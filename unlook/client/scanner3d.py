"""
High-level API for simplified 3D scanning with UnLook SDK.

This module provides a simplified API for performing 3D scans with the UnLook SDK.
It abstracts the complexity of structured light scanning, pattern generation,
camera control, and point cloud processing into a simple interface.
"""

import os
import time
import logging
import numpy as np
import json
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.info("open3d not installed. 3D mesh visualization and advanced filtering will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for o3dg when open3d is not available
    class PlaceholderO3DG:
        class PointCloud:
            pass
        class TriangleMesh:
            pass
    o3dg = PlaceholderO3DG

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    logger.info("h5py not installed. HDF5 file format support will be limited.")
    H5PY_AVAILABLE = False

from .. import UnlookClient
from .structured_light import (
    StereoStructuredLightScanner,
    StereoCalibrator,
    StereoCameraParameters,
    create_scanning_demo
)
from .advanced_structured_light import (
    EnhancedGrayCodeGenerator,
    PhaseShiftGenerator,
    EnhancedStereoScanner
)
from .robust_structured_light import (
    RobustGrayCodeGenerator,
    RobustStereoScanner
)


class UnlookScanner:
    """
    High-level 3D scanning API for the UnLook SDK.
    
    This class provides a simplified interface for 3D scanning, abstracting away the
    complexity of structured light scanning, pattern generation, camera control,
    and point cloud processing.
    
    Example:
        ```python
        # Create scanner with auto-detection of hardware
        scanner = UnlookScanner.auto_connect()
        
        # Perform a scan with default parameters
        point_cloud = scanner.perform_3d_scan()
        
        # Save the result
        scanner.save_scan(point_cloud, "my_scan.ply")
        
        # Optionally create and save a mesh
        mesh = scanner.create_mesh(point_cloud)
        scanner.save_mesh(mesh, "my_scan_mesh.obj")
        ```
    """
    
    def __init__(self, client: Optional[UnlookClient] = None, use_default_calibration: bool = True,
                 scanner_type: str = "robust"):
        """
        Initialize the UnlookScanner.
        
        Args:
            client: An existing UnlookClient instance, or None to create a new one
            use_default_calibration: Whether to use default calibration parameters
            scanner_type: Type of scanner to use:
                         - "basic": Original structured light implementation
                         - "enhanced": Enhanced implementation with Gray code and phase shift
                         - "robust": Robust implementation for real-world objects (recommended)
        """
        self.client = client or UnlookClient()
        self.is_connected = False
        self.scanner_info = None
        self.structured_light_scanner = None  # Basic scanner
        self.enhanced_scanner = None          # Enhanced scanner
        self.robust_scanner = None            # Robust scanner
        self.output_dir = os.path.join(os.getcwd(), "scans")
        self.use_default_calibration = use_default_calibration
        self.scanner_type = scanner_type
        self.scan_folders = {}  # Map of scan_id -> folders
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log scanner type
        logger.info(f"Initializing UnlookScanner with {scanner_type} implementation")
        
    @classmethod
    def auto_connect(cls, timeout: int = 5, use_default_calibration: bool = True,
                     scanner_type: str = "robust") -> 'UnlookScanner':
        """
        Create a scanner and automatically connect to the first available UnLook scanner.
        
        Args:
            timeout: Timeout in seconds for scanner discovery
            use_default_calibration: Whether to use default calibration parameters
            scanner_type: Type of scanner to use:
                         - "basic": Original structured light implementation
                         - "enhanced": Enhanced implementation with Gray code and phase shift
                         - "robust": Robust implementation for real-world objects (recommended)
            
        Returns:
            Connected UnlookScanner instance
        """
        # For backward compatibility
        if isinstance(scanner_type, bool):
            # If scanner_type is a boolean, it's the old use_enhanced_scanner parameter
            scanner_type = "enhanced" if scanner_type else "basic"
            logger.warning("Using deprecated boolean parameter for scanner type. "
                          "Please use scanner_type='enhanced' or scanner_type='robust' instead")
        
        scanner = cls(use_default_calibration=use_default_calibration, 
                      scanner_type=scanner_type)
        scanner.connect(timeout=timeout)
        return scanner
    
    def connect(self, scanner_id: Optional[str] = None, timeout: int = 5) -> bool:
        """
        Connect to an UnLook scanner.
        
        Args:
            scanner_id: Optional scanner ID to connect to. If None, connects to the first found scanner
            timeout: Timeout in seconds for scanner discovery
            
        Returns:
            True if connection was successful, False otherwise
        """
        if self.is_connected:
            logger.info("Already connected to a scanner")
            return True
            
        # Start discovery
        self.client.start_discovery()
        logger.info(f"Discovering scanners for {timeout} seconds...")
        time.sleep(timeout)
        
        # Get discovered scanners
        scanners = self.client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found")
            return False
            
        # Select scanner
        if scanner_id:
            selected_scanner = None
            for scanner in scanners:
                if scanner.uuid == scanner_id:
                    selected_scanner = scanner
                    break
                    
            if not selected_scanner:
                logger.error(f"Scanner with ID {scanner_id} not found")
                return False
        else:
            selected_scanner = scanners[0]
            
        # Connect to scanner
        logger.info(f"Connecting to scanner: {selected_scanner.name} ({selected_scanner.uuid})")
        if not self.client.connect(selected_scanner):
            logger.error("Failed to connect to scanner")
            return False
            
        self.is_connected = True
        self.scanner_info = selected_scanner
        
        # Initialize structured light scanner
        self._initialize_scanner()
        
        return True
        
    def _initialize_scanner(self) -> None:
        """
        Initialize the structured light scanner with the appropriate calibration.
        """
        # Create a calibration directory
        calib_dir = os.path.join(self.output_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)
        
        # Initialize scanners based on config
        # Always initialize basic scanner first (for fallback)
        if self.use_default_calibration:
            logger.info("Using default stereo calibration parameters")
            calib_file = os.path.join(calib_dir, "default_stereo_calibration.json")
            
            try:
                # Try to load existing calibration
                if os.path.exists(calib_file):
                    logger.info(f"Loading existing calibration from {calib_file}")
                    stereo_params = StereoCameraParameters.load(calib_file)
                    self.structured_light_scanner = StereoStructuredLightScanner(stereo_params)
                else:
                    # Create new calibration
                    logger.info("Creating new default calibration")
                    self.structured_light_scanner = create_scanning_demo(calib_dir)
            except Exception as e:
                logger.error(f"Error loading/creating basic scanner calibration: {e}")
                logger.warning("Creating fallback basic scanner with hardcoded parameters")
                self.structured_light_scanner = create_scanning_demo(calib_dir)
            
            # Get image size from basic scanner
            try:
                image_size = self.structured_light_scanner.camera_params.image_size
            except Exception:
                # Fallback to common HD resolution
                image_size = (1280, 720)
                logger.warning(f"Using fallback image size: {image_size}")
            
            # Initialize other scanner types based on configuration
            if self.scanner_type in ["enhanced", "robust"]:
                # Initialize the enhanced scanner if needed
                if self.scanner_type == "enhanced" or self.scanner_type == "robust":
                    try:
                        logger.info("Initializing enhanced structured light scanner")
                        self.enhanced_scanner = EnhancedStereoScanner.from_default_calibration(
                            image_size=image_size,
                            projector_width=1920,
                            projector_height=1080
                        )
                        logger.info("Enhanced structured light scanner initialized")
                    except Exception as e:
                        logger.error(f"Error initializing enhanced scanner, will use basic fallback: {e}")
                
                # Initialize the robust scanner if needed
                if self.scanner_type == "robust":
                    try:
                        logger.info("Initializing robust structured light scanner")
                        self.robust_scanner = RobustStereoScanner.from_default_calibration(
                            image_size=image_size,
                            projector_width=1920,
                            projector_height=1080
                        )
                        logger.info("Robust structured light scanner initialized")
                    except Exception as e:
                        logger.error(f"Error initializing robust scanner, will try enhanced or basic fallback: {e}")
        else:
            # TODO: Add support for custom calibration with actual images
            logger.warning("Custom calibration not yet implemented, using default")
            self.structured_light_scanner = create_scanning_demo(calib_dir)
            
            # For custom calibration, we still attempt to initialize the advanced scanners
            try:
                image_size = self.structured_light_scanner.camera_params.image_size
                
                # Initialize enhanced scanner
                if self.scanner_type == "enhanced" or self.scanner_type == "robust":
                    logger.info("Initializing enhanced structured light scanner with default params")
                    self.enhanced_scanner = EnhancedStereoScanner.from_default_calibration(
                        image_size=image_size
                    )
                    logger.info("Enhanced structured light scanner initialized")
                
                # Initialize robust scanner
                if self.scanner_type == "robust":
                    logger.info("Initializing robust structured light scanner with default params")
                    self.robust_scanner = RobustStereoScanner.from_default_calibration(
                        image_size=image_size
                    )
                    logger.info("Robust structured light scanner initialized")
            except Exception as e:
                logger.error(f"Error initializing advanced scanners: {e}")
        
        # Log scanner initialization status
        logger.info(f"Basic scanner initialized: {self.structured_light_scanner is not None}")
        logger.info(f"Enhanced scanner initialized: {self.enhanced_scanner is not None}")
        logger.info(f"Robust scanner initialized: {self.robust_scanner is not None}")
        
        # If the selected scanner type failed to initialize, try fallbacks
        if self.scanner_type == "robust" and self.robust_scanner is None:
            if self.enhanced_scanner is not None:
                logger.warning("Robust scanner failed to initialize, falling back to enhanced scanner")
                self.scanner_type = "enhanced"
            else:
                logger.warning("Advanced scanners failed to initialize, falling back to basic scanner")
                self.scanner_type = "basic"
        elif self.scanner_type == "enhanced" and self.enhanced_scanner is None:
            logger.warning("Enhanced scanner failed to initialize, falling back to basic scanner")
            self.scanner_type = "basic"
            
        logger.info(f"Using scanner type: {self.scanner_type}")
        
    def disconnect(self) -> None:
        """Disconnect from the scanner."""
        if self.is_connected:
            self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from scanner")
        
    def perform_3d_scan(
            self, 
            output_dir: Optional[str] = None,
            mask_threshold: int = 5,
            interval: float = 0.5,
            visualize: bool = False,
            scanner_type: Optional[str] = None,
            scan_quality: str = "medium",
            pattern_type: str = "gray_code",
            debug_output: bool = True
        ) -> o3dg.PointCloud:
        """
        Perform a complete 3D scan using structured light patterns.
        
        Args:
            output_dir: Optional output directory for scan results
            mask_threshold: Threshold for shadow/valid pixel detection
            interval: Time interval between pattern projections in seconds
            visualize: Whether to visualize the results (requires open3d)
            scanner_type: Override scanner type (otherwise uses instance setting)
                         - "basic": Original structured light implementation
                         - "enhanced": Enhanced implementation with Gray code and phase shift
                         - "robust": Robust implementation for real-world objects (recommended)
            scan_quality: Quality setting: "low", "medium", "high" or "ultra"
                         - "low": Faster but lower quality scan with fewer patterns
                         - "medium": Balanced quality and speed (default)
                         - "high": Higher quality scan with more patterns
                         - "ultra": Maximum quality with combined patterns and filtering
            pattern_type: Type of structured light patterns to use:
                         - "gray_code": Binary Gray code patterns (default)
                         - "phase_shift": Phase shift patterns for higher resolution
                         - "combined": Both Gray code and phase shift for best results
            debug_output: Whether to save debug information (useful for troubleshooting)
            
        Returns:
            3D point cloud
        """
        # For backward compatibility
        if isinstance(scanner_type, bool):
            # If scanner_type is a boolean, it's the old use_enhanced_scanner parameter
            scanner_type = "enhanced" if scanner_type else "basic"
            logger.warning("Using deprecated 'use_enhanced_scanner' parameter. "
                          "Please use scanner_type='robust' for best results")
        
        # Create debug sub-directory if enabled
        debug_dir = None
        if debug_output and output_dir:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
        if not self.is_connected:
            logger.error("Not connected to a scanner")
            return o3dg.PointCloud()
            
        # Determine scanner type to use
        use_scanner_type = scanner_type if scanner_type is not None else self.scanner_type
        
        # Check if requested scanner is available
        if use_scanner_type == "robust" and self.robust_scanner is None:
            if self.enhanced_scanner is not None:
                logger.warning("Robust scanner requested but not available, falling back to enhanced scanner")
                use_scanner_type = "enhanced"
            else:
                logger.warning("Advanced scanners not available, falling back to basic scanner")
                use_scanner_type = "basic"
        elif use_scanner_type == "enhanced" and self.enhanced_scanner is None:
            logger.warning("Enhanced scanner requested but not available, falling back to basic scanner")
            use_scanner_type = "basic"
        
        if use_scanner_type == "basic" and self.structured_light_scanner is None:
            logger.error("No structured light scanner available")
            return o3dg.PointCloud()
            
        # Create a timestamped scan folder if output_dir is not specified
        if output_dir:
            scan_dir = output_dir
            os.makedirs(scan_dir, exist_ok=True)
            captures_dir = os.path.join(scan_dir, "captures")
            results_dir = os.path.join(scan_dir, "results")
            os.makedirs(captures_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        else:
            scan_dir, captures_dir, results_dir = self._create_timestamped_folder()
            
        # Store the scan folders
        scan_id = os.path.basename(scan_dir)
        self.scan_folders[scan_id] = {
            "scan_dir": scan_dir,
            "captures_dir": captures_dir,
            "results_dir": results_dir
        }
        
        logger.info(f"Starting 3D scan with {use_scanner_type} scanner, saving results to {scan_dir}")
        
        # Get structured light patterns based on scanner type, pattern type and quality
        if use_scanner_type == "robust":
            # Use robust scanner (best for real objects)
            logger.info(f"Using robust scanner with quality: {scan_quality}")
            patterns = self.robust_scanner.generate_gray_code_patterns()
            
            # Adjust interval based on quality for robust scanner
            if scan_quality == "low":
                interval = max(interval, 0.3)
            elif scan_quality == "medium":
                interval = max(interval, 0.5)
            elif scan_quality == "high":
                interval = max(interval, 0.75)
            elif scan_quality == "ultra":
                interval = max(interval, 1.0)
            
        elif use_scanner_type == "enhanced":
            # Use enhanced scanner
            logger.info(f"Using enhanced scanner with {pattern_type} patterns, quality: {scan_quality}")
            
            if pattern_type == "gray_code":
                patterns = self.enhanced_scanner.generate_gray_code_patterns()
            elif pattern_type == "phase_shift":
                patterns = self.enhanced_scanner.generate_phase_shift_patterns()
            elif pattern_type == "combined":
                # Combine Gray code and phase shift patterns
                gray_patterns = self.enhanced_scanner.generate_gray_code_patterns()
                phase_patterns = self.enhanced_scanner.generate_phase_shift_patterns()
                patterns = gray_patterns + phase_patterns
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}, using Gray code")
                patterns = self.enhanced_scanner.generate_gray_code_patterns()
            
            # Adjust number of patterns based on quality setting
            if scan_quality == "low":
                # For low quality, use fewer patterns
                if pattern_type == "gray_code":
                    # Keep white/black and a subset of patterns
                    patterns = patterns[:2] + patterns[2:][::2]  # Take every other pattern
                elif pattern_type == "phase_shift":
                    # Use only largest frequency
                    keep_patterns = [0, 1]  # White and black
                    # Keep only one frequency
                    for i in range(2, len(patterns)):
                        if "phase" in patterns[i]["name"] and any(f in patterns[i]["name"] for f in ["8", "16"]):
                            keep_patterns.append(i)
                    patterns = [patterns[i] for i in keep_patterns]
                elif pattern_type == "combined":
                    # Only keep essential patterns from both
                    patterns = patterns[:10]  # Just keep basic patterns
            
            elif scan_quality == "medium":
                # For medium quality, use the default pattern count
                pass  # No adjustment needed
            
            elif scan_quality in ["high", "ultra"]:
                # For high quality, use all patterns and longer capture interval
                interval = max(interval, 0.75)  # Ensure longer interval for stability
        else:
            # Use basic scanner
            logger.info("Using basic structured light scanner")
            patterns = self.structured_light_scanner.generate_scan_patterns()
        
        logger.info(f"Generated {len(patterns)} structured light patterns")
        
        # Set up projector
        projector = self.client.projector
        
        # Set projector to black to start
        projector.show_solid_field("Black")
        
        # Set up cameras
        camera = self.client.camera
        cameras = camera.get_cameras()
        
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return o3dg.PointCloud()
        
        # Select the first two cameras
        left_camera_id = cameras[0]["id"]
        right_camera_id = cameras[1]["id"]
        
        # Configure cameras for optimal scanning
        try:
            # Adjust camera settings based on quality
            exposure = 100
            if scan_quality == "high":
                exposure = 80  # Slightly lower exposure for high quality
            elif scan_quality == "ultra":
                exposure = 60  # Even lower exposure for ultra quality
                
            camera_config = {
                "exposure": exposure,
                "auto_exposure": False,
                "format": "png",
                "quality": 100
            }
            
            for cam_id in [left_camera_id, right_camera_id]:
                camera.configure(cam_id, camera_config)
        except Exception as e:
            logger.warning(f"Camera configuration failed: {e}")
            
        # Prepare arrays for images
        left_images = []
        right_images = []
        
        # Display and capture for each pattern
        for i, pattern in enumerate(patterns):
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}")
            self._display_pattern(projector, pattern, i)
            
            # Wait for projector to update
            time.sleep(interval)
            
            # Capture images
            try:
                # Capture from cameras
                logger.info(f"Capturing image pair {i+1}")
                left_img = camera.capture(left_camera_id)
                right_img = camera.capture(right_camera_id)
                
                # Save images
                left_path = os.path.join(captures_dir, f"left_{i:03d}.png")
                right_path = os.path.join(captures_dir, f"right_{i:03d}.png")
                
                cv2.imwrite(left_path, left_img)
                cv2.imwrite(right_path, right_img)
                
                # Add to arrays
                left_images.append(left_img)
                right_images.append(right_img)
                
            except Exception as e:
                logger.error(f"Error capturing images: {e}")
                continue
                
        # Reset projector to black
        projector.show_solid_field("Black")
        
        # Process the scan
        if len(left_images) < 2 or len(right_images) < 2:
            logger.error("Not enough images captured for scanning")
            return o3dg.PointCloud()
            
        logger.info(f"Processing scan with {len(left_images)} image pairs")
        
        # Use the appropriate scanner for processing
        if use_scanner_type == "robust":
            # Process with robust scanner (best for real objects)
            logger.info("Processing with robust scanner")
            
            # The robust scanner has built-in debug output
            pcd, debug_info = self.robust_scanner.process_scan(
                left_images, 
                right_images,
                output_dir=debug_dir
            )
            
            # Log important debug information
            if debug_info:
                logger.info(f"Valid pixels: {debug_info.get('combined_valid_pixels', 0)}")
                logger.info(f"Correspondences found: {debug_info.get('correspondences', 0)}")
                logger.info(f"Points after filtering: {len(pcd.points)}")
                
                # Save debug info to file
                if debug_dir:
                    debug_file = os.path.join(debug_dir, "scan_debug_info.json")
                    try:
                        with open(debug_file, 'w') as f:
                            # Convert numpy values to Python native types for JSON serialization
                            clean_debug = {}
                            for k, v in debug_info.items():
                                if isinstance(v, (np.int32, np.int64, np.uint64)):
                                    clean_debug[k] = int(v)
                                elif isinstance(v, (np.float32, np.float64)):
                                    clean_debug[k] = float(v)
                                elif isinstance(v, np.ndarray):
                                    # Convert numpy arrays to lists
                                    clean_debug[k] = v.tolist()
                                elif isinstance(v, dict):
                                    # Handle nested dictionaries
                                    nested_clean = {}
                                    for kk, vv in v.items():
                                        if isinstance(vv, (np.int32, np.int64, np.uint64)):
                                            nested_clean[kk] = int(vv)
                                        elif isinstance(vv, (np.float32, np.float64)):
                                            nested_clean[kk] = float(vv)
                                        elif isinstance(vv, np.ndarray):
                                            nested_clean[kk] = vv.tolist()
                                        else:
                                            try:
                                                # Try JSON serialization as a test
                                                json.dumps(vv)
                                                nested_clean[kk] = vv
                                            except (TypeError, OverflowError):
                                                # If it's not serializable, convert to string
                                                nested_clean[kk] = str(vv)
                                    clean_debug[k] = nested_clean
                                else:
                                    try:
                                        # Try JSON serialization as a test
                                        json.dumps(v)
                                        clean_debug[k] = v
                                    except (TypeError, OverflowError):
                                        # If it's not serializable, convert to string
                                        clean_debug[k] = str(v)
                            import json
                            json.dump(clean_debug, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save debug info: {e}")
            
        elif use_scanner_type == "enhanced":
            # Process with enhanced scanner with options based on pattern type and quality
            use_gray_code = pattern_type in ["gray_code", "combined"]
            use_phase_shift = pattern_type in ["phase_shift", "combined"]
            
            pcd = self.enhanced_scanner.process_scan(
                left_images, 
                right_images, 
                use_gray_code=use_gray_code,
                use_phase_shift=use_phase_shift,
                mask_threshold=mask_threshold
            )
            
            # Apply additional filtering for ultra quality
            if scan_quality == "ultra" and OPEN3D_AVAILABLE and len(pcd.points) > 100:
                logger.info("Applying additional point cloud filtering for ultra quality")
                # Statistical outlier removal with stricter parameters
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
                # Radius outlier removal
                pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=1.5)
        else:
            # Process with basic scanner
            pcd = self.structured_light_scanner.process_scan(
                left_images, 
                right_images, 
                mask_threshold=mask_threshold
            )
        
        # If the point cloud has few points, try with more relaxed parameters
        if len(pcd.points) < 100:
            logger.warning(f"Initial scan produced only {len(pcd.points)} points. Trying with more relaxed parameters...")
            
            if use_scanner_type == "robust":
                # For robust scanner, we don't retry - it already uses adaptive parameters
                pass
            elif use_scanner_type == "enhanced":
                pcd = self.enhanced_scanner.process_scan(
                    left_images, 
                    right_images, 
                    use_gray_code=pattern_type in ["gray_code", "combined"],
                    use_phase_shift=pattern_type in ["phase_shift", "combined"],
                    mask_threshold=3
                )
            else:
                pcd = self.structured_light_scanner.process_scan(
                    left_images, 
                    right_images, 
                    mask_threshold=3
                )
            
        # Get the final point count
        point_count = len(pcd.points) if hasattr(pcd, 'points') else 0
        logger.info(f"Generated point cloud with {point_count} points")
        
        # Save point cloud
        if point_count > 0:
            point_cloud_path = os.path.join(results_dir, "scan_point_cloud.ply")
            
            if use_scanner_type == "robust":
                self.robust_scanner.save_point_cloud(pcd, point_cloud_path)
            elif use_scanner_type == "enhanced":
                self.enhanced_scanner.save_point_cloud(pcd, point_cloud_path)
            else:
                self.structured_light_scanner.save_point_cloud(pcd, point_cloud_path)
                
            logger.info(f"Saved point cloud to {point_cloud_path}")
            
        # Visualize if requested
        if visualize and OPEN3D_AVAILABLE and len(pcd.points) > 0:
            try:
                logger.info("Visualizing results")
                o3d.visualization.draw_geometries([pcd], window_name="3D Scan Results")
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
                
        return pcd
    
    def _display_pattern(self, projector, pattern, idx):
        """Helper method to display a pattern on the projector."""
        try:
            pattern_name = pattern.get('name', '').lower()
            pattern_type = pattern.get('pattern_type', '')
            
            if pattern_type == "raw_image":
                # For raw images, we can't directly set them, so we approximate
                if 'white' in pattern_name:
                    projector.show_solid_field("White")
                elif 'black' in pattern_name:
                    projector.show_solid_field("Black")
                elif 'gray_code_x' in pattern_name or 'horizontal' in pattern_name:
                    # Use horizontal lines
                    width = max(1, 4 - (idx % 4))
                    projector.show_horizontal_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=width,
                        background_width=width
                    )
                elif 'gray_code_y' in pattern_name or 'vertical' in pattern_name:
                    # Use vertical lines
                    width = max(1, 4 - (idx % 4))
                    projector.show_vertical_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=width,
                        background_width=width
                    )
                else:
                    # Fallback to a checkerboard pattern
                    projector.show_checkerboard(
                        foreground_color="White",
                        background_color="Black",
                        horizontal_count=6 + (idx % 6),
                        vertical_count=4 + (idx % 4)
                    )
            else:
                # For predefined patterns, use the appropriate method
                if pattern_type == "solid_field":
                    projector.show_solid_field(pattern.get("color", "White"))
                elif pattern_type == "horizontal_lines":
                    projector.show_horizontal_lines(
                        foreground_color=pattern.get("foreground_color", "White"),
                        background_color=pattern.get("background_color", "Black"),
                        foreground_width=pattern.get("foreground_width", 4),
                        background_width=pattern.get("background_width", 20)
                    )
                elif pattern_type == "vertical_lines":
                    projector.show_vertical_lines(
                        foreground_color=pattern.get("foreground_color", "White"),
                        background_color=pattern.get("background_color", "Black"),
                        foreground_width=pattern.get("foreground_width", 4),
                        background_width=pattern.get("background_width", 20)
                    )
                else:
                    # Default to a grid for other types
                    projector.show_grid(
                        foreground_color="White",
                        background_color="Black",
                        h_foreground_width=4,
                        h_background_width=20,
                        v_foreground_width=4,
                        v_background_width=20
                    )
        except Exception as e:
            logger.error(f"Error displaying pattern: {e}")
    
    def _create_timestamped_folder(self) -> Tuple[str, str, str]:
        """
        Create a timestamped folder structure for a scan.
        
        Returns:
            Tuple of (scan_dir, captures_dir, results_dir)
        """
        # Create timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(self.output_dir, f"Scan_{timestamp}")
        
        # Create scan directory
        os.makedirs(scan_dir, exist_ok=True)
        
        # Create subdirectories
        captures_dir = os.path.join(scan_dir, "captures")
        results_dir = os.path.join(scan_dir, "results")
        
        os.makedirs(captures_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        return scan_dir, captures_dir, results_dir
    
    def create_mesh(
            self, 
            point_cloud: o3dg.PointCloud, 
            depth: int = 9, 
            smooth_iterations: int = 5
        ) -> o3dg.TriangleMesh:
        """
        Create a triangle mesh from a point cloud.
        
        Args:
            point_cloud: Input point cloud
            depth: Depth parameter for Poisson reconstruction (higher = more detail)
            smooth_iterations: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Cannot create mesh from point cloud")
            return o3dg.TriangleMesh()
            
        return self.structured_light_scanner.create_mesh_from_point_cloud(
            point_cloud, 
            depth=depth, 
            smooth_iterations=smooth_iterations
        )
    
    def save_scan(self, point_cloud: o3dg.PointCloud, filepath: str) -> None:
        """
        Save a point cloud to a file.
        
        Args:
            point_cloud: Point cloud to save
            filepath: Output file path (supported formats: .ply, .pcd, .obj)
        """
        return self.structured_light_scanner.save_point_cloud(point_cloud, filepath)
    
    def save_mesh(self, mesh: o3dg.TriangleMesh, filepath: str) -> None:
        """
        Save a triangle mesh to a file.
        
        Args:
            mesh: Triangle mesh to save
            filepath: Output file path (supported formats: .ply, .obj, .off, .gltf)
        """
        return self.structured_light_scanner.save_mesh(mesh, filepath)
    
    def get_latest_scan_id(self) -> Optional[str]:
        """
        Get the ID of the most recent scan.
        
        Returns:
            Scan ID or None if no scans have been performed
        """
        if not self.scan_folders:
            return None
            
        return list(self.scan_folders.keys())[-1]
    
    def get_scan_folders(self, scan_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get the folder paths for a scan.
        
        Args:
            scan_id: Scan ID, or None to use the most recent scan
            
        Returns:
            Dictionary of folder paths (scan_dir, captures_dir, results_dir)
        """
        if not scan_id:
            scan_id = self.get_latest_scan_id()
            
        if not scan_id or scan_id not in self.scan_folders:
            return {}
            
        return self.scan_folders[scan_id]
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the base output directory for all scans.
        
        Args:
            output_dir: Path to the base output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Set output directory to {output_dir}")
        
    def set_calibration_params(self, 
                             camera_matrix_left: np.ndarray,
                             dist_coeffs_left: np.ndarray,
                             camera_matrix_right: np.ndarray,
                             dist_coeffs_right: np.ndarray,
                             R: np.ndarray,
                             T: np.ndarray,
                             image_size: Tuple[int, int] = None) -> None:
        """
        Set custom calibration parameters for the stereo camera system.
        
        This allows using calibration parameters from the StereoCalibrator
        or other external calibration tools.
        
        Args:
            camera_matrix_left: 3x3 camera matrix for left camera
            dist_coeffs_left: Distortion coefficients for left camera
            camera_matrix_right: 3x3 camera matrix for right camera
            dist_coeffs_right: Distortion coefficients for right camera
            R: 3x3 rotation matrix between cameras
            T: 3x1 translation vector between cameras
            image_size: Optional image size (width, height) - if not specified,
                       will use the size from existing scanners or a default value
        """
        # Use image size from parameters, or try to infer from existing scanners
        if image_size is None:
            if self.structured_light_scanner is not None:
                try:
                    image_size = self.structured_light_scanner.camera_params.image_size
                except Exception:
                    image_size = (1280, 720)  # Default fallback
            else:
                image_size = (1280, 720)  # Default fallback
        
        logger.info(f"Setting custom calibration parameters with image size {image_size}")
        
        # Create camera parameters from the provided matrices
        stereo_params = StereoCameraParameters(
            camera_matrix_left=camera_matrix_left,
            dist_coeffs_left=dist_coeffs_left,
            camera_matrix_right=camera_matrix_right,
            dist_coeffs_right=dist_coeffs_right,
            R=R,
            T=T,
            image_size=image_size
        )
        
        # Apply to basic scanner
        self.structured_light_scanner = StereoStructuredLightScanner(stereo_params)
        logger.info("Applied calibration to basic scanner")
        
        # Try to apply to enhanced scanner
        try:
            self.enhanced_scanner = EnhancedStereoScanner(
                stereo_params,
                projector_width=1920,
                projector_height=1080
            )
            logger.info("Applied calibration to enhanced scanner")
        except Exception as e:
            logger.warning(f"Failed to create enhanced scanner with custom calibration: {e}")
            self.enhanced_scanner = None
        
        # Try to apply to robust scanner
        try:
            self.robust_scanner = RobustStereoScanner(
                stereo_params,
                projector_width=1920,
                projector_height=1080
            )
            logger.info("Applied calibration to robust scanner")
        except Exception as e:
            logger.warning(f"Failed to create robust scanner with custom calibration: {e}")
            self.robust_scanner = None
        
        # Update scanner type based on available scanners
        if self.scanner_type == "robust" and self.robust_scanner is None:
            if self.enhanced_scanner is not None:
                logger.warning("Robust scanner not available with custom calibration, falling back to enhanced scanner")
                self.scanner_type = "enhanced"
            else:
                logger.warning("Advanced scanners not available with custom calibration, falling back to basic scanner")
                self.scanner_type = "basic"
        elif self.scanner_type == "enhanced" and self.enhanced_scanner is None:
            logger.warning("Enhanced scanner not available with custom calibration, falling back to basic scanner")
            self.scanner_type = "basic"
        
        # Update use_default_calibration flag
        self.use_default_calibration = False
        
        logger.info(f"Custom calibration applied, using scanner type: {self.scanner_type}")

    def visualize_point_cloud(self, point_cloud: o3dg.PointCloud) -> None:
        """
        Visualize a point cloud (requires open3d).
        
        Args:
            point_cloud: Point cloud to visualize
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Cannot visualize point cloud")
            return
            
        if len(point_cloud.points) == 0:
            logger.warning("Point cloud is empty, nothing to visualize")
            return
            
        try:
            o3d.visualization.draw_geometries([point_cloud], window_name="UnLook 3D Scan")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically disconnect from scanner."""
        self.disconnect()