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
from ..core.protocol import MessageType
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
from .single_camera_scanner import (
    SingleCameraCalibrator,
    SingleCameraStructuredLight
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
                 scanner_type: str = "robust", single_camera_mode: bool = False):
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
        self.single_camera_scanner = None     # Single camera scanner
        self.single_camera_mode = single_camera_mode
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
                     scanner_type: str = "robust", single_camera_mode: bool = False) -> 'UnlookScanner':
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
                      scanner_type=scanner_type,
                      single_camera_mode=single_camera_mode)
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

        # If single camera mode, initialize single camera scanner
        if self.single_camera_mode:
            self._initialize_single_camera_scanner()
        
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
        logger.info(f"Single camera scanner initialized: {self.single_camera_scanner is not None}")
        logger.info(f"Single camera mode: {self.single_camera_mode}")
        
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

    def _initialize_single_camera_scanner(self) -> None:
        """
        Initialize the single camera structured light scanner.
        """
        if not self.single_camera_mode:
            return

        # Create a calibration directory
        calib_dir = os.path.join(self.output_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)

        # Check for existing calibration file
        calib_file = os.path.join(calib_dir, "single_camera_calibration.npz")

        try:
            # Try to load existing calibration
            if os.path.exists(calib_file):
                logger.info(f"Loading single camera calibration from {calib_file}")
                self.single_camera_scanner = SingleCameraStructuredLight.from_calibration_file(
                    calib_file,
                    projector_width=1920,
                    projector_height=1080
                )
                logger.info("Single camera scanner initialized from calibration file")
            else:
                # Create with default parameters if no calibration file exists
                logger.info("No single camera calibration file found. Using default parameters.")
                # Get camera intrinsics (approximate)
                camera_matrix = np.array([
                    [1000.0, 0, 640.0],
                    [0, 1000.0, 480.0],
                    [0, 0, 1]
                ])
                camera_dist_coeffs = np.zeros(5)

                # Create projector parameters
                projector_matrix = np.array([
                    [1000.0, 0, 960.0],
                    [0, 1000.0, 540.0],
                    [0, 0, 1]
                ])
                projector_dist_coeffs = np.zeros(5)

                # Camera-projector transformation
                # The camera is usually to the right of the projector
                R = np.eye(3)  # Identity rotation
                T = np.array([[150.0], [0.0], [0.0]])  # 150mm baseline

                self.single_camera_scanner = SingleCameraStructuredLight(
                    camera_matrix, camera_dist_coeffs,
                    projector_matrix, projector_dist_coeffs,
                    R, T, 1920, 1080
                )
                logger.info("Single camera scanner initialized with default parameters")

                # Save default calibration for future use
                np.savez(
                    calib_file,
                    camera_matrix=camera_matrix,
                    camera_dist_coeffs=camera_dist_coeffs,
                    projector_matrix=projector_matrix,
                    projector_dist_coeffs=projector_dist_coeffs,
                    R=R,
                    T=T
                )
                logger.info(f"Saved default calibration to {calib_file}")

        except Exception as e:
            logger.error(f"Error initializing single camera scanner: {e}")
            self.single_camera_scanner = None
        
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
            debug_output: bool = True,
            use_single_camera: Optional[bool] = None
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

        # Determine whether to use single camera mode
        use_single_camera = use_single_camera if use_single_camera is not None else self.single_camera_mode

        # If using single camera mode, delegate to single camera scanner
        if use_single_camera:
            if self.single_camera_scanner is None:
                logger.error("Single camera scanner not initialized")
                return o3dg.PointCloud()

            return self._perform_single_camera_scan(
                output_dir=output_dir,
                mask_threshold=mask_threshold,
                interval=interval,
                visualize=visualize,
                scan_quality=scan_quality,
                pattern_type=pattern_type,
                debug_output=debug_output
            )

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

        # Configure projector for optimal pattern visibility
        try:
            if use_scanner_type == "enhanced":
                # Prepare the projector using only supported message types
                try:
                    # Set the projector to test pattern mode for optimal pattern projection
                    projector.set_mode("TestPatternGenerator")
                    logger.info("Set projector to test pattern mode for enhanced scanning")

                    # For maximum brightness, display a pure white pattern first
                    # This helps "warm up" the projector to full brightness
                    success = projector.show_solid_field("White")
                    if success:
                        logger.info("Displayed reference white field to prepare projector")
                        # Wait a moment for projector to adjust
                        time.sleep(0.25)

                    # Then show black as baseline for good contrast
                    success = projector.show_solid_field("Black")
                    if success:
                        logger.info("Displayed reference black field to prepare projector")
                        # Wait a moment for projector to adjust
                        time.sleep(0.25)

                    # Display a test pattern with lines to ensure projector is ready
                    # for detailed structured light patterns
                    if scan_quality in ["high", "ultra"]:
                        # For high quality, use finer lines to warm up the projector
                        success = projector.show_vertical_lines(
                            foreground_color="White",
                            background_color="Black",
                            foreground_width=4,
                            background_width=4
                        )
                        if success:
                            logger.info("Displayed fine test pattern to prepare projector for high quality scan")
                            time.sleep(0.2)
                    else:
                        # For other quality settings, use standard lines
                        success = projector.show_horizontal_lines(
                            foreground_color="White",
                            background_color="Black",
                            foreground_width=8,
                            background_width=8
                        )
                        if success:
                            logger.info("Displayed test pattern to prepare projector")
                            time.sleep(0.2)

                    logger.info("Successfully prepared projector for enhanced scanning")
                except Exception as mode_err:
                    logger.warning(f"Error preparing projector: {mode_err}")
        except Exception as e:
            logger.warning(f"Failed to prepare projector for scanning: {e}")

            # Fallback method - simple preparation
            try:
                projector.set_mode("TestPatternGenerator")
                projector.show_solid_field("White")
                time.sleep(0.2)
                projector.show_solid_field("Black")
                logger.info("Applied fallback projector preparation")
            except Exception:
                logger.warning("Failed to prepare projector")

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
            # Adjust camera settings based on quality and scanner type
            exposure = 100
            contrast = 1.0
            brightness = 0.0

            # For enhanced scanner, use optimized camera settings
            if use_scanner_type == "enhanced":
                # Much higher exposure and gain for better pattern visibility
                # These values are significantly increased based on observations of underexposed images
                exposure = 400  # Significantly increased from 150
                contrast = 2.0  # Higher contrast for better pattern visibility
                brightness = 0.2  # Increased brightness boost
                gain = 4.0  # Add analog gain to amplify the signal

                logger.info(f"Using high-visibility camera settings for enhanced scanner: exposure={exposure}, gain={gain}, contrast={contrast}, brightness={brightness}")
            else:
                # Default settings for other scanner types
                if scan_quality == "high":
                    exposure = 80  # Slightly lower exposure for high quality
                elif scan_quality == "ultra":
                    exposure = 60  # Even lower exposure for ultra quality

            # Apply camera settings based on quality level
            if scan_quality == "high" or scan_quality == "ultra":
                # Higher quality scans need more aggressive settings
                exposure *= 1.5
                gain = gain if 'gain' in locals() else 2.0  # Use existing gain or default to 2.0
                logger.info(f"Boosting exposure and gain for {scan_quality} quality scan")

            # Create camera configuration with exposure, gain, contrast and brightness settings
            camera_config = {
                "exposure": exposure,
                "auto_exposure": False,
                "gain": gain if 'gain' in locals() else 2.0,  # Use existing gain or default to 2.0
                "auto_gain": False,
                "format": "png",
                "quality": 100,
                "contrast": contrast,
                "brightness": brightness,
                "color_mode": "grayscale",  # Force grayscale mode for more consistent scanning
                "force_settings": True  # Force settings to be applied directly to camera hardware
            }

            # Configure both cameras with the higher exposure/gain settings
            for cam_id in [left_camera_id, right_camera_id]:
                success = camera.configure(cam_id, camera_config)
                if not success:
                    logger.warning(f"Failed to configure camera {cam_id}, trying direct method")
                    # Try direct method to set exposure and gain
                    camera.set_exposure(cam_id, exposure, gain=camera_config["gain"], auto_exposure=False, auto_gain=False)

                # Let's try to make sure settings are applied by configuring camera again
                try:
                    # Instead of using the unsupported CAMERA_APPLY_SETTINGS message type,
                    # we'll use the standard configuration method again to reinforce our settings
                    success = camera.configure(cam_id, camera_config)
                    if success:
                        logger.info(f"Applied additional configuration to camera {cam_id} to ensure settings take effect")
                    else:
                        # Let's try one more time with a delay
                        time.sleep(0.1)
                        success = camera.configure(cam_id, camera_config)
                        if success:
                            logger.info(f"Applied delayed configuration to camera {cam_id}")
                        else:
                            logger.warning(f"Failed to apply additional configuration to camera {cam_id}")
                except Exception as e:
                    logger.warning(f"Failed to ensure camera settings applied: {e}")
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
                mask_threshold=mask_threshold,
                output_dir=debug_dir
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
                    mask_threshold=3,
                    output_dir=debug_dir
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
    
    def _perform_single_camera_scan(
            self,
            output_dir: Optional[str] = None,
            mask_threshold: int = 5,
            interval: float = 0.5,
            visualize: bool = False,
            scan_quality: str = "medium",
            pattern_type: str = "gray_code",
            debug_output: bool = True
        ) -> o3dg.PointCloud:
        """
        Perform a 3D scan using single camera structured light.

        Args:
            output_dir: Optional output directory for scan results
            mask_threshold: Threshold for shadow/valid pixel detection
            interval: Time interval between pattern projections in seconds
            visualize: Whether to visualize the results (requires open3d)
            scan_quality: Quality setting: "low", "medium", "high" or "ultra"
            pattern_type: Type of structured light patterns to use
            debug_output: Whether to save debug information

        Returns:
            3D point cloud
        """
        # Create a timestamped scan folder if output_dir is not specified
        if output_dir:
            scan_dir = output_dir
            os.makedirs(scan_dir, exist_ok=True)
            captures_dir = os.path.join(scan_dir, "captures")
            results_dir = os.path.join(scan_dir, "results")
            os.makedirs(captures_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        else:
            scan_dir, captures_dir, results_dir = self._create_timestamped_folder("single_camera")

        # Create debug directory if needed
        debug_dir = os.path.join(scan_dir, "debug") if debug_output else None
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # Store the scan folders
        scan_id = os.path.basename(scan_dir)
        self.scan_folders[scan_id] = {
            "scan_dir": scan_dir,
            "captures_dir": captures_dir,
            "results_dir": results_dir
        }

        logger.info(f"Starting single camera 3D scan with {pattern_type} patterns, saving results to {scan_dir}")

        # Generate structured light patterns
        if pattern_type == "phase_shift":
            patterns = self.single_camera_scanner.generate_phase_shift_patterns()
        elif pattern_type == "combined":
            patterns = self.single_camera_scanner.generate_gray_code_patterns() + \
                       self.single_camera_scanner.generate_phase_shift_patterns()
        else:  # default to gray_code
            patterns = self.single_camera_scanner.generate_gray_code_patterns()

        # Adjust patterns based on quality
        if scan_quality == "low":
            # Use fewer patterns for faster scanning
            if pattern_type == "combined":
                # Just use gray code
                patterns = self.single_camera_scanner.generate_gray_code_patterns()
            elif pattern_type == "phase_shift":
                # Use fewer frequencies
                patterns = self.single_camera_scanner.generate_phase_shift_patterns(frequencies=[16])
            else:
                # Skip alternate patterns
                patterns = patterns[::2]
        elif scan_quality == "high" or scan_quality == "ultra":
            # Use more/all patterns for higher quality
            if pattern_type == "phase_shift":
                # Use more frequencies
                patterns = self.single_camera_scanner.generate_phase_shift_patterns(
                    frequencies=[8, 16, 32, 64]
                )

        logger.info(f"Generated {len(patterns)} structured light patterns")

        # Set up projector
        projector = self.client.projector

        # Configure projector for maximum visibility in single camera mode
        # Single camera mode requires good pattern contrast
        try:
            # Prepare the projector using only supported message types and methods
            # Set projector to test pattern mode for pattern projection
            try:
                # Set the projector to test pattern mode
                projector.set_mode("TestPatternGenerator")
                logger.info("Set projector to test pattern mode for single camera scanning")

                # For maximum brightness, we'll first show a pure white pattern
                # This can help "warm up" the projector to full brightness and prepare the hardware
                success = projector.show_solid_field("White")
                if success:
                    logger.info("Displayed reference white field to prepare projector")
                    # Wait a moment for projector to adjust
                    time.sleep(0.3)  # Slightly longer wait for single camera mode

                # Then show a black pattern for contrast optimization
                success = projector.show_solid_field("Black")
                if success:
                    logger.info("Displayed reference black field to prepare projector")
                    # Wait a moment for projector to adjust
                    time.sleep(0.3)  # Slightly longer wait for single camera mode

                # For single camera mode, we'll also display a test pattern with lines
                # to ensure the projector is properly warmed up for fine patterns
                success = projector.show_horizontal_lines(
                    foreground_color="White",
                    background_color="Black",
                    foreground_width=8,   # Wider white lines for better visibility
                    background_width=8    # Equal spacing for good contrast
                )
                if success:
                    logger.info("Displayed test line pattern to prepare projector")
                    time.sleep(0.2)

                logger.info("Successfully prepared projector for single camera scanning")
            except Exception as mode_err:
                logger.warning(f"Error preparing projector for single camera scanning: {mode_err}")
        except Exception as e:
            logger.warning(f"Failed to prepare projector for single camera scanning: {e}")

            # Fallback method - basic preparation
            try:
                projector.set_mode("TestPatternGenerator")
                projector.show_solid_field("White")
                time.sleep(0.2)
                logger.info("Applied fallback projector preparation for single camera scanning")
            except Exception:
                logger.warning("Failed to prepare projector for single camera scanning")

        # Set projector to black to start
        projector.show_solid_field("Black")

        # Set up camera
        camera = self.client.camera
        cameras = camera.get_cameras()

        if not cameras:
            logger.error("No cameras found")
            return o3dg.PointCloud()

        # Use the first camera
        camera_id = cameras[0]["id"]

        # Configure camera for optimal scanning
        try:
            # Adjust camera settings based on quality
            exposure = 100
            contrast = 1.0
            brightness = 0.0

            # Use very high exposure and gain settings for single camera mode for better pattern visibility
            # These values are significantly increased based on observations of underexposed images
            exposure = 500  # Significantly increased from 150 for single camera (needs more light)
            contrast = 2.0  # Higher contrast for better pattern visibility
            brightness = 0.2  # Increased brightness boost
            gain = 8.0  # Even higher gain for single camera mode to capture patterns

            if scan_quality == "high":
                exposure = 400  # High quality still needs significant exposure
                gain = 6.0      # High gain for better pattern visibility
            elif scan_quality == "ultra":
                exposure = 300  # Ultra quality needs careful exposure
                gain = 4.0      # More moderate gain for precise patterns

            logger.info(f"Using high-visibility camera settings for single camera: exposure={exposure}, gain={gain}, contrast={contrast}, brightness={brightness}")

            # Create advanced camera configuration with exposure, gain, contrast and brightness settings
            camera_config = {
                "exposure": exposure,
                "auto_exposure": False,
                "gain": gain,
                "auto_gain": False,
                "format": "png",
                "quality": 100,
                "contrast": contrast,
                "brightness": brightness,
                "color_mode": "grayscale",  # Force grayscale mode for more consistent scanning
                "force_settings": True,     # Force settings to be applied directly to camera hardware
                "denoise": False            # Disable denoise to preserve pattern details
            }

            # Apply configuration and ensure it takes effect at the hardware level
            success = camera.configure(camera_id, camera_config)
            if not success:
                logger.warning(f"Failed to configure camera {camera_id}, trying direct method")
                # Try direct method to set exposure and gain
                camera.set_exposure(camera_id, exposure, gain=gain, auto_exposure=False, auto_gain=False)

            # Ensure settings are applied by configuring camera again
            try:
                # Instead of using the unsupported CAMERA_APPLY_SETTINGS message type,
                # we'll apply the configuration again to reinforce settings
                success = camera.configure(camera_id, camera_config)
                if success:
                    logger.info(f"Applied additional configuration to camera {camera_id} to ensure settings take effect")
                else:
                    # Try once more with a delay
                    time.sleep(0.1)
                    success = camera.configure(camera_id, camera_config)
                    if success:
                        logger.info(f"Applied delayed configuration to camera {camera_id}")
                    else:
                        logger.warning(f"Failed to apply additional configuration to camera {camera_id}")

                # For single camera mode, we need to ensure good exposure
                # Try setting exposure directly one more time
                camera.set_exposure(camera_id, exposure, gain=gain, auto_exposure=False, auto_gain=False)
                logger.info(f"Reinforced exposure and gain settings for single camera mode")
            except Exception as e:
                logger.warning(f"Failed to ensure camera settings applied: {e}")
        except Exception as e:
            logger.warning(f"Camera configuration failed: {e}")

        # Prepare array for captured images
        captured_images = []

        # Display and capture for each pattern
        for i, pattern in enumerate(patterns):
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}")
            self._display_pattern(projector, pattern, i)

            # Wait for projector to update
            time.sleep(interval)

            # Capture image
            try:
                # Capture from camera
                logger.info(f"Capturing image {i+1}")
                img = camera.capture(camera_id)

                # Save image
                img_path = os.path.join(captures_dir, f"capture_{i:03d}.png")
                cv2.imwrite(img_path, img)

                # Add to array
                captured_images.append(img)

            except Exception as e:
                logger.error(f"Error capturing image: {e}")
                continue

        # Reset projector to black
        projector.show_solid_field("Black")

        # Process the scan
        if len(captured_images) < 2:
            logger.error("Not enough images captured for scanning")
            return o3dg.PointCloud()

        logger.info(f"Processing scan with {len(captured_images)} images")

        # Process with single camera scanner
        use_gray_code = pattern_type in ["gray_code", "combined"]
        use_phase_shift = pattern_type in ["phase_shift", "combined"]

        pcd = self.single_camera_scanner.process_scan(
            captured_images,
            use_gray_code=use_gray_code,
            use_phase_shift=use_phase_shift,
            mask_threshold=mask_threshold,
            output_dir=debug_dir
        )

        # If the point cloud has few points, try with more relaxed parameters
        if len(pcd.points) < 100:
            logger.warning(f"Initial scan produced only {len(pcd.points)} points. Trying with more relaxed parameters...")

            pcd = self.single_camera_scanner.process_scan(
                captured_images,
                use_gray_code=use_gray_code,
                use_phase_shift=use_phase_shift,
                mask_threshold=max(3, mask_threshold // 2),  # More relaxed threshold
                output_dir=debug_dir
            )

        # Get the final point count
        point_count = len(pcd.points) if hasattr(pcd, 'points') else 0
        logger.info(f"Generated point cloud with {point_count} points")

        # Save point cloud
        if point_count > 0:
            point_cloud_path = os.path.join(results_dir, "scan_point_cloud.ply")
            self.single_camera_scanner.save_point_cloud(pcd, point_cloud_path)
            logger.info(f"Saved point cloud to {point_cloud_path}")

            # Create mesh if we have enough points
            if point_count >= 500:
                logger.info("Creating mesh from point cloud")

                # Adjust mesh parameters based on quality
                depth = 8
                smooth_iterations = 2

                if scan_quality == "high":
                    depth = 9
                    smooth_iterations = 3
                elif scan_quality == "ultra":
                    depth = 10
                    smooth_iterations = 5

                mesh = self.single_camera_scanner.create_mesh(
                    pcd,
                    depth=depth,
                    smoothing=smooth_iterations
                )

                # Save mesh
                if len(mesh.triangles) > 0:
                    mesh_path = os.path.join(results_dir, "scan_mesh.obj")
                    self.single_camera_scanner.save_mesh(mesh, mesh_path)
                    logger.info(f"Saved mesh to {mesh_path}")

                # Visualize if requested
                if visualize and OPEN3D_AVAILABLE:
                    logger.info("Visualizing mesh")
                    o3d.visualization.draw_geometries([mesh], window_name="Single Camera Scan Mesh")

        # Visualize point cloud if requested
        if visualize and OPEN3D_AVAILABLE and point_count > 0:
            try:
                logger.info("Visualizing point cloud")
                o3d.visualization.draw_geometries([pcd], window_name="Single Camera Scan Point Cloud")
            except Exception as e:
                logger.error(f"Visualization failed: {e}")

        return pcd

    def _display_pattern(self, projector, pattern, idx):
        """Helper method to display a pattern on the projector with enhanced contrast."""
        try:
            pattern_name = pattern.get('name', '').lower()
            pattern_type = pattern.get('pattern_type', '')

            # For all structured light patterns, we want to ensure maximum possible visibility
            # This helps significantly with pattern detection on the object

            # NOTE: The previous version tried to use PROJECTOR_CONFIG which isn't supported
            # We'll use the standard projector methods which are supported by the server

            if pattern_type == "raw_image" and 'image' in pattern:
                # For raw images with binary data, try to use direct image display if available
                try:
                    # Check if the projector supports raw image display
                    if hasattr(projector, 'show_raw_image') and callable(getattr(projector, 'show_raw_image')):
                        # Use the raw image display method
                        projector.show_raw_image(pattern['image'])
                        logger.debug(f"Displayed raw image pattern {pattern_name} using show_raw_image")
                        return
                except Exception as img_err:
                    logger.warning(f"Could not display raw image directly: {img_err}, falling back to approximation")

            # Fall back to approximation methods if raw image display not available
            if pattern_type == "raw_image":
                # For raw images, we approximate with built-in patterns
                if 'white' in pattern_name:
                    # For white reference pattern, adjust for maximum brightness
                    # First set operating mode to test pattern
                    try:
                        projector.set_mode("TestPatternGenerator")
                    except Exception:
                        pass
                    projector.show_solid_field("White")
                elif 'black' in pattern_name:
                    # For black reference pattern
                    try:
                        projector.set_mode("TestPatternGenerator")
                    except Exception:
                        pass
                    projector.show_solid_field("Black")
                elif 'gray_code_x' in pattern_name or 'horizontal' in pattern_name:
                    # For horizontal Gray code patterns, use fine horizontal lines
                    # Calculate appropriate width based on bit position
                    bit_position = -1
                    if 'gray_code_x_' in pattern_name:
                        try:
                            bit_part = pattern_name.split('gray_code_x_')[1].split('_')[0]
                            if bit_part.isdigit():
                                bit_position = int(bit_part)
                        except (IndexError, ValueError):
                            pass

                    # Use finer lines for higher bits
                    if bit_position >= 0:
                        # Calculate stripe width based on bit position
                        # Higher bits need finer stripes
                        width = max(1, int(20 / (2 ** bit_position)))
                    else:
                        width = max(1, 4 - (idx % 4))

                    # Adjust for inverse patterns
                    is_inverse = 'inv' in pattern_name
                    fg_color = "Black" if is_inverse else "White"
                    bg_color = "White" if is_inverse else "Black"

                    # Use larger foreground width for better visibility
                    # This compensates for the lack of direct brightness control
                    if not is_inverse:
                        # For regular patterns, make white lines wider
                        fg_width = width + 1
                        bg_width = max(1, width - 1)
                    else:
                        # For inverse patterns, make black lines thinner
                        fg_width = max(1, width - 1)
                        bg_width = width + 1

                    projector.show_horizontal_lines(
                        foreground_color=fg_color,
                        background_color=bg_color,
                        foreground_width=fg_width,
                        background_width=bg_width
                    )
                elif 'gray_code_y' in pattern_name or 'vertical' in pattern_name:
                    # For vertical Gray code patterns, use fine vertical lines
                    # Calculate appropriate width based on bit position
                    bit_position = -1
                    if 'gray_code_y_' in pattern_name:
                        try:
                            bit_part = pattern_name.split('gray_code_y_')[1].split('_')[0]
                            if bit_part.isdigit():
                                bit_position = int(bit_part)
                        except (IndexError, ValueError):
                            pass

                    # Use finer lines for higher bits
                    if bit_position >= 0:
                        # Calculate stripe width based on bit position
                        # Higher bits need finer stripes
                        width = max(1, int(20 / (2 ** bit_position)))
                    else:
                        width = max(1, 4 - (idx % 4))

                    # Adjust for inverse patterns
                    is_inverse = 'inv' in pattern_name
                    fg_color = "Black" if is_inverse else "White"
                    bg_color = "White" if is_inverse else "Black"

                    # Use larger foreground width for better visibility
                    # This compensates for the lack of direct brightness control
                    if not is_inverse:
                        # For regular patterns, make white lines wider
                        fg_width = width + 1
                        bg_width = max(1, width - 1)
                    else:
                        # For inverse patterns, make black lines thinner
                        fg_width = max(1, width - 1)
                        bg_width = width + 1

                    projector.show_vertical_lines(
                        foreground_color=fg_color,
                        background_color=bg_color,
                        foreground_width=fg_width,
                        background_width=bg_width
                    )
                elif 'checkerboard' in pattern_name:
                    # Use a checkerboard pattern
                    projector.show_checkerboard(
                        foreground_color="White",
                        background_color="Black",
                        horizontal_count=8,
                        vertical_count=8
                    )
                elif 'phase' in pattern_name:
                    # Approximate phase pattern with vertical or horizontal sinusoidal
                    if 'h_phase' in pattern_name or 'horizontal' in pattern_name:
                        # Horizontal stripes of varying widths
                        # Get frequency if possible
                        freq = 16
                        if 'h_phase_' in pattern_name:
                            try:
                                freq_part = pattern_name.split('h_phase_')[1].split('_')[0]
                                if freq_part.isdigit():
                                    freq = int(freq_part)
                            except (IndexError, ValueError):
                                pass
                        # Adjust line width based on frequency
                        # Make white lines thicker for better visibility
                        width = max(1, int(64 / freq))
                        fw_width = width + 1  # Slightly wider white lines for better visibility
                        bg_width = max(1, width - 1)  # Slightly narrower black gaps
                        projector.show_horizontal_lines(
                            foreground_color="White",
                            background_color="Black",
                            foreground_width=fw_width,
                            background_width=bg_width
                        )
                    else:
                        # Vertical stripes of varying widths
                        # Get frequency if possible
                        freq = 16
                        if 'v_phase_' in pattern_name:
                            try:
                                freq_part = pattern_name.split('v_phase_')[1].split('_')[0]
                                if freq_part.isdigit():
                                    freq = int(freq_part)
                            except (IndexError, ValueError):
                                pass
                        # Adjust line width based on frequency
                        # Make white lines thicker for better visibility
                        width = max(1, int(64 / freq))
                        fw_width = width + 1  # Slightly wider white lines for better visibility
                        bg_width = max(1, width - 1)  # Slightly narrower black gaps
                        projector.show_vertical_lines(
                            foreground_color="White",
                            background_color="Black",
                            foreground_width=fw_width,
                            background_width=bg_width
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
                # First make sure we're in test pattern mode
                try:
                    projector.set_mode("TestPatternGenerator")
                except Exception:
                    pass

                # For predefined patterns, use the appropriate method
                if pattern_type == "solid_field":
                    projector.show_solid_field(pattern.get("color", "White"))
                elif pattern_type == "horizontal_lines":
                    # Adjust widths for optimal visibility - slightly wider white lines
                    fg_width = pattern.get("foreground_width", 4)
                    bg_width = pattern.get("background_width", 20)
                    fg_color = pattern.get("foreground_color", "White")

                    # If foreground is white, make it slightly wider for better visibility
                    if fg_color == "White":
                        fg_width += 1
                        bg_width = max(1, bg_width - 1)

                    projector.show_horizontal_lines(
                        foreground_color=fg_color,
                        background_color=pattern.get("background_color", "Black"),
                        foreground_width=fg_width,
                        background_width=bg_width
                    )
                elif pattern_type == "vertical_lines":
                    # Adjust widths for optimal visibility - slightly wider white lines
                    fg_width = pattern.get("foreground_width", 4)
                    bg_width = pattern.get("background_width", 20)
                    fg_color = pattern.get("foreground_color", "White")

                    # If foreground is white, make it slightly wider for better visibility
                    if fg_color == "White":
                        fg_width += 1
                        bg_width = max(1, bg_width - 1)

                    projector.show_vertical_lines(
                        foreground_color=fg_color,
                        background_color=pattern.get("background_color", "Black"),
                        foreground_width=fg_width,
                        background_width=bg_width
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
            # Log the pattern type being displayed
            logger.debug(f"Displayed pattern {idx}: {pattern_name} of type {pattern_type}")
        except Exception as e:
            logger.error(f"Error displaying pattern {idx} ({pattern.get('name', 'unknown')}): {e}")
    
    def _create_timestamped_folder(self, prefix: str = "scan") -> Tuple[str, str, str]:
        """
        Create a timestamped folder structure for a scan.
        
        Returns:
            Tuple of (scan_dir, captures_dir, results_dir)
        """
        # Create timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(self.output_dir, f"{prefix}_{timestamp}")
        
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