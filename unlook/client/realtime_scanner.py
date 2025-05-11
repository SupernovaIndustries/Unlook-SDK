"""
Real-time 3D Scanner (CPU-Optimized).

This module provides a simplified and reliable implementation of 3D scanning
designed for real-time operation using CPU processing only. It focuses on
robust pattern projection, capture, and decoding without requiring GPU.
"""

import os
import time
import logging
import threading
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

# Import projector adapter
from .projector_adapter import create_projector_adapter

# Configure logger
logger = logging.getLogger(__name__)

# Disable GPU usage for reliable operation
CUDA_AVAILABLE = False
logger.info("Using CPU-only mode for reliable operation")

# Import Open3D for point cloud processing if available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    logger.info(f"Open3D version: {o3d.__version__}")
except ImportError as e:
    logger.warning(f"Open3D not installed ({e}). Point cloud processing will be limited.")
    OPEN3D_AVAILABLE = False
    
    # Create placeholder for open3d when not available
    class PlaceholderO3D:
        class geometry:
            class PointCloud:
                def __init__(self):
                    self.points = []
            class TriangleMesh:
                pass
        class utility:
            class Vector3dVector:
                def __init__(self, points):
                    self.points = points
        class visualization:
            pass
        class io:
            pass
    o3d = PlaceholderO3D()

# Import structured light modules - basic pattern generation and processing
from .structured_light import Pattern, SolidPattern, GrayCodePattern, generate_patterns
from .scanner3d import ScanConfig

class RealTimeScanConfig:
    """Configuration for real-time 3D scanning with CPU-optimized settings."""

    def __init__(self):
        """Initialize with default settings for reliable CPU operation."""
        # Basic settings
        self.quality = "medium"  # Quality preset: fast, medium, high
        self.debug = False       # Enable debug output

        # Pattern generation parameters
        self.pattern_width = 1024
        self.pattern_height = 768
        self.num_gray_codes = 6       # Number of Gray code bits (horizontal)
        self.num_phase_shifts = 0     # Disable phase shifts for simplicity
        self.pattern_interval = 0.2   # Time between patterns (seconds)

        # Capture parameters
        self.capture_delay = 0.0      # Delay between projection and capture
        self.camera_exposure = 20     # Camera exposure time (milliseconds)
        self.camera_gain = 1.5        # Camera gain

        # Processing parameters
        self.downsample = True        # Enable point cloud downsampling
        self.downsample_voxel_size = 3.0  # Voxel size for downsampling (mm)
        self.epipolar_tolerance = 15.0    # Epipolar line matching tolerance (pixels)
        self.min_disparity = 5            # Minimum disparity value for valid matches
        self.max_disparity = 100          # Maximum disparity value for valid matches
        self.noise_filter = True          # Enable noise filtering
        self.noise_filter_radius = 10     # Statistical outlier radius
        self.noise_filter_std = 2.0       # Statistical outlier standard deviation multiplier
        self.skip_rectification = False   # Skip image rectification (for uncalibrated cameras)

        # Frame management
        self.continuous_scanning = False  # Continuous scanning mode
        self.max_fps = 3                 # Maximum scanning frames per second

        # Debug and troubleshooting options
        self.mask_threshold = 15         # Threshold for projector illumination detection
        self.gray_code_threshold = 20    # Threshold for Gray code bit decoding
        self.save_intermediate_images = False  # Save intermediate processing images
        self.fallback_to_default_mask = True   # Use full image mask if automatic mask fails
        self.use_adaptive_thresholding = True  # Use adaptive thresholding for better results
        self.verbose_logging = False     # Enable more detailed logging
    
    def set_quality_preset(self, quality: str):
        """Set parameters based on quality preset."""
        if quality == "fast":
            self.num_gray_codes = 4
            self.pattern_interval = 0.1
            self.downsample_voxel_size = 5.0
            self.epipolar_tolerance = 7.0
            self.noise_filter_radius = 5
            self.max_fps = 5
        elif quality == "medium":
            self.num_gray_codes = 6
            self.pattern_interval = 0.2
            self.downsample_voxel_size = 3.0
            self.epipolar_tolerance = 5.0
            self.noise_filter_radius = 10
            self.max_fps = 3
        elif quality == "high":
            self.num_gray_codes = 8
            self.pattern_interval = 0.3
            self.downsample_voxel_size = 2.0
            self.epipolar_tolerance = 3.0
            self.noise_filter_radius = 15
            self.max_fps = 1
        
        self.quality = quality
        logger.info(f"Set quality preset to {quality}")


class RealTimeScanner:
    """
    Real-time 3D scanner using structured light patterns and stereo cameras.
    
    This scanner is optimized for CPU-only operation and focuses on reliability
    rather than performance. It projects Gray code patterns, captures them using
    stereo cameras, and reconstructs 3D point clouds.
    """
    
    def __init__(
        self,
        client,
        config: Optional[RealTimeScanConfig] = None,
        calibration_file: Optional[str] = None,
        on_new_frame: Optional[Callable] = None
    ):
        """
        Initialize real-time scanner.

        Args:
            client: UnlookClient instance
            config: Scanner configuration (optional)
            calibration_file: Path to stereo calibration file (optional)
            on_new_frame: Callback for new frames (optional)
        """
        self.client = client
        self.config = config or RealTimeScanConfig()
        self.calibration_file = calibration_file
        self.on_new_frame = on_new_frame

        # Initialize projector adapter
        self.projector = create_projector_adapter(client.projector)
        logger.info("Projector adapter initialized")

        # Initialize state
        self.running = False
        self.scan_thread = None
        self.scan_count = 0
        self.fps = 0
        self.last_scan_time = 0
        self.current_point_cloud = None
        self.point_cloud_lock = threading.RLock()

        # Calibration data
        self.calibration = None
        self.map_x_left = None
        self.map_y_left = None
        self.map_x_right = None
        self.map_y_right = None
        self.projection_mat_left = None
        self.projection_mat_right = None
        self.Q = None

        # Gray code pattern generator
        self.pattern_generator = None
        self.pattern_sync_mode = "normal"  # "normal" or "strict"

        # Debug output
        # Create debug directory in current working directory for easier access
        self.debug_dir = os.path.join(os.getcwd(), "unlook_debug", f"scan_{time.strftime('%Y%m%d_%H%M%S')}")
        if self.config.debug:
            self._ensure_debug_dirs()
            logger.info(f"Debug output will be saved to: {self.debug_dir}")
            # Print absolute path for easier access
            abs_path = os.path.abspath(self.debug_dir)
            print(f"\nDEBUG INFO: Images will be saved to:\n{abs_path}\n")

        # Load calibration and initialize
        self._load_calibration()
        self._initialize_pattern_generator()

        logger.info("Real-time scanner initialized with CPU-optimized settings")
    
    def _ensure_debug_dirs(self):
        """Create necessary debug output directories."""
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "patterns"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "rectified"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "masks"), exist_ok=True)
    
    def _load_calibration(self):
        """
        Load stereo camera calibration data.
        
        This will either load from the specified calibration file or
        use a default calibration for testing.
        """
        if self.calibration_file and os.path.exists(self.calibration_file):
            logger.info(f"Loading calibration from {self.calibration_file}")
            try:
                self.calibration = self._load_calibration_file(self.calibration_file)
                
                # Generate rectification maps
                image_size = (1280, 720)  # Default image size
                M1 = self.calibration.get('camera_matrix_left', np.eye(3))
                M2 = self.calibration.get('camera_matrix_right', np.eye(3))
                d1 = self.calibration.get('dist_coeffs_left', np.zeros(5))
                d2 = self.calibration.get('dist_coeffs_right', np.zeros(5))
                R = self.calibration.get('R', np.eye(3))
                T = self.calibration.get('T', np.array([100, 0, 0]))
                
                # Compute rectification transforms
                flags = cv2.CALIB_ZERO_DISPARITY
                R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                    M1, d1, M2, d2, image_size, R, T, flags=flags, alpha=-1)
                
                # Compute rectification maps
                self.map_x_left, self.map_y_left = cv2.initUndistortRectifyMap(
                    M1, d1, R1, P1, image_size, cv2.CV_32FC1)
                self.map_x_right, self.map_y_right = cv2.initUndistortRectifyMap(
                    M2, d2, R2, P2, image_size, cv2.CV_32FC1)
                
                self.projection_mat_left = P1
                self.projection_mat_right = P2
                self.Q = Q
                
                logger.info("Calibration loaded and rectification maps generated")
                return True
                
            except Exception as e:
                logger.error(f"Error loading calibration file: {e}")
                self.calibration = None
        
        # If no calibration was loaded, use a default one for testing
        logger.warning("Using default calibration parameters")
        
        # Create a simple default calibration
        # These values won't be accurate but allow the system to function
        image_size = (1280, 720)
        focal_length = 819.2
        center = (image_size[0] / 2, image_size[1] / 2)
        M1 = np.array([[focal_length, 0, center[0]],
                       [0, focal_length, center[1]],
                       [0, 0, 1]])
        M2 = np.array([[focal_length, 0, center[0]],
                       [0, focal_length, center[1]],
                       [0, 0, 1]])
        d1 = np.zeros(5)
        d2 = np.zeros(5)
        R = np.eye(3)
        T = np.array([100.0, 0, 0])  # 100mm baseline
        
        # Generate rectification maps
        flags = cv2.CALIB_ZERO_DISPARITY
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            M1, d1, M2, d2, image_size, R, T, flags=flags, alpha=-1)
        
        # Compute rectification maps
        self.map_x_left, self.map_y_left = cv2.initUndistortRectifyMap(
            M1, d1, R1, P1, image_size, cv2.CV_32FC1)
        self.map_x_right, self.map_y_right = cv2.initUndistortRectifyMap(
            M2, d2, R2, P2, image_size, cv2.CV_32FC1)
        
        self.projection_mat_left = P1
        self.projection_mat_right = P2
        self.Q = Q
        
        # Create calibration dictionary
        self.calibration = {
            'camera_matrix_left': M1,
            'camera_matrix_right': M2,
            'dist_coeffs_left': d1,
            'dist_coeffs_right': d2,
            'R': R,
            'T': T,
            'image_size': image_size,
            'baseline': 100.0
        }
        
        return True
    
    def _load_calibration_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load calibration data from file.
        
        Args:
            file_path: Path to calibration file
            
        Returns:
            Dictionary with calibration data
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif ext == '.npz':
            data = np.load(file_path)
            return {key: data[key] for key in data.files}
        else:
            raise ValueError(f"Unsupported calibration file format: {ext}")
    
    def _initialize_pattern_generator(self):
        """Initialize pattern generator for structured light."""
        try:
            # Create OpenCV Gray code pattern generator
            self.pattern_generator = cv2.structured_light.GrayCodePattern.create(
                self.config.pattern_width, self.config.pattern_height)

            # Set white threshold for decoding
            self.pattern_generator.setWhiteThreshold(5)  # Low threshold for reliable decoding

            logger.info(f"Gray code pattern generator initialized with {self.config.num_gray_codes} bits")
        except Exception as e:
            logger.error(f"Error initializing pattern generator: {e}")
            logger.warning("Using fallback pattern generation method")
            # Set to None to indicate we should use the manual pattern generation
            self.pattern_generator = None
    
    def start(self, debug_mode: bool = False):
        """
        Start real-time scanning.
        
        Args:
            debug_mode: Enable debug mode
        """
        if self.running:
            logger.warning("Scanner is already running")
            return
        
        # Update debug flag
        if debug_mode:
            self.config.debug = True
            self._ensure_debug_dirs()
        
        # Configure camera settings for better capture
        try:
            self._configure_camera()
        except Exception as e:
            logger.error(f"Error configuring camera: {e}")
        
        # Start scanning thread
        self.running = True
        self.scan_thread = threading.Thread(target=self._scanning_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        
        logger.info("Real-time scanner started")
    
    def stop(self):
        """Stop real-time scanning."""
        self.running = False

        if self.scan_thread and self.scan_thread.is_alive():
            logger.info("Waiting for scan thread to terminate...")
            self.scan_thread.join(timeout=2.0)

        try:
            # Turn off projector pattern using the adapter
            self.projector.turn_off()
        except Exception as e:
            logger.error(f"Error turning off projector: {e}")

        logger.info("Real-time scanner stopped")
    
    def _configure_camera(self):
        """Configure cameras for structured light scanning."""
        logger.info("Configuring cameras for structured light scanning")

        # Set camera exposure mode
        try:
            if hasattr(self.client.camera, 'set_exposure_mode'):
                self.client.camera.set_exposure_mode("manual")
                logger.info("Set manual exposure mode")
            else:
                logger.warning("Camera does not support setting exposure mode")
        except Exception as e:
            logger.warning(f"Could not set exposure mode: {e}")

        # Set camera exposure and gain
        try:
            # Check for different method signatures
            if hasattr(self.client.camera, 'set_exposure'):
                try:
                    # Try with newer API (exposure_time parameter)
                    self.client.camera.set_exposure(exposure_time=self.config.camera_exposure)
                    logger.info(f"Set camera exposure to {self.config.camera_exposure}ms")
                except TypeError:
                    # Fall back to simpler API
                    self.client.camera.set_exposure(self.config.camera_exposure)
                    logger.info(f"Set camera exposure to {self.config.camera_exposure}ms (simple API)")
            else:
                logger.warning("Camera does not support setting exposure")

            # Set gain if available
            if hasattr(self.client.camera, 'set_gain'):
                self.client.camera.set_gain(self.config.camera_gain)
                logger.info(f"Set camera gain to {self.config.camera_gain}")
            else:
                logger.warning("Camera does not support setting gain")
        except Exception as e:
            logger.warning(f"Could not set exposure settings: {e}")

        # Configure image format for better quality
        try:
            if hasattr(self.client.camera, 'set_image_format'):
                self.client.camera.set_image_format(color_mode="grayscale", compression="jpeg", quality=95)
                logger.info("Set camera image format to high-quality grayscale")
            else:
                logger.warning("Camera does not support setting image format")
        except Exception as e:
            logger.warning(f"Could not set image format: {e}")
    
    def _scanning_loop(self):
        """Main scanning loop running in a separate thread."""
        logger.info("Scanning loop started")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Perform a single scan
                success = self._perform_single_scan()
                
                # Calculate FPS
                scan_time = time.time() - start_time
                self.fps = 1.0 / scan_time if scan_time > 0 else 0
                
                # Limit scan rate to avoid overloading the system
                if self.config.max_fps > 0:
                    min_interval = 1.0 / self.config.max_fps
                    if scan_time < min_interval:
                        time.sleep(min_interval - scan_time)
                
                # If not in continuous mode, break after one scan
                if not self.config.continuous_scanning:
                    if success:
                        logger.info("Single scan completed successfully")
                    else:
                        logger.warning("Single scan failed")
                    break
                
            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(1.0)  # Pause before retry
        
        logger.info("Scanning loop finished")
    
    def _perform_single_scan(self) -> bool:
        """
        Perform a single 3D scan.
        
        Returns:
            True if scan was successful, False otherwise
        """
        try:
            # 1. Project patterns and capture images
            pattern_images = self._project_and_capture_patterns()
            if not pattern_images:
                logger.error("Failed to capture pattern images")
                return False
            
            # 2. Rectify images
            left_images, right_images = self._rectify_images(pattern_images)
            
            # 3. Decode patterns to get projector coordinates
            left_coords, right_coords, mask_left, mask_right = self._decode_patterns(left_images, right_images)
            
            # 4. Find stereo correspondences
            points_left, points_right = self._find_stereo_correspondences(
                left_coords, right_coords, mask_left, mask_right,
                epipolar_tolerance=self.config.epipolar_tolerance
            )
            
            # Check if we have enough points
            if len(points_left) < 10:
                logger.warning(f"Too few correspondences found: {len(points_left)}")
                return False
            
            # 5. Triangulate 3D points
            point_cloud = self._triangulate_points(points_left, points_right)
            
            # 6. Post-process and update current point cloud
            if point_cloud is not None:
                with self.point_cloud_lock:
                    self.current_point_cloud = point_cloud
                    self.scan_count += 1
                    self.last_scan_time = time.time()
                
                # Call callback if provided
                if self.on_new_frame:
                    self.on_new_frame(point_cloud, self.scan_count, self.fps)
                
                logger.info(f"Scan #{self.scan_count} completed with {len(point_cloud.points)} points")
                return True
            else:
                logger.warning("Failed to create point cloud")
                return False
                
        except Exception as e:
            logger.error(f"Error during scan: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _project_and_capture_patterns(self) -> Dict[str, List[np.ndarray]]:
        """
        Project patterns and capture images.
        
        Returns:
            Dictionary with captured images for each pattern
        """
        logger.info("Projecting patterns and capturing images")
        
        # Generate Gray code patterns
        patterns = self._generate_gray_code_patterns()
        
        # Dictionary to store captured images
        captured_images = {
            'white': [],  # White reference images
            'black': [],  # Black reference images
            'gray_codes': []  # Gray code pattern images
        }
        
        try:
            # 1. Project and capture white pattern
            white_pattern = {
                "pattern_type": "solid_field",
                "color": "White",
                "name": "white_reference"
            }
            self.projector.project_pattern(white_pattern)

            # Apply capture delay if configured
            if hasattr(self.client, 'capture_delay') and self.client.capture_delay > 0:
                time.sleep(self.client.capture_delay)

            # Capture white reference with error handling
            try:
                # Try using capture_stereo_pair first
                if hasattr(self.client.camera, 'capture_stereo_pair'):
                    left_img, right_img = self.client.camera.capture_stereo_pair()
                else:
                    # Fall back to capturing from individual cameras
                    cameras = self.client.camera.get_cameras()
                    if len(cameras) < 2:
                        logger.error("Not enough cameras found for stereo scanning")
                        return {}

                    # Get camera IDs - get the first two cameras
                    camera_ids = [cam.get('id', f"camera_{i}") for i, cam in enumerate(cameras)][:2]
                    logger.info(f"Using cameras: {camera_ids}")

                    # Capture from both cameras
                    left_img = self.client.camera.capture_image(camera_ids[0])
                    right_img = self.client.camera.capture_image(camera_ids[1])

                # Store captured images
                captured_images['white'].append((left_img, right_img))
            except Exception as e:
                logger.error(f"Error capturing white reference: {e}")
                return {}

            # Save debug images if enabled
            if self.config.debug:
                try:
                    if left_img is not None and right_img is not None:
                        cv2.imwrite(os.path.join(self.debug_dir, "patterns", "white_left.png"), left_img)
                        cv2.imwrite(os.path.join(self.debug_dir, "patterns", "white_right.png"), right_img)
                    else:
                        logger.warning("Could not save white reference debug images - images are None")
                except Exception as e:
                    logger.warning(f"Could not save white reference debug images: {e}")

            # 2. Project and capture black pattern
            black_pattern = {
                "pattern_type": "solid_field",
                "color": "Black",
                "name": "black_reference"
            }
            self.projector.project_pattern(black_pattern)

            # Apply capture delay if configured
            if hasattr(self.client, 'capture_delay') and self.client.capture_delay > 0:
                time.sleep(self.client.capture_delay)

            # Capture black reference with error handling
            try:
                # Try using capture_stereo_pair first
                if hasattr(self.client.camera, 'capture_stereo_pair'):
                    left_img, right_img = self.client.camera.capture_stereo_pair()
                else:
                    # Fall back to capturing from individual cameras
                    cameras = self.client.camera.get_cameras()
                    if len(cameras) < 2:
                        logger.error("Not enough cameras found for stereo scanning")
                        return {}

                    # Get camera IDs - get the first two cameras
                    camera_ids = [cam.get('id', f"camera_{i}") for i, cam in enumerate(cameras)][:2]

                    # Capture from both cameras
                    left_img = self.client.camera.capture_image(camera_ids[0])
                    right_img = self.client.camera.capture_image(camera_ids[1])

                # Store captured images
                captured_images['black'].append((left_img, right_img))
            except Exception as e:
                logger.error(f"Error capturing black reference: {e}")
                return {}

            # Save debug images if enabled
            if self.config.debug:
                try:
                    if left_img is not None and right_img is not None:
                        cv2.imwrite(os.path.join(self.debug_dir, "patterns", "black_left.png"), left_img)
                        cv2.imwrite(os.path.join(self.debug_dir, "patterns", "black_right.png"), right_img)
                    else:
                        logger.warning("Could not save black reference debug images - images are None")
                except Exception as e:
                    logger.warning(f"Could not save black reference debug images: {e}")

            # 3. Project and capture Gray code patterns
            for i, pattern in enumerate(patterns):
                # Project pattern using the adapter
                self.projector.project_pattern(pattern)

                # Apply capture delay if configured
                if hasattr(self.client, 'capture_delay') and self.client.capture_delay > 0:
                    time.sleep(self.client.capture_delay)

                # Add extra delay for synchronization in strict mode
                if self.pattern_sync_mode == "strict":
                    time.sleep(0.05)

                # Capture pattern with error handling
                try:
                    # Try using capture_stereo_pair first
                    if hasattr(self.client.camera, 'capture_stereo_pair'):
                        left_img, right_img = self.client.camera.capture_stereo_pair()
                    else:
                        # Fall back to capturing from individual cameras
                        cameras = self.client.camera.get_cameras()
                        if len(cameras) < 2:
                            logger.error("Not enough cameras found for stereo scanning")
                            return {}

                        # Get camera IDs - get the first two cameras
                        camera_ids = [cam.get('id', f"camera_{i}") for i, cam in enumerate(cameras)][:2]

                        # Capture from both cameras
                        left_img = self.client.camera.capture_image(camera_ids[0])
                        right_img = self.client.camera.capture_image(camera_ids[1])

                    # Store captured images
                    captured_images['gray_codes'].append((left_img, right_img))
                except Exception as e:
                    logger.error(f"Error capturing gray code pattern: {e}")
                    # Continue with other patterns even if one fails

                # Save debug images if enabled
                if self.config.debug:
                    try:
                        if left_img is not None and right_img is not None:
                            cv2.imwrite(os.path.join(self.debug_dir, "patterns", f"pattern_{i}_left.png"), left_img)
                            cv2.imwrite(os.path.join(self.debug_dir, "patterns", f"pattern_{i}_right.png"), right_img)
                        else:
                            logger.warning(f"Could not save pattern_{i} debug images - images are None")
                    except Exception as e:
                        logger.warning(f"Could not save pattern_{i} debug images: {e}")

                # Add delay between patterns
                time.sleep(self.config.pattern_interval)
            
            logger.info(f"Captured {len(patterns)} patterns")
            return captured_images
            
        except Exception as e:
            logger.error(f"Error during pattern projection and capture: {e}")
            return {}
    
    def _generate_gray_code_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate Gray code patterns.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Generate all Gray code patterns (both regular and inverted)
        gray_code_patterns = []
        
        # Add horizontal Gray code patterns (only horizontal codes are needed)
        for bit in range(self.config.num_gray_codes):
            # Regular pattern
            gray_code_patterns.append({
                "pattern_type": "gray_code",
                "orientation": "horizontal",
                "bit": bit,
                "inverted": False,
                "name": f"gray_h_bit{bit}"
            })
            
            # Inverted pattern (for robust decoding)
            gray_code_patterns.append({
                "pattern_type": "gray_code",
                "orientation": "horizontal",
                "bit": bit,
                "inverted": True,
                "name": f"gray_h_bit{bit}_inv"
            })
        
        return gray_code_patterns
    
    def _rectify_images(self, pattern_images: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process captured pattern images with optional rectification.

        Args:
            pattern_images: Dictionary with captured images

        Returns:
            Tuple of (left_images, right_images) lists with processed images
        """
        if hasattr(self.config, 'skip_rectification') and self.config.skip_rectification:
            logger.info("Processing captured images (no rectification)")
        else:
            logger.info("Rectifying captured images")

        left_images = []
        right_images = []

        # Process reference images
        white_left, white_right = pattern_images['white'][0]
        black_left, black_right = pattern_images['black'][0]

        # Convert to grayscale if needed
        if len(white_left.shape) == 3:
            white_left = cv2.cvtColor(white_left, cv2.COLOR_BGR2GRAY)
            white_right = cv2.cvtColor(white_right, cv2.COLOR_BGR2GRAY)
            black_left = cv2.cvtColor(black_left, cv2.COLOR_BGR2GRAY)
            black_right = cv2.cvtColor(black_right, cv2.COLOR_BGR2GRAY)

        if hasattr(self.config, 'skip_rectification') and self.config.skip_rectification:
            # Skip rectification - use original images directly
            rect_white_left = white_left
            rect_white_right = white_right
            rect_black_left = black_left
            rect_black_right = black_right
        else:
            # Rectify images
            rect_white_left = cv2.remap(white_left, self.map_x_left, self.map_y_left, cv2.INTER_LINEAR)
            rect_white_right = cv2.remap(white_right, self.map_x_right, self.map_y_right, cv2.INTER_LINEAR)
            rect_black_left = cv2.remap(black_left, self.map_x_left, self.map_y_left, cv2.INTER_LINEAR)
            rect_black_right = cv2.remap(black_right, self.map_x_right, self.map_y_right, cv2.INTER_LINEAR)

        # Add reference images to the lists
        left_images.append(rect_white_left)
        right_images.append(rect_white_right)
        left_images.append(rect_black_left)
        right_images.append(rect_black_right)

        # Process Gray code pattern images
        for i, (left, right) in enumerate(pattern_images['gray_codes']):
            # Convert to grayscale if needed
            if len(left.shape) == 3:
                left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            if hasattr(self.config, 'skip_rectification') and self.config.skip_rectification:
                # Skip rectification - use original images directly
                rect_left = left
                rect_right = right
            else:
                # Rectify pattern images
                rect_left = cv2.remap(left, self.map_x_left, self.map_y_left, cv2.INTER_LINEAR)
                rect_right = cv2.remap(right, self.map_x_right, self.map_y_right, cv2.INTER_LINEAR)

            # Add to lists
            left_images.append(rect_left)
            right_images.append(rect_right)

            # Save processed images if in debug mode
            if self.config.debug:
                if hasattr(self.config, 'skip_rectification') and self.config.skip_rectification:
                    cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"orig_left_{i:03d}.png"), rect_left)
                    cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"orig_right_{i:03d}.png"), rect_right)
                else:
                    cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"rect_left_{i:03d}.png"), rect_left)
                    cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"rect_right_{i:03d}.png"), rect_right)

        if hasattr(self.config, 'skip_rectification') and self.config.skip_rectification:
            logger.info(f"Processed {len(left_images)} image pairs without rectification")
        else:
            logger.info(f"Rectified {len(left_images)} image pairs")

        return left_images, right_images
    
    def _decode_patterns(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to get projector coordinates.

        Args:
            left_images: List of processed left images
            right_images: List of processed right images

        Returns:
            Tuple of (left_coords, right_coords, mask_left, mask_right)
        """
        logger.info("Decoding Gray code patterns with enhanced sensitivity")

        # Get image dimensions
        h, w = left_images[2].shape[:2]  # Skip reference images

        # Extract reference images
        white_left, black_left = left_images[0], left_images[1]
        white_right, black_right = right_images[0], right_images[1]

        # Extract pattern images (skip reference images)
        left_patterns = left_images[2:]
        right_patterns = right_images[2:]

        # Convert pattern lists to the format expected by OpenCV
        left_patterns_cv = np.array(left_patterns)
        right_patterns_cv = np.array(right_patterns)

        # Compute shadow masks (pixels where the projector light is visible)
        # This is done by thresholding the difference between white and black reference images
        shadow_mask_left = np.zeros((h, w), dtype=np.uint8)
        shadow_mask_right = np.zeros((h, w), dtype=np.uint8)

        # Enhanced shadow mask generation for better results
        try:
            # Convert to float for safer calculation and to avoid overflows
            white_left_f = white_left.astype(np.float32)
            black_left_f = black_left.astype(np.float32)
            white_right_f = white_right.astype(np.float32)
            black_right_f = black_right.astype(np.float32)

            # Calculate normalized difference for more robust masking
            left_diff = white_left_f - black_left_f
            right_diff = white_right_f - black_right_f

            # Save raw difference images for debugging
            if self.config.debug:
                # Normalize and save raw differences
                left_diff_norm = np.clip((left_diff / np.max(left_diff) if np.max(left_diff) > 0 else left_diff) * 255, 0, 255).astype(np.uint8)
                right_diff_norm = np.clip((right_diff / np.max(right_diff) if np.max(right_diff) > 0 else right_diff) * 255, 0, 255).astype(np.uint8)

                cv2.imwrite(os.path.join(self.debug_dir, "masks", "raw_diff_left.png"), left_diff_norm)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "raw_diff_right.png"), right_diff_norm)

            # Use threshold from configuration - lowered to increase sensitivity
            threshold = max(1, self.config.mask_threshold // 2)  # More sensitive threshold
            logger.info(f"Using mask threshold: {threshold} (reduced for sensitivity)")

            # Create masks - both standard thresholding and optional adaptive approach
            if self.config.use_adaptive_thresholding:
                # Use absolute differences for better detection
                left_diff_abs = np.abs(left_diff)
                right_diff_abs = np.abs(right_diff)

                # Scale to full 8-bit range for better adaptive thresholding
                left_max = np.max(left_diff_abs) if np.max(left_diff_abs) > 0 else 1.0
                right_max = np.max(right_diff_abs) if np.max(right_diff_abs) > 0 else 1.0

                # Enhance contrast to make even small differences visible
                left_diff_enhanced = np.power(left_diff_abs / left_max, 0.5) * 255  # Square root to enhance small values
                right_diff_enhanced = np.power(right_diff_abs / right_max, 0.5) * 255

                left_diff_8bit = np.clip(left_diff_enhanced, 0, 255).astype(np.uint8)
                right_diff_8bit = np.clip(right_diff_enhanced, 0, 255).astype(np.uint8)

                # Save enhanced difference images
                if self.config.debug:
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "enhanced_diff_left.png"), left_diff_8bit)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "enhanced_diff_right.png"), right_diff_8bit)

                # Use adaptive thresholding with more sensitive parameters
                block_size = 7  # Smaller block for more detail
                c_value = -1    # Less negative C value for more sensitivity

                shadow_mask_left = cv2.adaptiveThreshold(
                    left_diff_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c_value)

                shadow_mask_right = cv2.adaptiveThreshold(
                    right_diff_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c_value)

                logger.info(f"Using enhanced adaptive thresholding (block={block_size}, C={c_value})")

                # Add direct thresholding for areas with clearly visible differences
                binary_threshold = 5  # Very low threshold for maximum sensitivity
                _, binary_mask_left = cv2.threshold(left_diff_8bit, binary_threshold, 255, cv2.THRESH_BINARY)
                _, binary_mask_right = cv2.threshold(right_diff_8bit, binary_threshold, 255, cv2.THRESH_BINARY)

                # Combine both masks (binary OR operation)
                shadow_mask_left = cv2.bitwise_or(shadow_mask_left, binary_mask_left)
                shadow_mask_right = cv2.bitwise_or(shadow_mask_right, binary_mask_right)

                # Save combined masks if in debug mode
                if self.config.debug:
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "combined_mask_left.png"), shadow_mask_left)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "combined_mask_right.png"), shadow_mask_right)

                # Apply morphological operations for better mask quality
                kernel = np.ones((3, 3), np.uint8)

                # First dilate to connect nearby regions
                shadow_mask_left = cv2.dilate(shadow_mask_left, kernel, iterations=1)
                shadow_mask_right = cv2.dilate(shadow_mask_right, kernel, iterations=1)

                # Then apply opening to remove noise
                shadow_mask_left = cv2.morphologyEx(shadow_mask_left, cv2.MORPH_OPEN, kernel)
                shadow_mask_right = cv2.morphologyEx(shadow_mask_right, cv2.MORPH_OPEN, kernel)
            else:
                # Simple fixed threshold but with normalization for better results
                max_left = np.max(left_diff)
                max_right = np.max(right_diff)

                if max_left > 0:
                    norm_left = left_diff / max_left * 255
                    shadow_mask_left[norm_left > threshold] = 255

                if max_right > 0:
                    norm_right = right_diff / max_right * 255
                    shadow_mask_right[norm_right > threshold] = 255

                logger.info(f"Using normalized thresholding - max values: left={max_left:.1f}, right={max_right:.1f}")

            # Try to enhance mask coverage if it's too low
            left_coverage = np.count_nonzero(shadow_mask_left) / (h * w) * 100
            right_coverage = np.count_nonzero(shadow_mask_right) / (h * w) * 100
            logger.info(f"Initial mask coverage: left={left_coverage:.1f}%, right={right_coverage:.1f}%")

            # If coverage is too low, try to enhance by lowering threshold
            if left_coverage < 10 or right_coverage < 10:
                logger.warning("Low mask coverage - attempting to enhance")

                # Try with even lower threshold
                lower_threshold = max(1, threshold // 2)
                logger.info(f"Trying with lower threshold: {lower_threshold}")

                # Use absolute differences for more reliable detection
                left_diff_abs = np.abs(left_diff)
                right_diff_abs = np.abs(right_diff)

                # Apply direct thresholding on absolute difference
                shadow_mask_left[left_diff_abs > lower_threshold] = 255
                shadow_mask_right[right_diff_abs > lower_threshold] = 255

                # Check new coverage
                new_left_coverage = np.count_nonzero(shadow_mask_left) / (h * w) * 100
                new_right_coverage = np.count_nonzero(shadow_mask_right) / (h * w) * 100
                logger.info(f"Enhanced mask coverage: left={new_left_coverage:.1f}%, right={new_right_coverage:.1f}%")

            # Final coverage check and detail logging
            final_left_coverage = np.count_nonzero(shadow_mask_left) / (h * w) * 100
            final_right_coverage = np.count_nonzero(shadow_mask_right) / (h * w) * 100
            logger.info(f"Final mask coverage: left={final_left_coverage:.1f}%, right={final_right_coverage:.1f}%")

            # Log detailed white-black difference stats
            logger.info(f"White-black difference stats: mean left={np.mean(left_diff):.1f}, max left={np.max(left_diff):.1f}")
            logger.info(f"White-black difference stats: mean right={np.mean(right_diff):.1f}, max right={np.max(right_diff):.1f}")

            # For extremely low coverage, use multiple fallback strategies
            if (final_left_coverage < 5 or final_right_coverage < 5) and self.config.fallback_to_default_mask:
                logger.warning("Extremely low mask coverage - trying multi-step fallback approach")

                # First attempt: Try direct difference thresholding with very low absolute threshold
                logger.info("Fallback step 1: Direct difference thresholding with minimal threshold")
                left_diff_abs = np.abs(left_diff)
                right_diff_abs = np.abs(right_diff)

                # Very minimal threshold, just to get any signal
                min_threshold = 1  # Absolute minimum threshold

                # Apply direct thresholding
                shadow_mask_left_attempt1 = np.zeros((h, w), dtype=np.uint8)
                shadow_mask_right_attempt1 = np.zeros((h, w), dtype=np.uint8)
                shadow_mask_left_attempt1[left_diff_abs > min_threshold] = 255
                shadow_mask_right_attempt1[right_diff_abs > min_threshold] = 255

                # Check coverage of this attempt
                attempt1_left_coverage = np.count_nonzero(shadow_mask_left_attempt1) / (h * w) * 100
                attempt1_right_coverage = np.count_nonzero(shadow_mask_right_attempt1) / (h * w) * 100
                logger.info(f"Fallback attempt 1 coverage: left={attempt1_left_coverage:.1f}%, right={attempt1_right_coverage:.1f}%")

                # Second attempt: Normalize and use OTSU thresholding
                logger.info("Fallback step 2: OTSU automatic thresholding")
                # Scale differences to full 8-bit range
                left_diff_scaled = np.clip((left_diff_abs / (np.max(left_diff_abs) if np.max(left_diff_abs) > 0 else 1)) * 255, 0, 255).astype(np.uint8)
                right_diff_scaled = np.clip((right_diff_abs / (np.max(right_diff_abs) if np.max(right_diff_abs) > 0 else 1)) * 255, 0, 255).astype(np.uint8)

                # Apply OTSU automatic thresholding
                _, shadow_mask_left_attempt2 = cv2.threshold(left_diff_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, shadow_mask_right_attempt2 = cv2.threshold(right_diff_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Check coverage of this attempt
                attempt2_left_coverage = np.count_nonzero(shadow_mask_left_attempt2) / (h * w) * 100
                attempt2_right_coverage = np.count_nonzero(shadow_mask_right_attempt2) / (h * w) * 100
                logger.info(f"Fallback attempt 2 coverage: left={attempt2_left_coverage:.1f}%, right={attempt2_right_coverage:.1f}%")

                # Third attempt: Create a central region of interest as last resort
                logger.info("Fallback step 3: Using central ROI as last resort")
                # Create a central region of interest as fallback
                center_x, center_y = w // 2, h // 2
                roi_size = min(w, h) // 2  # Start with half the image

                # Create ROI masks
                shadow_mask_left_attempt3 = np.zeros((h, w), dtype=np.uint8)
                shadow_mask_right_attempt3 = np.zeros((h, w), dtype=np.uint8)

                # Set central region as mask
                shadow_mask_left_attempt3[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size] = 255
                shadow_mask_right_attempt3[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size] = 255

                # Check coverage of this attempt (should be large)
                attempt3_left_coverage = np.count_nonzero(shadow_mask_left_attempt3) / (h * w) * 100
                attempt3_right_coverage = np.count_nonzero(shadow_mask_right_attempt3) / (h * w) * 100
                logger.info(f"Fallback attempt 3 coverage: left={attempt3_left_coverage:.1f}%, right={attempt3_right_coverage:.1f}%")

                # Save debug images if enabled
                if self.config.debug:
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "fallback1_left.png"), shadow_mask_left_attempt1)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "fallback1_right.png"), shadow_mask_right_attempt1)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "fallback2_left.png"), shadow_mask_left_attempt2)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "fallback2_right.png"), shadow_mask_right_attempt2)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "fallback3_left.png"), shadow_mask_left_attempt3)
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "fallback3_right.png"), shadow_mask_right_attempt3)

                # Choose the best approach: prefer attempt1 if it has reasonable coverage
                # Otherwise use attempt2, and only use attempt3 as a last resort
                if attempt1_left_coverage >= 5 and attempt1_right_coverage >= 5:
                    logger.info("Using direct thresholding fallback (attempt 1)")
                    shadow_mask_left = shadow_mask_left_attempt1
                    shadow_mask_right = shadow_mask_right_attempt1
                elif attempt2_left_coverage >= 5 and attempt2_right_coverage >= 5:
                    logger.info("Using OTSU thresholding fallback (attempt 2)")
                    shadow_mask_left = shadow_mask_left_attempt2
                    shadow_mask_right = shadow_mask_right_attempt2
                else:
                    logger.warning(f"Using central ROI of size {roi_size*2}x{roi_size*2} as last resort fallback")
                    shadow_mask_left = shadow_mask_left_attempt3
                    shadow_mask_right = shadow_mask_right_attempt3

        except Exception as e:
            logger.error(f"Error calculating shadow mask: {e}")
            # Create simple default masks as fallback
            shadow_mask_left[:] = 255
            shadow_mask_right[:] = 255
        
        # Apply morphological operations to improve the mask
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask_left = cv2.morphologyEx(shadow_mask_left, cv2.MORPH_OPEN, kernel)
        shadow_mask_right = cv2.morphologyEx(shadow_mask_right, cv2.MORPH_OPEN, kernel)
        
        # Save shadow masks if in debug mode
        if self.config.debug:
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "mask.png"), shadow_mask_left)
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "diff_raw.png"), white_left - black_left)
        
        # Create empty arrays for projector coordinates
        left_coords = np.zeros((h, w, 2), dtype=np.float32)
        right_coords = np.zeros((h, w, 2), dtype=np.float32)
        
        # Decode Gray code patterns using OpenCV
        for i in range(h):
            for j in range(w):
                # Process left image
                if shadow_mask_left[i, j] > 0:
                    try:
                        # Extract pixel values for this position
                        pixel_values = left_patterns_cv[:, i, j].astype(np.uint8)
                        
                        # Decode Gray code
                        success, proj_x, proj_y = self._decode_gray_code(pixel_values)
                        
                        if success:
                            left_coords[i, j, 0] = proj_x
                            left_coords[i, j, 1] = proj_y
                    except Exception as e:
                        # Skip errors during decoding
                        pass
                
                # Process right image
                if shadow_mask_right[i, j] > 0:
                    try:
                        # Extract pixel values for this position
                        pixel_values = right_patterns_cv[:, i, j].astype(np.uint8)
                        
                        # Decode Gray code
                        success, proj_x, proj_y = self._decode_gray_code(pixel_values)
                        
                        if success:
                            right_coords[i, j, 0] = proj_x
                            right_coords[i, j, 1] = proj_y
                    except Exception as e:
                        # Skip errors during decoding
                        pass
        
        # Save coordinate maps if in debug mode
        if self.config.debug:
            # Create a simple visualization of the coordinate maps
            coord_map = np.zeros((h, w, 3), dtype=np.uint8)
            mask = (left_coords[:, :, 0] > 0) | (left_coords[:, :, 1] > 0)
            coord_map[mask, 0] = 255  # Red channel for pixels with valid coordinates
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "coordinate_map.png"), coord_map)
        
        # Create final masks for valid coordinates
        mask_left = (left_coords[:, :, 0] > 0) & (left_coords[:, :, 1] > 0)
        mask_right = (right_coords[:, :, 0] > 0) & (right_coords[:, :, 1] > 0)
        
        valid_left = np.count_nonzero(mask_left)
        valid_right = np.count_nonzero(mask_right)
        logger.info(f"Decoded {valid_left} valid coordinates in left image")
        logger.info(f"Decoded {valid_right} valid coordinates in right image")
        
        return left_coords, right_coords, mask_left, mask_right
    
    def _decode_gray_code(self, pixel_values: np.ndarray) -> Tuple[bool, int, int]:
        """
        Decode Gray code pattern for a single pixel with enhanced robustness.
        This version is significantly improved for low-light conditions and noisy images.

        Args:
            pixel_values: Array of pixel values from Gray code patterns

        Returns:
            Tuple of (success, x, y) with projector coordinates
        """
        # Manual Gray code decoding for a single pixel
        # Each bit is encoded with two images (normal and inverted)
        # The number of patterns should be even (pairs of normal/inverted)

        if len(pixel_values) < 2:  # Need at least one bit (2 patterns)
            return False, 0, 0

        # Decode horizontal bits only (simpler)
        num_bits = len(pixel_values) // 2

        # Use an extremely lenient threshold for Gray code decoding
        # For very low light conditions, even small differences can be meaningful
        THRESHOLD = max(1, self.config.gray_code_threshold // 3)  # Even more sensitive threshold

        # Accept partial decoding with even fewer valid bits for extreme cases
        # This dramatically increases the chance of getting some valid coordinates
        required_valid_ratio = 0.3  # Only 30% of bits need to be valid - much more lenient
        min_valid_bits = max(1, int(num_bits * required_valid_ratio))

        # Decode the bits with tolerance for some errors
        decoded_bits = []
        valid_bit_count = 0
        uncertain_bits = []  # Track bits with low confidence
        confidence_scores = []  # Track confidence of each bit (0-1 scale)

        # First pass: normalize the values to increase contrast
        normalized_values = []
        for i in range(num_bits):
            normal_value = float(pixel_values[i*2]) if pixel_values[i*2] is not None else 0.0
            inverted_value = float(pixel_values[i*2 + 1]) if pixel_values[i*2 + 1] is not None else 0.0

            # Calculate normalized difference (-1 to 1 range)
            total = normal_value + inverted_value
            if total > 0:
                # Normalization formula: (normal - inverted) / (normal + inverted)
                # This gives values between -1 and 1, providing more robustness to lighting changes
                norm_diff = (normal_value - inverted_value) / total
            else:
                norm_diff = 0.0

            normalized_values.append(norm_diff)

        # Second pass: decode bits with normalized values
        for i in range(num_bits):
            normal_value = pixel_values[i*2]
            inverted_value = pixel_values[i*2 + 1]
            norm_diff = normalized_values[i]  # -1 to 1 range

            # Handle potential numerical overflow and type issues
            try:
                # Convert to standard integer types and ensure they're valid
                normal_int = int(normal_value) if normal_value is not None else 0
                inverted_int = int(inverted_value) if inverted_value is not None else 0

                # Check if values are within reasonable range (0-255 for typical grayscale)
                if not (0 <= normal_int <= 255) or not (0 <= inverted_int <= 255):
                    # Values out of expected range, but don't immediately fail
                    # Mark as uncertain and continue
                    uncertain_bits.append(i)
                    # Default to 0 for uncertain bits
                    decoded_bits.append(0)
                    confidence_scores.append(0.0)
                    continue

                # Calculate difference between normal and inverted images
                diff = normal_int - inverted_int
                abs_diff = abs(diff)

                # Calculate confidence score (0-1)
                # Higher absolute difference = higher confidence
                max_possible_diff = 255.0  # Maximum possible difference
                confidence = min(1.0, abs_diff / max_possible_diff)
                confidence_scores.append(confidence)

                # Use both the absolute difference and the normalized difference
                # for more robust decoding

                # For absolute value approach
                if diff > THRESHOLD:  # Normal brighter than inverted = 1 bit
                    decoded_bits.append(1)
                    valid_bit_count += 1
                elif diff < -THRESHOLD:  # Inverted brighter than normal = 0 bit
                    decoded_bits.append(0)
                    valid_bit_count += 1

                # If absolute difference is inconclusive, use normalized approach
                else:
                    # Use normalized value with a much smaller threshold
                    NORM_THRESHOLD = 0.05  # Just 5% difference in normalized space

                    if norm_diff > NORM_THRESHOLD:  # Normal side is stronger
                        decoded_bits.append(1)
                        valid_bit_count += 1
                    elif norm_diff < -NORM_THRESHOLD:  # Inverted side is stronger
                        decoded_bits.append(0)
                        valid_bit_count += 1
                    else:
                        # If the difference is extremely small, make a best guess
                        # based on the sign of the difference
                        if diff > 0 or norm_diff > 0:
                            decoded_bits.append(1)
                        else:
                            decoded_bits.append(0)
                        # Mark as uncertain
                        uncertain_bits.append(i)
            except (TypeError, ValueError, OverflowError) as e:
                # Handle any numerical conversion errors but don't immediately fail
                # Add a default bit and mark as uncertain
                decoded_bits.append(0)  # Default to 0 for error cases
                uncertain_bits.append(i)
                confidence_scores.append(0.0)

        # Calculate average confidence for debugging
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Check if we have enough valid bits
        if valid_bit_count < min_valid_bits:
            # Too few valid bits for reliable decoding
            return False, 0, 0

        # Convert Gray code to binary
        binary_value = self._gray_to_binary(decoded_bits)

        # Convert binary to projector coordinate
        # For simplicity, we only use horizontal coordinates (x)
        proj_x = binary_value
        proj_y = 0  # Not using vertical coordinates

        # If we're in debug mode, log some useful information occasionally
        if self.config.debug and np.random.random() < 0.001:  # Only log a small subset of pixels
            certainty = (num_bits - len(uncertain_bits)) / num_bits * 100
            logger.debug(f"Decoded with {certainty:.1f}% certainty: {decoded_bits} -> {binary_value}")

        return True, proj_x, proj_y
    
    def _gray_to_binary(self, gray_bits: List[int]) -> int:
        """
        Convert Gray code to binary.
        
        Args:
            gray_bits: List of Gray code bits
            
        Returns:
            Integer binary value
        """
        binary_bits = [gray_bits[0]]  # MSB is the same in both Gray and binary
        
        # Convert rest of the bits
        for i in range(1, len(gray_bits)):
            binary_bits.append(binary_bits[i-1] ^ gray_bits[i])  # XOR with previous binary bit
        
        # Convert binary bits to integer
        binary_value = 0
        for bit in binary_bits:
            binary_value = (binary_value << 1) | bit
        
        return binary_value
    
    def _find_stereo_correspondences(
        self,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        mask_left: np.ndarray,
        mask_right: np.ndarray,
        epipolar_tolerance: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find stereo correspondences between left and right images.

        Enhanced for non-rectified images - searches wider area for correspondences.
        This version adds multiple improvements for robustness in challenging conditions.

        Args:
            left_coords: Projector coordinates for left image
            right_coords: Projector coordinates for right image
            mask_left: Mask of valid pixels in left image
            mask_right: Mask of valid pixels in right image
            epipolar_tolerance: Tolerance for matching (pixels)

        Returns:
            Tuple of (points_left, points_right) arrays with corresponding points
        """
        logger.info("Finding stereo correspondences with enhanced robustness")

        h, w = mask_left.shape[:2]
        points_left = []
        points_right = []

        # Create disparity map for visualization
        disparity_map = np.zeros((h, w), dtype=np.float32)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        # Increase search area for challenging cases - make it significantly wider
        # This accommodates non-rectified images and potential calibration inaccuracies
        epipolar_y_range = int(epipolar_tolerance * 3)  # Much wider vertical search

        # Increase disparity range for challenging cases
        max_disparity_extended = self.config.max_disparity * 2.0
        min_disparity_relaxed = max(1, self.config.min_disparity // 2)  # More lenient minimum

        # Count the valid projector coordinates
        valid_left = np.count_nonzero(mask_left)
        valid_right = np.count_nonzero(mask_right)
        logger.info(f"Valid projector coordinates: left={valid_left}, right={valid_right}")

        # Skip pixels to speed up matching if the number of valid pixels is very large
        # This helps with performance while maintaining sufficient point density
        skip_rate = 1  # Default: process every pixel
        if valid_left > 10000:
            skip_rate = 2  # Process every 2nd pixel for large masks
        if valid_left > 50000:
            skip_rate = 3  # Process every 3rd pixel for very large masks

        logger.info(f"Using pixel skip rate of {skip_rate} for efficiency")

        # Enable sparse matching for speed
        sparse_matching = (valid_left > 20000) or (valid_right > 20000)
        logger.info(f"Using sparse matching: {sparse_matching}")

        # Create a list of candidates to match (for efficiency)
        candidates = []
        for y in range(0, h, skip_rate):
            for x in range(0, w, skip_rate):
                if not mask_left[y, x]:
                    continue

                # Get projector coordinates for this pixel
                proj_x = left_coords[y, x, 0]
                proj_y = left_coords[y, x, 1]

                # Skip if invalid
                if proj_x == 0 and proj_y == 0:
                    continue

                candidates.append((y, x, proj_x, proj_y))

        logger.info(f"Found {len(candidates)} candidate pixels for matching")

        # Adaptive matching approach based on the number of candidates
        # For very few candidates, we can be more thorough
        # For many candidates, we need to be more efficient
        use_exhaustive_search = len(candidates) < 1000
        logger.info(f"Using exhaustive search: {use_exhaustive_search}")

        # Extract right image projector coordinates for faster lookup
        right_coords_list = []
        for y_right in range(h):
            for x_right in range(w):
                if mask_right[y_right, x_right]:
                    right_proj_x = right_coords[y_right, x_right, 0]
                    right_proj_y = right_coords[y_right, x_right, 1]
                    if right_proj_x > 0 or right_proj_y > 0:
                        right_coords_list.append((y_right, x_right, right_proj_x, right_proj_y))

        logger.info(f"Found {len(right_coords_list)} right image points for matching")

        # Create a more efficient data structure for looking up right points
        # This divides the projector coordinate space into bins for faster matching
        if not use_exhaustive_search and len(right_coords_list) > 0:
            right_points_by_projcoord = {}
            bin_size = 4  # Bin size in projector coordinates

            for yr, xr, pr_x, pr_y in right_coords_list:
                bin_x = int(pr_x // bin_size)
                bin_y = int(pr_y // bin_size) if pr_y > 0 else 0
                bin_key = (bin_x, bin_y)

                if bin_key not in right_points_by_projcoord:
                    right_points_by_projcoord[bin_key] = []
                right_points_by_projcoord[bin_key].append((yr, xr, pr_x, pr_y))

            logger.info(f"Created {len(right_points_by_projcoord)} spatial bins for efficient matching")

        # Process each candidate
        match_count = 0
        multi_match_count = 0  # Count pixels with multiple possible matches

        for y, x, proj_x, proj_y in candidates:
            # Use different matching strategies based on the number of candidates
            if use_exhaustive_search:
                # Exhaustive search for few candidates
                best_matches = []  # Keep track of multiple possible matches
                best_match_dist = float('inf')

                # Define search region (with bounds checking)
                min_y = max(0, y - epipolar_y_range)
                max_y = min(h, y + epipolar_y_range + 1)

                # Search for all possible matches
                for y_right in range(min_y, max_y):
                    # Search from the left edge up to the current x position
                    # with an extended range for challenging cases
                    max_x_search = min(w, x + skip_rate)  # Allow slight rightward search for non-rectified images

                    for x_right in range(0, max_x_search, skip_rate):
                        if not mask_right[y_right, x_right]:
                            continue

                        # Get projector coordinates in right image
                        right_proj_x = right_coords[y_right, x_right, 0]
                        right_proj_y = right_coords[y_right, x_right, 1]

                        # Skip if invalid
                        if right_proj_x == 0 and right_proj_y == 0:
                            continue

                        # Check if projector coordinates match
                        # They should be the same point in the projector space
                        dist = np.sqrt((proj_x - right_proj_x)**2 + (proj_y - right_proj_y)**2)

                        # Use a sliding scale of tolerance based on distance
                        local_tolerance = epipolar_tolerance

                        # If this is a good match, add it to our list of possibilities
                        if dist < local_tolerance:
                            # Store all matches within tolerance
                            disparity = np.sqrt((x - x_right)**2 + (y - y_right)**2)

                            # Only consider matches with reasonable disparity
                            if disparity >= min_disparity_relaxed and disparity <= max_disparity_extended:
                                # Calculate confidence based on distance
                                confidence = 1.0 - (dist / local_tolerance)
                                best_matches.append((x_right, y_right, disparity, confidence, dist))

                                # Update best match distance
                                if dist < best_match_dist:
                                    best_match_dist = dist

                # Process the best matches
                if best_matches:
                    multi_match_count += 1 if len(best_matches) > 1 else 0

                    # Sort by distance (best matches first)
                    best_matches.sort(key=lambda m: m[4])

                    # Take the best match
                    best_x, best_y, best_disparity, best_confidence, _ = best_matches[0]

                    # Add to our list of correspondences
                    points_left.append([x, y])
                    points_right.append([best_x, best_y])
                    disparity_map[y, x] = best_disparity
                    confidence_map[y, x] = best_confidence
                    match_count += 1

            else:
                # Efficient search for many candidates using spatial bins
                bin_size = 4  # Same bin size as used earlier
                bin_x = int(proj_x // bin_size)
                bin_y = int(proj_y // bin_size) if proj_y > 0 else 0

                # Search in current bin and adjacent bins for better matching
                best_match = None
                best_match_dist = float('inf')

                # Check the current bin and adjacent bins
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        bin_key = (bin_x + dx, bin_y + dy)

                        if bin_key in right_points_by_projcoord:
                            for yr, xr, pr_x, pr_y in right_points_by_projcoord[bin_key]:
                                # Apply epipolar constraint - only consider points in a reasonable range
                                if abs(y - yr) > epipolar_y_range:
                                    continue

                                # Calculate projector coordinate distance
                                dist = np.sqrt((proj_x - pr_x)**2 + (proj_y - pr_y)**2)

                                # Check if this is a good match
                                if dist < epipolar_tolerance and dist < best_match_dist:
                                    # Calculate disparity
                                    disparity = np.sqrt((x - xr)**2 + (y - yr)**2)

                                    # Only consider matches with reasonable disparity
                                    if disparity >= min_disparity_relaxed and disparity <= max_disparity_extended:
                                        best_match_dist = dist
                                        best_match = (xr, yr, disparity, 1.0 - (dist / epipolar_tolerance))

                # If we found a match, add it to our list
                if best_match:
                    best_x, best_y, best_disparity, best_confidence = best_match

                    # Add to our list of correspondences
                    points_left.append([x, y])
                    points_right.append([best_x, best_y])
                    disparity_map[y, x] = best_disparity
                    confidence_map[y, x] = best_confidence
                    match_count += 1

        # Log match statistics
        if len(candidates) > 0:
            match_percentage = match_count / len(candidates) * 100
            logger.info(f"Found {match_count} correspondences ({match_percentage:.1f}% of candidates matched)")
            if multi_match_count > 0:
                logger.info(f"{multi_match_count} pixels had multiple potential matches")

        # If we have very few matches, try with more relaxed constraints
        # This is a fallback mechanism for challenging cases
        if len(points_left) < 50 and len(candidates) > 100:
            logger.warning(f"Very few matches found ({len(points_left)}), trying with relaxed constraints")

            # Double the tolerance and search range
            relaxed_tolerance = epipolar_tolerance * 2
            relaxed_epipolar_range = epipolar_y_range * 2

            # Clear previous matches
            points_left = []
            points_right = []
            disparity_map = np.zeros((h, w), dtype=np.float32)

            # Try again with relaxed parameters (simplified code for fallback)
            for y, x, proj_x, proj_y in candidates:
                # Define relaxed search region
                min_y = max(0, y - relaxed_epipolar_range)
                max_y = min(h, y + relaxed_epipolar_range + 1)

                best_match_x = -1
                best_match_y = -1
                best_match_dist = float('inf')

                # Scan the region looking for matching projector coordinates
                for y_right in range(min_y, max_y, skip_rate):
                    # Allow wider search in x direction too
                    max_x_right = min(w, x + skip_rate*2)
                    for x_right in range(0, max_x_right, skip_rate):
                        if not mask_right[y_right, x_right]:
                            continue

                        # Get projector coordinates in right image
                        right_proj_x = right_coords[y_right, x_right, 0]
                        right_proj_y = right_coords[y_right, x_right, 1]

                        # Skip if invalid
                        if right_proj_x == 0 and right_proj_y == 0:
                            continue

                        # Check if projector coordinates match with relaxed tolerance
                        dist = np.sqrt((proj_x - right_proj_x)**2 + (proj_y - right_proj_y)**2)

                        # Keep track of best match with relaxed tolerance
                        if dist < relaxed_tolerance and dist < best_match_dist:
                            best_match_dist = dist
                            best_match_x = x_right
                            best_match_y = y_right

                # If a match was found, add to the list of correspondences
                if best_match_x >= 0:
                    # Calculate disparity with very relaxed constraints
                    disparity = np.sqrt((x - best_match_x)**2 + (y - best_match_y)**2)

                    # Use even more relaxed disparity constraints
                    if disparity >= min_disparity_relaxed/2 and disparity <= max_disparity_extended*1.5:
                        points_left.append([x, y])
                        points_right.append([best_match_x, best_match_y])
                        disparity_map[y, x] = disparity

            logger.info(f"With relaxed constraints: found {len(points_left)} correspondences")

        # If we still have very few matches, create some artificial correspondences
        # as a last resort for visualization/debugging
        if len(points_left) < 10 and self.config.debug:
            logger.warning("Creating artificial correspondences for debugging (emergency fallback)")

            # Create a grid of artificial correspondences in the center of the image
            center_x, center_y = w // 2, h // 2
            grid_size = 5  # 5x5 grid
            grid_spacing = 20  # pixels

            for i in range(-grid_size, grid_size+1):
                for j in range(-grid_size, grid_size+1):
                    x_left = center_x + j * grid_spacing
                    y_left = center_y + i * grid_spacing

                    # Ensure within image bounds
                    if x_left < 0 or x_left >= w or y_left < 0 or y_left >= h:
                        continue

                    # Add artificial correspondence with a typical disparity
                    x_right = max(0, x_left - 20)  # 20 pixel fixed disparity
                    y_right = y_left  # Same row

                    points_left.append([x_left, y_left])
                    points_right.append([x_right, y_right])
                    disparity_map[y_left, x_left] = 20

            logger.info(f"Added {len(points_left)} artificial correspondences for debugging")

        # Convert to numpy arrays
        points_left = np.array(points_left).reshape(-1, 1, 2).astype(np.float32) if points_left else np.empty((0, 1, 2), dtype=np.float32)
        points_right = np.array(points_right).reshape(-1, 1, 2).astype(np.float32) if points_right else np.empty((0, 1, 2), dtype=np.float32)

        # Save disparity map if in debug mode
        if self.config.debug and len(points_left) > 0:
            # Normalize disparity for visualization
            if np.max(disparity_map) > 0:
                disparity_vis = (disparity_map * 255 / np.max(disparity_map)).astype(np.uint8)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "disparity.png"), disparity_vis)

                # Apply colormap for better visualization
                disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "disparity_color.png"), disparity_color)

                # Also save correspondence visualization
                if len(points_left) > 0:
                    vis_img = np.zeros((h, w*2, 3), dtype=np.uint8)

                    # Draw correspondences (sample up to 1000 for clarity)
                    max_vis_points = min(1000, len(points_left))
                    indices = np.linspace(0, len(points_left)-1, max_vis_points).astype(int)

                    for i in indices:
                        pt_left = points_left[i][0].astype(int)
                        pt_right = points_right[i][0].astype(int)

                        # Draw point in left image
                        cv2.circle(vis_img, (pt_left[0], pt_left[1]), 2, (0, 255, 0), -1)

                        # Draw point in right image (offset by width)
                        cv2.circle(vis_img, (pt_right[0] + w, pt_right[1]), 2, (0, 255, 0), -1)

                        # Draw line connecting them
                        cv2.line(vis_img, (pt_left[0], pt_left[1]),
                                (pt_right[0] + w, pt_right[1]), (0, 255, 255), 1)

                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "correspondences.png"), vis_img)

        logger.info(f"Found {len(points_left)} stereo correspondences")
        return points_left, points_right
    
    def _triangulate_points(self, points_left: np.ndarray, points_right: np.ndarray) -> Optional[o3d.geometry.PointCloud]:
        """
        Triangulate 3D points from stereo correspondences.

        Modified to handle non-rectified images with more robust error handling.

        Args:
            points_left: Array of points in left image
            points_right: Array of points in right image

        Returns:
            Open3D point cloud or None if triangulation failed
        """
        if len(points_left) < 4:
            logger.warning(f"Not enough correspondences for triangulation: {len(points_left)}")
            return None

        try:
            # For non-rectified images, we need to be more careful with triangulation
            # The projection matrices might not be accurate if we're not using rectification

            # Use fundamental matrix estimation for better triangulation with non-rectified images
            if len(points_left) >= 8:  # Need at least 8 points for fundamental matrix estimation
                try:
                    # Try to compute fundamental matrix for more accurate triangulation
                    F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC, 3.0)

                    if F is not None and mask is not None:
                        # Use inliers only
                        mask = mask.ravel() != 0
                        if np.sum(mask) >= 8:
                            points_left_filtered = points_left[mask]
                            points_right_filtered = points_right[mask]
                            logger.info(f"Using {np.sum(mask)} inliers from fundamental matrix estimation")

                            # Only use filtered points if we have enough of them
                            if len(points_left_filtered) >= 8:
                                points_left = points_left_filtered
                                points_right = points_right_filtered
                except Exception as e:
                    logger.warning(f"Could not estimate fundamental matrix: {e}")

            # Triangulate points
            points_4d = cv2.triangulatePoints(
                self.projection_mat_left,
                self.projection_mat_right,
                points_left,
                points_right
            )

            # Convert to 3D homogeneous coordinates
            points_3d = points_4d.T[:, 0:3] / points_4d.T[:, 3:4]

            # Validate triangulated points - remove points with very large values
            # or invalid values (NaN, inf)
            valid_mask = np.all(np.abs(points_3d) < 10000, axis=1) & \
                         np.all(~np.isnan(points_3d), axis=1) & \
                         np.all(~np.isinf(points_3d), axis=1)

            if np.sum(valid_mask) < 4:
                logger.warning(f"Too few valid triangulated points: {np.sum(valid_mask)}")
                return None

            # Filter points
            points_3d = points_3d[valid_mask]
            logger.info(f"Triangulated {len(points_3d)} valid 3D points")

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)

            # Apply statistical outlier removal if enabled
            if self.config.noise_filter and len(points_3d) > 10:
                try:
                    pcd, _ = pcd.remove_statistical_outlier(
                        nb_neighbors=self.config.noise_filter_radius,
                        std_ratio=self.config.noise_filter_std
                    )
                    logger.info(f"Applied statistical outlier removal, {len(pcd.points)} points remaining")
                except Exception as e:
                    logger.warning(f"Error during outlier removal: {e}")

            # Apply voxel downsampling if enabled
            if self.config.downsample and len(pcd.points) > 100:
                try:
                    pcd = pcd.voxel_down_sample(voxel_size=self.config.downsample_voxel_size)
                    logger.info(f"Applied voxel downsampling, {len(pcd.points)} points remaining")
                except Exception as e:
                    logger.warning(f"Error during downsampling: {e}")

            # Save point cloud if in debug mode
            if self.config.debug:
                try:
                    debug_ply = os.path.join(self.debug_dir, "point_cloud.ply")
                    o3d.io.write_point_cloud(debug_ply, pcd)
                    logger.info(f"Saved debug point cloud to {debug_ply}")
                except Exception as e:
                    logger.warning(f"Could not save debug point cloud: {e}")

            return pcd

        except Exception as e:
            logger.error(f"Error during triangulation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_current_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Get the current point cloud.
        
        Returns:
            Open3D point cloud or None if no point cloud is available
        """
        with self.point_cloud_lock:
            return self.current_point_cloud
    
    def get_fps(self) -> float:
        """
        Get the current scanning frame rate.
        
        Returns:
            Frames per second
        """
        return self.fps
    
    def get_scan_count(self) -> int:
        """
        Get the number of scans performed.
        
        Returns:
            Scan count
        """
        return self.scan_count
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostics to check the scanner status.
        
        Returns:
            Dictionary with diagnostic information
        """
        logger.info("Running scanner diagnostics")
        
        diagnostics = {
            'gpu': {
                'torch_cuda': False,
                'gpu_available': False,
                'opencv_cuda_enabled': False,
                'opencv_cuda': False,
                'open3d_cuda': False,
                'open3d_ml': False
            },
            'cameras': {
                'count': 0,
                'ids': []
            },
            'projector': {
                'test_result': 'UNKNOWN'
            },
            'calibration': {
                'loaded': False,
                'valid_camera_matrices': False,
                'valid_rotation': False,
                'valid_translation': False,
                'baseline': 0.0,
                'baseline_status': 'UNKNOWN'
            }
        }
        
        # Get camera information
        try:
            cameras = self.client.camera.get_cameras()
            diagnostics['cameras']['count'] = len(cameras)
            diagnostics['cameras']['ids'] = [cam.get('id', f"camera_{i}") for i, cam in enumerate(cameras)]
        except Exception as e:
            logger.error(f"Error getting camera information: {e}")
        
        # Check camera focus (simplified)
        try:
            focus_results = {}
            for camera_id in diagnostics['cameras']['ids']:
                # Evaluate focus based on the Laplacian variance method
                _, focus_result = self.client.camera.check_focus(camera_id)
                focus_score = focus_result[0] if isinstance(focus_result, tuple) else focus_result
                quality = "poor"
                
                if focus_score > 300:
                    quality = "excellent"
                elif focus_score > 150:
                    quality = "good"
                elif focus_score > 50:
                    quality = "moderate"
                
                focus_results[camera_id] = [float(focus_score), quality]
            
            diagnostics['focus'] = focus_results
            
            # Overall focus status
            focus_qualities = [result[1] for result in focus_results.values()]
            if all(q == "excellent" for q in focus_qualities):
                focus_status = "EXCELLENT - All cameras well focused"
            elif all(q in ["good", "excellent"] for q in focus_qualities):
                focus_status = "GOOD - All cameras adequately focused"
            elif any(q == "poor" for q in focus_qualities):
                focus_status = "POOR - Cameras need focus adjustment"
            else:
                focus_status = "MODERATE - Focus could be improved"
            
            diagnostics['focus_status'] = focus_status
        except Exception as e:
            logger.error(f"Error checking focus: {e}")
        
        # Test projector using the adapter
        try:
            # Project a simple pattern to test projector
            pattern = {"pattern_type": "solid_field", "color": "White", "name": "test_white"}
            result = self.projector.project_pattern(pattern)

            time.sleep(0.5)  # Wait for projection

            # Project back to black
            black_pattern = {"pattern_type": "solid_field", "color": "Black", "name": "test_black"}
            self.projector.project_pattern(black_pattern)

            diagnostics['projector']['test_result'] = "OK" if result else "FAILED"
        except Exception as e:
            logger.error(f"Error testing projector: {e}")
            diagnostics['projector']['test_result'] = f"ERROR: {str(e)}"
        
        # Check calibration
        if self.calibration:
            diagnostics['calibration']['loaded'] = True
            
            # Basic checks on calibration data
            diagnostics['calibration']['valid_camera_matrices'] = (
                'camera_matrix_left' in self.calibration and
                'camera_matrix_right' in self.calibration
            )
            
            diagnostics['calibration']['valid_rotation'] = ('R' in self.calibration)
            diagnostics['calibration']['valid_translation'] = ('T' in self.calibration)
            
            # Get baseline
            if 'T' in self.calibration:
                T = self.calibration['T']
                baseline = np.sqrt(T[0]**2 + T[1]**2 + T[2]**2)
                diagnostics['calibration']['baseline'] = float(baseline)
                
                # Baseline status
                if baseline > 50 and baseline < 200:
                    diagnostics['calibration']['baseline_status'] = "OK"
                elif baseline <= 0:
                    diagnostics['calibration']['baseline_status'] = "ERROR - Invalid baseline"
                elif baseline <= 50:
                    diagnostics['calibration']['baseline_status'] = "WARNING - Baseline too small"
                else:
                    diagnostics['calibration']['baseline_status'] = "WARNING - Baseline unusually large"
            
            # Add calibration file info
            diagnostics['calibration']['file'] = self.calibration_file
            
            # Add optical parameters
            if 'camera_matrix_left' in self.calibration:
                M = self.calibration['camera_matrix_left']
                focal_length = (M[0, 0] + M[1, 1]) / 2
                diagnostics['calibration']['focal_length_left'] = float(focal_length)
            
            if 'camera_matrix_right' in self.calibration:
                M = self.calibration['camera_matrix_right']
                focal_length = (M[0, 0] + M[1, 1]) / 2
                diagnostics['calibration']['focal_length_right'] = float(focal_length)
        
        # Save diagnostics to file if in debug mode
        if self.config.debug:
            with open(os.path.join(self.debug_dir, "diagnostics.json"), "w") as f:
                json.dump(diagnostics, f, indent=2)
        
        return diagnostics


def create_realtime_scanner(
    client,
    config=None,
    calibration_file=None,
    on_new_frame=None
) -> RealTimeScanner:
    """
    Create a real-time 3D scanner.
    
    Args:
        client: UnlookClient instance
        config: Scanner configuration (optional)
        calibration_file: Path to stereo calibration file (optional)
        on_new_frame: Callback for new frames (optional)
    
    Returns:
        RealTimeScanner instance
    """
    if config is None:
        config = RealTimeScanConfig()
    
    return RealTimeScanner(
        client=client,
        config=config,
        calibration_file=calibration_file,
        on_new_frame=on_new_frame
    )