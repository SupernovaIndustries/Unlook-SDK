"""
Static 3D Scanner for Unlook SDK.

This module provides a simplified implementation for static 3D scanning
designed for high-quality single scans. Unlike real-time scanning, this
approach captures all patterns at once then processes them for maximum
accuracy. It leverages the stereo camera calibration, structured light
patterns, and 3D reconstruction techniques.
"""

import os
import time
import logging
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable

# Import Open3D for point cloud processing if available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    logging.info(f"Open3D version: {o3d.__version__}")
except ImportError as e:
    logging.warning(f"Open3D not installed ({e}). Using limited point cloud functionality.")
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

# Import projector adapter
from .projector_adapter import create_projector_adapter

# Import enhanced gray code patterns
from .enhanced_gray_code import generate_enhanced_gray_code_patterns

# Import calibration module
from .camera_calibration import StereoCalibrator, load_calibration

# Import GPU acceleration and neural network modules
try:
    from .gpu_utils import get_gpu_accelerator, is_gpu_available
    GPU_AVAILABLE = is_gpu_available()
    if GPU_AVAILABLE:
        logging.info("GPU acceleration is available")
    else:
        logging.warning("GPU acceleration not available")
except ImportError:
    logging.warning("GPU acceleration module not found. Running on CPU only.")
    GPU_AVAILABLE = False

try:
    from .nn_processing import get_point_cloud_processor, is_nn_processing_available
    NN_AVAILABLE = is_nn_processing_available()
    if NN_AVAILABLE:
        logging.info("Neural network processing is available")
    else:
        logging.warning("Neural network processing not available")
except ImportError:
    logging.warning("Neural network module not found. Advanced processing disabled.")
    NN_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

class StaticScanConfig:
    """Configuration for static 3D scanning with high-quality settings."""

    def __init__(self):
        """Initialize with default settings for high-quality static scanning."""
        # Basic settings
        self.quality = "high"  # Quality preset: medium, high, ultra
        self.debug = False     # Enable debug output

        # Pattern generation parameters
        self.pattern_width = 1024
        self.pattern_height = 768
        self.num_gray_codes = 10      # More bits for higher quality
        self.use_phase_shift = True   # Enable phase shift patterns for increased accuracy
        self.num_phase_shifts = 4     # Number of phase shift patterns
        self.pattern_interval = 0.3   # Time between patterns (seconds)

        # Capture parameters
        self.capture_delay = 0.3      # Delay between projection and capture
        self.camera_exposure = 20     # Camera exposure time (milliseconds)
        self.camera_gain = 1.5        # Camera gain

        # Processing parameters
        self.downsample = True        # Enable point cloud downsampling
        self.downsample_voxel_size = 1.0  # Voxel size for downsampling (mm)
        self.epipolar_tolerance = 3.0     # Epipolar line matching tolerance (pixels)
        self.min_disparity = 5        # Minimum disparity value for valid matches
        self.max_disparity = 100      # Maximum disparity value for valid matches
        self.noise_filter = True      # Enable noise filtering
        self.noise_filter_radius = 20 # Statistical outlier radius
        self.noise_filter_std = 2.0   # Statistical outlier standard deviation multiplier
        self.skip_rectification = False  # Skip image rectification (for uncalibrated cameras)

        # Debug and troubleshooting options
        self.mask_threshold = 10      # Threshold for projector illumination detection
        self.gray_code_threshold = 15 # Threshold for Gray code bit decoding
        self.save_intermediate_images = True  # Save intermediate processing images
        self.use_adaptive_thresholding = True  # Use adaptive thresholding for better results

        # Output options
        self.output_format = "ply"    # Output format for point clouds (ply, pcd)
        self.texture_mapping = True   # Enable texture mapping for colored point clouds

        # GPU and neural network options
        self.use_gpu = True                # Use GPU acceleration if available
        self.use_neural_network = False    # Use neural network processing if available
        self.nn_denoise = True             # Apply neural network denoising
        self.nn_fill_holes = True          # Apply neural network hole filling
        self.nn_enhance_details = False    # Apply neural network detail enhancement
        self.nn_model_path = None          # Path to neural network model weights
        self.ml_backend = None             # Preferred ML backend (pytorch or tensorflow)
    
    def set_quality_preset(self, quality: str):
        """Set parameters based on quality preset."""
        if quality == "medium":
            self.num_gray_codes = 8
            self.use_phase_shift = False
            self.pattern_interval = 0.2
            self.capture_delay = 0.2
            self.downsample_voxel_size = 2.0
            self.epipolar_tolerance = 5.0
            self.noise_filter_radius = 15
            self.use_gpu = True
            self.use_neural_network = False
        elif quality == "high":
            self.num_gray_codes = 10
            self.use_phase_shift = True
            self.num_phase_shifts = 4
            self.pattern_interval = 0.3
            self.capture_delay = 0.3
            self.downsample_voxel_size = 1.0
            self.epipolar_tolerance = 3.0
            self.noise_filter_radius = 20
            self.use_gpu = True
            self.use_neural_network = False
        elif quality == "ultra":
            self.num_gray_codes = 12
            self.use_phase_shift = True
            self.num_phase_shifts = 8
            self.pattern_interval = 0.4
            self.capture_delay = 0.4
            self.downsample_voxel_size = 0.5
            self.epipolar_tolerance = 1.0
            self.noise_filter_radius = 30
            self.use_gpu = True
            self.use_neural_network = True
            self.nn_denoise = True
            self.nn_fill_holes = True
            self.nn_enhance_details = False  # Keep this off by default until we have a good model

        self.quality = quality
        logger.info(f"Set quality preset to {quality}")


class StaticScanner:
    """
    Static 3D scanner using structured light patterns and stereo cameras.
    
    This scanner is optimized for high-quality single scans rather than
    real-time performance. It captures a full sequence of patterns before
    processing to generate a detailed 3D point cloud.
    """
    
    def __init__(
        self,
        client,
        config: Optional[StaticScanConfig] = None,
        calibration_file: Optional[str] = None
    ):
        """
        Initialize static scanner.

        Args:
            client: UnlookClient instance
            config: Scanner configuration (optional)
            calibration_file: Path to stereo calibration file (optional)
        """
        self.client = client
        self.config = config or StaticScanConfig()
        self.calibration_file = calibration_file

        # Initialize projector adapter
        self.projector = create_projector_adapter(client.projector)
        logger.info("Projector adapter initialized")

        # Initialize state
        self.point_cloud = None
        self.captured_images = {}
        self.processing_time = 0

        # Create StereoCalibrator instance
        self.calibrator = None
        if calibration_file:
            self.calibrator = load_calibration(calibration_file)
            if self.calibrator:
                logger.info(f"Loaded calibration from {calibration_file}")
            else:
                logger.warning(f"Failed to load calibration from {calibration_file}, using default values")
                self.calibrator = StereoCalibrator()  # Create with default values
        else:
            logger.warning("No calibration file provided, using default values")
            self.calibrator = StereoCalibrator()  # Create with default values

        # Initialize GPU acceleration if available and enabled
        self.gpu_accelerator = None
        if GPU_AVAILABLE and self.config.use_gpu:
            logger.info("Initializing GPU acceleration")
            try:
                self.gpu_accelerator = get_gpu_accelerator(enable_gpu=True)
                logger.info("GPU acceleration initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")

        # Initialize neural network processor if available and enabled
        self.nn_processor = None
        if NN_AVAILABLE and self.config.use_neural_network:
            logger.info("Initializing neural network processing")
            try:
                self.nn_processor = get_point_cloud_processor(
                    use_gpu=self.config.use_gpu,
                    model_path=self.config.nn_model_path,
                    ml_backend=self.config.ml_backend
                )
                logger.info("Neural network processing initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize neural network processing: {e}")

        # Debug output
        self.debug_dir = os.path.join(os.getcwd(), "unlook_debug", f"static_scan_{time.strftime('%Y%m%d_%H%M%S')}")
        if self.config.debug or self.config.save_intermediate_images:
            self._ensure_debug_dirs()
            logger.info(f"Debug output will be saved to: {self.debug_dir}")
            # Print absolute path for easier access
            abs_path = os.path.abspath(self.debug_dir)
            print(f"\nDEBUG INFO: Images will be saved to:\n{abs_path}\n")
    
    def _ensure_debug_dirs(self):
        """Create necessary debug output directories."""
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "patterns"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "rectified"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "masks"), exist_ok=True)
    
    def _configure_camera(self):
        """Configure cameras for structured light scanning."""
        logger.info("Configuring cameras for static scanning")

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
    
    def generate_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate structured light patterns for static scanning.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Add white and black reference patterns first
        patterns.append({
            "pattern_type": "solid_field",
            "color": "White",
            "name": "white_reference"
        })
        
        patterns.append({
            "pattern_type": "solid_field",
            "color": "Black",
            "name": "black_reference"
        })
        
        # Add Gray code patterns (both horizontal and vertical)
        # Horizontal patterns
        for bit in range(self.config.num_gray_codes):
            # Regular pattern
            patterns.append({
                "pattern_type": "gray_code",
                "orientation": "horizontal",
                "bit": bit,
                "inverted": False,
                "name": f"gray_h_bit{bit}"
            })
            
            # Inverted pattern (for robust decoding)
            patterns.append({
                "pattern_type": "gray_code",
                "orientation": "horizontal",
                "bit": bit,
                "inverted": True,
                "name": f"gray_h_bit{bit}_inv"
            })
        
        # Add phase shift patterns if enabled
        if self.config.use_phase_shift:
            # Add phase shift patterns for increased sub-pixel accuracy
            for i in range(self.config.num_phase_shifts):
                phase = i * (2 * np.pi / self.config.num_phase_shifts)
                patterns.append({
                    "pattern_type": "phase_shift",
                    "orientation": "horizontal",
                    "phase": phase,
                    "frequency": 16,  # Higher frequency for more detail
                    "name": f"phase_h_{i}"
                })
        
        logger.info(f"Generated {len(patterns)} structured light patterns")
        return patterns
    
    def capture_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Project and capture all patterns for static scanning.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Dictionary with captured images for each pattern
        """
        logger.info(f"Capturing {len(patterns)} patterns for static scanning")
        
        # Dictionary to store captured images
        captured_images = {
            'white': [],  # White reference images
            'black': [],  # Black reference images
            'gray_codes': [],  # Gray code pattern images
            'phase_shifts': []  # Phase shift pattern images
        }
        
        # Project and capture each pattern
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get("pattern_type")
            pattern_name = pattern.get("name", f"pattern_{i}")
            
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern_name}")
            
            # Project pattern
            self.projector.project_pattern(pattern)
            
            # Apply capture delay
            time.sleep(self.config.capture_delay)
            
            # Capture image
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
                
                # Convert to grayscale if needed
                if len(left_img.shape) == 3:
                    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                else:
                    left_img_gray = left_img
                
                if len(right_img.shape) == 3:
                    right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                else:
                    right_img_gray = right_img
                
                # Store images based on pattern type
                if pattern_type == "solid_field":
                    if "white" in pattern_name.lower():
                        captured_images['white'].append((left_img_gray, right_img_gray))
                    elif "black" in pattern_name.lower():
                        captured_images['black'].append((left_img_gray, right_img_gray))
                elif pattern_type == "gray_code":
                    captured_images['gray_codes'].append((left_img_gray, right_img_gray))
                elif pattern_type == "phase_shift":
                    captured_images['phase_shifts'].append((left_img_gray, right_img_gray))
                
                # Save debug images if enabled
                if self.config.debug or self.config.save_intermediate_images:
                    cv2.imwrite(os.path.join(self.debug_dir, "patterns", f"{pattern_name}_left.png"), left_img_gray)
                    cv2.imwrite(os.path.join(self.debug_dir, "patterns", f"{pattern_name}_right.png"), right_img_gray)
            
            except Exception as e:
                logger.error(f"Error capturing pattern {pattern_name}: {e}")
                return {}
            
            # Wait between patterns
            time.sleep(self.config.pattern_interval)
        
        # Turn off projector
        self.projector.turn_off()
        
        # Store captured images for later use
        self.captured_images = captured_images
        
        logger.info(f"Successfully captured {len(patterns)} patterns")
        return captured_images
    
    def rectify_images(self, captured_images: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process captured pattern images with rectification if calibration is available.

        Args:
            captured_images: Dictionary with captured images

        Returns:
            Tuple of (left_images, right_images) lists with processed images
        """
        if self.config.skip_rectification or self.calibrator is None:
            logger.info("Processing captured images (no rectification)")
            skip_rectification = True
        else:
            logger.info("Rectifying captured images using calibration data")
            skip_rectification = False
        
        left_images = []
        right_images = []
        
        # Process reference images
        if not captured_images['white'] or not captured_images['black']:
            logger.error("Missing white or black reference images")
            return [], []
        
        white_left, white_right = captured_images['white'][0]
        black_left, black_right = captured_images['black'][0]
        
        if skip_rectification:
            # Skip rectification - use original images directly
            rect_white_left = white_left
            rect_white_right = white_right
            rect_black_left = black_left
            rect_black_right = black_right
        else:
            # Rectify images using calibration data
            rect_white_left, rect_white_right = self.calibrator.undistort_rectify_image_pair(white_left, white_right)
            rect_black_left, rect_black_right = self.calibrator.undistort_rectify_image_pair(black_left, black_right)
        
        # Add reference images to the lists
        left_images.append(rect_white_left)
        right_images.append(rect_white_right)
        left_images.append(rect_black_left)
        right_images.append(rect_black_right)
        
        # Process Gray code pattern images
        for i, (left, right) in enumerate(captured_images['gray_codes']):
            if skip_rectification:
                # Skip rectification - use original images directly
                rect_left = left
                rect_right = right
            else:
                # Rectify pattern images
                rect_left, rect_right = self.calibrator.undistort_rectify_image_pair(left, right)
            
            # Add to lists
            left_images.append(rect_left)
            right_images.append(rect_right)
            
            # Save processed images if in debug mode
            if self.config.debug or self.config.save_intermediate_images:
                prefix = "orig" if skip_rectification else "rect"
                cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"{prefix}_left_{i:03d}.png"), rect_left)
                cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"{prefix}_right_{i:03d}.png"), rect_right)
        
        # Process phase shift pattern images if available
        for i, (left, right) in enumerate(captured_images.get('phase_shifts', [])):
            if skip_rectification:
                # Skip rectification - use original images directly
                rect_left = left
                rect_right = right
            else:
                # Rectify pattern images
                rect_left, rect_right = self.calibrator.undistort_rectify_image_pair(left, right)
            
            # Add to lists
            left_images.append(rect_left)
            right_images.append(rect_right)
            
            # Save processed images if in debug mode
            if self.config.debug or self.config.save_intermediate_images:
                prefix = "orig" if skip_rectification else "rect"
                cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"{prefix}_phase_left_{i:03d}.png"), rect_left)
                cv2.imwrite(os.path.join(self.debug_dir, "rectified", f"{prefix}_phase_right_{i:03d}.png"), rect_right)
        
        logger.info(f"Processed {len(left_images)} image pairs" + 
                  ("" if skip_rectification else " with rectification"))
        
        return left_images, right_images
    
    def decode_patterns(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode structured light patterns to get projector coordinates.

        Args:
            left_images: List of processed left images
            right_images: List of processed right images

        Returns:
            Tuple of (left_coords, right_coords, mask_left, mask_right)
        """
        logger.info("Decoding structured light patterns")
        
        # Get image dimensions
        h, w = left_images[2].shape[:2]  # Skip reference images
        
        # Extract reference images
        white_left, black_left = left_images[0], left_images[1]
        white_right, black_right = right_images[0], right_images[1]
        
        # Compute shadow masks (pixels where the projector light is visible)
        # This is done by thresholding the difference between white and black reference images
        shadow_mask_left = np.zeros((h, w), dtype=np.uint8)
        shadow_mask_right = np.zeros((h, w), dtype=np.uint8)
        
        # Enhanced shadow mask generation
        # Convert to float for safer calculation
        white_left_f = white_left.astype(np.float32)
        black_left_f = black_left.astype(np.float32)
        white_right_f = white_right.astype(np.float32)
        black_right_f = black_right.astype(np.float32)
        
        # Calculate difference
        left_diff = white_left_f - black_left_f
        right_diff = white_right_f - black_right_f
        
        # Save raw difference images for debugging
        if self.config.debug or self.config.save_intermediate_images:
            left_diff_norm = np.clip((left_diff / np.max(left_diff) if np.max(left_diff) > 0 else left_diff) * 255, 0, 255).astype(np.uint8)
            right_diff_norm = np.clip((right_diff / np.max(right_diff) if np.max(right_diff) > 0 else right_diff) * 255, 0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "raw_diff_left.png"), left_diff_norm)
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "raw_diff_right.png"), right_diff_norm)
        
        # Use adaptive thresholding for better mask generation
        if self.config.use_adaptive_thresholding:
            # Scale differences to full 8-bit range
            left_diff_abs = np.abs(left_diff)
            right_diff_abs = np.abs(right_diff)
            
            left_max = np.max(left_diff_abs) if np.max(left_diff_abs) > 0 else 1.0
            right_max = np.max(right_diff_abs) if np.max(right_diff_abs) > 0 else 1.0
            
            # Enhance contrast
            left_diff_enhanced = np.power(left_diff_abs / left_max, 0.5) * 255
            right_diff_enhanced = np.power(right_diff_abs / right_max, 0.5) * 255
            
            left_diff_8bit = np.clip(left_diff_enhanced, 0, 255).astype(np.uint8)
            right_diff_8bit = np.clip(right_diff_enhanced, 0, 255).astype(np.uint8)
            
            # Save enhanced difference images
            if self.config.debug or self.config.save_intermediate_images:
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "enhanced_diff_left.png"), left_diff_8bit)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "enhanced_diff_right.png"), right_diff_8bit)
            
            # Use adaptive thresholding
            block_size = 7  # Smaller block for more detail
            c_value = -1    # Less negative C value for more sensitivity
            
            shadow_mask_left = cv2.adaptiveThreshold(
                left_diff_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_value)
            
            shadow_mask_right = cv2.adaptiveThreshold(
                right_diff_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_value)
            
            # Add direct thresholding for areas with clearly visible differences
            binary_threshold = self.config.mask_threshold  # Threshold from config
            _, binary_mask_left = cv2.threshold(left_diff_8bit, binary_threshold, 255, cv2.THRESH_BINARY)
            _, binary_mask_right = cv2.threshold(right_diff_8bit, binary_threshold, 255, cv2.THRESH_BINARY)
            
            # Combine both masks
            shadow_mask_left = cv2.bitwise_or(shadow_mask_left, binary_mask_left)
            shadow_mask_right = cv2.bitwise_or(shadow_mask_right, binary_mask_right)
            
            # Save combined masks
            if self.config.debug or self.config.save_intermediate_images:
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "combined_mask_left.png"), shadow_mask_left)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "combined_mask_right.png"), shadow_mask_right)
        else:
            # Simple fixed threshold
            threshold = self.config.mask_threshold
            shadow_mask_left[left_diff > threshold] = 255
            shadow_mask_right[right_diff > threshold] = 255
        
        # Apply morphological operations for better mask quality
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask_left = cv2.morphologyEx(shadow_mask_left, cv2.MORPH_OPEN, kernel)
        shadow_mask_right = cv2.morphologyEx(shadow_mask_right, cv2.MORPH_OPEN, kernel)
        
        # Save shadow masks
        if self.config.debug or self.config.save_intermediate_images:
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "mask_left.png"), shadow_mask_left)
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "mask_right.png"), shadow_mask_right)
        
        # Extract pattern images (skip reference images)
        left_patterns = left_images[2:]
        right_patterns = right_images[2:]
        
        # Convert pattern lists to numpy arrays
        left_patterns_cv = np.array(left_patterns)
        right_patterns_cv = np.array(right_patterns)
        
        # Create arrays for projector coordinates
        left_coords = np.zeros((h, w, 2), dtype=np.float32)
        right_coords = np.zeros((h, w, 2), dtype=np.float32)
        
        # Process Gray code patterns
        num_bits = self.config.num_gray_codes
        gray_code_pairs = len(left_patterns) // 2  # Each bit has regular and inverted pattern
        
        # Iterate through each pixel
        for y in range(h):
            for x in range(w):
                # Process left image
                if shadow_mask_left[y, x] > 0:
                    # Decode Gray code
                    gray_code = 0
                    for bit in range(num_bits):
                        # Get normal and inverted pattern values
                        normal_idx = bit * 2
                        inverted_idx = bit * 2 + 1
                        
                        if normal_idx < len(left_patterns) and inverted_idx < len(left_patterns):
                            normal_val = left_patterns[normal_idx][y, x]
                            inverted_val = left_patterns[inverted_idx][y, x]
                            
                            # Decode bit - use safe integer comparison to avoid overflow
                            normal_int = int(normal_val)
                            inverted_int = int(inverted_val)
                            threshold = int(self.config.gray_code_threshold)

                            if normal_int > inverted_int + threshold:
                                # Set bit if normal is brighter than inverted
                                gray_code |= (1 << bit)
                    
                    # Store projector coordinates
                    left_coords[y, x, 0] = gray_code
                
                # Process right image
                if shadow_mask_right[y, x] > 0:
                    # Decode Gray code
                    gray_code = 0
                    for bit in range(num_bits):
                        # Get normal and inverted pattern values
                        normal_idx = bit * 2
                        inverted_idx = bit * 2 + 1
                        
                        if normal_idx < len(right_patterns) and inverted_idx < len(right_patterns):
                            normal_val = right_patterns[normal_idx][y, x]
                            inverted_val = right_patterns[inverted_idx][y, x]
                            
                            # Decode bit - use safe integer comparison to avoid overflow
                            normal_int = int(normal_val)
                            inverted_int = int(inverted_val)
                            threshold = int(self.config.gray_code_threshold)

                            if normal_int > inverted_int + threshold:
                                # Set bit if normal is brighter than inverted
                                gray_code |= (1 << bit)
                    
                    # Store projector coordinates
                    right_coords[y, x, 0] = gray_code
        
        # Save coordinate maps if in debug mode
        if self.config.debug or self.config.save_intermediate_images:
            # Create visualization of coordinate maps
            left_coord_vis = np.clip((left_coords[:, :, 0] * 255 / (2**num_bits - 1)), 0, 255).astype(np.uint8)
            right_coord_vis = np.clip((right_coords[:, :, 0] * 255 / (2**num_bits - 1)), 0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "coordinate_map_left.png"), left_coord_vis)
            cv2.imwrite(os.path.join(self.debug_dir, "masks", "coordinate_map_right.png"), right_coord_vis)
        
        # Create masks for valid coordinates
        mask_left = left_coords[:, :, 0] > 0
        mask_right = right_coords[:, :, 0] > 0
        
        logger.info(f"Decoded projector coordinates: {np.sum(mask_left)} left, {np.sum(mask_right)} right")
        
        return left_coords, right_coords, mask_left, mask_right
    
    def find_correspondences(self, left_coords: np.ndarray, right_coords: np.ndarray,
                           mask_left: np.ndarray, mask_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find stereo correspondences between left and right images.

        Args:
            left_coords: Projector coordinates for left image
            right_coords: Projector coordinates for right image
            mask_left: Mask of valid pixels in left image
            mask_right: Mask of valid pixels in right image

        Returns:
            Tuple of (points_left, points_right) arrays with corresponding points
        """
        logger.info("Finding stereo correspondences")

        # Try to use GPU-accelerated version first if available
        if self.gpu_accelerator is not None and self.config.use_gpu:
            logger.info("Using GPU-accelerated correspondence matching")
            start_time = time.time()

            # Call GPU-accelerated implementation
            points_left_gpu, points_right_gpu = self.gpu_accelerator.find_correspondences_gpu(
                left_coords, right_coords, mask_left, mask_right,
                epipolar_tolerance=self.config.epipolar_tolerance,
                min_disparity=self.config.min_disparity,
                max_disparity=self.config.max_disparity,
                gray_code_threshold=self.config.gray_code_threshold
            )

            # If GPU implementation was successful, return its results
            if points_left_gpu is not None and points_right_gpu is not None:
                elapsed = time.time() - start_time
                logger.info(f"GPU correspondence matching found {len(points_left_gpu)} matches in {elapsed:.2f} seconds")
                return points_left_gpu, points_right_gpu
            else:
                logger.warning("GPU correspondence matching failed, falling back to CPU implementation")

        # Use CPU implementation if GPU not available or failed
        start_time = time.time()
        logger.info("Using CPU correspondence matching")

        h, w = mask_left.shape[:2]
        points_left = []
        points_right = []

        # Create disparity map for visualization
        disparity_map = np.zeros((h, w), dtype=np.float32)

        # Use epipolar tolerance from config
        epipolar_tolerance = self.config.epipolar_tolerance
        min_disparity = self.config.min_disparity
        max_disparity = self.config.max_disparity

        # Pre-compute valid points to improve performance
        valid_left_indices = np.where(mask_left)
        valid_left_y, valid_left_x = valid_left_indices
        valid_left_count = len(valid_left_y)

        # Process in smaller batches for better performance
        batch_size = 1000
        total_matches = 0

        for batch_start in range(0, valid_left_count, batch_size):
            batch_end = min(batch_start + batch_size, valid_left_count)
            batch_indices = range(batch_start, batch_end)

            # Process this batch of points
            for i in batch_indices:
                y, x = valid_left_y[i], valid_left_x[i]

                # Get projector coordinate for this pixel
                proj_coord = left_coords[y, x, 0]
                if proj_coord == 0:
                    continue

                # Define epipolar search region
                min_y = max(0, y - int(epipolar_tolerance))
                max_y = min(h - 1, y + int(epipolar_tolerance))

                best_match_x = -1
                best_match_y = -1
                best_match_dist = float('inf')

                # Search along epipolar line more efficiently
                for y_right in range(min_y, max_y + 1):
                    # Get valid pixels in the right image on this epipolar line
                    line_mask = mask_right[y_right, :min(x, w - 1)]
                    if not np.any(line_mask):
                        continue

                    # Only process the valid pixels
                    valid_x_rights = np.where(line_mask)[0]
                    if len(valid_x_rights) == 0:
                        continue

                    # Get all projector coordinates for valid pixels on this line
                    right_coords_line = right_coords[y_right, valid_x_rights, 0]

                    # Calculate distances for all valid pixels at once
                    dists = np.abs(proj_coord - right_coords_line)

                    # Find the best match
                    if len(dists) > 0:
                        min_idx = np.argmin(dists)
                        min_dist = dists[min_idx]
                        x_right = valid_x_rights[min_idx]

                        # Update best match if better than current best
                        if min_dist < best_match_dist and min_dist < self.config.gray_code_threshold:
                            # Calculate disparity
                            disparity = x - x_right

                            # Only consider matches with reasonable disparity
                            if disparity >= min_disparity and disparity <= max_disparity:
                                best_match_dist = min_dist
                                best_match_x = x_right
                                best_match_y = y_right

                # If a match was found, add to correspondences
                if best_match_x >= 0:
                    points_left.append([x, y])
                    points_right.append([best_match_x, best_match_y])
                    disparity_map[y, x] = x - best_match_x
                    total_matches += 1

            # Report progress for large datasets
            if valid_left_count > 10000 and (batch_end % 10000 == 0 or batch_end == valid_left_count):
                progress = 100 * batch_end / valid_left_count
                elapsed = time.time() - start_time
                logger.info(f"Correspondence matching progress: {progress:.1f}% ({batch_end}/{valid_left_count}), "
                          f"matches so far: {total_matches}, elapsed: {elapsed:.1f}s")
        
        # Save disparity map if in debug mode
        if self.config.debug or self.config.save_intermediate_images:
            if np.max(disparity_map) > 0:
                # Normalize for visualization
                disp_vis = (disparity_map * 255 / np.max(disparity_map)).astype(np.uint8)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "disparity.png"), disp_vis)
                
                # Apply colormap for better visualization
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(self.debug_dir, "masks", "disparity_color.png"), disp_color)
                
                # Create correspondence visualization
                correspondence_img = np.zeros((h, w*2, 3), dtype=np.uint8)
                
                # Draw a sample of correspondences (up to 1000 for clarity)
                max_vis = min(1000, len(points_left))
                if max_vis > 0:
                    sample_idx = np.linspace(0, len(points_left) - 1, max_vis).astype(int)
                    
                    for idx in sample_idx:
                        x_left, y_left = points_left[idx]
                        x_right, y_right = points_right[idx]
                        
                        # Draw points and line
                        cv2.circle(correspondence_img, (x_left, y_left), 2, (0, 255, 0), -1)
                        cv2.circle(correspondence_img, (x_right + w, y_right), 2, (0, 255, 0), -1)
                        cv2.line(correspondence_img, (x_left, y_left), (x_right + w, y_right), (0, 255, 255), 1)
                    
                    cv2.imwrite(os.path.join(self.debug_dir, "masks", "correspondences.png"), correspondence_img)
        
        # Convert to numpy arrays in the format expected by OpenCV
        points_left = np.array(points_left).reshape(-1, 1, 2).astype(np.float32) if points_left else np.empty((0, 1, 2), dtype=np.float32)
        points_right = np.array(points_right).reshape(-1, 1, 2).astype(np.float32) if points_right else np.empty((0, 1, 2), dtype=np.float32)
        
        logger.info(f"Found {len(points_left)} stereo correspondences")
        
        return points_left, points_right
    
    def triangulate_points(self, points_left: np.ndarray, points_right: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Triangulate 3D points from stereo correspondences.

        Args:
            points_left: Array of points in left image
            points_right: Array of points in right image

        Returns:
            Open3D point cloud
        """
        logger.info("Triangulating 3D points")
        
        # Check if we have enough correspondences
        if len(points_left) < 8:
            logger.error(f"Not enough correspondences for triangulation: {len(points_left)}")
            return None
        
        # Get projection matrices
        if self.calibrator and hasattr(self.calibrator, 'P1') and hasattr(self.calibrator, 'P2'):
            # Use calibration data if available
            P1 = self.calibrator.P1
            P2 = self.calibrator.P2
        else:
            # Create default projection matrices if no calibration
            # These are very approximate and will not give accurate results
            logger.warning("Using default projection matrices (results will not be accurate)")
            
            # Approximate camera matrices
            fx = 800  # Focal length in pixels
            fy = 800
            cx = 640  # Principal point
            cy = 360
            baseline = 100  # mm
            
            # Left camera projection matrix (identity rotation and zero translation)
            P1 = np.array([
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0]
            ], dtype=np.float32)
            
            # Right camera projection matrix (identity rotation and translation along x-axis)
            P2 = np.array([
                [fx, 0, cx, -fx * baseline],
                [0, fy, cy, 0],
                [0, 0, 1, 0]
            ], dtype=np.float32)
        
        try:
            # Make sure we have the right input format and sufficient points
            if len(points_left) < 10 or len(points_right) < 10:
                logger.error(f"Not enough points for triangulation: {len(points_left)}")
                return None

            # Make sure P1 and P2 are not None and have the right format (3x4 projection matrices)
            # Create default matrices if they're None
            if P1 is None or P2 is None:
                logger.warning("Projection matrices P1 or P2 are None, creating default matrices")
                # Create default projection matrices
                fx = 800  # Focal length
                fy = 800
                cx = 640  # Principal point
                cy = 360
                baseline = 100  # mm

                # Left camera projection matrix
                P1 = np.array([
                    [fx, 0, cx, 0],
                    [0, fy, cy, 0],
                    [0, 0, 1, 0]
                ], dtype=np.float32)

                # Right camera projection matrix
                P2 = np.array([
                    [fx, 0, cx, -fx * baseline],
                    [0, fy, cy, 0],
                    [0, 0, 1, 0]
                ], dtype=np.float32)
            # Check if they're properly shaped
            elif not (P1.shape == (3, 4) and P2.shape == (3, 4)):
                logger.warning(f"Projection matrices have wrong shape: P1 {P1.shape}, P2 {P2.shape}")

                # Try to ensure they're proper 3x4 matrices - reshape if we got a flat array
                if len(P1.shape) == 1 and P1.size == 12:
                    P1 = P1.reshape(3, 4)
                    logger.info(f"Reshaped P1 to {P1.shape}")
                if len(P2.shape) == 1 and P2.size == 12:
                    P2 = P2.reshape(3, 4)
                    logger.info(f"Reshaped P2 to {P2.shape}")

                # If matrices still don't have the right shape, we need to create defaults
                if not (P1.shape == (3, 4) and P2.shape == (3, 4)):
                    logger.warning("Creating default 3x4 projection matrices")

                    # Simple projection matrices
                    fx = 800  # Focal length
                    fy = 800
                    cx = 640  # Principal point
                    cy = 360
                    baseline = 100  # mm

                    # Left camera projection matrix
                    P1 = np.array([
                        [fx, 0, cx, 0],
                        [0, fy, cy, 0],
                        [0, 0, 1, 0]
                    ], dtype=np.float32)

                    # Right camera projection matrix
                    P2 = np.array([
                        [fx, 0, cx, -fx * baseline],
                        [0, fy, cy, 0],
                        [0, 0, 1, 0]
                    ], dtype=np.float32)

            # Ensure contiguous arrays with correct dtype
            P1 = np.ascontiguousarray(P1, dtype=np.float32)
            P2 = np.ascontiguousarray(P2, dtype=np.float32)

            # Debug calibration matrix shapes
            logger.info(f"Final projection matrix shapes: P1 {P1.shape}, P2 {P2.shape}")

            # Extract raw 2D points from the Nx1x2 format
            pts_left_raw = points_left.reshape(-1, 2)  # Convert to Nx2
            pts_right_raw = points_right.reshape(-1, 2)  # Convert to Nx2

            # Convert to the format OpenCV expects (2xN: first row=x coords, second row=y coords)
            pts_left = pts_left_raw.T.astype(np.float32)  # Transpose to 2xN
            pts_right = pts_right_raw.T.astype(np.float32)  # Transpose to 2xN

            # Make sure we have proper contiguous arrays (OpenCV requirement)
            pts_left = np.ascontiguousarray(pts_left, dtype=np.float32)
            pts_right = np.ascontiguousarray(pts_right, dtype=np.float32)

            # Log final formats for debugging
            logger.info(f"Points shape before triangulation: pts_left {pts_left.shape}, pts_right {pts_right.shape}")
            logger.info(f"Points data type: {pts_left.dtype}, contiguous: {pts_left.flags['C_CONTIGUOUS']}")

            # Use GPU acceleration if available and enabled
            if self.gpu_accelerator is not None and self.config.use_gpu:
                logger.info("Using GPU-accelerated triangulation")
                start_time = time.time()
                points_3d = self.gpu_accelerator.triangulate_points_gpu(P1, P2, pts_left, pts_right)
                logger.info(f"GPU triangulation completed in {time.time() - start_time:.2f} seconds")
            else:
                # OpenCV's triangulatePoints requires:
                # - 3x4 or 4x4 projection matrices (P1, P2) as np.float32
                # - 2xN point arrays (pts_left, pts_right) as np.float32
                logger.info("Using CPU triangulation")
                start_time = time.time()
                points_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)

                # Convert to 3D homogeneous coordinates
                points_3d = points_4d.T
                points_3d = points_3d[:, :3] / points_3d[:, 3:4]
                logger.info(f"CPU triangulation completed in {time.time() - start_time:.2f} seconds")
            
            # Filter out invalid points
            valid_mask = np.all(np.abs(points_3d) < 10000, axis=1) & \
                         np.all(~np.isnan(points_3d), axis=1) & \
                         np.all(~np.isinf(points_3d), axis=1)
            
            points_3d = points_3d[valid_mask]
            
            # Create Open3D point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_3d)

            # Apply neural network processing if enabled
            if self.nn_processor is not None and self.config.use_neural_network:
                logger.info("Applying neural network point cloud processing")
                start_time = time.time()

                # Configure processing parameters
                nn_params = {
                    "denoise_strength": 1.0,
                    "hole_fill_resolution": self.config.downsample_voxel_size,
                    "detail_level": 1.0
                }

                # Apply processing pipeline
                point_cloud = self.nn_processor.process_point_cloud(
                    point_cloud,
                    denoise=self.config.nn_denoise,
                    fill_holes=self.config.nn_fill_holes,
                    enhance_details=self.config.nn_enhance_details,
                    parameters=nn_params
                )

                logger.info(f"Neural network processing completed in {time.time() - start_time:.2f} seconds")
            else:
                # Apply traditional processing methods

                # Apply noise filtering if enabled
                if self.config.noise_filter and OPEN3D_AVAILABLE and len(points_3d) > 20:
                    logger.info("Applying statistical outlier removal")
                    start_time = time.time()
                    try:
                        point_cloud, _ = point_cloud.remove_statistical_outlier(
                            nb_neighbors=self.config.noise_filter_radius,
                            std_ratio=self.config.noise_filter_std
                        )
                        logger.info(f"Applied statistical outlier removal: {len(points_3d)} -> {len(point_cloud.points)} points in {time.time() - start_time:.2f} seconds")
                    except Exception as e:
                        logger.warning(f"Error during outlier removal: {e}")

                # Apply voxel downsampling if enabled
                if self.config.downsample and OPEN3D_AVAILABLE and len(point_cloud.points) > 1000:
                    logger.info("Applying voxel downsampling")
                    start_time = time.time()
                    try:
                        point_cloud = point_cloud.voxel_down_sample(voxel_size=self.config.downsample_voxel_size)
                        logger.info(f"Applied voxel downsampling to {len(point_cloud.points)} points in {time.time() - start_time:.2f} seconds")
                    except Exception as e:
                        logger.warning(f"Error during downsampling: {e}")
            
            # Save to debug directory if enabled
            if self.config.debug or self.config.save_intermediate_images:
                if OPEN3D_AVAILABLE:
                    o3d.io.write_point_cloud(os.path.join(self.debug_dir, "point_cloud.ply"), point_cloud)
            
            logger.info(f"Triangulated {len(point_cloud.points)} 3D points")
            
            return point_cloud
        
        except Exception as e:
            logger.error(f"Error during triangulation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def perform_scan(self) -> o3d.geometry.PointCloud:
        """
        Perform a complete 3D scan using structured light.

        Returns:
            Open3D point cloud
        """
        logger.info("Starting static 3D scan")
        start_time = time.time()
        
        # Configure camera
        self._configure_camera()
        
        # Generate patterns
        patterns = self.generate_patterns()
        
        # Project and capture patterns
        captured_images = self.capture_patterns(patterns)
        if not captured_images:
            logger.error("Failed to capture patterns")
            return None
        
        # Rectify images
        left_images, right_images = self.rectify_images(captured_images)
        if not left_images or not right_images:
            logger.error("Failed to process captured images")
            return None
        
        # Decode patterns
        left_coords, right_coords, mask_left, mask_right = self.decode_patterns(left_images, right_images)
        
        # Find correspondences
        points_left, points_right = self.find_correspondences(left_coords, right_coords, mask_left, mask_right)
        
        # Triangulate points
        point_cloud = self.triangulate_points(points_left, points_right)
        
        # Store the result
        self.point_cloud = point_cloud
        
        # Calculate processing time
        self.processing_time = time.time() - start_time
        
        logger.info(f"Completed static 3D scan in {self.processing_time:.2f} seconds")
        logger.info(f"Resulting point cloud has {len(point_cloud.points) if point_cloud else 0} points")
        
        return point_cloud
    
    def save_point_cloud(self, output_file: str) -> bool:
        """
        Save the point cloud to a file.

        Args:
            output_file: Output file path (PLY or PCD format)

        Returns:
            True if successful, False otherwise
        """
        if self.point_cloud is None:
            logger.error("No point cloud to save")
            return False
        
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for saving point clouds")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save point cloud
            o3d.io.write_point_cloud(output_file, self.point_cloud)
            
            logger.info(f"Point cloud saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving point cloud: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics for the scan.

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "point_count": len(self.point_cloud.points) if self.point_cloud else 0,
            "processing_time": self.processing_time,
            "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quality_preset": self.config.quality,
            "calibration_file": self.calibration_file,
            "debug_dir": self.debug_dir if self.config.debug or self.config.save_intermediate_images else None
        }
        
        return stats


def create_static_scanner(
    client,
    config=None,
    calibration_file=None
) -> StaticScanner:
    """
    Create a static 3D scanner.
    
    Args:
        client: UnlookClient instance
        config: Scanner configuration (optional)
        calibration_file: Path to stereo calibration file (optional)
    
    Returns:
        StaticScanner instance
    """
    if config is None:
        config = StaticScanConfig()
    
    return StaticScanner(
        client=client,
        config=config,
        calibration_file=calibration_file
    )


def perform_static_scan(
    client,
    output_file=None,
    calibration_file=None,
    quality="high",
    debug=False,
    use_gpu=None,
    use_neural_network=None
) -> o3d.geometry.PointCloud:
    """
    Perform a static 3D scan with minimal configuration.

    This is a simplified interface for users who want to perform a scan
    with minimal code.

    Args:
        client: UnlookClient instance
        output_file: Output file path for the point cloud (optional)
        calibration_file: Path to stereo calibration file (optional)
        quality: Quality preset ("medium", "high", or "ultra")
        debug: Enable debug output
        use_gpu: Whether to use GPU acceleration (overrides quality preset)
        use_neural_network: Whether to use neural network processing (overrides quality preset)

    Returns:
        Open3D point cloud
    """
    # Create configuration
    config = StaticScanConfig()
    config.set_quality_preset(quality)
    config.debug = debug

    # Override GPU and neural network settings if specified
    if use_gpu is not None:
        config.use_gpu = use_gpu
    if use_neural_network is not None:
        config.use_neural_network = use_neural_network
    
    # Create scanner
    scanner = create_static_scanner(
        client=client,
        config=config,
        calibration_file=calibration_file
    )
    
    # Perform scan
    point_cloud = scanner.perform_scan()
    
    # Save point cloud if output file is provided
    if output_file and point_cloud:
        scanner.save_point_cloud(output_file)
    
    return point_cloud