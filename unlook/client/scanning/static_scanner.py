"""
Static 3D Scanner for Unlook SDK.

This module provides a fast implementation for static 3D scanning
designed for quick yet high-quality scans. It uses vectorized operations
for enhanced speed and simplified hardware requirements, making it
more accessible while still providing good quality results.
"""

import os
import time
import json
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Try to import Open3D for point cloud processing
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    logging.info(f"Open3D version: {o3d.__version__}")
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not installed. Visualization will be limited.")

# Configure logger
logger = logging.getLogger(__name__)

# Import pattern generation utilities from appropriate modules
from .patterns.enhanced_gray_code import generate_enhanced_gray_code_patterns, decode_patterns as decode_gray_code_patterns
from .patterns.enhanced_phaseshift import generate_phase_shift_patterns as _generate_phase_shift_patterns_raw
from .patterns.enhanced_patterns import (
    generate_multi_scale_patterns,
    generate_multi_frequency_patterns,
    generate_variable_width_gray_code,
)

# For backward compatibility, rename the enhanced function
generate_gray_code_patterns = generate_enhanced_gray_code_patterns

# Wrapper function to convert phase shift patterns to expected format
def generate_phase_shift_patterns(width, height, num_steps=8, include_references=False):
    """
    Generate phase shift patterns in the expected dictionary format.
    
    Args:
        width: Pattern width
        height: Pattern height
        num_steps: Number of phase steps
        include_references: Whether to include white/black reference patterns
        
    Returns:
        List of pattern dictionaries
    """
    patterns = []
    
    # Add reference patterns if requested
    if include_references:
        patterns.append({
            "pattern_type": "solid_field",
            "color": "White",
            "name": "white_reference",
            "image": np.ones((height, width), dtype=np.uint8) * 255
        })
        patterns.append({
            "pattern_type": "solid_field",
            "color": "Black", 
            "name": "black_reference",
            "image": np.zeros((height, width), dtype=np.uint8)
        })
    
    # Generate phase shift patterns using single frequency
    raw_patterns = _generate_phase_shift_patterns_raw(
        width=width,
        height=height,
        frequencies=[1],  # Use single frequency for compatibility
        steps_per_frequency=num_steps
    )
    
    # Convert to dictionary format
    for i, pattern_image in enumerate(raw_patterns):
        patterns.append({
            "pattern_type": "phase_shift",
            "name": f"phase_shift_{i}",
            "step": i,
            "total_steps": num_steps,
            "image": pattern_image
        })
    
    return patterns

# Import decoding functions
# Use the actual decode_patterns from enhanced_gray_code
decode_patterns = decode_gray_code_patterns

# For now, create placeholder implementations for phase shift decoding
def decode_phase_shift(*args, **kwargs):
    logger.warning("decode_phase_shift not implemented yet")
    return None

class StaticScanner:
    """
    Fast and optimized static 3D scanner implementation with structured light.
    
    This scanner is designed for quick yet accurate static 3D scans using 
    structured light patterns. It focuses on efficiency and ease of use, 
    making it suitable for a wide range of applications.
    
    The scanner can adapt to different quality needs from fast preview scans
    to high-quality detailed scans.
    """
    def __init__(self, client, config=None, calibration_file=None):
        """
        Initialize the static 3D scanner.
        
        Args:
            client: Unlook client instance for hardware access
            config: Scanner configuration
            calibration_file: Path to stereo calibration file
        """
        self.client = client
        
        # Set default config if none provided
        if config is None:
            config = StaticScanConfig()
        self.config = config
        
        # Debug settings
        self.debug_enabled = getattr(config, 'debug', False)
        self.debug_path = None
        self.debug_metadata = {}
        
        # Initialize projector
        self.projector = client.projector
        
        # Set quality preset
        self.set_quality_preset(self.config.quality)
        
        # Initialize calibration
        self.calibration_file = calibration_file
        self.P1 = None  # Left camera projection matrix 
        self.P2 = None  # Right camera projection matrix
        self.Q = None   # Disparity-to-depth mapping matrix
        self.R1 = None  # Left rectification matrix
        self.R2 = None  # Right rectification matrix
        self.map1x = None  # Left camera x map for rectification
        self.map1y = None  # Left camera y map for rectification
        self.map2x = None  # Right camera x map for rectification
        self.map2y = None  # Right camera y map for rectification
        
        # Load calibration if provided, otherwise try auto-detection
        if calibration_file:
            self.load_calibration(calibration_file)
        else:
            self._try_auto_detect_calibration()
        
        # Initialize storage for the point cloud
        self.point_cloud = None
        
        # Initialize debug directory
        examples_dir = str(Path(__file__).resolve().parent.parent / "examples")
        self.debug_dir = os.path.join(examples_dir, "unlook_debug", f"scan_{time.strftime('%Y%m%d_%H%M%S')}")
        if self.config.debug or self.config.save_intermediate_images:
            self._ensure_debug_dirs()
            logger.info(f"Debug output will be saved to: {self.debug_dir}")
            # Print absolute path for easier access
            abs_path = os.path.abspath(self.debug_dir)
            print(f"\nDEBUG INFO: Scan data will be saved to:\n{abs_path}\n")
    
    def set_quality_preset(self, quality: str):
        """
        Configure scanner based on quality preset.
        
        Args:
            quality: Quality preset ('fast', 'balanced', or 'high')
        """
        if quality == "fast":
            # Optimize for speed
            self.config.pattern_width = 640
            self.config.pattern_height = 480
            self.config.num_gray_codes = 6
            self.config.num_phase_shifts = 4
            self.config.num_frequencies = 1
            self.config.denoise_level = 1
            self.config.downsample_voxel_size = 0.01
        elif quality == "balanced":
            # Balanced approach
            self.config.pattern_width = 800
            self.config.pattern_height = 600
            self.config.num_gray_codes = 8
            self.config.num_phase_shifts = 8
            self.config.num_frequencies = 2
            self.config.denoise_level = 2
            self.config.downsample_voxel_size = 0.005
        else:  # high quality
            # Maximum quality
            self.config.pattern_width = 1024
            self.config.pattern_height = 768
            self.config.num_gray_codes = 10
            self.config.num_phase_shifts = 12
            self.config.num_frequencies = 3
            self.config.denoise_level = 3
            self.config.downsample_voxel_size = 0.002

    def load_calibration(self, calibration_file: str) -> bool:
        """
        Load stereo calibration data from file.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            True if calibration loaded successfully, False otherwise
        """
        if not os.path.exists(calibration_file):
            logger.error(f"Calibration file not found: {calibration_file}")
            return False
            
        try:
            # Try to load as JSON first
            if calibration_file.endswith('.json'):
                with open(calibration_file, 'r') as f:
                    calibration = json.load(f)
                    
                # Extract projection matrices
                if 'P1' in calibration and 'P2' in calibration:
                    # Convert to numpy arrays
                    self.P1 = np.array(calibration['P1'], dtype=np.float32).reshape(3, 4)
                    self.P2 = np.array(calibration['P2'], dtype=np.float32).reshape(3, 4)
                    
                    logger.info(f"Loaded projection matrices from {calibration_file}")
                    logger.info(f"P1: {self.P1}")
                    logger.info(f"P2: {self.P2}")
                else:
                    logger.error("Calibration file does not contain P1 and P2 matrices")
                    return False
                    
                # Extract rectification matrices if available
                if 'R1' in calibration and 'R2' in calibration:
                    self.R1 = np.array(calibration['R1'], dtype=np.float32)
                    self.R2 = np.array(calibration['R2'], dtype=np.float32)
                
                # Extract disparity-to-depth mapping matrix
                if 'Q' in calibration:
                    self.Q = np.array(calibration['Q'], dtype=np.float32)
                else:
                    # Calculate Q matrix from projection matrices
                    logger.info("Q matrix not found, calculating from projection matrices")
                    self.Q = self.calculate_Q_matrix()
                
                # Extract rectification maps if available
                if ('map1x' in calibration and 'map1y' in calibration and 
                    'map2x' in calibration and 'map2y' in calibration):
                    self.map1x = np.array(calibration['map1x'], dtype=np.float32)
                    self.map1y = np.array(calibration['map1y'], dtype=np.float32)
                    self.map2x = np.array(calibration['map2x'], dtype=np.float32)
                    self.map2y = np.array(calibration['map2y'], dtype=np.float32)
            else:
                # Try to load as numpy files (.npy or text)
                p1_path = os.path.join(os.path.dirname(calibration_file), 'P1.txt')
                p2_path = os.path.join(os.path.dirname(calibration_file), 'P2.txt')
                
                if os.path.exists(p1_path) and os.path.exists(p2_path):
                    # Load text files
                    self.P1 = np.loadtxt(p1_path).astype(np.float32).reshape(3, 4)
                    self.P2 = np.loadtxt(p2_path).astype(np.float32).reshape(3, 4)
                    logger.info(f"Loaded projection matrices from text files in {os.path.dirname(calibration_file)}")
                else:
                    # Try to load the specific file (might be .npy)
                    self.P1 = np.load(calibration_file).astype(np.float32).reshape(3, 4)
                    p2_path = calibration_file.replace('P1', 'P2')
                    self.P2 = np.load(p2_path).astype(np.float32).reshape(3, 4)
                    logger.info(f"Loaded projection matrices from numpy files: {calibration_file}")
                
                # Calculate Q matrix from projection matrices
                self.Q = self.calculate_Q_matrix()
                
            # Validate the loaded projection matrices
            self.validate_projection_matrices()
                
            return True
        except Exception as e:
            logger.warning(f"Error loading calibration file: {e}")
            return False

    def _try_auto_detect_calibration(self):
        """
        Try to automatically detect and load calibration files.
        """
        # Try common locations for calibration files
        common_paths = [
            # Current working directory
            os.path.join(os.getcwd(), "stereo_calibration.json"),
            # Examples directory
            os.path.join(os.path.dirname(__file__), "..", "examples", "stereo_calibration.json"),
            # Examples/calibration directory
            os.path.join(os.path.dirname(__file__), "..", "examples", "calibration", "stereo_calibration.json"),
            # Look for projection matrices directly
            os.path.join(os.path.dirname(__file__), "..", "examples", "projection_matrices", "P1.txt"),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"Found calibration file: {path}")
                success = self.load_calibration(path)
                if success:
                    logger.info(f"Successfully loaded calibration from {path}")
                    return
                else:
                    logger.warning(f"Failed to load calibration from {path}")
        
        logger.warning("No calibration files found. Scanning may not work correctly.")

    def _ensure_debug_dirs(self):
        """
        Create debug directories for output files.
        """
        if self.config.debug or self.config.save_intermediate_images:
            debug_dirs = [
                self.debug_dir,  # Base debug directory
                os.path.join(self.debug_dir, "01_patterns"),  # Pattern images
                os.path.join(self.debug_dir, "01_patterns", "raw"),  # Raw pattern captures
                os.path.join(self.debug_dir, "02_rectified"),  # Rectified images
                os.path.join(self.debug_dir, "03_decoded"),  # Decoded patterns
                os.path.join(self.debug_dir, "04_correspondences"),  # Correspondence visualizations
                os.path.join(self.debug_dir, "05_point_cloud"),  # Point cloud data
            ]
            
            for directory in debug_dirs:
                os.makedirs(directory, exist_ok=True)
                
            logger.info(f"Created debug directories in {self.debug_dir}")
            
            # Save configuration for reference
            self._save_scan_config()

    def _save_scan_config(self):
        """
        Save the current configuration to a JSON file for reference.
        """
        try:
            config_path = os.path.join(self.debug_dir, "scan_config.json")
            config_dict = self.config.__dict__.copy()
            
            # Convert any non-serializable objects to strings
            for key, value in config_dict.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config_dict[key] = str(value)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
                
            logger.info(f"Saved scan configuration to {config_path}")
        except Exception as e:
            logger.warning(f"Error saving scan configuration: {e}")

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
            'phase_shifts': [],  # Phase shift pattern images
            'color': [],  # Store original color images of the object
            'multi_scale': [],  # New enhanced pattern types
            'multi_frequency': [],
            'variable_width': []
        }
        
        # Log the patterns for debugging
        logger.info(f"Capturing {len(patterns)} patterns with the following types:")
        pattern_types = {}
        for p in patterns:
            p_type = p.get("pattern_type", "unknown")
            pattern_types[p_type] = pattern_types.get(p_type, 0) + 1
        for p_type, count in pattern_types.items():
            logger.info(f"  {p_type}: {count} patterns")
        
        # Create debug directory
        debug_dir = None
        if hasattr(self, 'debug_dir'):
            debug_dir = os.path.join(self.debug_dir, "01_patterns", "raw")
            os.makedirs(debug_dir, exist_ok=True)
        
        # Capture reference images to help diagnose issues with the point cloud
        try:
            # Capture ambient light image (no pattern)
            logger.info("Capturing ambient light reference image...")
            
            # First turn off projector
            if hasattr(self.projector, 'turn_off'):
                self.projector.turn_off()
                time.sleep(0.5)  # Wait for projector to stabilize
            
            # Capture ambient light reference image
            if hasattr(self.client.camera, 'capture_stereo_pair'):
                ambient_left, ambient_right = self.client.camera.capture_stereo_pair()
                
                # Save ambient light reference images if debug directory exists
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "ambient_left.png"), ambient_left)
                    cv2.imwrite(os.path.join(debug_dir, "ambient_right.png"), ambient_right)
            
            # Now project white and capture color image of the object (with white illumination)
            logger.info("Capturing color reference image with white illumination...")
            if hasattr(self.projector, 'project_white'):
                self.projector.project_white()
                time.sleep(0.8)  # Longer delay to ensure projector is fully displaying white
            
            # Capture color reference image
            if hasattr(self.client.camera, 'capture_stereo_pair'):
                left_color, right_color = self.client.camera.capture_stereo_pair()
                
                # Save original color images if debug directory exists
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "object_left.png"), left_color)
                    cv2.imwrite(os.path.join(debug_dir, "object_right.png"), right_color)
                
                # Store color images in the result dictionary
                captured_images['color'].append((left_color, right_color))
        except Exception as e:
            logger.warning(f"Could not capture color reference images: {e}")
        
        # Project and capture each pattern
        # Try using pattern sequence approach if projector supports it
        # This tends to be more reliable as it's a single server command
        if hasattr(self.projector, 'project_sequence'):
            logger.info("Using pattern sequence approach for more reliable pattern projection")
            # Project patterns as a sequence with manual stepping
            try:
                # Need to process each pattern after projection, so custom handling
                logger.info(f"Starting manual pattern sequence for {len(patterns)} patterns")
                
                # First, prepare by turning off projector
                if hasattr(self.projector, 'turn_off'):
                    self.projector.turn_off()
                    time.sleep(0.2)  # Ensure projector has time to process
                
                # Now iterate through patterns one by one
                for i, pattern in enumerate(patterns):
                    pattern_type = pattern.get("pattern_type")
                    pattern_name = pattern.get("name", f"pattern_{i}")
                    
                    logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern_name}")
                    
                    # Project this single pattern with detailed logging
                    logger.info(f"About to project pattern: {pattern_name}, type: {pattern_type}")
                    success = self.projector.project_pattern(pattern)
                    if success:
                        logger.info(f"Successfully projected pattern: {pattern_name}")
                    else:
                        logger.error(f"Failed to project pattern: {pattern_name}")
                        
                    # Use a more reliable delay for pattern stabilization
                    time.sleep(0.2)  # Always wait at least 200ms
                    
                    # Apply a slightly longer delay for Gray code patterns
                    if pattern_type in ["gray_code", "multi_scale", "variable_width", "multi_frequency", "phase_shift"]:
                        delay = 0.2  # Base delay for most patterns
                        time.sleep(delay)
                    
                    # Try to capture the pattern
                    try:
                        # Capture images for this pattern
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
                        
                        # Process and store the captured images
                        # Convert to grayscale if needed
                        if len(left_img.shape) == 3:
                            left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                        else:
                            left_img_gray = left_img
                            
                        if len(right_img.shape) == 3:
                            right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                        else:
                            right_img_gray = right_img
                        
                        # Store in appropriate category
                        if pattern_type == "solid_field":
                            color = pattern.get("color", "White")
                            if color == "White":
                                captured_images['white'].append((left_img_gray, right_img_gray))
                            elif color == "Black":
                                captured_images['black'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "gray_code":
                            captured_images['gray_codes'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "phase_shift":
                            captured_images['phase_shifts'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "multi_scale":
                            captured_images['multi_scale'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "multi_frequency":
                            captured_images['multi_frequency'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "variable_width":
                            captured_images['variable_width'].append((left_img_gray, right_img_gray))
                        
                        # Save the raw captured images if requested
                        if debug_dir:
                            # Save both original and grayscale versions
                            cv2.imwrite(os.path.join(debug_dir, f"{pattern_name}_left.png"), left_img)
                            cv2.imwrite(os.path.join(debug_dir, f"{pattern_name}_right.png"), right_img)
                    except Exception as e:
                        logger.error(f"Error capturing pattern {pattern_name}: {e}")
            except Exception as e:
                logger.error(f"Error processing pattern sequence: {e}")
                return {}
        else:
            # Fall back to individual pattern approach
            for i, pattern in enumerate(patterns):
                pattern_type = pattern.get("pattern_type")
                pattern_name = pattern.get("name", f"pattern_{i}")
                
                logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern_name}")
                
                # Project pattern with better synchronization
                # Ensure projector is ready by clearing any prior patterns
                if hasattr(self.projector, 'clear'):
                    self.projector.clear()
                    time.sleep(0.1)  # Short delay to ensure projector is cleared
                elif hasattr(self.projector, 'turn_off') and hasattr(self.projector, 'turn_on'):
                    # Alternative method if clear() is not available
                    self.projector.turn_off()
                    time.sleep(0.2)  # Longer delay for more reliable reset
                    self.projector.turn_on()
                time.sleep(0.2)

                # Project the pattern with detailed logging and extra error handling
                logger.info(f"About to project pattern: {pattern_name}, type: {pattern_type}")
                
                # Add detailed parameters for debugging
                params = {k: v for k, v in pattern.items() if k != "image"}  # Exclude binary data
                logger.info(f"Pattern parameters: {params}")
                
                # Try projection with more detailed error handling
                try:
                    success = self.projector.project_pattern(pattern)
                    if success:
                        logger.info(f"Successfully projected pattern: {pattern_name}")
                    else:
                        logger.error(f"Failed to project pattern: {pattern_name}, trying again...")
                        # Try again with a delay
                        time.sleep(0.5)
                        success = self.projector.project_pattern(pattern)
                        if success:
                            logger.info(f"Second attempt succeeded for pattern: {pattern_name}")
                        else:
                            logger.error(f"Second attempt also failed for pattern: {pattern_name}")
                except Exception as e:
                    logger.error(f"Exception projecting pattern {pattern_name}: {e}")
                    # Don't exit the loop, try to continue with other patterns
                    success = False
                
                # Only continue with capture if projection was successful
                if success:
                    # Apply delay before capture
                    time.sleep(0.2)  # Standard delay for pattern stabilization
                    
                    # Set custom delay based on pattern type
                    if pattern_type in ["gray_code", "multi_scale", "variable_width", "multi_frequency"]:
                        time.sleep(self.config.pattern_interval if hasattr(self.config, 'pattern_interval') else 0.2)
                    
                    # Capture images
                    try:
                        # Capture stereo pair
                        if hasattr(self.client.camera, 'capture_stereo_pair'):
                            left_img, right_img = self.client.camera.capture_stereo_pair()
                        else:
                            # Fall back to individual cameras
                            cameras = self.client.camera.get_cameras()
                            if len(cameras) < 2:
                                logger.error("Not enough cameras found for stereo scanning")
                                continue
                            
                            # Capture from both cameras
                            camera_ids = [cam.get('id', f"camera_{i}") for i, cam in enumerate(cameras)][:2]
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
                        
                        # Store in appropriate category
                        if pattern_type == "solid_field":
                            color = pattern.get("color", "White")
                            if color == "White":
                                captured_images['white'].append((left_img_gray, right_img_gray))
                            elif color == "Black":
                                captured_images['black'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "gray_code":
                            captured_images['gray_codes'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "phase_shift":
                            captured_images['phase_shifts'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "multi_scale":
                            captured_images['multi_scale'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "multi_frequency":
                            captured_images['multi_frequency'].append((left_img_gray, right_img_gray))
                        elif pattern_type == "variable_width":
                            captured_images['variable_width'].append((left_img_gray, right_img_gray))
                        
                        # Save raw captures if debug enabled
                        if debug_dir:
                            # Save both original and grayscale versions
                            cv2.imwrite(os.path.join(debug_dir, f"{pattern_name}_left.png"), left_img)
                            cv2.imwrite(os.path.join(debug_dir, f"{pattern_name}_right.png"), right_img)
                            
                        # Save additional data for debugging
                        try:
                            if self.config.save_raw_images:
                                raw_dir = os.path.join(self.config.output_directory, "raw_captures")
                                os.makedirs(raw_dir, exist_ok=True)
                                
                                # Save both original and grayscale versions
                                cv2.imwrite(os.path.join(raw_dir, f"{pattern_name}_left.png"), left_img)
                                cv2.imwrite(os.path.join(raw_dir, f"{pattern_name}_right.png"), right_img)
                                
                                # If raw images are available and different from grayscale, save those too
                                if left_img is not left_img_gray:
                                    cv2.imwrite(os.path.join(raw_dir, f"{pattern_name}_left_gray.png"), left_img_gray)
                                    cv2.imwrite(os.path.join(raw_dir, f"{pattern_name}_right_gray.png"), right_img_gray)
                        except Exception as e:
                            logger.error(f"Error processing pattern {pattern_name}: {e}")
                            # Continue to the next pattern
                            continue
                    except Exception as e:
                        logger.error(f"Error capturing pattern {pattern_name}: {e}")
        
        # Log summary of captured images
        total_patterns = sum(len(images) for images in captured_images.values())
        logger.info(f"Captured {total_patterns} pattern pairs in total:")
        for cat, images in captured_images.items():
            if images:
                logger.info(f"  {cat}: {len(images)} pairs")
        
        return captured_images
        
    def perform_scan(self) -> o3d.geometry.PointCloud:
        """
        Perform a complete 3D scan using structured light.
        
        This method coordinates the entire scanning process:
        1. Camera configuration
        2. Pattern generation and projection
        3. Image capturing and rectification
        4. Correspondence finding
        5. Triangulation to produce a 3D point cloud
        
        Returns:
            Open3D point cloud object with the scan results
        """
        logger.info("Starting 3D scan with structured light patterns")
        start_time = time.time()
        
        # Initialize debug output
        if self.debug_enabled:
            self._initialize_debug()
            self._save_calibration_data()
        
        # Configure camera
        self._configure_camera()
        
        # Generate patterns based on configuration
        logger.info("Generating structured light patterns...")
        patterns = self.generate_patterns()
        
        # Save patterns for debug
        if self.debug_enabled and patterns:
            for i, pattern in enumerate(patterns):
                if "image" in pattern:
                    self._save_single_debug_image("01_patterns/projected", 
                                                f"pattern_{i:03d}_{pattern.get('name', '')}", 
                                                pattern["image"])
        
        # Project and capture patterns
        logger.info(f"Projecting and capturing {len(patterns)} patterns...")
        captured_images = self.capture_patterns(patterns)
        if not captured_images or len(captured_images) == 0:
            logger.error("Failed to capture patterns")
            return o3d.geometry.PointCloud()
        
        # Save raw captured images for debug
        if self.debug_enabled:
            # captured_images is a dictionary of pattern types
            pattern_idx = 0
            for pattern_type, image_list in captured_images.items():
                for i, (left_img, right_img) in enumerate(image_list):
                    self._save_debug_images("01_patterns/raw", 
                                         f"capture_{pattern_idx:03d}_{pattern_type}_{i:02d}", 
                                         (left_img, right_img))
                    pattern_idx += 1
        
        # Rectify captured images
        logger.info("Rectifying captured images...")
        left_images, right_images = self.rectify_images(captured_images)
        if not left_images or not right_images or len(left_images) == 0 or len(right_images) == 0:
            logger.error("Failed to rectify images")
            return o3d.geometry.PointCloud()
        
        # Save rectified images for debug
        if self.debug_enabled:
            for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
                self._save_debug_images("02_rectified", f"rectified_{i:03d}", (left_img, right_img))
        
        # Use enhanced pattern processor if patterns have low contrast
        if self.config.use_enhanced_processor or self.config.enhancement_level > 0:
            logger.info("Using enhanced pattern processor for low-contrast patterns...")
            from .patterns.enhanced_pattern_processor import EnhancedPatternProcessor
            
            processor = EnhancedPatternProcessor(
                threshold_method='adaptive',
                enhancement_level=self.config.enhancement_level
            )
            
            # Process patterns with enhancement
            # Combine all images including references
            all_left_patterns = [
                captured_images.get('black', [(None, None)])[0][0],  # black ref
                captured_images.get('white', [(None, None)])[0][0],  # white ref
            ] + left_images
            
            all_right_patterns = [
                captured_images.get('black', [(None, None)])[0][1],  # black ref  
                captured_images.get('white', [(None, None)])[0][1],  # white ref
            ] + right_images
            
            # Process patterns
            left_coords, right_coords = processor.process_stereo_patterns(
                all_left_patterns,
                all_right_patterns,
                {}  # calibration data
            )
            
            # Create masks
            height, width = left_images[0].shape[:2]
            mask_left = np.zeros((height, width), dtype=bool)
            mask_right = np.zeros((height, width), dtype=bool)
            
            if len(left_coords) > 0:
                for x, y in left_coords:
                    if 0 <= int(y) < height and 0 <= int(x) < width:
                        mask_left[int(y), int(x)] = True
                        
                for x, y in right_coords:
                    if 0 <= int(y) < height and 0 <= int(x) < width:
                        mask_right[int(y), int(x)] = True
        else:
            # Decode patterns and find correspondences
            logger.info("Decoding patterns and finding correspondences...")
            left_coords, right_coords, mask_left, mask_right = self.decode_patterns(left_images, right_images)
        
        # Save masks and decoded coordinates
        if self.debug_enabled:
            self._save_debug_images("03_masks", "mask", (mask_left, mask_right))
            
            # Save decoded coordinates
            np.save(self.debug_path / "04_correspondence/maps/left_coords.npy", left_coords)
            np.save(self.debug_path / "04_correspondence/maps/right_coords.npy", right_coords)
        
        # The decode_patterns already returns the actual correspondence points
        points_left = left_coords
        points_right = right_coords
        logger.info(f"Using {len(points_left)} correspondences from decode_patterns")
        
        if len(points_left) == 0 or len(points_right) == 0:
            logger.error("No valid correspondences found")
            return o3d.geometry.PointCloud()
        
        # Save correspondence points
        if self.debug_enabled:
            np.save(self.debug_path / "04_correspondence/maps/points_left.npy", points_left)
            np.save(self.debug_path / "04_correspondence/maps/points_right.npy", points_right)
        
        # Triangulate to create 3D point cloud
        logger.info("Triangulating 3D points...")
        point_cloud = self.triangulate_points(points_left, points_right)
        
        # Create and save disparity maps before point cloud creation
        if self.debug_enabled:
            self._create_and_save_disparity_maps(points_left, points_right)
        
        # Store point cloud for later access
        self.point_cloud = point_cloud
        
        if point_cloud is None:
            logger.warning("No point cloud was generated")
            self.point_cloud = o3d.geometry.PointCloud()  # Create empty point cloud
        
        num_points = len(point_cloud.points) if point_cloud and hasattr(point_cloud, 'points') else 0
        logger.info(f"3D scan completed in {time.time() - start_time:.2f} seconds with {num_points} points")
        
        # Save final debug data
        if self.debug_enabled:
            if point_cloud and hasattr(point_cloud, 'points'):
                points_array = np.asarray(point_cloud.points)
                self._save_point_cloud_debug(points_array, "raw")
                
                # Save filtered version if it exists
                if hasattr(self, 'filtered_point_cloud') and self.filtered_point_cloud:
                    points_filtered = np.asarray(self.filtered_point_cloud.points)
                    self._save_point_cloud_debug(points_filtered, "filtered")
            
            self._save_debug_metadata()
            logger.info(f"Debug data saved to: {self.debug_path}")
        
        return point_cloud

    def _configure_camera(self):
        """
        Configure cameras for structured light scanning.
        """
        logger.info("Configuring cameras for structured light scanning")
        
        try:
            # Set camera parameters for structured light scanning
            # Turn off auto-exposure and auto-white balance for better results
            if hasattr(self.client.camera, 'set_exposure'):
                logger.info(f"Setting exposure to {self.config.exposure}")
                self.client.camera.set_exposure(self.config.exposure)
            else:
                logger.warning("Camera does not support setting exposure")
            
            # Set fixed gain for consistent brightness
            if hasattr(self.client.camera, 'set_gain'):
                logger.info(f"Setting gain to {self.config.gain}")
                self.client.camera.set_gain(self.config.gain)
            else:
                logger.warning("Camera does not support setting gain")
                
            # Disable auto exposure as it can change between pattern projections
            if hasattr(self.client.camera, 'set_auto_exposure'):
                logger.info("Disabling auto exposure for consistent brightness")
                self.client.camera.set_auto_exposure(False)
            else:
                logger.warning("Camera does not support disabling auto exposure")
                
            # Wait for settings to take effect
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error configuring camera: {e}")

    def generate_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate structured light patterns based on configuration.
        
        Returns:
            List of pattern dictionaries ready for projection
        """
        logger.info("Generating structured light patterns...")
        patterns = []
        
        # Pattern generation parameters
        pattern_width = self.config.pattern_width
        pattern_height = self.config.pattern_height
        
        # Generate patterns based on the specified pattern type
        pattern_type = self.config.pattern_type if hasattr(self.config, 'pattern_type') else 'gray_code'
        
        logger.info(f"Using pattern type: {pattern_type}")
        
        if pattern_type == 'multi_scale':
            # Generate multi-scale patterns (better for complex scenes)
            num_scales = self.config.num_frequencies if hasattr(self.config, 'num_frequencies') else 2
            logger.info(f"Generating multi-scale patterns with {num_scales} scales")
            
            # Add white and black reference patterns
            patterns.append({
                "pattern_type": "solid_field",
                "name": "white",
                "color": "White",
                "image": np.ones((pattern_height, pattern_width), dtype=np.uint8) * 255
            })
            
            patterns.append({
                "pattern_type": "solid_field",
                "name": "black",
                "color": "Black",
                "image": np.zeros((pattern_height, pattern_width), dtype=np.uint8)
            })
            
            # Generate multi-scale patterns
            return generate_multi_scale_patterns(
                width=pattern_width, 
                height=pattern_height,
                num_bits=self.config.num_gray_codes if hasattr(self.config, 'num_gray_codes') else 10,
                orientation="both"
            )
            
        elif pattern_type == 'multi_frequency':
            # Generate multi-frequency patterns
            num_frequencies = self.config.num_frequencies if hasattr(self.config, 'num_frequencies') else 3
            steps_per_frequency = self.config.num_phase_shifts if hasattr(self.config, 'num_phase_shifts') else 8
            
            logger.info(f"Generating multi-frequency patterns with {num_frequencies} frequencies and {steps_per_frequency} steps per frequency")
            
            # The function takes frequencies list, not num_frequencies
            frequencies = [2**i for i in range(num_frequencies)]  # Generate power-of-2 frequencies
            
            return generate_multi_frequency_patterns(
                width=pattern_width,
                height=pattern_height,
                frequencies=frequencies,
                steps_per_frequency=steps_per_frequency,
                orientation="both"
            )
            
        elif pattern_type == 'variable_width':
            # Generate variable-width gray code patterns
            return generate_variable_width_gray_code(
                width=pattern_width,
                height=pattern_height,
                min_bits=4,
                max_bits=self.config.num_gray_codes if hasattr(self.config, 'num_gray_codes') else 10,
                orientation="both"
            )
            
        else:
            # Default to combined Gray code and phase shift patterns
            num_gray_codes = self.config.num_gray_codes if hasattr(self.config, 'num_gray_codes') else 8
            num_phase_shifts = self.config.num_phase_shifts if hasattr(self.config, 'num_phase_shifts') else 8
            
            logger.info(f"Generating Gray code patterns with {num_gray_codes} bits")
            gray_code_patterns = generate_gray_code_patterns(
                width=pattern_width,
                height=pattern_height,
                num_bits=num_gray_codes,
                orientation="both"  # enhanced_gray_code doesn't have include_references param
            )
            
            logger.info(f"Generating phase shift patterns with {num_phase_shifts} steps")
            phase_shift_patterns = generate_phase_shift_patterns(
                width=pattern_width,
                height=pattern_height,
                num_steps=num_phase_shifts,
                include_references=False  # References already included in Gray code patterns
            )
            
            # Combine patterns
            patterns = gray_code_patterns + phase_shift_patterns
            
        logger.info(f"Generated {len(patterns)} total patterns")
        return patterns
        
    def rectify_images(self, captured_images: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process captured pattern images with rectification if calibration is available.

        Args:
            captured_images: Dictionary with captured images

        Returns:
            Tuple of (left_images, right_images) lists with processed images
        """
        if not self.P1 is not None and self.P2 is not None:
            logger.warning("No calibration available for rectification, using raw images")
            return [], []
        
        # Combine all captured images in proper order
        left_images = []
        right_images = []
        
        # First, add color images if available
        for left, right in captured_images.get('color', []):
            left_images.append(left)
            right_images.append(right)
        
        # Add white reference images
        for left, right in captured_images.get('white', []):
            left_images.append(left)
            right_images.append(right)
            
        # Add black reference images
        for left, right in captured_images.get('black', []):
            left_images.append(left)
            right_images.append(right)
            
        # Add Gray code patterns
        for left, right in captured_images.get('gray_codes', []):
            left_images.append(left)
            right_images.append(right)
            
        # Add phase shift patterns
        for left, right in captured_images.get('phase_shifts', []):
            left_images.append(left)
            right_images.append(right)
            
        # Add enhanced pattern types
        for pattern_type in ['multi_scale', 'multi_frequency', 'variable_width']:
            for left, right in captured_images.get(pattern_type, []):
                left_images.append(left)
                right_images.append(right)
        
        # Perform rectification if calibration is available
        if self.P1 is not None and self.P2 is not None:
            try:
                logger.info(f"Rectifying {len(left_images)} image pairs...")
                
                # Use rectification maps if available, otherwise compute them
                if self.map1x is None or self.map1y is None or self.map2x is None or self.map2y is None:
                    logger.info("Computing rectification maps...")
                    # TODO: Generate rectification maps from projection matrices
                    pass
                
                # TODO: Apply rectification maps to images
                # For now, just return the original images
                
                # Save rectified images if debug is enabled
                if self.config.debug or self.config.save_intermediate_images:
                    debug_dir = os.path.join(self.debug_dir, "02_rectified")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save a few representative images
                    for i, (left, right) in enumerate(zip(left_images, right_images)):
                        if i % max(1, len(left_images) // 5) == 0:  # Save ~5 examples
                            cv2.imwrite(os.path.join(debug_dir, f"rectified_left_{i:02d}.png"), left)
                            cv2.imwrite(os.path.join(debug_dir, f"rectified_right_{i:02d}.png"), right)
            except Exception as e:
                logger.error(f"Error during rectification: {e}")
                # Continue with unrectified images
        else:
            logger.warning("No calibration available, using unrectified images")
        
        return left_images, right_images
        
    def decode_patterns(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode structured light patterns to find correspondences.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            
        Returns:
            Tuple of (left_coords, right_coords, mask_left, mask_right)
            - left_coords: Coordinates of detected points in left image
            - right_coords: Corresponding coordinates in right image
            - mask_left: Mask for valid points in left image
            - mask_right: Mask for valid points in right image
        """
        logger.info(f"Decoding {len(left_images)} pattern pairs...")
        
        if len(left_images) == 0 or len(right_images) == 0:
            logger.error("No images to decode")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Get dimensions from first image
        height, width = left_images[0].shape[:2]
        
        # Extract reference images and pattern images
        # Based on the pattern generation order:
        # 0: white reference
        # 1: black reference  
        # 2-11: horizontal gray codes (5 bits, normal and inverted)
        # 12-21: vertical gray codes (5 bits, normal and inverted)
        # 22-33: phase shift patterns (12 steps)
        
        if len(left_images) < 22:
            logger.error(f"Not enough images for Gray code decoding: {len(left_images)}")
            return self._fallback_decode(left_images, right_images)
        
        # Convert to grayscale if needed
        def to_gray(img):
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        
        # Extract reference images
        white_left = to_gray(left_images[0])
        white_right = to_gray(right_images[0])
        black_left = to_gray(left_images[1])
        black_right = to_gray(right_images[1])
        
        # Extract Gray code patterns
        # Horizontal patterns (bits 0-4)
        h_patterns_left = [to_gray(img) for img in left_images[2:12]]
        h_patterns_right = [to_gray(img) for img in right_images[2:12]]
        
        # Vertical patterns (bits 5-9)
        v_patterns_left = [to_gray(img) for img in left_images[12:22]]
        v_patterns_right = [to_gray(img) for img in right_images[12:22]]
        
        # Apply enhanced processing if enabled
        if self.config.use_enhanced_processor:
            logger.info("Applying enhanced pattern processing...")
            from .patterns.enhanced_pattern_processor import EnhancedPatternProcessor
            
            processor = EnhancedPatternProcessor()
            
            # Process left patterns
            h_patterns_left = processor.preprocess_images(h_patterns_left, black_left, white_left)
            v_patterns_left = processor.preprocess_images(v_patterns_left, black_left, white_left)
            
            # Process right patterns
            h_patterns_right = processor.preprocess_images(h_patterns_right, black_right, white_right)
            v_patterns_right = processor.preprocess_images(v_patterns_right, black_right, white_right)
            
            # Also process reference images for better contrast
            white_left = processor.enhance_single_image(white_left, self.config.enhancement_level)
            white_right = processor.enhance_single_image(white_right, self.config.enhancement_level)
            black_left = processor.enhance_single_image(black_left, self.config.enhancement_level)
            black_right = processor.enhance_single_image(black_right, self.config.enhancement_level)
        
        # Decode horizontal patterns (for X coordinates)
        logger.info("Decoding horizontal Gray code patterns...")
        x_coord_left, x_conf_left, x_mask_left = decode_patterns(
            white_left, black_left, h_patterns_left, 
            num_bits=5, orientation="horizontal"
        )
        
        x_coord_right, x_conf_right, x_mask_right = decode_patterns(
            white_right, black_right, h_patterns_right,
            num_bits=5, orientation="horizontal"
        )
        
        # Decode vertical patterns (for Y coordinates)
        logger.info("Decoding vertical Gray code patterns...")
        y_coord_left, y_conf_left, y_mask_left = decode_patterns(
            white_left, black_left, v_patterns_left,
            num_bits=5, orientation="vertical"
        )
        
        y_coord_right, y_conf_right, y_mask_right = decode_patterns(
            white_right, black_right, v_patterns_right,
            num_bits=5, orientation="vertical"
        )
        
        # Combine masks - only keep points visible in both patterns
        mask_left = x_mask_left & y_mask_left
        mask_right = x_mask_right & y_mask_right
        
        # Create coordinate grids based on pattern resolution
        # The Gray code patterns map projector coordinates to camera coordinates
        # We need to find correspondences between left and right cameras
        
        # For each valid point in the left image, find its correspondence in the right image
        # This is done by matching projector coordinates
        
        # Get valid coordinates
        valid_left = np.where(mask_left)
        valid_right = np.where(mask_right)
        
        points_left = []
        points_right = []
        
        # Debug: Save intermediate decode results
        if self.debug_enabled:
            decode_dir = self.debug_path / "03_decoded"
            decode_dir.mkdir(exist_ok=True, parents=True)
            
            # Create visualizations of decoded coordinates
            # Normalize coordinates for visualization
            x_vis_left = x_coord_left.copy()
            x_vis_right = x_coord_right.copy()
            y_vis_left = y_coord_left.copy()
            y_vis_right = y_coord_right.copy()
            
            # Ensure proper scaling and type for colormap
            if np.max(x_coord_left) > 0:
                x_vis_left = ((x_coord_left - np.min(x_coord_left)) / (np.max(x_coord_left) - np.min(x_coord_left)) * 255).astype(np.uint8)
            else:
                x_vis_left = np.zeros_like(x_coord_left, dtype=np.uint8)
                
            if np.max(x_coord_right) > 0:
                x_vis_right = ((x_coord_right - np.min(x_coord_right)) / (np.max(x_coord_right) - np.min(x_coord_right)) * 255).astype(np.uint8)
            else:
                x_vis_right = np.zeros_like(x_coord_right, dtype=np.uint8)
                
            if np.max(y_coord_left) > 0:
                y_vis_left = ((y_coord_left - np.min(y_coord_left)) / (np.max(y_coord_left) - np.min(y_coord_left)) * 255).astype(np.uint8)
            else:
                y_vis_left = np.zeros_like(y_coord_left, dtype=np.uint8)
                
            if np.max(y_coord_right) > 0:
                y_vis_right = ((y_coord_right - np.min(y_coord_right)) / (np.max(y_coord_right) - np.min(y_coord_right)) * 255).astype(np.uint8)
            else:
                y_vis_right = np.zeros_like(y_coord_right, dtype=np.uint8)
            
            # Apply colormaps for better visualization
            x_vis_left_color = cv2.applyColorMap(x_vis_left, cv2.COLORMAP_JET)
            x_vis_right_color = cv2.applyColorMap(x_vis_right, cv2.COLORMAP_JET)
            y_vis_left_color = cv2.applyColorMap(y_vis_left, cv2.COLORMAP_JET)
            y_vis_right_color = cv2.applyColorMap(y_vis_right, cv2.COLORMAP_JET)
            
            # Mask out invalid regions
            x_vis_left_color[~mask_left] = 0
            x_vis_right_color[~mask_right] = 0
            y_vis_left_color[~mask_left] = 0
            y_vis_right_color[~mask_right] = 0
            
            cv2.imwrite(str(decode_dir / "x_coord_left_color.png"), x_vis_left_color)
            cv2.imwrite(str(decode_dir / "x_coord_right_color.png"), x_vis_right_color)
            cv2.imwrite(str(decode_dir / "y_coord_left_color.png"), y_vis_left_color)
            cv2.imwrite(str(decode_dir / "y_coord_right_color.png"), y_vis_right_color)
            
            # Save coordinate statistics
            coord_stats = {
                "x_coord_left": {
                    "min": float(np.min(x_coord_left[mask_left])) if np.any(mask_left) else 0,
                    "max": float(np.max(x_coord_left[mask_left])) if np.any(mask_left) else 0,
                    "mean": float(np.mean(x_coord_left[mask_left])) if np.any(mask_left) else 0,
                    "valid_count": int(np.sum(mask_left))
                },
                "x_coord_right": {
                    "min": float(np.min(x_coord_right[mask_right])) if np.any(mask_right) else 0,
                    "max": float(np.max(x_coord_right[mask_right])) if np.any(mask_right) else 0,
                    "mean": float(np.mean(x_coord_right[mask_right])) if np.any(mask_right) else 0,
                    "valid_count": int(np.sum(mask_right))
                },
                "y_coord_left": {
                    "min": float(np.min(y_coord_left[mask_left])) if np.any(mask_left) else 0,
                    "max": float(np.max(y_coord_left[mask_left])) if np.any(mask_left) else 0,
                    "mean": float(np.mean(y_coord_left[mask_left])) if np.any(mask_left) else 0,
                },
                "y_coord_right": {
                    "min": float(np.min(y_coord_right[mask_right])) if np.any(mask_right) else 0,
                    "max": float(np.max(y_coord_right[mask_right])) if np.any(mask_right) else 0,
                    "mean": float(np.mean(y_coord_right[mask_right])) if np.any(mask_right) else 0,
                }
            }
            
            with open(decode_dir / "decode_stats.json", 'w') as f:
                json.dump(coord_stats, f, indent=2)
        
        # Use proper correspondence finder
        from .reconstruction.proper_correspondence_finder import find_stereo_correspondences_from_projector_coords, verify_correspondences
        
        # Find correspondences using projector coordinates
        points_left, points_right = find_stereo_correspondences_from_projector_coords(
            x_coord_left, y_coord_left, mask_left,
            x_coord_right, y_coord_right, mask_right,
            epipolar_threshold=5.0
        )
        
        # Verify correspondences are reasonable
        points_left, points_right = verify_correspondences(points_left, points_right)
        
        logger.info(f"Found {len(points_left)} valid correspondences")
        
        # Debug: Create correspondence visualization
        if self.debug_enabled and len(points_left) > 0:
            corr_vis = np.zeros((height, width * 2, 3), dtype=np.uint8)
            # Create base images
            if len(left_images) > 0:
                left_base = cv2.cvtColor(left_images[0], cv2.COLOR_GRAY2BGR) if left_images[0].ndim == 2 else left_images[0]
                right_base = cv2.cvtColor(right_images[0], cv2.COLOR_GRAY2BGR) if right_images[0].ndim == 2 else right_images[0]
                corr_vis[:, :width] = left_base
                corr_vis[:, width:] = right_base
            
            # Draw some correspondences for visualization
            step = max(1, len(points_left) // 100)  # Show up to 100 correspondences
            for i in range(0, len(points_left), step):
                x_left, y_left = int(points_left[i][0]), int(points_left[i][1])
                x_right, y_right = int(points_right[i][0]), int(points_right[i][1])
                
                color = (0, 255, 0)  # Green
                cv2.circle(corr_vis, (x_left, y_left), 3, color, -1)
                cv2.circle(corr_vis, (x_right + width, y_right), 3, color, -1)
                cv2.line(corr_vis, (x_left, y_left), (x_right + width, y_right), color, 1)
            
            cv2.imwrite(str(self.debug_path / "04_correspondence/visualizations/correspondences.png"), corr_vis)
        
        # If too few correspondences, fall back to simple method
        if len(points_left) < 100:
            logger.warning("Too few Gray code correspondences, using fallback method")
            return self._fallback_decode(left_images, right_images)
        
        # Create final masks
        final_mask_left = np.zeros((height, width), dtype=bool)
        final_mask_right = np.zeros((height, width), dtype=bool)
        
        if len(points_left) > 0:
            for x, y in points_left:
                final_mask_left[int(y), int(x)] = True
                
            for x, y in points_right:
                final_mask_right[int(y), int(x)] = True
        
        # Save decoded coordinates and confidence maps for debugging
        if self.debug_enabled:
            debug_dir = self.debug_path / "03_decoded"
            debug_dir.mkdir(exist_ok=True, parents=True)
            
            # Save decoded coordinates
            cv2.imwrite(str(debug_dir / "x_coord_left.png"), (x_coord_left / x_coord_left.max() * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "x_coord_right.png"), (x_coord_right / x_coord_right.max() * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "y_coord_left.png"), (y_coord_left / y_coord_left.max() * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "y_coord_right.png"), (y_coord_right / y_coord_right.max() * 255).astype(np.uint8))
            
            # Save confidence maps
            cv2.imwrite(str(debug_dir / "x_conf_left.png"), (x_conf_left * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "x_conf_right.png"), (x_conf_right * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "y_conf_left.png"), (y_conf_left * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "y_conf_right.png"), (y_conf_right * 255).astype(np.uint8))
            
            # Save final masks  
            cv2.imwrite(str(debug_dir / "mask_left.png"), (mask_left * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "mask_right.png"), (mask_right * 255).astype(np.uint8))
        
        return points_left, points_right, final_mask_left, final_mask_right
    
    def _fallback_decode(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fallback decoding method using simple feature matching."""
        logger.warning("Using fallback pattern decoding")
        
        # Get dimensions from first image
        height, width = left_images[0].shape[:2]
        
        # Use SIFT or ORB for feature matching
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
        
        # Use the first white reference image
        left_img = cv2.cvtColor(left_images[0], cv2.COLOR_BGR2GRAY) if len(left_images[0].shape) == 3 else left_images[0]
        right_img = cv2.cvtColor(right_images[0], cv2.COLOR_BGR2GRAY) if len(right_images[0].shape) == 3 else right_images[0]
        
        # Detect features
        kp1, des1 = detector.detectAndCompute(left_img, None)
        kp2, des2 = detector.detectAndCompute(right_img, None)
        
        if des1 is None or des2 is None:
            logger.error("No features detected")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Match features
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Extract matched points
        points_left = []
        points_right = []
        
        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # Check epipolar constraint (y coordinates should be similar)
            if abs(pt1[1] - pt2[1]) < 5:
                points_left.append(pt1)
                points_right.append(pt2)
        
        points_left = np.array(points_left)
        points_right = np.array(points_right)
        
        # Create masks
        mask_left = np.zeros((height, width), dtype=bool)
        mask_right = np.zeros((height, width), dtype=bool)
        
        for x, y in points_left:
            if 0 <= x < width and 0 <= y < height:
                mask_left[int(y), int(x)] = True
                
        for x, y in points_right:
            if 0 <= x < width and 0 <= y < height:
                mask_right[int(y), int(x)] = True
        
        logger.info(f"Found {len(points_left)} feature correspondences")
        
        return points_left, points_right, mask_left, mask_right
        
        # Save decoded coordinates visualization if debug is enabled
        if self.config.debug or self.config.save_intermediate_images:
            debug_dir = os.path.join(self.debug_dir, "03_decoded")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create visualization of decoded coordinates
            if left_images and mask_left is not None:
                # Use first image as background
                left_vis = left_images[0].copy()
                if len(left_vis.shape) == 2:
                    left_vis = cv2.cvtColor(left_vis, cv2.COLOR_GRAY2BGR)
                
                # Draw decoded points
                points = np.where(mask_left)
                for y, x in zip(points[0], points[1]):
                    cv2.circle(left_vis, (x, y), 1, (0, 255, 0), -1)
                
                cv2.imwrite(os.path.join(debug_dir, "decoded_left.png"), left_vis)
            
            if right_images and mask_right is not None:
                # Use first image as background
                right_vis = right_images[0].copy()
                if len(right_vis.shape) == 2:
                    right_vis = cv2.cvtColor(right_vis, cv2.COLOR_GRAY2BGR)
                
                # Draw decoded points
                points = np.where(mask_right)
                for y, x in zip(points[0], points[1]):
                    cv2.circle(right_vis, (x, y), 1, (0, 255, 0), -1)
                
                cv2.imwrite(os.path.join(debug_dir, "decoded_right.png"), right_vis)
        
        return left_coords, right_coords, mask_left, mask_right

    def find_correspondences(self, left_coords: np.ndarray, right_coords: np.ndarray,
                           mask_left: np.ndarray, mask_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find corresponding points between left and right images.
        
        Args:
            left_coords: Coordinates in left image (N,2)
            right_coords: Coordinates in right image (N,2)
            mask_left: Mask for valid points in left image
            mask_right: Mask for valid points in right image
            
        Returns:
            Tuple of (points_left, points_right) arrays with corresponding point coordinates
        """
        logger.info("Finding corresponding points between images...")
        
        if left_coords.size == 0 or right_coords.size == 0 or mask_left.size == 0 or mask_right.size == 0:
            logger.warning("Empty input data for correspondence finding")
            return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1, 2), dtype=np.float32)
        
        # Extract points where both masks are valid
        valid_mask = np.logical_and(mask_left, mask_right)
        
        # Reshape coordinates for triangulation
        left_coords_reshaped = left_coords.reshape(-1, 1, 2).astype(np.float32)
        right_coords_reshaped = right_coords.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply mask to get only valid correspondences
        final_mask_left = valid_mask
        final_mask_right = valid_mask
        
        return left_coords_reshaped, right_coords_reshaped, final_mask_left, final_mask_right

    def triangulate_points(self, points_left: np.ndarray, points_right: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Triangulate 3D points from corresponding 2D points.
        
        Args:
            points_left: Array of points in left image
            points_right: Array of points in right image
            
        Returns:
            Open3D point cloud
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for point cloud processing")
            return None
        
        logger.info(f"Triangulating {len(points_left)} points...")
        
        if len(points_left) == 0 or len(points_right) == 0:
            logger.warning("No points to triangulate")
            return o3d.geometry.PointCloud()
        
        # Ensure we have calibration data
        if self.P1 is None or self.P2 is None:
            logger.error("No calibration data available for triangulation")
            return None
        
        # Create a point cloud to store the results
        point_cloud = o3d.geometry.PointCloud()
        
        # Triangulate all points
        try:
            # Convert points to the right format for triangulation
            points_left_array = np.array(points_left).reshape(-1, 2)
            points_right_array = np.array(points_right).reshape(-1, 2)
            
            # Triangulate points using the projection matrices
            points_4d = cv2.triangulatePoints(self.P1, self.P2, points_left_array.T, points_right_array.T)
            
            # Convert from homogeneous coordinates to 3D
            points_3d = points_4d[:3, :] / points_4d[3, :]
            points_3d = points_3d.T
            
            # Debug: Save triangulation data
            if self.debug_enabled:
                debug_path = self.debug_path / "05_triangulation/data"
                
                # Save input points
                np.save(debug_path / "points_left_triangulation.npy", points_left_array)
                np.save(debug_path / "points_right_triangulation.npy", points_right_array)
                
                # Save projection matrices
                np.save(debug_path / "P1.npy", self.P1)
                np.save(debug_path / "P2.npy", self.P2)
                
                # Save triangulated points
                np.save(debug_path / "points_4d.npy", points_4d)
                np.save(debug_path / "points_3d.npy", points_3d)
                
                # Log statistics
                logger.info(f"Triangulation stats:")
                logger.info(f"  P1 shape: {self.P1.shape}")
                logger.info(f"  P2 shape: {self.P2.shape}")
                logger.info(f"  Input points: {len(points_left_array)}")
                logger.info(f"  Output points: {len(points_3d)}")
                logger.info(f"  3D point range: X[{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}], "
                           f"Y[{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}], "
                           f"Z[{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
            
            # Create point cloud
            point_cloud.points = o3d.utility.Vector3dVector(points_3d)
            
            # Add simple white color to all points
            white_color = np.ones((len(points_3d), 3), dtype=np.float64)
            point_cloud.colors = o3d.utility.Vector3dVector(white_color)
            
            # Apply post-processing
            if hasattr(self.config, 'denoise_level') and self.config.denoise_level > 0:
                # Remove outliers
                logger.info("Removing outliers from point cloud...")
                point_cloud, _ = point_cloud.remove_statistical_outlier(
                    nb_neighbors=20, 
                    std_ratio=self.config.denoise_level / 10.0
                )
            
            # Downsample to reduce size if needed
            if hasattr(self.config, 'downsample_voxel_size') and self.config.downsample_voxel_size > 0:
                # Scale the voxel size based on the scale of the point cloud
                # If points are in mm, voxel size should be larger
                points = np.asarray(point_cloud.points)
                if len(points) > 0:
                    # Check the scale of the points
                    bbox = point_cloud.get_axis_aligned_bounding_box()
                    bbox_size = bbox.get_extent()
                    max_dimension = max(bbox_size)
                    
                    # If max dimension is > 100, assume we're in mm scale
                    voxel_size = self.config.downsample_voxel_size
                    if max_dimension > 100:
                        # Scale up voxel size for mm scale (multiply by 10)
                        voxel_size = self.config.downsample_voxel_size * 10
                        logger.info(f"Detected mm scale, adjusting voxel size from {self.config.downsample_voxel_size} to {voxel_size}")
                    
                    # Ensure voxel size is not too small
                    min_voxel_size = max_dimension / 10000  # At most 10000 voxels per dimension
                    if voxel_size < min_voxel_size:
                        voxel_size = min_voxel_size
                        logger.warning(f"Voxel size too small, using {voxel_size}")
                    
                    logger.info(f"Downsampling point cloud with voxel size {voxel_size}...")
                    try:
                        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
                    except Exception as e:
                        logger.warning(f"Error downsampling: {e}. Skipping downsampling.")
                else:
                    logger.warning("Empty point cloud, skipping downsampling")
            
            # Save point cloud if debug is enabled
            if self.config.debug or self.config.save_intermediate_images:
                debug_dir = os.path.join(self.debug_dir, "05_point_cloud")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save as PLY file for easy viewing
                o3d.io.write_point_cloud(
                    os.path.join(debug_dir, "point_cloud.ply"), 
                    point_cloud
                )
            
            logger.info(f"Created point cloud with {len(point_cloud.points)} points")
            return point_cloud
        except Exception as e:
            logger.error(f"Error triangulating points: {e}")
            return None

    def save_point_cloud(self, output_file: str) -> bool:
        """
        Save the point cloud to a file.
        
        Args:
            output_file: Path to save the point cloud
            
        Returns:
            True if successful, False otherwise
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for saving point clouds")
            return False
            
        if self.point_cloud is None or len(self.point_cloud.points) == 0:
            logger.error("No point cloud to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save point cloud
            o3d.io.write_point_cloud(output_file, self.point_cloud)
            logger.info(f"Saved point cloud with {len(self.point_cloud.points)} points to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving point cloud: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processed scan.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "point_cloud_size": len(self.point_cloud.points) if self.point_cloud is not None else 0,
            "configuration": {k: v for k, v in self.config.__dict__.items() 
                              if isinstance(v, (int, float, str, bool))}
        }
        
        return stats
    
    def _initialize_debug(self, scan_name: str = None):
        """Initialize debug directories and metadata."""
        if not self.debug_enabled:
            return
            
        # Create debug directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = scan_name or f"scan_{timestamp}"
        
        # Create debug path 
        debug_root = Path(self.config.output_directory) / "unlook_debug" / base_name
        self.debug_path = debug_root
        
        # Create directory structure
        dirs = [
            "01_patterns/projected",
            "01_patterns/raw",
            "02_rectified/left",
            "02_rectified/right",
            "03_masks/left",
            "03_masks/right",
            "04_correspondence/maps",
            "04_correspondence/visualizations",
            "05_triangulation/data",
            "05_triangulation/visualizations",
            "06_point_cloud/raw",
            "06_point_cloud/filtered",
            "06_point_cloud/visualizations",
            "07_disparity_maps",
            "08_metadata"
        ]
        
        for dir_path in dirs:
            full_path = debug_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize metadata
        self.debug_metadata = {
            "timestamp": timestamp,
            "scan_name": base_name,
            "config": self.config.__dict__.copy(),
            "calibration": {}
        }
        
        logger.info(f"Debug output initialized at: {debug_root}")
    
    def _save_debug_images(self, category: str, name: str, image_data: Any, 
                          metadata: Dict = None):
        """Save debug images with optional metadata."""
        if not self.debug_enabled or self.debug_path is None:
            return
            
        # Handle different image types
        if isinstance(image_data, tuple):
            # Stereo pair
            left_img, right_img = image_data
            self._save_single_debug_image(f"{category}/left", name, left_img)
            self._save_single_debug_image(f"{category}/right", name, right_img)
        else:
            # Single image
            self._save_single_debug_image(category, name, image_data)
            
        # Save metadata if provided
        if metadata:
            meta_path = self.debug_path / category / f"{name}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _save_single_debug_image(self, category: str, name: str, image: np.ndarray):
        """Save a single debug image."""
        save_path = self.debug_path / category / f"{name}.png"
        
        # Convert if needed
        if image.dtype == bool:
            # Convert boolean mask to uint8
            img_uint8 = (image * 255).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            # Convert float images to uint8
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
            
        cv2.imwrite(str(save_path), img_uint8)
    
    def _save_calibration_data(self):
        """Save calibration matrices and parameters for debugging."""
        if not self.debug_enabled or self.debug_path is None:
            return
            
        metadata_path = self.debug_path / "08_metadata"
        
        # Save calibration matrices
        if hasattr(self, 'K1') and self.K1 is not None:
            np.savetxt(metadata_path / "camera_matrix_left.txt", self.K1)
        if hasattr(self, 'K2') and self.K2 is not None:
            np.savetxt(metadata_path / "camera_matrix_right.txt", self.K2)
        if hasattr(self, 'D1') and self.D1 is not None:
            np.savetxt(metadata_path / "dist_coeffs_left.txt", self.D1)
        if hasattr(self, 'D2') and self.D2 is not None:
            np.savetxt(metadata_path / "dist_coeffs_right.txt", self.D2)
        if hasattr(self, 'P1') and self.P1 is not None:
            np.savetxt(metadata_path / "projection_matrix_left.txt", self.P1)
        if hasattr(self, 'P2') and self.P2 is not None:
            np.savetxt(metadata_path / "projection_matrix_right.txt", self.P2)
        if hasattr(self, 'R') and self.R is not None:
            np.savetxt(metadata_path / "rotation_matrix.txt", self.R)
        if hasattr(self, 'T') and self.T is not None:
            np.savetxt(metadata_path / "translation_vector.txt", self.T)
            
        # Update metadata
        self.debug_metadata["calibration"] = {
            "baseline_mm": self.baseline if hasattr(self, 'baseline') else None,
            "fx": self.K1[0, 0] if hasattr(self, 'K1') and self.K1 is not None else None,
            "fy": self.K1[1, 1] if hasattr(self, 'K1') and self.K1 is not None else None,
            "cx": self.K1[0, 2] if hasattr(self, 'K1') and self.K1 is not None else None,
            "cy": self.K1[1, 2] if hasattr(self, 'K1') and self.K1 is not None else None,
        }
    
    def _save_debug_metadata(self):
        """Save final debug metadata."""
        if not self.debug_enabled or self.debug_path is None:
            return
            
        # Save configuration
        config_path = self.debug_path / "08_metadata" / "scan_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.debug_metadata, f, indent=2)
    
    def _save_point_cloud_debug(self, points_3d: np.ndarray, stage: str = "raw"):
        """Save point cloud data for debugging."""
        if not self.debug_enabled or self.debug_path is None:
            return
            
        # Save as numpy array
        npy_path = self.debug_path / f"06_point_cloud/{stage}" / "points_3d.npy"
        np.save(npy_path, points_3d)
        
        # Save as text file for easy inspection
        txt_path = self.debug_path / f"06_point_cloud/{stage}" / "points_3d.txt"
        np.savetxt(txt_path, points_3d, fmt='%.6f')
        
        # Save visualization if Open3D available
        if OPEN3D_AVAILABLE and len(points_3d) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # Save as PLY
            ply_path = self.debug_path / f"06_point_cloud/{stage}" / "points.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            
            # Create and save visualization
            vis_path = self.debug_path / f"06_point_cloud/visualizations" / f"{stage}_preview.png"
            # Skip visualization for now - can be implemented later
    
    def _create_and_save_disparity_maps(self, points_left: np.ndarray, points_right: np.ndarray):
        """Create and save disparity maps from correspondence points."""
        if not self.debug_enabled or self.debug_path is None:
            return
            
        logger.info("Creating disparity maps...")
        
        # Get image dimensions (assume from calibration or first image)
        if hasattr(self, 'image_width') and hasattr(self, 'image_height'):
            width, height = self.image_width, self.image_height
        else:
            # Default dimensions
            width, height = 640, 480
            
        # Create empty disparity map
        disparity_map = np.zeros((height, width), dtype=np.float32)
        disparity_count = np.zeros((height, width), dtype=np.int32)
        
        # Fill disparity map
        # Ensure points are in the right format
        if isinstance(points_left, list):
            points_left = np.array(points_left)
        if isinstance(points_right, list):
            points_right = np.array(points_right)
            
        # Handle different array shapes
        if points_left.ndim == 1:
            # Single point
            points_left = points_left.reshape(1, -1)
            points_right = points_right.reshape(1, -1)
            
        for i in range(len(points_left)):
            try:
                x_left = int(points_left[i, 0])
                y_left = int(points_left[i, 1])
                x_right = int(points_right[i, 0])
                y_right = int(points_right[i, 1])
            except (IndexError, TypeError) as e:
                logger.warning(f"Error accessing point {i}: {e}")
                logger.warning(f"points_left shape: {points_left.shape}, type: {type(points_left)}")
                logger.warning(f"points_left[{i}]: {points_left[i]}")
                continue
            
            # Calculate disparity (left_x - right_x)
            disparity = x_left - x_right
            
            if 0 <= x_left < width and 0 <= y_left < height:
                disparity_map[y_left, x_left] += disparity
                disparity_count[y_left, x_left] += 1
        
        # Average disparities where multiple values exist
        mask = disparity_count > 0
        disparity_map[mask] = disparity_map[mask] / disparity_count[mask]
        
        # Save disparity visualizations
        disp_dir = self.debug_path / "07_disparity_maps"
        disp_dir.mkdir(exist_ok=True, parents=True)
        
        # Save raw disparity data
        np.save(disp_dir / "disparity_raw.npy", disparity_map)
        
        # Save disparity statistics
        valid_disparities = disparity_map[mask]
        stats = {
            "min_disparity": float(np.min(valid_disparities)) if len(valid_disparities) > 0 else 0,
            "max_disparity": float(np.max(valid_disparities)) if len(valid_disparities) > 0 else 0,
            "mean_disparity": float(np.mean(valid_disparities)) if len(valid_disparities) > 0 else 0,
            "std_disparity": float(np.std(valid_disparities)) if len(valid_disparities) > 0 else 0,
            "num_valid_points": int(np.sum(mask)),
            "coverage_percentage": float(np.sum(mask) / (width * height) * 100)
        }
        
        with open(disp_dir / "disparity_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Create visualizations
        if len(valid_disparities) > 0:
            # Normalize for visualization
            disp_min = np.min(valid_disparities)
            disp_max = np.max(valid_disparities)
            
            if disp_max > disp_min:
                # Grayscale visualization
                disp_norm = (disparity_map - disp_min) / (disp_max - disp_min)
                disp_norm[~mask] = 0
                disp_gray = (disp_norm * 255).astype(np.uint8)
                cv2.imwrite(str(disp_dir / "disparity_grayscale.png"), disp_gray)
                
                # Colormap visualizations
                colormaps = {
                    'jet': cv2.COLORMAP_JET,
                    'hot': cv2.COLORMAP_HOT,
                    'viridis': cv2.COLORMAP_VIRIDIS,
                    'plasma': cv2.COLORMAP_PLASMA,
                    'magma': cv2.COLORMAP_MAGMA,
                    'rainbow': cv2.COLORMAP_RAINBOW
                }
                
                for name, colormap in colormaps.items():
                    disp_color = cv2.applyColorMap(disp_gray, colormap)
                    disp_color[~mask] = 0
                    cv2.imwrite(str(disp_dir / f"disparity_{name}.png"), disp_color)
                
                # Enhanced visualization with filtering
                disp_filtered = cv2.medianBlur(disp_gray, 5)
                disp_filtered[~mask] = 0
                cv2.imwrite(str(disp_dir / "disparity_median.png"), disp_filtered)
                
                # Enhanced colormaps on filtered data
                for name, colormap in colormaps.items():
                    disp_color = cv2.applyColorMap(disp_filtered, colormap)
                    disp_color[~mask] = 0
                    cv2.imwrite(str(disp_dir / f"disparity_median_{name}.png"), disp_color)
                
                # Create histogram
                plt.figure(figsize=(10, 6))
                plt.hist(valid_disparities, bins=50, alpha=0.7, color='blue')
                plt.xlabel('Disparity')
                plt.ylabel('Count')
                plt.title('Disparity Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(str(disp_dir / "disparity_histogram.png"))
                plt.close()
                
                # Create depth map (inverse of disparity)
                depth_map = np.zeros_like(disparity_map)
                depth_map[mask] = 1.0 / (disparity_map[mask] + 1e-6)
                
                # Normalize depth for visualization
                depth_min = np.min(depth_map[mask])
                depth_max = np.max(depth_map[mask])
                
                if depth_max > depth_min:
                    depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
                    depth_norm[~mask] = 0
                    depth_gray = (depth_norm * 255).astype(np.uint8)
                    cv2.imwrite(str(disp_dir / "depth_map.png"), depth_gray)
                    
                    # Depth colormap visualizations
                    depth_jet = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
                    depth_jet[~mask] = 0
                    cv2.imwrite(str(disp_dir / "depth_map_jet.png"), depth_jet)
                    
                    depth_viridis = cv2.applyColorMap(depth_gray, cv2.COLORMAP_VIRIDIS)
                    depth_viridis[~mask] = 0
                    cv2.imwrite(str(disp_dir / "depth_map_viridis.png"), depth_viridis)
        
        logger.info(f"Saved disparity maps with {len(points_left)} correspondences")


class StaticScanConfig:
    """
    Configuration for static 3D scanner.
    """
    def __init__(
        self, 
        quality: str = "balanced",
        pattern_type: str = "gray_code", 
        pattern_width: int = 800,
        pattern_height: int = 600,
        num_gray_codes: int = 8,
        num_phase_shifts: int = 8,
        num_frequencies: int = 2,
        exposure: int = 100,
        gain: float = 1.0,
        pattern_interval: float = 0.2,
        min_reliability: float = 0.7,
        denoise_level: int = 2,
        downsample_voxel_size: float = 0.005,
        save_intermediate_images: bool = False,
        save_raw_images: bool = False,
        output_directory: str = "output",
        debug: bool = False,
        use_enhanced_processor: bool = True,  # Changed to True by default
        enhancement_level: int = 3,  # Changed to maximum level by default
    ):
        """
        Initialize scan configuration.
        
        Args:
            quality: Quality preset ('fast', 'balanced', 'high')
            pattern_type: Type of patterns to use ('gray_code', 'multi_scale', 'multi_frequency', 'variable_width')
            pattern_width: Width of the generated patterns
            pattern_height: Height of the generated patterns
            num_gray_codes: Number of Gray code bits to use
            num_phase_shifts: Number of phase shift steps to use
            num_frequencies: Number of frequencies for multi-frequency patterns
            exposure: Camera exposure setting
            gain: Camera gain setting
            pattern_interval: Delay between patterns
            min_reliability: Minimum reliability threshold for decoding
            denoise_level: Level of denoising to apply (0-5)
            downsample_voxel_size: Voxel size for downsampling point cloud
            save_intermediate_images: Whether to save intermediate images
            save_raw_images: Whether to save raw captures
            output_directory: Directory to save output files
            debug: Enable debug output
            use_enhanced_processor: Use enhanced pattern processor for low-contrast images
            enhancement_level: Enhancement level (0-3) for pattern processing
        """
        self.quality = quality
        self.pattern_type = pattern_type
        self.pattern_width = pattern_width
        self.pattern_height = pattern_height
        self.num_gray_codes = num_gray_codes
        self.num_phase_shifts = num_phase_shifts
        self.num_frequencies = num_frequencies
        self.exposure = exposure
        self.gain = gain
        self.pattern_interval = pattern_interval
        self.min_reliability = min_reliability
        self.denoise_level = denoise_level
        self.downsample_voxel_size = downsample_voxel_size
        self.save_intermediate_images = save_intermediate_images
        self.save_raw_images = save_raw_images
        self.output_directory = output_directory
        self.debug = debug
        self.use_enhanced_processor = use_enhanced_processor
        self.enhancement_level = enhancement_level


def create_static_scanner(client, config=None, calibration_file=None):
    """
    Create a static 3D scanner instance.
    
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


def scan3d(
    client,
    output_file=None,
    calibration_file=None,
    quality="balanced",
    pattern_type="gray_code",
    debug=False,
):
    """
    Perform a 3D scan with the static scanner.
    
    Args:
        client: UnlookClient instance
        output_file: Path to save the resulting point cloud
        calibration_file: Path to stereo calibration file
        quality: Scan quality preset ('fast', 'balanced', 'high')
        pattern_type: Type of patterns to use ('gray_code', 'multi_scale', etc.)
        debug: Enable debug output
    
    Returns:
        Point cloud object
    """
    logger.info(f"Starting 3D scan with quality={quality}, pattern_type={pattern_type}")
    
    # Create configuration
    config = StaticScanConfig(
        quality=quality,
        pattern_type=pattern_type,
        debug=debug
    )
    
    # Create scanner
    scanner = StaticScanner(
        client=client,
        config=config,
        calibration_file=calibration_file
    )
    
    # Perform scan
    point_cloud = scanner.perform_scan()

    # Save point cloud if output file is provided
    if output_file and point_cloud:
        scanner.save_point_cloud(output_file)
        logger.info(f"Saved point cloud to {output_file} ({len(point_cloud.points)} points)")

    return point_cloud


# Legacy function maintained for backward compatibility
def perform_static_scan(
    client,
    output_file=None,
    calibration_file=None,
    quality="balanced",
    debug=False,
    use_gpu=None,  # Ignored, kept for compatibility
    use_neural_network=None  # Ignored, kept for compatibility
):
    """
    Legacy function for backward compatibility.
    
    Use scan3d or create_static_scanner instead.
    """
    return scan3d(
        client=client,
        output_file=output_file,
        calibration_file=calibration_file,
        quality=quality,
        debug=debug
    )