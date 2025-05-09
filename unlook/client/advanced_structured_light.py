"""
Advanced structured light scanning implementation with improved algorithms.

This module provides enhanced algorithms for structured light scanning,
incorporating techniques from academic research and best practices.
It focuses on robust pattern generation, decoding, and point cloud reconstruction.
"""

import os
import time
import logging
import numpy as np
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


class EnhancedGrayCodeGenerator:
    """
    Enhanced Gray code pattern generator with robust decoding capabilities.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, min_brightness: int = 5):
        """
        Initialize the enhanced Gray code generator.
        
        Args:
            width: Projector width in pixels
            height: Projector height in pixels
            min_brightness: Minimum brightness difference for decoding
        """
        self.proj_w = width
        self.proj_h = height
        self.min_brightness = min_brightness
        self.shadow_threshold = 40  # For shadow detection
        
        # Compute required number of patterns
        self.bits_x = int(np.ceil(np.log2(width)))
        self.bits_y = int(np.ceil(np.log2(height)))
        self.num_patterns = 2 + 2 * (self.bits_x + self.bits_y)  # Including white and black
        
        logger.info(f"Initialized Enhanced Gray Code generator: {width}x{height}, {self.num_patterns} patterns")
    
    def generate_pattern_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate a complete Gray code pattern sequence with calibration patterns.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Generate white and black calibration images
        white_image = np.ones((self.proj_h, self.proj_w), dtype=np.uint8) * 255
        black_image = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        
        # Add white and black images
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(white_image),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(black_image),
            "name": "black"
        })
        
        # Generate horizontal patterns (X axis)
        for bit in range(self.bits_x):
            # Generate Gray code pattern for this bit
            pattern_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
            
            for x in range(self.proj_w):
                # Convert binary to Gray code
                gray_x = x ^ (x >> 1)
                # Check if this bit is set in the Gray code
                if (gray_x >> bit) & 1:
                    pattern_img[:, x] = 255
            
            # Add pattern and its inverse
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(pattern_img),
                "name": f"gray_code_x_{bit}"
            })
            
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(255 - pattern_img),
                "name": f"gray_code_x_inv_{bit}"
            })
        
        # Generate vertical patterns (Y axis)
        for bit in range(self.bits_y):
            # Generate Gray code pattern for this bit
            pattern_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
            
            for y in range(self.proj_h):
                # Convert binary to Gray code
                gray_y = y ^ (y >> 1)
                # Check if this bit is set in the Gray code
                if (gray_y >> bit) & 1:
                    pattern_img[y, :] = 255
            
            # Add pattern and its inverse
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(pattern_img),
                "name": f"gray_code_y_{bit}"
            })
            
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(255 - pattern_img),
                "name": f"gray_code_y_inv_{bit}"
            })
        
        # Add fine checkerboard pattern for feature detection
        checkerboard_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        square_size = 16
        for y in range(0, self.proj_h, square_size):
            for x in range(0, self.proj_w, square_size):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    y_end = min(y + square_size, self.proj_h)
                    x_end = min(x + square_size, self.proj_w)
                    checkerboard_img[y:y_end, x:x_end] = 255
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(checkerboard_img),
            "name": "checkerboard"
        })
        
        # Add horizontal and vertical sinusoidal patterns (phase shift)
        for freq in [8, 16, 32]:
            # Horizontal sinusoidal pattern
            h_sin_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
            for x in range(self.proj_w):
                val = 127.5 + 127.5 * np.sin(2 * np.pi * x / freq)
                h_sin_img[:, x] = val
            
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(h_sin_img),
                "name": f"h_sin_{freq}"
            })
            
            # Vertical sinusoidal pattern
            v_sin_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
            for y in range(self.proj_h):
                val = 127.5 + 127.5 * np.sin(2 * np.pi * y / freq)
                v_sin_img[y, :] = val
            
            patterns.append({
                "pattern_type": "raw_image", 
                "image": self._encode_image(v_sin_img),
                "name": f"v_sin_{freq}"
            })
        
        logger.info(f"Generated {len(patterns)} structured light patterns")
        return patterns
    
    def decode_patterns(self, captured_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to obtain projector-camera correspondence.
        
        Args:
            captured_images: List of captured images (white, black, patterns)
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        if len(captured_images) < 2 + 2 * (self.bits_x + self.bits_y):
            logger.error(f"Not enough patterns: {len(captured_images)} provided, " 
                        f"{2 + 2 * (self.bits_x + self.bits_y)} required")
            return None, None
        
        # Extract white and black images
        white_img = captured_images[0]
        black_img = captured_images[1]
        
        # Convert to grayscale if needed
        if len(white_img.shape) == 3:
            white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        if len(black_img.shape) == 3:
            black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask
        mask = self._compute_shadow_mask(black_img, white_img)
        
        # Extract pattern images (starting from index 2)
        pattern_images = captured_images[2:2 + 2 * (self.bits_x + self.bits_y)]
        
        # Convert pattern images to grayscale if needed
        for i in range(len(pattern_images)):
            if len(pattern_images[i].shape) == 3:
                pattern_images[i] = cv2.cvtColor(pattern_images[i], cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = mask.shape
        
        # Initialize projector correspondence map
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Decode X coordinate (horizontal patterns)
                x_code = 0
                for bit in range(self.bits_x):
                    # Get normal pattern and its inverse for this bit
                    idx = 2 + 2 * bit
                    normal = pattern_images[idx][y, x]
                    inverse = pattern_images[idx + 1][y, x]
                    
                    # Check if difference is significant enough
                    if abs(int(normal) - int(inverse)) > self.min_brightness:
                        # Set bit if normal is brighter than inverse
                        if normal > inverse:
                            x_code |= (1 << bit)
                
                # Decode Y coordinate (vertical patterns)
                y_code = 0
                for bit in range(self.bits_y):
                    # Get normal pattern and its inverse for this bit
                    idx = 2 + 2 * self.bits_x + 2 * bit
                    normal = pattern_images[idx][y, x]
                    inverse = pattern_images[idx + 1][y, x]
                    
                    # Check if difference is significant enough
                    if abs(int(normal) - int(inverse)) > self.min_brightness:
                        # Set bit if normal is brighter than inverse
                        if normal > inverse:
                            y_code |= (1 << bit)
                
                # Convert from Gray code to binary
                x_proj = self._gray_to_binary(x_code)
                y_proj = self._gray_to_binary(y_code)
                
                # Store projector coordinates if within bounds
                if 0 <= x_proj < self.proj_w and 0 <= y_proj < self.proj_h:
                    cam_proj[y, x, 0] = y_proj  # Row (V)
                    cam_proj[y, x, 1] = x_proj  # Column (U)
        
        return cam_proj, mask
    
    def _compute_shadow_mask(self, black_img: np.ndarray, white_img: np.ndarray) -> np.ndarray:
        """
        Compute robust shadow mask using adaptive methods.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            
        Returns:
            Binary mask (1 for valid pixels, 0 for shadowed)
        """
        # Compute difference between white and black images
        diff = cv2.absdiff(white_img, black_img)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Apply adaptive thresholding
        try:
            # Normalize for improved contrast
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            
            # Try adaptive threshold
            mask = cv2.adaptiveThreshold(
                normalized, 
                1, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21, 
                -5
            )
        except Exception as e:
            logger.warning(f"Adaptive thresholding failed, using simple threshold: {e}")
            # Fall back to simple thresholding
            _, mask = cv2.threshold(blurred, self.shadow_threshold, 1, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _gray_to_binary(self, n: int) -> int:
        """Convert Gray code to binary."""
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """Encode image as JPEG."""
        success, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return data.tobytes() if success else b''


class PhaseShiftGenerator:
    """
    Phase shift pattern generator for structured light scanning.
    Uses sinusoidal phase shift patterns for higher accuracy.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, num_phases: int = 4):
        """
        Initialize phase shift generator.
        
        Args:
            width: Projector width
            height: Projector height
            num_phases: Number of phase shifts per frequency
        """
        self.proj_w = width
        self.proj_h = height
        self.num_phases = num_phases
        self.frequencies = [16, 32, 64, 128]  # Multiple frequencies for unwrapping
        
        logger.info(f"Initialized Phase Shift generator: {width}x{height}, {num_phases} phases")
    
    def generate_pattern_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate phase shift pattern sequence.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Generate white and black calibration images
        white_image = np.ones((self.proj_h, self.proj_w), dtype=np.uint8) * 255
        black_image = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(white_image),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(black_image),
            "name": "black"
        })
        
        # Generate horizontal phase shift patterns
        for freq in self.frequencies:
            for phase in range(self.num_phases):
                phase_offset = 2 * np.pi * phase / self.num_phases
                
                # Create horizontal phase pattern
                h_pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
                for x in range(self.proj_w):
                    val = 127.5 + 127.5 * np.cos(2 * np.pi * x / freq + phase_offset)
                    h_pattern[:, x] = val
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._encode_image(h_pattern),
                    "name": f"h_phase_{freq}_{phase}"
                })
        
        # Generate vertical phase shift patterns
        for freq in self.frequencies:
            for phase in range(self.num_phases):
                phase_offset = 2 * np.pi * phase / self.num_phases
                
                # Create vertical phase pattern
                v_pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
                for y in range(self.proj_h):
                    val = 127.5 + 127.5 * np.cos(2 * np.pi * y / freq + phase_offset)
                    v_pattern[y, :] = val
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._encode_image(v_pattern),
                    "name": f"v_phase_{freq}_{phase}"
                })
        
        logger.info(f"Generated {len(patterns)} phase shift patterns")
        return patterns
    
    def decode_patterns(self, captured_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode phase shift patterns using multi-frequency phase unwrapping.
        
        Args:
            captured_images: List of captured images (white, black, patterns)
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        # Extract white and black images
        white_img = captured_images[0]
        black_img = captured_images[1]
        
        # Convert to grayscale if needed
        if len(white_img.shape) == 3:
            white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        if len(black_img.shape) == 3:
            black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask
        mask = self._compute_shadow_mask(black_img, white_img)
        
        # Get image dimensions
        height, width = mask.shape
        
        # Initialize projector correspondence map
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Extract pattern images (starting from index 2)
        pattern_images = []
        for i in range(2, len(captured_images)):
            if len(captured_images[i].shape) == 3:
                pattern_images.append(cv2.cvtColor(captured_images[i], cv2.COLOR_BGR2GRAY))
            else:
                pattern_images.append(captured_images[i])
        
        # Process horizontal patterns (for x-coordinate)
        x_phases = self._decode_phase_shifts(pattern_images[:len(self.frequencies) * self.num_phases], mask)
        
        # Process vertical patterns (for y-coordinate)
        v_patterns = pattern_images[len(self.frequencies) * self.num_phases:]
        y_phases = self._decode_phase_shifts(v_patterns, mask)
        
        # Fill correspondence map
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    proj_x = x_phases[y, x]
                    proj_y = y_phases[y, x]
                    
                    if 0 <= proj_x < self.proj_w and 0 <= proj_y < self.proj_h:
                        cam_proj[y, x, 1] = proj_x  # Column (U)
                        cam_proj[y, x, 0] = proj_y  # Row (V)
        
        return cam_proj, mask
    
    def _decode_phase_shifts(self, patterns: List[np.ndarray], mask: np.ndarray) -> np.ndarray:
        """
        Decode phase shift patterns using multi-frequency phase unwrapping.
        
        Args:
            patterns: List of phase shift patterns
            mask: Shadow mask
            
        Returns:
            Unwrapped phase map scaled to projector coordinates
        """
        height, width = mask.shape
        
        # Initialize arrays for each frequency
        wrapped_phases = []
        
        # Calculate wrapped phase for each frequency
        for f in range(len(self.frequencies)):
            # Get patterns for this frequency
            freq_patterns = patterns[f * self.num_phases:(f + 1) * self.num_phases]
            
            # Initialize arrays for sine and cosine accumulation
            numerator = np.zeros((height, width), dtype=np.float32)
            denominator = np.zeros((height, width), dtype=np.float32)
            
            # Accumulate sine and cosine components
            for p in range(self.num_phases):
                phase = 2 * np.pi * p / self.num_phases
                pattern = freq_patterns[p].astype(np.float32)
                
                numerator += pattern * np.sin(phase)
                denominator += pattern * np.cos(phase)
            
            # Calculate wrapped phase
            wrapped_phase = np.arctan2(numerator, denominator)
            wrapped_phases.append(wrapped_phase)
        
        # Unwrap phase using multi-frequency approach
        unwrapped_phase = self._unwrap_phase_multi_freq(wrapped_phases, self.frequencies, mask)
        
        # Scale to projector coordinates
        max_freq = max(self.frequencies)
        scaled_coords = (unwrapped_phase / (2 * np.pi)) * max_freq
        
        return scaled_coords
    
    def _unwrap_phase_multi_freq(self, wrapped_phases: List[np.ndarray], 
                               frequencies: List[int], mask: np.ndarray) -> np.ndarray:
        """
        Unwrap phase using multi-frequency approach.
        
        Args:
            wrapped_phases: List of wrapped phase maps
            frequencies: List of frequencies
            mask: Shadow mask
            
        Returns:
            Unwrapped phase map
        """
        # Start with highest frequency
        freq_idx = len(frequencies) - 1
        high_freq = frequencies[freq_idx]
        unwrapped = wrapped_phases[freq_idx].copy()
        
        # Unwrap using hierarchical approach
        for i in range(freq_idx - 1, -1, -1):
            low_freq = frequencies[i]
            ratio = high_freq / low_freq
            
            # Calculate phase difference
            diff = unwrapped - wrapped_phases[i] * ratio
            
            # Adjust difference to be in [-π, π]
            diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
            
            # Update unwrapped phase
            unwrapped = wrapped_phases[i] * ratio + diff
        
        # Apply mask
        unwrapped = unwrapped * mask
        
        return unwrapped
    
    def _compute_shadow_mask(self, black_img: np.ndarray, white_img: np.ndarray) -> np.ndarray:
        """
        Compute shadow mask.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            
        Returns:
            Binary mask (1 for valid pixels, 0 for shadowed)
        """
        # Compute difference
        diff = cv2.absdiff(white_img, black_img)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Apply adaptive threshold
        try:
            # Normalize
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                normalized,
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                -5
            )
        except Exception:
            # Simple threshold as fallback
            _, binary = cv2.threshold(blurred, 20, 1, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """Encode image as JPEG."""
        success, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return data.tobytes() if success else b''


class EnhancedStereoScanner:
    """
    Enhanced stereo structured light scanner with robust reconstruction.
    """
    
    def __init__(self, camera_matrix_left: np.ndarray, dist_coeffs_left: np.ndarray,
                camera_matrix_right: np.ndarray, dist_coeffs_right: np.ndarray,
                R: np.ndarray, T: np.ndarray, image_size: Tuple[int, int],
                projector_width: int = 1920, projector_height: int = 1080):
        """
        Initialize the enhanced stereo scanner.
        
        Args:
            camera_matrix_left: Intrinsic matrix for left camera
            dist_coeffs_left: Distortion coefficients for left camera
            camera_matrix_right: Intrinsic matrix for right camera
            dist_coeffs_right: Distortion coefficients for right camera
            R: Rotation matrix between cameras
            T: Translation vector between cameras
            image_size: Image size (width, height)
            projector_width: Projector width
            projector_height: Projector height
        """
        self.camera_matrix_left = camera_matrix_left
        self.dist_coeffs_left = dist_coeffs_left
        self.camera_matrix_right = camera_matrix_right
        self.dist_coeffs_right = dist_coeffs_right
        self.R = R
        self.T = T
        self.image_size = image_size
        self.projector_width = projector_width
        self.projector_height = projector_height
        
        # Compute stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        
        # Compute undistortion maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_32FC1
        )
        
        # Create pattern generators
        self.gray_code = EnhancedGrayCodeGenerator(projector_width, projector_height)
        self.phase_shift = PhaseShiftGenerator(projector_width, projector_height)
        
        logger.info("Initialized Enhanced Stereo Scanner")
    
    @classmethod
    def from_calibration_file(cls, calib_file: str, 
                             projector_width: int = 1920, 
                             projector_height: int = 1080) -> 'EnhancedStereoScanner':
        """
        Create scanner from calibration file.
        
        Args:
            calib_file: Path to calibration file
            projector_width: Projector width
            projector_height: Projector height
            
        Returns:
            Initialized scanner
        """
        # Load calibration data
        calib_data = np.load(calib_file)
        
        # Extract calibration parameters
        camera_matrix_left = calib_data['camera_matrix_left']
        dist_coeffs_left = calib_data['dist_coeffs_left']
        camera_matrix_right = calib_data['camera_matrix_right']
        dist_coeffs_right = calib_data['dist_coeffs_right']
        R = calib_data['R']
        T = calib_data['T']
        image_size = tuple(calib_data['image_size'])
        
        return cls(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            R, T, image_size,
            projector_width, projector_height
        )
    
    @classmethod
    def from_default_calibration(cls, image_size: Tuple[int, int] = (1280, 720),
                                projector_width: int = 1920, 
                                projector_height: int = 1080) -> 'EnhancedStereoScanner':
        """
        Create scanner with default calibration.
        
        Args:
            image_size: Image size (width, height)
            projector_width: Projector width
            projector_height: Projector height
            
        Returns:
            Initialized scanner
        """
        # Create default calibration parameters
        width, height = image_size
        fx = width * 0.8  # Focal length x
        fy = width * 0.8  # Focal length y
        cx = width / 2    # Principal point x
        cy = height / 2   # Principal point y
        
        # Intrinsic matrices
        camera_matrix_left = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        camera_matrix_right = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # No distortion
        dist_coeffs_left = np.zeros(5)
        dist_coeffs_right = np.zeros(5)
        
        # Identity rotation
        R = np.eye(3)
        
        # Translation 100mm along X
        T = np.array([100.0, 0.0, 0.0]).reshape(3, 1)
        
        return cls(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            R, T, image_size,
            projector_width, projector_height
        )
    
    def generate_gray_code_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate Gray code patterns.
        
        Returns:
            List of pattern dictionaries
        """
        return self.gray_code.generate_pattern_sequence()
    
    def generate_phase_shift_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate phase shift patterns.
        
        Returns:
            List of pattern dictionaries
        """
        return self.phase_shift.generate_pattern_sequence()
    
    def rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo images.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            
        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        # Rectify images
        left_rect = cv2.remap(left_img, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect
    
    def process_scan(self, left_images: List[np.ndarray], right_images: List[np.ndarray],
                   use_gray_code: bool = True, use_phase_shift: bool = False,
                   mask_threshold: int = 5) -> o3dg.PointCloud:
        """
        Process scan data to generate point cloud.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            use_gray_code: Whether to use Gray code for decoding
            use_phase_shift: Whether to use phase shift for decoding
            mask_threshold: Threshold for shadow mask
            
        Returns:
            Point cloud
        """
        # Ensure we have images
        if not left_images or not right_images:
            logger.error("No images provided")
            return o3dg.PointCloud()
        
        if len(left_images) != len(right_images):
            logger.error("Number of left and right images must match")
            return o3dg.PointCloud()
        
        # Rectify all images
        rectified_left = []
        rectified_right = []
        
        logger.info("Rectifying stereo images...")
        for left_img, right_img in zip(left_images, right_images):
            left_rect, right_rect = self.rectify_images(left_img, right_img)
            rectified_left.append(left_rect)
            rectified_right.append(right_rect)
        
        # Decode patterns
        logger.info("Decoding structured light patterns...")
        
        if use_gray_code:
            # Use Gray code decoding
            left_cam_proj, left_mask = self.gray_code.decode_patterns(rectified_left)
            right_cam_proj, right_mask = self.gray_code.decode_patterns(rectified_right)
            
            # If phase shift is enabled, use it for refinement
            if use_phase_shift and len(left_images) >= len(self.gray_code.generate_pattern_sequence()) + 2:
                # Get phase shift patterns (after Gray code patterns)
                phase_left = rectified_left[len(self.gray_code.generate_pattern_sequence()):]
                phase_right = rectified_right[len(self.gray_code.generate_pattern_sequence()):]
                
                # Decode phase patterns for refinement
                left_phase, _ = self.phase_shift.decode_patterns(phase_left)
                right_phase, _ = self.phase_shift.decode_patterns(phase_right)
                
                # Refine Gray code with phase shift
                left_cam_proj = self._refine_correspondence(left_cam_proj, left_phase, left_mask)
                right_cam_proj = self._refine_correspondence(right_cam_proj, right_phase, right_mask)
        else:
            # Use phase shift decoding only
            left_cam_proj, left_mask = self.phase_shift.decode_patterns(rectified_left)
            right_cam_proj, right_mask = self.phase_shift.decode_patterns(rectified_right)
        
        # Combine masks with threshold
        if mask_threshold <= 3:
            # Very lenient - use union
            combined_mask = np.logical_or(left_mask, right_mask).astype(np.uint8)
            logger.info("Using union of masks (very lenient)")
        else:
            # Normal - use intersection
            combined_mask = np.logical_and(left_mask, right_mask).astype(np.uint8)
            logger.info("Using intersection of masks")
        
        # Find stereo correspondences
        logger.info("Finding stereo correspondences...")
        points_3d = self._find_correspondences(left_cam_proj, right_cam_proj, combined_mask)
        
        # Create point cloud
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, returning numpy array of points")
            return points_3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Filter outliers
        filtered_pcd = self._filter_point_cloud(pcd)
        
        logger.info(f"Generated point cloud with {len(filtered_pcd.points)} points")
        return filtered_pcd
    
    def _refine_correspondence(self, gray_correspondence: np.ndarray, 
                              phase_correspondence: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
        """
        Refine Gray code correspondence with phase shift.
        
        Args:
            gray_correspondence: Gray code correspondence map
            phase_correspondence: Phase shift correspondence map
            mask: Validity mask
            
        Returns:
            Refined correspondence map
        """
        refined = gray_correspondence.copy()
        
        # Where both are valid, use weighted average
        valid_mask = (gray_correspondence[:,:,0] > 0) & (phase_correspondence[:,:,0] > 0) & (mask > 0)
        
        # Compute weights - Gray code has higher weight for integer values
        gray_weight = 0.7
        phase_weight = 0.3
        
        # Weighted average
        refined[valid_mask, 0] = (gray_correspondence[valid_mask, 0] * gray_weight + 
                                phase_correspondence[valid_mask, 0] * phase_weight)
        refined[valid_mask, 1] = (gray_correspondence[valid_mask, 1] * gray_weight + 
                                phase_correspondence[valid_mask, 1] * phase_weight)
        
        return refined
    
    def _find_correspondences(self, left_cam_proj: np.ndarray, 
                             right_cam_proj: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
        """
        Find correspondences between left and right cameras using projector as intermediary.
        
        Args:
            left_cam_proj: Left camera-projector correspondence
            right_cam_proj: Right camera-projector correspondence
            mask: Combined mask
            
        Returns:
            Array of 3D points
        """
        height, width = mask.shape
        points_left = []
        points_right = []
        
        # Enhanced correspondence matching with spatial neighborhood and epipolar constraints
        
        # Build projector-to-right camera map with spatial neighborhood
        proj_to_right = {}
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                
                # Get projector coordinates
                proj_v = int(right_cam_proj[y, x, 0])
                proj_u = int(right_cam_proj[y, x, 1])
                
                if proj_v <= 0 or proj_u <= 0 or proj_v >= self.projector_height or proj_u >= self.projector_width:
                    continue
                
                # Store with neighborhood
                for dv in [-1, 0, 1]:
                    for du in [-1, 0, 1]:
                        key = (proj_v + dv, proj_u + du)
                        if key not in proj_to_right:
                            proj_to_right[key] = []
                        proj_to_right[key].append((x, y))
        
        # Find correspondences
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                
                # Get projector coordinates for left camera
                proj_v = int(left_cam_proj[y, x, 0])
                proj_u = int(left_cam_proj[y, x, 1])
                
                if proj_v <= 0 or proj_u <= 0 or proj_v >= self.projector_height or proj_u >= self.projector_width:
                    continue
                
                # Search for matches
                key = (proj_v, proj_u)
                if key in proj_to_right and proj_to_right[key]:
                    # Select best match (closest to epipolar line)
                    # For rectified images, epipolar lines are horizontal
                    best_match = None
                    min_v_diff = float('inf')
                    
                    for rx, ry in proj_to_right[key]:
                        # In rectified images, corresponding points should have similar y-coordinates
                        v_diff = abs(y - ry)
                        if v_diff < min_v_diff:
                            min_v_diff = v_diff
                            best_match = (rx, ry)
                    
                    # Use match only if within epipolar tolerance
                    if min_v_diff <= 3:  # 3-pixel tolerance for epipolar constraint
                        right_x, right_y = best_match
                        points_left.append([x, y])
                        points_right.append([right_x, right_y])
        
        # Convert to numpy arrays
        points_left = np.array(points_left, dtype=np.float32).reshape(-1, 1, 2)
        points_right = np.array(points_right, dtype=np.float32).reshape(-1, 1, 2)
        
        if len(points_left) < 10:
            logger.warning("Too few correspondences for triangulation")
            return np.array([])
        
        logger.info(f"Found {len(points_left)} corresponding points")
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(self.P1, self.P2, points_left, points_right)
        points_4d = points_4d.T
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:, :3] / points_4d[:, 3:]
        
        # Filter by distance
        dist_from_origin = np.linalg.norm(points_3d, axis=1)
        valid_dist_mask = dist_from_origin < 5000  # Adjust based on your scene scale
        
        points_3d = points_3d[valid_dist_mask]
        logger.info(f"After distance filtering: {len(points_3d)} points")
        
        return points_3d
    
    def _filter_point_cloud(self, pcd: o3dg.PointCloud, 
                          nb_neighbors: int = 30, 
                          std_ratio: float = 1.0) -> o3dg.PointCloud:
        """
        Filter point cloud to remove outliers.
        
        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors for statistical filtering
            std_ratio: Standard deviation ratio for filtering
            
        Returns:
            Filtered point cloud
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, skipping filtering")
            return pcd
        
        if len(pcd.points) == 0:
            logger.warning("Empty point cloud, nothing to filter")
            return pcd
        
        try:
            # Statistical outlier removal
            filtered_pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=min(nb_neighbors, len(pcd.points) // 2), 
                std_ratio=std_ratio
            )
            
            # Remove noise with radius outlier removal
            filtered_pcd, _ = filtered_pcd.remove_radius_outlier(
                nb_points=16, 
                radius=2.0
            )
            
            # Estimate normals
            filtered_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
            )
            
            logger.info(f"Filtered point cloud: {len(pcd.points)} → {len(filtered_pcd.points)} points")
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"Error filtering point cloud: {e}")
            return pcd
    
    def create_mesh(self, pcd: o3dg.PointCloud, 
                   depth: int = 8, 
                   smoothing: int = 2) -> o3dg.TriangleMesh:
        """
        Create mesh from point cloud.
        
        Args:
            pcd: Input point cloud
            depth: Depth parameter for Poisson reconstruction
            smoothing: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, cannot create mesh")
            return o3dg.TriangleMesh()
        
        if len(pcd.points) < 100:
            logger.warning(f"Too few points ({len(pcd.points)}) to create a mesh")
            return o3dg.TriangleMesh()
        
        try:
            # Ensure normals are available
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(k=15)
            
            # Create mesh using Poisson reconstruction
            logger.info(f"Creating mesh with Poisson reconstruction (depth={depth})")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=0, scale=1.1, linear_fit=True
            )
            
            # Remove low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Apply smoothing
            if smoothing > 0:
                logger.info(f"Smoothing mesh with {smoothing} iterations")
                mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing)
            
            logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            return o3dg.TriangleMesh()
    
    def save_point_cloud(self, pcd: Union[o3dg.PointCloud, np.ndarray], filepath: str) -> None:
        """
        Save point cloud to file.
        
        Args:
            pcd: Point cloud to save
            filepath: Output filepath
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        if not OPEN3D_AVAILABLE:
            # Save as numpy if open3d not available
            if isinstance(pcd, np.ndarray):
                np.save(filepath, pcd)
                logger.info(f"Saved point cloud as NumPy array: {filepath}")
            else:
                logger.error("Cannot save point cloud: open3d not available and input is not numpy array")
            return
        
        # Convert numpy array to point cloud if needed
        if isinstance(pcd, np.ndarray):
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
            pcd = o3d_pcd
        
        # Get file extension
        ext = os.path.splitext(filepath)[1].lower()
        
        # Save based on extension
        if ext in ['.ply', '.pcd', '.xyz', '.pts']:
            o3d.io.write_point_cloud(filepath, pcd)
        else:
            # Default to PLY
            new_path = os.path.splitext(filepath)[0] + '.ply'
            o3d.io.write_point_cloud(new_path, pcd)
            logger.info(f"Changed extension to .ply: {new_path}")
        
        logger.info(f"Saved point cloud with {len(pcd.points)} points to {filepath}")
    
    def save_mesh(self, mesh: o3dg.TriangleMesh, filepath: str) -> None:
        """
        Save mesh to file.
        
        Args:
            mesh: Mesh to save
            filepath: Output filepath
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Cannot save mesh: open3d not available")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Get file extension
        ext = os.path.splitext(filepath)[1].lower()
        
        # Save based on extension
        if ext in ['.ply', '.obj', '.off', '.stl', '.gltf']:
            o3d.io.write_triangle_mesh(filepath, mesh)
        else:
            # Default to OBJ
            new_path = os.path.splitext(filepath)[0] + '.obj'
            o3d.io.write_triangle_mesh(new_path, mesh)
            logger.info(f"Changed extension to .obj: {new_path}")
        
        logger.info(f"Saved mesh with {len(mesh.triangles)} triangles to {filepath}")