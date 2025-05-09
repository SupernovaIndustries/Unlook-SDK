"""
Robust structured light scanning implementation for UnLook SDK.

This module provides a comprehensive structured light scanning system combining 
Gray code and Phase Shift techniques for reliable 3D reconstruction in various 
lighting conditions.
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
    logger.warning("open3d not installed. 3D mesh visualization and advanced filtering will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for o3dg when open3d is not available
    class PlaceholderO3DG:
        class PointCloud:
            pass
        class TriangleMesh:
            pass
    o3dg = PlaceholderO3DG


class StructuredLightPatternGenerator:
    """
    Pattern generator for structured light scanning, combining Gray code and Phase Shift techniques
    for robust 3D reconstruction in various lighting conditions.
    """
    
    def __init__(self, 
                 projector_width: int = 1920, 
                 projector_height: int = 1080, 
                 use_gray_code: bool = True,
                 use_phase_shift: bool = True,
                 phase_shift_steps: int = 3, 
                 phase_shift_frequencies: List[int] = [8, 16, 32]):
        """
        Initialize the pattern generator.
        
        Args:
            projector_width: Width of the projector in pixels
            projector_height: Height of the projector in pixels
            use_gray_code: Whether to generate Gray code patterns
            use_phase_shift: Whether to generate phase shift patterns
            phase_shift_steps: Number of phase shifts per frequency
            phase_shift_frequencies: List of frequencies for phase shift patterns
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.use_gray_code = use_gray_code
        self.use_phase_shift = use_phase_shift
        self.phase_shift_steps = phase_shift_steps
        self.phase_shift_frequencies = phase_shift_frequencies
        
        logger.info(f"Initializing StructuredLightPatternGenerator with projector size: {projector_width}x{projector_height}")
        
        # Initialize OpenCV Gray code generator
        if self.use_gray_code:
            self.gray_code_generator = cv2.structured_light.GrayCodePattern.create(width=projector_width, height=projector_height)
            logger.info(f"Gray code patterns: {self.gray_code_generator.getNumberOfPatternImages()} patterns")
        
    def generate_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate a complete sequence of structured light patterns.
        
        Returns:
            List of pattern dictionaries with keys:
                - name: Name of the pattern
                - pattern_type: Type of pattern (raw_image, solid_field, etc.)
                - image: Raw pattern data (if raw_image)
                - other pattern-specific parameters
        """
        patterns = []
        
        # Always generate white and black reference patterns
        white_pattern = np.ones((self.projector_height, self.projector_width), dtype=np.uint8) * 255
        black_pattern = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
        
        patterns.append({
            "name": "white_reference",
            "pattern_type": "raw_image",
            "image": self._encode_image(white_pattern)
        })
        
        patterns.append({
            "name": "black_reference",
            "pattern_type": "raw_image",
            "image": self._encode_image(black_pattern)
        })
        
        # Gray code patterns (horizontal and vertical)
        if self.use_gray_code:
            # Generate patterns using OpenCV Gray code generator
            _, gray_patterns = self.gray_code_generator.generate()
            
            # Add each pattern with metadata
            for i, pattern in enumerate(gray_patterns):
                # Determine pattern orientation and bit position
                if i < len(gray_patterns) // 2:
                    orientation = "horizontal"
                    bit_pos = i // 2
                    is_inverse = (i % 2) == 1
                else:
                    orientation = "vertical"
                    bit_pos = (i - len(gray_patterns) // 2) // 2
                    is_inverse = (i % 2) == 1
                
                # Name pattern based on orientation, bit position, and whether it's inverted
                pattern_name = f"gray_code_{orientation[0]}_bit{bit_pos:02d}"
                if is_inverse:
                    pattern_name += "_inv"
                
                patterns.append({
                    "name": pattern_name,
                    "pattern_type": "raw_image",
                    "image": self._encode_image(pattern),
                    "orientation": orientation,
                    "bit_position": bit_pos,
                    "is_inverse": is_inverse
                })
        
        # Phase shift patterns
        if self.use_phase_shift:
            # Generate horizontal and vertical phase shift patterns for each frequency
            for orientation in ["horizontal", "vertical"]:
                for freq in self.phase_shift_frequencies:
                    for step in range(self.phase_shift_steps):
                        # Calculate phase shift
                        phase_offset = 2 * np.pi * step / self.phase_shift_steps
                        
                        # Create the sinusoidal pattern
                        pattern = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
                        
                        if orientation == "horizontal":
                            for x in range(self.projector_width):
                                val = 127.5 + 127.5 * np.cos(2 * np.pi * x / freq + phase_offset)
                                pattern[:, x] = val
                        else:  # vertical
                            for y in range(self.projector_height):
                                val = 127.5 + 127.5 * np.cos(2 * np.pi * y / freq + phase_offset)
                                pattern[y, :] = val
                        
                        pattern_name = f"phase_{orientation[0]}_freq{freq}_step{step}"
                        
                        patterns.append({
                            "name": pattern_name,
                            "pattern_type": "raw_image",
                            "image": self._encode_image(pattern),
                            "orientation": orientation,
                            "frequency": freq,
                            "step": step,
                            "phase_offset": phase_offset
                        })
        
        logger.info(f"Generated {len(patterns)} structured light patterns")
        return patterns
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """Encode image as JPEG."""
        success, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return data.tobytes() if success else b''


class StructuredLightDecoder:
    """
    Robust decoder for structured light patterns, combining Gray code and Phase Shift techniques.
    """
    
    def __init__(self, 
                 projector_width: int = 1920, 
                 projector_height: int = 1080,
                 white_threshold: int = 5,
                 black_threshold: int = 40,
                 min_reliability: float = 0.1):
        """
        Initialize the structured light decoder.
        
        Args:
            projector_width: Width of the projector in pixels
            projector_height: Height of the projector in pixels
            white_threshold: Threshold for differentiating between encoded values
            black_threshold: Threshold for shadow detection
            min_reliability: Minimum reliability for decoding
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.white_threshold = white_threshold
        self.black_threshold = black_threshold
        self.min_reliability = min_reliability
        
        # Initialize OpenCV Gray code decoder
        self.gray_code = cv2.structured_light.GrayCodePattern.create(width=projector_width, height=projector_height)
        self.gray_code.setWhiteThreshold(white_threshold)
        self.gray_code.setBlackThreshold(black_threshold)
        
        logger.info(f"Initialized StructuredLightDecoder with projector size: {projector_width}x{projector_height}")
    
    def decode_gray_code(self, 
                        captured_patterns: List[np.ndarray], 
                        white_ref: np.ndarray, 
                        black_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to obtain projector-camera correspondence.
        
        Args:
            captured_patterns: List of captured Gray code patterns
            white_ref: White reference image
            black_ref: Black reference image
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        # Ensure all images are grayscale
        gray_patterns = []
        for img in captured_patterns:
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_patterns.append(gray_img)
            else:
                gray_patterns.append(img.copy())
        
        # Apply preprocessing for better contrast and noise reduction
        enhanced_patterns = self._enhance_patterns(gray_patterns)
        
        # Convert white and black reference images to grayscale if needed
        if len(white_ref.shape) == 3:
            white_ref = cv2.cvtColor(white_ref, cv2.COLOR_BGR2GRAY)
        if len(black_ref.shape) == 3:
            black_ref = cv2.cvtColor(black_ref, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask
        shadow_mask = self._compute_shadow_mask(black_ref, white_ref)
        
        # Prepare arrays for OpenCV Gray code decoding
        pattern_arr = np.array(enhanced_patterns)
        empty_pattern = np.zeros_like(enhanced_patterns[0])
        black_arr = black_ref
        white_arr = white_ref
        
        # Decode patterns to get projector coordinates
        ret, proj_coords = self.gray_code.decode(
            pattern_arr, 
            empty_pattern,
            black_arr,
            white_arr
        )
        
        if not ret:
            logger.error("Gray code decoding failed")
            return np.zeros((white_ref.shape[0], white_ref.shape[1], 2), dtype=np.float32), shadow_mask
        
        # Create camera-projector correspondence map
        height, width = shadow_mask.shape
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Extract projector coordinates
        for y in range(height):
            for x in range(width):
                if shadow_mask[y, x] != 0 and proj_coords[y, x] != -1:  # Valid pixel
                    # Convert packed projector coordinate to (row, col)
                    proj_y = proj_coords[y, x] // self.projector_width
                    proj_x = proj_coords[y, x] % self.projector_width
                    
                    if 0 <= proj_x < self.projector_width and 0 <= proj_y < self.projector_height:
                        cam_proj[y, x, 0] = proj_y  # Row (V)
                        cam_proj[y, x, 1] = proj_x  # Column (U)
        
        return cam_proj, shadow_mask
    
    def decode_phase_shift(self, 
                          captured_patterns: List[np.ndarray], 
                          frequencies: List[int],
                          steps_per_freq: int,
                          white_ref: np.ndarray, 
                          black_ref: np.ndarray, 
                          orientation: str = "horizontal") -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode phase shift patterns to obtain projector-camera correspondence.
        
        Args:
            captured_patterns: List of captured phase shift patterns
            frequencies: List of frequencies used
            steps_per_freq: Number of phase shift steps per frequency
            white_ref: White reference image
            black_ref: Black reference image
            orientation: Pattern orientation ("horizontal" or "vertical")
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        # Ensure all images are grayscale
        gray_patterns = []
        for img in captured_patterns:
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_patterns.append(gray_img)
            else:
                gray_patterns.append(img.copy())
        
        # Convert white and black reference images to grayscale if needed
        if len(white_ref.shape) == 3:
            white_ref = cv2.cvtColor(white_ref, cv2.COLOR_BGR2GRAY)
        if len(black_ref.shape) == 3:
            black_ref = cv2.cvtColor(black_ref, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask
        shadow_mask = self._compute_shadow_mask(black_ref, white_ref)
        
        # Get image dimensions
        height, width = shadow_mask.shape
        
        # Initialize projector correspondence map
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Process each frequency separately
        wrapped_phases = []
        for f_idx, freq in enumerate(frequencies):
            start_idx = f_idx * steps_per_freq
            end_idx = start_idx + steps_per_freq
            
            if end_idx > len(gray_patterns):
                logger.warning(f"Not enough patterns for frequency {freq}, skipping")
                continue
            
            # Get patterns for this frequency
            freq_patterns = gray_patterns[start_idx:end_idx]
            
            # Calculate wrapped phase for this frequency
            wrapped_phase = self._calculate_wrapped_phase(freq_patterns, steps_per_freq)
            wrapped_phases.append((freq, wrapped_phase))
        
        # Unwrap phase using multi-frequency approach
        if wrapped_phases:
            # Sort by frequency (highest first)
            wrapped_phases.sort(key=lambda x: x[0], reverse=True)
            
            # Unwrap phase
            unwrapped_phase = self._unwrap_phase_multi_freq(
                [wp[1] for wp in wrapped_phases], 
                [wp[0] for wp in wrapped_phases], 
                shadow_mask
            )
            
            # Convert to projector coordinates
            max_freq = wrapped_phases[0][0]  # Highest frequency
            if orientation == "horizontal":
                for y in range(height):
                    for x in range(width):
                        if shadow_mask[y, x] != 0:
                            # Scale to projector coordinates
                            proj_x = (unwrapped_phase[y, x] / (2 * np.pi)) * max_freq
                            if 0 <= proj_x < self.projector_width:
                                cam_proj[y, x, 1] = proj_x  # Column (U)
                                cam_proj[y, x, 0] = 0  # Set row to 0 (will be filled by vertical patterns)
            else:  # vertical
                for y in range(height):
                    for x in range(width):
                        if shadow_mask[y, x] != 0:
                            # Scale to projector coordinates
                            proj_y = (unwrapped_phase[y, x] / (2 * np.pi)) * max_freq
                            if 0 <= proj_y < self.projector_height:
                                cam_proj[y, x, 0] = proj_y  # Row (V)
                                cam_proj[y, x, 1] = 0  # Set column to 0 (will be filled by horizontal patterns)
        
        return cam_proj, shadow_mask
    
    def combine_decodings(self, 
                        gray_code_result: Tuple[np.ndarray, np.ndarray], 
                        horiz_phase_result: Tuple[np.ndarray, np.ndarray],
                        vert_phase_result: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine Gray code and phase shift decodings for improved accuracy.
        
        Args:
            gray_code_result: Tuple of (cam_proj_correspondence, mask) from Gray code
            horiz_phase_result: Tuple of (cam_proj_correspondence, mask) from horizontal phase shift
            vert_phase_result: Tuple of (cam_proj_correspondence, mask) from vertical phase shift
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        gray_proj, gray_mask = gray_code_result
        horiz_proj, horiz_mask = horiz_phase_result
        vert_proj, vert_mask = vert_phase_result
        
        # Combined mask (intersection of all masks)
        combined_mask = gray_mask.copy()
        if horiz_mask is not None:
            combined_mask = combined_mask * horiz_mask
        if vert_mask is not None:
            combined_mask = combined_mask * vert_mask
        
        # Initialize combined projector correspondence map
        combined_proj = gray_proj.copy()
        
        # In areas where phase shift is valid, refine with phase shift results
        height, width = combined_mask.shape
        
        # Weight factors for combining results
        gray_weight = 0.6
        phase_weight = 0.4
        
        for y in range(height):
            for x in range(width):
                if combined_mask[y, x] != 0:
                    # Get projector coordinates from each method
                    gray_coord = gray_proj[y, x].copy()
                    
                    # Only refine when phase shift results are valid
                    if horiz_proj is not None and horiz_proj[y, x, 1] > 0:
                        # Update horizontal coordinate (column/U) with weighted average
                        combined_proj[y, x, 1] = (gray_coord[1] * gray_weight + 
                                                 horiz_proj[y, x, 1] * phase_weight)
                    
                    if vert_proj is not None and vert_proj[y, x, 0] > 0:
                        # Update vertical coordinate (row/V) with weighted average
                        combined_proj[y, x, 0] = (gray_coord[0] * gray_weight + 
                                                 vert_proj[y, x, 0] * phase_weight)
        
        return combined_proj, combined_mask
    
    def _enhance_patterns(self, patterns: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance patterns for better contrast and noise reduction.
        
        Args:
            patterns: List of grayscale patterns
            
        Returns:
            List of enhanced patterns
        """
        enhanced = []
        for img in patterns:
            # Apply adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img)
            
            # Apply bilateral filter to reduce noise while preserving edges
            img_filtered = cv2.bilateralFilter(img_enhanced, d=5, sigmaColor=75, sigmaSpace=75)
            
            enhanced.append(img_filtered)
        
        return enhanced
    
    def _compute_shadow_mask(self, black_img: np.ndarray, white_img: np.ndarray) -> np.ndarray:
        """
        Compute shadow mask using the difference between white and black images.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            
        Returns:
            Binary mask (1 for valid pixels, 0 for shadowed)
        """
        # Compute difference
        diff = cv2.absdiff(white_img, black_img)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Apply adaptive thresholding
        try:
            # Normalize for improved contrast
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply adaptive threshold
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
            _, mask = cv2.threshold(blurred, self.black_threshold, 1, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _calculate_wrapped_phase(self, patterns: List[np.ndarray], steps: int) -> np.ndarray:
        """
        Calculate wrapped phase from phase shift patterns.
        
        Args:
            patterns: List of phase shift patterns
            steps: Number of phase shift steps
            
        Returns:
            Wrapped phase map
        """
        # Get image dimensions
        height, width = patterns[0].shape
        
        # Initialize arrays for sine and cosine accumulation
        numerator = np.zeros((height, width), dtype=np.float32)
        denominator = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate sine and cosine components
        for p, pattern in enumerate(patterns):
            phase = 2 * np.pi * p / steps
            pattern = pattern.astype(np.float32)
            
            numerator += pattern * np.sin(phase)
            denominator += pattern * np.cos(phase)
        
        # Calculate wrapped phase
        wrapped_phase = np.arctan2(numerator, denominator)
        
        return wrapped_phase
    
    def _unwrap_phase_multi_freq(self, wrapped_phases: List[np.ndarray],
                              frequencies: List[int], mask: np.ndarray) -> np.ndarray:
        """
        Unwrap phase using multi-frequency approach.
        
        Args:
            wrapped_phases: List of wrapped phase maps (sorted by frequency, highest first)
            frequencies: List of frequencies (sorted, highest first)
            mask: Shadow mask
            
        Returns:
            Unwrapped phase map
        """
        # Handle case with no phase maps
        if not wrapped_phases:
            logger.warning("No wrapped phase maps available, returning zeros")
            return np.zeros_like(mask, dtype=np.float32)
        
        # If we only have one frequency, just return it (no unwrapping needed)
        if len(wrapped_phases) == 1:
            logger.warning("Only one frequency available, skipping unwrapping")
            return wrapped_phases[0] * mask
        
        # Start with highest frequency
        unwrapped = wrapped_phases[0].copy()
        high_freq = frequencies[0]
        
        # Unwrap using hierarchical approach
        for i in range(1, len(wrapped_phases)):
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


class StereoTriangulator:
    """
    Triangulates corresponding points between stereo cameras to create a 3D point cloud.
    """
    
    def __init__(self, 
                 camera_matrix_left: np.ndarray, 
                 dist_coeffs_left: np.ndarray,
                 camera_matrix_right: np.ndarray, 
                 dist_coeffs_right: np.ndarray,
                 R: np.ndarray, 
                 T: np.ndarray, 
                 image_size: Tuple[int, int]):
        """
        Initialize the stereo triangulator.
        
        Args:
            camera_matrix_left: Intrinsic matrix for left camera
            dist_coeffs_left: Distortion coefficients for left camera
            camera_matrix_right: Intrinsic matrix for right camera
            dist_coeffs_right: Distortion coefficients for right camera
            R: Rotation matrix between cameras
            T: Translation vector between cameras
            image_size: Image size (width, height)
        """
        self.camera_matrix_left = camera_matrix_left
        self.dist_coeffs_left = dist_coeffs_left
        self.camera_matrix_right = camera_matrix_right
        self.dist_coeffs_right = dist_coeffs_right
        self.R = R
        self.T = T
        self.image_size = image_size
        
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
        
        logger.info("Initialized StereoTriangulator")
    
    def rectify_images(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Rectify stereo image pairs.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            
        Returns:
            Tuple of (rectified_left_images, rectified_right_images)
        """
        rectified_left = []
        rectified_right = []
        
        for left_img, right_img in zip(left_images, right_images):
            # Rectify left image
            left_rect = cv2.remap(left_img, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
            
            # Rectify right image
            right_rect = cv2.remap(right_img, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
            
            rectified_left.append(left_rect)
            rectified_right.append(right_rect)
        
        return rectified_left, rectified_right
    
    def find_correspondences(self, 
                          left_cam_proj: np.ndarray, 
                          right_cam_proj: np.ndarray, 
                          left_mask: np.ndarray, 
                          right_mask: np.ndarray,
                          epipolar_tolerance: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find corresponding points between left and right cameras using projector as intermediary.
        
        Args:
            left_cam_proj: Left camera-projector correspondence
            right_cam_proj: Right camera-projector correspondence
            left_mask: Left camera shadow mask
            right_mask: Right camera shadow mask
            epipolar_tolerance: Maximum distance from epipolar line (in pixels)
            
        Returns:
            Tuple of (left_points, right_points) where each is an array of shape (N, 2)
        """
        # Get image dimensions
        height, width = left_mask.shape
        
        # Initialize result arrays
        left_points = []
        right_points = []
        
        # Create a mapping from projector coordinates to right camera coordinates
        proj_to_right = {}
        
        # Build projection map for right camera
        for y in range(height):
            for x in range(width):
                if right_mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Get projector coordinates
                proj_v = int(right_cam_proj[y, x, 0])
                proj_u = int(right_cam_proj[y, x, 1])
                
                if proj_v <= 0 or proj_u <= 0 or proj_v >= self.P2.shape[0] or proj_u >= self.P2.shape[1]:
                    continue  # Skip invalid projector coordinates
                
                # Use integer projector coordinates as keys
                key = (proj_v, proj_u)
                
                # Store coordinates, allowing multiple right camera points for the same projector point
                if key not in proj_to_right:
                    proj_to_right[key] = []
                
                proj_to_right[key].append((x, y))
        
        # Find correspondences
        for y in range(height):
            for x in range(width):
                if left_mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Get projector coordinates
                proj_v = int(left_cam_proj[y, x, 0])
                proj_u = int(left_cam_proj[y, x, 1])
                
                if proj_v <= 0 or proj_u <= 0 or proj_v >= self.P1.shape[0] or proj_u >= self.P1.shape[1]:
                    continue  # Skip invalid projector coordinates
                
                # Use integer projector coordinates as keys
                key = (proj_v, proj_u)
                
                # Find right camera points with the same projector coordinates
                if key in proj_to_right:
                    # For rectified images, corresponding points should have similar y-coordinates
                    # Find right camera point with y-coordinate closest to left camera y-coordinate
                    best_match = None
                    min_y_diff = float('inf')
                    
                    for rx, ry in proj_to_right[key]:
                        y_diff = abs(y - ry)
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_match = (rx, ry)
                    
                    # Only use match if it's close to the epipolar line
                    if min_y_diff <= epipolar_tolerance:
                        right_x, right_y = best_match
                        left_points.append([x, y])
                        right_points.append([right_x, right_y])
        
        return np.array(left_points), np.array(right_points)
    
    def triangulate(self, left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
        """
        Triangulate corresponding points to 3D.
        
        Args:
            left_points: Left camera points of shape (N, 2)
            right_points: Right camera points of shape (N, 2)
            
        Returns:
            3D points of shape (N, 3)
        """
        # Reshape points for triangulation
        left_pts = left_points.reshape(-1, 1, 2).astype(np.float32)
        right_pts = right_points.reshape(-1, 1, 2).astype(np.float32)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(self.P1, self.P2, left_pts, right_pts)
        points_4d = points_4d.T
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:, :3] / points_4d[:, 3:4]
        
        return points_3d
    
    def filter_points(self, points_3d: np.ndarray, max_dist: float = 10000.0) -> np.ndarray:
        """
        Filter 3D points to remove outliers.
        
        Args:
            points_3d: 3D points of shape (N, 3)
            max_dist: Maximum distance from origin
            
        Returns:
            Filtered 3D points
        """
        # Remove NaN and infinite values
        valid_mask = ~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)
        clean_pts = points_3d[valid_mask]
        
        if len(clean_pts) == 0:
            logger.warning("No valid points after removing NaN and infinite values")
            return np.array([])
        
        # Filter points that are too far from the origin
        dist = np.linalg.norm(clean_pts, axis=1)
        dist_mask = dist < max_dist
        clean_pts = clean_pts[dist_mask]
        
        if len(clean_pts) == 0:
            logger.warning(f"No valid points after distance filtering with max_dist={max_dist}")
            return np.array([])
        
        # Filter statistical outliers with Open3D if available
        if OPEN3D_AVAILABLE and len(clean_pts) > 50:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clean_pts)
            
            try:
                # Remove statistical outliers
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                # Remove radius outliers
                pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=1.0)
                
                # Convert back to numpy array
                clean_pts = np.asarray(pcd.points)
            except Exception as e:
                logger.warning(f"Failed to filter outliers with Open3D: {e}")
        
        return clean_pts


class RobustStructuredLightScanner:
    """
    Complete structured light scanning system combining Gray code and Phase Shift
    for robust 3D reconstruction in various lighting conditions.
    """
    
    def __init__(self, 
                 camera_matrix_left: np.ndarray, 
                 dist_coeffs_left: np.ndarray,
                 camera_matrix_right: np.ndarray, 
                 dist_coeffs_right: np.ndarray,
                 R: np.ndarray, 
                 T: np.ndarray, 
                 image_size: Tuple[int, int],
                 projector_width: int = 1920,
                 projector_height: int = 1080):
        """
        Initialize the robust structured light scanner.
        
        Args:
            camera_matrix_left: Intrinsic matrix for left camera
            dist_coeffs_left: Distortion coefficients for left camera
            camera_matrix_right: Intrinsic matrix for right camera
            dist_coeffs_right: Distortion coefficients for right camera
            R: Rotation matrix between cameras
            T: Translation vector between cameras
            image_size: Image size (width, height)
            projector_width: Width of the projector in pixels
            projector_height: Height of the projector in pixels
        """
        self.image_size = image_size
        self.projector_width = projector_width
        self.projector_height = projector_height
        
        # Initialize pattern generator
        self.pattern_generator = StructuredLightPatternGenerator(
            projector_width=projector_width,
            projector_height=projector_height,
            use_gray_code=True,
            use_phase_shift=True
        )
        
        # Initialize decoder
        self.decoder = StructuredLightDecoder(
            projector_width=projector_width,
            projector_height=projector_height,
            white_threshold=5,
            black_threshold=30
        )
        
        # Initialize triangulator
        self.triangulator = StereoTriangulator(
            camera_matrix_left=camera_matrix_left,
            dist_coeffs_left=dist_coeffs_left,
            camera_matrix_right=camera_matrix_right,
            dist_coeffs_right=dist_coeffs_right,
            R=R,
            T=T,
            image_size=image_size
        )
        
        logger.info("Initialized RobustStructuredLightScanner")
    
    @classmethod
    def from_calibration_file(cls, calib_file: str, 
                             image_size: Tuple[int, int],
                             projector_width: int = 1920, 
                             projector_height: int = 1080) -> 'RobustStructuredLightScanner':
        """
        Create scanner from calibration file.
        
        Args:
            calib_file: Path to calibration file
            image_size: Image size (width, height)
            projector_width: Width of the projector in pixels
            projector_height: Height of the projector in pixels
            
        Returns:
            Initialized scanner
        """
        # Load calibration data
        import h5py
        
        calib_data = {}
        try:
            with h5py.File(calib_file, 'r') as f:
                for k, v in f.items():
                    calib_data[k] = np.array(v)
        except Exception as e:
            # Try loading with numpy if h5py fails
            try:
                calib_data = dict(np.load(calib_file))
            except Exception as e2:
                raise ValueError(f"Failed to load calibration data: {e}, {e2}")
        
        # Extract calibration parameters
        camera_matrix_left = calib_data.get('M1', calib_data.get('camera_matrix_left'))
        dist_coeffs_left = calib_data.get('d1', calib_data.get('dist_coeffs_left'))
        camera_matrix_right = calib_data.get('M2', calib_data.get('camera_matrix_right'))
        dist_coeffs_right = calib_data.get('d2', calib_data.get('dist_coeffs_right'))
        R = calib_data.get('R')
        T = calib_data.get('t', calib_data.get('T'))
        
        if any(param is None for param in [camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T]):
            raise ValueError("Missing calibration parameters in calibration file")
        
        return cls(
            camera_matrix_left=camera_matrix_left,
            dist_coeffs_left=dist_coeffs_left,
            camera_matrix_right=camera_matrix_right,
            dist_coeffs_right=dist_coeffs_right,
            R=R,
            T=T,
            image_size=image_size,
            projector_width=projector_width,
            projector_height=projector_height
        )
    
    @classmethod
    def from_default_calibration(cls, 
                                image_size: Tuple[int, int],
                                projector_width: int = 1920, 
                                projector_height: int = 1080,
                                baseline: float = 100.0) -> 'RobustStructuredLightScanner':
        """
        Create scanner with default calibration parameters.
        
        Args:
            image_size: Image size (width, height)
            projector_width: Width of the projector in pixels
            projector_height: Height of the projector in pixels
            baseline: Distance between cameras in mm
            
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
        
        # Translation along X
        T = np.array([baseline, 0.0, 0.0]).reshape(3, 1)
        
        return cls(
            camera_matrix_left=camera_matrix_left,
            dist_coeffs_left=dist_coeffs_left,
            camera_matrix_right=camera_matrix_right,
            dist_coeffs_right=dist_coeffs_right,
            R=R,
            T=T,
            image_size=image_size,
            projector_width=projector_width,
            projector_height=projector_height
        )
    
    def generate_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate structured light patterns.
        
        Returns:
            List of pattern dictionaries
        """
        return self.pattern_generator.generate_patterns()
    
    def process_scan(self, 
                    left_images: List[np.ndarray], 
                    right_images: List[np.ndarray],
                    output_dir: Optional[str] = None) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Process structured light scan to generate point cloud.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            output_dir: Optional output directory for debug information
            
        Returns:
            3D point cloud (Open3D PointCloud if available, otherwise numpy array)
        """
        # Ensure we have the same number of images
        if len(left_images) != len(right_images):
            min_len = min(len(left_images), len(right_images))
            logger.warning(f"Number of left ({len(left_images)}) and right ({len(right_images)}) images don't match. Using first {min_len} images.")
            left_images = left_images[:min_len]
            right_images = right_images[:min_len]
        
        # Rectify images
        logger.info("Rectifying stereo images...")
        rectified_left, rectified_right = self.triangulator.rectify_images(left_images, right_images)
        
        # Save rectified images if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, (left, right) in enumerate(zip(rectified_left, rectified_right)):
                left_path = os.path.join(output_dir, f"left_rectified_{i:02d}.png")
                right_path = os.path.join(output_dir, f"right_rectified_{i:02d}.png")
                cv2.imwrite(left_path, left)
                cv2.imwrite(right_path, right)
        
        # Extract reference images
        left_white = rectified_left[0]
        left_black = rectified_left[1]
        right_white = rectified_right[0]
        right_black = rectified_right[1]
        
        # Decode Gray code patterns (all patterns beyond white and black)
        logger.info("Decoding Gray code patterns...")

        # Get the number of patterns for Gray code
        gray_code_patterns = self.decoder.gray_code.getNumberOfPatternImages()

        # Ensure we have enough patterns
        if len(rectified_left) < gray_code_patterns + 2:  # +2 for white and black
            logger.error(f"Not enough patterns. Need at least {gray_code_patterns + 2}, have {len(rectified_left)}")
            return o3dg.PointCloud() if OPEN3D_AVAILABLE else np.array([])

        # Split patterns for Gray code and phase shift
        left_gray_patterns = rectified_left[2:2+gray_code_patterns]
        right_gray_patterns = rectified_right[2:2+gray_code_patterns]
        
        # Decode with separate method for each camera for easier debugging
        left_cam_proj, left_mask = self.decoder.decode_gray_code(
            left_gray_patterns, 
            left_white, 
            left_black
        )
        
        right_cam_proj, right_mask = self.decoder.decode_gray_code(
            right_gray_patterns, 
            right_white, 
            right_black
        )
        
        # Debug info
        logger.info(f"Left mask: {np.sum(left_mask)} valid pixels")
        logger.info(f"Right mask: {np.sum(right_mask)} valid pixels")
        
        # Save masks if output directory provided
        if output_dir:
            left_mask_path = os.path.join(output_dir, "left_mask.png")
            right_mask_path = os.path.join(output_dir, "right_mask.png")
            cv2.imwrite(left_mask_path, left_mask * 255)
            cv2.imwrite(right_mask_path, right_mask * 255)
            
            # Save combined mask
            combined_mask = left_mask * right_mask
            combined_mask_path = os.path.join(output_dir, "combined_mask.png")
            cv2.imwrite(combined_mask_path, combined_mask * 255)
        
        # Find correspondence pairs between left and right cameras
        logger.info("Finding stereo correspondences...")
        left_points, right_points = self.triangulator.find_correspondences(
            left_cam_proj, 
            right_cam_proj, 
            left_mask, 
            right_mask,
            epipolar_tolerance=3.0
        )
        
        # Debug info
        logger.info(f"Found {len(left_points)} stereo correspondences")
        
        # If we don't have enough correspondences, return empty point cloud
        if len(left_points) < 10:
            logger.error(f"Too few correspondences ({len(left_points)}), cannot triangulate")
            return o3dg.PointCloud() if OPEN3D_AVAILABLE else np.array([])
        
        # Triangulate 3D points
        logger.info("Triangulating points...")
        points_3d = self.triangulator.triangulate(left_points, right_points)
        
        # Filter points
        logger.info("Filtering points...")
        filtered_points = self.triangulator.filter_points(points_3d)
        
        # Debug info
        logger.info(f"Final point cloud: {len(filtered_points)} points")
        
        # Save point cloud if output directory provided
        if output_dir and OPEN3D_AVAILABLE and len(filtered_points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            
            # Try to add color information from left camera images
            if len(left_images) > 0 and len(left_images[0].shape) >= 3:
                # Use the first color image for texture
                ref_img = left_images[0]
                if len(ref_img.shape) == 2:
                    # Convert grayscale to RGB if needed
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
                
                # Extract colors from 2D points
                colors = []
                for point in left_points:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= y < ref_img.shape[0] and 0 <= x < ref_img.shape[1]:
                        color = ref_img[y, x] / 255.0
                        colors.append(color[::-1])  # BGR to RGB
                    else:
                        colors.append([0.7, 0.7, 0.7])  # Default gray
                
                pcd.colors = o3d.utility.Vector3dVector(colors[:len(filtered_points)])
            
            # Save as PLY
            output_path = os.path.join(output_dir, "point_cloud.ply")
            o3d.io.write_point_cloud(output_path, pcd)
            logger.info(f"Saved point cloud to {output_path}")
            
            return pcd
        elif OPEN3D_AVAILABLE and len(filtered_points) > 0:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            return pcd
        else:
            # Return numpy array
            return filtered_points
    
    def create_mesh(self, 
                   point_cloud: Union[o3dg.PointCloud, np.ndarray], 
                   depth: int = 9, 
                   smoothing: int = 5) -> o3dg.TriangleMesh:
        """
        Create a triangle mesh from point cloud.
        
        Args:
            point_cloud: Point cloud (Open3D PointCloud or numpy array)
            depth: Depth parameter for Poisson reconstruction
            smoothing: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D not available, cannot create mesh")
            return o3dg.TriangleMesh()
        
        # Convert numpy array to Open3D point cloud if needed
        if isinstance(point_cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            pcd = point_cloud
        
        # Ensure we have enough points
        if len(pcd.points) < 100:
            logger.error(f"Not enough points ({len(pcd.points)}) to create mesh")
            return o3dg.TriangleMesh()
        
        try:
            # Estimate normals if not already computed
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
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
            logger.error(f"Failed to create mesh: {e}")
            return o3dg.TriangleMesh()
    
    def save_point_cloud(self, 
                        point_cloud: Union[o3dg.PointCloud, np.ndarray], 
                        filepath: str) -> None:
        """
        Save point cloud to file.
        
        Args:
            point_cloud: Point cloud (Open3D PointCloud or numpy array)
            filepath: Output file path
        """
        if not OPEN3D_AVAILABLE and not isinstance(point_cloud, np.ndarray):
            logger.error("Open3D not available and input is not a numpy array")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        if OPEN3D_AVAILABLE:
            # Convert numpy array to Open3D point cloud if needed
            if isinstance(point_cloud, np.ndarray):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
            else:
                pcd = point_cloud
            
            # Save point cloud
            o3d.io.write_point_cloud(filepath, pcd)
            logger.info(f"Saved point cloud to {filepath}")
        else:
            # Save as numpy file
            np.save(filepath, point_cloud)
            logger.info(f"Saved point cloud as numpy array to {filepath}")
    
    def save_mesh(self, 
                 mesh: o3dg.TriangleMesh, 
                 filepath: str) -> None:
        """
        Save mesh to file.
        
        Args:
            mesh: Triangle mesh
            filepath: Output file path
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D not available, cannot save mesh")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save mesh
        o3d.io.write_triangle_mesh(filepath, mesh)
        logger.info(f"Saved mesh to {filepath}")


if __name__ == "__main__":
    # Test pattern generation
    pattern_generator = StructuredLightPatternGenerator()
    patterns = pattern_generator.generate_patterns()
    print(f"Generated {len(patterns)} patterns")