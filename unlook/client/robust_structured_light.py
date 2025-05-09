"""
Robust structured light scanning implementation for the UnLook SDK.

This module provides a more robust implementation of structured light scanning,
incorporating proven techniques from established libraries and research.
It focuses on reliable pattern detection, correspondence finding, and
point cloud generation for real-world objects.
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


class RobustGrayCodeGenerator:
    """
    Robust Gray code pattern generator with improved decoding.
    
    This implementation draws heavily from the OpenCV structured light module
    and 3DUNDERWORLD algorithm approach with additional robustness features.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, 
                 white_threshold: int = 5, black_threshold: int = 40):
        """
        Initialize the robust Gray code generator.
        
        Args:
            width: Projector width in pixels
            height: Projector height in pixels
            white_threshold: Minimum intensity difference for a valid pixel (higher = more strict)
            black_threshold: Minimum intensity for a pixel to be considered not in shadow
        """
        self.proj_w = width
        self.proj_h = height
        self.white_threshold = white_threshold
        self.black_threshold = black_threshold
        
        # Compute required number of patterns
        self.bits_x = int(np.ceil(np.log2(width)))
        self.bits_y = int(np.ceil(np.log2(height)))
        
        # Number of patterns: 2 (all-white/all-black) + 2 * (bits_x + bits_y) normal and inverted patterns
        self.num_patterns = 2 + 2 * (self.bits_x + self.bits_y)
        
        logger.info(f"Initialized Robust Gray Code generator: {width}x{height}, {self.num_patterns} patterns")
        logger.info(f"Using {self.bits_x} bits for X axis, {self.bits_y} bits for Y axis")
    
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
        
        logger.info(f"Generated {len(patterns)} structured light patterns")
        return patterns
    
    def decode_patterns(self, captured_images: List[np.ndarray], 
                        output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Decode Gray code patterns to obtain projector-camera correspondence.
        
        Args:
            captured_images: List of captured images (white, black, patterns)
            output_dir: Optional directory to save debug images
            
        Returns:
            Tuple of (cam_proj_correspondence, mask, debug_info)
        """
        debug_info = {}
        
        if len(captured_images) < 2 + 2 * (self.bits_x + self.bits_y):
            logger.error(f"Not enough patterns: {len(captured_images)} provided, " 
                        f"{2 + 2 * (self.bits_x + self.bits_y)} required")
            return None, None, debug_info
        
        # Extract white and black images
        white_img = captured_images[0]
        black_img = captured_images[1]
        
        # Convert to grayscale if needed
        if len(white_img.shape) == 3:
            white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        if len(black_img.shape) == 3:
            black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask with adaptive approach
        shadow_mask, raw_diff = self._compute_shadow_mask(black_img, white_img)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "white.png"), white_img)
            cv2.imwrite(os.path.join(output_dir, "black.png"), black_img)
            cv2.imwrite(os.path.join(output_dir, "shadow_mask.png"), shadow_mask * 255)
            cv2.imwrite(os.path.join(output_dir, "raw_diff.png"), raw_diff)
        
        debug_info["shadow_mask_count"] = np.sum(shadow_mask)
        debug_info["image_size"] = shadow_mask.shape
        
        # Normalize white and black images for better robustness
        # This helps with varying lighting conditions
        norm_white = white_img.astype(np.float32)
        norm_black = black_img.astype(np.float32)
        brightness_range = np.maximum(norm_white - norm_black, 1.0)
        
        # Extract pattern images (starting from index 2)
        pattern_images = captured_images[2:2 + 2 * (self.bits_x + self.bits_y)]
        
        # Convert pattern images to grayscale if needed and normalize
        norm_patterns = []
        for i in range(len(pattern_images)):
            if len(pattern_images[i].shape) == 3:
                img = cv2.cvtColor(pattern_images[i], cv2.COLOR_BGR2GRAY)
            else:
                img = pattern_images[i].copy()
            
            # Normalize pattern image based on white/black range
            norm_img = (img.astype(np.float32) - norm_black) / brightness_range
            norm_img = np.clip(norm_img * 255, 0, 255).astype(np.uint8)
            norm_patterns.append(norm_img)
            
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"pattern_{i:02d}.png"), img)
                cv2.imwrite(os.path.join(output_dir, f"norm_pattern_{i:02d}.png"), norm_img)
        
        # Get image dimensions
        height, width = shadow_mask.shape
        
        # Initialize projector correspondence map
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Additional masks for debugging
        valid_bits = np.zeros_like(shadow_mask)
        ambiguous_bits = np.zeros_like(shadow_mask)
        
        # Process each pixel using a more robust method
        for y in range(height):
            for x in range(width):
                if shadow_mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Decode X coordinate (horizontal patterns)
                x_code = 0
                x_valid_bit_count = 0
                
                for bit in range(self.bits_x):
                    # Get normal pattern and its inverse for this bit
                    idx = 2 * bit
                    normal = norm_patterns[idx][y, x]
                    inverse = norm_patterns[idx + 1][y, x]
                    
                    # Check if difference is significant enough
                    if abs(int(normal) - int(inverse)) > self.white_threshold:
                        x_valid_bit_count += 1
                        # Set bit if normal is brighter than inverse
                        if normal > inverse:
                            x_code |= (1 << bit)
                
                # Decode Y coordinate (vertical patterns)
                y_code = 0
                y_valid_bit_count = 0
                
                for bit in range(self.bits_y):
                    # Get normal pattern and its inverse for this bit
                    idx = 2 * self.bits_x + 2 * bit
                    normal = norm_patterns[idx][y, x]
                    inverse = norm_patterns[idx + 1][y, x]
                    
                    # Check if difference is significant enough
                    if abs(int(normal) - int(inverse)) > self.white_threshold:
                        y_valid_bit_count += 1
                        # Set bit if normal is brighter than inverse
                        if normal > inverse:
                            y_code |= (1 << bit)
                
                # Update validity masks
                if x_valid_bit_count >= self.bits_x * 0.6 and y_valid_bit_count >= self.bits_y * 0.6:
                    valid_bits[y, x] = 1
                else:
                    # If we have some valid bits but not enough, mark as ambiguous
                    if x_valid_bit_count > 0 or y_valid_bit_count > 0:
                        ambiguous_bits[y, x] = 1
                    continue
                
                # Convert from Gray code to binary
                x_proj = self._gray_to_binary(x_code)
                y_proj = self._gray_to_binary(y_code)
                
                # Store projector coordinates if within bounds
                if 0 <= x_proj < self.proj_w and 0 <= y_proj < self.proj_h:
                    cam_proj[y, x, 0] = y_proj  # Row (V)
                    cam_proj[y, x, 1] = x_proj  # Column (U)
        
        # Update mask to include only valid pixels
        final_mask = shadow_mask.copy()
        final_mask[valid_bits == 0] = 0
        
        # Save additional debug masks
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "valid_bits_mask.png"), valid_bits * 255)
            cv2.imwrite(os.path.join(output_dir, "ambiguous_bits_mask.png"), ambiguous_bits * 255)
            cv2.imwrite(os.path.join(output_dir, "final_mask.png"), final_mask * 255)
            
            # Create visualization of the decoded projector coordinates
            proj_vis = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    if final_mask[y, x] > 0:
                        # Scale projector coordinates to 0-255 for visualization
                        r = int(255 * cam_proj[y, x, 1] / self.proj_w)
                        g = int(255 * cam_proj[y, x, 0] / self.proj_h)
                        b = 128
                        proj_vis[y, x] = [b, g, r]
            
            cv2.imwrite(os.path.join(output_dir, "proj_coords_vis.png"), proj_vis)
        
        # Add debug info
        debug_info["valid_pixels"] = np.sum(valid_bits)
        debug_info["ambiguous_pixels"] = np.sum(ambiguous_bits)
        debug_info["final_valid_pixels"] = np.sum(final_mask)
        
        return cam_proj, final_mask, debug_info
    
    def _compute_shadow_mask(self, black_img: np.ndarray, white_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust shadow mask using adaptive methods.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            
        Returns:
            Tuple of (binary_mask (1 for valid pixels, 0 for shadowed), diff_image)
        """
        # Compute difference between white and black images
        diff = cv2.absdiff(white_img, black_img)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Try multiple thresholding approaches and combine them for robustness
        try:
            # Method 1: Adaptive thresholding (works well in varied lighting)
            # Normalize for improved contrast
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply adaptive threshold
            adaptive_mask = cv2.adaptiveThreshold(
                normalized, 
                1, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21, 
                -5
            )
            
            # Method 2: Otsu thresholding (works well for bimodal histograms)
            _, otsu_mask = cv2.threshold(
                blurred,
                0,
                1,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Method 3: Simple threshold based on black_threshold
            _, simple_mask = cv2.threshold(
                blurred, 
                self.black_threshold, 
                1, 
                cv2.THRESH_BINARY
            )
            
            # Combine masks with OR operation for more relaxed masking
            # This helps include more potentially useful pixels
            combined_mask = np.logical_or(
                np.logical_or(adaptive_mask, otsu_mask),
                simple_mask
            ).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Advanced thresholding failed: {e}, using simple threshold")
            # Fall back to simple thresholding
            _, combined_mask = cv2.threshold(
                blurred, 
                self.black_threshold, 
                1, 
                cv2.THRESH_BINARY
            )
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask, blurred
    
    def _gray_to_binary(self, n: int) -> int:
        """
        Convert Gray code to binary.
        
        Args:
            n: Gray code value
            
        Returns:
            Binary value
        """
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """
        Encode image as JPEG.
        
        Args:
            image: Image to encode
            
        Returns:
            JPEG encoded image as bytes
        """
        success, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return data.tobytes() if success else b''


class RobustStereoScanner:
    """
    Robust stereo structured light scanner for improved 3D reconstruction.
    
    This implementation focuses on practical scanning of real-world objects,
    with improved correspondence finding and point cloud generation.
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
        Initialize the robust stereo scanner.
        
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
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1  # -1 for valid pixels only, 0 for all pixels valid
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
        
        # Create pattern generator with more relaxed parameters for real-world objects
        self.graycode = RobustGrayCodeGenerator(
            projector_width, 
            projector_height,
            white_threshold=5,  # Lower for more points (less strict)
            black_threshold=10  # Lower for more points in darker areas
        )
        
        logger.info("Initialized Robust Stereo Scanner")
    
    @classmethod
    def from_calibration_file(cls, calib_file: str, 
                             projector_width: int = 1920, 
                             projector_height: int = 1080) -> 'RobustStereoScanner':
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
        try:
            calib_data = np.load(calib_file)
            # Extract calibration parameters
            camera_matrix_left = calib_data['camera_matrix_left']
            dist_coeffs_left = calib_data['dist_coeffs_left']
            camera_matrix_right = calib_data['camera_matrix_right']
            dist_coeffs_right = calib_data['dist_coeffs_right']
            R = calib_data['R']
            T = calib_data['T']
            image_size = tuple(calib_data['image_size'])
        except Exception as e:
            logger.error(f"Error loading calibration file: {e}")
            logger.warning("Using default calibration parameters")
            return cls.from_default_calibration(
                image_size=(1280, 720),
                projector_width=projector_width,
                projector_height=projector_height
            )
        
        return cls(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            R, T, image_size,
            projector_width, projector_height
        )
    
    @classmethod
    def from_default_calibration(cls, image_size: Tuple[int, int] = (1280, 720),
                                projector_width: int = 1920, 
                                projector_height: int = 1080) -> 'RobustStereoScanner':
        """
        Create scanner with default calibration suitable for quick testing.
        
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
        
        # Translation 10cm along X axis - typical stereo setup
        T = np.array([100.0, 0.0, 0.0]).reshape(3, 1)
        
        logger.info("Using default calibration parameters")
        return cls(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            R, T, image_size,
            projector_width, projector_height
        )
    
    def generate_gray_code_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate Gray code patterns for projection.
        
        Returns:
            List of pattern dictionaries
        """
        return self.graycode.generate_pattern_sequence()
    
    def rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo images for correspondence finding.
        
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
                   output_dir: Optional[str] = None) -> Tuple[o3dg.PointCloud, Dict[str, Any]]:
        """
        Process scan data to generate point cloud with detailed debugging.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            output_dir: Optional directory to save debug images and data
            
        Returns:
            Tuple of (point_cloud, debug_info)
        """
        debug_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_images": len(left_images)
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving debug information to {output_dir}")
        
        # Ensure we have images
        if not left_images or not right_images:
            logger.error("No images provided")
            return o3dg.PointCloud(), debug_info
        
        if len(left_images) != len(right_images):
            logger.error("Number of left and right images must match")
            return o3dg.PointCloud(), debug_info
        
        # Create debug directories
        left_debug_dir = os.path.join(output_dir, "left_debug") if output_dir else None
        right_debug_dir = os.path.join(output_dir, "right_debug") if output_dir else None
        
        if left_debug_dir:
            os.makedirs(left_debug_dir, exist_ok=True)
        if right_debug_dir:
            os.makedirs(right_debug_dir, exist_ok=True)
        
        # Rectify all images
        rectified_left = []
        rectified_right = []
        
        logger.info("Rectifying stereo images...")
        for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
            # Save original images
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"left_original_{i:02d}.png"), left_img)
                cv2.imwrite(os.path.join(output_dir, f"right_original_{i:02d}.png"), right_img)
            
            # Rectify images
            left_rect, right_rect = self.rectify_images(left_img, right_img)
            rectified_left.append(left_rect)
            rectified_right.append(right_rect)
            
            # Save rectified images
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"left_rectified_{i:02d}.png"), left_rect)
                cv2.imwrite(os.path.join(output_dir, f"right_rectified_{i:02d}.png"), right_rect)
        
        # Decode patterns for both cameras
        logger.info("Decoding structured light patterns...")
        
        # Decode left camera patterns
        left_cam_proj, left_mask, left_debug = self.graycode.decode_patterns(
            rectified_left, left_debug_dir
        )
        
        # Decode right camera patterns
        right_cam_proj, right_mask, right_debug = self.graycode.decode_patterns(
            rectified_right, right_debug_dir
        )
        
        # Store debug info
        debug_info["left_decode"] = left_debug
        debug_info["right_decode"] = right_debug
        
        if left_mask is None or right_mask is None:
            logger.error("Pattern decoding failed")
            return o3dg.PointCloud(), debug_info
        
        logger.info(f"Left camera: {left_debug['final_valid_pixels']} valid pixels")
        logger.info(f"Right camera: {right_debug['final_valid_pixels']} valid pixels")
        
        # Combine masks
        combined_mask = np.logical_and(left_mask, right_mask).astype(np.uint8)
        
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "combined_mask.png"), combined_mask * 255)
        
        debug_info["combined_valid_pixels"] = np.sum(combined_mask)
        logger.info(f"Combined: {debug_info['combined_valid_pixels']} valid pixels")
        
        # Find stereo correspondences - relaxed approach
        logger.info("Finding stereo correspondences...")
        points_3d, corr_info = self._find_correspondences_relaxed(
            left_cam_proj, right_cam_proj, combined_mask, left_images, right_images, output_dir
        )
        
        debug_info.update(corr_info)
        
        # Create point cloud
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, returning numpy array of points")
            return points_3d, debug_info
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Filter outliers with multiple methods for better results
        filtered_pcd, filter_info = self._filter_point_cloud_multi(pcd, output_dir)
        debug_info.update(filter_info)
        
        logger.info(f"Generated point cloud with {len(filtered_pcd.points)} points")
        
        # Save raw and filtered point clouds
        if output_dir and OPEN3D_AVAILABLE and len(pcd.points) > 0:
            raw_pcd_path = os.path.join(output_dir, "raw_point_cloud.ply")
            o3d.io.write_point_cloud(raw_pcd_path, pcd)
            
            filtered_pcd_path = os.path.join(output_dir, "filtered_point_cloud.ply")
            o3d.io.write_point_cloud(filtered_pcd_path, filtered_pcd)
        
        return filtered_pcd, debug_info
    
    def _find_correspondences_relaxed(self, left_cam_proj: np.ndarray, 
                                    right_cam_proj: np.ndarray,
                                    mask: np.ndarray,
                                    left_images: List[np.ndarray] = None,
                                    right_images: List[np.ndarray] = None,
                                    output_dir: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Find correspondences between left and right cameras with relaxed constraints.
        
        Args:
            left_cam_proj: Left camera-projector correspondence
            right_cam_proj: Right camera-projector correspondence
            mask: Combined mask
            left_images: Original left camera images for visualization (optional)
            right_images: Original right camera images for visualization (optional)
            output_dir: Optional directory to save debug images
            
        Returns:
            Tuple of (array of 3D points, debug_info)
        """
        debug_info = {}
        height, width = mask.shape
        points_left = []
        points_right = []
        
        # Epipolar constraint relaxation - for real-world scanning 
        epipolar_tolerance = 5  # Pixels - higher value is more relaxed
        
        # Projector coordinate tolerance - for better matching
        proj_coord_tolerance = 3  # Pixels - higher is more relaxed
        
        # Create projector-to-right camera map with spatial neighborhood
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
                
                # Store with neighborhood for better matching
                for dv in range(-proj_coord_tolerance, proj_coord_tolerance + 1):
                    for du in range(-proj_coord_tolerance, proj_coord_tolerance + 1):
                        key = (proj_v + dv, proj_u + du)
                        if key not in proj_to_right:
                            proj_to_right[key] = []
                        proj_to_right[key].append((x, y))
        
        # Debug for projector map coverage
        debug_info["proj_map_keys"] = len(proj_to_right)
        
        # Find correspondences
        correspondence_counts = []
        epipolar_diffs = []
        
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                
                # Get projector coordinates for left camera
                proj_v = int(left_cam_proj[y, x, 0])
                proj_u = int(left_cam_proj[y, x, 1])
                
                if proj_v <= 0 or proj_u <= 0 or proj_v >= self.projector_height or proj_u >= self.projector_width:
                    continue
                
                # Search for matches with expanded search
                matches = []
                for dv in range(-proj_coord_tolerance, proj_coord_tolerance + 1):
                    for du in range(-proj_coord_tolerance, proj_coord_tolerance + 1):
                        key = (proj_v + dv, proj_u + du)
                        if key in proj_to_right and proj_to_right[key]:
                            matches.extend(proj_to_right[key])
                
                # If we found matches
                if matches:
                    correspondence_counts.append(len(matches))
                    
                    # Select best match (closest to epipolar line)
                    # For rectified images, epipolar lines are horizontal
                    best_match = None
                    min_v_diff = float('inf')
                    
                    for rx, ry in matches:
                        # In rectified images, corresponding points should have similar y-coordinates
                        v_diff = abs(y - ry)
                        if v_diff < min_v_diff:
                            min_v_diff = v_diff
                            best_match = (rx, ry)
                    
                    # Store epipolar difference for debugging
                    epipolar_diffs.append(min_v_diff)
                    
                    # Use match only if within epipolar tolerance
                    if min_v_diff <= epipolar_tolerance:
                        right_x, right_y = best_match
                        points_left.append([x, y])
                        points_right.append([right_x, right_y])
        
        # Convert to numpy arrays
        if not points_left:
            logger.error("No correspondences found")
            return np.array([]), {"correspondences": 0}
        
        points_left = np.array(points_left, dtype=np.float32).reshape(-1, 1, 2)
        points_right = np.array(points_right, dtype=np.float32).reshape(-1, 1, 2)
        
        debug_info["correspondences"] = len(points_left)
        
        if len(points_left) < 10:
            logger.warning("Too few correspondences for triangulation")
            return np.array([]), debug_info
        
        logger.info(f"Found {len(points_left)} corresponding points")
        
        # Compute average epipolar difference for debugging
        if epipolar_diffs:
            debug_info["avg_epipolar_diff"] = sum(epipolar_diffs) / len(epipolar_diffs)
            debug_info["max_epipolar_diff"] = max(epipolar_diffs)
        
        # Compute average correspondence count for debugging
        if correspondence_counts:
            debug_info["avg_correspondence_count"] = sum(correspondence_counts) / len(correspondence_counts)
            debug_info["max_correspondence_count"] = max(correspondence_counts)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(self.P1, self.P2, points_left, points_right)
        points_4d = points_4d.T
        
        # Convert from homogeneous to 3D with numerical stability improvements
        # Avoid division by very small numbers
        w = points_4d[:, 3:].copy()
        # Replace zero or very small values with 1.0 to avoid division problems
        mask = np.abs(w) < 1e-10
        w[mask] = 1.0
        
        # Now safely divide
        points_3d = points_4d[:, :3] / w
        
        # Basic distance filtering - can be adjusted for scene scale
        dist_from_origin = np.linalg.norm(points_3d, axis=1)
        valid_dist_mask = (dist_from_origin > 10) & (dist_from_origin < 5000)
        
        points_3d = points_3d[valid_dist_mask]
        debug_info["after_dist_filter"] = len(points_3d)
        
        # Save correspondence visualization
        if output_dir and len(points_left) > 0:
            try:
                # Create visualization of correspondences
                if left_images and right_images and len(left_images) > 0 and len(right_images) > 0:
                    # Use the captured images if available
                    left_img = left_images[0].copy()
                    right_img = right_images[0].copy()
                else:
                    # Otherwise create blank images with the right size
                    left_img = np.zeros((height, width, 3), dtype=np.uint8)
                    right_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Ensure images are in color for visualization
                if len(left_img.shape) == 2:
                    left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                if len(right_img.shape) == 2:
                    right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
                
                # Draw points
                for i in range(min(len(points_left), 100)):  # Limit to 100 points for clarity
                    x, y = points_left[i][0]
                    cv2.circle(left_img, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    x, y = points_right[i][0]
                    cv2.circle(right_img, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # Resize images to ensure they have the same height
                if left_img.shape[0] != right_img.shape[0]:
                    # Resize to match the smaller height
                    target_height = min(left_img.shape[0], right_img.shape[0])
                    scale_left = target_height / left_img.shape[0]
                    scale_right = target_height / right_img.shape[0]
                    
                    left_img = cv2.resize(left_img, (int(left_img.shape[1] * scale_left), target_height))
                    right_img = cv2.resize(right_img, (int(right_img.shape[1] * scale_right), target_height))
                
                # Combine images side by side
                combined = np.hstack((left_img, right_img))
                
                # Draw epipolar lines
                for y in range(0, combined.shape[0], 50):  # Draw every 50 pixels
                    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 0, 255), 1)
                
                # Add text labels
                cv2.putText(combined, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Right Camera", (left_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save the visualization
                cv2.imwrite(os.path.join(output_dir, "correspondence_vis.png"), combined)
                
                logger.info(f"Saved correspondence visualization to {os.path.join(output_dir, 'correspondence_vis.png')}")
            except Exception as e:
                logger.warning(f"Failed to create correspondence visualization: {e}")
        
        return points_3d, debug_info
    
    def _filter_point_cloud_multi(self, pcd: o3dg.PointCloud, 
                                output_dir: Optional[str] = None) -> Tuple[o3dg.PointCloud, Dict[str, Any]]:
        """
        Filter point cloud using multiple methods for better results.
        
        Args:
            pcd: Input point cloud
            output_dir: Optional output directory for debug files
            
        Returns:
            Tuple of (filtered_point_cloud, debug_info)
        """
        debug_info = {"original_points": len(pcd.points)}
        
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, skipping filtering")
            return pcd, debug_info
        
        if len(pcd.points) == 0:
            logger.warning("Empty point cloud, nothing to filter")
            return pcd, debug_info
        
        try:
            # Check how many points we have
            pcd_size = len(pcd.points)
            debug_info["original_point_count"] = pcd_size
            
            # For small point clouds, use very light filtering to avoid removing too many points
            if pcd_size < 100:
                logger.warning(f"Small point cloud with only {pcd_size} points, using minimal filtering")
                # Only remove extreme outliers
                filtered_pcd, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=min(10, max(3, pcd_size // 5)),
                    std_ratio=5.0  # Very relaxed filtering
                )
                debug_info["statistical_filter_points"] = len(filtered_pcd.points)
                
                # Skip radius filtering for very small point clouds
                if len(filtered_pcd.points) > 0 and output_dir:
                    stage1_path = os.path.join(output_dir, "stage1_minimal_filter.ply")
                    o3d.io.write_point_cloud(stage1_path, filtered_pcd)
            
            # For medium-sized point clouds, use gentle filtering
            elif pcd_size < 500:
                logger.info(f"Medium-sized point cloud with {pcd_size} points, using gentle filtering")
                # Use more relaxed filtering parameters
                filtered_pcd, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=min(15, max(5, pcd_size // 20)),
                    std_ratio=3.0  # Relaxed filtering
                )
                debug_info["statistical_filter_points"] = len(filtered_pcd.points)
                
                if output_dir and len(filtered_pcd.points) > 0:
                    stage1_path = os.path.join(output_dir, "stage1_gentle_filter.ply")
                    o3d.io.write_point_cloud(stage1_path, filtered_pcd)
                
                # Only do radius filtering if we still have enough points
                if len(filtered_pcd.points) > 50:
                    filtered_pcd, ind = filtered_pcd.remove_radius_outlier(
                        nb_points=min(5, max(2, len(filtered_pcd.points) // 50)),
                        radius=10.0  # Larger radius, more relaxed
                    )
                    debug_info["radius_filter_points"] = len(filtered_pcd.points)
                
                if output_dir and len(filtered_pcd.points) > 0:
                    stage2_path = os.path.join(output_dir, "stage2_gentle_radius_filter.ply")
                    o3d.io.write_point_cloud(stage2_path, filtered_pcd)
            
            # For larger point clouds, use normal filtering
            else:
                logger.info(f"Normal-sized point cloud with {pcd_size} points, using standard filtering")
                # Stage 1: Statistical outlier removal with relaxed parameters
                filtered_pcd, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=min(20, max(10, len(pcd.points) // 20)), 
                    std_ratio=2.5  # More relaxed for real-world objects
                )
                debug_info["statistical_filter_points"] = len(filtered_pcd.points)
                
                if output_dir and len(filtered_pcd.points) > 0:
                    stage1_path = os.path.join(output_dir, "stage1_statistical_filter.ply")
                    o3d.io.write_point_cloud(stage1_path, filtered_pcd)
                
                # Stage 2: Radius outlier removal for more isolated points
                filtered_pcd, ind = filtered_pcd.remove_radius_outlier(
                    nb_points=min(10, max(3, len(filtered_pcd.points) // 50)),  
                    radius=8.0  # Adjusted for scale, more relaxed
                )
                debug_info["radius_filter_points"] = len(filtered_pcd.points)
                
                if output_dir and len(filtered_pcd.points) > 0:
                    stage2_path = os.path.join(output_dir, "stage2_radius_filter.ply")
                    o3d.io.write_point_cloud(stage2_path, filtered_pcd)
            
            # Safety check - if we've filtered too aggressively, revert to original
            if len(filtered_pcd.points) < max(10, pcd_size * 0.05):  # Less than 5% of original points
                logger.warning(f"Filtering removed too many points ({len(filtered_pcd.points)}/{pcd_size}), reverting to original point cloud")
                filtered_pcd = pcd
                debug_info["filtering_reverted"] = True
            
            if output_dir and len(filtered_pcd.points) > 0:
                stage2_path = os.path.join(output_dir, "stage2_radius_filter.ply")
                o3d.io.write_point_cloud(stage2_path, filtered_pcd)
            
            # If point cloud is still large enough, estimate normals
            if len(filtered_pcd.points) > 100:
                try:
                    filtered_pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30)
                    )
                    filtered_pcd.orient_normals_consistent_tangent_plane(k=20)
                except Exception as e:
                    logger.warning(f"Normal estimation failed: {e}")
            
            logger.info(f"Filtered point cloud: {len(pcd.points)} → {len(filtered_pcd.points)} points")
            return filtered_pcd, debug_info
            
        except Exception as e:
            logger.error(f"Error filtering point cloud: {e}")
            return pcd, debug_info
    
    def create_mesh(self, pcd: o3dg.PointCloud, 
                   depth: int = 8, 
                   smoothing: int = 2,
                   output_dir: Optional[str] = None) -> o3dg.TriangleMesh:
        """
        Create mesh from point cloud with multiple methods for better success rates.
        
        Args:
            pcd: Input point cloud
            depth: Depth parameter for Poisson reconstruction
            smoothing: Number of smoothing iterations
            output_dir: Optional output directory for debug files
            
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
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(k=20)
            
            # Try multiple reconstruction methods for better results
            
            # Method 1: Poisson reconstruction (best for smooth, clean point clouds)
            try:
                logger.info(f"Creating mesh with Poisson reconstruction (depth={depth})")
                poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, width=0, scale=1.1, linear_fit=True
                )
                
                # Remove low-density vertices for better results
                vertices_to_remove = densities < np.quantile(densities, 0.1)
                poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
                
                if output_dir and len(poisson_mesh.triangles) > 0:
                    poisson_path = os.path.join(output_dir, "poisson_mesh.ply")
                    o3d.io.write_triangle_mesh(poisson_path, poisson_mesh)
            except Exception as e:
                logger.warning(f"Poisson reconstruction failed: {e}")
                poisson_mesh = o3d.geometry.TriangleMesh()
            
            # Method 2: Ball pivoting (better for flat surfaces and sharp features)
            try:
                logger.info("Creating mesh with Ball Pivoting Algorithm")
                # Estimate radius for ball pivoting
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]  # Progressive radii
                
                ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                
                if output_dir and len(ball_mesh.triangles) > 0:
                    ball_path = os.path.join(output_dir, "ball_mesh.ply")
                    o3d.io.write_triangle_mesh(ball_path, ball_mesh)
            except Exception as e:
                logger.warning(f"Ball pivoting failed: {e}")
                ball_mesh = o3d.geometry.TriangleMesh()
            
            # Choose the mesh with more triangles
            if len(poisson_mesh.triangles) > len(ball_mesh.triangles):
                mesh = poisson_mesh
                logger.info(f"Using Poisson mesh with {len(mesh.triangles)} triangles")
            else:
                mesh = ball_mesh
                logger.info(f"Using Ball Pivoting mesh with {len(mesh.triangles)} triangles")
            
            # Apply smoothing if needed
            if smoothing > 0 and len(mesh.triangles) > 0:
                logger.info(f"Smoothing mesh with {smoothing} iterations")
                mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing)
            
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