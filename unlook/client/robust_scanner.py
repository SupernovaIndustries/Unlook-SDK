"""
Robust Structured Light Scanner Module

This module implements a robust structured light scanning system
based on techniques from SLStudio and our existing implementation.
The focus is on stable stereo correspondence and accurate triangulation.
"""

import os
import time
import logging
import numpy as np
import cv2
import json
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies (Open3D for point cloud processing)
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. 3D mesh visualization and processing will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for open3d when not available
    class PlaceholderO3D:
        class geometry:
            class PointCloud:
                pass
        class utility:
            class Vector3dVector:
                pass
    o3d = PlaceholderO3D()
    o3dg = PlaceholderO3D.geometry

# Try to use CUDA if available
try:
    # Check if CUDA is available via OpenCV
    cv2.cuda.getCudaEnabledDeviceCount()
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if CUDA_AVAILABLE:
        logger.info("CUDA support detected, GPU acceleration enabled")
    else:
        logger.warning("CUDA not available, using CPU processing only")
except:
    CUDA_AVAILABLE = False
    logger.warning("CUDA not available, using CPU processing only")


class Pattern:
    """Base class for structured light patterns."""
    
    def __init__(self, name: str, pattern_type: str, width: int, height: int):
        """
        Initialize base pattern.
        
        Args:
            name: Pattern name
            pattern_type: Type of pattern
            width: Pattern width
            height: Pattern height
        """
        self.name = name
        self.pattern_type = pattern_type
        self.width = width
        self.height = height
        self.data = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for projection."""
        raise NotImplementedError("Subclasses must implement to_dict()")


class GrayCodePattern(Pattern):
    """Gray code pattern for structured light scanning."""
    
    def __init__(
        self, 
        bit: int, 
        orientation: str = "horizontal",
        inverted: bool = False,
        width: int = 1024, 
        height: int = 768
    ):
        """
        Initialize Gray code pattern.
        
        Args:
            bit: Bit position for Gray code
            orientation: "horizontal" or "vertical"
            inverted: Whether pattern is inverted
            width: Pattern width
            height: Pattern height
        """
        inv_text = "_inv" if inverted else ""
        name = f"gray_code_{orientation[0]}_bit{bit:02d}{inv_text}"
        super().__init__(name=name, pattern_type="raw_image", width=width, height=height)
        
        self.bit = bit
        self.orientation = orientation
        self.inverted = inverted
        
        # Generate the pattern image
        self._generate_pattern()
    
    def _generate_pattern(self):
        """Generate Gray code pattern image."""
        # Create a blank image
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Calculate stripe width based on bit position
        stripe_width = 2 ** self.bit
        
        # Fill with Gray code pattern
        if self.orientation == "horizontal":
            for x in range(self.width):
                # Use Gray code binary pattern based on position
                if ((x // stripe_width) % 2) == 0:
                    img[:, x] = 255 if not self.inverted else 0
                else:
                    img[:, x] = 0 if not self.inverted else 255
        else:  # vertical
            for y in range(self.height):
                if ((y // stripe_width) % 2) == 0:
                    img[y, :] = 255 if not self.inverted else 0
                else:
                    img[y, :] = 0 if not self.inverted else 255
        
        # Store the raw image data
        success, encoded = cv2.imencode('.png', img)
        if success:
            self.data = encoded.tobytes()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for projection."""
        return {
            "pattern_type": "raw_image",
            "name": self.name,
            "orientation": self.orientation,
            "bit_position": self.bit,
            "is_inverse": self.inverted,
            "image": self.data
        }


class RobustPatternSet:
    """
    Generates robust structured light pattern sets for 3D scanning.
    
    This class creates patterns optimized for reliable stereo correspondence
    based on techniques from SLStudio and our existing implementation.
    """
    
    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        num_gray_codes: int = 8,
        use_phase_shift: bool = False,
        num_phase_shifts: int = 4,
        use_gpu: bool = False
    ):
        """
        Initialize pattern generator.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
            num_gray_codes: Number of Gray code bits
            use_phase_shift: Whether to include phase shift patterns
            num_phase_shifts: Number of phase shifts if enabled
            use_gpu: Whether to use GPU acceleration
        """
        self.width = width
        self.height = height
        self.num_gray_codes = num_gray_codes
        self.use_phase_shift = use_phase_shift
        self.num_phase_shifts = num_phase_shifts
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        
        # Generate patterns
        self.patterns = self._generate_patterns()
        logger.info(f"Generated {len(self.patterns)} robust patterns")
    
    def _generate_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate robust pattern set.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Add white and black reference patterns (always needed)
        patterns.append({"pattern_type": "solid_field", "color": "White", "name": "white_reference"})
        patterns.append({"pattern_type": "solid_field", "color": "Black", "name": "black_reference"})
        
        # Add Gray code patterns (both horizontal and vertical)
        # Horizontal patterns encode vertical coordinates
        # Vertical patterns encode horizontal coordinates
        for bit in range(self.num_gray_codes):
            # Horizontal patterns
            h_normal = GrayCodePattern(bit=bit, orientation="horizontal", 
                                    inverted=False, width=self.width, height=self.height)
            h_inverted = GrayCodePattern(bit=bit, orientation="horizontal", 
                                        inverted=True, width=self.width, height=self.height)
            patterns.append(h_normal.to_dict())
            patterns.append(h_inverted.to_dict())
            
            # Vertical patterns
            v_normal = GrayCodePattern(bit=bit, orientation="vertical",
                                     inverted=False, width=self.width, height=self.height)
            v_inverted = GrayCodePattern(bit=bit, orientation="vertical",
                                        inverted=True, width=self.width, height=self.height)
            patterns.append(v_normal.to_dict())
            patterns.append(v_inverted.to_dict())
        
        # TODO: Add phase shift patterns if requested
        # This would be similar to our existing PhaseShiftPattern class
        
        return patterns
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get the pattern set."""
        return self.patterns


class GrayCodeDecoder:
    """
    Robust Gray code decoder for structured light scanning.
    
    This class implements a reliable decoding algorithm based on
    techniques from SLStudio, with optimizations for our use case.
    """
    
    def __init__(self, pattern_width: int, pattern_height: int, use_gpu: bool = False):
        """
        Initialize Gray code decoder.
        
        Args:
            pattern_width: Width of the projection pattern
            pattern_height: Height of the projection pattern
            use_gpu: Whether to use GPU for decoding
        """
        self.pattern_width = pattern_width
        self.pattern_height = pattern_height
        self.use_gpu = use_gpu and CUDA_AVAILABLE
    
    def decode(
        self,
        gray_images: List[np.ndarray],
        white_ref: np.ndarray,
        black_ref: np.ndarray,
        orientation: str = "vertical",
        debug_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns.
        
        Args:
            gray_images: List of Gray code images
            white_ref: White reference image
            black_ref: Black reference image
            orientation: "horizontal" or "vertical"
            debug_dir: Optional directory to save debug images
            
        Returns:
            Tuple of (decoded_coords, mask)
        """
        # Compute shadow mask from reference images
        diff = cv2.absdiff(white_ref, black_ref)
        
        # Convert to grayscale if needed
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff
        
        # Use a relatively low threshold for better sensitivity
        _, mask = cv2.threshold(diff_gray, 15, 1, cv2.THRESH_BINARY)
        
        # Save debug images if requested
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "white_ref.png"), white_ref)
            cv2.imwrite(os.path.join(debug_dir, "black_ref.png"), black_ref)
            cv2.imwrite(os.path.join(debug_dir, "diff.png"), diff)
            cv2.imwrite(os.path.join(debug_dir, "mask.png"), mask * 255)
        
        # Convert images to grayscale if needed
        gray_processed = []
        for i, img in enumerate(gray_images):
            if img is None:
                logger.warning(f"Gray code image {i} is None!")
                img = np.zeros_like(white_ref if white_ref is not None else np.zeros((100, 100)))
            
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_processed.append(gray_img)
            else:
                gray_processed.append(img)
            
            # Save gray code images for debugging
            if debug_dir and i < 4:  # Save first few only
                cv2.imwrite(os.path.join(debug_dir, f"gray_code_{i}.png"), img)
        
        # Determine number of patterns (must be even with half for normal and half for inverted)
        num_patterns = len(gray_processed)
        if num_patterns % 2 != 0:
            logger.warning(f"Odd number of Gray code patterns: {num_patterns}, expected even number")
            gray_processed = gray_processed[:num_patterns-1]
            num_patterns = len(gray_processed)
        
        # Number of bits is half the patterns (each bit has normal + inverted pattern)
        num_bits = num_patterns // 2
        
        # Create binary representation
        binary_codes = np.zeros((mask.shape[0], mask.shape[1], num_bits), dtype=np.uint8)
        
        # Decode using GPU if available
        if self.use_gpu:
            try:
                # Upload mask to GPU
                mask_gpu = cv2.cuda_GpuMat(mask)
                
                # Process each bit
                for i in range(num_bits):
                    normal_idx = i * 2
                    inverted_idx = i * 2 + 1
                    
                    # Ensure grayscale format
                    normal = gray_processed[normal_idx]
                    inverted = gray_processed[inverted_idx]
                    
                    # Upload to GPU
                    normal_gpu = cv2.cuda_GpuMat(normal)
                    inverted_gpu = cv2.cuda_GpuMat(inverted)
                    
                    # Compute absolute difference for robust thresholding
                    diff_gpu = cv2.cuda.absdiff(normal_gpu, inverted_gpu)
                    _, thresh_gpu = cv2.cuda.threshold(diff_gpu, 20, 255, cv2.THRESH_BINARY)
                    
                    # Download results
                    normal_cpu = normal_gpu.download()
                    inverted_cpu = inverted_gpu.download()
                    thresh_cpu = thresh_gpu.download()
                    
                    # 1 where normal > inverted, 0 otherwise
                    bit_val = (normal_cpu > inverted_cpu).astype(np.uint8)
                    
                    # Apply mask - only set bits where difference is significant
                    bit_val = bit_val & (thresh_cpu > 0)
                    
                    # Store in binary_codes
                    binary_codes[:, :, i] = bit_val
                
                logger.debug("GPU Gray code decoding completed successfully")
                
            except Exception as e:
                logger.error(f"GPU Gray code decoding failed: {e}, falling back to CPU")
                # Fall back to CPU implementation
                self._decode_cpu(gray_processed, num_bits, binary_codes)
        else:
            # CPU implementation
            self._decode_cpu(gray_processed, num_bits, binary_codes)
        
        # Convert binary codes to pixel coordinates
        # This follows SLStudio's approach with optimizations
        return self._binary_to_coords(binary_codes, mask, debug_dir)
    
    def _decode_cpu(self, gray_processed, num_bits, binary_codes):
        """CPU-based Gray code decoding."""
        for i in range(num_bits):
            normal_idx = i * 2
            inverted_idx = i * 2 + 1
            
            # Normal and inverted patterns
            normal = gray_processed[normal_idx]
            inverted = gray_processed[inverted_idx]
            
            # Threshold based on difference (SLStudio approach)
            diff = cv2.absdiff(normal, inverted)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            
            # Determine bit value
            # 1 where normal > inverted, 0 otherwise
            bit_val = (normal > inverted).astype(np.uint8)
            
            # Apply mask - only set bits where difference is significant
            bit_val = bit_val & (thresh > 0)
            
            # Store in binary_codes
            binary_codes[:, :, i] = bit_val
    
    def _binary_to_coords(self, binary_codes, mask, debug_dir=None):
        """
        Convert binary codes to projector coordinates.
        
        This is based on SLStudio's approach but optimized for our use case.
        """
        img_height, img_width = mask.shape
        num_bits = binary_codes.shape[2]
        
        # Convert binary to standard binary (not Gray code yet)
        decoded = np.zeros(mask.shape, dtype=np.int32) - 1  # -1 for invalid pixels
        
        # Use vectorized operations where possible
        y_coords, x_coords = np.where(mask > 0)
        
        for idx in range(len(y_coords)):
            y, x = y_coords[idx], x_coords[idx]
            
            # Extract binary code for this pixel
            binary_code = binary_codes[y, x, :]
            
            # Convert binary to Gray code
            # This approach uses the Gray code conversion formula from SLStudio
            gray_code = 0
            for j in range(num_bits):
                bit_val = binary_code[num_bits - j - 1]
                gray_code = (gray_code << 1) | bit_val
            
            # Convert Gray code to binary using SLStudio's algorithm
            binary = gray_code
            for shift in range(1, num_bits):
                binary ^= (binary >> shift)
            
            # Set decoded value only if within range
            if binary < self.pattern_width * self.pattern_height:
                decoded[y, x] = binary
        
        # Create coordinate map
        coord_map = np.zeros((img_height, img_width, 2), dtype=np.float32)
        
        # For each valid pixel, convert projector coordinate to row, col
        valid_coords = np.where((decoded >= 0) & (mask > 0))
        y_coords, x_coords = valid_coords
        
        for idx in range(len(y_coords)):
            y, x = y_coords[idx], x_coords[idx]
            value = decoded[y, x]
            
            # Convert projector coordinate to row, col
            proj_y = value // self.pattern_width
            proj_x = value % self.pattern_width
            
            if 0 <= proj_x < self.pattern_width and 0 <= proj_y < self.pattern_height:
                coord_map[y, x, 0] = proj_y  # Row (V)
                coord_map[y, x, 1] = proj_x  # Column (U)
        
        # Save debug visualization if requested
        if debug_dir:
            # Create visualization of projected coordinates
            coord_viz = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            for y in range(img_height):
                for x in range(img_width):
                    if mask[y, x] > 0:
                        proj_y = int(coord_map[y, x, 0])
                        proj_x = int(coord_map[y, x, 1])
                        
                        if proj_y >= 0 and proj_x >= 0:
                            r = min(255, (proj_y * 255) // self.pattern_height)
                            g = min(255, (proj_x * 255) // self.pattern_width)
                            coord_viz[y, x] = [r, g, 255]
            
            cv2.imwrite(os.path.join(debug_dir, "proj_coords_viz.png"), coord_viz)
        
        return coord_map, mask


class StereoTriangulator:
    """
    Robust stereo triangulation for structured light scanning.
    
    This class implements a reliable triangulation algorithm based on
    techniques from SLStudio, with optimizations for our use case.
    """
    
    def __init__(self, calibration_data: Dict[str, Any]):
        """
        Initialize triangulator with calibration data.
        
        Args:
            calibration_data: Stereo calibration parameters
        """
        self.calibration = calibration_data
        
        # Extract calibration parameters
        self.camera_matrix_left = np.array(self.calibration["camera_matrix_left"])
        self.dist_coeffs_left = np.array(self.calibration["dist_coeffs_left"])
        self.camera_matrix_right = np.array(self.calibration["camera_matrix_right"])
        self.dist_coeffs_right = np.array(self.calibration["dist_coeffs_right"])
        self.R = np.array(self.calibration["R"])
        self.T = np.array(self.calibration["T"])
        self.image_size = self.calibration.get("image_size", (1280, 720))
        
        # Precompute stereo rectification
        self._compute_rectification()
        
        # Following SLStudio, precompute determinant tensor for faster triangulation
        # This is skipped for now as it's complex and we'll rely on OpenCV initially
    
    def _compute_rectification(self):
        """Compute stereo rectification parameters."""
        # Compute stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Store matrices
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.roi1 = roi1
        self.roi2 = roi2
        
        # Compute undistortion maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, R1, P1, self.image_size, cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, R2, P2, self.image_size, cv2.CV_32FC1
        )
    
    def rectify_images(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Rectify stereo image pairs.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            
        Returns:
            Tuple of (rectified_left_images, rectified_right_images)
        """
        rect_left = []
        rect_right = []
        
        for left_img, right_img in zip(left_images, right_images):
            # Rectify left image
            left_rect = cv2.remap(left_img, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
            # Rectify right image
            right_rect = cv2.remap(right_img, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
            
            rect_left.append(left_rect)
            rect_right.append(right_rect)
        
        return rect_left, rect_right
    
    def find_correspondences(
        self,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        epipolar_tolerance: float = 2.0,  # Tighter epipolar constraint
        debug_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find stereo correspondences between left and right cameras.
        
        Args:
            left_coords: Left camera-projector correspondences
            right_coords: Right camera-projector correspondences
            left_mask: Left camera shadow mask
            right_mask: Right camera shadow mask
            epipolar_tolerance: Max distance from epipolar line
            debug_dir: Optional directory to save debug images
            
        Returns:
            Tuple of (left_points, right_points) as Nx2 arrays
        """
        height, width = left_mask.shape
        
        # Create mapping from projector to right camera
        # This is similar to our approach but with better validation
        proj_to_right = {}
        valid_right_count = 0
        
        # Build projection map (following SLStudio approach)
        for y in range(height):
            for x in range(width):
                if right_mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Get projector coordinates
                proj_v = int(right_coords[y, x, 0])
                proj_u = int(right_coords[y, x, 1])
                
                # Stricter check for valid projector coordinates
                if proj_v < 0 or proj_u < 0 or proj_v >= height or proj_u >= width:
                    continue  # Skip invalid coordinates
                
                valid_right_count += 1
                
                # Store coordinates
                key = (proj_v, proj_u)
                if key not in proj_to_right:
                    proj_to_right[key] = []
                proj_to_right[key].append((x, y))
        
        logger.info(f"Valid projector coordinates in right image: {valid_right_count}")
        logger.info(f"Unique projector coordinates: {len(proj_to_right)}")
        
        # Find correspondences
        left_points = []
        right_points = []
        valid_left_count = 0
        match_count = 0
        
        for y in range(height):
            for x in range(width):
                if left_mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Get projector coordinates
                proj_v = int(left_coords[y, x, 0])
                proj_u = int(left_coords[y, x, 1])
                
                # Stricter check for valid projector coordinates
                if proj_v < 0 or proj_u < 0 or proj_v >= height or proj_u >= width:
                    continue  # Skip invalid coordinates
                
                valid_left_count += 1
                
                # Find matches in right image
                key = (proj_v, proj_u)
                if key in proj_to_right:
                    # Find best match based on epipolar constraint
                    # This is the key innovation from SLStudio - very tight epipolar constraints
                    best_match = None
                    min_y_diff = float('inf')
                    
                    for rx, ry in proj_to_right[key]:
                        y_diff = abs(y - ry)
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_match = (rx, ry)
                    
                    # Only use if close to epipolar line (stricter constraint)
                    if min_y_diff <= epipolar_tolerance:
                        match_count += 1
                        right_x, right_y = best_match
                        left_points.append([x, y])
                        right_points.append([right_x, right_y])
        
        # Debug output
        logger.info(f"Valid projector coordinates in left image: {valid_left_count}")
        logger.info(f"Matched points: {match_count}")
        
        # Always save debug images if requested
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create coordinate visualizations
            coord_viz_left = np.zeros((height, width, 3), dtype=np.uint8)
            coord_viz_right = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Color pixels based on their projector coordinates
            for y in range(height):
                for x in range(width):
                    if left_mask[y, x] > 0:
                        proj_v = int(left_coords[y, x, 0])
                        proj_u = int(left_coords[y, x, 1])
                        if 0 <= proj_v < height and 0 <= proj_u < width:
                            r = min(255, (proj_v * 255) // height)
                            g = min(255, (proj_u * 255) // width)
                            coord_viz_left[y, x] = [r, g, 255]
                    
                    if right_mask[y, x] > 0:
                        proj_v = int(right_coords[y, x, 0])
                        proj_u = int(right_coords[y, x, 1])
                        if 0 <= proj_v < height and 0 <= proj_u < width:
                            r = min(255, (proj_v * 255) // height)
                            g = min(255, (proj_u * 255) // width)
                            coord_viz_right[y, x] = [r, g, 255]
            
            # Save visualizations
            cv2.imwrite(os.path.join(debug_dir, "left_coords_viz.png"), coord_viz_left)
            cv2.imwrite(os.path.join(debug_dir, "right_coords_viz.png"), coord_viz_right)
            
            # If we found correspondences, visualize them too
            if len(left_points) > 0:
                # Create a combined visualization
                stereo_viz = np.zeros((height, width*2, 3), dtype=np.uint8)
                stereo_viz[:, :width] = coord_viz_left
                stereo_viz[:, width:] = coord_viz_right
                
                # Draw lines between corresponding points
                for i in range(min(100, len(left_points))):  # Limit to 100 for clarity
                    pt1 = (int(left_points[i][0]), int(left_points[i][1]))
                    pt2 = (int(right_points[i][0]) + width, int(right_points[i][1]))
                    cv2.line(stereo_viz, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(stereo_viz, pt1, 3, (255, 0, 0), -1)
                    cv2.circle(stereo_viz, pt2, 3, (0, 0, 255), -1)
                
                cv2.imwrite(os.path.join(debug_dir, "stereo_correspondences.png"), stereo_viz)
        
        return np.array(left_points), np.array(right_points)
    
    def triangulate_points(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray
    ) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences.
        
        Args:
            left_points: Left image points (Nx2)
            right_points: Right image points (Nx2)
            
        Returns:
            3D points (Nx3)
        """
        # Check if we have any points to triangulate
        if len(left_points) == 0 or len(right_points) == 0:
            logger.warning("No points to triangulate, returning empty point cloud")
            return np.array([])
        
        # Reshape points for triangulation
        left_pts = left_points.reshape(-1, 1, 2).astype(np.float32)
        right_pts = right_points.reshape(-1, 1, 2).astype(np.float32)
        
        # OpenCV triangulatePoints expects points to be 2xN, not Nx2
        left_pts_2xn = np.transpose(left_pts, (2, 1, 0)).reshape(2, -1)
        right_pts_2xn = np.transpose(right_pts, (2, 1, 0)).reshape(2, -1)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(self.P1, self.P2, left_pts_2xn, right_pts_2xn)
        points_4d = points_4d.T
        
        # Convert to 3D points
        points_3d = points_4d[:, :3] / points_4d[:, 3:4]
        
        return points_3d
    
    def filter_point_cloud(
        self,
        points_3d: np.ndarray,
        max_distance: float = 1000.0
    ) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Filter and clean 3D point cloud.
        
        Args:
            points_3d: 3D points (Nx3)
            max_distance: Maximum distance from origin
            
        Returns:
            Filtered point cloud
        """
        # Check if points_3d is empty
        if points_3d.size == 0:
            logger.warning("Empty point cloud, nothing to filter")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Check dimensions before applying axis operations
        if points_3d.ndim > 1:
            # For multi-dimensional arrays (normal case)
            mask = ~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)
        else:
            # For 1D arrays
            mask = ~np.isnan(points_3d) & ~np.isinf(points_3d)
        
        clean_pts = points_3d[mask]
        
        if len(clean_pts) == 0:
            logger.warning("No valid points after removing NaN and infinite values")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Filter by distance
        if clean_pts.ndim > 1:
            dist = np.linalg.norm(clean_pts, axis=1)
        else:
            # For 1D array, just use its absolute value as distance
            dist = np.abs(clean_pts)
        
        mask = dist < max_distance
        clean_pts = clean_pts[mask]
        
        if len(clean_pts) == 0:
            logger.warning(f"No valid points after distance filtering (max_dist={max_distance})")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Use Open3D for advanced filtering if available
        if OPEN3D_AVAILABLE and len(clean_pts) > 20:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clean_pts)
            
            # Remove outliers (tighter filtering from SLStudio)
            if len(pcd.points) > 50:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=20,  # More neighbors for better outlier detection
                    std_ratio=2.0     # Standard outlier threshold
                )
            
            return pcd
        
        # If Open3D not available, just return filtered points
        return clean_pts


class RobustStereoScanner:
    """
    Robust structured light scanner for 3D scanning.
    
    This class combines techniques from SLStudio and our existing implementation,
    focusing on stable stereo correspondence and accurate triangulation.
    """
    
    def __init__(
        self,
        client,
        calibration_file: Optional[str] = None,
        pattern_resolution: Tuple[int, int] = (1024, 768),
        num_gray_codes: int = 8,
        use_gpu: bool = False,
        debug_mode: bool = False
    ):
        """
        Initialize robust scanner.
        
        Args:
            client: Unlook client instance
            calibration_file: Path to stereo calibration file
            pattern_resolution: Resolution of projection patterns
            num_gray_codes: Number of Gray code bits
            use_gpu: Whether to use GPU acceleration
            debug_mode: Whether to enable debug mode with additional output
        """
        self.client = client
        self.calibration_file = calibration_file
        self.pattern_resolution = pattern_resolution
        self.num_gray_codes = num_gray_codes
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.debug_mode = debug_mode
        
        # State variables
        self.current_point_cloud = None
        
        # Create debug directory
        if self.debug_mode:
            self.debug_dir = os.path.join(os.getcwd(), "robust_scanner_debug")
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug mode enabled, outputs saved to: {self.debug_dir}")
        else:
            self.debug_dir = None
        
        # Load calibration data
        self.calibration_data = self._load_calibration(calibration_file)
        
        # Initialize pattern set
        self.pattern_set = RobustPatternSet(
            width=pattern_resolution[0],
            height=pattern_resolution[1],
            num_gray_codes=num_gray_codes,
            use_gpu=use_gpu
        )
        
        # Initialize decoder
        self.decoder = GrayCodeDecoder(
            pattern_width=pattern_resolution[0],
            pattern_height=pattern_resolution[1],
            use_gpu=use_gpu
        )
        
        # Initialize triangulator
        self.triangulator = StereoTriangulator(self.calibration_data)
        
        logger.info(f"Initialized RobustStereoScanner with {num_gray_codes} Gray code bits")
    
    def _load_calibration(self, calibration_file: Optional[str]) -> Dict[str, Any]:
        """
        Load stereo calibration data from file.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            Calibration data dictionary
        """
        if not calibration_file:
            logger.info("No calibration file provided, using default calibration")
            return self._create_default_calibration()
        
        if not os.path.exists(calibration_file):
            logger.warning(f"Calibration file not found: {calibration_file}")
            logger.info("Using default calibration instead")
            return self._create_default_calibration()
        
        try:
            # Try JSON format first
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
                
                # Check required parameters
                required_keys = ["camera_matrix_left", "dist_coeffs_left",
                              "camera_matrix_right", "dist_coeffs_right",
                              "R", "T"]
                
                if all(key in calib_data for key in required_keys):
                    logger.info(f"Loaded calibration from {calibration_file}")
                    return calib_data
                else:
                    logger.warning("Calibration file is missing required parameters")
        except:
            logger.warning(f"Failed to parse JSON calibration file: {calibration_file}")
        
        try:
            # Try numpy format
            calib_data = dict(np.load(calibration_file, allow_pickle=True))
            
            # Check required parameters (with alternate names)
            if all(key in calib_data for key in ["M1", "d1", "M2", "d2", "R", "T"]):
                # Rename keys to standardized format
                return {
                    "camera_matrix_left": calib_data["M1"],
                    "dist_coeffs_left": calib_data["d1"],
                    "camera_matrix_right": calib_data["M2"],
                    "dist_coeffs_right": calib_data["d2"],
                    "R": calib_data["R"],
                    "T": calib_data["T"]
                }
            else:
                logger.warning("NumPy calibration file is missing required parameters")
        except Exception as e:
            logger.warning(f"Failed to load NumPy calibration file: {e}")
        
        logger.warning("Using default calibration as fallback")
        return self._create_default_calibration()
    
    def _create_default_calibration(self) -> Dict[str, Any]:
        """Create default stereo calibration parameters."""
        # Use projector resolution for image size
        width, height = self.pattern_resolution
        
        # Standard intrinsic camera matrix
        fx = width * 0.8
        fy = height * 0.8
        cx = width / 2
        cy = height / 2
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # No distortion by default
        dist_coeffs = np.zeros(5)
        
        # Identity rotation between cameras
        R = np.eye(3)
        
        # 100mm baseline along X
        T = np.array([100.0, 0.0, 0.0]).reshape(3, 1)
        
        return {
            "camera_matrix_left": camera_matrix,
            "dist_coeffs_left": dist_coeffs,
            "camera_matrix_right": camera_matrix.copy(),
            "dist_coeffs_right": dist_coeffs.copy(),
            "R": R,
            "T": T,
            "image_size": (width, height)
        }
    
    def scan(self, output_dir: Optional[str] = None) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Perform a 3D scan with structured light.
        
        Args:
            output_dir: Optional directory to save output files
            
        Returns:
            Reconstructed point cloud
        """
        # Create output directory if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        debug_dir = output_dir or self.debug_dir
        
        # Project patterns and capture images
        left_images, right_images = self._project_and_capture(debug_dir)
        
        if not left_images or not right_images:
            logger.error("Failed to capture images")
            return None
        
        # Process images to reconstruct point cloud
        point_cloud = self._process_scan(left_images, right_images, debug_dir)
        
        # Store result
        self.current_point_cloud = point_cloud
        
        # Save point cloud if requested
        if output_dir and OPEN3D_AVAILABLE and isinstance(point_cloud, o3d.geometry.PointCloud):
            output_path = os.path.join(output_dir, "point_cloud.ply")
            o3d.io.write_point_cloud(output_path, point_cloud)
            logger.info(f"Point cloud saved to {output_path}")
        
        return point_cloud
    
    def _project_and_capture(
        self,
        debug_dir: Optional[str] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Project patterns and capture stereo images.
        
        Args:
            debug_dir: Optional directory to save debug images
            
        Returns:
            Tuple of (left_images, right_images)
        """
        # Get camera IDs
        cameras = self.client.camera.get_cameras()
        if not cameras or len(cameras) < 2:
            logger.error("Need at least 2 cameras for stereo scanning")
            return [], []
        
        left_camera_id = cameras[0]["id"]
        right_camera_id = cameras[1]["id"]
        
        # Get patterns
        patterns = self.pattern_set.get_patterns()
        
        # Prepare for capture
        left_images = []
        right_images = []
        
        # Project patterns and capture images
        logger.info(f"Projecting {len(patterns)} patterns and capturing images...")
        
        for i, pattern in enumerate(patterns):
            pattern_name = pattern.get("name", f"pattern_{i}")
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern_name}")
            
            # Use the appropriate projector method based on pattern type
            pattern_type = pattern.get("pattern_type", "")
            
            success = False
            if pattern_type == "solid_field":
                # Use direct solid field method
                success = self.client.projector.show_solid_field(pattern.get("color", "White"))
            elif pattern_type == "raw_image" and "image" in pattern:
                # For raw images with binary data
                if hasattr(self.client.projector, "show_raw_image"):
                    success = self.client.projector.show_raw_image(pattern["image"])
                else:
                    logger.warning(f"Projector doesn't support raw images, skipping pattern: {pattern_name}")
                    continue
            else:
                logger.warning(f"Unsupported pattern type: {pattern_type}")
                continue
            
            if not success:
                logger.warning(f"Failed to project pattern {pattern_name}")
                continue
            
            # Wait for projector to update (SLStudio uses hardware trigger, we use software delay)
            time.sleep(0.2)  # Slightly longer delay for reliability
            
            # Capture stereo pair
            try:
                left_img = self.client.camera.capture(left_camera_id)
                right_img = self.client.camera.capture(right_camera_id)
                
                # Save debug images if requested
                if debug_dir:
                    pattern_dir = os.path.join(debug_dir, "capture")
                    os.makedirs(pattern_dir, exist_ok=True)
                    
                    if left_img is not None:
                        cv2.imwrite(os.path.join(pattern_dir, f"left_{i:02d}.png"), left_img)
                    if right_img is not None:
                        cv2.imwrite(os.path.join(pattern_dir, f"right_{i:02d}.png"), right_img)
                
                # Append to image lists
                left_images.append(left_img)
                right_images.append(right_img)
                
            except Exception as e:
                logger.error(f"Error capturing images: {e}")
                continue
        
        # Reset projector to black field
        self.client.projector.show_solid_field("Black")
        
        return left_images, right_images
    
    def _process_scan(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        debug_dir: Optional[str] = None
    ) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Process captured images to reconstruct point cloud.
        
        Args:
            left_images: Left camera images
            right_images: Right camera images
            debug_dir: Optional directory to save debug outputs
            
        Returns:
            Reconstructed point cloud
        """
        # Create debug subdirectories
        if debug_dir:
            decode_dir = os.path.join(debug_dir, "decode")
            os.makedirs(decode_dir, exist_ok=True)
            
            stereo_dir = os.path.join(debug_dir, "stereo")
            os.makedirs(stereo_dir, exist_ok=True)
        else:
            decode_dir = None
            stereo_dir = None
        
        # Rectify images (SLStudio approach)
        rect_left, rect_right = self.triangulator.rectify_images(left_images, right_images)
        
        # Get white and black reference images
        white_left = rect_left[0]  # First image is white
        black_left = rect_left[1]  # Second image is black
        white_right = rect_right[0]
        black_right = rect_right[1]
        
        # Gray code images start from index 2
        # The pattern set has alternating normal and inverted pairs
        gray_left = rect_left[2:]
        gray_right = rect_right[2:]
        
        # Decode Gray code patterns
        # SLStudio approach focuses on robust thresholding
        left_coords, left_mask = self.decoder.decode(
            gray_left, white_left, black_left, 
            orientation="vertical",  # Determines coordinate direction
            debug_dir=decode_dir
        )
        
        right_coords, right_mask = self.decoder.decode(
            gray_right, white_right, black_right,
            orientation="vertical",
            debug_dir=decode_dir
        )
        
        logger.info(f"Decoded projector coordinates - Left mask: {np.count_nonzero(left_mask)}, "
                  f"Right mask: {np.count_nonzero(right_mask)}")
        
        # Find correspondences
        # SLStudio approach uses very tight epipolar constraints
        left_points, right_points = self.triangulator.find_correspondences(
            left_coords, right_coords, left_mask, right_mask,
            debug_dir=stereo_dir
        )
        
        logger.info(f"Found {len(left_points)} stereo correspondences")
        
        if len(left_points) == 0:
            logger.warning("No stereo correspondences found, returning empty point cloud")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Triangulate points
        points_3d = self.triangulator.triangulate_points(left_points, right_points)
        
        # Filter point cloud
        point_cloud = self.triangulator.filter_point_cloud(points_3d)
        
        # Log results
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            logger.info(f"Reconstructed point cloud with {len(point_cloud.points)} points")
        else:
            logger.info(f"Reconstructed point cloud with {len(point_cloud)} points")
        
        return point_cloud
    
    def get_point_cloud(self) -> Union[o3dg.PointCloud, np.ndarray]:
        """Get the latest reconstructed point cloud."""
        return self.current_point_cloud

# Helper function to create a scanner instance
def create_robust_scanner(
    client,
    calibration_file: Optional[str] = None,
    pattern_resolution: Tuple[int, int] = (1024, 768),
    num_gray_codes: int = 8,
    use_gpu: bool = False,
    debug_mode: bool = False
) -> RobustStereoScanner:
    """
    Create a robust stereo scanner instance.
    
    Args:
        client: Unlook client instance
        calibration_file: Path to stereo calibration file
        pattern_resolution: Resolution of projection patterns
        num_gray_codes: Number of Gray code bits
        use_gpu: Whether to use GPU acceleration
        debug_mode: Whether to enable debug mode with additional output
        
    Returns:
        Configured RobustStereoScanner instance
    """
    return RobustStereoScanner(
        client=client,
        calibration_file=calibration_file,
        pattern_resolution=pattern_resolution,
        num_gray_codes=num_gray_codes,
        use_gpu=use_gpu,
        debug_mode=debug_mode
    )