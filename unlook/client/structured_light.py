"""
Advanced structured light scanning module for UnLook SDK.
Provides utilities for structured light pattern generation and decoding.
Enhanced with OpenCV's structured light module and stereo calibration capabilities.
"""

import os
import numpy as np
import cv2
import logging
import json
import glob
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union

# Configure logger first
logger = logging.getLogger(__name__)

# Optional dependencies - make them truly optional
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    logger.warning("h5py not installed. HDF5 file format support will be limited.")
    H5PY_AVAILABLE = False

# Check for OpenCV contrib modules (structured_light)
try:
    if not hasattr(cv2, 'structured_light'):
        logger.warning("OpenCV structured_light module not found. This is included in opencv-contrib-python.")
        logger.warning("To enable structured light functionality, install opencv-contrib-python.")
        # Try to check if we can import it directly (sometimes available this way)
        try:
            from cv2 import structured_light
            cv2.structured_light = structured_light
            OPENCV_STRUCTURED_LIGHT_AVAILABLE = True
        except ImportError:
            OPENCV_STRUCTURED_LIGHT_AVAILABLE = False
    else:
        # Check if we can create a GrayCodePattern
        try:
            test_pattern = cv2.structured_light.GrayCodePattern_create(10, 10)
            # Check which API is available by testing the methods
            # Some OpenCV versions use getWhiteImage, others use getWhitePatternImage
            try:
                test_pattern.getWhiteImage()
                OPENCV_STRUCTURED_LIGHT_API_VERSION = "old"
            except AttributeError:
                try:
                    test_pattern.getWhitePatternImage()
                    OPENCV_STRUCTURED_LIGHT_API_VERSION = "new"
                except AttributeError:
                    logger.warning("Unrecognized OpenCV structured_light API version")
                    OPENCV_STRUCTURED_LIGHT_API_VERSION = "unknown"
            
            OPENCV_STRUCTURED_LIGHT_AVAILABLE = True
        except Exception as e:
            logger.warning(f"Error testing OpenCV structured_light: {e}")
            OPENCV_STRUCTURED_LIGHT_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error checking OpenCV structured_light module: {e}")
    OPENCV_STRUCTURED_LIGHT_AVAILABLE = False
    OPENCV_STRUCTURED_LIGHT_API_VERSION = "unknown"

# Set API version to "unknown" if the module is not available
if not 'OPENCV_STRUCTURED_LIGHT_API_VERSION' in locals():
    OPENCV_STRUCTURED_LIGHT_API_VERSION = "unknown"

# Try to import open3d
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

logger = logging.getLogger(__name__)

def numpy_to_open3d_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3dg.PointCloud:
    """
    Convert a numpy array of points to an Open3D point cloud.
    
    Args:
        points: Array of 3D points (N, 3)
        colors: Optional array of RGB colors (N, 3), values in range [0, 1]
        
    Returns:
        Open3D point cloud or a simple wrapper object if Open3D is not available
    """
    # Check if Open3D is available
    if not OPEN3D_AVAILABLE:
        logger.warning("open3d not available. Returning points as a simple wrapper object")
        
        # Create a simple wrapper to maintain API compatibility
        class SimplePointCloud:
            def __init__(self, pts):
                self.points = pts
                self.colors = None
                
            def __len__(self):
                return len(self.points)
        
        # Filter valid points
        valid_mask = (~np.isnan(points).any(axis=1)) & (~np.isinf(points).any(axis=1))
        valid_points = points[valid_mask]
        
        if valid_points.shape[0] == 0:
            logger.warning("No valid points found in point cloud")
            return SimplePointCloud([])
            
        logger.info(f"Creating simple point cloud with {valid_points.shape[0]} points")
        return SimplePointCloud(valid_points)
    
    # Filter out invalid points
    valid_mask = (~np.isnan(points).any(axis=1)) & (~np.isinf(points).any(axis=1))
    valid_points = points[valid_mask]
    
    if valid_points.shape[0] == 0:
        logger.warning("No valid points found in point cloud")
        # Return empty point cloud
        return o3dg.PointCloud()
    
    logger.info(f"Creating point cloud with {valid_points.shape[0]} points")
    
    # Create Open3D point cloud
    pcd = o3dg.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    # Add colors if provided
    if colors is not None and colors.shape[0] >= valid_points.shape[0]:
        valid_colors = colors[valid_mask]
        # Ensure color values are in [0, 1] range
        if valid_colors.max() > 1.0:
            valid_colors = valid_colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    
    return pcd

def filter_point_cloud(pcd: o3dg.PointCloud, nb_neighbors: int = 20, std_ratio: float = 0.5) -> o3dg.PointCloud:
    """
    Filter a point cloud to remove outliers.
    
    Args:
        pcd: Input point cloud
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Filtered point cloud
    """
    # Check if Open3D is available
    if not OPEN3D_AVAILABLE:
        logger.warning("open3d not available. Skipping point cloud filtering")
        return pcd
        
    if len(pcd.points) == 0:
        logger.warning("Empty point cloud, nothing to filter")
        return pcd
        
    # Apply statistical outlier removal filter
    try:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        filtered_pcd = pcd.select_by_index(ind)
        logger.info(f"Filtered point cloud: removed {len(pcd.points) - len(filtered_pcd.points)} outliers")
        return filtered_pcd
    except Exception as e:
        logger.error(f"Error filtering point cloud: {e}")
        return pcd


@dataclass
class CameraParameters:
    """Camera intrinsic and distortion parameters."""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray    # Distortion coefficients
    resolution: Tuple[int, int]  # Width, height
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraParameters':
        """Create CameraParameters from a dictionary."""
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        resolution = tuple(data["resolution"])
        return cls(camera_matrix, dist_coeffs, resolution)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CameraParameters to a dictionary."""
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "resolution": list(self.resolution)
        }


@dataclass
class StereoCameraParameters:
    """Stereo calibration parameters for a pair of cameras."""
    left: CameraParameters
    right: CameraParameters
    R: np.ndarray  # 3x3 rotation matrix
    T: np.ndarray  # 3x1 translation vector
    E: np.ndarray  # Essential matrix
    F: np.ndarray  # Fundamental matrix
    R1: Optional[np.ndarray] = None  # Rectification rotation for left camera
    R2: Optional[np.ndarray] = None  # Rectification rotation for right camera
    P1: Optional[np.ndarray] = None  # Projection matrix for left camera
    P2: Optional[np.ndarray] = None  # Projection matrix for right camera
    Q: Optional[np.ndarray] = None   # Disparity-to-depth mapping matrix
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StereoCameraParameters':
        """Create StereoCameraParameters from a dictionary."""
        left = CameraParameters.from_dict(data["left"])
        right = CameraParameters.from_dict(data["right"])
        R = np.array(data["R"], dtype=np.float64)
        T = np.array(data["T"], dtype=np.float64)
        E = np.array(data["E"], dtype=np.float64)
        F = np.array(data["F"], dtype=np.float64)
        
        # Optional rectification parameters
        R1 = np.array(data.get("R1", []), dtype=np.float64) if "R1" in data else None
        R2 = np.array(data.get("R2", []), dtype=np.float64) if "R2" in data else None
        P1 = np.array(data.get("P1", []), dtype=np.float64) if "P1" in data else None
        P2 = np.array(data.get("P2", []), dtype=np.float64) if "P2" in data else None
        Q = np.array(data.get("Q", []), dtype=np.float64) if "Q" in data else None
        
        return cls(left, right, R, T, E, F, R1, R2, P1, P2, Q)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert StereoCameraParameters to a dictionary."""
        data = {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist()
        }
        
        # Add optional rectification parameters if available
        if self.R1 is not None:
            data["R1"] = self.R1.tolist()
        if self.R2 is not None:
            data["R2"] = self.R2.tolist()
        if self.P1 is not None:
            data["P1"] = self.P1.tolist()
        if self.P2 is not None:
            data["P2"] = self.P2.tolist()
        if self.Q is not None:
            data["Q"] = self.Q.tolist()
            
        return data
    
    def compute_rectification(self, image_size: Optional[Tuple[int, int]] = None) -> None:
        """Compute stereo rectification parameters."""
        if image_size is None:
            # Use the resolution from the first camera
            image_size = self.left.resolution
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left.camera_matrix, self.left.dist_coeffs,
            self.right.camera_matrix, self.right.dist_coeffs,
            image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Store the rectification parameters
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        
        logger.info(f"Computed stereo rectification for image size: {image_size}")
    
    def save(self, filepath: str) -> None:
        """Save stereo parameters to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved stereo parameters to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StereoCameraParameters':
        """Load stereo parameters from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        stereo_params = cls.from_dict(data)
        logger.info(f"Loaded stereo parameters from {filepath}")
        return stereo_params

class GrayCodeGenerator:
    """
    Gray code pattern generator for structured light scanning.
    Implements the methods from the Structured-light-stereo repository.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, white_threshold: int = 5):
        """
        Initialize the Gray code generator.
        
        Args:
            width: Projector width
            height: Projector height
            white_threshold: Threshold for white detection during decoding
        """
        self.proj_w = width
        self.proj_h = height
        self.white_threshold = white_threshold
        self.black_threshold = 40  # Threshold for shadow detection
        
        # Check if OpenCV structured_light module is available
        if not OPENCV_STRUCTURED_LIGHT_AVAILABLE:
            logger.warning("OpenCV structured_light module not available.")
            logger.warning("Using a basic custom Gray code implementation instead.")
            logger.warning("Install opencv-contrib-python for full Gray code functionality.")
            
            # Use our own custom Gray code implementation
            self._use_custom_implementation = True
            self.num_required_imgs = 2 * (int(np.log2(width)) + int(np.log2(height)))
            logger.info(f"Custom Gray code requires {self.num_required_imgs} pattern images")
        else:
            # Create OpenCV Gray code pattern generator
            self._use_custom_implementation = False
            
            # Different OpenCV versions have different API
            try:
                # Newer versions use GrayCodePattern_create
                if hasattr(cv2.structured_light, 'GrayCodePattern_create'):
                    self.graycode = cv2.structured_light.GrayCodePattern_create(width=self.proj_w, height=self.proj_h)
                else:
                    # Older versions use GrayCodePattern.create
                    self.graycode = cv2.structured_light.GrayCodePattern.create(width=self.proj_w, height=self.proj_h)
                
                # Set white threshold
                if hasattr(self.graycode, 'setWhiteThreshold'):
                    self.graycode.setWhiteThreshold(self.white_threshold)
                
                # Store number of required images
                if hasattr(self.graycode, 'getNumberOfPatternImages'):
                    self.num_required_imgs = self.graycode.getNumberOfPatternImages()
                else:
                    # Fallback method for number of patterns
                    self.num_required_imgs = 2 * (int(np.log2(width)) + int(np.log2(height)))
                
                logger.info(f"Gray code requires {self.num_required_imgs} pattern images")
                
                # Check which API version to use for getting patterns
                if OPENCV_STRUCTURED_LIGHT_API_VERSION == "new":
                    logger.info("Using newer OpenCV API for Gray code patterns")
                    self._use_new_api = True
                else:
                    logger.info("Using older OpenCV API for Gray code patterns")
                    self._use_new_api = False
            
            except Exception as e:
                logger.warning(f"Error initializing OpenCV GrayCodePattern: {e}")
                logger.warning("Falling back to custom Gray code implementation")
                self._use_custom_implementation = True
                self.num_required_imgs = 2 * (int(np.log2(width)) + int(np.log2(height)))
                logger.info(f"Custom Gray code requires {self.num_required_imgs} pattern images")
        
    def generate_pattern_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate a complete Gray code pattern sequence.
        
        Returns:
            List of pattern dictionaries for the projector
        """
        patterns = []
        
        if self._use_custom_implementation:
            # Custom implementation - Improved to use actual Gray code
            # Generate white and black calibration images
            white_image = np.ones((self.proj_h, self.proj_w), dtype=np.uint8) * 255
            black_image = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
            
            # Add white and black images
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._process_cv_image(white_image),
                "name": "white"
            })
            
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._process_cv_image(black_image),
                "name": "black"
            })
            
            # Generate horizontal patterns (encoding X coordinates) with true Gray code
            bits_x = int(np.ceil(np.log2(self.proj_w)))
            for bit in range(bits_x):
                # Generate normal gray code pattern
                pattern_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
                
                for x in range(self.proj_w):
                    # Convert binary to Gray code using bitwise XOR
                    gray_x = x ^ (x >> 1)
                    # Check if this bit is set in the Gray code representation
                    if (gray_x >> bit) & 1:
                        pattern_img[:, x] = 255
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._process_cv_image(pattern_img),
                    "name": f"gray_code_x_{bit}"
                })
                
                # Generate inverted pattern (complement)
                inv_pattern_img = 255 - pattern_img
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._process_cv_image(inv_pattern_img),
                    "name": f"gray_code_x_inv_{bit}"
                })
            
            # Generate vertical patterns (encoding Y coordinates) with true Gray code
            bits_y = int(np.ceil(np.log2(self.proj_h)))
            for bit in range(bits_y):
                # Generate normal gray code pattern
                pattern_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
                
                for y in range(self.proj_h):
                    # Convert binary to Gray code using bitwise XOR
                    gray_y = y ^ (y >> 1)
                    # Check if this bit is set in the Gray code representation
                    if (gray_y >> bit) & 1:
                        pattern_img[y, :] = 255
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._process_cv_image(pattern_img),
                    "name": f"gray_code_y_{bit}"
                })
                
                # Generate inverted pattern (complement)
                inv_pattern_img = 255 - pattern_img
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._process_cv_image(inv_pattern_img),
                    "name": f"gray_code_y_inv_{bit}"
                })
                
            # Add additional high-contrast patterns like the reference repo to help with calibration
            # Add a fine checkerboard pattern
            checkerboard_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
            square_size_x = 16
            square_size_y = 16
            for y in range(0, self.proj_h, square_size_y):
                for x in range(0, self.proj_w, square_size_x):
                    if ((y // square_size_y) + (x // square_size_x)) % 2 == 0:
                        y_end = min(y + square_size_y, self.proj_h)
                        x_end = min(x + square_size_x, self.proj_w)
                        checkerboard_img[y:y_end, x:x_end] = 255
                        
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._process_cv_image(checkerboard_img),
                "name": "checkerboard_fine"
            })
        
        else:
            # OpenCV structured_light implementation
            # There are API differences between OpenCV versions
            try:
                # Get black and white images based on API version
                if hasattr(self, '_use_new_api') and self._use_new_api:
                    # Newer API
                    _, white_image = self.graycode.getWhitePatternImage()
                    _, black_image = self.graycode.getBlackPatternImage()
                else:
                    # Older API or fallback
                    try:
                        _, white_image = self.graycode.getWhiteImage()
                        _, black_image = self.graycode.getBlackImage()
                    except AttributeError:
                        # If both APIs fail, create basic white/black images
                        logger.warning("Could not get white/black images from OpenCV API, using basic images")
                        white_image = np.ones((self.proj_h, self.proj_w), dtype=np.uint8) * 255
                        black_image = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
                
                # Convert OpenCV images to patterns
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._process_cv_image(white_image),
                    "name": "white"
                })
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._process_cv_image(black_image),
                    "name": "black"
                })
                
                # Get Gray code patterns
                pattern_cv_images = []
                try:
                    if hasattr(self.graycode, 'getImagesForProjection'):
                        self.graycode.getImagesForProjection(pattern_cv_images)
                    elif hasattr(self.graycode, 'generate'):
                        pattern_cv_images = self.graycode.generate()
                    else:
                        raise AttributeError("No pattern generation method found")
                    
                    # Add each pattern to the sequence
                    for i, pattern_image in enumerate(pattern_cv_images):
                        patterns.append({
                            "pattern_type": "raw_image",
                            "image": self._process_cv_image(pattern_image),
                            "name": f"gray_code_{i}"
                        })
                        
                    # Add additional high-contrast patterns like the reference repo to help with calibration
                    # Add a fine checkerboard pattern
                    checkerboard_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
                    square_size_x = 16
                    square_size_y = 16
                    for y in range(0, self.proj_h, square_size_y):
                        for x in range(0, self.proj_w, square_size_x):
                            if ((y // square_size_y) + (x // square_size_x)) % 2 == 0:
                                y_end = min(y + square_size_y, self.proj_h)
                                x_end = min(x + square_size_x, self.proj_w)
                                checkerboard_img[y:y_end, x:x_end] = 255
                                
                    patterns.append({
                        "pattern_type": "raw_image",
                        "image": self._process_cv_image(checkerboard_img),
                        "name": "checkerboard_fine"
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating OpenCV Gray code patterns: {e}")
                    logger.warning("Falling back to custom pattern generation")
                    # Call the method again with custom implementation
                    self._use_custom_implementation = True
                    return self.generate_pattern_sequence()
            except Exception as e:
                logger.error(f"Error in OpenCV Gray code pattern generation: {e}")
                logger.warning("Falling back to custom pattern generation")
                # Call the method again with custom implementation
                self._use_custom_implementation = True
                return self.generate_pattern_sequence()
            
        logger.info(f"Generated {len(patterns)} Gray code patterns")
        return patterns
    
    def _process_cv_image(self, cv_image: np.ndarray) -> bytes:
        """
        Convert an OpenCV pattern image to a compatible format.
        
        Args:
            cv_image: OpenCV image
            
        Returns:
            JPEG encoded image
        """
        # Ensure image is 8-bit
        if cv_image.dtype != np.uint8:
            cv_image = cv_image.astype(np.uint8)
            
        # Encode to JPEG
        success, jpeg_data = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            return jpeg_data.tobytes()
        else:
            logger.error("Failed to encode pattern image")
            return b''
    
    def _binary_to_gray(self, n):
        """Convert a binary number to Gray code."""
        return n ^ (n >> 1)
    
    def _gray_to_binary(self, n):
        """Convert a Gray code to binary."""
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
            
    def decode_pattern_images(self, pattern_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode a set of captured images to generate projector-camera correspondence.
        
        Args:
            pattern_images: List of captured images (including white, black and Gray code patterns)
            
        Returns:
            Tuple of (cam_proj, shadow_mask)
        """
        if len(pattern_images) < self.num_required_imgs + 2:
            logger.error(f"Not enough pattern images: {len(pattern_images)} (need {self.num_required_imgs + 2})")
            return None, None
            
        # Extract white, black and pattern images
        white_img = pattern_images[0]
        black_img = pattern_images[1]
        gray_patterns = pattern_images[2:]
        
        # Compute shadow mask (areas occluded from projector)
        shadow_mask = self._compute_shadow_mask(black_img, white_img, self.black_threshold)
        
        # Decode Gray code patterns
        img_h, img_w = pattern_images[0].shape[:2]
        cam_proj = np.zeros((img_h, img_w, 2), dtype=np.float32)
        
        if self._use_custom_implementation:
            # Custom implementation
            bits_x = int(np.ceil(np.log2(self.proj_w)))
            bits_y = int(np.ceil(np.log2(self.proj_h)))
            
            # Process each pixel
            for y in range(img_h):
                for x in range(img_w):
                    # Skip shadowed pixels
                    if shadow_mask[y, x] == 0:
                        continue
                    
                    # Decode X coordinate
                    x_code = 0
                    for bit in range(bits_x):
                        # Get regular and inverted patterns
                        pattern_idx = bit * 2
                        normal_val = gray_patterns[pattern_idx][y, x]
                        inverted_val = gray_patterns[pattern_idx + 1][y, x]
                        
                        # Only use the bit if the difference is significant
                        if abs(int(normal_val) - int(inverted_val)) > self.white_threshold:
                            # Set the bit if normal > inverted
                            if normal_val > inverted_val:
                                x_code |= (1 << bit)
                    
                    # Decode Y coordinate
                    y_code = 0
                    for bit in range(bits_y):
                        # Get regular and inverted patterns
                        pattern_idx = 2 * bits_x + bit * 2
                        if pattern_idx < len(gray_patterns) - 1:
                            normal_val = gray_patterns[pattern_idx][y, x]
                            inverted_val = gray_patterns[pattern_idx + 1][y, x]
                            
                            # Only use the bit if the difference is significant
                            if abs(int(normal_val) - int(inverted_val)) > self.white_threshold:
                                # Set the bit if normal > inverted
                                if normal_val > inverted_val:
                                    y_code |= (1 << bit)
                    
                    # Convert Gray code to binary
                    proj_x = self._gray_to_binary(x_code)
                    proj_y = self._gray_to_binary(y_code)
                    
                    # Store projection mapping if within projector bounds
                    if 0 <= proj_x < self.proj_w and 0 <= proj_y < self.proj_h:
                        cam_proj[y, x, 1] = proj_x  # u (column)
                        cam_proj[y, x, 0] = proj_y  # v (row)
                    
        else:
            # OpenCV structured_light implementation
            # Process each pixel
            for y in range(img_h):
                for x in range(img_w):
                    # Skip shadowed pixels
                    if shadow_mask[y, x] == 0:
                        continue
                        
                    # Get projector pixel corresponding to this camera pixel
                    try:
                        if hasattr(self.graycode, 'getProjPixel'):
                            error, proj_pixel = self.graycode.getProjPixel(gray_patterns, x, y)
                        elif hasattr(self.graycode, 'decode'):
                            ret, proj_pixel = self.graycode.decode(gray_patterns, x, y)
                            error = not ret
                        else:
                            # If no method available, fall back to custom implementation
                            logger.warning("No suitable decode method found in GrayCodePattern")
                            self._use_custom_implementation = True
                            return self.decode_pattern_images(pattern_images)
                        
                        if not error:
                            # Store projection mapping (projector v, u)
                            cam_proj[y, x, 0] = proj_pixel[1]  # v (row)
                            cam_proj[y, x, 1] = proj_pixel[0]  # u (column)
                    except Exception as e:
                        logger.warning(f"Error decoding pixel ({x},{y}): {e}")
        
        logger.info(f"Decoded {np.count_nonzero(cam_proj[:,:,0] > 0)} correspondence points")
        return cam_proj, shadow_mask
    
    @staticmethod
    def _compute_shadow_mask(black_img: np.ndarray, white_img: np.ndarray, threshold: int) -> np.ndarray:
        """
        Compute a shadow mask to identify areas that are not illuminated by the projector.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            threshold: Brightness difference threshold
            
        Returns:
            Binary mask where 0 indicates shadowed areas
        """
        # Improved shadow mask calculation
        diff = cv2.absdiff(white_img, black_img)
        
        # Normalize for better visibility
        if np.max(diff) > 0:
            normalized_diff = (diff.astype(float) / np.max(diff) * 255).astype(np.uint8)
        else:
            normalized_diff = diff
        
        # Apply adaptive thresholding for better results
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(normalized_diff, (5, 5), 0)
            
            # Try adaptive threshold first
            shadow_mask = cv2.adaptiveThreshold(blurred, 1, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 21, -2)
        except Exception as e:
            logger.warning(f"Adaptive thresholding failed: {e}, using simple threshold")
            # Fallback to simple thresholding
            shadow_mask = np.zeros_like(black_img)
            shadow_mask[white_img > black_img + threshold] = 1
            
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        return shadow_mask
        
    @staticmethod
    def generate_point_cloud(cam_l_proj: np.ndarray, cam_r_proj: np.ndarray, 
                           P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Generate a point cloud from camera-projector correspondences.
        
        Args:
            cam_l_proj: Left camera-projector correspondence
            cam_r_proj: Right camera-projector correspondence
            P1: Left camera projection matrix
            P2: Right camera projection matrix
            
        Returns:
            3D point cloud
        """
        # Find pixel correspondence between cameras
        pts_l = []
        pts_r = []
        
        # Create projector-to-camera map for right camera
        img_h, img_w = cam_r_proj.shape[:2]
        proj_cam_r = np.zeros((1080, 1920, 2))  # Assuming projector is 1920x1080
        
        # Fill projector-to-camera map
        for y in range(img_h):
            for x in range(img_w):
                if cam_r_proj[y, x, 0] > 0 and cam_r_proj[y, x, 1] > 0:
                    proj_y = int(cam_r_proj[y, x, 0])
                    proj_x = int(cam_r_proj[y, x, 1])
                    
                    if 0 <= proj_y < 1080 and 0 <= proj_x < 1920:
                        proj_cam_r[proj_y, proj_x, 0] = y
                        proj_cam_r[proj_y, proj_x, 1] = x
        
        # Find correspondence points
        for y in range(img_h):
            for x in range(img_w):
                if cam_l_proj[y, x, 0] > 0 and cam_l_proj[y, x, 1] > 0:
                    proj_y = int(cam_l_proj[y, x, 0])
                    proj_x = int(cam_l_proj[y, x, 1])
                    
                    if 0 <= proj_y < 1080 and 0 <= proj_x < 1920:
                        cam_r_y = int(proj_cam_r[proj_y, proj_x, 0])
                        cam_r_x = int(proj_cam_r[proj_y, proj_x, 1])
                        
                        if cam_r_y > 0 and cam_r_x > 0:
                            pts_l.append([x, y])
                            pts_r.append([cam_r_x, cam_r_y])
        
        # Convert to numpy arrays for triangulation
        pts_l = np.array(pts_l)[:, np.newaxis, :]
        pts_r = np.array(pts_r)[:, np.newaxis, :]
        
        # Triangulate points
        pts4D = cv2.triangulatePoints(P1, P2, np.float32(pts_l), np.float32(pts_r)).T
        pts3D = pts4D[:, :3] / pts4D[:, -1:]
        
        return pts3D


class PhaseShiftPatternGenerator:
    """
    Phase shift pattern generator for structured light scanning.
    Implements sinusoidal phase shift patterns at multiple frequencies.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, phases: int = 4, frequencies: List[int] = None):
        """
        Initialize the phase shift pattern generator.
        
        Args:
            width: Projector width
            height: Projector height
            phases: Number of phase shifts per frequency
            frequencies: List of pattern frequencies (periods)
        """
        self.proj_w = width
        self.proj_h = height
        self.phases = phases
        self.frequencies = frequencies or [16, 32, 64, 128]  # Default frequencies
        
    def generate_pattern_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate a complete phase shift pattern sequence.
        
        Returns:
            List of pattern dictionaries for the projector
        """
        patterns = []
        
        # Add white and black reference patterns
        white_img = np.ones((self.proj_h, self.proj_w), dtype=np.uint8) * 255
        black_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(white_img),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(black_img),
            "name": "black"
        })
        
        # Generate horizontal and vertical patterns for each frequency
        directions = ['horizontal', 'vertical']
        for direction in directions:
            for freq in self.frequencies:
                # Generate patterns with different phase shifts
                for phase in range(self.phases):
                    phase_offset = (2 * np.pi * phase) / self.phases
                    
                    # Create the pattern
                    pattern_img = self._generate_phase_pattern(freq, phase_offset, direction)
                    
                    # Add to sequence
                    patterns.append({
                        "pattern_type": "raw_image",
                        "image": self._encode_image(pattern_img),
                        "name": f"{direction}_f{freq}_p{phase}"
                    })
        
        logger.info(f"Generated {len(patterns)} phase shift patterns")
        return patterns
    
    def _generate_phase_pattern(self, frequency: int, phase_offset: float, direction: str) -> np.ndarray:
        """
        Generate a single phase shift pattern.
        
        Args:
            frequency: Pattern frequency (period in pixels)
            phase_offset: Phase shift in radians
            direction: 'horizontal' or 'vertical'
            
        Returns:
            Pattern image
        """
        pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        
        # Create coordinate grid
        x = np.arange(self.proj_w)
        y = np.arange(self.proj_h)
        
        if direction == 'horizontal':
            # Horizontal sinusoidal pattern
            for i in range(self.proj_h):
                # Calculate sinusoidal intensity
                intensity = 0.5 + 0.5 * np.cos((2 * np.pi * x / frequency) + phase_offset)
                pattern[i, :] = (intensity * 255).astype(np.uint8)
        else:
            # Vertical sinusoidal pattern
            for i in range(self.proj_w):
                # Calculate sinusoidal intensity
                intensity = 0.5 + 0.5 * np.cos((2 * np.pi * y / frequency) + phase_offset)
                pattern[:, i] = (intensity * 255).astype(np.uint8)
        
        return pattern
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """
        Encode an image as JPEG.
        
        Args:
            image: Input image
            
        Returns:
            JPEG encoded image
        """
        success, jpeg_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            return jpeg_data.tobytes()
        else:
            logger.error("Failed to encode pattern image")
            return b''
            
            
class StereoStructuredLightScanner:
    """
    Enhanced stereo structured light scanner that uses calibrated stereo cameras for accurate 3D reconstruction.
    This class implements advanced techniques for structured light scanning with better point cloud generation
    based on the Structured-light-stereo repository.
    """
    
    def __init__(self, stereo_params: StereoCameraParameters, 
                 projector_width: int = 1920, projector_height: int = 1080):
        """
        Initialize the stereo structured light scanner.
        
        Args:
            stereo_params: Stereo camera calibration parameters
            projector_width: Width of the projector in pixels
            projector_height: Height of the projector in pixels
        """
        self.stereo_params = stereo_params
        self.projector_width = projector_width
        self.projector_height = projector_height
        
        # Make sure rectification parameters are computed
        if self.stereo_params.P1 is None or self.stereo_params.P2 is None:
            logger.info("Computing stereo rectification parameters")
            self.stereo_params.compute_rectification()
        
        # Initialize Gray code generator for projector-camera correspondence
        self.gray_code = GrayCodeGenerator(
            width=projector_width, 
            height=projector_height
        )
        
        # Create undistortion maps for faster processing
        self._init_undistortion_maps()
        
        logger.info("Initialized StereoStructuredLightScanner with rectified stereo parameters")
        
    def _init_undistortion_maps(self) -> None:
        """Initialize undistortion and rectification maps for faster processing."""
        # Get image size from camera parameters
        left_size = self.stereo_params.left.resolution
        right_size = self.stereo_params.right.resolution
        
        # Create undistortion maps for left camera
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.stereo_params.left.camera_matrix,
            self.stereo_params.left.dist_coeffs,
            self.stereo_params.R1,
            self.stereo_params.P1,
            left_size,
            cv2.CV_32FC1
        )
        
        # Create undistortion maps for right camera
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.stereo_params.right.camera_matrix,
            self.stereo_params.right.dist_coeffs,
            self.stereo_params.R2,
            self.stereo_params.P2,
            right_size,
            cv2.CV_32FC1
        )
        
        logger.info(f"Created undistortion maps for left camera {left_size} and right camera {right_size}")
    
    def generate_scan_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate a complete pattern sequence for structured light scanning.
        
        Returns:
            List of pattern dictionaries for the projector
        """
        return self.gray_code.generate_pattern_sequence()
    
    def rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo images using the calibration parameters.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            
        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        # Ensure images are the right format
        if len(left_img.shape) == 3 and left_img.shape[2] == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            
        if len(right_img.shape) == 3 and right_img.shape[2] == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img
        
        # Apply rectification using precomputed maps
        left_rectified = cv2.remap(left_gray, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_gray, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def process_scan(self, left_images: List[np.ndarray], right_images: List[np.ndarray], 
                     mask_threshold: int = 5) -> o3dg.PointCloud:
        """
        Process structured light scan images and generate a 3D point cloud.
        
        Args:
            left_images: List of left camera images (should match pattern sequence)
            right_images: List of right camera images (should match pattern sequence)
            mask_threshold: Threshold for shadow/valid pixel detection
            
        Returns:
            3D point cloud (filtered)
        """
        logger.info(f"Processing structured light scan with {len(left_images)} left images and {len(right_images)} right images")
        
        if len(left_images) != len(right_images):
            logger.error("Number of left and right images must match")
            return o3dg.PointCloud()
        
        if len(left_images) < 2:  # Need at least white and black images
            logger.error("Not enough images for structured light processing")
            return o3dg.PointCloud()
        
        # Save some debug info for the first image pair to help with troubleshooting
        try:
            # Save shape info
            left_shape = left_images[0].shape
            right_shape = right_images[0].shape
            logger.info(f"Image shapes: left={left_shape}, right={right_shape}")
            
            # Check if images are color or grayscale
            left_is_color = len(left_shape) == 3 and left_shape[2] == 3
            right_is_color = len(right_shape) == 3 and right_shape[2] == 3
            logger.info(f"Image types: left={'color' if left_is_color else 'grayscale'}, " +
                       f"right={'color' if right_is_color else 'grayscale'}")
            
            # Check intensity range
            left_min, left_max = np.min(left_images[0]), np.max(left_images[0])
            right_min, right_max = np.min(right_images[0]), np.max(right_images[0])
            logger.info(f"Intensity ranges: left=[{left_min}, {left_max}], right=[{right_min}, {right_max}]")
        except Exception as e:
            logger.warning(f"Could not analyze input images: {e}")
        
        # Rectify all images - convert to grayscale first if needed
        left_rectified = []
        right_rectified = []
        
        for left_img, right_img in zip(left_images, right_images):
            # Ensure images are grayscale
            if len(left_img.shape) == 3 and left_img.shape[2] == 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            if len(right_img.shape) == 3 and right_img.shape[2] == 3:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                
            # Apply rectification
            l_rect, r_rect = self.rectify_images(left_img, right_img)
            left_rectified.append(l_rect)
            right_rectified.append(r_rect)
        
        logger.info("Rectified all images")
        
        # Enhanced decoding with robust error handling
        try:
            # Decode Gray code patterns and get projector-camera correspondences
            left_cam_proj, left_mask = self.gray_code.decode_pattern_images(left_rectified)
            right_cam_proj, right_mask = self.gray_code.decode_pattern_images(right_rectified)
            
            if left_cam_proj is None or right_cam_proj is None:
                logger.error("Failed to decode Gray code patterns")
                return o3dg.PointCloud()
            
            # Save masks for debugging
            try:
                left_mask_vis = (left_mask * 255).astype(np.uint8)
                right_mask_vis = (right_mask * 255).astype(np.uint8)
                
                # Find a temporary folder to save these masks
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                cv2.imwrite(os.path.join(debug_dir, "left_mask.png"), left_mask_vis)
                cv2.imwrite(os.path.join(debug_dir, "right_mask.png"), right_mask_vis)
                
                logger.info(f"Saved masks to {debug_dir} for debugging")
                logger.info(f"Left mask valid pixels: {np.count_nonzero(left_mask)}")
                logger.info(f"Right mask valid pixels: {np.count_nonzero(right_mask)}")
            except Exception as e:
                logger.warning(f"Could not save debug masks: {e}")
            
            # Combine masks (areas visible from both cameras and the projector)
            # Use a more lenient approach - pixels visible in either camera
            if mask_threshold <= 3:
                # For very low thresholds, use a union instead of intersection
                combined_mask = np.logical_or(left_mask, right_mask).astype(np.uint8)
                logger.info("Using union of masks (lenient mode) since threshold is very low")
            else:
                # Normal mode - intersection of masks
                combined_mask = np.logical_and(left_mask, right_mask).astype(np.uint8)
                logger.info("Using intersection of masks (standard mode)")
            
            logger.info(f"Combined mask valid pixels: {np.count_nonzero(combined_mask)}")
            
            # If the mask has too few valid pixels, try a more lenient threshold
            if np.count_nonzero(combined_mask) < 100 and mask_threshold > 2:
                logger.warning(f"Too few valid pixels ({np.count_nonzero(combined_mask)}). Trying more lenient mask...")
                combined_mask = np.logical_or(left_mask, right_mask).astype(np.uint8)
                logger.info(f"New combined mask valid pixels: {np.count_nonzero(combined_mask)}")
            
            # Generate 3D point cloud
            logger.info("Generating point cloud from decoded patterns")
            points3d = self._triangulate_points(left_cam_proj, right_cam_proj, combined_mask)
            
            # Check if we got any points
            if points3d.size == 0:
                logger.error("No points generated from triangulation")
                return o3dg.PointCloud()
            
            # Create Open3D point cloud
            pcd = numpy_to_open3d_point_cloud(points3d)
            
            # If point cloud is very sparse, skip filtering
            if len(pcd.points) < 50:
                logger.warning(f"Point cloud has only {len(pcd.points)} points, skipping filtering")
                return pcd
            
            # Apply statistical outlier removal with adaptive parameters
            # For sparse clouds, use a higher standard deviation threshold
            std_ratio = 1.0 if len(pcd.points) < 1000 else 0.5
            filtered_pcd = filter_point_cloud(pcd, nb_neighbors=min(30, len(pcd.points) // 2), std_ratio=std_ratio)
            
            logger.info(f"Generated point cloud with {len(filtered_pcd.points)} points after filtering")
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"Error in structured light processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return o3dg.PointCloud()
    
    def _triangulate_points(self, left_cam_proj: np.ndarray, right_cam_proj: np.ndarray, 
                           mask: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from camera-projector correspondences.
        
        Args:
            left_cam_proj: Left camera-projector correspondence map
            right_cam_proj: Right camera-projector correspondence map
            mask: Combined mask of valid pixels
            
        Returns:
            Array of 3D points
        """
        # Get projection matrices from stereo calibration
        P1 = self.stereo_params.P1
        P2 = self.stereo_params.P2
        
        # Find corresponding points between cameras using the projector as an intermediary
        height, width = mask.shape[:2]
        points_left = []
        points_right = []
        colors = []  # For point cloud coloring if we have color images
        
        # More robust correspondence finding with tolerance
        # Projector coordinates can have some error, so use a neighborhood search
        proj_to_right = {}
        
        # First, build a map from projector pixels to right camera pixels with +/- 1 pixel tolerance
        logger.info("Building projector-to-right camera mapping with tolerance...")
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                
                # Get projector coordinates for right camera
                proj_v = int(right_cam_proj[y, x, 0])  # projector row
                proj_u = int(right_cam_proj[y, x, 1])  # projector column
                
                if proj_v > 0 and proj_u > 0 and proj_v < self.projector_height and proj_u < self.projector_width:
                    # Store with 1-pixel tolerance in all directions for better matching
                    for dv in [-1, 0, 1]:
                        for du in [-1, 0, 1]:
                            key = (proj_v + dv, proj_u + du)
                            if key not in proj_to_right:
                                proj_to_right[key] = []
                            proj_to_right[key].append((x, y))
        
        logger.info(f"Built mapping with {len(proj_to_right)} projector coordinates")
        
        # Now find correspondences using this mapping
        logger.info("Finding stereo correspondences...")
        correspondence_count = 0
        skipped_count = 0
        
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                
                # Get projector coordinates for left camera
                proj_v = int(left_cam_proj[y, x, 0])  # projector row
                proj_u = int(left_cam_proj[y, x, 1])  # projector column
                
                if proj_v <= 0 or proj_u <= 0 or proj_v >= self.projector_height or proj_u >= self.projector_width:
                    skipped_count += 1
                    continue
                
                # Look up the right camera coordinates with tolerance
                key = (proj_v, proj_u)
                if key in proj_to_right and proj_to_right[key]:
                    # Multiple matches might exist, take the best one
                    # In this case, we'll use the first match for simplicity
                    # A more sophisticated approach could use a best match criteria
                    right_x, right_y = proj_to_right[key][0]
                    
                    # Add the correspondence
                    points_left.append([x, y])
                    points_right.append([right_x, right_y])
                    correspondence_count += 1
                    
                    # Add color information if available (for future coloring of point cloud)
                    # This assumes left_images[0] is available with RGB data
                    # colors.append(left_images[0][y, x] / 255.0)  # Normalized color
        
        # Convert to numpy arrays
        points_left = np.array(points_left, dtype=np.float32)
        points_right = np.array(points_right, dtype=np.float32)
        
        logger.info(f"Found {len(points_left)} corresponding points between cameras (skipped {skipped_count} invalid points)")
        
        if len(points_left) < 10:
            logger.warning("Too few correspondences for triangulation")
            return np.array([])
        
        # Reshape for triangulation
        points_left = points_left.reshape(-1, 1, 2)
        points_right = points_right.reshape(-1, 1, 2)
        
        # Triangulate points
        logger.info("Triangulating points...")
        try:
            points_4d = cv2.triangulatePoints(P1, P2, points_left, points_right)
            points_4d = points_4d.T
            
            # Convert from homogeneous to 3D coordinates
            points_3d = points_4d[:, :3] / points_4d[:, 3:]
            
            # Filter out points that are too far from the cameras
            dist_from_origin = np.linalg.norm(points_3d, axis=1)
            valid_dist_mask = dist_from_origin < 5000  # Adjust this threshold based on your scene scale
            points_3d = points_3d[valid_dist_mask]
            
            logger.info(f"Triangulated {len(points_3d)} valid 3D points")
            
            return points_3d
            
        except Exception as e:
            logger.error(f"Error during triangulation: {e}")
            return np.array([])
        
    
    def save_point_cloud(self, pcd: o3dg.PointCloud, filepath: str) -> None:
        """
        Save a point cloud to a file.
        
        Args:
            pcd: Point cloud to save
            filepath: Output file path (supported formats: .ply, .pcd, .obj)
        """
        # Extract file extension
        ext = os.path.splitext(filepath)[1].lower()
        
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Saving point cloud in alternative format")
            # Save as NumPy array if open3d is not available
            try:
                if hasattr(pcd, 'points') and not isinstance(pcd.points, list):
                    points = pcd.points
                else:
                    points = np.array([])
                    
                # Choose format based on extension
                if ext == '.h5' and H5PY_AVAILABLE:
                    # Save as HDF5 format
                    with h5py.File(filepath, 'w') as f:
                        f.create_dataset('points', data=points)
                        if hasattr(pcd, 'colors') and pcd.colors is not None:
                            f.create_dataset('colors', data=pcd.colors)
                    logger.info(f"Saved point cloud as HDF5 file to {filepath}")
                else:
                    # Default to NumPy format
                    np_path = filepath
                    if ext not in ['.npy', '.npz']:
                        np_path = filepath + '.npy'
                    np.save(np_path, points)
                    logger.info(f"Saved point cloud as NumPy array to {np_path}")
            except Exception as e:
                logger.error(f"Failed to save point cloud: {e}")
            return
        
        # Use Open3D if available
        if ext in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
            o3d.io.write_point_cloud(filepath, pcd)
            logger.info(f"Saved point cloud with {len(pcd.points)} points to {filepath}")
        else:
            # Default to PLY format if extension not supported
            new_path = os.path.splitext(filepath)[0] + '.ply'
            o3d.io.write_point_cloud(new_path, pcd)
            logger.info(f"Saved point cloud with {len(pcd.points)} points to {new_path} (changed extension to .ply)")
    
    @staticmethod
    def create_mesh_from_point_cloud(pcd: o3dg.PointCloud, 
                                    depth: int = 9, 
                                    smooth_iterations: int = 5) -> o3dg.TriangleMesh:
        """
        Create a triangle mesh from a point cloud using Poisson surface reconstruction.
        
        Args:
            pcd: Input point cloud
            depth: Depth parameter for Poisson reconstruction (higher = more detail)
            smooth_iterations: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Cannot create mesh from point cloud")
            return o3dg.TriangleMesh()
            
        # Estimate normals if not already present
        if not pcd.has_normals():
            logger.info("Estimating normals for point cloud")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Perform Poisson surface reconstruction
        logger.info(f"Performing Poisson reconstruction with depth={depth}")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=True
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Apply smoothing
        if smooth_iterations > 0:
            logger.info(f"Smoothing mesh with {smooth_iterations} iterations")
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        
        logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
        return mesh
    
    @staticmethod
    def save_mesh(mesh: o3dg.TriangleMesh, filepath: str) -> None:
        """
        Save a triangle mesh to a file.
        
        Args:
            mesh: Triangle mesh to save
            filepath: Output file path (supported formats: .ply, .obj, .off, .gltf)
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Cannot save mesh")
            return
            
        o3d.io.write_triangle_mesh(filepath, mesh)
        logger.info(f"Saved mesh with {len(mesh.triangles)} triangles to {filepath}")
    
    @classmethod
    def create_from_calibration_folder(cls, calib_folder: str, 
                                      projector_width: int = 1920, 
                                      projector_height: int = 1080) -> 'StereoStructuredLightScanner':
        """
        Create a scanner from a calibration folder containing stereo parameters.
        
        Args:
            calib_folder: Path to folder containing stereo calibration files
            projector_width: Width of the projector
            projector_height: Height of the projector
            
        Returns:
            Configured StereoStructuredLightScanner
        """
        # Look for stereo calibration file
        calib_files = glob.glob(os.path.join(calib_folder, "*stereo*.json"))
        
        if not calib_files:
            raise FileNotFoundError(f"No stereo calibration file found in {calib_folder}")
        
        # Load the first found calibration file
        stereo_params = StereoCameraParameters.load(calib_files[0])
        
        return cls(stereo_params, projector_width, projector_height)


class StereoCalibrator:
    """
    Stereo camera calibration class for creating calibration parameters for structured light scanning.
    This is a utility class to help generate the necessary calibration files for the StereoStructuredLightScanner.
    """
    
    def __init__(self, board_size: Tuple[int, int] = (10, 7), square_size: float = 20.0):
        """
        Initialize the stereo calibrator.
        
        Args:
            board_size: Dimensions of the checkerboard (width, height) in inner corners
            square_size: Size of the checkerboard squares in millimeters
        """
        self.board_size = board_size
        self.square_size = square_size
        self.obj_points = self._create_object_points()
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        logger.info(f"Initialized StereoCalibrator with {board_size} checkerboard and {square_size}mm squares")
        
    def _create_object_points(self) -> np.ndarray:
        """Create 3D points for the checkerboard corners in world coordinates."""
        # Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(width-1,height-1,0)
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        # Scale by square size (convert to mm)
        objp *= self.square_size
        return objp
    
    def find_chessboard_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Find checkerboard corners in an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (success, corners)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners2
        else:
            return False, None
    
    def calibrate_stereo_cameras(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> StereoCameraParameters:
        """
        Calibrate stereo cameras using a set of checkerboard images.
        
        Args:
            left_images: List of images from the left camera
            right_images: List of images from the right camera
            
        Returns:
            Stereo camera calibration parameters
        """
        if len(left_images) != len(right_images) or len(left_images) < 5:
            raise ValueError(f"Need at least 5 image pairs, got {len(left_images)}")
        
        logger.info(f"Calibrating stereo cameras with {len(left_images)} image pairs")
        
        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world space
        left_img_points = []  # 2D points in left image plane
        right_img_points = []  # 2D points in right image plane
        
        # Process all image pairs
        for left_img, right_img in zip(left_images, right_images):
            # Find corners in left image
            left_found, left_corners = self.find_chessboard_corners(left_img)
            
            # Find corners in right image
            right_found, right_corners = self.find_chessboard_corners(right_img)
            
            # Only use pairs where corners were found in both images
            if left_found and right_found:
                obj_points.append(self.obj_points)
                left_img_points.append(left_corners)
                right_img_points.append(right_corners)
        
        if not obj_points:
            raise ValueError("No valid checkerboard corners found in any image pair")
        
        logger.info(f"Found checkerboard corners in {len(obj_points)} image pairs")
        
        # Get image size from first image
        left_size = left_images[0].shape[:2][::-1]  # width, height
        right_size = right_images[0].shape[:2][::-1]  # width, height
        
        # Calibrate each camera individually
        logger.info("Calibrating left camera...")
        left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
            obj_points, left_img_points, left_size, None, None
        )
        
        logger.info("Calibrating right camera...")
        right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
            obj_points, right_img_points, right_size, None, None
        )
        
        # Stereo calibration
        logger.info("Performing stereo calibration...")
        stereo_flags = cv2.CALIB_FIX_INTRINSIC  # Use intrinsics from individual calibrations
        ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
            obj_points, left_img_points, right_img_points,
            left_mtx, left_dist, right_mtx, right_dist,
            left_size, None, None, None, None, stereo_flags, self.criteria
        )
        
        # Create stereo parameters object
        left_params = CameraParameters(left_mtx, left_dist, left_size)
        right_params = CameraParameters(right_mtx, right_dist, right_size)
        stereo_params = StereoCameraParameters(left_params, right_params, R, T, E, F)
        
        # Compute rectification parameters
        stereo_params.compute_rectification()
        
        logger.info("Stereo calibration completed successfully")
        return stereo_params
    
    @staticmethod
    def draw_chessboard_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected chessboard corners on an image.
        
        Args:
            image: Input image
            corners: Corners detected by find_chessboard_corners
            
        Returns:
            Image with drawn corners
        """
        # Create a color copy of the image if it's grayscale
        if len(image.shape) == 2:
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = image.copy()
            
        # Draw the corners
        cv2.drawChessboardCorners(vis_img, (10, 7), corners, True)
        return vis_img
    
    @staticmethod
    def create_default_stereo_parameters(output_file: str) -> StereoCameraParameters:
        """
        Create a default set of stereo parameters with reasonable values for testing.
        This is useful when you don't have calibration images but want to test the scanner.
        
        Args:
            output_file: Path to save the calibration file
            
        Returns:
            Generated stereo parameters
        """
        # Create default intrinsic matrix for a standard camera
        K1 = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        K2 = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        # No distortion
        dist1 = np.zeros(5)
        dist2 = np.zeros(5)
        
        # Cameras separated by 100mm on the X axis
        # Identity rotation matrix (cameras are aligned)
        R = np.eye(3)
        # Translation 100mm to the right (X axis)
        T = np.array([100.0, 0.0, 0.0]).reshape(3, 1)
        
        # Calculate essential and fundamental matrices
        E = np.cross(T.reshape(3), R, axisb=0).reshape(3, 3)
        F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)
        
        # Create camera parameters
        left_params = CameraParameters(K1, dist1, (1920, 1080))
        right_params = CameraParameters(K2, dist2, (1920, 1080))
        
        # Create stereo parameters
        stereo_params = StereoCameraParameters(left_params, right_params, R, T, E, F)
        
        # Compute rectification parameters
        stereo_params.compute_rectification()
        
        # Save to file if specified
        if output_file:
            stereo_params.save(output_file)
            logger.info(f"Saved default stereo parameters to {output_file}")
        
        return stereo_params


# Utility function to create a calibration and scanning demo
def create_scanning_demo(output_dir: str, use_default_calibration: bool = True) -> StereoStructuredLightScanner:
    """
    Create a structured light scanning demo with default or custom calibration.
    
    Args:
        output_dir: Directory to save calibration and output files
        use_default_calibration: Whether to use default calibration parameters
        
    Returns:
        Configured StereoStructuredLightScanner
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create calibration
    if use_default_calibration:
        logger.info("Using default stereo calibration parameters")
        calib_file = os.path.join(output_dir, "default_stereo_calibration.json")
        stereo_params = StereoCalibrator.create_default_stereo_parameters(calib_file)
    else:
        # In a real application, this would use actual calibration images
        raise NotImplementedError("Custom calibration requires stereo image pairs of a checkerboard pattern")
    
    # Create scanner
    scanner = StereoStructuredLightScanner(stereo_params)
    
    # Generate pattern sequence and save info
    patterns = scanner.generate_scan_patterns()
    with open(os.path.join(output_dir, "scan_patterns.json"), "w") as f:
        pattern_info = {
            "pattern_count": len(patterns),
            "pattern_names": [p.get("name", f"pattern_{i}") for i, p in enumerate(patterns)]
        }
        json.dump(pattern_info, f, indent=2)
    
    logger.info(f"Created scanning demo in {output_dir} with {len(patterns)} patterns")
    return scanner