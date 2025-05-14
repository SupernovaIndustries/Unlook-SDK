"""
3D Scanning Module for UnLook SDK.

This module provides a complete 3D scanning solution using structured light 
techniques, triangulation, and point cloud processing. It is designed to work 
with the UnLook hardware system and client architecture.
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

# Try to import optional dependencies
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
            class TriangleMesh:
                pass
        class utility:
            class Vector3dVector:
                pass
        class visualization:
            pass
        class io:
            pass
    o3d = PlaceholderO3D()
    o3dg = PlaceholderO3D.geometry

# We now use Open3D exclusively for point cloud processing
# No need for python-pcl anymore
PCL_AVAILABLE = False  # Legacy flag kept for backward compatibility

# Import structured light module
# Since these classes might not exist, we'll handle them gracefully
PatternGenerator = None
StructuredLightController = None
generate_patterns = None  
project_patterns = None

# Log the issue for debugging
import logging
logging.getLogger(__name__).warning("Some structured light components may not be available")



class ScanConfig:
    """Configuration for 3D scanning."""
    
    def __init__(self):
        """Initialize scan configuration with default values."""
        # Pattern generation parameters
        self.pattern_resolution = (1024, 768)  # Width x Height
        self.num_gray_codes = 10
        self.num_phase_shifts = 8
        self.phase_shift_frequencies = [1, 8, 16]
        
        # Scanning parameters
        self.pattern_interval = 0.5  # Time between patterns (seconds)
        self.quality = "medium"  # Quality preset: fast, medium, high, ultra
        
        # Processing parameters
        self.max_distance = 1000.0  # Maximum point distance (mm)
        self.voxel_size = 0.5       # Voxel size for downsampling (mm)
        self.outlier_std = 2.0      # Standard deviation for outlier removal
        self.mesh_depth = 9         # Poisson reconstruction depth
        self.mesh_smoothing = 5     # Mesh smoothing iterations
    
    def set_quality_preset(self, quality: str):
        """Set parameters based on quality preset."""
        if quality == "fast":
            self.num_gray_codes = 8
            self.num_phase_shifts = 4
            self.phase_shift_frequencies = [1, 16]
            self.pattern_interval = 0.3
            self.voxel_size = 1.0
            self.mesh_depth = 8
            self.mesh_smoothing = 2
        elif quality == "medium":
            self.num_gray_codes = 10
            self.num_phase_shifts = 6
            self.phase_shift_frequencies = [1, 8, 16]
            self.pattern_interval = 0.5
            self.voxel_size = 0.5
            self.mesh_depth = 9
            self.mesh_smoothing = 3
        elif quality == "high":
            self.num_gray_codes = 10
            self.num_phase_shifts = 8
            self.phase_shift_frequencies = [1, 8, 16, 32]
            self.pattern_interval = 0.75
            self.voxel_size = 0.25
            self.mesh_depth = 10
            self.mesh_smoothing = 4
        elif quality == "ultra":
            self.num_gray_codes = 12
            self.num_phase_shifts = 12
            self.phase_shift_frequencies = [1, 4, 8, 16, 32]
            self.pattern_interval = 1.0
            self.voxel_size = 0.1
            self.mesh_depth = 11
            self.mesh_smoothing = 5
        else:
            logger.warning(f"Unknown quality preset: {quality}, using medium")
            self.set_quality_preset("medium")
        
        self.quality = quality
        logger.info(f"Set quality preset to: {quality}")


class ScanResult:
    """Result of a 3D scan."""
    
    def __init__(
        self,
        point_cloud: Optional[Union[o3dg.PointCloud, np.ndarray]] = None,
        mesh: Optional[o3dg.TriangleMesh] = None,
        images: Optional[Dict[str, List[np.ndarray]]] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize scan result.
        
        Args:
            point_cloud: 3D point cloud (Open3D or numpy array)
            mesh: 3D mesh
            images: Dictionary of captured images
            debug_info: Debug information
        """
        self.point_cloud = point_cloud
        self.mesh = mesh
        self.images = images or {}
        self.debug_info = debug_info or {}
        
        # Calculate statistics
        self.num_points = len(point_cloud.points) if hasattr(point_cloud, "points") else 0
        self.num_triangles = len(mesh.triangles) if mesh and hasattr(mesh, "triangles") else 0
    
    def has_point_cloud(self) -> bool:
        """Check if result contains a point cloud."""
        if self.point_cloud is None:
            return False
        if hasattr(self.point_cloud, "points"):
            return len(self.point_cloud.points) > 0
        return len(self.point_cloud) > 0
    
    def has_mesh(self) -> bool:
        """Check if result contains a mesh."""
        if self.mesh is None:
            return False
        if hasattr(self.mesh, "triangles"):
            return len(self.mesh.triangles) > 0
        return False
    
    def save(self, output_dir: str):
        """Save scan results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        results_dir = os.path.join(output_dir, "results")
        debug_dir = os.path.join(output_dir, "debug")
        captures_dir = os.path.join(output_dir, "captures")
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(captures_dir, exist_ok=True)
        
        # Save point cloud
        if self.has_point_cloud():
            if OPEN3D_AVAILABLE and isinstance(self.point_cloud, o3d.geometry.PointCloud):
                pc_path = os.path.join(results_dir, "point_cloud.ply")
                o3d.io.write_point_cloud(pc_path, self.point_cloud)
            elif isinstance(self.point_cloud, np.ndarray):
                pc_path = os.path.join(results_dir, "point_cloud.npy")
                np.save(pc_path, self.point_cloud)
            logger.info(f"Saved point cloud to: {os.path.basename(pc_path)}")
        
        # Save mesh
        if self.has_mesh() and OPEN3D_AVAILABLE:
            mesh_path = os.path.join(results_dir, "mesh.ply")
            o3d.io.write_triangle_mesh(mesh_path, self.mesh)
            logger.info(f"Saved mesh to: {os.path.basename(mesh_path)}")
        
        # Only save images if debug flag is enabled
        if os.environ.get("UNLOOK_SAVE_DEBUG_IMAGES", "0") == "1":
            logger.debug("Saving captured images...")
            for camera_name, image_list in self.images.items():
                for i, img in enumerate(image_list):
                    img_path = os.path.join(captures_dir, f"{camera_name}_{i:03d}.png")
                    cv2.imwrite(img_path, img)
        
        # Save debug info
        if self.debug_info:
            debug_path = os.path.join(debug_dir, "debug_info.json")
            with open(debug_path, "w") as f:
                clean_debug = {}
                # Convert numpy types to Python native types for JSON serialization
                for k, v in self.debug_info.items():
                    if isinstance(v, (np.int32, np.int64, np.uint64)):
                        clean_debug[k] = int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        clean_debug[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_debug[k] = v.tolist()
                    else:
                        try:
                            json.dumps({k: v})  # Test if serializable
                            clean_debug[k] = v
                        except TypeError:
                            clean_debug[k] = str(v)
                json.dump(clean_debug, f, indent=2)
            logger.info(f"Saved debug info to: {os.path.basename(debug_path)}")


class Scanner3D:
    """
    3D Scanner using structured light techniques.
    
    This scanner uses a combination of Gray code and Phase shift structured light
    techniques to robustly reconstruct 3D point clouds from stereo image pairs.
    """
    
    def __init__(
        self,
        client,
        config: Optional[ScanConfig] = None,
        calibration_file: Optional[str] = None
    ):
        """
        Initialize the 3D scanner.
        
        Args:
            client: UnlookClient instance
            config: Scanner configuration
            calibration_file: Path to stereo calibration file
        """
        self.client = client
        self.config = config or ScanConfig()
        self.calibration_file = calibration_file
        
        # Load calibration data if provided
        self.calibration_data = self._load_calibration(calibration_file)
        
        # Create structured light controller
        self.controller = StructuredLightController(
            projector_client=client.projector,
            camera_client=client.camera,
            width=self.config.pattern_resolution[0],
            height=self.config.pattern_resolution[1],
            pattern_interval=self.config.pattern_interval
        )
        
        logger.info(f"Initialized Scanner3D with {self.config.quality} quality preset")
    
    def _load_calibration(self, calibration_file: Optional[str]) -> Dict[str, Any]:
        """
        Load calibration data from file.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            Dictionary with calibration parameters
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
        except json.JSONDecodeError:
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
        """
        Create default stereo calibration parameters.
        
        Returns:
            Dictionary with calibration parameters
        """
        # Use projector resolution for image size
        width, height = self.config.pattern_resolution
        
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
    
    def compute_rectification(
        self,
        camera_matrix_left: np.ndarray,
        dist_coeffs_left: np.ndarray,
        camera_matrix_right: np.ndarray,
        dist_coeffs_right: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Compute stereo rectification parameters.
        
        Args:
            camera_matrix_left: Left camera matrix
            dist_coeffs_left: Left distortion coefficients
            camera_matrix_right: Right camera matrix
            dist_coeffs_right: Right distortion coefficients
            R: Rotation matrix between cameras
            T: Translation vector between cameras
            image_size: Image size (width, height)
            
        Returns:
            Dictionary with rectification parameters
        """
        # Compute stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Compute undistortion maps
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_32FC1
        )
        
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_32FC1
        )
        
        return {
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "roi1": roi1, "roi2": roi2,
            "map_left_x": map_left_x, "map_left_y": map_left_y,
            "map_right_x": map_right_x, "map_right_y": map_right_y
        }
    
    def rectify_images(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        rectification_params: Dict[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Rectify stereo image pairs.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            rectification_params: Rectification parameters
            
        Returns:
            Tuple of (rectified_left_images, rectified_right_images)
        """
        rect_left = []
        rect_right = []
        
        map_left_x = rectification_params["map_left_x"]
        map_left_y = rectification_params["map_left_y"]
        map_right_x = rectification_params["map_right_x"]
        map_right_y = rectification_params["map_right_y"]
        
        for left_img, right_img in zip(left_images, right_images):
            # Rectify left image
            left_rect = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
            # Rectify right image
            right_rect = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
            
            rect_left.append(left_rect)
            rect_right.append(right_rect)
        
        return rect_left, rect_right
    
    def decode_gray_code(
        self,
        images: List[np.ndarray],
        white_ref: np.ndarray,
        black_ref: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns from images.

        Args:
            images: List of captured Gray code images
            white_ref: White reference image
            black_ref: Black reference image
            width: Projector width
            height: Projector height

        Returns:
            Tuple of (decoded_coords, mask)
        """
        # Create Gray code generator in OpenCV
        gray_code = cv2.structured_light.GrayCodePattern.create(width=width, height=height)

        # Compute shadow mask
        diff = cv2.absdiff(white_ref, black_ref)
        _, mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Convert images to grayscale if needed
        gray_images = []
        for img in images:
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_images.append(gray_img)
            else:
                gray_images.append(img)

        # Prepare arrays for OpenCV decoding
        pattern_images = np.array(gray_images)

        # Set defaults in case decoding fails
        ret = False
        decoded = None

        # Try to decode with proper error handling
        # Direct approach using low-level OpenCV functions for better compatibility
        try:
            # First, determine number of patterns (must be even with half for normal and half for inverted)
            num_patterns = len(pattern_images)
            if num_patterns % 2 != 0:
                logger.warning(f"Odd number of Gray code patterns: {num_patterns}, expected even number")
                # Truncate to even number if needed
                pattern_images = pattern_images[:num_patterns-1]
                num_patterns = len(pattern_images)

            # Number of bits is half the patterns (each bit has normal + inverted pattern)
            num_bits = num_patterns // 2
            logger.info(f"Decoding {num_bits} bits from {num_patterns} Gray code patterns")

            # Create binary representations
            binary_codes = np.zeros((mask.shape[0], mask.shape[1], num_bits), dtype=np.uint8)

            # For each bit, compute difference between normal and inverted pattern
            for i in range(num_bits):
                normal_idx = i * 2
                inverted_idx = i * 2 + 1

                # Normal and inverted patterns
                normal = pattern_images[normal_idx]
                inverted = pattern_images[inverted_idx]

                # Threshold based on difference
                diff = cv2.absdiff(normal, inverted)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                # Determine bit value
                # 1 where normal > inverted, 0 otherwise
                bit_val = (normal > inverted).astype(np.uint8)

                # Apply mask - only set bits where difference is significant
                bit_val = bit_val & (thresh > 0)

                # Store in binary_codes
                binary_codes[:, :, i] = bit_val

            # Convert binary codes to pixel coordinates
            decoded = np.zeros(mask.shape, dtype=np.int32) - 1  # -1 for invalid pixels

            # For each valid pixel
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y, x] > 0:
                        # Extract binary code for this pixel
                        binary_code = binary_codes[y, x, :]

                        # Convert binary to gray code
                        value = 0
                        for j in range(num_bits):
                            bit_val = binary_code[num_bits - j - 1]
                            value = (value << 1) | bit_val

                        # Set decoded value
                        if value < width * height:
                            decoded[y, x] = value

            # Success
            ret = True

        except Exception as e:
            logger.warning(f"Custom Gray code decoding failed: {e}, trying OpenCV's decoder...")
            try:
                # Fallback to OpenCV's decoder with various argument patterns

                # Method 1: Basic decode (OpenCV 3.x style)
                try:
                    result = gray_code.decode(pattern_images)
                    if isinstance(result, tuple) and len(result) >= 2:
                        ret, decoded = result[:2]
                    else:
                        # Direct result
                        decoded = result
                        ret = True
                except Exception as e1:
                    logger.warning(f"OpenCV basic decode failed: {e1}")

                    # Method 2: With reference images (OpenCV 4.x style)
                    try:
                        result = gray_code.decode(
                            pattern_images,
                            np.zeros_like(white_ref),  # Disparity map (not used)
                            black_ref,
                            white_ref
                        )

                        if isinstance(result, tuple) and len(result) >= 2:
                            ret, decoded = result[:2]
                        else:
                            decoded = result
                            ret = True
                    except Exception as e2:
                        logger.error(f"All decode methods failed: {e1}, {e2}")
                        ret = False
                        decoded = None
            except Exception as final_e:
                logger.error(f"All Gray code decoding approaches failed: {final_e}")
                ret = False
                decoded = None
        
        # Check decoding success
        if not ret or decoded is None:
            logger.error("Gray code decoding failed")
            # Return empty arrays with correct shape
            img_height, img_width = images[0].shape[:2]
            empty_coords = np.zeros((img_height, img_width, 2), dtype=np.float32)
            empty_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            return empty_coords, empty_mask
        
        # Create coordinate map
        img_height, img_width = mask.shape
        coord_map = np.zeros((img_height, img_width, 2), dtype=np.float32)
        
        # Extract coordinates
        for y in range(img_height):
            for x in range(img_width):
                if mask[y, x] != 0:
                    try:
                        # Try to get decoded value with error handling
                        if decoded is not None and decoded[y, x] != -1:
                            # Convert projector coordinate to row, col
                            proj_y = decoded[y, x] // width
                            proj_x = decoded[y, x] % width
                            
                            if 0 <= proj_x < width and 0 <= proj_y < height:
                                coord_map[y, x, 0] = proj_y  # Row (V)
                                coord_map[y, x, 1] = proj_x  # Column (U)
                    except Exception as e:
                        # If error for this pixel, just skip it
                        logger.debug(f"Error processing decoded pixel ({x},{y}): {e}")
                        continue
        
        return coord_map, mask
    
    def find_correspondences(
        self,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        epipolar_tolerance: float = 5.0  # Increased tolerance for better coverage
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find corresponding points between stereo images with improved robustness.

        Args:
            left_coords: Left camera-projector correspondences
            right_coords: Right camera-projector correspondences
            left_mask: Left camera shadow mask
            right_mask: Right camera shadow mask
            epipolar_tolerance: Max distance from epipolar line

        Returns:
            Tuple of (left_points, right_points) as Nx2 arrays
        """
        height, width = left_mask.shape

        # Create mapping from projector to right camera with confidence scores
        proj_to_right = {}

        # Create a more lenient mask by dilating the original mask
        kernel = np.ones((3, 3), np.uint8)
        right_mask_dilated = cv2.dilate(right_mask, kernel, iterations=1)
        left_mask_dilated = cv2.dilate(left_mask, kernel, iterations=1)

        # Build projection map with confidence scores
        for y in range(height):
            for x in range(width):
                if right_mask_dilated[y, x] == 0:
                    continue  # Skip shadowed pixels

                # Get projector coordinates with boundary checking
                try:
                    proj_v = int(right_coords[y, x, 0])
                    proj_u = int(right_coords[y, x, 1])

                    # More lenient validity check
                    if proj_v < 0 or proj_u < 0:
                        continue  # Skip invalid coords

                    # Calculate confidence based on mask value and coordinates
                    confidence = 1.0
                    if right_mask[y, x] > 0:  # Higher confidence if in original mask
                        confidence += 1.0

                    # Store coordinates with confidence
                    key = (proj_v, proj_u)
                    if key not in proj_to_right:
                        proj_to_right[key] = []
                    proj_to_right[key].append((x, y, confidence))
                except Exception as e:
                    # Skip pixels that cause errors
                    continue

        logger.info(f"Built projection map with {len(proj_to_right)} unique projector coordinates")

        # Find correspondences
        left_points = []
        right_points = []

        # Track coordinates for better statistics
        used_projector_coords = set()

        for y in range(height):
            for x in range(width):
                if left_mask_dilated[y, x] == 0:
                    continue  # Skip shadowed pixels

                try:
                    # Get projector coordinates
                    proj_v = int(left_coords[y, x, 0])
                    proj_u = int(left_coords[y, x, 1])

                    # More lenient validity check
                    if proj_v < 0 or proj_u < 0:
                        continue  # Skip invalid coords

                    # Find matches in right image
                    key = (proj_v, proj_u)
                    if key in proj_to_right:
                        # Find best match based on epipolar constraint and confidence
                        best_match = None
                        best_score = float('-inf')

                        for rx, ry, confidence in proj_to_right[key]:
                            y_diff = abs(y - ry)
                            if y_diff <= epipolar_tolerance:
                                # Score combines epipolar alignment and confidence
                                score = confidence - y_diff / epipolar_tolerance
                                if score > best_score:
                                    best_score = score
                                    best_match = (rx, ry)

                        # Use the best match if found
                        if best_match:
                            right_x, right_y = best_match
                            left_points.append([x, y])
                            right_points.append([right_x, right_y])
                            used_projector_coords.add(key)
                except Exception as e:
                    # Skip pixels that cause errors
                    continue

        # Convert to numpy arrays
        left_points_arr = np.array(left_points)
        right_points_arr = np.array(right_points)

        # Log detailed statistics
        logger.info(f"Found {len(left_points)} correspondences from {len(used_projector_coords)} unique projector points")
        if len(left_points) > 0:
            logger.info(f"Correspondence coverage: {len(left_points) / (np.sum(left_mask) * np.sum(right_mask) / 255**2) * 100:.2f}%")

        return left_points_arr, right_points_arr
    
    def triangulate_points(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        rectification_params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences.
        
        This method uses the robust triangulation implementation from direct_triangulator.py
        with adaptive scaling and baseline correction to ensure accurate real-world measurements.
        
        Key features:
        - Automatic baseline correction to ensure proper real-world scale
        - Adaptive scaling detection for proper depth units
        - Comprehensive error handling and outlier filtering
        - Fallback mechanisms for error cases
        
        Args:
            left_points: Left image points (Nx2)
            right_points: Right image points (Nx2)
            rectification_params: Rectification parameters with projection matrices
                
        Returns:
            3D points (Nx3) in millimeter scale
        """
        # Check if we have any points to triangulate
        if len(left_points) == 0 or len(right_points) == 0:
            logger.warning("No points to triangulate, returning empty point cloud")
            return np.array([])
            
        # Use the centralized triangulation implementation
        from .scanning.reconstruction.direct_triangulator import triangulate_with_baseline_correction
        
        # The baseline will be extracted from calibration data
        # No need to specify baseline_mm parameter
        
        # Set depth limits from configuration if available
        max_depth = getattr(self.config, 'max_depth', 5000.0) if hasattr(self, 'config') else 5000.0
        min_depth = getattr(self.config, 'min_depth', 10.0) if hasattr(self, 'config') else 10.0
        
        # Get debug directory if available
        debug_dir = self.debug_dir if hasattr(self, 'debug_dir') else None
        
        # Triangulate points using the robust implementation
        points_3d = triangulate_with_baseline_correction(
            left_points, right_points, 
            rectification_params,
            max_depth=max_depth,
            min_depth=min_depth,
            debug_dir=debug_dir
        )
        
        return points_3d
    
    def filter_point_cloud(
        self,
        points_3d: np.ndarray,
        max_distance: float = 1000.0,
        voxel_size: float = 0.5
    ) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Filter and clean 3D point cloud.
        
        Args:
            points_3d: 3D points (Nx3)
            max_distance: Maximum distance from origin
            voxel_size: Voxel size for downsampling
            
        Returns:
            Filtered point cloud
        """
        # Remove invalid points
        mask = ~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)
        clean_pts = points_3d[mask]
        
        if len(clean_pts) == 0:
            logger.warning("No valid points after removing NaN and infinite values")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Filter by distance
        dist = np.linalg.norm(clean_pts, axis=1)
        mask = dist < max_distance
        clean_pts = clean_pts[mask]
        
        if len(clean_pts) == 0:
            logger.warning(f"No valid points after distance filtering (max_dist={max_distance})")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Use Open3D for advanced filtering if available
        if OPEN3D_AVAILABLE and len(clean_pts) > 50:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clean_pts)
            
            # Remove statistical outliers
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=self.config.outlier_std
            )
            
            # Voxel downsample
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            
            return pcd
        
        # If Open3D not available, just return filtered points
        return clean_pts
    
    def create_mesh(
        self,
        pcd: o3dg.PointCloud,
        depth: int = 9,
        smoothing: int = 5
    ) -> Optional[o3dg.TriangleMesh]:
        """
        Create a 3D mesh from point cloud.
        
        Args:
            pcd: Point cloud
            depth: Poisson reconstruction depth
            smoothing: Number of smoothing iterations
            
        Returns:
            Triangle mesh or None if failed
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available, cannot create mesh")
            return None
        
        if not pcd or not hasattr(pcd, 'points') or len(pcd.points) < 10:
            logger.warning("Not enough points to create mesh")
            return None
        
        try:
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
            )
            
            # Orient normals
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
            
            # Create mesh with Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Apply smoothing
            if smoothing > 0:
                logger.info(f"Smoothing mesh with {smoothing} iterations")
                mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing)
            
            logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
            return mesh
        except Exception as e:
            logger.error(f"Failed to create mesh: {e}")
            return None
    
    def scan(
        self,
        output_dir: Optional[str] = None,
        generate_mesh: bool = False,
        visualize: bool = False
    ) -> ScanResult:
        """
        Perform a complete 3D scan using structured light.
        
        Args:
            output_dir: Directory to save scan results
            generate_mesh: Whether to generate a 3D mesh
            visualize: Whether to visualize results
            
        Returns:
            ScanResult object with point cloud, mesh, and debug info
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            captures_dir = os.path.join(output_dir, "captures")
            debug_dir = os.path.join(output_dir, "debug")
            results_dir = os.path.join(output_dir, "results")
            
            os.makedirs(captures_dir, exist_ok=True)
            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        
        # Generate patterns
        logger.info("Generating structured light patterns...")

        # For robust scanning, focus more on Gray code patterns and less on phase shift
        # This makes scanning more reliable when hardware can't display full grayscale
        if self.config.quality in ['high', 'ultra']:
            # For high quality, use both Gray code and phase shift
            patterns = generate_patterns(
                width=self.config.pattern_resolution[0],
                height=self.config.pattern_resolution[1],
                num_gray_codes=self.config.num_gray_codes,
                num_phase_shifts=self.config.num_phase_shifts,
                phase_shift_frequencies=self.config.phase_shift_frequencies
            )
        else:
            # For medium and fast quality, focus more on Gray code
            # We'll use fewer phase shift frequencies and steps for better reliability
            frequencies = self.config.phase_shift_frequencies[:1]  # Just use the lowest frequency
            patterns = generate_patterns(
                width=self.config.pattern_resolution[0],
                height=self.config.pattern_resolution[1],
                num_gray_codes=self.config.num_gray_codes,
                num_phase_shifts=4,  # Reduce phase shift steps
                phase_shift_frequencies=frequencies
            )
        logger.info(f"Generated {len(patterns)} structured light patterns")
        
        # Project patterns and capture images
        logger.info("Projecting patterns and capturing images...")
        try:
            left_images, right_images = self.controller.project_and_capture(
                patterns=patterns,
                output_dir=output_dir
            )
            logger.info(f"Captured {len(left_images)} image pairs")
        except Exception as e:
            logger.error(f"Error during pattern projection and capture: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={"error": str(e), "stage": "capture"}
            )
            return result
        
        # Check if we have enough images
        if len(left_images) < 2 or len(right_images) < 2:
            logger.error(f"Not enough images captured: {len(left_images)} left, {len(right_images)} right")
            # Create empty result with error info
            result = ScanResult(
                debug_info={"error": "Not enough images captured", "stage": "capture"}
            )
            return result
        
        # Get calibration and rectification parameters
        logger.info("Computing rectification parameters...")
        try:
            # Extract calibration parameters
            camera_matrix_left = np.array(self.calibration_data["camera_matrix_left"])
            dist_coeffs_left = np.array(self.calibration_data["dist_coeffs_left"])
            camera_matrix_right = np.array(self.calibration_data["camera_matrix_right"])
            dist_coeffs_right = np.array(self.calibration_data["dist_coeffs_right"])
            R = np.array(self.calibration_data["R"])
            T = np.array(self.calibration_data["T"])
            image_size = self.calibration_data.get("image_size", (1280, 720))
            
            # Compute rectification
            rectification_params = self.compute_rectification(
                camera_matrix_left, dist_coeffs_left,
                camera_matrix_right, dist_coeffs_right,
                R, T, image_size
            )
            logger.info("Rectification parameters computed")
        except Exception as e:
            logger.error(f"Error computing rectification parameters: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={"error": str(e), "stage": "rectification"}
            )
            return result
        
        # Rectify images
        logger.info("Rectifying stereo images...")
        try:
            rect_left, rect_right = self.rectify_images(
                left_images, right_images, rectification_params
            )
            logger.info("Images rectified")
            
            # Only save rectified debug images if explicitly enabled
            if output_dir and os.environ.get("UNLOOK_SAVE_DEBUG_IMAGES", "0") == "1":
                logger.debug("Saving rectified debug images...")
                for i, (left, right) in enumerate(zip(rect_left, rect_right)):
                    left_path = os.path.join(debug_dir, f"rect_left_{i:03d}.png")
                    right_path = os.path.join(debug_dir, f"rect_right_{i:03d}.png")
                    cv2.imwrite(left_path, left)
                    cv2.imwrite(right_path, right)
        except Exception as e:
            logger.error(f"Error rectifying images: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={"error": str(e), "stage": "rectification"},
                images={"left": left_images, "right": right_images}
            )
            return result
        
        # Get white and black reference images
        white_left = rect_left[0]  # First image is white
        black_left = rect_left[1]  # Second image is black
        white_right = rect_right[0]
        black_right = rect_right[1]
        
        # Gray code images start from index 2
        gray_left = rect_left[2:2+self.config.num_gray_codes*4]  # 4 patterns per bit (H/V, normal/inverse)
        gray_right = rect_right[2:2+self.config.num_gray_codes*4]
        
        # Decode Gray code patterns
        logger.info("Decoding Gray code patterns...")
        try:
            left_coords, left_mask = self.decode_gray_code(
                gray_left, white_left, black_left,
                self.config.pattern_resolution[0], self.config.pattern_resolution[1]
            )
            
            right_coords, right_mask = self.decode_gray_code(
                gray_right, white_right, black_right,
                self.config.pattern_resolution[0], self.config.pattern_resolution[1]
            )
            
            # Save masks for debugging
            if output_dir and os.environ.get("UNLOOK_SAVE_DEBUG_IMAGES", "0") == "1":
                cv2.imwrite(os.path.join(debug_dir, "left_mask.png"), left_mask * 255)
                cv2.imwrite(os.path.join(debug_dir, "right_mask.png"), right_mask * 255)
            
            # Debug info
            left_valid = np.sum(left_mask)
            right_valid = np.sum(right_mask)
            logger.info(f"Valid pixels: {left_valid} left, {right_valid} right")
        except Exception as e:
            logger.error(f"Error decoding Gray code patterns: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={"error": str(e), "stage": "decoding"},
                images={"left": left_images, "right": right_images}
            )
            return result
        
        # Find stereo correspondences
        logger.info("Finding stereo correspondences...")
        try:
            left_points, right_points = self.find_correspondences(
                left_coords, right_coords, left_mask, right_mask,
                epipolar_tolerance=3.0
            )
            
            if len(left_points) < 10:
                logger.error(f"Too few correspondences: {len(left_points)}")
                # Create empty result with error info
                result = ScanResult(
                    debug_info={
                        "error": "Too few correspondences",
                        "stage": "correspondences",
                        "num_correspondences": len(left_points),
                        "left_valid": int(np.sum(left_mask)),
                        "right_valid": int(np.sum(right_mask))
                    },
                    images={"left": left_images, "right": right_images}
                )
                return result
            
            logger.info(f"Found {len(left_points)} stereo correspondences")
        except Exception as e:
            logger.error(f"Error finding stereo correspondences: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={"error": str(e), "stage": "correspondences"},
                images={"left": left_images, "right": right_images}
            )
            return result
        
        # Triangulate 3D points
        logger.info("Triangulating 3D points...")
        try:
            points_3d = self.triangulate_points(
                left_points, right_points, rectification_params
            )
            logger.info(f"Triangulated {len(points_3d)} 3D points")
        except Exception as e:
            logger.error(f"Error triangulating points: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={
                    "error": str(e),
                    "stage": "triangulation",
                    "num_correspondences": len(left_points)
                },
                images={"left": left_images, "right": right_images}
            )
            return result
        
        # Filter and process point cloud
        logger.info("Filtering point cloud...")
        try:
            point_cloud = self.filter_point_cloud(
                points_3d,
                max_distance=self.config.max_distance,
                voxel_size=self.config.voxel_size
            )
            
            if isinstance(point_cloud, o3dg.PointCloud):
                num_points = len(point_cloud.points)
            else:
                num_points = len(point_cloud)
            
            logger.info(f"Point cloud filtered: {num_points} points after filtering")
        except Exception as e:
            logger.error(f"Error filtering point cloud: {e}")
            # Create empty result with error info
            result = ScanResult(
                debug_info={
                    "error": str(e),
                    "stage": "filtering",
                    "num_points": len(points_3d)
                },
                images={"left": left_images, "right": right_images}
            )
            return result
        
        # Generate mesh if requested
        mesh = None
        if generate_mesh and OPEN3D_AVAILABLE and isinstance(point_cloud, o3dg.PointCloud) and len(point_cloud.points) > 0:
            logger.info("Generating 3D mesh...")
            try:
                mesh = self.create_mesh(
                    point_cloud,
                    depth=self.config.mesh_depth,
                    smoothing=self.config.mesh_smoothing
                )
                if mesh:
                    logger.info(f"Mesh generated with {len(mesh.triangles)} triangles")
                else:
                    logger.warning("Mesh generation failed")
            except Exception as e:
                logger.error(f"Error creating mesh: {e}")
        
        # Visualize results if requested
        if visualize and OPEN3D_AVAILABLE and isinstance(point_cloud, o3dg.PointCloud) and len(point_cloud.points) > 0:
            logger.info("Visualizing results...")
            try:
                if mesh:
                    o3d.visualization.draw_geometries([point_cloud, mesh])
                else:
                    o3d.visualization.draw_geometries([point_cloud])
            except Exception as e:
                logger.error(f"Error visualizing results: {e}")
        
        # Create and save result
        debug_info = {
            "num_patterns": len(patterns),
            "num_captured_images": len(left_images),
            "left_valid_pixels": int(np.sum(left_mask)) if 'left_mask' in locals() else 0,
            "right_valid_pixels": int(np.sum(right_mask)) if 'right_mask' in locals() else 0,
            "num_correspondences": len(left_points) if 'left_points' in locals() else 0,
            "num_triangulated_points": len(points_3d) if 'points_3d' in locals() else 0,
            "num_filtered_points": num_points if 'num_points' in locals() else 0,
            "calibration_file": self.calibration_file,
            "quality_preset": self.config.quality,
            "max_distance": self.config.max_distance,
            "voxel_size": self.config.voxel_size
        }
        
        result = ScanResult(
            point_cloud=point_cloud,
            mesh=mesh,
            images={"left": left_images, "right": right_images},
            debug_info=debug_info
        )
        
        # Save result if output directory is specified
        if output_dir:
            result.save(output_dir)
        
        return result


def create_scanner(
    client,
    quality: str = "high",
    calibration_file: Optional[str] = None
) -> Scanner3D:
    """
    Create a 3D scanner with specified quality preset.
    
    Args:
        client: UnlookClient instance
        quality: Quality preset (fast, medium, high, ultra)
        calibration_file: Path to stereo calibration file
    
    Returns:
        Configured Scanner3D instance
    """
    config = ScanConfig()
    config.set_quality_preset(quality)
    
    scanner = Scanner3D(
        client=client,
        config=config,
        calibration_file=calibration_file
    )
    
    return scanner