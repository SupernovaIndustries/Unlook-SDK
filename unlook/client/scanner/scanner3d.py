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

# Import local modules
from .scan_config import ScanConfig
from .scan_result import ScanResult
from .pattern_decoder import PatternDecoder
from .point_cloud_processor import PointCloudProcessor

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
        # Prepare images list with white and black references at the beginning
        all_images = [white_ref, black_ref] + images
        
        # Use PatternDecoder for actual decoding
        x_coords, y_coords, mask = PatternDecoder.decode_gray_code(
            all_images, width, height, threshold=5.0,
            debug_dir=self.debug_dir if hasattr(self, 'debug_dir') else None
        )
        
        # Combine x and y coordinates into single array for compatibility
        decoded_coords = np.stack([x_coords, y_coords], axis=-1)
        
        return decoded_coords, mask
    
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
            
        # Use the unified triangulation implementation
        from unlook.client.scanning.reconstruction.triangulator import Triangulator
        
        # The baseline will be extracted from calibration data
        # No need to specify baseline_mm parameter
        
        # Set depth limits from configuration if available
        max_depth = getattr(self.config, 'max_depth', 5000.0) if hasattr(self, 'config') else 5000.0
        min_depth = getattr(self.config, 'min_depth', 10.0) if hasattr(self, 'config') else 10.0
        
        # Get debug directory if available
        debug_dir = self.debug_dir if hasattr(self, 'debug_dir') else None
        
        # Create triangulator instance
        triangulator = Triangulator(
            rectification_params,
            max_depth=max_depth,
            min_depth=min_depth,
            enable_gpu=False  # GPU not implemented yet
        )
        
        # Triangulate points using the unified implementation
        result = triangulator.triangulate(left_points, right_points)
        
        # Extract just the 3D points for backward compatibility
        # In the future, we could also use result.uncertainties for ISO compliance
        points_3d = result.points_3d
        
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
        # Delegate to PointCloudProcessor
        return PointCloudProcessor.filter_point_cloud(
            points_3d,
            max_distance=max_distance,
            voxel_size=voxel_size,
            remove_outliers=True,
            outlier_std_ratio=self.config.outlier_std
        )
    
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
        # Delegate to PointCloudProcessor
        return PointCloudProcessor.create_mesh(
            pcd,
            method="poisson",
            depth=depth,
            remove_degenerate=True,
            remove_duplicated=True
        )
    
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