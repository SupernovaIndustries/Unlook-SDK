"""
Pattern decoding utilities for structured light 3D scanning.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class PatternDecoder:
    """Handles decoding of structured light patterns."""
    
    @staticmethod
    def decode_gray_code(
        images: List[np.ndarray],
        pattern_width: int,
        pattern_height: int,
        threshold: float = 5.0,
        debug_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to get projector coordinates.
        
        Args:
            images: List of captured images with Gray code patterns
            pattern_width: Width of projected patterns
            pattern_height: Height of projected patterns
            threshold: Threshold for binary decoding
            debug_dir: Directory to save debug images
            
        Returns:
            Tuple of (x_coords, y_coords, mask) where mask indicates valid pixels
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images (white and black reference)")
        
        # Convert all images to grayscale first to avoid broadcasting errors
        gray_images = []
        for img in images:
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img
            gray_images.append(gray_img)
        
        h, w = gray_images[0].shape
        
        # Get white and black reference images (now guaranteed to be 2D)
        white_img = gray_images[0].astype(np.float32)
        black_img = gray_images[1].astype(np.float32)
        
        # Calculate threshold image
        thresh_img = (white_img - black_img) * 0.5
        
        # Create mask for valid pixels
        mask = (white_img - black_img) > threshold
        
        # Initialize coordinate maps
        x_coords = np.zeros((h, w), dtype=np.float32)
        y_coords = np.zeros((h, w), dtype=np.float32)
        
        # Decode horizontal patterns (for X coordinates)
        num_x_bits = int(np.log2(pattern_width))
        x_bits = []
        
        for i in range(num_x_bits):
            if 2 + i * 2 + 1 >= len(images):
                logger.warning(f"Not enough images for {num_x_bits} horizontal bits")
                break
                
            # Get positive and inverted patterns (now guaranteed to be 2D)
            pos_img = gray_images[2 + i * 2].astype(np.float32)
            inv_img = gray_images[2 + i * 2 + 1].astype(np.float32)
            
            # Decode bit
            bit_value = ((pos_img - black_img) > (inv_img - black_img)).astype(np.uint8)
            x_bits.append(bit_value)
        
        # Convert Gray code to binary
        if x_bits:
            x_binary = x_bits[0].copy()
            for i in range(1, len(x_bits)):
                x_binary = x_binary ^ x_bits[i]
                x_coords += x_binary * (pattern_width / (2 ** (i + 1)))
            x_coords += x_bits[-1] * (pattern_width / (2 ** len(x_bits)))
        
        # Decode vertical patterns (for Y coordinates)
        num_y_bits = int(np.log2(pattern_height))
        y_bits = []
        
        start_idx = 2 + num_x_bits * 2
        for i in range(num_y_bits):
            if start_idx + i * 2 + 1 >= len(images):
                logger.warning(f"Not enough images for {num_y_bits} vertical bits")
                break
                
            # Get positive and inverted patterns (now guaranteed to be 2D)
            pos_img = gray_images[start_idx + i * 2].astype(np.float32)
            inv_img = gray_images[start_idx + i * 2 + 1].astype(np.float32)
            
            # Decode bit
            bit_value = ((pos_img - black_img) > (inv_img - black_img)).astype(np.uint8)
            y_bits.append(bit_value)
        
        # Convert Gray code to binary
        if y_bits:
            y_binary = y_bits[0].copy()
            for i in range(1, len(y_bits)):
                y_binary = y_binary ^ y_bits[i]
                y_coords += y_binary * (pattern_height / (2 ** (i + 1)))
            y_coords += y_bits[-1] * (pattern_height / (2 ** len(y_bits)))
        
        # Apply mask to coordinates
        x_coords[~mask] = -1
        y_coords[~mask] = -1
        
        # Save debug images if requested
        if debug_dir:
            PatternDecoder._save_debug_images(
                x_coords, y_coords, mask, thresh_img, debug_dir
            )
        
        return x_coords, y_coords, mask
    
    @staticmethod
    def decode_phase_shift(
        images: List[np.ndarray],
        num_shifts: int = 4,
        threshold: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode phase shift patterns.
        
        Args:
            images: List of phase-shifted sinusoidal patterns
            num_shifts: Number of phase shifts
            threshold: Minimum modulation threshold
            
        Returns:
            Tuple of (phase_map, modulation_map)
        """
        if len(images) < num_shifts:
            raise ValueError(f"Need at least {num_shifts} images for phase shift decoding")
        
        h, w = images[0].shape[:2]
        
        # Initialize arrays
        numerator = np.zeros((h, w), dtype=np.float32)
        denominator = np.zeros((h, w), dtype=np.float32)
        
        # Calculate phase using N-step phase shifting algorithm
        for i in range(num_shifts):
            phase = 2 * np.pi * i / num_shifts
            img = images[i].astype(np.float32)
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            numerator += img * np.sin(phase)
            denominator += img * np.cos(phase)
        
        # Calculate phase map
        phase_map = np.arctan2(numerator, denominator)
        
        # Calculate modulation (quality metric)
        modulation = np.sqrt(numerator**2 + denominator**2) * 2 / num_shifts
        
        # Create mask based on modulation threshold
        mask = modulation > threshold
        phase_map[~mask] = -np.pi
        
        return phase_map, modulation
    
    @staticmethod
    def unwrap_phase(
        phase_map: np.ndarray,
        gray_code_x: Optional[np.ndarray] = None,
        gray_code_y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Unwrap phase map using Gray code as guide.
        
        Args:
            phase_map: Wrapped phase map (-pi to pi)
            gray_code_x: X coordinates from Gray code (optional)
            gray_code_y: Y coordinates from Gray code (optional)
            
        Returns:
            Unwrapped phase map
        """
        h, w = phase_map.shape
        unwrapped = np.zeros_like(phase_map)
        
        # Simple spatial unwrapping if no Gray code available
        if gray_code_x is None and gray_code_y is None:
            # Use OpenCV phase unwrapping or simple row-by-row unwrapping
            for y in range(h):
                unwrapped[y, :] = np.unwrap(phase_map[y, :])
            return unwrapped
        
        # Use Gray code to guide unwrapping
        if gray_code_x is not None:
            # Determine period from Gray code
            period_x = w / np.max(gray_code_x[gray_code_x > 0])
            
            # Unwrap using Gray code as guide
            k = np.round(gray_code_x / period_x)
            unwrapped = phase_map + 2 * np.pi * k
        
        return unwrapped
    
    @staticmethod
    def _save_debug_images(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        mask: np.ndarray,
        threshold_img: np.ndarray,
        debug_dir: str
    ):
        """Save debug images for pattern decoding."""
        import os
        os.makedirs(debug_dir, exist_ok=True)
        
        # Normalize coordinates for visualization
        x_vis = np.zeros_like(x_coords, dtype=np.uint8)
        y_vis = np.zeros_like(y_coords, dtype=np.uint8)
        
        valid_mask = mask & (x_coords >= 0) & (y_coords >= 0)
        if np.any(valid_mask):
            x_vis[valid_mask] = (x_coords[valid_mask] / np.max(x_coords[valid_mask]) * 255).astype(np.uint8)
            y_vis[valid_mask] = (y_coords[valid_mask] / np.max(y_coords[valid_mask]) * 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(os.path.join(debug_dir, "x_coordinates.png"), x_vis)
        cv2.imwrite(os.path.join(debug_dir, "y_coordinates.png"), y_vis)
        cv2.imwrite(os.path.join(debug_dir, "valid_mask.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(debug_dir, "threshold.png"), 
                    (threshold_img / np.max(threshold_img) * 255).astype(np.uint8))
        
        # Create color-coded coordinate map
        coords_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        coords_color[:, :, 0] = x_vis  # Red channel for X
        coords_color[:, :, 1] = y_vis  # Green channel for Y
        coords_color[:, :, 2] = mask.astype(np.uint8) * 255  # Blue channel for mask
        
        cv2.imwrite(os.path.join(debug_dir, "coordinates_color.png"), coords_color)
        
        logger.info(f"Saved debug images to {debug_dir}")


class ProjectorCameraTriangulator:
    """
    Triangulator for projector-camera structured light scanning.
    
    This class implements proper triangulation between camera and projector
    using calibrated parameters for high-density 3D reconstruction.
    """
    
    def __init__(self, calibration_path: Optional[str] = None):
        """
        Initialize triangulator with calibration data.
        
        Args:
            calibration_path: Path to projector-camera calibration file
        """
        self.camera_intrinsics = None
        self.camera_distortion = None
        self.projector_intrinsics = None
        self.projector_distortion = None
        self.rotation_matrix = None
        self.translation_vector = None
        self.essential_matrix = None
        self.fundamental_matrix = None
        self.q_matrix = None
        self.is_calibrated = False
        
        if calibration_path:
            self.load_calibration(calibration_path)
    
    def load_calibration(self, calibration_path: str) -> bool:
        """
        Load projector-camera calibration from file.
        
        Args:
            calibration_path: Path to calibration JSON file
            
        Returns:
            True if calibration loaded successfully
        """
        try:
            import json
            with open(calibration_path, 'r') as f:
                calib_data = json.load(f)
            
            # Load camera calibration
            if 'camera' in calib_data:
                cam = calib_data['camera']
                self.camera_intrinsics = np.array(cam['camera_matrix'])
                self.camera_distortion = np.array(cam['distortion_coefficients'])
            
            # Load projector calibration
            if 'projector' in calib_data:
                proj = calib_data['projector']
                self.projector_intrinsics = np.array(proj['camera_matrix'])
                self.projector_distortion = np.array(proj['distortion_coefficients'])
            
            # Load stereo calibration
            if 'stereo' in calib_data:
                stereo = calib_data['stereo']
                self.rotation_matrix = np.array(stereo['R'])
                self.translation_vector = np.array(stereo['T'])
                self.essential_matrix = np.array(stereo['E'])
                self.fundamental_matrix = np.array(stereo['F'])
                self.q_matrix = np.array(stereo['Q']) if 'Q' in stereo else None
            
            # Validate calibration
            if (self.camera_intrinsics is not None and 
                self.projector_intrinsics is not None and 
                self.rotation_matrix is not None and 
                self.translation_vector is not None):
                self.is_calibrated = True
                logger.info(f"Successfully loaded projector-camera calibration from {calibration_path}")
                return True
            else:
                logger.error(f"Incomplete calibration data in {calibration_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False
    
    def triangulate_points(self, 
                          camera_points: np.ndarray,
                          projector_points: np.ndarray,
                          undistort: bool = True) -> np.ndarray:
        """
        Triangulate 3D points from camera and projector correspondences.
        
        Args:
            camera_points: Camera pixel coordinates (Nx2)
            projector_points: Projector pixel coordinates (Nx2)
            undistort: Whether to undistort points before triangulation
            
        Returns:
            3D points in camera coordinate system (Nx3)
        """
        if not self.is_calibrated:
            raise ValueError("Triangulator not calibrated")
        
        if len(camera_points) != len(projector_points):
            raise ValueError("Camera and projector points must have same length")
        
        # Undistort points if requested
        if undistort:
            camera_undist = cv2.undistortPoints(
                camera_points.reshape(-1, 1, 2).astype(np.float32),
                self.camera_intrinsics,
                self.camera_distortion
            ).reshape(-1, 2)
            
            projector_undist = cv2.undistortPoints(
                projector_points.reshape(-1, 1, 2).astype(np.float32),
                self.projector_intrinsics,
                self.projector_distortion
            ).reshape(-1, 2)
        else:
            camera_undist = camera_points
            projector_undist = projector_points
        
        # Create projection matrices
        P_camera = np.hstack([self.camera_intrinsics, np.zeros((3, 1))])
        P_projector = np.dot(
            self.projector_intrinsics,
            np.hstack([
                self.rotation_matrix,
                self.translation_vector.reshape(-1, 1)
            ])
        )
        
        # Triangulate points using OpenCV
        points_4d = cv2.triangulatePoints(
            P_camera,
            P_projector,
            camera_undist.T.astype(np.float32),
            projector_undist.T.astype(np.float32)
        )
        
        # Convert from homogeneous to 3D coordinates
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        return points_3d
    
    def triangulate_from_phase_map(self,
                                  phase_map: np.ndarray,
                                  quality_mask: np.ndarray,
                                  projector_width: int = 1280,
                                  projector_height: int = 720,
                                  orientation: str = "vertical") -> np.ndarray:
        """
        Triangulate 3D points from a phase map.
        
        Args:
            phase_map: Unwrapped phase map from phase shift decoding
            quality_mask: Quality mask indicating valid pixels
            projector_width: Projector resolution width
            projector_height: Projector resolution height
            orientation: Phase encoding orientation ("vertical" or "horizontal")
            
        Returns:
            3D point cloud (HxWx3) with NaN for invalid points
        """
        if not self.is_calibrated:
            raise ValueError("Triangulator not calibrated")
        
        h, w = phase_map.shape
        points_3d = np.full((h, w, 3), np.nan, dtype=np.float32)
        
        # Get valid pixel coordinates
        valid_coords = np.where(quality_mask)
        if len(valid_coords[0]) == 0:
            logger.warning("No valid pixels in phase map")
            return points_3d
        
        # Create camera pixel coordinates
        camera_u, camera_v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert phase to projector coordinates
        if orientation == "vertical":
            # Phase encodes projector X coordinate
            max_phase = 2 * np.pi
            projector_u = (phase_map / max_phase) * projector_width
            projector_v = camera_v  # Assume no vertical encoding for now
        else:  # horizontal
            # Phase encodes projector Y coordinate
            max_phase = 2 * np.pi
            projector_u = camera_u  # Assume no horizontal encoding for now
            projector_v = (phase_map / max_phase) * projector_height
        
        # Extract valid correspondences
        camera_points = np.column_stack([
            camera_u[valid_coords].flatten(),
            camera_v[valid_coords].flatten()
        ])
        
        projector_points = np.column_stack([
            projector_u[valid_coords].flatten(),
            projector_v[valid_coords].flatten()
        ])
        
        # Triangulate 3D points
        try:
            triangulated_3d = self.triangulate_points(camera_points, projector_points)
            
            # Place back into image coordinates
            for i, (y, x) in enumerate(zip(valid_coords[0], valid_coords[1])):
                points_3d[y, x] = triangulated_3d[i]
                
        except Exception as e:
            logger.error(f"Triangulation failed: {e}")
        
        return points_3d
    
    def triangulate_from_gray_code(self,
                                  x_coords: np.ndarray,
                                  y_coords: np.ndarray,
                                  mask: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from Gray code coordinates.
        
        Args:
            x_coords: Projector X coordinates from Gray code
            y_coords: Projector Y coordinates from Gray code
            mask: Validity mask for coordinates
            
        Returns:
            3D point cloud (HxWx3) with NaN for invalid points
        """
        if not self.is_calibrated:
            raise ValueError("Triangulator not calibrated")
        
        h, w = x_coords.shape
        points_3d = np.full((h, w, 3), np.nan, dtype=np.float32)
        
        # Get valid pixel coordinates
        valid_coords = np.where(mask & (x_coords >= 0) & (y_coords >= 0))
        if len(valid_coords[0]) == 0:
            logger.warning("No valid coordinates in Gray code")
            return points_3d
        
        # Create camera pixel coordinates
        camera_u, camera_v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Extract valid correspondences
        camera_points = np.column_stack([
            camera_u[valid_coords].flatten(),
            camera_v[valid_coords].flatten()
        ])
        
        projector_points = np.column_stack([
            x_coords[valid_coords].flatten(),
            y_coords[valid_coords].flatten()
        ])
        
        # Triangulate 3D points
        try:
            triangulated_3d = self.triangulate_points(camera_points, projector_points)
            
            # Place back into image coordinates
            for i, (y, x) in enumerate(zip(valid_coords[0], valid_coords[1])):
                points_3d[y, x] = triangulated_3d[i]
                
        except Exception as e:
            logger.error(f"Triangulation failed: {e}")
        
        return points_3d
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information summary."""
        if not self.is_calibrated:
            return {"calibrated": False}
        
        return {
            "calibrated": True,
            "camera_focal_length": [
                float(self.camera_intrinsics[0, 0]),
                float(self.camera_intrinsics[1, 1])
            ],
            "camera_principal_point": [
                float(self.camera_intrinsics[0, 2]),
                float(self.camera_intrinsics[1, 2])
            ],
            "projector_focal_length": [
                float(self.projector_intrinsics[0, 0]),
                float(self.projector_intrinsics[1, 1])
            ],
            "projector_principal_point": [
                float(self.projector_intrinsics[0, 2]),
                float(self.projector_intrinsics[1, 2])
            ],
            "baseline": float(np.linalg.norm(self.translation_vector)),
            "has_q_matrix": self.q_matrix is not None
        }