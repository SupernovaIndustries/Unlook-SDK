"""
Projector Calibration for Structured Light Systems

Calibrates the projector as an "inverse camera" to enable precise
projector-camera triangulation for 3D reconstruction.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ProjectorCalibrator:
    """
    Professional projector calibration for structured light systems.
    
    Treats the projector as an "inverse camera" and calibrates both
    intrinsic parameters and extrinsic relationship with the camera.
    """
    
    def __init__(self, 
                 projector_width: int = 1920,
                 projector_height: int = 1080,
                 checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):  # mm
        """
        Initialize projector calibrator.
        
        Args:
            projector_width: Projector resolution width
            projector_height: Projector resolution height
            checkerboard_size: Checkerboard pattern size (corners_x, corners_y)
            square_size: Physical size of checkerboard squares in mm
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Calibration data storage
        self.camera_intrinsics = None
        self.camera_distortion = None
        self.projector_intrinsics = None
        self.projector_distortion = None
        self.rotation_matrix = None
        self.translation_vector = None
        self.fundamental_matrix = None
        self.essential_matrix = None
        
        # 3D world points for checkerboard
        self.world_points = self._generate_world_points()
        
    def _generate_world_points(self) -> np.ndarray:
        """Generate 3D world coordinates for checkerboard corners."""
        corners_x, corners_y = self.checkerboard_size
        
        # Create grid of 3D points (Z=0 plane)
        world_points = np.zeros((corners_x * corners_y, 3), np.float32)
        world_points[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
        world_points *= self.square_size
        
        return world_points
    
    def calibrate_camera(self, 
                        camera_images: List[np.ndarray],
                        detect_corners: bool = True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Calibrate camera using checkerboard images.
        
        Args:
            camera_images: List of camera images with checkerboard
            detect_corners: Whether to detect corners or use provided points
            
        Returns:
            Tuple of (intrinsics, distortion, image_points_list)
        """
        logger.info(f"Calibrating camera with {len(camera_images)} images")
        
        # Storage for calibration data
        world_points_list = []
        image_points_list = []
        
        # Process each image
        for i, img in enumerate(camera_images):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Find checkerboard corners
            found, corners = cv2.findChessboardCorners(
                gray, 
                self.checkerboard_size, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if found:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                world_points_list.append(self.world_points)
                image_points_list.append(corners_refined)
                
                logger.debug(f"Image {i+1}: Found checkerboard corners")
            else:
                logger.warning(f"Image {i+1}: Could not find checkerboard corners")
        
        if len(image_points_list) < 3:
            raise ValueError(f"Need at least 3 valid images, got {len(image_points_list)}")
        
        # Calibrate camera
        logger.info(f"Performing camera calibration with {len(image_points_list)} valid images")
        
        img_height, img_width = camera_images[0].shape[:2]
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            world_points_list,
            image_points_list, 
            (img_width, img_height),
            None,
            None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        # Store results
        self.camera_intrinsics = camera_matrix
        self.camera_distortion = dist_coeffs
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(world_points_list)):
            projected_points, _ = cv2.projectPoints(
                world_points_list[i], rvecs[i], tvecs[i], 
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(image_points_list[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
        
        avg_error = total_error / len(world_points_list)
        logger.info(f"Camera calibration completed - Average reprojection error: {avg_error:.4f} pixels")
        
        return camera_matrix, dist_coeffs, image_points_list
    
    def calibrate_projector_camera_system(self,
                                        camera_images: List[np.ndarray],
                                        projected_patterns: List[np.ndarray],
                                        camera_image_points: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Calibrate complete projector-camera system using Gray code patterns.
        
        Args:
            camera_images: Camera images with checkerboard under projection
            projected_patterns: Corresponding Gray code patterns  
            camera_image_points: Pre-detected camera corner points
            
        Returns:
            Complete calibration results dictionary
        """
        logger.info("Calibrating projector-camera system")
        
        # Step 1: Calibrate camera if not already done
        if self.camera_intrinsics is None:
            if camera_image_points is None:
                self.calibrate_camera(camera_images)
                camera_image_points = []
                for img in camera_images:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                    found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size)
                    if found:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        camera_image_points.append(corners_refined)
            else:
                logger.info("Using provided camera image points")
        
        # Step 2: Decode projector coordinates using Gray code
        projector_image_points = self._decode_projector_coordinates(
            projected_patterns, camera_image_points
        )
        
        # Step 3: Calibrate projector as inverse camera
        self._calibrate_projector_intrinsics(projector_image_points)
        
        # Step 4: Compute stereo calibration (projector-camera)
        self._calibrate_stereo_system(camera_image_points, projector_image_points)
        
        # Step 5: Compute quality metrics
        quality_metrics = self._compute_calibration_quality(
            camera_image_points, projector_image_points
        )
        
        return {
            'camera_intrinsics': self.camera_intrinsics,
            'camera_distortion': self.camera_distortion,
            'projector_intrinsics': self.projector_intrinsics,
            'projector_distortion': self.projector_distortion,
            'rotation_matrix': self.rotation_matrix,
            'translation_vector': self.translation_vector,
            'fundamental_matrix': self.fundamental_matrix,
            'essential_matrix': self.essential_matrix,
            'quality_metrics': quality_metrics
        }
    
    def _decode_projector_coordinates(self,
                                    projected_patterns: List[np.ndarray],
                                    camera_image_points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Decode projector pixel coordinates using Gray code patterns.
        
        This is a simplified version - in practice you'd use full Gray code decoding
        from the pattern_decoder module.
        """
        logger.info("Decoding projector coordinates from Gray code patterns")
        
        projector_points = []
        
        for i, camera_points in enumerate(camera_image_points):
            # For each detected corner in camera image, 
            # find corresponding projector coordinate
            proj_coords = []
            
            for corner in camera_points:
                x, y = int(corner[0][0]), int(corner[0][1])
                
                # Simplified Gray code decoding 
                # (In real implementation, use proper Gray code decoder)
                proj_x = self._decode_gray_code_x(projected_patterns, x, y)
                proj_y = self._decode_gray_code_y(projected_patterns, x, y)
                
                proj_coords.append([[proj_x, proj_y]])
            
            projector_points.append(np.array(proj_coords, dtype=np.float32))
        
        logger.info(f"Decoded projector coordinates for {len(projector_points)} images")
        return projector_points
    
    def _decode_gray_code_x(self, patterns: List[np.ndarray], x: int, y: int) -> float:
        """Simplified Gray code decoding for X coordinate."""
        # This is a placeholder - implement proper Gray code decoding
        # For now, use linear mapping as approximation
        return x * (self.projector_width / patterns[0].shape[1])
    
    def _decode_gray_code_y(self, patterns: List[np.ndarray], x: int, y: int) -> float:
        """Simplified Gray code decoding for Y coordinate.""" 
        # This is a placeholder - implement proper Gray code decoding
        # For now, use linear mapping as approximation
        return y * (self.projector_height / patterns[0].shape[0])
    
    def _calibrate_projector_intrinsics(self, projector_image_points: List[np.ndarray]):
        """Calibrate projector intrinsics treating it as inverse camera."""
        logger.info("Calibrating projector intrinsics")
        
        # Prepare world points (same as for camera)
        world_points_list = [self.world_points] * len(projector_image_points)
        
        # Calibrate projector
        ret, proj_matrix, proj_dist, rvecs, tvecs = cv2.calibrateCamera(
            world_points_list,
            projector_image_points,
            (self.projector_width, self.projector_height),
            None,
            None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        self.projector_intrinsics = proj_matrix
        self.projector_distortion = proj_dist
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(world_points_list)):
            projected_points, _ = cv2.projectPoints(
                world_points_list[i], rvecs[i], tvecs[i],
                proj_matrix, proj_dist
            )
            error = cv2.norm(projector_image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
        
        avg_error = total_error / len(world_points_list)
        logger.info(f"Projector calibration completed - Average reprojection error: {avg_error:.4f} pixels")
    
    def _calibrate_stereo_system(self,
                               camera_points: List[np.ndarray],
                               projector_points: List[np.ndarray]):
        """Calibrate stereo system between camera and projector."""
        logger.info("Calibrating stereo camera-projector system")
        
        world_points_list = [self.world_points] * len(camera_points)
        
        # Stereo calibration
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            world_points_list,
            camera_points,
            projector_points,
            self.camera_intrinsics,
            self.camera_distortion,
            self.projector_intrinsics, 
            self.projector_distortion,
            (self.projector_width, self.projector_height),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        self.rotation_matrix = R
        self.translation_vector = T
        self.essential_matrix = E
        self.fundamental_matrix = F
        
        logger.info(f"Stereo calibration completed - Baseline: {np.linalg.norm(T):.2f} mm")
    
    def _compute_calibration_quality(self,
                                   camera_points: List[np.ndarray],
                                   projector_points: List[np.ndarray]) -> Dict[str, float]:
        """Compute calibration quality metrics."""
        
        # Epipolar constraint error
        epipolar_errors = []
        for cam_pts, proj_pts in zip(camera_points, projector_points):
            for cam_pt, proj_pt in zip(cam_pts, proj_pts):
                # Convert to homogeneous coordinates
                cam_h = np.array([cam_pt[0][0], cam_pt[0][1], 1.0])
                proj_h = np.array([proj_pt[0][0], proj_pt[0][1], 1.0])
                
                # Compute epipolar error
                error = abs(np.dot(cam_h, np.dot(self.fundamental_matrix, proj_h)))
                epipolar_errors.append(error)
        
        avg_epipolar_error = np.mean(epipolar_errors)
        max_epipolar_error = np.max(epipolar_errors)
        
        # Baseline quality
        baseline_length = np.linalg.norm(self.translation_vector)
        
        return {
            'average_epipolar_error': avg_epipolar_error,
            'max_epipolar_error': max_epipolar_error,
            'baseline_length_mm': baseline_length,
            'num_calibration_images': len(camera_points)
        }
    
    def save_calibration(self, filepath: Path) -> Path:
        """Save calibration results to file."""
        calibration_data = {
            'projector_resolution': [self.projector_width, self.projector_height],
            'checkerboard_size': self.checkerboard_size,
            'square_size_mm': self.square_size,
            'camera_intrinsics': self.camera_intrinsics.tolist() if self.camera_intrinsics is not None else None,
            'camera_distortion': self.camera_distortion.tolist() if self.camera_distortion is not None else None,
            'projector_intrinsics': self.projector_intrinsics.tolist() if self.projector_intrinsics is not None else None,
            'projector_distortion': self.projector_distortion.tolist() if self.projector_distortion is not None else None,
            'rotation_matrix': self.rotation_matrix.tolist() if self.rotation_matrix is not None else None,
            'translation_vector': self.translation_vector.tolist() if self.translation_vector is not None else None,
            'fundamental_matrix': self.fundamental_matrix.tolist() if self.fundamental_matrix is not None else None,
            'essential_matrix': self.essential_matrix.tolist() if self.essential_matrix is not None else None
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
        return filepath
    
    def load_calibration(self, filepath: Path) -> bool:
        """Load calibration results from file."""
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.projector_width, self.projector_height = data['projector_resolution']
            self.checkerboard_size = tuple(data['checkerboard_size'])
            self.square_size = data['square_size_mm']
            
            self.camera_intrinsics = np.array(data['camera_intrinsics']) if data['camera_intrinsics'] else None
            self.camera_distortion = np.array(data['camera_distortion']) if data['camera_distortion'] else None
            self.projector_intrinsics = np.array(data['projector_intrinsics']) if data['projector_intrinsics'] else None
            self.projector_distortion = np.array(data['projector_distortion']) if data['projector_distortion'] else None
            self.rotation_matrix = np.array(data['rotation_matrix']) if data['rotation_matrix'] else None
            self.translation_vector = np.array(data['translation_vector']) if data['translation_vector'] else None
            self.fundamental_matrix = np.array(data['fundamental_matrix']) if data['fundamental_matrix'] else None
            self.essential_matrix = np.array(data['essential_matrix']) if data['essential_matrix'] else None
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def get_projection_matrix(self, for_projector: bool = True) -> np.ndarray:
        """Get projection matrix for camera or projector."""
        if for_projector:
            if self.projector_intrinsics is None:
                raise ValueError("Projector not calibrated")
            return self.projector_intrinsics
        else:
            if self.camera_intrinsics is None:
                raise ValueError("Camera not calibrated")
            return self.camera_intrinsics
    
    def is_calibrated(self) -> bool:
        """Check if full system calibration is complete."""
        return all([
            self.camera_intrinsics is not None,
            self.projector_intrinsics is not None,
            self.rotation_matrix is not None,
            self.translation_vector is not None
        ])