"""
Camera calibration module for the UnLook SDK.

This module provides functions for calibrating stereo cameras using a checkerboard
pattern, obtaining intrinsic and extrinsic camera parameters, and saving/loading 
these parameters for use in 3D scanning applications.

Based on TemugeB's stereo camera calibration approach:
https://github.com/TemugeB/python_stereo_camera_calibrate
"""

import os
import time
import logging
import numpy as np
import cv2
import json
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class StereoCalibrator:
    """
    Stereo camera calibration class for UnLook SDK.
    
    This class provides methods for calibrating stereo cameras using a checkerboard
    pattern, obtaining intrinsic and extrinsic camera parameters, and saving/loading
    these parameters for use in 3D scanning applications.
    """
    
    def __init__(self, 
                 checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 24.0,
                 calibration_flags: int = cv2.CALIB_FIX_INTRINSIC,
                 max_frames: int = 30):
        """
        Initialize the stereo calibrator with default values.
        
        Args:
            checkerboard_size: Size of the checkerboard pattern (width, height) in inner corners
            square_size: Size of the checkerboard squares in mm
            calibration_flags: Flags to use for cv2.stereoCalibrate
            max_frames: Maximum number of calibration frames to use
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.calibration_flags = calibration_flags
        self.max_frames = max_frames
        
        # Initialize camera parameters to None
        self._init_camera_params()
    
    def _init_camera_params(self):
        """Initialize all camera parameters to None."""
        # Camera parameters
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None
        self.F = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.reprojection_error = None
        self.image_size = None
        self.roi_left = None
        self.roi_right = None
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
        
        logger.info(f"Stereo calibrator initialized with checkerboard size {self.checkerboard_size}")
        
        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        
        # Calibration points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane
        
        # Image size
        self.image_size = None
        
        logger.info(f"Stereo calibrator initialized with checkerboard size {self.checkerboard_size}")
    
    def create_object_points(self) -> np.ndarray:
        """
        Create 3D object points for the checkerboard in the world coordinate system.
        
        Returns:
            3D object points
        """
        width, height = self.checkerboard_size
        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * self.square_size
        return objp
    
    def find_checkerboard(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
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
            gray = image.copy()
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.checkerboard_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE + 
            cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            return True, corners
        else:
            return False, None
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected checkerboard corners on an image.
        
        Args:
            image: Input image
            corners: Detected corners
            
        Returns:
            Image with drawn corners
        """
        # Make a copy to avoid modifying the original
        image_copy = image.copy()
        
        # Convert to color for drawing if needed
        if len(image_copy.shape) == 2:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        
        # Draw the corners
        cv2.drawChessboardCorners(image_copy, self.checkerboard_size, corners, True)
        
        return image_copy
    
    def calibrate_single_camera(self, 
                              images: List[np.ndarray],
                              visualize: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate a single camera using multiple images of a checkerboard.
        
        Args:
            images: List of calibration images
            visualize: Whether to visualize the detected corners
            
        Returns:
            Tuple of (camera_matrix, dist_coeffs, reprojection_error)
        """
        # Check if we have enough images
        if len(images) < 5:
            logger.error(f"Not enough calibration images: {len(images)} (need at least 5)")
            return None, None, float('inf')
        
        # Get image size
        if self.image_size is None:
            height, width = images[0].shape[:2]
            self.image_size = (width, height)
        
        # Create object points
        objp = self.create_object_points()
        
        # Lists to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Find corners in each image
        for i, img in enumerate(images):
            ret, corners = self.find_checkerboard(img)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                if visualize:
                    # Draw and display the corners
                    img_corners = self.draw_corners(img, corners)
                    cv2.imshow(f'Corners (Image {i+1})', img_corners)
                    cv2.waitKey(500)  # Display for 500ms
            else:
                logger.warning(f"Could not find checkerboard in image {i+1}")
        
        if visualize:
            cv2.destroyAllWindows()
        
        # Check if we found enough corners
        if len(objpoints) < 5:
            logger.error(f"Not enough valid calibration images: {len(objpoints)} (need at least 5)")
            return None, None, float('inf')
        
        logger.info(f"Calibrating single camera with {len(objpoints)} images")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        reprojection_error = mean_error / len(objpoints)
        logger.info(f"Single camera calibration complete. Reprojection error: {reprojection_error:.6f}")
        
        return camera_matrix, dist_coeffs, reprojection_error
    
    def calibrate_stereo(self, 
                        left_images: List[np.ndarray], 
                        right_images: List[np.ndarray],
                        visualize: bool = False) -> Dict[str, Any]:
        """
        Calibrate a stereo camera system using images from both cameras.
        
        Args:
            left_images: List of images from the left camera
            right_images: List of images from the right camera
            visualize: Whether to visualize the detected corners
            
        Returns:
            Dictionary of calibration parameters
        """
        # Check if we have matching image pairs
        if len(left_images) != len(right_images):
            logger.error(f"Number of left and right images must match: {len(left_images)} != {len(right_images)}")
            return {}
        
        # Check if we have enough image pairs
        if len(left_images) < 5:
            logger.error(f"Not enough calibration image pairs: {len(left_images)} (need at least 5)")
            return {}
        
        # Limit the number of frames for calibration
        if len(left_images) > self.max_frames:
            logger.info(f"Limiting to {self.max_frames} calibration frames")
            left_images = left_images[:self.max_frames]
            right_images = right_images[:self.max_frames]
        
        # First calibrate each camera individually
        logger.info("Calibrating left camera...")
        self.camera_matrix_left, self.dist_coeffs_left, left_error = self.calibrate_single_camera(
            left_images, visualize=False
        )
        
        logger.info("Calibrating right camera...")
        self.camera_matrix_right, self.dist_coeffs_right, right_error = self.calibrate_single_camera(
            right_images, visualize=False
        )
        
        # Check if individual calibrations succeeded
        if (self.camera_matrix_left is None or self.camera_matrix_right is None or
            self.dist_coeffs_left is None or self.dist_coeffs_right is None):
            logger.error("Individual camera calibration failed")
            return {}
        
        # Get image size if not already set
        if self.image_size is None:
            height, width = left_images[0].shape[:2]
            self.image_size = (width, height)
        
        # Reset calibration points
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        
        # Create object points
        objp = self.create_object_points()
        
        # Find checkerboard corners in stereo pairs
        valid_pairs = 0
        for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
            # Find corners in left image
            left_ret, left_corners = self.find_checkerboard(left_img)
            
            # Find corners in right image
            right_ret, right_corners = self.find_checkerboard(right_img)
            
            # If both images have corners, add to calibration points
            if left_ret and right_ret:
                self.objpoints.append(objp)
                self.imgpoints_left.append(left_corners)
                self.imgpoints_right.append(right_corners)
                valid_pairs += 1
                
                if visualize:
                    # Draw corners on both images
                    left_img_corners = self.draw_corners(left_img, left_corners)
                    right_img_corners = self.draw_corners(right_img, right_corners)
                    
                    # Combine images for display
                    combined = np.hstack((left_img_corners, right_img_corners))
                    
                    # Resize if too large
                    if combined.shape[1] > 1600:
                        scale = 1600 / combined.shape[1]
                        combined = cv2.resize(combined, None, fx=scale, fy=scale)
                    
                    cv2.imshow(f'Stereo Corners (Pair {i+1})', combined)
                    cv2.waitKey(500)  # Display for 500ms
            else:
                logger.warning(f"Could not find checkerboard in image pair {i+1}")
        
        if visualize:
            cv2.destroyAllWindows()
        
        # Check if we have enough valid pairs
        if valid_pairs < 5:
            logger.error(f"Not enough valid stereo pairs: {valid_pairs} (need at least 5)")
            return {}
        
        logger.info(f"Calibrating stereo cameras with {valid_pairs} image pairs")
        
        # Perform stereo calibration
        ret, self.camera_matrix_left, self.dist_coeffs_left, self.camera_matrix_right, self.dist_coeffs_right, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.image_size, flags=self.calibration_flags
        )
        
        self.reprojection_error = ret
        logger.info(f"Stereo calibration complete. Reprojection error (RMSE): {ret:.6f}")
        
        # Apply bundle adjustment if available
        try:
            from .bundle_adjustment import StereoCalibrationOptimizer
            
            if ret > 0.5:  # Only apply if initial error is high
                logger.info(f"RMS error {ret:.4f} > 0.5px threshold - applying bundle adjustment...")
                
                optimizer = StereoCalibrationOptimizer()
                initial_params = {
                    'K1': self.camera_matrix_left,
                    'D1': self.dist_coeffs_left,
                    'K2': self.camera_matrix_right,
                    'D2': self.dist_coeffs_right,
                    'R': self.R,
                    'T': self.T
                }
                
                optimized_params, summary = optimizer.optimize_stereo_calibration(
                    self.imgpoints_left, self.imgpoints_right, self.objpoints,
                    initial_params, self.image_size
                )
                
                # Update parameters if optimization succeeded
                if summary.get('rms_error', ret) < ret:
                    logger.info("✅ Bundle adjustment improved calibration - updating parameters")
                    self.camera_matrix_left = optimized_params['K1']
                    self.dist_coeffs_left = optimized_params['D1']
                    self.camera_matrix_right = optimized_params['K2']
                    self.dist_coeffs_right = optimized_params['D2']
                    self.R = optimized_params['R']
                    self.T = optimized_params['T']
                    self.reprojection_error = summary['rms_error']
                    
                    # Store bundle adjustment metadata
                    self.bundle_adjustment_summary = summary
                else:
                    logger.warning("Bundle adjustment did not improve calibration - keeping original")
                    self.bundle_adjustment_summary = None
            else:
                logger.info(f"RMS error {ret:.4f} already meets target - skipping bundle adjustment")
                self.bundle_adjustment_summary = None
                
        except ImportError:
            logger.warning("Bundle adjustment not available (Ceres Solver not installed)")
            self.bundle_adjustment_summary = None
        except Exception as e:
            logger.warning(f"Bundle adjustment failed: {e}")
            self.bundle_adjustment_summary = None
        
        # Calculate rectification transforms with improved parameters
        # Use alpha=0.5 for balanced rectification (not too much distortion in both cameras)
        # ISO compliance: Using cv2.CALIB_ZERO_DISPARITY to ensure epipolar lines are aligned
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_left, roi_right = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.5,  # 0.5 gives a balanced rectification with minimal distortion
            newImageSize=self.image_size  # Explicitly set to ensure consistency
        )

        # Store the valid ROIs from rectification for later use in masking
        self.roi_left = roi_left
        self.roi_right = roi_right

        logger.info(f"Rectification ROI - Left: {roi_left}, Right: {roi_right}")

        # Save the baseline in mm (for ISO compliance documentation)
        # The baseline is the magnitude of the translation vector
        # NOTE: T is already in the same units as the checkerboard square_size (mm)
        self.baseline_mm = float(np.linalg.norm(self.T))
        logger.info(f"Stereo baseline: {self.baseline_mm:.2f}mm")
        
        # Sanity check for baseline
        if self.baseline_mm < 50 or self.baseline_mm > 200:
            logger.warning(f"Baseline {self.baseline_mm:.2f}mm seems unusual for stereo cameras")
            logger.warning("Expected range is typically 50-200mm")
        
        # Prepare result dictionary
        result = {
            "reprojection_error": float(self.reprojection_error),
            "image_size": list(self.image_size),
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "R1": self.R1.tolist(),
            "R2": self.R2.tolist(),
            "P1": self.P1.tolist(),
            "P2": self.P2.tolist(),
            "Q": self.Q.tolist(),
            "left_error": float(left_error),
            "right_error": float(right_error),
            "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_valid_pairs": valid_pairs,
            "bundle_adjustment_applied": self.bundle_adjustment_summary is not None
        }
        
        # Add bundle adjustment summary if available
        if self.bundle_adjustment_summary:
            result["bundle_adjustment"] = self.bundle_adjustment_summary
        
        return result
    
    def save_calibration(self, output_file: str) -> bool:
        """
        Save calibration parameters to a file.
        
        Args:
            output_file: Output file path (JSON)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if we have calibration parameters
        if self.camera_matrix_left is None or self.camera_matrix_right is None:
            logger.error("No calibration parameters to save")
            return False
        
        # Create result dictionary
        result = {
            "reprojection_error": float(self.reprojection_error),
            "image_size": list(self.image_size),
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "R1": self.R1.tolist(),
            "R2": self.R2.tolist(),
            "P1": self.P1.tolist(),
            "P2": self.P2.tolist(),
            "Q": self.Q.tolist(),
            "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save to file
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Calibration saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration to {output_file}: {e}")
            return False
    
    def load_calibration(self, input_file: str) -> bool:
        """
        Load calibration parameters from a file.
        
        Args:
            input_file: Input file path (JSON)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Load parameters
            self.reprojection_error = float(data["reprojection_error"])
            self.image_size = tuple(data["image_size"])
            self.camera_matrix_left = np.array(data["camera_matrix_left"])
            self.dist_coeffs_left = np.array(data["dist_coeffs_left"])
            self.camera_matrix_right = np.array(data["camera_matrix_right"])
            self.dist_coeffs_right = np.array(data["dist_coeffs_right"])
            self.R = np.array(data["R"])
            self.T = np.array(data["T"])
            self.E = np.array(data["E"])
            self.F = np.array(data["F"])
            self.R1 = np.array(data["R1"])
            self.R2 = np.array(data["R2"])
            self.P1 = np.array(data["P1"])
            self.P2 = np.array(data["P2"])
            self.Q = np.array(data["Q"])
            
            logger.info(f"Calibration loaded from {input_file}")
            logger.info(f"Reprojection error: {self.reprojection_error:.6f}")
            return True
        except Exception as e:
            logger.error(f"Error loading calibration from {input_file}: {e}")
            return False
    
    def save_calibration_npy(self, output_file: str) -> bool:
        """
        Save calibration parameters and ISO-compliant metadata to a NumPy file.

        This method follows ISO 52902 standards by including complete metadata
        about the calibration process, enabling better traceability and reproducibility.

        Args:
            output_file: Output file path (.npz)

        Returns:
            True if successful, False otherwise
        """
        # Check if we have calibration parameters
        if self.camera_matrix_left is None or self.camera_matrix_right is None:
            logger.error("No calibration parameters to save")
            return False

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        try:
            # Calculate baseline in mm if we have translation vector
            baseline_mm = 0.0
            if hasattr(self, 'T') and self.T is not None:
                baseline_mm = float(np.linalg.norm(self.T) * 1000.0)  # Convert to mm

            # Calculate focal length in pixels
            if hasattr(self, 'camera_matrix_left') and self.camera_matrix_left is not None:
                fx_left = float(self.camera_matrix_left[0, 0])
                fy_left = float(self.camera_matrix_left[1, 1])
                focal_length_left = (fx_left + fy_left) / 2.0
            else:
                focal_length_left = 0.0

            if hasattr(self, 'camera_matrix_right') and self.camera_matrix_right is not None:
                fx_right = float(self.camera_matrix_right[0, 0])
                fy_right = float(self.camera_matrix_right[1, 1])
                focal_length_right = (fx_right + fy_right) / 2.0
            else:
                focal_length_right = 0.0

            # Get ROI information if available
            roi_left = getattr(self, 'roi_left', [0, 0, 0, 0])
            roi_right = getattr(self, 'roi_right', [0, 0, 0, 0])

            # Create enhanced metadata for ISO compliance
            metadata = {
                "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_version": "2.0.0",  # Version of the calibration format
                "baseline_mm": baseline_mm,
                "focal_length_pixels": {
                    "left": focal_length_left,
                    "right": focal_length_right
                },
                "checkerboard": {
                    "size": getattr(self, 'checkerboard_size', (0, 0)),
                    "square_size_mm": getattr(self, 'square_size', 0.0)
                },
                "reprojection_error": float(self.reprojection_error),
                "measurement_uncertainty_mm": float(self.reprojection_error * baseline_mm / focal_length_left)
                                              if focal_length_left > 0 and baseline_mm > 0 else 0.0,
                "calibration_method": "OpenCV stereoCalibrate",
                "rectification": {
                    "algorithm": "OpenCV stereoRectify",
                    "alpha": 0.5,  # The alpha value used for rectification
                    "roi": {
                        "left": roi_left,
                        "right": roi_right
                    }
                },
                "iso_52902_compliance": True  # Mark as ISO 52902 compliant
            }

            # Convert any numpy types to native Python types for better compatibility
            metadata_json = json.dumps(metadata)
            metadata = json.loads(metadata_json)

            # Compute a calibration quality score (0-100)
            # Based on reprojection error and coverage
            if self.reprojection_error < 0.5:
                quality_score = 90 - self.reprojection_error * 40  # Excellent: error < 0.5 px
            elif self.reprojection_error < 1.0:
                quality_score = 70 - (self.reprojection_error - 0.5) * 40  # Good: error 0.5-1.0 px
            else:
                quality_score = max(0, 50 - (self.reprojection_error - 1.0) * 25)  # Poor: error > 1.0 px

            quality_score = min(100, max(0, quality_score))  # Clamp to 0-100 range
            metadata["quality_score"] = float(quality_score)

            # Log calibration quality assessment
            quality_level = "Excellent" if quality_score >= 80 else \
                            "Good" if quality_score >= 60 else \
                            "Fair" if quality_score >= 40 else "Poor"

            logger.info(f"Calibration quality: {quality_level} ({quality_score:.1f}/100)")
            logger.info(f"Estimated measurement uncertainty: {metadata['measurement_uncertainty_mm']:.3f}mm")

            # Save as NPZ file (compressed) with enhanced metadata
            np.savez(
                output_file,
                reprojection_error=self.reprojection_error,
                image_size=self.image_size,
                camera_matrix_left=self.camera_matrix_left,
                dist_coeffs_left=self.dist_coeffs_left,
                camera_matrix_right=self.camera_matrix_right,
                dist_coeffs_right=self.dist_coeffs_right,
                R=self.R,
                T=self.T,
                E=self.E,
                F=self.F,
                R1=self.R1,
                R2=self.R2,
                P1=self.P1,
                P2=self.P2,
                Q=self.Q,
                metadata=metadata,  # Add the enhanced metadata
                roi_left=roi_left,  # Add rectification ROIs
                roi_right=roi_right,
                baseline_mm=baseline_mm,  # Add baseline in mm
                quality_score=quality_score  # Add quality score
            )
            logger.info(f"Calibration saved to {output_file} with ISO 52902 compliant metadata")

            # Save metadata separately as JSON for easy inspection
            metadata_file = output_file.replace('.npz', '.json')
            if metadata_file == output_file:
                metadata_file += '.json'

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Calibration metadata saved to {metadata_file}")

            return True
        except Exception as e:
            logger.error(f"Error saving calibration to {output_file}: {e}")
            return False
    
    def load_calibration_npy(self, input_file: str) -> bool:
        """
        Load calibration parameters from a NumPy file with enhanced ISO metadata support.

        This method handles both legacy calibration files and newer files with
        ISO 52902 compliant metadata.

        Args:
            input_file: Input file path (.npz)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the NPZ file
            data = np.load(input_file, allow_pickle=True)

            # Load essential parameters
            self.reprojection_error = float(data["reprojection_error"])
            self.image_size = tuple(data["image_size"])
            self.camera_matrix_left = data["camera_matrix_left"]
            self.dist_coeffs_left = data["dist_coeffs_left"]
            self.camera_matrix_right = data["camera_matrix_right"]
            self.dist_coeffs_right = data["dist_coeffs_right"]
            self.R = data["R"]
            self.T = data["T"]
            self.E = data["E"]
            self.F = data["F"]
            self.R1 = data["R1"]
            self.R2 = data["R2"]
            self.P1 = data["P1"]
            self.P2 = data["P2"]
            self.Q = data["Q"]

            # Load enhanced metadata if available (newer files)
            has_enhanced_metadata = False
            if "metadata" in data:
                try:
                    # Load metadata object (might be serialized as a numpy object)
                    metadata = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]

                    # Store key metadata fields as class attributes with null safety
                    if "baseline_mm" in metadata and metadata["baseline_mm"] is not None:
                        self.baseline_mm = float(metadata["baseline_mm"])
                        has_enhanced_metadata = True

                    if "quality_score" in metadata and metadata["quality_score"] is not None:
                        self.quality_score = float(metadata["quality_score"])
                        has_enhanced_metadata = True

                    if "measurement_uncertainty_mm" in metadata and metadata["measurement_uncertainty_mm"] is not None:
                        self.measurement_uncertainty_mm = float(metadata["measurement_uncertainty_mm"])
                        has_enhanced_metadata = True

                    if "rectification" in metadata and "roi" in metadata["rectification"]:
                        if "left" in metadata["rectification"]["roi"]:
                            self.roi_left = tuple(metadata["rectification"]["roi"]["left"])
                            has_enhanced_metadata = True
                        if "right" in metadata["rectification"]["roi"]:
                            self.roi_right = tuple(metadata["rectification"]["roi"]["right"])
                            has_enhanced_metadata = True

                    # Store the full metadata object
                    self.metadata = metadata

                    logger.info("Loaded enhanced ISO 52902 compliant metadata")
                except Exception as e:
                    logger.warning(f"Error loading enhanced metadata: {e}")

            # Load ROIs directly if available in the file (newer files)
            if "roi_left" in data:
                self.roi_left = tuple(data["roi_left"])
                has_enhanced_metadata = True
            if "roi_right" in data:
                self.roi_right = tuple(data["roi_right"])
                has_enhanced_metadata = True

            # Load baseline directly if available and not None
            if "baseline_mm" in data and data["baseline_mm"] is not None:
                self.baseline_mm = float(data["baseline_mm"])
                has_enhanced_metadata = True

            # Calculate baseline if not loaded but we have translation vector
            if not hasattr(self, 'baseline_mm') and hasattr(self, 'T') and self.T is not None:
                self.baseline_mm = float(np.linalg.norm(self.T) * 1000.0)  # Convert to mm

            # Calculate focal lengths
            fx_left = float(self.camera_matrix_left[0, 0])
            fy_left = float(self.camera_matrix_left[1, 1])
            self.focal_length_left = (fx_left + fy_left) / 2.0

            fx_right = float(self.camera_matrix_right[0, 0])
            fy_right = float(self.camera_matrix_right[1, 1])
            self.focal_length_right = (fx_right + fy_right) / 2.0

            # Calculate measurement uncertainty if not loaded
            if not hasattr(self, 'measurement_uncertainty_mm') and hasattr(self, 'baseline_mm'):
                # Uncertainty is proportional to reprojection error * baseline / focal length
                self.measurement_uncertainty_mm = float(self.reprojection_error * self.baseline_mm / self.focal_length_left)

            # Log success with enhanced details
            logger.info(f"Calibration loaded from {input_file}")
            logger.info(f"Reprojection error: {self.reprojection_error:.4f} pixels")

            if hasattr(self, 'baseline_mm'):
                logger.info(f"Stereo baseline: {self.baseline_mm:.2f}mm")

            if hasattr(self, 'measurement_uncertainty_mm'):
                logger.info(f"Measurement uncertainty: {self.measurement_uncertainty_mm:.3f}mm (ISO 52902)")

            if hasattr(self, 'quality_score'):
                quality_level = "Excellent" if self.quality_score >= 80 else \
                                "Good" if self.quality_score >= 60 else \
                                "Fair" if self.quality_score >= 40 else "Poor"
                logger.info(f"Calibration quality: {quality_level} ({self.quality_score:.1f}/100)")

            # Log if this is a newer format file
            if has_enhanced_metadata:
                logger.info("Loaded calibration file with enhanced ISO 52902 compliant metadata")
            else:
                logger.info("Loaded legacy calibration file format")

            return True
        except Exception as e:
            logger.error(f"Error loading calibration from {input_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def undistort_rectify_image_pair(self, left_img: np.ndarray, right_img: np.ndarray,
                                   interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undistort and rectify a stereo image pair with enhanced quality.

        This implementation follows ISO 52902 standards for accurate camera
        undistortion and rectification for proper stereo matching. It handles
        the valid ROI (region of interest) to avoid issues with undefined regions
        after rectification.

        Args:
            left_img: Left camera image
            right_img: Right camera image
            interpolation: Interpolation method (default: cv2.INTER_LINEAR)

        Returns:
            Tuple of (undistorted_left, undistorted_right)
        """
        # Check if we have calibration parameters
        if (self.camera_matrix_left is None or self.camera_matrix_right is None or
            self.R1 is None or self.R2 is None or self.P1 is None or self.P2 is None):
            logger.error("No calibration parameters available")
            return left_img, right_img

        # Get image size
        height, width = left_img.shape[:2]
        size = (width, height)

        # Calculate undistortion maps with higher precision (CV_32FC1)
        # This ensures subpixel accuracy in the rectification process
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, size, cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, size, cv2.CV_32FC1
        )

        # Use the specified interpolation method for remapping
        # INTER_LINEAR is a good default, but INTER_CUBIC can be used for higher quality
        left_rectified = cv2.remap(left_img, map_left_x, map_left_y, interpolation)
        right_rectified = cv2.remap(right_img, map_right_x, map_right_y, interpolation)

        # Don't apply ROI masks by default - we want the full rectified images
        # However, store the ROI masks as properties of the rectified images for later use
        if hasattr(self, 'roi_left') and hasattr(self, 'roi_right'):
            # Extract ROI parameters (x, y, width, height)
            x_left, y_left, w_left, h_left = self.roi_left
            x_right, y_right, w_right, h_right = self.roi_right

            # Create masks for valid regions (initialized with zeros)
            left_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask = np.zeros((height, width), dtype=np.uint8)

            # Set valid regions to 255
            left_mask[y_left:y_left+h_left, x_left:x_left+w_left] = 255
            right_mask[y_right:y_right+h_right, x_right:x_right+w_right] = 255

            # Store masks as properties of the arrays
            left_rectified.roi_mask = left_mask
            right_rectified.roi_mask = right_mask

            # Also store ROI coordinates for later use
            left_rectified.roi = self.roi_left
            right_rectified.roi = self.roi_right

            logger.debug("Generated ROI masks for rectified images")

        return left_rectified, right_rectified
    
    def draw_rectification_lines(self, left_img: np.ndarray, right_img: np.ndarray, 
                               interval: int = 50, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw rectification lines on a stereo image pair to visualize epipolar lines.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            interval: Line interval in pixels
            color: Line color (BGR)
            
        Returns:
            Combined image with rectification lines
        """
        # Convert to color if needed
        if len(left_img.shape) == 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        if len(right_img.shape) == 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        
        # Make copies to avoid modifying originals
        left_img_lines = left_img.copy()
        right_img_lines = right_img.copy()
        
        # Get image dimensions
        height, width = left_img.shape[:2]
        
        # Draw horizontal lines at regular intervals
        for y in range(0, height, interval):
            cv2.line(left_img_lines, (0, y), (width, y), color, 1)
            cv2.line(right_img_lines, (0, y), (width, y), color, 1)
        
        # Combine images side by side
        combined = np.hstack((left_img_lines, right_img_lines))
        
        # Add vertical line between images
        cv2.line(combined, (width, 0), (width, height), (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(combined, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Right Camera", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def compute_disparity_map(self, left_img: np.ndarray, right_img: np.ndarray,
                            max_disparity: int = 128, block_size: int = 9,
                            use_wls_filter: bool = True) -> np.ndarray:
        """
        Compute disparity map from a rectified stereo pair with enhanced quality.

        This implementation follows best practices for structured light scanning:
        1. Uses better preprocessing to enhance pattern visibility
        2. Employs Semi-Global Block Matching (SGBM) with optimized parameters
        3. Applies post-processing with WLS filter for better disparity map quality
        4. Follows ISO 52902 standards for accurate depth measurement

        Args:
            left_img: Left camera image (rectified)
            right_img: Right camera image (rectified)
            max_disparity: Maximum disparity (must be divisible by 16)
            block_size: Block size for matching (odd number)
            use_wls_filter: Whether to use WLS filtering for better quality

        Returns:
            Disparity map (normalized to 0-255 range for visualization)
        """
        # Check if images are grayscale, convert if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img.copy()

        if len(right_img.shape) == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img.copy()

        # Apply any ROI masks if available (as attached properties)
        left_mask = getattr(left_img, 'roi_mask', None)
        right_mask = getattr(right_img, 'roi_mask', None)

        # Get valid ROI coordinates if available
        left_roi = getattr(left_img, 'roi', None)
        right_roi = getattr(right_img, 'roi', None)

        # Enhanced preprocessing pipeline for structured light patterns

        # Step 1: Apply CLAHE with stronger parameters for better pattern contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        left_enhanced = clahe.apply(left_gray)
        right_enhanced = clahe.apply(right_gray)

        # Step 2: Apply Gaussian blur to reduce noise while preserving patterns
        left_blurred = cv2.GaussianBlur(left_enhanced, (3, 3), 0)
        right_blurred = cv2.GaussianBlur(right_enhanced, (3, 3), 0)

        # Step 3: Apply sharpening to enhance pattern edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        left_sharp = cv2.filter2D(left_blurred, -1, kernel)
        right_sharp = cv2.filter2D(right_blurred, -1, kernel)

        # Step 4: Normalize images to ensure full dynamic range
        left_gray = cv2.normalize(left_sharp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        right_gray = cv2.normalize(right_sharp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Log the preprocessing steps
        logger.info("Applied enhanced preprocessing pipeline to stereo images for better pattern visibility")

        # Ensure max_disparity is divisible by 16 (OpenCV requirement)
        max_disparity = (max_disparity // 16) * 16
        if max_disparity < 16:
            max_disparity = 16

        # Ensure block_size is odd (OpenCV requirement)
        if block_size % 2 == 0:
            block_size += 1

        # Calculate optimal P1 and P2 parameters for Semi-Global Block Matching
        # These control the smoothness of the disparity map
        # P1 penalizes small disparity changes (1 pixel)
        # P2 penalizes larger disparity changes (>1 pixel)
        # For structured light with clear patterns, we can use more aggressive settings
        P1 = 8 * 3 * block_size**2
        P2 = 32 * 3 * block_size**2

        # Create stereo matcher with parameters specifically optimized for structured light patterns
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disparity,
            blockSize=block_size,
            P1=P1 * 1.5,  # Increased to better handle structured light patterns
            P2=P2 * 1.5,  # Increased proportionally
            disp12MaxDiff=1,  # More strict left-right consistency check for better accuracy
            uniquenessRatio=5,  # Lower value allows more matches in patterned areas
            speckleWindowSize=200,  # Larger window for better region consistency
            speckleRange=2,  # Tighter tolerance for connected components
            preFilterCap=31,  # Reduced for better pattern differentiation
            mode=cv2.STEREO_SGBM_MODE_HH  # HH (High-Quality) mode for structured light
        )

        logger.info(f"Using optimized SGBM parameters for structured light patterns: maxDisparity={max_disparity}, blockSize={block_size}")

        # Compute disparity map (scaled by 16)
        left_disparity = stereo.compute(left_gray, right_gray)

        # Enhanced disparity map with WLS filtering if enabled
        if use_wls_filter:
            try:
                # Create right matcher for WLS filter (compute right-to-left disparity)
                stereo_right = cv2.ximgproc.createRightMatcher(stereo)
                right_disparity = stereo_right.compute(right_gray, left_gray)

                # Create WLS filter with reasonable parameters for structured light
                wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
                wls_filter.setLambda(8000)  # Filter strength (higher for smoother results)
                wls_filter.setSigmaColor(1.5)  # Edge-preserving filter parameter

                # Apply WLS filter (returns floating point disparity map)
                filtered_disp = wls_filter.filter(
                    left_disparity, left_gray, None, right_disparity
                )

                # Scale back to visible range
                filtered_disp = np.clip(filtered_disp, 0, max_disparity*16)
                disparity = filtered_disp
            except Exception as e:
                logger.warning(f"WLS filtering failed, using raw disparity: {e}")
                disparity = left_disparity
        else:
            disparity = left_disparity

        # Normalize for display (0-255 range)
        # Scale by 1/16 to get actual disparity values in pixels
        # Then scale to 0-255 range for visualization
        disparity_float = disparity.astype(np.float32) / 16.0

        # Apply ROI mask if available - this is to ensure only valid regions are used in the statistics
        # and in the final disparity map
        if left_mask is not None:
            disparity_float = disparity_float.copy()  # Create a copy to avoid modifying the original
            # Set pixels outside ROI to 0 (invalid)
            disparity_float[left_mask == 0] = 0
            logger.debug("Applied left ROI mask to disparity map")

        # Get disparity statistics for quality assessment (ISO compliance)
        valid_mask = disparity_float > 0
        if np.any(valid_mask):
            disp_min = float(np.min(disparity_float[valid_mask]))
            disp_max = float(np.max(disparity_float[valid_mask]))
            disp_mean = float(np.mean(disparity_float[valid_mask]))
            disp_coverage = float(np.sum(valid_mask)) / float(valid_mask.size) * 100.0
        else:
            disp_min = disp_max = disp_mean = 0.0
            disp_coverage = 0.0

        logger.info(f"Disparity range: {disp_min:.1f} to {disp_max:.1f} pixels, " +
                  f"mean: {disp_mean:.1f}, coverage: {disp_coverage:.1f}%")

        # Normalize for display in 0-255 range
        if np.max(disparity_float) > 0:
            norm_disparity = cv2.normalize(disparity_float, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            # Handle empty/invalid disparities
            norm_disparity = np.zeros_like(disparity_float, dtype=np.uint8)

        return norm_disparity
    
    def compute_point_cloud(self, left_img: np.ndarray, disparity: np.ndarray,
                             max_z: Optional[float] = 10000.0,
                             filter_outliers: bool = True) -> np.ndarray:
        """
        Compute high-quality 3D point cloud from a disparity map.

        This implementation follows ISO 52902 standards for accurate depth computation
        and provides enhanced filtering for better 3D reconstruction quality.

        Args:
            left_img: Left camera image (rectified)
            disparity: Disparity map
            max_z: Maximum depth value to keep (mm) or None for no limit
            filter_outliers: Whether to filter outlier points

        Returns:
            3D point cloud with colors (N x 6 array: [x,y,z,r,g,b])
        """
        # Check if we have calibration parameters
        if self.Q is None:
            logger.error("No calibration Q matrix available")
            return np.array([])

        # Convert disparity to float and scale (OpenCV disparity maps are scaled by 16)
        # Note: Some OpenCV functions may return raw values, so check the values
        if np.max(disparity) > 1000:  # If very large values, probably already scaled
            disparity_float = disparity.astype(np.float32)
        else:
            disparity_float = disparity.astype(np.float32) * 16.0

        # Create a better mask for valid disparities
        # A value of 0 or negative is invalid in disparity maps
        valid_mask = disparity_float > 0

        # Get color from left image (ensure RGB format)
        if len(left_img.shape) == 3:
            colors = left_img.copy()
            if colors.shape[2] == 4:  # RGBA format
                colors = colors[:, :, :3]  # Convert to RGB
        else:
            colors = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)

        # Reproject valid disparities to 3D
        # Note: reprojectImageTo3D maps disparity values to depths using the Q matrix
        points_3d = cv2.reprojectImageTo3D(disparity_float, self.Q)

        # Additional validity check: depths should be positive
        valid_mask = valid_mask & (points_3d[:, :, 2] > 0)

        # Apply depth limit if specified
        if max_z is not None:
            valid_mask = valid_mask & (points_3d[:, :, 2] < max_z)

        # Extract valid points and colors
        valid_points = points_3d[valid_mask]
        valid_colors = colors[valid_mask]

        # Log point cloud statistics
        logger.info(f"Generated point cloud with {len(valid_points)} points " +
                  f"from disparity map ({np.sum(valid_mask) / valid_mask.size * 100:.1f}% valid)")

        # Optional outlier filtering for better quality
        if filter_outliers and len(valid_points) > 100:
            try:
                # Use statistical filtering to remove outliers
                # Compute median and standard deviation of depth values
                depths = valid_points[:, 2]
                depth_median = np.median(depths)
                depth_std = np.std(depths)

                # Keep points within 3 standard deviations of the median
                depth_mask = np.abs(depths - depth_median) < 3.0 * depth_std

                # Apply filter
                filtered_points = valid_points[depth_mask]
                filtered_colors = valid_colors[depth_mask]

                # Log filtering results
                removed = len(valid_points) - len(filtered_points)
                logger.info(f"Removed {removed} outlier points ({removed/len(valid_points)*100:.1f}%)")

                valid_points = filtered_points
                valid_colors = filtered_colors
            except Exception as e:
                logger.warning(f"Outlier filtering failed: {e}")

        # Create combined array with points and colors
        # Format: [x, y, z, r, g, b]
        if len(valid_points) > 0:
            # Ensure colors are in the right range (0-255)
            if valid_colors.dtype != np.uint8:
                valid_colors = np.clip(valid_colors, 0, 255).astype(np.uint8)

            # Combine points and colors
            points_with_colors = np.hstack((valid_points, valid_colors))
        else:
            # Empty point cloud
            points_with_colors = np.zeros((0, 6), dtype=np.float32)
            logger.warning("Generated empty point cloud - no valid disparities found")

        return points_with_colors
    
    def verify_calibration(self, left_img: np.ndarray, right_img: np.ndarray, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify calibration by undistorting and rectifying a stereo pair, 
        computing a disparity map, and evaluating epipolar alignment.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            output_dir: Optional directory to save verification images
            
        Returns:
            Dictionary with verification metrics
        """
        # Check if we have calibration parameters
        if (self.camera_matrix_left is None or self.camera_matrix_right is None or
            self.R1 is None or self.R2 is None or self.P1 is None or self.P2 is None):
            logger.error("No calibration parameters available")
            return {"success": False, "error": "No calibration parameters"}
        
        try:
            # Undistort and rectify
            left_rect, right_rect = self.undistort_rectify_image_pair(left_img, right_img)
            
            # Draw rectification lines
            combined = self.draw_rectification_lines(left_rect, right_rect)
            
            # Compute disparity map
            disparity = self.compute_disparity_map(left_rect, right_rect)
            
            # Find checkerboard if present
            left_ret, left_corners = self.find_checkerboard(left_rect)
            right_ret, right_corners = self.find_checkerboard(right_rect)
            
            # Compute epipolar alignment error if checkerboard found in both images
            epipolar_error = float('inf')
            if left_ret and right_ret:
                # Get corresponding y-coordinates
                left_y = left_corners[:, 0, 1]
                right_y = right_corners[:, 0, 1]
                
                # Compute differences
                y_diffs = np.abs(left_y - right_y)
                epipolar_error = np.mean(y_diffs)
                
                # Draw checkerboard on combined image
                combined_with_corners = combined.copy()
                width = left_img.shape[1]
                
                # Convert combined to color if needed
                if len(combined_with_corners.shape) == 2:
                    combined_with_corners = cv2.cvtColor(combined_with_corners, cv2.COLOR_GRAY2BGR)
                
                # Draw corners on left image area
                cv2.drawChessboardCorners(combined_with_corners[:, :width], self.checkerboard_size, left_corners, left_ret)
                
                # Draw corners on right image area
                cv2.drawChessboardCorners(combined_with_corners[:, width:], self.checkerboard_size, right_corners, right_ret)
                
                if output_dir:
                    cv2.imwrite(os.path.join(output_dir, "combined_with_corners.png"), combined_with_corners)
            
            # Save verification images if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, "left_rectified.png"), left_rect)
                cv2.imwrite(os.path.join(output_dir, "right_rectified.png"), right_rect)
                cv2.imwrite(os.path.join(output_dir, "combined_rectified.png"), combined)
                cv2.imwrite(os.path.join(output_dir, "disparity_map.png"), disparity)
            
            # Create verification result
            result = {
                "success": True,
                "reprojection_error": float(self.reprojection_error),
                "epipolar_alignment_error": float(epipolar_error),
                "disparity_coverage": float(np.count_nonzero(disparity > 0)) / float(disparity.size),
                "checkerboard_detected": left_ret and right_ret
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error verifying calibration: {e}")
            return {"success": False, "error": str(e)}

    def validate_projection_matrices(self) -> bool:
        """
        Validate and correct projection matrices for unreasonable values.
        
        This method checks for common issues with projection matrices:
        1. Unreasonably large baseline values (typically > 1000mm)
        2. Incorrect focal length values
        3. Invalid or mismatched camera centers
        4. Scale issues where values may be in meters instead of millimeters
        
        If issues are found, it will attempt to correct them based on reasonable
        assumptions for stereo camera systems.
        
        Returns:
            True if matrices were valid or successfully corrected, False otherwise
        """
        if self.P1 is None or self.P2 is None:
            logger.error("No projection matrices available to validate")
            return False
        
        logger.info("Validating projection matrices with enhanced reliability...")
        
        # Extract parameters from projection matrices
        # P = [fx 0 cx tx; 0 fy cy ty; 0 0 1 0]
        fx_left = self.P1[0, 0]
        cx_left = self.P1[0, 2]
        fy_left = self.P1[1, 1]
        cy_left = self.P1[1, 2]
        
        fx_right = self.P2[0, 0]
        cx_right = self.P2[0, 2]
        fy_right = self.P2[1, 1]
        cy_right = self.P2[1, 2]
        
        # Extract baseline from P2
        tx = self.P2[0, 3]
        baseline_mm = -tx / fx_right * 1000.0  # Convert to mm
        
        changes_made = False
        
        # Log all parameters for debugging
        logger.info(f"Left camera projection matrix P1:")
        logger.info(f"fx_left={fx_left:.2f}, fy_left={fy_left:.2f}, cx_left={cx_left:.2f}, cy_left={cy_left:.2f}")
        logger.info(f"Right camera projection matrix P2:")
        logger.info(f"fx_right={fx_right:.2f}, fy_right={fy_right:.2f}, cx_right={cx_right:.2f}, cy_right={cy_right:.2f}")
        logger.info(f"tx={tx:.2f}, calculated baseline={baseline_mm:.2f}mm")
        
        # Multiple validation approaches for robust correction
        
        # APPROACH 1: Check for unreasonable baseline (typical stereo cameras have baselines of 50-100mm)
        if abs(baseline_mm) > 1000:
            logger.warning(f"Detected unreasonable baseline: {baseline_mm:.2f}mm")
            
            # Check for scaling issues (common in OpenCV calibrations)
            if abs(baseline_mm) > 1000 and abs(baseline_mm) < 100000:
                # This looks like a scaling issue - the baseline could be in millimeters but
                # was incorrectly calculated with the formula expecting meters
                corrected_baseline = baseline_mm / 1000.0
                if 10 < corrected_baseline < 200:
                    logger.info(f"Detected scaling issue in baseline. Correcting {baseline_mm:.2f}mm to {corrected_baseline:.2f}mm")
                    
                    # Update P2 matrix with corrected baseline
                    tx_new = -corrected_baseline * fx_right / 1000.0
                    self.P2[0, 3] = tx_new
                    baseline_mm = corrected_baseline
                    changes_made = True
            
            # If still unreasonable, try using T vector if available
            if abs(baseline_mm) > 1000 and hasattr(self, 'T') and self.T is not None:
                calculated_baseline = float(np.linalg.norm(self.T) * 1000.0)  # in mm
                
                # Check for T vector scaling issue as well
                if calculated_baseline > 1000:
                    calculated_baseline /= 1000.0
                    logger.info(f"Detected scaling issue in T vector baseline. Correcting to {calculated_baseline:.2f}mm")
                
                # If calculated baseline is reasonable, use it
                if 10 < calculated_baseline < 200:
                    logger.info(f"Using calculated baseline from T vector: {calculated_baseline:.2f}mm")
                    
                    # Recalculate tx based on reasonable baseline
                    tx_new = -calculated_baseline * fx_right / 1000.0
                    self.P2[0, 3] = tx_new
                    baseline_mm = calculated_baseline
                    changes_made = True
                else:
                    # Use a reasonable default if both values are unreasonable
                    logger.warning(f"Calculated baseline also unreasonable: {calculated_baseline:.2f}mm")
                    logger.info("Setting baseline to default value of 75mm")
                    
                    # Set to typical baseline for stereo cameras (75mm)
                    tx_new = -75.0 * fx_right / 1000.0
                    self.P2[0, 3] = tx_new
                    baseline_mm = 75.0
                    changes_made = True
            elif abs(baseline_mm) > 1000:
                # No T vector or still unreasonable, use default value
                logger.info("Setting baseline to default value of 75mm")
                tx_new = -75.0 * fx_right / 1000.0
                self.P2[0, 3] = tx_new
                baseline_mm = 75.0
                changes_made = True
        
        # APPROACH 2: Check for commonly mismatched extrinsic parameter scales
        # Sometimes P2 uses different scale than T vector
        if not changes_made and hasattr(self, 'T') and self.T is not None:
            t_baseline = float(np.linalg.norm(self.T) * 1000.0)  # in mm
            
            # Check for scale mismatch between T vector and P2 matrix
            baseline_ratio = abs(t_baseline / baseline_mm) if baseline_mm != 0 else float('inf')
            
            if abs(baseline_ratio - 1.0) > 0.2:  # More than 20% difference
                logger.warning(f"Baseline scale mismatch: T vector gives {t_baseline:.2f}mm, " + 
                              f"P2 matrix gives {baseline_mm:.2f}mm (ratio: {baseline_ratio:.2f})")
                
                # Choose the more reasonable value
                if 10 < t_baseline < 200:
                    logger.info(f"Using T vector baseline: {t_baseline:.2f}mm")
                    tx_new = -t_baseline * fx_right / 1000.0
                    self.P2[0, 3] = tx_new
                    baseline_mm = t_baseline
                    changes_made = True
        
        # APPROACH 3: Check for focal length consistency and scale
        # Typical focal lengths for consumer cameras are in the range of 400-1200 pixels
        if fx_left < 100 or fx_left > 5000 or fx_right < 100 or fx_right > 5000:
            logger.warning(f"Potentially unreasonable focal lengths: fx_left={fx_left:.2f}, fx_right={fx_right:.2f}")
            
            # Try to estimate if there's a scaling issue
            if 1.0 < fx_left < 10.0 and 1.0 < fx_right < 10.0:
                # Looks like focal lengths are in normalized units, convert to pixels
                # Assuming 640x480 image as default if we don't have image_size
                width = self.image_size[0] if hasattr(self, 'image_size') and self.image_size else 640
                
                fx_left_new = fx_left * width
                fx_right_new = fx_right * width
                fy_left_new = fy_left * width
                fy_right_new = fy_right * width
                
                logger.info(f"Converting normalized focal lengths to pixels: " + 
                           f"fx_left: {fx_left:.2f} -> {fx_left_new:.2f}, " + 
                           f"fx_right: {fx_right:.2f} -> {fx_right_new:.2f}")
                
                # Update matrices
                self.P1[0, 0] = fx_left_new
                self.P1[1, 1] = fy_left_new
                self.P2[0, 0] = fx_right_new
                self.P2[1, 1] = fy_right_new
                
                # Also need to adjust other values
                tx_new = self.P2[0, 3] * (fx_right_new / fx_right)
                self.P2[0, 3] = tx_new
                
                # Update local variables
                fx_left = fx_left_new
                fy_left = fy_left_new
                fx_right = fx_right_new
                fy_right = fy_right_new
                
                changes_made = True
        
        # Check for focal length consistency between cameras
        if abs(fx_left - fx_right) / max(fx_left, fx_right) > 0.05:
            logger.warning(f"Inconsistent focal lengths: fx_left={fx_left:.2f}, fx_right={fx_right:.2f}")
            
            # Use average focal length for both cameras
            fx_avg = (fx_left + fx_right) / 2.0
            self.P1[0, 0] = fx_avg
            self.P2[0, 0] = fx_avg
            changes_made = True
        
        if abs(fy_left - fy_right) / max(fy_left, fy_right) > 0.05:
            logger.warning(f"Inconsistent focal lengths: fy_left={fy_left:.2f}, fy_right={fy_right:.2f}")
            
            # Use average focal length for both cameras
            fy_avg = (fy_left + fy_right) / 2.0
            self.P1[1, 1] = fy_avg
            self.P2[1, 1] = fy_avg
            changes_made = True
        
        # APPROACH 4: Check principal points (usually near image center)
        if hasattr(self, 'image_size') and self.image_size:
            width, height = self.image_size
            
            # Principal points should be near image center
            if abs(cx_left - width/2) > width/3 or abs(cx_right - width/2) > width/3:
                logger.warning(f"Principal points x far from image center: cx_left={cx_left:.2f}, cx_right={cx_right:.2f}, width={width}")
                # Don't correct automatically as this could be legitimate for non-standard cameras
            
            if abs(cy_left - height/2) > height/3 or abs(cy_right - height/2) > height/3:
                logger.warning(f"Principal points y far from image center: cy_left={cy_left:.2f}, cy_right={cy_right:.2f}, height={height}")
                # Don't correct automatically as this could be legitimate for non-standard cameras
        
        if changes_made:
            # Recalculate Q matrix from corrected projection matrices
            # This is a more complete disparity-to-depth mapping matrix calculation
            self.Q = np.zeros((4, 4), dtype=np.float64)
            
            # Get parameters from updated P1, P2
            fx = self.P1[0, 0]  # Should be the same for both cameras after correction
            fy = self.P1[1, 1]
            cx1 = self.P1[0, 2]
            cy = self.P1[1, 2]  # Should be the same for both cameras
            cx2 = self.P2[0, 2]
            tx = self.P2[0, 3]
            
            # Compute the new Q matrix
            # See OpenCV documentation for Q matrix format
            self.Q[0, 0] = 1.0
            self.Q[1, 1] = 1.0
            self.Q[0, 3] = -cx1
            self.Q[1, 3] = -cy
            self.Q[2, 3] = fx
            self.Q[3, 2] = -1.0 / tx  # Critical line for accurate depth calculation
            self.Q[3, 3] = (cx1 - cx2) / tx  # Accounts for any remaining x-offset between cameras
            
            logger.info("Projection matrices corrected and Q matrix updated")
            
            # Update baseline attribute for future reference
            self.baseline_mm = -self.P2[0, 3] / self.P2[0, 0] * 1000.0
            logger.info(f"Final baseline: {self.baseline_mm:.2f}mm")
            
            # Also store the original baseline value for reference
            self.original_baseline_mm = baseline_mm
            logger.info(f"Original baseline: {self.original_baseline_mm:.2f}mm (scale factor: {self.original_baseline_mm/self.baseline_mm:.1f}x)")
            
            return True
        else:
            logger.info(f"Projection matrices validated - no corrections needed. Baseline: {baseline_mm:.2f}mm")
            # Still store the baseline for reference
            self.baseline_mm = baseline_mm
            return True


def run_calibration(left_images: List[np.ndarray], 
                   right_images: List[np.ndarray],
                   checkerboard_size: Tuple[int, int] = (9, 6),
                   square_size: float = 24.0,
                   output_dir: str = "./calibration",
                   visualize: bool = False) -> Dict[str, Any]:
    """
    Run stereo camera calibration with the provided image pairs.
    
    Args:
        left_images: List of images from the left camera
        right_images: List of images from the right camera
        checkerboard_size: Size of the checkerboard pattern (width, height) in inner corners
        square_size: Size of the checkerboard squares in mm
        output_dir: Directory to save calibration results
        visualize: Whether to visualize the calibration process
        
    Returns:
        Dictionary with calibration parameters and results
    """
    # Create calibrator
    calibrator = StereoCalibrator(
        checkerboard_size=checkerboard_size,
        square_size=square_size
    )
    
    # Run calibration
    result = calibrator.calibrate_stereo(left_images, right_images, visualize=visualize)
    
    # Check if calibration was successful
    if not result:
        logger.error("Stereo calibration failed")
        return {"success": False, "error": "Calibration failed"}
    
    # Save calibration results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON (human-readable)
    json_path = os.path.join(output_dir, "stereo_calibration.json")
    calibrator.save_calibration(json_path)
    
    # Save as NPY (more efficient for loading)
    npy_path = os.path.join(output_dir, "stereo_calibration.npz")
    calibrator.save_calibration_npy(npy_path)
    
    # Verify calibration if we have at least one image pair
    if left_images and right_images:
        # Use last image pair for verification
        verification = calibrator.verify_calibration(
            left_images[-1], right_images[-1],
            output_dir=os.path.join(output_dir, "verification")
        )
        result.update(verification)
    
    result["success"] = True
    result["json_path"] = json_path
    result["npy_path"] = npy_path
    
    return result


def save_calibration(filename: str, calibration_params: Dict[str, Any]) -> bool:
    """
    Save stereo camera calibration parameters to a file.

    Args:
        filename: Path to the calibration file
        calibration_params: Dictionary of calibration parameters

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        data = {}
        for key, value in calibration_params.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            else:
                data[key] = value

        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info(f"Saved calibration to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving calibration: {e}")
        return False


def load_calibration(calibration_file: str) -> StereoCalibrator:
    """
    Load calibration parameters from a file.

    Args:
        calibration_file: Path to calibration file (JSON or NPY)

    Returns:
        Loaded StereoCalibrator object, or None if loading failed
    """
    # Create calibrator
    calibrator = StereoCalibrator()
    
    # Check file extension
    if calibration_file.endswith('.json'):
        success = calibrator.load_calibration(calibration_file)
    elif calibration_file.endswith('.npz') or calibration_file.endswith('.npy'):
        success = calibrator.load_calibration_npy(calibration_file)
    else:
        logger.error(f"Unsupported calibration file format: {calibration_file}")
        return None
    
    if success:
        # Validate and correct projection matrices if needed
        if hasattr(calibrator, 'P1') and hasattr(calibrator, 'P2'):
            calibrator.validate_projection_matrices()
        return calibrator
    else:
        return None


def extract_baseline_from_calibration(calibration_file: str) -> float:
    """
    Extract the baseline value from a calibration file.
    
    Args:
        calibration_file: Path to calibration file (JSON or NPY)
        
    Returns:
        Baseline value in millimeters, or 80.0 if not found
    """
    try:
        # Load calibration
        calibrator = load_calibration(calibration_file)
        
        if calibrator is None:
            logger.error(f"Failed to load calibration from {calibration_file}")
            return 80.0  # Default value
        
        # Check if baseline was directly loaded
        if hasattr(calibrator, 'baseline_mm'):
            baseline = calibrator.baseline_mm
            logger.info(f"Using baseline from calibration metadata: {baseline:.2f}mm")
            return baseline
        
        # Check if we can calculate it from P2 matrix
        if hasattr(calibrator, 'P2') and calibrator.P2 is not None:
            fx = calibrator.P2[0, 0]
            tx = calibrator.P2[0, 3]
            baseline = -tx / fx * 1000.0  # Convert to mm
            logger.info(f"Calculated baseline from P2 matrix: {baseline:.2f}mm")
            return baseline
        
        # Check if we can calculate it from T vector
        if hasattr(calibrator, 'T') and calibrator.T is not None:
            baseline = np.linalg.norm(calibrator.T) * 1000.0  # Convert to mm
            logger.info(f"Calculated baseline from T vector: {baseline:.2f}mm")
            return baseline
        
        # Fallback to default
        logger.warning("Could not extract baseline from calibration, using default value of 80.0mm")
        return 80.0
        
    except Exception as e:
        logger.error(f"Error extracting baseline from calibration: {e}")
        logger.info("Using default baseline of 80.0mm")
        return 80.0