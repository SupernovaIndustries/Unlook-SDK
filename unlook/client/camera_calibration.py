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
        Initialize the stereo calibrator.
        
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
        
        # Camera parameters
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.reprojection_error = None
        
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
        
        logger.info(f"Stereo calibrator initialized with checkerboard size {checkerboard_size}")
    
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
        
        # Calculate rectification transforms
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.image_size, self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )
        
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
            "num_valid_pairs": valid_pairs
        }
        
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
        Save calibration parameters to a NumPy file (for compatibility with other tools).
        
        Args:
            output_file: Output file path (.npy)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if we have calibration parameters
        if self.camera_matrix_left is None or self.camera_matrix_right is None:
            logger.error("No calibration parameters to save")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save to file
        try:
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
                Q=self.Q
            )
            logger.info(f"Calibration saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration to {output_file}: {e}")
            return False
    
    def load_calibration_npy(self, input_file: str) -> bool:
        """
        Load calibration parameters from a NumPy file.
        
        Args:
            input_file: Input file path (.npy)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = np.load(input_file)
            
            # Load parameters
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
            
            logger.info(f"Calibration loaded from {input_file}")
            logger.info(f"Reprojection error: {self.reprojection_error:.6f}")
            return True
        except Exception as e:
            logger.error(f"Error loading calibration from {input_file}: {e}")
            return False
    
    def undistort_rectify_image_pair(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undistort and rectify a stereo image pair.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            
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
        
        # Calculate undistortion maps
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, size, cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, size, cv2.CV_32FC1
        )
        
        # Undistort and rectify
        left_rectified = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
        
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
                            max_disparity: int = 128, block_size: int = 15) -> np.ndarray:
        """
        Compute disparity map from a rectified stereo pair.
        
        Args:
            left_img: Left camera image (rectified)
            right_img: Right camera image (rectified)
            max_disparity: Maximum disparity (must be divisible by 16)
            block_size: Block size for matching (odd number)
            
        Returns:
            Disparity map
        """
        # Check if images are grayscale
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img.copy()
        
        if len(right_img.shape) == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img.copy()
        
        # Ensure max_disparity is divisible by 16
        max_disparity = (max_disparity // 16) * 16
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disparity,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray)
        
        # Normalize for display
        norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return norm_disparity
    
    def compute_point_cloud(self, left_img: np.ndarray, disparity: np.ndarray) -> np.ndarray:
        """
        Compute 3D point cloud from a disparity map.
        
        Args:
            left_img: Left camera image (rectified)
            disparity: Disparity map
            
        Returns:
            3D point cloud (N x 3 array)
        """
        # Check if we have calibration parameters
        if self.Q is None:
            logger.error("No calibration Q matrix available")
            return np.array([])
        
        # Convert disparity to float and scale
        disparity_float = disparity.astype(np.float32) / 16.0
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity_float, self.Q)
        
        # Get color from left image
        if len(left_img.shape) == 3:
            colors = left_img
        else:
            colors = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        
        # Create mask for valid points (disparity > 0)
        mask = disparity > 0
        
        # Filter points
        points = points_3d[mask]
        colors = colors[mask]
        
        # Combine points and colors
        points_with_colors = np.hstack((points, colors))
        
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
        return calibrator
    else:
        return None