"""
Stereo rectification module for fixing distorted point clouds.

This module applies stereo rectification to align epipolar lines horizontally,
which is essential for accurate stereo matching and 3D reconstruction.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class StereoRectifier:
    """
    Stereo camera rectification for correcting point cloud distortions.
    
    Applies stereo rectification using calibration data to ensure epipolar
    lines are horizontal, which is crucial for accurate correspondence matching.
    """
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize stereo rectifier.
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_data = None
        self.rectify_maps_left = None
        self.rectify_maps_right = None
        self.is_initialized = False
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, calibration_file: str) -> bool:
        """
        Load calibration data from JSON file.
        
        Args:
            calibration_file: Path to calibration JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            # Validate required parameters
            required_keys = ['R1', 'R2', 'P1', 'P2', 'camera_matrix_left', 
                           'camera_matrix_right', 'dist_coeffs_left', 'dist_coeffs_right',
                           'image_size']
            
            for key in required_keys:
                if key not in self.calibration_data:
                    logger.error(f"Missing calibration parameter: {key}")
                    return False
            
            # Convert to numpy arrays
            self.R1 = np.array(self.calibration_data['R1'])
            self.R2 = np.array(self.calibration_data['R2'])
            self.P1 = np.array(self.calibration_data['P1'])
            self.P2 = np.array(self.calibration_data['P2'])
            self.camera_matrix_left = np.array(self.calibration_data['camera_matrix_left'])
            self.camera_matrix_right = np.array(self.calibration_data['camera_matrix_right'])
            self.dist_coeffs_left = np.array(self.calibration_data['dist_coeffs_left'])
            self.dist_coeffs_right = np.array(self.calibration_data['dist_coeffs_right'])
            self.image_size = tuple(self.calibration_data['image_size'])
            
            # Initialize rectification maps
            self._initialize_rectification_maps()
            
            logger.info(f"Calibration loaded from {calibration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def _initialize_rectification_maps(self):
        """Initialize rectification maps for fast rectification."""
        try:
            # Create rectification maps for both cameras
            self.rectify_maps_left = cv2.initUndistortRectifyMap(
                self.camera_matrix_left, self.dist_coeffs_left,
                self.R1, self.P1, self.image_size, cv2.CV_32FC1
            )
            
            self.rectify_maps_right = cv2.initUndistortRectifyMap(
                self.camera_matrix_right, self.dist_coeffs_right,
                self.R2, self.P2, self.image_size, cv2.CV_32FC1
            )
            
            self.is_initialized = True
            logger.info(f"Rectification maps initialized for size {self.image_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize rectification maps: {e}")
            self.is_initialized = False
    
    def _scale_calibration_to_image_size(self, actual_image_size):
        """Scale calibration parameters to match actual image dimensions."""
        actual_w, actual_h = actual_image_size
        calib_w, calib_h = self.image_size
        
        if actual_w == calib_w and actual_h == calib_h:
            return  # No scaling needed
        
        # Calculate scaling factors
        scale_x = actual_w / calib_w
        scale_y = actual_h / calib_h
        
        logger.info(f"Scaling calibration from {self.image_size} to {actual_image_size}, scale: {scale_x:.3f}x{scale_y:.3f}")
        
        # Important: Store original calibration parameters if not already done
        if not hasattr(self, '_original_calibration_stored'):
            self._original_K1 = self.camera_matrix_left.copy()
            self._original_K2 = self.camera_matrix_right.copy()
            self._original_P1 = self.P1.copy()
            self._original_P2 = self.P2.copy()
            self._original_image_size = self.image_size
            self._original_calibration_stored = True
        
        # Scale camera matrices
        self.camera_matrix_left = self.camera_matrix_left.copy()
        self.camera_matrix_right = self.camera_matrix_right.copy()
        
        # Scale focal lengths and principal points
        self.camera_matrix_left[0, 0] *= scale_x  # fx_left
        self.camera_matrix_left[1, 1] *= scale_y  # fy_left 
        self.camera_matrix_left[0, 2] *= scale_x  # cx_left
        self.camera_matrix_left[1, 2] *= scale_y  # cy_left
        
        self.camera_matrix_right[0, 0] *= scale_x  # fx_right
        self.camera_matrix_right[1, 1] *= scale_y  # fy_right
        self.camera_matrix_right[0, 2] *= scale_x  # cx_right
        self.camera_matrix_right[1, 2] *= scale_y  # cy_right
        
        # Scale projection matrices
        self.P1 = self.P1.copy()
        self.P2 = self.P2.copy()
        
        self.P1[0, 0] *= scale_x  # fx
        self.P1[1, 1] *= scale_y  # fy
        self.P1[0, 2] *= scale_x  # cx
        self.P1[1, 2] *= scale_y  # cy
        
        self.P2[0, 0] *= scale_x  # fx  
        self.P2[1, 1] *= scale_y  # fy
        self.P2[0, 2] *= scale_x  # cx
        self.P2[1, 2] *= scale_y  # cy
        self.P2[0, 3] *= scale_x  # baseline adjustment
        
        # Update image size
        self.image_size = (actual_w, actual_h)
    
    def rectify_images(self, left_image: np.ndarray, right_image: np.ndarray, 
                      use_opencl: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            use_opencl: Use OpenCL GPU acceleration if available
            
        Returns:
            Tuple of (rectified_left, rectified_right) images
        """
        if not self.is_initialized:
            logger.error("Rectifier not initialized")
            return left_image, right_image
        
        try:
            # Check if we need to scale calibration parameters to match actual image size
            actual_h, actual_w = left_image.shape[:2]
            actual_image_size = (actual_w, actual_h)
            
            if actual_image_size != self.image_size:
                logger.info(f"Image size mismatch: calibration {self.image_size} vs actual {actual_image_size}")
                self._scale_calibration_to_image_size(actual_image_size)
                # Regenerate rectification maps with scaled parameters
                self._initialize_rectification_maps()
                logger.info("Rectification maps regenerated with scaled calibration")
            # Apply rectification using precomputed maps
            if use_opencl and cv2.ocl.haveOpenCL():
                # GPU-accelerated rectification with OpenCL
                try:
                    left_gpu = cv2.UMat(left_image)
                    right_gpu = cv2.UMat(right_image)
                    
                    # Convert maps to UMat for GPU processing
                    map1_left_gpu = cv2.UMat(self.rectify_maps_left[0])
                    map2_left_gpu = cv2.UMat(self.rectify_maps_left[1])
                    map1_right_gpu = cv2.UMat(self.rectify_maps_right[0])
                    map2_right_gpu = cv2.UMat(self.rectify_maps_right[1])
                    
                    rectified_left_gpu = cv2.remap(left_gpu, map1_left_gpu, map2_left_gpu, cv2.INTER_LINEAR)
                    rectified_right_gpu = cv2.remap(right_gpu, map1_right_gpu, map2_right_gpu, cv2.INTER_LINEAR)
                    
                    rectified_left = rectified_left_gpu.get()
                    rectified_right = rectified_right_gpu.get()
                    
                    logger.debug("OpenCL GPU rectification completed")
                except Exception as opencl_error:
                    logger.warning(f"OpenCL rectification failed: {opencl_error}, falling back to CPU")
                    # Fallback to CPU
                    rectified_left = cv2.remap(left_image, *self.rectify_maps_left, cv2.INTER_LINEAR)
                    rectified_right = cv2.remap(right_image, *self.rectify_maps_right, cv2.INTER_LINEAR)
                    logger.debug("CPU rectification completed (fallback)")
            else:
                # CPU rectification
                rectified_left = cv2.remap(left_image, *self.rectify_maps_left, cv2.INTER_LINEAR)
                rectified_right = cv2.remap(right_image, *self.rectify_maps_right, cv2.INTER_LINEAR)
                
                logger.debug("CPU rectification completed")
            
            # Verify rectification quality
            if self._verify_rectification(rectified_left, rectified_right):
                logger.debug("Rectification quality check passed")
            else:
                logger.warning("Rectification quality check failed - epipolar lines may not be aligned")
            
            return rectified_left, rectified_right
            
        except Exception as e:
            logger.error(f"Rectification failed: {e}")
            return left_image, right_image
    
    def rectify_image_batch(self, image_pairs: list, use_opencl: bool = True) -> list:
        """
        Rectify a batch of stereo image pairs.
        
        Args:
            image_pairs: List of (left_image, right_image) tuples
            use_opencl: Use OpenCL GPU acceleration if available
            
        Returns:
            List of (rectified_left, rectified_right) tuples
        """
        if not self.is_initialized:
            logger.error("Rectifier not initialized")
            return image_pairs
        
        rectified_pairs = []
        
        for i, (left_img, right_img) in enumerate(image_pairs):
            try:
                rect_left, rect_right = self.rectify_images(left_img, right_img, use_opencl)
                rectified_pairs.append((rect_left, rect_right))
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Rectified {i + 1}/{len(image_pairs)} image pairs")
                    
            except Exception as e:
                logger.error(f"Failed to rectify image pair {i}: {e}")
                rectified_pairs.append((left_img, right_img))  # Fallback to original
        
        logger.info(f"Batch rectification completed: {len(rectified_pairs)} pairs processed")
        return rectified_pairs
    
    def get_rectified_calibration(self) -> Dict[str, Any]:
        """
        Get rectified calibration data for triangulation.
        
        Returns:
            Dictionary with rectified calibration parameters
        """
        if not self.is_initialized:
            logger.error("Rectifier not initialized")
            return {}
        
        return {
            'P1': self.P1.tolist(),
            'P2': self.P2.tolist(),
            'R1': self.R1.tolist(),
            'R2': self.R2.tolist(),
            'image_size': self.image_size,
            'rectified': True,
            'baseline_mm': self.calibration_data.get('baseline_mm', 79.5)
        }
    
    def _verify_rectification(self, left_rect: np.ndarray, right_rect: np.ndarray) -> bool:
        """Verify that rectification produces horizontally aligned epipolar lines."""
        try:
            # Convert to grayscale if needed
            if len(left_rect.shape) == 3:
                left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_rect
                right_gray = right_rect
                
            # Use ORB detector (patent-free alternative to SIFT)
            detector = cv2.ORB_create(nfeatures=100)
            kp_left, desc_left = detector.detectAndCompute(left_gray, None)
            kp_right, desc_right = detector.detectAndCompute(right_gray, None)
            
            if desc_left is None or desc_right is None:
                return True  # Can't verify, assume OK
                
            # Match features
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc_left, desc_right)
            
            if len(matches) < 10:
                return True  # Not enough matches to verify
                
            # Check if matched points have similar y-coordinates (should be on same epipolar line)
            y_differences = []
            for match in matches[:20]:  # Check first 20 matches
                pt_left = kp_left[match.queryIdx].pt
                pt_right = kp_right[match.trainIdx].pt
                y_diff = abs(pt_left[1] - pt_right[1])
                y_differences.append(y_diff)
                
            mean_y_diff = np.mean(y_differences)
            
            # Epipolar lines should be horizontal, so y-difference should be small
            if mean_y_diff > 5.0:  # More than 5 pixels average difference
                logger.warning(f"Rectification issue: mean y-difference is {mean_y_diff:.1f} pixels")
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Could not verify rectification: {e}")
            return True  # Assume OK if verification fails

def rectify_captured_session(session_dir: str, calibration_file: str = None, 
                           output_suffix: str = "_rectified") -> bool:
    """
    Rectify all images in a captured session directory.
    
    Args:
        session_dir: Path to captured session directory
        calibration_file: Path to calibration file (defaults to session calibration)
        output_suffix: Suffix to add to rectified image filenames
        
    Returns:
        True if successful, False otherwise
    """
    session_path = Path(session_dir)
    
    # Use session calibration if not specified
    if calibration_file is None:
        calibration_file = session_path / "calibration.json"
    
    # Initialize rectifier
    rectifier = StereoRectifier(calibration_file)
    if not rectifier.is_initialized:
        logger.error("Failed to initialize rectifier")
        return False
    
    # Find all image pairs
    left_images = sorted(session_path.glob("left_*.jpg"))
    right_images = sorted(session_path.glob("right_*.jpg"))
    
    if len(left_images) != len(right_images):
        logger.error(f"Mismatched image counts: {len(left_images)} left, {len(right_images)} right")
        return False
    
    logger.info(f"Rectifying {len(left_images)} image pairs...")
    
    # Process each image pair
    for left_path, right_path in zip(left_images, right_images):
        try:
            # Load images
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))
            
            if left_img is None or right_img is None:
                logger.error(f"Failed to load images: {left_path}, {right_path}")
                continue
            
            # Rectify
            rect_left, rect_right = rectifier.rectify_images(left_img, right_img)
            
            # Save rectified images
            left_output = session_path / f"{left_path.stem}{output_suffix}.jpg"
            right_output = session_path / f"{right_path.stem}{output_suffix}.jpg"
            
            cv2.imwrite(str(left_output), rect_left)
            cv2.imwrite(str(right_output), rect_right)
            
            logger.debug(f"Rectified: {left_path.name} -> {left_output.name}")
            
        except Exception as e:
            logger.error(f"Failed to rectify {left_path.name}: {e}")
            return False
    
    # Save rectified calibration data
    rect_calib = rectifier.get_rectified_calibration()
    rect_calib_file = session_path / "calibration_rectified.json"
    
    try:
        with open(rect_calib_file, 'w') as f:
            json.dump(rect_calib, f, indent=4)
        logger.info(f"Rectified calibration saved to {rect_calib_file}")
    except Exception as e:
        logger.error(f"Failed to save rectified calibration: {e}")
    
    logger.info(f"Session rectification completed: {session_dir}")
    return True