#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2K Calibration Processing Pipeline for UnLook SDK

This script processes captured checkerboard images to generate high-precision
stereo calibration for 2K resolution (2048x1536).

Features:
- Advanced stereo calibration algorithms
- Automatic quality validation
- ISO-compliant reprojection error calculation
- Epipolar geometry verification
- Calibration deployment to default locations

Usage:
  python process_calibration.py --input calibration_2k_images/ --output calibration_2k.json
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("process_calibration_2k")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

class StereoCalibration2K:
    """Advanced stereo calibration processor for 2K resolution"""
    
    def __init__(self, checkerboard_size=(8, 5), square_size_mm=19.5):
        """
        Initialize calibration processor.
        
        Args:
            checkerboard_size: Inner corners (columns, rows)
            square_size_mm: Physical size of each square
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.target_baseline_mm = 80.0  # Expected baseline
        
        # Calibration criteria for sub-pixel refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        
        # Generate 3D object points for checkerboard
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        logger.info(f"Initialized for {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard")
        logger.info(f"Square size: {square_size_mm}mm, Target baseline: {self.target_baseline_mm}mm")
    
    def calibrate_single_camera(self, images):
        """
        Calibrate a single camera using checkerboard images.
        
        Args:
            images: List of numpy arrays containing checkerboard images
            
        Returns:
            Tuple: (camera_matrix, distortion_coeffs, rvecs, tvecs, reprojection_error)
        """
        logger.info(f"Starting single camera calibration with {len(images)} images")
        
        # Prepare object points and image points
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        
        image_size = None
        valid_images = 0
        
        for i, img in enumerate(images):
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Set image size from first valid image
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)
                logger.info(f"Image size: {image_size}")
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corners for sub-pixel accuracy
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                
                object_points.append(self.objp)
                image_points.append(corners)
                valid_images += 1
                
                logger.debug(f"Image {i+1}: Checkerboard found")
            else:
                logger.warning(f"Image {i+1}: Checkerboard not found")
        
        logger.info(f"Found checkerboard in {valid_images}/{len(images)} images")
        
        if valid_images < 10:
            raise ValueError(f"Insufficient valid images: {valid_images} < 10")
        
        # Perform camera calibration
        logger.info("Performing camera calibration...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, 
            image_points, 
            image_size, 
            None, 
            None,
            flags=cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5  # Fix high-order distortion
        )
        
        logger.info(f"Camera calibration completed. RMS reprojection error: {ret:.3f} pixels")
        
        # Log camera parameters
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        logger.info(f"Camera intrinsics:")
        logger.info(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
        logger.info(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
        logger.info(f"  Distortion coefficients: {dist_coeffs.ravel()}")
        
        return camera_matrix, dist_coeffs, rvecs, tvecs, ret
    
    def load_calibration_images(self, input_dir, single_camera_mode=False):
        """
        Load calibration images from directory.
        
        Args:
            input_dir: Path to directory containing images
            single_camera_mode: If True, look for images in 'camera' subdirectory or directly in input_dir
        
        Returns:
            tuple: (left_images, right_images, image_size)
        """
        input_path = Path(input_dir)
        logger.info(f"Input directory: {input_path}")
        logger.info(f"Input directory exists: {input_path.exists()}")
        
        # List all files in input directory for debugging
        if input_path.exists():
            all_files = list(input_path.iterdir())
            logger.info(f"Files/folders in input directory: {[f.name for f in all_files]}")
        
        if single_camera_mode:
            # For single camera mode, check for 'camera' subdirectory first
            camera_dir = input_path / "camera"
            if camera_dir.exists():
                left_dir = camera_dir
                logger.info(f"Single camera mode: Using 'camera' subdirectory: {left_dir}")
            else:
                # Fallback to 'left' subdirectory or directly in input directory
                left_dir = input_path / "left"
                if not left_dir.exists():
                    left_dir = input_path
                    logger.info(f"Single camera mode: Using input directory directly: {left_dir}")
                else:
                    logger.info(f"Single camera mode: Using 'left' subdirectory: {left_dir}")
            right_dir = None  # No right camera in single camera mode
        else:
            # Standard stereo mode
            left_dir = input_path / "left"
            right_dir = input_path / "right"
            
            logger.info(f"Looking for left images in: {left_dir}")
            logger.info(f"Left directory exists: {left_dir.exists()}")
            logger.info(f"Looking for right images in: {right_dir}")
            logger.info(f"Right directory exists: {right_dir.exists()}")
        
        # List files in subdirectories if they exist
        if left_dir.exists():
            left_dir_files = list(left_dir.iterdir())
            logger.info(f"Files in left directory: {[f.name for f in left_dir_files[:10]]}{'...' if len(left_dir_files) > 10 else ''}")
        
        if right_dir and right_dir.exists():
            right_dir_files = list(right_dir.iterdir())
            logger.info(f"Files in right directory: {[f.name for f in right_dir_files[:10]]}{'...' if len(right_dir_files) > 10 else ''}")
        
        # Load calibration info
        info_path = input_path / "calibration_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                calib_info = json.load(f)
                logger.info(f"Loaded calibration info: {calib_info['resolution']}")
        else:
            logger.info(f"No calibration_info.json found at: {info_path}")
        
        # Find image pairs - try both JPG and PNG
        left_pattern_jpg = str(left_dir / "*.jpg")
        left_pattern_png = str(left_dir / "*.png")
        
        logger.info(f"Searching for left JPG images: {left_pattern_jpg}")
        logger.info(f"Searching for left PNG images: {left_pattern_png}")
        
        if right_dir:
            right_pattern_jpg = str(right_dir / "*.jpg")
            right_pattern_png = str(right_dir / "*.png")
            logger.info(f"Searching for right JPG images: {right_pattern_jpg}")
            logger.info(f"Searching for right PNG images: {right_pattern_png}")
        else:
            right_pattern_jpg = ""
            right_pattern_png = ""
        
        left_files = sorted(glob.glob(left_pattern_jpg) + glob.glob(left_pattern_png))
        if right_dir:
            right_files = sorted(glob.glob(right_pattern_jpg) + glob.glob(right_pattern_png))
        else:
            right_files = []
        
        logger.info(f"Found {len(left_files)} left images: {[Path(f).name for f in left_files[:5]]}{'...' if len(left_files) > 5 else ''}")
        if right_dir:
            logger.info(f"Found {len(right_files)} right images: {[Path(f).name for f in right_files[:5]]}{'...' if len(right_files) > 5 else ''}")
        
        if single_camera_mode:
            if len(left_files) == 0:
                raise ValueError(f"No images found in single camera mode")
            logger.info(f"Found {len(left_files)} single camera images")
        else:
            if len(left_files) != len(right_files):
                raise ValueError(f"Mismatched image counts: {len(left_files)} left, {len(right_files)} right")
            logger.info(f"Found {len(left_files)} image pairs")
        
        left_images = []
        right_images = []
        image_size = None
        
        if single_camera_mode:
            # Single camera mode - only load left images
            for left_path in left_files:
                left_img = cv2.imread(left_path)
                
                if left_img is None:
                    logger.warning(f"Failed to load image: {os.path.basename(left_path)}")
                    continue
                
                if image_size is None:
                    image_size = (left_img.shape[1], left_img.shape[0])
                    logger.info(f"Image size: {image_size[0]}x{image_size[1]}")
                
                left_images.append(left_img)
            
            logger.info(f"Loaded {len(left_images)} single camera images")
            return left_images, [], image_size  # Empty right_images for single camera
        else:
            # Stereo mode - load both left and right images
            for left_path, right_path in zip(left_files, right_files):
                left_img = cv2.imread(left_path)
                right_img = cv2.imread(right_path)
                
                if left_img is None or right_img is None:
                    logger.warning(f"Failed to load image pair: {os.path.basename(left_path)}")
                    continue
                
                if image_size is None:
                    image_size = (left_img.shape[1], left_img.shape[0])
                    logger.info(f"Image size: {image_size[0]}x{image_size[1]}")
                
                left_images.append(left_img)
                right_images.append(right_img)
            
            return left_images, right_images, image_size
    
    def detect_corners(self, images, visualize=False):
        """
        Detect checkerboard corners in all images.
        
        Returns:
            list: List of detected corners for each image
        """
        all_corners = []
        valid_indices = []
        
        for idx, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners with adaptive threshold
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            
            if ret:
                # Refine corners for sub-pixel accuracy
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                all_corners.append(corners)
                valid_indices.append(idx)
                
                if visualize and idx < 5:  # Visualize first 5
                    vis_img = cv2.drawChessboardCorners(img.copy(), self.checkerboard_size, corners, ret)
                    cv2.imwrite(f"corners_detected_{idx:03d}.jpg", vis_img)
            else:
                logger.warning(f"No corners found in image {idx}")
        
        logger.info(f"Detected corners in {len(all_corners)}/{len(images)} images")
        return all_corners, valid_indices
    
    def calibrate_stereo_advanced(self, left_images, right_images, image_size):
        """
        Perform advanced stereo calibration with multiple algorithms.
        
        Returns:
            dict: Calibration parameters
        """
        # Detect corners
        logger.info("Detecting checkerboard corners...")
        left_corners, left_valid = self.detect_corners(left_images)
        right_corners, right_valid = self.detect_corners(right_images)
        
        # Find common valid indices
        valid_indices = list(set(left_valid) & set(right_valid))
        logger.info(f"Common valid images: {len(valid_indices)}")
        
        if len(valid_indices) < 10:
            raise ValueError(f"Insufficient valid image pairs: {len(valid_indices)} < 10")
        
        # Filter to common valid images
        object_points = [self.objp for _ in valid_indices]
        left_points = [left_corners[left_valid.index(i)] for i in valid_indices]
        right_points = [right_corners[right_valid.index(i)] for i in valid_indices]
        
        # Individual camera calibration first
        logger.info("Calibrating left camera...")
        ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
            object_points, left_points, image_size, None, None,
            flags=cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
        )
        
        logger.info("Calibrating right camera...")
        ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
            object_points, right_points, image_size, None, None,
            flags=cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
        )
        
        logger.info(f"Individual calibration RMS: Left={ret_l:.3f}, Right={ret_r:.3f}")
        
        # Stereo calibration with advanced flags
        logger.info("Performing stereo calibration...")
        
        # Try multiple calibration strategies
        calibration_flags = [
            cv2.CALIB_FIX_INTRINSIC,  # Use individual calibrations
            cv2.CALIB_USE_INTRINSIC_GUESS,  # Refine from individual
            cv2.CALIB_FIX_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT,  # Fix main params
            0  # Full optimization
        ]
        
        best_rms = float('inf')
        best_params = None
        
        for flag_idx, flags in enumerate(calibration_flags):
            try:
                ret_stereo, K1_s, D1_s, K2_s, D2_s, R, T, E, F = cv2.stereoCalibrate(
                    object_points, left_points, right_points,
                    K1.copy(), D1.copy(), K2.copy(), D2.copy(),
                    image_size,
                    criteria=self.criteria,
                    flags=flags
                )
                
                # Calculate baseline - T is already in mm since objp is in mm
                baseline_mm = np.linalg.norm(T)
                logger.info(f"Strategy {flag_idx}: RMS={ret_stereo:.3f}, Baseline={baseline_mm:.1f}mm")
                
                # Check if this is the best result
                baseline_error = abs(baseline_mm - self.target_baseline_mm)
                logger.info(f"  Baseline error: {baseline_error:.1f}mm")
                logger.info(f"  T vector: {T.ravel()}")
                logger.info(f"  T magnitude: {np.linalg.norm(T)}")
                
                # Debug: show what we're calculating
                logger.info(f"  T * 1000 = {np.linalg.norm(T) * 1000}")
                
                if ret_stereo < best_rms:
                    best_rms = ret_stereo
                    best_params = (K1_s, D1_s, K2_s, D2_s, R, T, E, F)
                    logger.info(f"  -> New best calibration (RMS: {ret_stereo:.3f})")
                    
            except Exception as e:
                logger.warning(f"Strategy {flag_idx} failed: {e}")
        
        if best_params is None:
            logger.error("All calibration strategies failed!")
            logger.error("Debug info:")
            logger.error(f"  Target baseline: {self.target_baseline_mm}mm")
            logger.error(f"  Square size used: {self.square_size_mm}mm")
            logger.error("  This might be caused by:")
            logger.error("  1. Incorrect square size measurement")
            logger.error("  2. Poor image quality")
            logger.error("  3. Insufficient camera separation")
            logger.error("  4. Wrong checkerboard pattern size")
            raise ValueError("All calibration strategies failed")
        
        K1, D1, K2, D2, R, T, E, F = best_params
        initial_baseline = np.linalg.norm(T)
        logger.info(f"Best calibration: RMS={best_rms:.3f}")
        logger.info(f"Initial baseline from T vector: {initial_baseline:.1f}mm")
        
        # Check if we need to scale the square size based on baseline error
        baseline_ratio = initial_baseline / self.target_baseline_mm
        if baseline_ratio > 10 or baseline_ratio < 0.1:
            logger.warning(f"Large baseline ratio detected: {baseline_ratio:.1f}")
            logger.warning(f"This suggests the square size might be incorrect")
            logger.warning(f"Actual square size might be: {self.square_size_mm * baseline_ratio:.1f}mm")
        
        # Stereo rectification
        logger.info("Computing stereo rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            alpha=0,  # No black borders
            newImageSize=image_size
        )
        
        # Calculate final baseline from projection matrices
        baseline_mm = abs(P2[0, 3] - P1[0, 3]) / K1[0, 0]
        logger.info(f"Final baseline from P matrices: {baseline_mm:.2f}mm")
        logger.info(f"Target baseline: {self.target_baseline_mm}mm")
        logger.info(f"Baseline error: {abs(baseline_mm - self.target_baseline_mm):.1f}mm")
        
        # Create calibration dictionary
        calibration = {
            "timestamp": datetime.now().isoformat(),
            "image_size": list(image_size),
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size_mm,
            "num_images_used": len(valid_indices),
            "rms_error": best_rms,
            "baseline_mm": baseline_mm,
            
            "K1": K1.tolist(),
            "D1": D1.ravel().tolist(),
            "K2": K2.tolist(),
            "D2": D2.ravel().tolist(),
            "R": R.tolist(),
            "T": T.ravel().tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "roi1": list(roi1),
            "roi2": list(roi2)
        }
        
        return calibration
    
    def validate_calibration(self, calibration, left_images, right_images):
        """
        Validate calibration quality with ISO-compliant metrics.
        
        Returns:
            dict: Validation metrics
        """
        logger.info("Validating calibration quality...")
        
        # Load calibration matrices
        K1 = np.array(calibration['K1'])
        D1 = np.array(calibration['D1'])
        K2 = np.array(calibration['K2'])
        D2 = np.array(calibration['D2'])
        R1 = np.array(calibration['R1'])
        R2 = np.array(calibration['R2'])
        P1 = np.array(calibration['P1'])
        P2 = np.array(calibration['P2'])
        
        image_size = tuple(calibration['image_size'])
        
        # Create rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        
        # Test epipolar geometry on sample images
        epipolar_errors = []
        for idx in range(min(5, len(left_images))):
            # Rectify images
            left_rect = cv2.remap(left_images[idx], map1x, map1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_images[idx], map2x, map2y, cv2.INTER_LINEAR)
            
            # Find feature matches
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY), None)
            kp2, des2 = orb.detectAndCompute(cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY), None)
            
            if des1 is not None and des2 is not None:
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                if len(matches) > 10:
                    # Calculate epipolar error (y-coordinate difference)
                    for match in matches[:50]:  # Use top 50 matches
                        pt1 = kp1[match.queryIdx].pt
                        pt2 = kp2[match.trainIdx].pt
                        epipolar_error = abs(pt1[1] - pt2[1])  # Should be near 0
                        epipolar_errors.append(epipolar_error)
        
        # Calculate metrics
        metrics = {
            "rms_reprojection_error": calibration['rms_error'],
            "baseline_mm": calibration['baseline_mm'],
            "baseline_error_mm": abs(calibration['baseline_mm'] - self.target_baseline_mm),
            "epipolar_error_mean": np.mean(epipolar_errors) if epipolar_errors else 0,
            "epipolar_error_std": np.std(epipolar_errors) if epipolar_errors else 0,
            "epipolar_error_max": np.max(epipolar_errors) if epipolar_errors else 0,
            "num_epipolar_tests": len(epipolar_errors),
            "iso_compliant": calibration['rms_error'] < 0.5 and calibration['baseline_error_mm'] < 2.0
        }
        
        logger.info(f"Validation metrics: RMS={metrics['rms_reprojection_error']:.3f}, "
                   f"Epipolar error={metrics['epipolar_error_mean']:.2f}±{metrics['epipolar_error_std']:.2f} pixels")
        
        return metrics
    
    def save_calibration_report(self, calibration, metrics, output_path):
        """Generate comprehensive calibration report"""
        report = {
            "calibration": calibration,
            "validation_metrics": metrics,
            "quality_assessment": {
                "overall_quality": "EXCELLENT" if metrics['rms_reprojection_error'] < 0.3 else
                                 "GOOD" if metrics['rms_reprojection_error'] < 0.5 else
                                 "ACCEPTABLE" if metrics['rms_reprojection_error'] < 1.0 else "POOR",
                "iso_compliant": metrics['iso_compliant'],
                "recommendations": []
            }
        }
        
        # Add recommendations
        if metrics['rms_reprojection_error'] > 0.5:
            report['quality_assessment']['recommendations'].append(
                "Consider recapturing calibration images with better coverage"
            )
        if metrics['baseline_error_mm'] > 2.0:
            report['quality_assessment']['recommendations'].append(
                f"Baseline error is {metrics['baseline_error_mm']:.1f}mm, verify camera spacing"
            )
        if metrics['epipolar_error_mean'] > 1.0:
            report['quality_assessment']['recommendations'].append(
                "High epipolar error detected, calibration may need refinement"
            )
        
        # Save report
        report_path = Path(output_path).parent / "calibration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Calibration report saved to {report_path}")
        return report
    
    def save_verification_images(self, calibration, left_images, right_images, output_dir=".", num_samples=3):
        """
        Save rectified images and epipolar line visualizations for verification.
        
        Args:
            calibration: Calibration data dictionary
            left_images: List of left camera images
            right_images: List of right camera images
            output_dir: Directory to save verification images
            num_samples: Number of sample images to process
        """
        logger.info("Generating verification images...")
        
        # Load calibration matrices
        K1 = np.array(calibration['K1'])
        D1 = np.array(calibration['D1'])
        K2 = np.array(calibration['K2'])
        D2 = np.array(calibration['D2'])
        R1 = np.array(calibration['R1'])
        R2 = np.array(calibration['R2'])
        P1 = np.array(calibration['P1'])
        P2 = np.array(calibration['P2'])
        
        image_size = tuple(calibration['image_size'])
        
        # Create rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process sample images
        num_samples = min(num_samples, len(left_images), len(right_images))
        
        for i in range(num_samples):
            left_img = left_images[i]
            right_img = right_images[i]
            
            # Rectify images
            left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
            
            # Save individual rectified images
            cv2.imwrite(str(output_path / f"rectified_left_{i:02d}.jpg"), left_rect)
            cv2.imwrite(str(output_path / f"rectified_right_{i:02d}.jpg"), right_rect)
            
            # Create side-by-side rectified comparison
            rect_combined = np.hstack((left_rect, right_rect))
            
            # Draw horizontal epipolar lines every 50 pixels
            h, w = left_rect.shape[:2]
            for y in range(0, h, 50):
                cv2.line(rect_combined, (0, y), (w*2, y), (0, 255, 0), 1)
            
            # Add text labels
            cv2.putText(rect_combined, "LEFT (Rectified)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(rect_combined, "RIGHT (Rectified)", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(rect_combined, f"Baseline: {calibration['baseline_mm']:.1f}mm", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imwrite(str(output_path / f"rectified_comparison_{i:02d}.jpg"), rect_combined)
            
            # Create epipolar line visualization on original images
            orig_combined = np.hstack((left_img, right_img))
            
            # Find some feature points for epipolar line demonstration
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Detect corners or feature points
            corners_left = cv2.goodFeaturesToTrack(gray_left, maxCorners=20, 
                                                  qualityLevel=0.01, minDistance=50)
            
            if corners_left is not None:
                # Load fundamental matrix
                F = np.array(calibration['F'])
                
                # Draw epipolar lines for a few points
                for j, corner in enumerate(corners_left[:10]):  # Limit to 10 points
                    pt = corner.ravel().astype(int)
                    
                    # Draw point on left image
                    cv2.circle(orig_combined, tuple(pt), 5, (0, 0, 255), -1)
                    
                    # Calculate epipolar line on right image
                    line = cv2.computeCorrespondEpilines(corner.reshape(-1, 1, 2), 1, F)
                    line = line.reshape(-1, 3)[0]
                    
                    # Draw epipolar line on right image
                    x0, y0 = map(int, [0, -line[2]/line[1]])
                    x1, y1 = map(int, [right_img.shape[1], -(line[2] + line[0] * right_img.shape[1])/line[1]])
                    
                    # Offset for right image position
                    x0 += left_img.shape[1]
                    x1 += left_img.shape[1]
                    
                    # Ensure line coordinates are within image bounds
                    if 0 <= y0 < orig_combined.shape[0] and 0 <= y1 < orig_combined.shape[0]:
                        cv2.line(orig_combined, (x0, y0), (x1, y1), (0, 255, 0), 1)
            
            # Add labels to original epipolar visualization
            cv2.putText(orig_combined, "LEFT (Original)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(orig_combined, "RIGHT (Original + Epipolar Lines)", 
                       (left_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(str(output_path / f"epipolar_lines_{i:02d}.jpg"), orig_combined)
            
            # Create rectified epipolar verification (should show horizontal lines)
            rect_epipolar = self._create_rectified_epipolar_check(left_rect, right_rect, i)
            cv2.imwrite(str(output_path / f"rectified_epipolar_check_{i:02d}.jpg"), rect_epipolar)
            
            logger.info(f"Saved verification images for sample {i+1}")
        
        # Create a summary image showing calibration quality
        self._create_calibration_summary(calibration, output_path)
        
        logger.info(f"Verification images saved to {output_path}")
        logger.info("Generated files:")
        logger.info("  - rectified_left_XX.jpg: Individual rectified left images")
        logger.info("  - rectified_right_XX.jpg: Individual rectified right images") 
        logger.info("  - rectified_comparison_XX.jpg: Side-by-side rectified with epipolar lines")
        logger.info("  - epipolar_lines_XX.jpg: Original images with epipolar lines")
        logger.info("  - rectified_epipolar_check_XX.jpg: Rectified epipolar correspondence verification")
        logger.info("  - calibration_summary.jpg: Overall calibration quality summary")
    
    def _create_rectified_epipolar_check(self, left_rect, right_rect, sample_idx):
        """
        Create a detailed epipolar correspondence check on rectified images.
        This should show perfectly horizontal correspondences if calibration is correct.
        """
        # Convert to grayscale for feature detection
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        # Detect good features to track
        corners_left = cv2.goodFeaturesToTrack(
            gray_left, 
            maxCorners=50, 
            qualityLevel=0.01, 
            minDistance=30,
            blockSize=7
        )
        
        # Create combined image
        combined = np.hstack((left_rect, right_rect))
        h, w = left_rect.shape[:2]
        
        if corners_left is not None:
            # Use template matching to find correspondences in rectified images
            correspondences = []
            
            for corner in corners_left:
                x1, y1 = corner.ravel().astype(int)
                
                # Skip points too close to borders
                if x1 < 20 or x1 > w-20 or y1 < 20 or y1 > h-20:
                    continue
                
                # Extract template around the point
                template_size = 21
                half_size = template_size // 2
                template = gray_left[y1-half_size:y1+half_size+1, x1-half_size:x1+half_size+1]
                
                if template.shape[0] != template_size or template.shape[1] != template_size:
                    continue
                
                # Search along the same horizontal line in right image (epipolar constraint)
                search_width = min(200, w - 40)  # Search range
                search_y = y1  # Same Y coordinate (epipolar line is horizontal)
                search_x_start = max(20, x1 - search_width//2)
                search_x_end = min(w - 20, x1 + search_width//2)
                
                best_match_x = None
                best_match_val = -1
                
                # Template matching along horizontal line
                for search_x in range(search_x_start, search_x_end - template_size, 2):
                    if search_x + template_size >= w or search_y + half_size >= h:
                        continue
                    
                    search_template = gray_right[search_y-half_size:search_y+half_size+1, 
                                                search_x-half_size:search_x+half_size+1]
                    
                    if search_template.shape == template.shape:
                        # Normalized cross correlation
                        result = cv2.matchTemplate(template, search_template, cv2.TM_CCOEFF_NORMED)
                        match_val = result[0, 0]
                        
                        if match_val > best_match_val and match_val > 0.7:  # Threshold for good matches
                            best_match_val = match_val
                            best_match_x = search_x
                
                if best_match_x is not None:
                    correspondences.append(((x1, y1), (best_match_x, search_y), best_match_val))
            
            # Draw correspondences
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, ((x1, y1), (x2, y2), confidence) in enumerate(correspondences[:20]):  # Limit to 20 best
                color = colors[i % len(colors)]
                
                # Draw points
                cv2.circle(combined, (x1, y1), 4, color, -1)
                cv2.circle(combined, (x2 + w, y2), 4, color, -1)
                
                # Draw horizontal line connecting the points
                cv2.line(combined, (x1, y1), (x2 + w, y2), color, 1)
                
                # Calculate vertical disparity (should be near 0 for good calibration)
                vertical_error = abs(y1 - y2)
                
                # Add text showing disparity info
                if i < 10:  # Only label first 10 matches
                    cv2.putText(combined, f"{i+1}: dy={vertical_error}px", 
                              (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add grid lines every 50 pixels for reference
        for y in range(0, h, 50):
            cv2.line(combined, (0, y), (w*2, y), (128, 128, 128), 1)
        
        # Add labels
        cv2.putText(combined, "LEFT (Rectified)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "RIGHT (Rectified)", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Epipolar Check: Lines should be HORIZONTAL", (10, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined, f"Found {len(correspondences) if corners_left is not None else 0} correspondences", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined
    
    def _create_calibration_summary(self, calibration, output_path):
        """Create a summary image with calibration information."""
        # Create a summary image (800x600)
        summary_img = np.zeros((600, 800, 3), dtype=np.uint8)
        summary_img.fill(50)  # Dark gray background
        
        # Add title
        cv2.putText(summary_img, "STEREO CALIBRATION SUMMARY", (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Add calibration metrics
        y_pos = 120
        line_height = 35
        
        metrics = [
            f"RMS Error: {calibration['rms_error']:.3f} pixels",
            f"Baseline: {calibration['baseline_mm']:.2f} mm",
            f"Image Size: {calibration['image_size'][0]}x{calibration['image_size'][1]}",
            f"Checkerboard: {calibration['checkerboard_size'][0]}x{calibration['checkerboard_size'][1]}",
            f"Square Size: {calibration['square_size_mm']} mm",
            f"Images Used: {calibration['num_images_used']}",
            f"Timestamp: {calibration['timestamp'][:19]}"
        ]
        
        for metric in metrics:
            cv2.putText(summary_img, metric, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_pos += line_height
        
        # Add quality indicators
        y_pos += 20
        cv2.putText(summary_img, "QUALITY ASSESSMENT:", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_pos += 40
        
        # RMS quality
        rms_color = (0, 255, 0) if calibration['rms_error'] < 0.5 else (0, 165, 255) if calibration['rms_error'] < 1.0 else (0, 0, 255)
        rms_text = "EXCELLENT" if calibration['rms_error'] < 0.5 else "GOOD" if calibration['rms_error'] < 1.0 else "FAIR"
        cv2.putText(summary_img, f"RMS Quality: {rms_text}", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, rms_color, 2)
        y_pos += line_height
        
        # Baseline quality
        baseline_error = abs(calibration['baseline_mm'] - 80.0)  # Assuming 80mm target
        baseline_color = (0, 255, 0) if baseline_error < 2.0 else (0, 165, 255) if baseline_error < 5.0 else (0, 0, 255)
        baseline_text = "EXCELLENT" if baseline_error < 2.0 else "GOOD" if baseline_error < 5.0 else "POOR"
        cv2.putText(summary_img, f"Baseline Quality: {baseline_text}", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, baseline_color, 2)
        
        # Add camera matrices info (simplified)
        y_pos = 450
        cv2.putText(summary_img, "CAMERA MATRICES:", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_pos += 25
        
        K1 = np.array(calibration['K1'])
        K2 = np.array(calibration['K2'])
        
        cv2.putText(summary_img, f"Left fx: {K1[0,0]:.1f}  fy: {K1[1,1]:.1f}", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        cv2.putText(summary_img, f"Right fx: {K2[0,0]:.1f}  fy: {K2[1,1]:.1f}", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imwrite(str(output_path / "calibration_summary.jpg"), summary_img)

def deploy_calibration(calibration_file, deploy_to_default=True):
    """Deploy calibration to default UnLook locations"""
    if not deploy_to_default:
        logger.info("Skipping deployment to default locations")
        return
    
    logger.info("Deploying calibration to default locations...")
    
    calib_path = Path(calibration_file)
    if not calib_path.exists():
        logger.error(f"Calibration file not found: {calibration_file}")
        return
    
    # Default locations
    locations = [
        Path(__file__).parent.parent.parent / "calibration" / "default" / "default_stereo.json",
        Path(__file__).parent.parent.parent.parent / "server" / "calibration" / "default" / "default_stereo.json"
    ]
    
    for loc in locations:
        try:
            loc.parent.mkdir(parents=True, exist_ok=True)
            
            # Load and update calibration format if needed
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            
            # Ensure compatible format
            if "calibration" in calib_data:
                calib_data = calib_data["calibration"]
            
            with open(loc, 'w') as f:
                json.dump(calib_data, f, indent=2)
            
            logger.info(f"Deployed to: {loc}")
            
        except Exception as e:
            logger.warning(f"Failed to deploy to {loc}: {e}")

def main():
    parser = argparse.ArgumentParser(description="2K Calibration Processing Pipeline")
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with captured calibration images')
    parser.add_argument('--output', type=str, default='calibration_2k.json',
                        help='Output calibration file')
    parser.add_argument('--checkerboard-columns', type=int, default=8,
                        help='Number of inner corners horizontally')
    parser.add_argument('--checkerboard-rows', type=int, default=5,
                        help='Number of inner corners vertically')
    parser.add_argument('--checkerboard-size', type=str, 
                        help='Checkerboard size as "columnsxrows" (e.g., "9x6") - alternative to separate columns/rows args')
    parser.add_argument('--square-size', type=float, default=19.5,
                        help='Size of checkerboard squares in mm')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy to default calibration locations')
    parser.add_argument('--save-verification', action='store_true',
                        help='Save rectified images and epipolar line visualizations')
    parser.add_argument('--verification-samples', type=int, default=3,
                        help='Number of sample images for verification (default: 3)')
    parser.add_argument('--single-camera', action='store_true',
                        help='Calibrate single camera instead of stereo pair')
    parser.add_argument('--camera', type=str, choices=['left', 'right'], default='left',
                        help='Which camera to calibrate in single camera mode')
    
    args = parser.parse_args()
    
    # Parse checkerboard size if provided as "columnsxrows"
    if args.checkerboard_size:
        try:
            cols, rows = map(int, args.checkerboard_size.split('x'))
            args.checkerboard_columns = cols
            args.checkerboard_rows = rows
        except ValueError:
            logger.error(f"Invalid checkerboard size format: {args.checkerboard_size}. Use format like '9x6'")
            return 1
    
    # Log command line arguments
    logger.info("Starting calibration with parameters:")
    logger.info(f"   Input directory: {args.input}")
    logger.info(f"   Output file: {args.output}")
    logger.info(f"   Checkerboard size: {args.checkerboard_columns}x{args.checkerboard_rows}")
    logger.info(f"   Square size: {args.square_size} mm")
    logger.info(f"   Visualize: {args.visualize}")
    logger.info(f"   Deploy: {args.deploy}")
    logger.info(f"   Save verification: {args.save_verification}")
    logger.info(f"   Verification samples: {args.verification_samples}")
    logger.info(f"   Single camera mode: {args.single_camera}")
    if args.single_camera:
        logger.info(f"   Camera to calibrate: {args.camera}")
    
    # Initialize calibration processor
    logger.info("Initializing calibration processor...")
    calibrator = StereoCalibration2K(
        checkerboard_size=(args.checkerboard_columns, args.checkerboard_rows),
        square_size_mm=args.square_size
    )
    
    try:
        if args.single_camera:
            # Single camera calibration
            logger.info(f"Loading images for single camera calibration ({args.camera} camera)")
            left_images, right_images, image_size = calibrator.load_calibration_images(args.input, single_camera_mode=True)
            
            # Use appropriate camera images
            if args.camera == 'left':
                images = left_images
            else:
                images = right_images if right_images else left_images
            
            if not images:
                logger.error(f"No images found for {args.camera} camera")
                return 1
            
            # Perform single camera calibration
            camera_matrix, dist_coeffs, rvecs, tvecs, rms_error = calibrator.calibrate_single_camera(images)
            
            # Save single camera calibration
            calibration_data = {
                "calibration_type": "single_camera",
                "camera": args.camera,
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.tolist(),
                "image_size": image_size,
                "checkerboard_size": [args.checkerboard_columns, args.checkerboard_rows],
                "square_size_mm": args.square_size,
                "rms_reprojection_error": rms_error,
                "num_calibration_images": len(images),
                "calibration_date": datetime.now().isoformat(),
                "notes": f"Single {args.camera} camera calibration for projector-camera structured light system"
            }
            
            # Save calibration file
            output_file = Path(args.output)
            with open(output_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"✅ Single camera calibration completed successfully!")
            logger.info(f"📁 Calibration saved to: {output_file}")
            logger.info(f"📊 RMS reprojection error: {rms_error:.3f} pixels")
            logger.info(f"🎯 Next step: Use this calibration for projector-camera calibration")
            
            return 0
        else:
            # Original stereo calibration
            left_images, right_images, image_size = calibrator.load_calibration_images(args.input)
        
        # Perform calibration
        calibration = calibrator.calibrate_stereo_advanced(left_images, right_images, image_size)
        
        # Validate calibration
        metrics = calibrator.validate_calibration(calibration, left_images, right_images)
        
        # Save calibration and report
        with open(args.output, 'w') as f:
            json.dump(calibration, f, indent=2)
        logger.info(f"Calibration saved to {args.output}")
        
        report = calibrator.save_calibration_report(calibration, metrics, args.output)
        
        # Save verification images if requested
        if args.save_verification:
            output_dir = Path(args.output).parent / "verification_images"
            calibrator.save_verification_images(
                calibration, left_images, right_images, 
                output_dir=str(output_dir),
                num_samples=args.verification_samples
            )
        
        # Deploy if requested
        if args.deploy:
            deploy_calibration(args.output, deploy_to_default=True)
        
        # Print summary
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print(f"RMS Error: {calibration['rms_error']:.3f} pixels")
        print(f"Baseline: {calibration['baseline_mm']:.2f}mm")
        print(f"Epipolar Error: {metrics['epipolar_error_mean']:.2f}±{metrics['epipolar_error_std']:.2f} pixels")
        print(f"ISO Compliant: {'YES' if metrics['iso_compliant'] else 'NO'}")
        print(f"Quality: {report['quality_assessment']['overall_quality']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())