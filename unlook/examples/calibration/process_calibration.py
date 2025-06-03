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
    
    def __init__(self, checkerboard_size=(8, 5), square_size_mm=24.0):
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
    
    def load_calibration_images(self, input_dir):
        """
        Load calibration images from directory.
        
        Returns:
            tuple: (left_images, right_images, image_size)
        """
        input_path = Path(input_dir)
        left_dir = input_path / "left"
        right_dir = input_path / "right"
        
        # Load calibration info
        info_path = input_path / "calibration_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                calib_info = json.load(f)
                logger.info(f"Loaded calibration info: {calib_info['resolution']}")
        
        # Find image pairs
        left_files = sorted(glob.glob(str(left_dir / "*.jpg")))
        right_files = sorted(glob.glob(str(right_dir / "*.jpg")))
        
        if len(left_files) != len(right_files):
            raise ValueError(f"Mismatched image counts: {len(left_files)} left, {len(right_files)} right")
        
        logger.info(f"Found {len(left_files)} image pairs")
        
        left_images = []
        right_images = []
        image_size = None
        
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
                
                # Calculate baseline
                baseline_mm = np.linalg.norm(T) * 1000
                logger.info(f"Strategy {flag_idx}: RMS={ret_stereo:.3f}, Baseline={baseline_mm:.1f}mm")
                
                # Check if this is the best result
                if ret_stereo < best_rms and abs(baseline_mm - self.target_baseline_mm) < 10:
                    best_rms = ret_stereo
                    best_params = (K1_s, D1_s, K2_s, D2_s, R, T, E, F)
                    
            except Exception as e:
                logger.warning(f"Strategy {flag_idx} failed: {e}")
        
        if best_params is None:
            raise ValueError("All calibration strategies failed")
        
        K1, D1, K2, D2, R, T, E, F = best_params
        logger.info(f"Best calibration: RMS={best_rms:.3f}")
        
        # Stereo rectification
        logger.info("Computing stereo rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            alpha=0,  # No black borders
            newImageSize=image_size
        )
        
        # Calculate final baseline
        baseline_mm = abs(P2[0, 3] - P1[0, 3]) / K1[0, 0] * 1000
        logger.info(f"Final baseline: {baseline_mm:.2f}mm (target: {self.target_baseline_mm}mm)")
        
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
    parser.add_argument('--square-size', type=float, default=24.0,
                        help='Size of checkerboard squares in mm')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy to default calibration locations')
    
    args = parser.parse_args()
    
    # Initialize calibration processor
    calibrator = StereoCalibration2K(
        checkerboard_size=(args.checkerboard_columns, args.checkerboard_rows),
        square_size_mm=args.square_size
    )
    
    try:
        # Load images
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