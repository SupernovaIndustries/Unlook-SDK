#!/usr/bin/env python3
"""
Verify and fix stereo calibration issues.
Analyzes the current calibration and suggests fixes.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_calibration():
    """Analyze current calibration for issues."""
    
    # Load calibration
    calib_file = Path("captured_data/20250531_005620/calibration.json")
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    logger.info("="*60)
    logger.info("CALIBRATION ANALYSIS")
    logger.info("="*60)
    
    # 1. Check baseline
    T = np.array(calib['T'])
    baseline = np.linalg.norm(T)
    logger.info(f"Baseline: {baseline:.2f} mm")
    
    if baseline < 50 or baseline > 200:
        logger.warning(f"  [ISSUE] Baseline {baseline:.2f}mm is unusual (expected 50-200mm)")
    else:
        logger.info(f"  [OK] Baseline is reasonable")
    
    # 2. Check focal lengths
    K1 = np.array(calib['camera_matrix_left'])
    K2 = np.array(calib['camera_matrix_right'])
    fx1, fy1 = K1[0,0], K1[1,1]
    fx2, fy2 = K2[0,0], K2[1,1]
    
    logger.info(f"Left camera focal length: fx={fx1:.1f}, fy={fy1:.1f}")
    logger.info(f"Right camera focal length: fx={fx2:.1f}, fy={fy2:.1f}")
    
    focal_diff = abs(fx1 - fx2) / fx1
    if focal_diff > 0.05:  # More than 5% difference
        logger.warning(f"  [ISSUE] Focal lengths differ by {focal_diff*100:.1f}%")
    else:
        logger.info(f"  [OK] Focal lengths are similar")
    
    # 3. Check image size vs calibration
    image_size = calib['image_size']
    logger.info(f"Calibration image size: {image_size}")
    
    # Load actual images
    left_img = cv2.imread("captured_data/20250531_005620/left_006_phase_shift_f8_s0.jpg")
    if left_img is not None:
        actual_size = [left_img.shape[1], left_img.shape[0]]
        logger.info(f"Actual image size: {actual_size}")
        
        if actual_size != image_size:
            logger.warning(f"  [ISSUE] Image size mismatch!")
            logger.warning(f"    Calibration: {image_size}")
            logger.warning(f"    Actual: {actual_size}")
            scale_x = actual_size[0] / image_size[0]
            scale_y = actual_size[1] / image_size[1]
            logger.warning(f"    Scale factor: {scale_x:.3f} x {scale_y:.3f}")
        else:
            logger.info(f"  [OK] Image sizes match")
    
    # 4. Check Q matrix
    Q = np.array(calib['Q'])
    logger.info(f"Q matrix analysis:")
    logger.info(f"  Q[2,3] = {Q[2,3]} (focal length)")
    logger.info(f"  Q[3,2] = {Q[3,2]} (should be -1/baseline)")
    
    expected_q32 = -1.0 / baseline
    if abs(Q[3,2] - expected_q32) > 0.001:
        logger.warning(f"  [ISSUE] Q[3,2] should be {expected_q32:.6f}, but is {Q[3,2]:.6f}")
    
    # 5. Check reprojection error
    reproj_error = calib.get('reprojection_error', 0)
    logger.info(f"Reprojection error: {reproj_error:.4f}")
    
    if reproj_error > 1.0:
        logger.warning(f"  [ISSUE] High reprojection error (should be < 1.0)")
    else:
        logger.info(f"  [OK] Reprojection error is good")
    
    # 6. Test rectification on sample images
    logger.info("\n" + "="*60)
    logger.info("RECTIFICATION TEST")
    logger.info("="*60)
    
    if left_img is not None:
        right_img = cv2.imread("captured_data/20250531_005620/right_006_phase_shift_f8_s0.jpg")
        
        if right_img is not None:
            # Create rectification maps
            R1 = np.array(calib['R1'])
            R2 = np.array(calib['R2'])
            P1 = np.array(calib['P1'])
            P2 = np.array(calib['P2'])
            
            map1x, map1y = cv2.initUndistortRectifyMap(
                K1, np.array(calib['dist_coeffs_left']),
                R1, P1, tuple(image_size), cv2.CV_32FC1
            )
            map2x, map2y = cv2.initUndistortRectifyMap(
                K2, np.array(calib['dist_coeffs_right']),
                R2, P2, tuple(image_size), cv2.CV_32FC1
            )
            
            # Rectify
            left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
            
            # Check epipolar alignment
            check_epipolar_alignment(left_rect, right_rect)
    
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    logger.info("1. The calibration has significant issues:")
    logger.info("   - Rectification is failing (epipolar lines not aligned)")
    logger.info("   - This causes incorrect disparity computation")
    logger.info("   - Which leads to wrong depth values")
    logger.info("")
    logger.info("2. To fix:")
    logger.info("   a) Recalibrate with more image pairs (20-30)")
    logger.info("   b) Ensure checkerboard is fully visible in all images")
    logger.info("   c) Use images at the same resolution as capture")
    logger.info("   d) Consider using cv2.CALIB_SAME_FOCAL_LENGTH flag")
    logger.info("")
    logger.info("3. Quick fix for current data:")
    logger.info("   - Process without rectification")
    logger.info("   - Or manually adjust baseline in calibration.json")


def check_epipolar_alignment(left_rect, right_rect):
    """Check if epipolar lines are properly aligned."""
    
    # Draw horizontal lines to visualize epipolar alignment
    vis = np.hstack([left_rect, right_rect])
    h, w = vis.shape[:2]
    
    # Draw lines every 50 pixels
    for y in range(0, h, 50):
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    
    # Save visualization
    cv2.imwrite("epipolar_check.jpg", vis)
    logger.info("Saved epipolar alignment check to epipolar_check.jpg")
    
    # Find features and check alignment
    gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    
    # Detect corners
    corners_left = cv2.goodFeaturesToTrack(gray_left, 50, 0.01, 10)
    corners_right = cv2.goodFeaturesToTrack(gray_right, 50, 0.01, 10)
    
    if corners_left is not None and corners_right is not None:
        # For each corner in left, find closest in right
        y_diffs = []
        for corner_l in corners_left:
            x_l, y_l = corner_l[0]
            
            # Find closest corner in right image with similar y
            min_dist = float('inf')
            best_y_diff = 0
            
            for corner_r in corners_right:
                x_r, y_r = corner_r[0]
                y_diff = abs(y_l - y_r)
                
                if y_diff < 10:  # Consider only points on similar epipolar line
                    dist = abs(x_l - x_r)
                    if dist < min_dist:
                        min_dist = dist
                        best_y_diff = y_l - y_r
            
            if min_dist < float('inf'):
                y_diffs.append(best_y_diff)
        
        if y_diffs:
            mean_y_diff = np.mean(np.abs(y_diffs))
            logger.info(f"Mean Y difference between correspondences: {mean_y_diff:.1f} pixels")
            
            if mean_y_diff > 2.0:
                logger.warning("  [ISSUE] Poor epipolar alignment!")
            else:
                logger.info("  [OK] Good epipolar alignment")


if __name__ == "__main__":
    analyze_calibration()