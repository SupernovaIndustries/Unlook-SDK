#!/usr/bin/env python3
"""
Fix calibration for resolution mismatch by recalculating rectification parameters.
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


def fix_calibration_for_resolution(calib_file, target_size):
    """Fix calibration parameters for different resolution."""
    
    # Load calibration
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    # Original calibration size
    orig_w, orig_h = calib['image_size']
    target_w, target_h = target_size
    
    # Calculate scale factors
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    logger.info(f"Scaling calibration from {orig_w}x{orig_h} to {target_w}x{target_h}")
    logger.info(f"Scale factors: {scale_x:.3f} x {scale_y:.3f}")
    
    # Scale camera matrices
    K1 = np.array(calib['camera_matrix_left'])
    K2 = np.array(calib['camera_matrix_right'])
    
    # Scale focal lengths and principal points
    K1_scaled = K1.copy()
    K1_scaled[0, 0] *= scale_x  # fx
    K1_scaled[1, 1] *= scale_y  # fy
    K1_scaled[0, 2] *= scale_x  # cx
    K1_scaled[1, 2] *= scale_y  # cy
    
    K2_scaled = K2.copy()
    K2_scaled[0, 0] *= scale_x  # fx
    K2_scaled[1, 1] *= scale_y  # fy
    K2_scaled[0, 2] *= scale_x  # cx
    K2_scaled[1, 2] *= scale_y  # cy
    
    # Get other calibration parameters
    dist1 = np.array(calib['dist_coeffs_left'])
    dist2 = np.array(calib['dist_coeffs_right'])
    R = np.array(calib['R'])
    T = np.array(calib['T'])
    
    # Recalculate rectification for the new image size
    logger.info("Recalculating rectification parameters for scaled resolution...")
    
    flags = cv2.CALIB_ZERO_DISPARITY
    alpha = 0  # 0 = crop to valid pixels only, 1 = keep all pixels
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1_scaled, dist1, K2_scaled, dist2,
        (target_w, target_h), R, T,
        flags=flags, alpha=alpha
    )
    
    # Create updated calibration
    updated_calib = calib.copy()
    
    # Update with scaled parameters
    updated_calib['camera_matrix_left'] = K1_scaled.tolist()
    updated_calib['camera_matrix_right'] = K2_scaled.tolist()
    updated_calib['image_size'] = [target_w, target_h]
    updated_calib['R1'] = R1.tolist()
    updated_calib['R2'] = R2.tolist()
    updated_calib['P1'] = P1.tolist()
    updated_calib['P2'] = P2.tolist()
    updated_calib['Q'] = Q.tolist()
    updated_calib['roi1'] = roi1
    updated_calib['roi2'] = roi2
    
    # Calculate baseline from scaled parameters
    baseline_mm = np.linalg.norm(T)
    updated_calib['baseline_mm'] = float(baseline_mm)
    
    # Save the fixed calibration
    output_file = Path(calib_file).parent / "calibration_fixed.json"
    with open(output_file, 'w') as f:
        json.dump(updated_calib, f, indent=4)
    
    logger.info(f"Fixed calibration saved to: {output_file}")
    
    # Verify the fix
    logger.info("\nVerifying rectification with fixed calibration...")
    verify_rectification(str(output_file), target_size)
    
    return str(output_file)


def verify_rectification(calib_file, image_size):
    """Verify rectification quality."""
    
    # Load calibration
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    # Create dummy checkerboard images to test rectification
    w, h = image_size
    
    # Create checkerboard pattern
    square_size = 50
    pattern_size = (9, 6)
    
    board_w = pattern_size[0] * square_size
    board_h = pattern_size[1] * square_size
    
    # Center the board
    offset_x = (w - board_w) // 2
    offset_y = (h - board_h) // 2
    
    # Create test images
    test_left = np.ones((h, w, 3), dtype=np.uint8) * 255
    test_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Draw checkerboard
    for i in range(pattern_size[1]):
        for j in range(pattern_size[0]):
            if (i + j) % 2 == 0:
                y1 = offset_y + i * square_size
                y2 = offset_y + (i + 1) * square_size
                x1 = offset_x + j * square_size
                x2 = offset_x + (j + 1) * square_size
                test_left[y1:y2, x1:x2] = 0
                test_right[y1:y2, x1:x2] = 0
    
    # Apply rectification
    K1 = np.array(calib['camera_matrix_left'])
    K2 = np.array(calib['camera_matrix_right'])
    dist1 = np.array(calib['dist_coeffs_left'])
    dist2 = np.array(calib['dist_coeffs_right'])
    R1 = np.array(calib['R1'])
    R2 = np.array(calib['R2'])
    P1 = np.array(calib['P1'])
    P2 = np.array(calib['P2'])
    
    # Create rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, image_size, cv2.CV_32FC1)
    
    # Rectify
    rect_left = cv2.remap(test_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(test_right, map2x, map2y, cv2.INTER_LINEAR)
    
    # Check if horizontal lines are aligned
    vis = np.hstack([rect_left, rect_right])
    h, w = vis.shape[:2]
    
    # Draw epipolar lines
    for y in range(0, h, 50):
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    
    cv2.imwrite("calibration_test_epipolar.jpg", vis)
    logger.info("Saved calibration_test_epipolar.jpg")
    
    # Test on actual images
    left_img = cv2.imread("captured_data/20250531_005620/left_006_phase_shift_f8_s0.jpg")
    right_img = cv2.imread("captured_data/20250531_005620/right_006_phase_shift_f8_s0.jpg")
    
    if left_img is not None and right_img is not None:
        rect_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
        
        # Check feature alignment
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        
        detector = cv2.ORB_create(nfeatures=100)
        kp_left, desc_left = detector.detectAndCompute(gray_left, None)
        kp_right, desc_right = detector.detectAndCompute(gray_right, None)
        
        if desc_left is not None and desc_right is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc_left, desc_right)
            
            y_diffs = []
            for match in matches[:30]:
                pt_left = kp_left[match.queryIdx].pt
                pt_right = kp_right[match.trainIdx].pt
                y_diffs.append(abs(pt_left[1] - pt_right[1]))
            
            if y_diffs:
                mean_y_diff = np.mean(y_diffs)
                logger.info(f"Mean Y difference on real images: {mean_y_diff:.2f} pixels")
                
                if mean_y_diff < 2.0:
                    logger.info("[SUCCESS] Rectification is working correctly!")
                else:
                    logger.warning(f"[WARNING] Y-difference is {mean_y_diff:.2f} pixels")


if __name__ == "__main__":
    # Fix calibration for 728x544 resolution
    calib_file = "captured_data/20250531_005620/calibration.json"
    target_size = (728, 544)
    
    fixed_calib_file = fix_calibration_for_resolution(calib_file, target_size)