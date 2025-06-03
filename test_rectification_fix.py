#!/usr/bin/env python3
"""
Test the rectification fix for image size mismatch.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from unlook.client.scanning.calibration.stereo_rectification import StereoRectifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rectification():
    """Test rectification with the fix."""
    
    # Load calibration
    calib_file = Path("captured_data/20250531_005620/calibration.json")
    
    # Initialize rectifier
    rectifier = StereoRectifier(str(calib_file))
    
    # Load test images
    left_img = cv2.imread("captured_data/20250531_005620/left_006_phase_shift_f8_s0.jpg")
    right_img = cv2.imread("captured_data/20250531_005620/right_006_phase_shift_f8_s0.jpg")
    
    if left_img is None or right_img is None:
        logger.error("Failed to load test images")
        return
    
    logger.info(f"Original image size: {left_img.shape[1]}x{left_img.shape[0]}")
    
    # Test rectification
    logger.info("Testing rectification with automatic scaling...")
    rect_left, rect_right = rectifier.rectify_images(left_img, right_img)
    
    # Save rectified images
    cv2.imwrite("test_rectified_left.jpg", rect_left)
    cv2.imwrite("test_rectified_right.jpg", rect_right)
    
    # Create visualization with epipolar lines
    vis = np.hstack([rect_left, rect_right])
    h, w = vis.shape[:2]
    
    # Draw horizontal lines every 50 pixels
    for y in range(0, h, 50):
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    
    cv2.imwrite("test_epipolar_check.jpg", vis)
    logger.info("Saved test_epipolar_check.jpg")
    
    # Test feature matching on rectified images
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    
    # Detect features
    detector = cv2.ORB_create(nfeatures=200)
    kp_left, desc_left = detector.detectAndCompute(gray_left, None)
    kp_right, desc_right = detector.detectAndCompute(gray_right, None)
    
    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_left, desc_right)
    
    # Check epipolar constraint
    y_diffs = []
    for match in matches[:50]:  # Check first 50 matches
        pt_left = kp_left[match.queryIdx].pt
        pt_right = kp_right[match.trainIdx].pt
        y_diff = abs(pt_left[1] - pt_right[1])
        y_diffs.append(y_diff)
    
    mean_y_diff = np.mean(y_diffs) if y_diffs else 0
    logger.info(f"Mean Y difference after rectification: {mean_y_diff:.2f} pixels")
    
    if mean_y_diff < 2.0:
        logger.info("[SUCCESS] Rectification is working correctly!")
    else:
        logger.warning(f"[WARNING] Rectification may still have issues. Y-diff: {mean_y_diff:.2f}")
    
    # Draw matches
    matches_img = cv2.drawMatches(
        rect_left, kp_left, rect_right, kp_right,
        matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("test_matches.jpg", matches_img)
    logger.info("Saved test_matches.jpg")


if __name__ == "__main__":
    test_rectification()