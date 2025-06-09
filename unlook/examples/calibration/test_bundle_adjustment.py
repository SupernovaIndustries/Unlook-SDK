#!/usr/bin/env python3
"""Test script for bundle adjustment optimization.

This script demonstrates how to apply bundle adjustment to improve
stereo calibration accuracy using Ceres Solver.

Usage:
    python test_bundle_adjustment.py --calibration calibration_2k.json
    python test_bundle_adjustment.py --calibration calibration.json --images calibration_images/
"""

import argparse
import sys
import logging
from pathlib import Path
import cv2
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanning.calibration.bundle_adjustment import (
    StereoCalibrationOptimizer, 
    optimize_stereo_calibration_with_bundle_adjustment,
    CERES_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibration_images(images_dir: Path):
    """Load calibration images for testing."""
    left_images = []
    right_images = []
    
    # Look for common calibration image patterns
    for left_file in sorted(images_dir.glob("left_*.jpg")):
        right_file = images_dir / left_file.name.replace("left_", "right_")
        if right_file.exists():
            left_img = cv2.imread(str(left_file), cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(str(right_file), cv2.IMREAD_GRAYSCALE)
            if left_img is not None and right_img is not None:
                left_images.append(left_img)
                right_images.append(right_img)
                
    return left_images, right_images


def extract_checkerboard_points(images, checkerboard_size=(9, 6)):
    """Extract checkerboard points from calibration images."""
    objpoints = []
    imgpoints = []
    
    # Create object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= 24.0  # 24mm squares
    
    for img in images:
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(img, checkerboard_size, None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners.reshape(-1, 2))
            
    return objpoints, imgpoints


def test_bundle_adjustment_synthetic():
    """Test bundle adjustment with synthetic data."""
    logger.info("🧪 Testing bundle adjustment with synthetic data...")
    
    # Create synthetic calibration data
    image_size = (1920, 1080)
    
    # Synthetic camera parameters (with some noise)
    K1 = np.array([
        [1500 + np.random.normal(0, 10), 0, 960 + np.random.normal(0, 5)],
        [0, 1500 + np.random.normal(0, 10), 540 + np.random.normal(0, 5)],
        [0, 0, 1]
    ])
    
    K2 = np.array([
        [1500 + np.random.normal(0, 10), 0, 960 + np.random.normal(0, 5)],
        [0, 1500 + np.random.normal(0, 10), 540 + np.random.normal(0, 5)],
        [0, 0, 1]
    ])
    
    D1 = np.random.normal(0, 0.1, (5, 1))
    D2 = np.random.normal(0, 0.1, (5, 1))
    
    # Stereo parameters
    baseline = 60.0  # 60mm baseline
    R = np.eye(3)  # No rotation for simplicity
    T = np.array([baseline, 0, 0]).reshape(-1, 1)
    
    initial_params = {
        'K1': K1, 'D1': D1,
        'K2': K2, 'D2': D2,
        'R': R, 'T': T
    }
    
    # Generate synthetic observation data
    num_views = 10
    points_per_view = 50
    
    image_points_left = []
    image_points_right = []
    object_points = []
    
    for view in range(num_views):
        # Generate random 3D points
        obj_pts = np.random.uniform(-100, 100, (points_per_view, 3))
        obj_pts[:, 2] += 300  # Move points away from cameras
        
        # Project to left camera (add noise)
        left_pts, _ = cv2.projectPoints(obj_pts, np.zeros(3), np.zeros(3), K1, D1)
        left_pts = left_pts.reshape(-1, 2) + np.random.normal(0, 0.5, (points_per_view, 2))
        
        # Project to right camera (add noise)
        right_pts, _ = cv2.projectPoints(obj_pts, R, T, K2, D2)
        right_pts = right_pts.reshape(-1, 2) + np.random.normal(0, 0.5, (points_per_view, 2))
        
        object_points.append(obj_pts)
        image_points_left.append(left_pts)
        image_points_right.append(right_pts)
    
    # Test optimization
    optimizer = StereoCalibrationOptimizer()
    optimized_params, summary = optimizer.optimize_stereo_calibration(
        image_points_left, image_points_right, object_points,
        initial_params, image_size
    )
    
    # Validate results
    validation = optimizer.validate_calibration_improvement(
        initial_params, optimized_params,
        image_points_left, image_points_right, object_points
    )
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA TEST RESULTS")
    print("="*60)
    print(f"Initial RMS error: {validation['original_rms_error']:.4f} pixels")
    print(f"Optimized RMS error: {validation['optimized_rms_error']:.4f} pixels")
    print(f"Improvement: {validation['improvement_percent']:.1f}%")
    print(f"Meets target (<0.5px): {'YES' if validation['meets_target'] else 'NO'}")
    print("="*60)
    
    return validation['meets_target']


def test_bundle_adjustment_real(calibration_file: str, images_dir: str = None):
    """Test bundle adjustment with real calibration data."""
    logger.info(f"🔬 Testing bundle adjustment with real data: {calibration_file}")
    
    if not Path(calibration_file).exists():
        logger.error(f"Calibration file not found: {calibration_file}")
        return False
    
    # If images directory provided, use real images
    if images_dir and Path(images_dir).exists():
        logger.info(f"Loading calibration images from: {images_dir}")
        left_images, right_images = load_calibration_images(Path(images_dir))
        
        if not left_images:
            logger.error("No calibration images found")
            return False
            
        logger.info(f"Found {len(left_images)} calibration image pairs")
        
        # Extract checkerboard points
        object_points, image_points_left = extract_checkerboard_points(left_images)
        _, image_points_right = extract_checkerboard_points(right_images)
        
        if not object_points:
            logger.error("No valid checkerboard patterns found")
            return False
            
        logger.info(f"Extracted points from {len(object_points)} images")
        image_size = (left_images[0].shape[1], left_images[0].shape[0])
        
    else:
        logger.warning("No images directory provided - using synthetic data for demonstration")
        # Use synthetic data if no images available
        return test_bundle_adjustment_synthetic()
    
    # Apply bundle adjustment
    output_file = str(Path(calibration_file).with_suffix('.optimized.json'))
    
    optimized_params, summary = optimize_stereo_calibration_with_bundle_adjustment(
        calibration_file, image_points_left, image_points_right, 
        object_points, image_size, output_file
    )
    
    print("\n" + "="*60)
    print("REAL DATA TEST RESULTS")
    print("="*60)
    print(f"Initial RMS error: {summary['original_rms_error']:.4f} pixels")
    print(f"Optimized RMS error: {summary['optimized_rms_error']:.4f} pixels")
    print(f"Improvement: {summary['improvement_percent']:.1f}%")
    print(f"Meets target (<0.5px): {'YES' if summary['meets_target'] else 'NO'}")
    print(f"Optimized calibration saved: {output_file}")
    print("="*60)
    
    return summary['meets_target']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test bundle adjustment for stereo calibration optimization'
    )
    
    parser.add_argument('--calibration', type=str,
                       help='Path to calibration JSON file')
    parser.add_argument('--images', type=str,
                       help='Directory containing calibration images')
    parser.add_argument('--synthetic', action='store_true',
                       help='Test with synthetic data only')
    
    args = parser.parse_args()
    
    # Check if Ceres is available
    print("="*60)
    print("BUNDLE ADJUSTMENT TEST")
    print("="*60)
    print(f"Ceres Solver available: {'YES' if CERES_AVAILABLE else 'NO'}")
    
    if not CERES_AVAILABLE:
        print("\n❌ Ceres Solver not available!")
        print("Install instructions:")
        print("1. Ubuntu/Debian: sudo apt-get install libceres-dev")
        print("2. Install Python bindings: pip install pyceres")
        print("3. Or build from source: https://github.com/ceres-solver/ceres-solver")
        return 1
    
    success = False
    
    # Run synthetic test
    if args.synthetic or not args.calibration:
        success = test_bundle_adjustment_synthetic()
    
    # Run real data test
    if args.calibration:
        success = test_bundle_adjustment_real(args.calibration, args.images)
    
    if success:
        print("\n✅ Bundle adjustment test PASSED!")
        return 0
    else:
        print("\n❌ Bundle adjustment test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())