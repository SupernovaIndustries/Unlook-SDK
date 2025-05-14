#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load Calibration Images Tool

This script loads previously captured calibration images and runs the calibration process.
This is useful when you want to reuse already captured images without taking new ones.

Usage:
  python load_calibration_images.py --images-dir calibration_images --output calibration.json
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("load_calibration_images")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from unlook.client.camera_calibration import StereoCalibrator, save_calibration
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    sys.exit(1)

def load_images_from_directory(directory):
    """
    Load left and right images from a directory.
    
    Args:
        directory: Directory containing calibration images
        
    Returns:
        Tuple of (left_images, right_images)
    """
    print(f"Loading calibration images from: {directory}")
    
    # Find all left and right images
    left_files = sorted(glob.glob(os.path.join(directory, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(directory, "right_*.png")))
    
    if not left_files or not right_files:
        # Try alternative pattern
        left_files = sorted(glob.glob(os.path.join(directory, "*_left.png")))
        right_files = sorted(glob.glob(os.path.join(directory, "*_right.png")))
    
    if not left_files or not right_files:
        # Try another pattern
        left_files = sorted(glob.glob(os.path.join(directory, "*left*.png")))
        right_files = sorted(glob.glob(os.path.join(directory, "*right*.png")))
    
    if not left_files or not right_files:
        print(f"No image files found in {directory}")
        print("Expected file pattern: left_*.png and right_*.png")
        return [], []
    
    # Make sure we have matching pairs
    min_count = min(len(left_files), len(right_files))
    left_files = left_files[:min_count]
    right_files = right_files[:min_count]
    
    print(f"Found {min_count} image pairs")
    
    # Load images
    left_images = []
    right_images = []
    
    for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
        try:
            left_img = cv2.imread(left_file)
            right_img = cv2.imread(right_file)
            
            if left_img is None or right_img is None:
                print(f"Warning: Could not load image pair {i+1}")
                continue
                
            left_images.append(left_img)
            right_images.append(right_img)
            print(f"Loaded image pair {i+1}: {os.path.basename(left_file)} and {os.path.basename(right_file)}")
        except Exception as e:
            print(f"Error loading image pair {i+1}: {e}")
    
    print(f"Successfully loaded {len(left_images)} image pairs")
    return left_images, right_images

def main():
    """Main function for loading calibration images."""
    parser = argparse.ArgumentParser(description="Load Calibration Images Tool")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing calibration images")
    parser.add_argument("--output", type=str, default="stereo_calibration.json",
                        help="Output calibration file")
    parser.add_argument("--squares-size", type=float, default=19.4,
                        help="Size of the checkerboard squares in mm")
    parser.add_argument("--checkerboard", type=str, default="8x5",
                        help="Checkerboard size (width x height in inner corners)")
    parser.add_argument("--baseline", type=float, default=80.0,
                        help="Camera baseline in mm (for validation)")
    args = parser.parse_args()
    
    # Parse checkerboard size
    try:
        width, height = map(int, args.checkerboard.split("x"))
        checkerboard_size = (width, height)
    except Exception:
        print(f"Invalid checkerboard size: {args.checkerboard}. Using default 8x5.")
        checkerboard_size = (8, 5)
    
    print("\n" + "="*80)
    print(" UNLOOK SDK - LOAD CALIBRATION IMAGES TOOL")
    print("="*80)
    
    print("\nThis tool will calibrate cameras using your saved calibration images.")
    print(f"Using a {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard with {args.squares_size}mm squares.")
    print(f"Expected camera baseline: {args.baseline}mm")
    print(f"Saving calibration to: {args.output}")
    
    # Check if images directory exists
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory does not exist: {args.images_dir}")
        return 1
    
    # Load images
    left_images, right_images = load_images_from_directory(args.images_dir)
    
    if len(left_images) < 5:
        print("Not enough valid calibration images (need at least 5). Please check your images directory.")
        return 1
    
    try:
        # Create stereo calibrator
        calibrator = StereoCalibrator(
            checkerboard_size=checkerboard_size,
            square_size=args.squares_size
        )
        
        print(f"\nPerforming stereo calibration with {len(left_images)} image pairs...")
        
        # Perform calibration
        calibration_params = calibrator.calibrate_stereo(
            left_images,
            right_images,
            visualize=False  # No visualization in headless mode
        )
        
        if not calibration_params:
            print("Calibration failed. Please try again with better images.")
            return 1
        
        # Check baseline
        if "T" in calibration_params and calibration_params["T"] is not None:
            # The baseline is the x-component of the translation vector (in mm)
            if isinstance(calibration_params["T"], list):
                baseline_mm = abs(calibration_params["T"][0][0])
            else:
                baseline_mm = abs(calibration_params["T"][0, 0])
                
            print(f"\nMeasured baseline: {baseline_mm:.2f}mm (expected: {args.baseline}mm)")
            
            # Check if baseline is reasonable
            if abs(baseline_mm - args.baseline) > 20:
                print("WARNING: Measured baseline differs significantly from expected value.")
                print("This may indicate calibration issues.")
            else:
                print("Baseline measurement is reasonable, calibration looks good.")
        
        # Save calibration
        output_path = os.path.abspath(args.output)
        save_calibration(output_path, calibration_params)
        print(f"\nCalibration saved to: {output_path}")
        
        # Check P matrices if available
        if "P1" in calibration_params and "P2" in calibration_params:
            P1 = calibration_params["P1"]
            P2 = calibration_params["P2"]
            
            # Convert to numpy arrays if needed
            if isinstance(P1, list):
                P1 = np.array(P1)
            if isinstance(P2, list):
                P2 = np.array(P2)
            
            # Save projection matrices
            p_dir = os.path.join(os.path.dirname(output_path), "projection_matrices")
            os.makedirs(p_dir, exist_ok=True)
            
            np.savetxt(os.path.join(p_dir, "P1.txt"), P1, fmt="%.6f")
            np.savetxt(os.path.join(p_dir, "P2.txt"), P2, fmt="%.6f")
            
            # Check P2[0,3]
            if P2.shape[0] >= 1 and P2.shape[1] >= 4:
                p2_03 = P2[0, 3]
                expected_p2_03 = -P1[0, 0] * args.baseline / 1000.0
                print(f"\nP2[0,3] = {p2_03:.6f} (expected: {expected_p2_03:.6f})")
                
                with open(os.path.join(p_dir, "p2_03_value.txt"), 'w') as f:
                    f.write(f"P2[0,3] = {p2_03}\n")
                    f.write(f"This value should be close to -fx*baseline_m\n")
                    f.write(f"If baseline is {args.baseline}mm and fx is {P1[0, 0]:.1f}px,\n")
                    f.write(f"Then P2[0,3] should be approx: {-P1[0, 0]*args.baseline/1000.0}\n")
        
        print("\nCalibration successful!")
        print("You can now use this calibration file for 3D scanning:")
        print(f"python static_scanning_example.py --calibration {args.output}")
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"Error during calibration: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())