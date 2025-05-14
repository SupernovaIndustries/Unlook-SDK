#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headless Camera Calibration Tool for Unlook SDK

This script provides a simple interface for performing stereo camera calibration
without requiring GUI support. It's designed for systems where OpenCV's GUI
functionality is not available.

The calibration process involves:
1. Capturing multiple images of a checkerboard from both cameras
2. Finding the checkerboard corners in each image
3. Calculating the intrinsic and extrinsic camera parameters
4. Computing the stereo rectification matrices

The calibration results are saved to a file for use in 3D scanning applications.

Usage:
  python headless_calibration.py [--output FILENAME] [--squares-size MM] [--num-images N]
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("headless_calibration")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from unlook import UnlookClient
    from unlook.client.camera_calibration import StereoCalibrator, save_calibration
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    sys.exit(1)

def capture_calibration_images(client, num_images=20, delay=2.0, save_dir=None):
    """
    Capture stereo image pairs for calibration.
    
    Args:
        client: Unlook client instance
        num_images: Number of image pairs to capture
        delay: Delay between captures (seconds)
        save_dir: Directory to save the images (optional)
        
    Returns:
        Tuple of (left_images, right_images)
    """
    print("\nCapturing calibration images.")
    print("Please move the checkerboard pattern around the field of view:")
    print("1. Hold the checkerboard at different distances (30-80cm)")
    print("2. Rotate the checkerboard to different angles (0°, ±15°, ±30°)")
    print("3. Position the checkerboard in all areas of the image (center, corners, edges)")
    print("4. Ensure the entire checkerboard is visible in both cameras\n")
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving calibration images to: {save_dir}")
    
    left_images = []
    right_images = []
    
    # Capture images
    for i in range(num_images):
        print(f"Capturing image pair {i+1}/{num_images}... ", end="")
        sys.stdout.flush()
        
        # Prompt user
        if i > 0:
            print(f"\nPlease move the checkerboard to a new position/orientation")
            print(f"Waiting {delay:.1f} seconds before next capture...")
            time.sleep(delay)
        
        # Capture image pair
        try:
            left_img, right_img = client.camera.capture_stereo_pair()
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # Check images
        if left_img is None or right_img is None:
            print("Failed!")
            continue
            
        # Save images if directory provided
        if save_dir:
            left_path = os.path.join(save_dir, f"left_{i:02d}.png")
            right_path = os.path.join(save_dir, f"right_{i:02d}.png")
            cv2.imwrite(left_path, left_img)
            cv2.imwrite(right_path, right_img)
        
        # Save images to memory
        left_images.append(left_img)
        right_images.append(right_img)
        
        print("Success!")
    
    return left_images, right_images

def main():
    """Main function for the camera calibration tool."""
    parser = argparse.ArgumentParser(description="Headless Camera Calibration Tool for Unlook SDK")
    parser.add_argument("--output", type=str, default="stereo_calibration.json",
                        help="Output calibration file")
    parser.add_argument("--squares-size", type=float, default=19.4,
                        help="Size of the checkerboard squares in mm")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of image pairs to capture")
    parser.add_argument("--checkerboard", type=str, default="9x6",
                        help="Checkerboard size (width x height in inner corners)")
    parser.add_argument("--baseline", type=float, default=80.0,
                        help="Camera baseline in mm (for validation)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save calibration images to disk")
    parser.add_argument("--image-delay", type=float, default=2.0,
                        help="Delay between image captures (seconds)")
    args = parser.parse_args()
    
    # Parse checkerboard size
    try:
        width, height = map(int, args.checkerboard.split("x"))
        checkerboard_size = (width, height)
    except Exception:
        print(f"Invalid checkerboard size: {args.checkerboard}. Using default 9x6.")
        checkerboard_size = (9, 6)
    
    # Create image save directory
    save_dir = None
    if args.save_images:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "calibration_images")
    
    print("\n" + "="*80)
    print(" UNLOOK SDK - HEADLESS STEREO CAMERA CALIBRATION TOOL")
    print("="*80)
    
    print("\nThis tool will help you calibrate your stereo cameras for 3D scanning.")
    print(f"Using a {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard with {args.squares_size}mm squares.")
    print(f"Expected camera baseline: {args.baseline}mm")
    print(f"Saving calibration to: {args.output}")
    
    # Create client and connect to scanner
    print("\nConnecting to scanner...")
    client = UnlookClient(auto_discover=True)
    client.start_discovery()
    time.sleep(3)  # Wait for scanners to be discovered
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("No scanners found. Please ensure scanner hardware is connected.")
        return 1
    
    # Connect to first available scanner
    print(f"Connecting to scanner: {scanners[0].name} ({scanners[0].uuid})")
    client.connect(scanners[0])
    print(f"Successfully connected to scanner: {scanners[0].name}\n")
    
    try:
        # Create stereo calibrator
        calibrator = StereoCalibrator(
            checkerboard_size=checkerboard_size,
            square_size=args.squares_size
        )
        
        # Capture calibration images
        left_images, right_images = capture_calibration_images(
            client, 
            num_images=args.num_images,
            delay=args.image_delay,
            save_dir=save_dir
        )
        
        if len(left_images) < 5:
            print("Not enough valid calibration images (need at least 5). Please try again.")
            return 1
        
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
            baseline_mm = abs(calibration_params["T"][0][0])
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
        
        # Save some debug info
        if save_dir:
            # Save projection matrices
            if "P1" in calibration_params and "P2" in calibration_params:
                P1 = calibration_params["P1"]
                P2 = calibration_params["P2"]
                
                np.savetxt(os.path.join(save_dir, "P1.txt"), P1, fmt="%.6f")
                np.savetxt(os.path.join(save_dir, "P2.txt"), P2, fmt="%.6f")
                
                # Calculate and save P2[0,3] which determines the baseline scaling
                if P2 is not None and P2.shape[0] >= 1 and P2.shape[1] >= 4:
                    p2_03 = P2[0, 3]
                    with open(os.path.join(save_dir, "p2_03_value.txt"), 'w') as f:
                        f.write(f"P2[0,3] = {p2_03}\n")
                        f.write(f"This value should be close to -fx*baseline_m\n")
                        f.write(f"If baseline is {args.baseline}mm and fx is {P1[0,0]:.1f}px,\n")
                        f.write(f"Then P2[0,3] should be approx: {-P1[0,0]*args.baseline/1000.0}\n")
        
        print("\nCalibration successful!")
        print("You can now use this calibration file for 3D scanning.")
        print(f"python static_scanning_example.py --calibration {args.output}")
        
        # Disconnect from scanner
        client.disconnect()
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"Error during calibration: {e}")
        traceback.print_exc()
        
        # Disconnect from scanner
        client.disconnect()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())