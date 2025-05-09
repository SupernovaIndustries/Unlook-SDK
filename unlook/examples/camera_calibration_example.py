#!/usr/bin/env python
"""
Example script for calibrating stereo cameras using the UnlookScanner.

This script demonstrates how to:
1. Capture calibration images from both cameras
2. Run the calibration process
3. Save calibration parameters
4. Verify calibration quality
5. Optionally test with a 3D scan

Usage:
  python camera_calibration_example.py --mode capture --output_dir ./calibration_images
  python camera_calibration_example.py --mode calibrate --input_dir ./calibration_images --output_file ./calibration_params.json
  python camera_calibration_example.py --mode verify --calibration_file ./calibration_params.json
"""

import os
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from unlook.client import UnlookClient
from unlook.client.camera_calibration import StereoCalibrator
from unlook.client.scanner3d import UnlookScanner


def capture_calibration_images(output_dir: str, 
                              num_images: int = 20, 
                              delay: float = 2.0,
                              checkerboard_size: Tuple[int, int] = (9, 6),
                              visualize: bool = True) -> None:
    """
    Capture a series of checkerboard images from both cameras for calibration.
    
    Args:
        output_dir: Directory to save the captured images
        num_images: Number of image pairs to capture
        delay: Delay between captures in seconds
        checkerboard_size: Size of the checkerboard (width, height) in inner corners
        visualize: Whether to display the captured images
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create left and right subdirectories
    left_dir = output_path / "left"
    right_dir = output_path / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    
    # Connect to the scanner
    client = UnlookClient.auto_connect(timeout=5)
    if client is None:
        print("Failed to connect to UnlookClient. Make sure the server is running.")
        return
    
    # Prepare for image capture
    client.camera.start_streaming()
    
    print(f"Capturing {num_images} image pairs for calibration")
    print("Position the checkerboard in different orientations and distances")
    print(f"Checkerboard should have {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
    print("Press 'c' to capture or 'q' to quit when the preview window is in focus")
    
    count = 0
    last_capture_time = 0
    
    try:
        while count < num_images:
            # Get frames from both cameras
            left_frame = client.camera.get_left_frame()
            right_frame = client.camera.get_right_frame()
            
            if left_frame is None or right_frame is None:
                print("Failed to capture frames from cameras")
                time.sleep(0.1)
                continue
            
            # Display frames
            if visualize:
                display = np.hstack((left_frame, right_frame))
                display = cv2.resize(display, (0, 0), fx=0.5, fy=0.5)
                cv2.putText(display, f"Captured: {count}/{num_images}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Calibration Capture", display)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                
                # Capture on 'c' key press or after delay if auto mode
                current_time = time.time()
                if key == ord('c') or (current_time - last_capture_time > delay and count == 0):
                    auto_mode = key != ord('c') and count == 0
                    if auto_mode:
                        print("Starting automatic capture mode")
                    
                    # Try to detect the checkerboard
                    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    
                    left_found, _ = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
                    right_found, _ = cv2.findChessboardCorners(gray_right, checkerboard_size, None)
                    
                    if left_found and right_found:
                        # Save images
                        left_path = left_dir / f"left_{count:03d}.png"
                        right_path = right_dir / f"right_{count:03d}.png"
                        
                        cv2.imwrite(str(left_path), left_frame)
                        cv2.imwrite(str(right_path), right_frame)
                        
                        print(f"Saved image pair {count+1}/{num_images}")
                        count += 1
                        last_capture_time = current_time
                    elif key == ord('c'):
                        print("Checkerboard not detected in both images")
                        
                # In auto mode, capture if delay has passed and checkerboard is detected
                elif auto_mode and current_time - last_capture_time > delay:
                    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    
                    left_found, _ = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
                    right_found, _ = cv2.findChessboardCorners(gray_right, checkerboard_size, None)
                    
                    if left_found and right_found:
                        # Save images
                        left_path = left_dir / f"left_{count:03d}.png"
                        right_path = right_dir / f"right_{count:03d}.png"
                        
                        cv2.imwrite(str(left_path), left_frame)
                        cv2.imwrite(str(right_path), right_frame)
                        
                        print(f"Saved image pair {count+1}/{num_images}")
                        count += 1
                        last_capture_time = current_time
            else:
                # Non-visual mode, just capture at regular intervals
                current_time = time.time()
                if current_time - last_capture_time > delay:
                    # Save images without checkerboard validation
                    left_path = left_dir / f"left_{count:03d}.png"
                    right_path = right_dir / f"right_{count:03d}.png"
                    
                    cv2.imwrite(str(left_path), left_frame)
                    cv2.imwrite(str(right_path), right_frame)
                    
                    print(f"Saved image pair {count+1}/{num_images}")
                    count += 1
                    last_capture_time = current_time
                
                time.sleep(0.1)
    
    finally:
        # Clean up
        client.camera.stop_streaming()
        if visualize:
            cv2.destroyAllWindows()
    
    print(f"Successfully captured {count} image pairs for calibration")
    print(f"Images saved to {output_dir}/left and {output_dir}/right")


def run_calibration(input_dir: str, 
                   output_file: str,
                   checkerboard_size: Tuple[int, int] = (9, 6),
                   square_size: float = 0.025,  # 25mm squares
                   visualize: bool = True) -> None:
    """
    Run stereo camera calibration using captured images.
    
    Args:
        input_dir: Directory containing the calibration images
        output_file: File to save the calibration parameters
        checkerboard_size: Size of the checkerboard (width, height) in inner corners
        square_size: Size of the checkerboard squares in meters
        visualize: Whether to visualize the calibration results
    """
    input_path = Path(input_dir)
    left_dir = input_path / "left"
    right_dir = input_path / "right"
    
    if not left_dir.exists() or not right_dir.exists():
        print(f"Error: Input directories not found: {left_dir} or {right_dir}")
        return
    
    # Load left images
    left_images_paths = sorted(left_dir.glob("*.png"))
    right_images_paths = sorted(right_dir.glob("*.png"))
    
    if len(left_images_paths) != len(right_images_paths):
        print("Error: Number of left and right images doesn't match")
        return
    
    if len(left_images_paths) < 10:
        print(f"Warning: Found only {len(left_images_paths)} image pairs. For good calibration, at least 10 pairs are recommended.")
    
    print(f"Found {len(left_images_paths)} image pairs for calibration")
    
    # Load images
    left_images = [cv2.imread(str(path)) for path in left_images_paths]
    right_images = [cv2.imread(str(path)) for path in right_images_paths]
    
    # Create calibrator
    calibrator = StereoCalibrator(
        checkerboard_size=checkerboard_size,
        square_size=square_size
    )
    
    # Run calibration
    print("Running stereo calibration...")
    calibration_result = calibrator.calibrate_stereo(
        left_images=left_images,
        right_images=right_images,
        visualize=visualize
    )
    
    if calibration_result["success"]:
        print(f"Calibration successful. RMS error: {calibration_result['rms_error']:.6f}")
        
        # Save calibration
        calibrator.save_calibration(output_file)
        print(f"Calibration parameters saved to {output_file}")
        
        # Compute and display rectification
        if visualize and len(left_images) > 0:
            # Use the first image pair for visualization
            left_img = left_images[0]
            right_img = right_images[0]
            
            # Rectify images
            rectified_pair = calibrator.rectify_stereo_pair(left_img, right_img)
            
            # Display original and rectified images
            h, w = left_img.shape[:2]
            original_pair = np.hstack((left_img, right_img))
            
            # Draw horizontal lines to check rectification
            rectified_with_lines = rectified_pair.copy()
            for i in range(0, h, 30):
                cv2.line(rectified_with_lines, (0, i), (2*w, i), (0, 255, 0), 1)
            
            # Display
            cv2.imshow("Original Images", cv2.resize(original_pair, (0, 0), fx=0.5, fy=0.5))
            cv2.imshow("Rectified Images", cv2.resize(rectified_with_lines, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Calibration failed")


def verify_calibration(calibration_file: str, 
                      test_image_left: Optional[str] = None,
                      test_image_right: Optional[str] = None,
                      visualize: bool = True) -> None:
    """
    Verify the quality of a stereo calibration.
    
    Args:
        calibration_file: Path to the calibration file
        test_image_left: Path to a test left image (optional)
        test_image_right: Path to a test right image (optional)
        visualize: Whether to visualize the verification results
    """
    # Load calibration
    calibrator = StereoCalibrator()
    if not calibrator.load_calibration(calibration_file):
        print(f"Failed to load calibration from {calibration_file}")
        return
    
    print("Calibration loaded successfully")
    print(f"Camera matrix left:\n{calibrator.camera_matrix_left}")
    print(f"Camera matrix right:\n{calibrator.camera_matrix_right}")
    print(f"Translation between cameras:\n{calibrator.T}")
    
    # If test images are provided, verify the calibration on them
    if test_image_left and test_image_right:
        left_img = cv2.imread(test_image_left)
        right_img = cv2.imread(test_image_right)
        
        if left_img is None or right_img is None:
            print("Failed to load test images")
            return
        
        # Rectify the images
        rectified_pair = calibrator.rectify_stereo_pair(left_img, right_img)
        
        if visualize:
            # Draw horizontal lines to check rectification
            h, w = left_img.shape[:2]
            original_pair = np.hstack((left_img, right_img))
            
            rectified_with_lines = rectified_pair.copy()
            for i in range(0, h, 30):
                cv2.line(rectified_with_lines, (0, i), (2*w, i), (0, 255, 0), 1)
            
            # Display
            cv2.imshow("Original Images", cv2.resize(original_pair, (0, 0), fx=0.5, fy=0.5))
            cv2.imshow("Rectified Images", cv2.resize(rectified_with_lines, (0, 0), fx=0.5, fy=0.5))
            
            # Compute disparity map if possible
            rect_left = rectified_pair[:, :w]
            rect_right = rectified_pair[:, w:]
            
            try:
                disparity_map = calibrator.compute_disparity_map(rect_left, rect_right)
                disparity_visual = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                cv2.imshow("Disparity Map", disparity_visual)
                
                # Generate point cloud
                points_3d = calibrator.disparity_to_point_cloud(disparity_map, rect_left)
                print(f"Generated point cloud with {len(points_3d)} points")
                
                # Visualize point cloud with color (simplified display)
                if len(points_3d) > 0:
                    # Convert to format suitable for visualization
                    # Just display a subset of points
                    sample_idx = np.random.choice(len(points_3d), min(10000, len(points_3d)), replace=False)
                    
                    # Create a simple visualization of point cloud as 2D projections
                    z_values = points_3d[sample_idx, 2]
                    min_z, max_z = np.min(z_values), np.max(z_values)
                    
                    # Normalize Z coordinates for coloring
                    if max_z > min_z:
                        normalized_z = (z_values - min_z) / (max_z - min_z) * 255
                    else:
                        normalized_z = np.zeros_like(z_values)
                    
                    # Create visualization images for top and side views
                    vis_size = 500
                    top_view = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
                    side_view = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
                    
                    # Scale points to fit in visualization
                    x_vals = points_3d[sample_idx, 0]
                    y_vals = points_3d[sample_idx, 1]
                    z_vals = points_3d[sample_idx, 2]
                    
                    x_min, x_max = np.min(x_vals), np.max(x_vals)
                    y_min, y_max = np.min(y_vals), np.max(y_vals)
                    z_min, z_max = np.min(z_vals), np.max(z_vals)
                    
                    if x_max > x_min and y_max > y_min and z_max > z_min:
                        for i in range(len(sample_idx)):
                            x = int((x_vals[i] - x_min) / (x_max - x_min) * (vis_size - 1))
                            y = int((y_vals[i] - y_min) / (y_max - y_min) * (vis_size - 1))
                            z = int((z_vals[i] - z_min) / (z_max - z_min) * (vis_size - 1))
                            
                            # Color based on depth
                            color = cv2.applyColorMap(np.array([[int(normalized_z[i])]], dtype=np.uint8), 
                                                     cv2.COLORMAP_JET)[0, 0].tolist()
                            
                            # Draw points in top view (X-Y plane)
                            cv2.circle(top_view, (x, y), 1, color, -1)
                            
                            # Draw points in side view (X-Z plane)
                            cv2.circle(side_view, (x, z), 1, color, -1)
                    
                        cv2.imshow("Point Cloud - Top View (X-Y)", top_view)
                        cv2.imshow("Point Cloud - Side View (X-Z)", side_view)
            
            except Exception as e:
                print(f"Error creating disparity map or point cloud: {e}")
            
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # If no test images, connect to cameras and capture a pair for testing
        print("No test images provided. Connecting to cameras to capture test images...")
        
        client = UnlookClient.auto_connect(timeout=5)
        if client is None:
            print("Failed to connect to UnlookClient. Make sure the server is running.")
            return
        
        # Start streaming
        client.camera.start_streaming()
        
        try:
            print("Press 'c' to capture test image or 'q' to quit")
            
            while True:
                # Get frames
                left_frame = client.camera.get_left_frame()
                right_frame = client.camera.get_right_frame()
                
                if left_frame is None or right_frame is None:
                    print("Failed to capture frames from cameras")
                    time.sleep(0.1)
                    continue
                
                # Display frames
                if visualize:
                    display = np.hstack((left_frame, right_frame))
                    display = cv2.resize(display, (0, 0), fx=0.5, fy=0.5)
                    cv2.putText(display, "Press 'c' to capture, 'q' to quit", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Test Capture", display)
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    
                    if key == ord('c'):
                        # Rectify the captured frames
                        rectified_pair = calibrator.rectify_stereo_pair(left_frame, right_frame)
                        
                        # Draw horizontal lines to check rectification
                        h, w = left_frame.shape[:2]
                        original_pair = np.hstack((left_frame, right_frame))
                        
                        rectified_with_lines = rectified_pair.copy()
                        for i in range(0, h, 30):
                            cv2.line(rectified_with_lines, (0, i), (2*w, i), (0, 255, 0), 1)
                        
                        # Display
                        cv2.imshow("Original Images", cv2.resize(original_pair, (0, 0), fx=0.5, fy=0.5))
                        cv2.imshow("Rectified Images", cv2.resize(rectified_with_lines, (0, 0), fx=0.5, fy=0.5))
                        
                        # Compute disparity map
                        try:
                            rect_left = rectified_pair[:, :w]
                            rect_right = rectified_pair[:, w:]
                            
                            disparity_map = calibrator.compute_disparity_map(rect_left, rect_right)
                            disparity_visual = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            cv2.imshow("Disparity Map", disparity_visual)
                            
                            print("Disparity map computed. Press any key to continue...")
                            cv2.waitKey(0)
                        except Exception as e:
                            print(f"Error computing disparity map: {e}")
                
                else:
                    # In non-visual mode, just capture and process one pair
                    rectified_pair = calibrator.rectify_stereo_pair(left_frame, right_frame)
                    print("Test rectification completed")
                    break
            
        finally:
            # Clean up
            client.camera.stop_streaming()
            if visualize:
                cv2.destroyAllWindows()


def test_with_scan(calibration_file: str, 
                  output_dir: str = "./scan_output",
                  scanner_type: str = "robust") -> None:
    """
    Test the calibration by performing a 3D scan.
    
    Args:
        calibration_file: Path to the calibration file
        output_dir: Directory to save scan results
        scanner_type: Type of scanner to use (basic, enhanced, robust)
    """
    print(f"Testing calibration with a {scanner_type} 3D scan")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize scanner with custom calibration
    scanner = UnlookScanner.auto_connect(
        timeout=5,
        use_default_calibration=False,  # Don't use default calibration
        scanner_type=scanner_type
    )
    
    if scanner is None:
        print("Failed to connect to UnlookScanner")
        return
    
    try:
        # Load custom calibration
        calibrator = StereoCalibrator()
        if not calibrator.load_calibration(calibration_file):
            print(f"Failed to load calibration from {calibration_file}")
            return
        
        # Set custom calibration parameters in scanner
        scanner.set_calibration_params(
            camera_matrix_left=calibrator.camera_matrix_left,
            dist_coeffs_left=calibrator.dist_coeffs_left,
            camera_matrix_right=calibrator.camera_matrix_right,
            dist_coeffs_right=calibrator.dist_coeffs_right,
            R=calibrator.R,
            T=calibrator.T
        )
        
        # Perform scan
        print("Starting 3D scan with custom calibration...")
        result = scanner.scan(
            output_dir=output_dir,
            num_patterns=10,  # Use fewer patterns for a quick test
            save_debug=True
        )
        
        if result["success"]:
            print(f"Scan completed successfully. Point cloud saved to {result['point_cloud_path']}")
            print(f"Generated {result['num_points']} 3D points")
        else:
            print(f"Scan failed: {result['error']}")
    
    finally:
        scanner.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Camera calibration example for UnlookScanner")
    parser.add_argument("--mode", type=str, required=True, choices=["capture", "calibrate", "verify", "test_scan"],
                       help="Operation mode")
    
    # Parameters for all modes
    parser.add_argument("--checkerboard_width", type=int, default=9,
                       help="Number of inner corners along the width of the checkerboard")
    parser.add_argument("--checkerboard_height", type=int, default=6,
                       help="Number of inner corners along the height of the checkerboard")
    parser.add_argument("--square_size", type=float, default=0.025,
                       help="Size of the checkerboard squares in meters (default: 25mm)")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Disable visualization")
    
    # Mode-specific parameters
    parser.add_argument("--output_dir", type=str, default="./calibration_images",
                       help="Output directory for captured images or scan results")
    parser.add_argument("--input_dir", type=str, default="./calibration_images",
                       help="Input directory for calibration images")
    parser.add_argument("--output_file", type=str, default="./calibration_params.json",
                       help="Output file for calibration parameters")
    parser.add_argument("--calibration_file", type=str, default="./calibration_params.json",
                       help="Input calibration file for verification or testing")
    parser.add_argument("--test_image_left", type=str,
                       help="Left test image for verification (optional)")
    parser.add_argument("--test_image_right", type=str,
                       help="Right test image for verification (optional)")
    parser.add_argument("--num_images", type=int, default=20,
                       help="Number of image pairs to capture")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between automatic captures in seconds")
    parser.add_argument("--scanner_type", type=str, default="robust",
                       choices=["basic", "enhanced", "robust"],
                       help="Scanner type to use for test scan")
    
    args = parser.parse_args()
    
    # Process based on mode
    if args.mode == "capture":
        capture_calibration_images(
            output_dir=args.output_dir,
            num_images=args.num_images,
            delay=args.delay,
            checkerboard_size=(args.checkerboard_width, args.checkerboard_height),
            visualize=not args.no_visualize
        )
    
    elif args.mode == "calibrate":
        run_calibration(
            input_dir=args.input_dir,
            output_file=args.output_file,
            checkerboard_size=(args.checkerboard_width, args.checkerboard_height),
            square_size=args.square_size,
            visualize=not args.no_visualize
        )
    
    elif args.mode == "verify":
        verify_calibration(
            calibration_file=args.calibration_file,
            test_image_left=args.test_image_left,
            test_image_right=args.test_image_right,
            visualize=not args.no_visualize
        )
    
    elif args.mode == "test_scan":
        test_with_scan(
            calibration_file=args.calibration_file,
            output_dir=args.output_dir,
            scanner_type=args.scanner_type
        )


if __name__ == "__main__":
    main()