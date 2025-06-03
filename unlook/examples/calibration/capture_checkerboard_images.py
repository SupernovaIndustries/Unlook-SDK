#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2K Calibration Image Capture Tool for UnLook SDK

This script captures high-resolution (2048x1536) checkerboard images for stereo calibration.
Specifically designed for 2K resolution calibration to achieve maximum accuracy.

Features:
- 2K resolution capture (2048x1536)
- 5x8 checkerboard pattern detection
- Real-time checkerboard detection feedback
- Automatic quality validation
- Server configuration using unlook_config_2k.json

Usage:
  python capture_checkerboard_images.py --output calibration_2k_images/ --num-images 40
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("capture_checkerboard_2k")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from unlook import UnlookClient
    from unlook.core.discovery import discover_scanners
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    sys.exit(1)

class Checkerboard2KCapture:
    """2K resolution checkerboard capture system"""
    
    def __init__(self, checkerboard_size=(8, 5), square_size_mm=24.0):
        """
        Initialize capture system for 2K calibration.
        
        Args:
            checkerboard_size: Inner corners of checkerboard (columns, rows)
            square_size_mm: Size of each square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 2K resolution settings
        self.target_resolution = (2048, 1536)
        self.jpeg_quality = 95
        
        logger.info(f"Initialized for {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard")
        logger.info(f"Target resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        
    def configure_scanner_2k(self, scanner):
        """Configure scanner for 2K resolution capture"""
        try:
            # Load 2K configuration
            config_path = Path(__file__).parent.parent.parent.parent / "unlook_config_2k.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_2k = json.load(f)
                    logger.info("Loaded 2K configuration from unlook_config_2k.json")
            else:
                logger.warning("2K config file not found, using hardcoded settings")
                config_2k = None
            
            # Apply 2K camera settings
            scanner.set_camera_resolution(self.target_resolution[0], self.target_resolution[1])
            scanner.set_jpeg_quality(self.jpeg_quality)
            scanner.set_camera_fps(15)  # Reduced FPS for 2K stability
            
            # Configure exposure for calibration
            scanner.set_camera_exposure_mode('manual')
            scanner.set_camera_exposure_value(15000)  # Fixed exposure for consistency
            
            logger.info("Scanner configured for 2K calibration capture")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure scanner: {e}")
            return False
    
    def detect_checkerboard(self, image):
        """
        Detect checkerboard corners in image.
        
        Returns:
            tuple: (success, corners) where success is boolean and corners are the detected points
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corners for sub-pixel accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
        return ret, corners
    
    def validate_image_quality(self, image, corners=None):
        """
        Validate captured image quality for calibration.
        
        Returns:
            tuple: (is_valid, quality_score, message)
        """
        height, width = image.shape[:2]
        
        # Check resolution
        if (width, height) != self.target_resolution:
            return False, 0.0, f"Wrong resolution: {width}x{height}, expected {self.target_resolution[0]}x{self.target_resolution[1]}"
        
        # Check image brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            return False, 0.2, "Image too dark"
        elif mean_brightness > 200:
            return False, 0.2, "Image too bright"
        
        # Check image sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return False, 0.3, "Image too blurry"
        
        # Check checkerboard coverage if corners provided
        quality_score = 0.7  # Base score for good brightness and sharpness
        
        if corners is not None:
            # Calculate checkerboard area coverage
            min_x = np.min(corners[:, 0, 0])
            max_x = np.max(corners[:, 0, 0])
            min_y = np.min(corners[:, 0, 1])
            max_y = np.max(corners[:, 0, 1])
            
            coverage_x = (max_x - min_x) / width
            coverage_y = (max_y - min_y) / height
            coverage = coverage_x * coverage_y
            
            if coverage < 0.2:
                return False, 0.4, "Checkerboard too small in frame"
            elif coverage > 0.8:
                return False, 0.4, "Checkerboard too close (may be cut off)"
            
            quality_score = 0.7 + (0.3 * coverage)
        
        return True, quality_score, "Good quality"
    
    def capture_calibration_set(self, client, output_dir, num_images=40, min_delay=2.0):
        """
        Capture a complete set of calibration images at 2K resolution.
        
        Args:
            client: UnlookClient instance
            output_dir: Directory to save captured images
            num_images: Number of image pairs to capture
            min_delay: Minimum delay between captures
            
        Returns:
            dict: Calibration capture results
        """
        # Create output directory structure
        output_path = Path(output_dir)
        left_dir = output_path / "left"
        right_dir = output_path / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        
        # Save calibration info
        calib_info = {
            "capture_date": datetime.now().isoformat(),
            "resolution": list(self.target_resolution),
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size_mm,
            "num_images": num_images,
            "jpeg_quality": self.jpeg_quality
        }
        
        with open(output_path / "calibration_info.json", 'w') as f:
            json.dump(calib_info, f, indent=2)
        
        print("\n" + "="*60)
        print("2K CALIBRATION IMAGE CAPTURE")
        print("="*60)
        print(f"Resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} inner corners")
        print(f"Output directory: {output_path}")
        print("\nInstructions:")
        print("1. Hold checkerboard at various distances (30-80cm)")
        print("2. Rotate to different angles (±30°)")
        print("3. Cover all areas of the image")
        print("4. Ensure ENTIRE checkerboard is visible")
        print("="*60 + "\n")
        
        captured_count = 0
        failed_count = 0
        capture_results = []
        
        while captured_count < num_images:
            print(f"\nImage pair {captured_count + 1}/{num_images}")
            
            if captured_count > 0:
                print(f"Move checkerboard to new position...")
                print(f"Waiting {min_delay} seconds...")
                time.sleep(min_delay)
            
            try:
                # Capture stereo pair
                print("Capturing... ", end="", flush=True)
                left_img, right_img = client.camera.capture_stereo_pair()
                
                if left_img is None or right_img is None:
                    print("FAILED - No image received")
                    failed_count += 1
                    continue
                
                # Detect checkerboards
                print("Detecting checkerboard... ", end="", flush=True)
                left_found, left_corners = self.detect_checkerboard(left_img)
                right_found, right_corners = self.detect_checkerboard(right_img)
                
                if not left_found or not right_found:
                    print("FAILED - Checkerboard not found in both images")
                    failed_count += 1
                    continue
                
                # Validate quality
                left_valid, left_score, left_msg = self.validate_image_quality(left_img, left_corners)
                right_valid, right_score, right_msg = self.validate_image_quality(right_img, right_corners)
                
                if not left_valid or not right_valid:
                    print(f"FAILED - Quality issue: L:{left_msg}, R:{right_msg}")
                    failed_count += 1
                    continue
                
                # Save images
                filename = f"calib_{captured_count:04d}.jpg"
                cv2.imwrite(str(left_dir / filename), left_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                cv2.imwrite(str(right_dir / filename), right_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                
                # Record results
                capture_results.append({
                    "index": captured_count,
                    "filename": filename,
                    "left_quality": left_score,
                    "right_quality": right_score,
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"SUCCESS - Quality: L:{left_score:.2f}, R:{right_score:.2f}")
                captured_count += 1
                
            except Exception as e:
                print(f"ERROR - {str(e)}")
                failed_count += 1
                
            # Show progress
            if failed_count > 0:
                print(f"Progress: {captured_count}/{num_images} captured, {failed_count} failed attempts")
        
        # Save capture results
        results = {
            "capture_info": calib_info,
            "statistics": {
                "captured": captured_count,
                "failed_attempts": failed_count,
                "success_rate": captured_count / (captured_count + failed_count) if (captured_count + failed_count) > 0 else 0
            },
            "captures": capture_results
        }
        
        with open(output_path / "capture_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("CAPTURE COMPLETE")
        print(f"Successfully captured: {captured_count} image pairs")
        print(f"Failed attempts: {failed_count}")
        print(f"Output directory: {output_path}")
        print("="*60)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="2K Calibration Image Capture Tool")
    parser.add_argument('--output', type=str, default='calibration_2k_images',
                        help='Output directory for calibration images')
    parser.add_argument('--num-images', type=int, default=40,
                        help='Number of image pairs to capture (default: 40)')
    parser.add_argument('--checkerboard-columns', type=int, default=8,
                        help='Number of inner corners horizontally (default: 8)')
    parser.add_argument('--checkerboard-rows', type=int, default=5,
                        help='Number of inner corners vertically (default: 5)')
    parser.add_argument('--square-size', type=float, default=24.0,
                        help='Size of checkerboard squares in mm (default: 24.0)')
    parser.add_argument('--scanner-ip', type=str,
                        help='Scanner IP address (auto-discover if not specified)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Minimum delay between captures in seconds')
    
    args = parser.parse_args()
    
    # Initialize capture system
    capture_system = Checkerboard2KCapture(
        checkerboard_size=(args.checkerboard_columns, args.checkerboard_rows),
        square_size_mm=args.square_size
    )
    
    # Connect to scanner
    if args.scanner_ip:
        scanner_ip = args.scanner_ip
    else:
        print("Discovering scanners...")
        scanners = discover_scanners(timeout=5.0)
        if not scanners:
            logger.error("No scanners found on network")
            return 1
        scanner_ip = scanners[0]['ip']
        print(f"Found scanner at {scanner_ip}")
    
    # Create client
    try:
        client = UnlookClient(scanner_ip)
        print(f"Connected to scanner at {scanner_ip}")
        
        # Configure for 2K
        if not capture_system.configure_scanner_2k(client):
            logger.error("Failed to configure scanner for 2K")
            return 1
        
        # Capture calibration images
        results = capture_system.capture_calibration_set(
            client,
            args.output,
            num_images=args.num_images,
            min_delay=args.delay
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        return 1
    finally:
        if 'client' in locals():
            client.disconnect()

if __name__ == "__main__":
    sys.exit(main())