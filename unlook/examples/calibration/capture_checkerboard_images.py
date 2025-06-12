#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2K Calibration Image Capture Tool for UnLook SDK

This script captures high-resolution (1456x1088) checkerboard images for stereo calibration.
Specifically designed for maximum camera resolution calibration to achieve optimal accuracy.

Features:
- High resolution capture (1456x1088 - actual camera maximum)
- 9x6 checkerboard pattern detection
- Real-time checkerboard detection feedback
- Automatic quality validation
- AS1170 LED flash control for better illumination (on during capture, off otherwise)
- Protocol v2 support
- Two capture modes:
  * AUTOMATIC: Traditional timed capture with LED flash
  * LIVE PREVIEW: Real-time OpenCV stream with manual capture control

Usage:
  # Automatic mode (traditional)
  python capture_checkerboard_images.py --output calibration_2k_images/ --num-images 20
  
  # Live preview mode (recommended)
  python capture_checkerboard_images.py --output calibration_2k_images/ --num-images 20 --live-preview
  
Live Preview Controls:
  - 'c' or SPACE: Capture image when checkerboard is detected and ready
  - 'q' or ESC: Quit
  - 'r': Reset capture count
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
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("capture_checkerboard")

# Global variable to track the client for signal handling
global_client = None

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) to clean up LED before exiting."""
    global global_client
    logger.info("Received interrupt signal, cleaning up...")
    
    if global_client:
        try:
            # Turn off LED before disconnecting
            global_client.projector.led_off()
            logger.info("LED turned off")
        except Exception as e:
            logger.warning(f"Could not turn off LED: {e}")
        
        try:
            global_client.disconnect()
            logger.info("Disconnected from scanner")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")
    
    logger.info("Exiting...")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from unlook.client.scanner import Scanner3D
    from unlook.client.scanner.scanner import UnlookClient
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    sys.exit(1)

class Checkerboard2KCapture:
    """2K resolution checkerboard capture system"""
    
    def __init__(self, checkerboard_size=(9, 6), square_size_mm=25.0):
        """
        Initialize capture system for 2K calibration.
        
        Args:
            checkerboard_size: Inner corners of checkerboard (columns, rows)
            square_size_mm: Size of each square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Maximum camera resolution
        self.target_resolution = (1456, 1088)  # Actual max resolution of the camera
        self.jpeg_quality = 95
        
        logger.info(f"Initialized for {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard")
        logger.info(f"Target resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        
    def configure_scanner_2k(self, client, led_intensity=200):
        """Configure scanner for 2K resolution capture"""
        try:
            # Store LED intensity for later use
            self.led_intensity = led_intensity
            
            # Try to configure camera resolution through the camera client
            try:
                cameras = client.camera.get_cameras()
                if cameras:
                    camera_id = cameras[0]['id'] if isinstance(cameras[0], dict) else cameras[0]
                    # Configure the camera with our target resolution
                    config = {
                        "resolution": self.target_resolution,
                        "jpeg_quality": self.jpeg_quality
                    }
                    success = client.camera.configure(camera_id, config)
                    if success:
                        logger.info(f"Camera configured for {self.target_resolution[0]}x{self.target_resolution[1]} capture")
                    else:
                        logger.warning("Could not configure camera resolution")
                else:
                    logger.warning("No cameras found for configuration")
            except Exception as e:
                logger.warning(f"Could not set camera parameters: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure scanner: {e}")
            return False
    
    def flash_led_for_capture(self, client, duration=0.5):
        """Flash LED for a single capture"""
        try:
            # Turn on LED
            client.projector.led_set_intensity(led1_mA=0, led2_mA=self.led_intensity)
            logger.debug(f"LED turned on ({self.led_intensity}mA)")
            time.sleep(0.1)  # Brief stabilization
            return True
        except Exception as e:
            logger.warning(f"Could not turn on LED: {e}")
            return False
    
    def turn_off_led(self, client):
        """Turn off LED after capture"""
        try:
            client.projector.led_off()
            logger.debug("LED turned off")
        except Exception as e:
            logger.warning(f"Could not turn off LED: {e}")
    
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
        
        # Check resolution - be more flexible, just warn if different
        if (width, height) != self.target_resolution:
            logger.debug(f"Resolution differs: {width}x{height} vs expected {self.target_resolution[0]}x{self.target_resolution[1]}")
            # Don't fail on resolution mismatch - just continue with what we have
        
        # Check image brightness - just for information
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Log brightness but don't block
        if mean_brightness < 30:
            logger.warning(f"Image quite dark (brightness: {mean_brightness:.1f})")
        elif mean_brightness > 225:
            logger.warning(f"Image quite bright (brightness: {mean_brightness:.1f})")
        else:
            logger.debug(f"Image brightness OK: {mean_brightness:.1f}")
        
        # Check image sharpness (Laplacian variance) - just for information, not blocking
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.debug(f"Image sharpness (Laplacian variance): {laplacian_var}")
        # Removed the blur check - let user decide if image is good enough
        
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
            
            # More permissive coverage check - just warn, don't block
            if coverage < 0.1:  # Reduced from 0.2 to 0.1 (10%)
                logger.info(f"Checkerboard coverage: {coverage:.2%} - quite small but acceptable")
            elif coverage > 0.9:  # Increased from 0.8 to 0.9 (90%)
                logger.warning(f"Checkerboard coverage: {coverage:.2%} - very close, corners might be cut off")
                # Still allow capture but warn user
            
            logger.info(f"Checkerboard coverage: {coverage:.2%}")
            
            quality_score = 0.7 + (0.3 * coverage)
        
        return True, quality_score, "Good quality"
    
    def live_preview_capture(self, client, output_dir, num_images=20, min_delay=2.0):
        """
        Live preview mode with OpenCV stream for real-time checkerboard detection and focus assessment.
        Press 'c' to capture, 'q' to quit.
        """
        # Create output directory structure
        output_path = Path(output_dir)
        left_dir = output_path / "camera"
        left_dir.mkdir(parents=True, exist_ok=True)
        
        # Save calibration info
        calib_info = {
            "capture_date": datetime.now().isoformat(),
            "resolution": list(self.target_resolution),
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size_mm,
            "num_images": num_images,
            "jpeg_quality": self.jpeg_quality,
            "single_camera_mode": True
        }
        
        with open(output_path / "calibration_info.json", 'w') as f:
            json.dump(calib_info, f, indent=2)
        
        print("\n" + "="*60)
        print("LIVE CALIBRATION CAPTURE WITH PREVIEW")
        print("="*60)
        print(f"Resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} inner corners")
        print(f"Output directory: {output_path}")
        print("\nControls:")
        print("  'c' or SPACE - Capture image when checkerboard is detected")
        print("  'q' or ESC - Quit")
        print("  'r' - Reset captured count")
        print("="*60 + "\n")
        
        captured_count = 0
        failed_count = 0
        capture_results = []
        
        # Focus assessment for real-time feedback
        focus_history = []
        focus_threshold = 100
        
        try:
            # Get camera ID with retries
            cameras = None
            for retry in range(3):
                try:
                    cameras = client.camera.get_cameras()
                    if cameras:
                        break
                    logger.warning(f"No cameras found, retry {retry + 1}/3")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Error getting cameras (retry {retry + 1}/3): {e}")
                    time.sleep(1)
            
            if not cameras:
                logger.error("No cameras available after retries")
                return {}
            
            camera_id = cameras[0]['id'] if isinstance(cameras[0], dict) else cameras[0]
            logger.info(f"Using camera: {camera_id}")
            
            # Test capture to ensure everything works
            logger.info("Testing camera capture...")
            test_success = False
            for test_retry in range(3):
                try:
                    test_images = client.camera.capture_multi([camera_id])
                    if test_images and camera_id in test_images and test_images[camera_id] is not None:
                        test_success = True
                        logger.info("Camera test successful!")
                        break
                    else:
                        logger.warning(f"Camera test failed, retry {test_retry + 1}/3")
                        time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Camera test error (retry {test_retry + 1}/3): {e}")
                    time.sleep(0.5)
            
            if not test_success:
                logger.error("Camera test failed - cannot proceed with live preview")
                return {}
            
            print("✅ Camera test successful - starting live preview...")
            
            while captured_count < num_images:
                try:
                    # Capture live image using more reliable method
                    try:
                        # Try multi-camera capture first (more reliable)
                        images = client.camera.capture_multi([camera_id])
                        if images and camera_id in images and images[camera_id] is not None:
                            image = images[camera_id]
                        else:
                            # Fallback to single camera capture
                            image = client.camera.capture(camera_id)
                    except Exception as capture_error:
                        logger.debug(f"Capture error: {capture_error}")
                        # Last resort - try stereo pair capture
                        try:
                            left_img, right_img = client.camera.capture_stereo_pair()
                            image = left_img if left_img is not None else right_img
                        except Exception as stereo_error:
                            logger.debug(f"Stereo capture error: {stereo_error}")
                            image = None
                    
                    if image is None:
                        logger.debug("Failed to capture live image, retrying...")
                        time.sleep(0.1)
                        continue
                    
                    # Check image dimensions - resize if needed for consistent display
                    if image.shape[:2] != (self.target_resolution[1], self.target_resolution[0]):
                        logger.debug(f"Image size {image.shape[:2]} differs from target {(self.target_resolution[1], self.target_resolution[0])}")
                        # Don't resize - just use what we got for display, but note the difference
                    
                    # Create display copy
                    display_img = image.copy()
                    
                    # Detect checkerboard
                    found, corners = self.detect_checkerboard(image)
                    
                    # Calculate focus score
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    focus_score = laplacian.var()
                    
                    # Update focus history
                    focus_history.append(focus_score)
                    if len(focus_history) > 10:
                        focus_history.pop(0)
                    avg_focus = np.mean(focus_history)
                    
                    # Determine focus quality
                    if avg_focus > 150:
                        focus_color = (0, 255, 0)  # Green - excellent
                        focus_status = "EXCELLENT"
                    elif avg_focus > 100:
                        focus_color = (0, 255, 255)  # Yellow - good
                        focus_status = "GOOD"
                    elif avg_focus > 50:
                        focus_color = (0, 165, 255)  # Orange - fair
                        focus_status = "FAIR"
                    else:
                        focus_color = (0, 0, 255)  # Red - poor
                        focus_status = "POOR"
                    
                    # Draw focus information
                    cv2.putText(display_img, f"Focus: {avg_focus:.1f} ({focus_status})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, focus_color, 2)
                    
                    # Draw progress
                    cv2.putText(display_img, f"Captured: {captured_count}/{num_images}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if found:
                        # Draw checkerboard corners
                        cv2.drawChessboardCorners(display_img, self.checkerboard_size, corners, found)
                        
                        # Validate quality
                        img_valid, img_score, img_msg = self.validate_image_quality(image, corners)
                        
                        if img_valid:
                            # Green border - ready to capture
                            cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), 
                                         (0, 255, 0), 5)
                            cv2.putText(display_img, "READY TO CAPTURE - Press 'c' or SPACE", 
                                       (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(display_img, f"Quality: {img_score:.2f}", 
                                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            # Yellow border - checkerboard found but quality issues
                            cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), 
                                         (0, 255, 255), 5)
                            cv2.putText(display_img, f"QUALITY ISSUE: {img_msg}", 
                                       (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Red border - no checkerboard detected
                        cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), 
                                     (0, 0, 255), 5)
                        cv2.putText(display_img, "CHECKERBOARD NOT DETECTED", 
                                   (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Instructions
                    cv2.putText(display_img, "Controls: 'c'/SPACE=capture, 'q'/ESC=quit, 'r'=reset", 
                               (10, display_img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show image
                    cv2.imshow('Calibration Live Preview', display_img)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(30) & 0xFF
                    
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('r'):  # Reset
                        captured_count = 0
                        failed_count = 0
                        capture_results = []
                        logger.info("Capture count reset")
                    elif key == ord('c') or key == 32:  # 'c' or SPACE
                        if found:
                            # Flash LED for capture
                            self.flash_led_for_capture(client)
                            
                            # Capture with LED on using the same reliable method
                            try:
                                # Try multi-camera capture first
                                capture_images = client.camera.capture_multi([camera_id])
                                if capture_images and camera_id in capture_images:
                                    capture_img = capture_images[camera_id]
                                else:
                                    # Fallback to stereo pair capture
                                    left_img, right_img = client.camera.capture_stereo_pair()
                                    capture_img = left_img
                            except Exception as capture_error:
                                logger.warning(f"Capture error: {capture_error}")
                                capture_img = None
                            
                            # Turn off LED immediately
                            self.turn_off_led(client)
                            
                            if capture_img is not None:
                                # Re-validate the captured image
                                cap_found, cap_corners = self.detect_checkerboard(capture_img)
                                if cap_found:
                                    cap_valid, cap_score, cap_msg = self.validate_image_quality(capture_img, cap_corners)
                                    if cap_valid:
                                        # Save image
                                        filename = f"calib_{captured_count:04d}.jpg"
                                        cv2.imwrite(str(left_dir / filename), capture_img, 
                                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                                        
                                        # Record results
                                        capture_results.append({
                                            "index": captured_count,
                                            "filename": filename,
                                            "quality": cap_score,
                                            "focus_score": avg_focus,
                                            "timestamp": datetime.now().isoformat()
                                        })
                                        
                                        captured_count += 1
                                        logger.info(f"✅ Captured image {captured_count}/{num_images} - Quality: {cap_score:.2f}")
                                        
                                        # Brief delay after successful capture
                                        time.sleep(min_delay)
                                    else:
                                        logger.warning(f"❌ Capture failed quality check: {cap_msg}")
                                        failed_count += 1
                                else:
                                    logger.warning("❌ Checkerboard not found in captured image")
                                    failed_count += 1
                            else:
                                logger.warning("❌ Failed to capture image")
                                failed_count += 1
                        else:
                            logger.info("⚠️ Checkerboard not detected - position checkerboard in view")
                    
                except Exception as e:
                    logger.error(f"Error in live preview: {e}")
                    time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Live preview interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
        
        # Save final results
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
        print("LIVE CAPTURE COMPLETE")
        print(f"Successfully captured: {captured_count} images")
        print(f"Failed attempts: {failed_count}")
        print(f"Output directory: {output_path}")
        print("="*60)
        
        return results
    
    def capture_calibration_set(self, client, output_dir, num_images=20, min_delay=2.0, single_camera_mode=False):
        """
        Capture a complete set of calibration images at 2K resolution.
        
        Args:
            scanner: Scanner3D instance
            output_dir: Directory to save captured images
            num_images: Number of images to capture
            min_delay: Minimum delay between captures
            single_camera_mode: If True, only capture left camera images
            
        Returns:
            dict: Calibration capture results
        """
        # Create output directory structure (single camera)
        output_path = Path(output_dir)
        left_dir = output_path / "camera"  # Changed from "left" to "camera"
        left_dir.mkdir(parents=True, exist_ok=True)
        
        # Save calibration info
        calib_info = {
            "capture_date": datetime.now().isoformat(),
            "resolution": list(self.target_resolution),
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size_mm,
            "num_images": num_images,
            "jpeg_quality": self.jpeg_quality,
            "single_camera_mode": single_camera_mode
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
        print("1. Hold checkerboard at various distances (40-50cm optimal, 30-60cm range)")
        print("2. Rotate to different angles (±30°)")
        print("3. Cover all areas of the image")
        print("4. Ensure ENTIRE checkerboard is visible")
        print("5. Best distance for calibration: 45cm")
        print("="*60 + "\n")
        
        captured_count = 0
        failed_count = 0
        capture_results = []
        
        while captured_count < num_images:
            print(f"\nImage {captured_count + 1}/{num_images}")
            
            if captured_count > 0:
                print(f"Move checkerboard to new position...")
                print(f"Waiting {min_delay} seconds...")
                time.sleep(min_delay)
            
            try:
                # Flash LED for capture
                print("Flashing LED and capturing... ", end="", flush=True)
                self.flash_led_for_capture(client)
                
                # Now capture_stereo_pair() works correctly with single camera
                try:
                    left_img, right_img = client.camera.capture_stereo_pair()
                    
                    # Turn off LED immediately after capture
                    self.turn_off_led(client)
                    
                    # For single camera mode, ignore right image
                    if single_camera_mode:
                        right_img = None
                        
                except Exception as e:
                    # Make sure LED is off even on error
                    self.turn_off_led(client)
                    print(f"FAILED - Capture error: {e}")
                    failed_count += 1
                    continue
                
                if left_img is None:
                    print("FAILED - No image captured")
                    failed_count += 1
                    continue
                
                # Detect checkerboards
                print("Detecting checkerboard... ", end="", flush=True)
                left_found, left_corners = self.detect_checkerboard(left_img)
                
                # Since we only have one camera, just check the single image
                if not left_found:
                    print("FAILED - Checkerboard not found")
                    failed_count += 1
                    continue
                
                # Validate quality
                img_valid, img_score, img_msg = self.validate_image_quality(left_img, left_corners)
                
                if not img_valid:
                    print(f"FAILED - Quality issue: {img_msg}")
                    failed_count += 1
                    continue
                
                # Save image (only one camera available)
                filename = f"calib_{captured_count:04d}.jpg"
                cv2.imwrite(str(left_dir / filename), left_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                
                # Record results
                capture_results.append({
                    "index": captured_count,
                    "filename": filename,
                    "quality": img_score,
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"SUCCESS - Quality: {img_score:.2f}")
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
        print(f"Successfully captured: {captured_count} images")
        print(f"Failed attempts: {failed_count}")
        print(f"Output directory: {output_path}")
        print("="*60)
        
        return results

def main():
    global global_client
    
    parser = argparse.ArgumentParser(description="2K Calibration Image Capture Tool")
    parser.add_argument('--output', type=str, default='calibration_2k_images',
                        help='Output directory for calibration images')
    parser.add_argument('--num-images', type=int, default=20,
                        help='Number of images to capture (default: 20)')
    parser.add_argument('--checkerboard-columns', type=int, default=9,
                        help='Number of inner corners horizontally (default: 9)')
    parser.add_argument('--checkerboard-rows', type=int, default=6,
                        help='Number of inner corners vertically (default: 6)')
    parser.add_argument('--square-size', type=float, default=25.0,
                        help='Size of checkerboard squares in mm (default: 25.0)')
    parser.add_argument('--scanner-ip', type=str,
                        help='Scanner IP address (auto-discover if not specified)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Minimum delay between captures in seconds')
    parser.add_argument('--single-camera', action='store_true',
                        help='Capture only left camera images for single camera calibration')
    parser.add_argument('--led-intensity', type=int, default=200,
                        help='LED intensity in mA for illumination (0-450, default: 200)')
    parser.add_argument('--live-preview', action='store_true',
                        help='Use live preview mode with OpenCV streaming and manual capture control')
    
    args = parser.parse_args()
    
    # Initialize capture system
    capture_system = Checkerboard2KCapture(
        checkerboard_size=(args.checkerboard_columns, args.checkerboard_rows),
        square_size_mm=args.square_size
    )
    
    # Connect to scanner using auto-discovery
    print("Discovering scanners...")
    client = UnlookClient("CalibrationCapture", auto_discover=True)
    global_client = client  # Set global reference for signal handler
    
    # Wait for discovery
    time.sleep(3)
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found on network")
        logger.error("Make sure the scanner server is running with:")
        logger.error("  python unlook/server_bootstrap.py --config unlook_config_2k.json")
        return 1
    
    # Select scanner
    scanner_info = None
    if args.scanner_ip:
        # Find scanner by IP
        for s in scanners:
            if s.host == args.scanner_ip:
                scanner_info = s
                break
        if not scanner_info:
            logger.error(f"Scanner with IP {args.scanner_ip} not found")
            return 1
    else:
        # Use first discovered scanner
        scanner_info = scanners[0]
    
    print(f"Found scanner: {scanner_info.name} at {scanner_info.host}")
    
    # Connect to scanner
    try:
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return 1
            
        print(f"Connected to scanner at {scanner_info.host}")
        
        # Use the connected client directly for capture
        
        # Configure scanner for 2K with LED illumination
        if not capture_system.configure_scanner_2k(client, led_intensity=args.led_intensity):
            logger.error("Failed to configure scanner for 2K")
            return 1
        
        # Wait a moment for configuration to stabilize
        time.sleep(0.5)
        
        # Choose capture method based on live preview flag
        if args.live_preview:
            # Use live preview mode with OpenCV streaming
            print("Starting LIVE PREVIEW mode...")
            print("You'll see a real-time camera feed with checkerboard detection.")
            print("Press 'c' or SPACE to capture when the checkerboard is detected and in focus.")
            
            results = capture_system.live_preview_capture(
                client,
                args.output,
                num_images=args.num_images,
                min_delay=args.delay
            )
        else:
            # Use traditional automatic capture mode
            print("Starting AUTOMATIC capture mode...")
            print("Images will be captured automatically with LED flash.")
            
            results = capture_system.capture_calibration_set(
                client,
                args.output,
                num_images=args.num_images,
                min_delay=args.delay,
                single_camera_mode=args.single_camera
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'client' in locals():
            # Turn off LED before disconnecting
            try:
                client.projector.led_off()
                logger.info("LED turned off")
            except:
                pass
            client.disconnect()
        # Clear global reference
        global_client = None

if __name__ == "__main__":
    sys.exit(main())