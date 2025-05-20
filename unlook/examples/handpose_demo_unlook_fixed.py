#!/usr/bin/env python3
"""Hand pose detection using UnLook hardware cameras - Fixed version with LED control."""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import UnLook SDK
from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker

def preprocess_image(image):
    """
    Enhance image for better hand detection in poor lighting.
    
    This function:
    1. Increases brightness and contrast
    2. Applies adaptive histogram equalization for better feature visibility
    3. Applies light bilateral filtering to reduce noise while preserving edges
    
    Args:
        image: Input BGR image
        
    Returns:
        Enhanced image
    """
    # Skip preprocessing if image is None
    if image is None:
        return None
    
    # Make a copy to avoid modifying the original
    enhanced = image.copy()
    
    # Convert to LAB color space (L = lightness)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to lightness channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back and convert to BGR
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
    
    return enhanced


def auto_adjust_led_intensity(client, frame_left, frame_right, current_intensity=450, steps=5):
    """
    Automatically adjust LED2 intensity for optimal hand detection.
    
    This function:
    1. Starts with the current intensity
    2. Tests multiple intensity levels
    3. Evaluates image quality for each intensity
    4. Returns the optimal intensity for LED2
    5. Always sets LED1 to 0 for hardware safety
    
    Args:
        client: UnlookClient instance
        frame_left: Left camera frame
        frame_right: Right camera frame
        current_intensity: Current LED2 intensity in mA
        steps: Number of intensity levels to test
        
    Returns:
        Optimal LED2 intensity in mA (LED1 will always be 0)
    """
    from unlook.core import MessageType
    
    logger.info("Starting automatic LED intensity calibration...")
    
    # Define test range (from 40% to 100% of max)
    min_intensity = 180  # 40% of max (450)
    max_intensity = 450  # Maximum supported by hardware
    
    # Generate test intensities
    if current_intensity > max_intensity:
        current_intensity = max_intensity
    
    test_intensities = np.linspace(min_intensity, max_intensity, steps).astype(int)
    best_score = -1
    optimal_intensity = current_intensity
    
    # Measure starting quality
    starting_quality = evaluate_image_quality(frame_left) + evaluate_image_quality(frame_right)
    logger.info(f"Starting image quality: {starting_quality:.2f}")
    
    for intensity in test_intensities:
        logger.info(f"Testing LED intensity: {intensity}mA")
        
        # Set LED intensity - always set LED1 to 0
        success, response, _ = client.send_message(MessageType.LED_SET_INTENSITY, {
            'led1': 0,  # Always 0 for hardware safety
            'led2': intensity  # Use intensity for LED2
        })
        
        if not success:
            logger.warning(f"Failed to set LED intensity to {intensity}mA")
            continue
            
        # Wait for camera exposure to adjust
        time.sleep(0.5)
        
        # Capture frames
        frame_buffer = {'left': None, 'right': None}
        frame_lock = {'left': False, 'right': False}
        
        def temp_cb_left(frame, _):
            frame_buffer['left'] = frame
            frame_lock['left'] = True
            
        def temp_cb_right(frame, _):
            frame_buffer['right'] = frame
            frame_lock['right'] = True
        
        # Register temporary callbacks
        client.stream.register_callback(temp_cb_left)
        client.stream.register_callback(temp_cb_right)
        
        # Wait for frames
        timeout = 0
        while not (frame_lock['left'] and frame_lock['right']) and timeout < 20:
            time.sleep(0.1)
            timeout += 1
        
        # Evaluate quality
        if frame_buffer['left'] is not None and frame_buffer['right'] is not None:
            quality_left = evaluate_image_quality(frame_buffer['left'])
            quality_right = evaluate_image_quality(frame_buffer['right'])
            total_quality = quality_left + quality_right
            
            logger.info(f"Quality at {intensity}mA: {total_quality:.2f} (L: {quality_left:.2f}, R: {quality_right:.2f})")
            
            # Update best if better
            if total_quality > best_score:
                best_score = total_quality
                optimal_intensity = intensity
    
    logger.info(f"Optimal LED2 intensity: {optimal_intensity}mA (LED1 will be 0, quality score: {best_score:.2f})")
    
    # Set optimal intensity - always set LED1 to 0
    success, response, _ = client.send_message(MessageType.LED_SET_INTENSITY, {
        'led1': 0,  # Always 0 for hardware safety
        'led2': optimal_intensity  # Use optimal intensity for LED2
    })
    
    return optimal_intensity


def evaluate_image_quality(image):
    """
    Evaluate image quality for hand detection.
    
    This function calculates a score based on:
    1. Image contrast (higher is better)
    2. Image brightness (medium is better)
    3. Edge detail (higher is better)
    
    Args:
        image: Input image
        
    Returns:
        Quality score (higher is better)
    """
    if image is None:
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    # Calculate brightness (mean)
    brightness = np.mean(gray)
    
    # Calculate edge detail (Sobel gradient magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_detail = np.mean(edge_magnitude)
    
    # Penalize too bright or too dark images (ideal around 127)
    brightness_penalty = 1.0 - abs(brightness - 127) / 127
    
    # Calculate score (weighted sum)
    score = (contrast * 0.5) + (edge_detail * 0.3) + (brightness_penalty * 100)
    
    return score

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LED control is done through the server


def run_unlook_hand_tracking_demo(calibration_file=None, output_file=None, visualize_3d=False, timeout=10, verbose=False,
                                  use_led=True, led1_intensity=0, led2_intensity=200, auto_led_adjust=True):
    # Note: led1_intensity parameter is kept for backward compatibility but is always set to 0
    """
    Run 3D hand tracking demo using UnLook scanner cameras.
    
    Args:
        calibration_file: Path to stereo calibration file
        output_file: Path to save tracking data
        visualize_3d: Enable 3D visualization
        timeout: Discovery timeout in seconds
        verbose: Enable verbose debug logging
        use_led: Enable AS1170 LED control if available
        led1_intensity: LED1 intensity in mA (0-450)
        led2_intensity: LED2 intensity in mA (0-450)
    """
    
    # Initialize LED through server if requested
    if use_led:
        try:
            logger.info(f"Initializing LED control (LED1={led1_intensity}mA, LED2={led2_intensity}mA)")
            # Will be initialized after client connection
        except Exception as e:
            logger.error(f"Failed to initialize LED: {e}")
            use_led = False
    
    # Initialize UnLook client
    logger.info("Initializing UnLook client...")
    client = UnlookClient(auto_discover=False)
    
    # Follow the same pattern as static_scanning_example_fixed.py
    try:
        # Start discovery
        client.start_discovery()
        logger.info(f"Discovering scanners for {timeout} seconds...")
        time.sleep(timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Check that your hardware is connected and powered on.")
            return 1
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner.")
            return 1
        
        logger.info(f"Successfully connected to scanner: {scanner_info.name}")
        
        # Initialize LED control through server if requested
        if use_led:
            try:
                # Send LED control command to server - always set LED1 to 0
                from unlook.core import MessageType
                success, response, _ = client.send_message(MessageType.LED_SET_INTENSITY, {
                    'led1': 0,  # Always 0 for hardware safety
                    'led2': led2_intensity
                })
                if success and response and response.payload.get('status') == 'success':
                    logger.info(f"LED flood illuminator activated on server (LED1=0mA, LED2={led2_intensity}mA)")
                else:
                    logger.warning("Failed to activate LED on server")
                    use_led = False
            except Exception as e:
                logger.error(f"Failed to control LED on server: {e}")
                use_led = False
        
        # Get auto-loaded calibration if no calibration file specified
        if not calibration_file:
            calibration_file = client.camera.get_calibration_file_path()
            if calibration_file:
                logger.info(f"Using auto-loaded calibration: {calibration_file}")
            else:
                logger.warning("No calibration file available")
        
        # Initialize hand tracker with much lower confidence thresholds for better detection
        tracker = HandTracker(
            calibration_file=calibration_file, 
            max_num_hands=4,  # Allow more hands for better detection
            detection_confidence=0.2,  # Even lower threshold for initial detection
            tracking_confidence=0.2    # Even lower threshold for tracking
        )
        
        # Configure cameras
        logger.info("Configuring cameras...")
        cameras = client.camera.get_cameras()
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return 1
        
        # Use first two cameras as left and right
        left_camera = cameras[0]['id']
        right_camera = cameras[1]['id']
        camera_names = [cameras[0]['name'], cameras[1]['name']]
        logger.info(f"Using cameras: {camera_names[0]} (left), {camera_names[1]} (right)")
        
        print("\nStarting UnLook 3D hand tracking demo...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame data")
        print("Press 'v' to toggle 3D visualization")
        print("Press 'd' to toggle debug mode (show raw images)")
        print("Press 'r' to recalibrate LED intensity for current lighting")
        print("\nTips for better detection:")
        print("- Ensure hands are well-lit and visible to both cameras")
        print("- Keep hands within 30-60cm from cameras")
        print("- Try adjusting camera angles for better coverage")
        
        show_3d = visualize_3d
        debug_mode = False  # Toggle for showing raw images
        frame_count = 0
        fps_timer = time.time()
        fps_count = 0
        current_fps = 0
        
        # Use streaming with callback instead of direct capture
        # Create storage for frames
        frame_buffer = {'left': None, 'right': None}
        frame_lock = {'left': False, 'right': False}
        
        def frame_callback_left(frame, metadata):
            frame_buffer['left'] = frame
            frame_lock['left'] = True
            
        def frame_callback_right(frame, metadata):
            frame_buffer['right'] = frame
            frame_lock['right'] = True
        
        # Note: Camera synchronization may not be available in all versions
        # client.camera.set_sync_mode(enabled=True)  # Commented out - method doesn't exist
        
        # Start streaming
        logger.info("Starting camera streams...")
        client.stream.start(left_camera, frame_callback_left, fps=30)
        client.stream.start(right_camera, frame_callback_right, fps=30)
        
        # Give streams time to start
        time.sleep(0.5)
        
        # Auto-adjust LED intensity if requested
        if use_led and auto_led_adjust:
            # Wait for initial frames 
            timeout_count = 0
            while not (frame_lock['left'] and frame_lock['right']):
                time.sleep(0.1)
                timeout_count += 1
                if timeout_count > 30:  # 3 seconds timeout
                    logger.warning("Timeout waiting for frames for LED calibration")
                    break
            
            # Get frames for calibration
            frame_left_cal = frame_buffer['left']
            frame_right_cal = frame_buffer['right']
            
            # Reset locks
            frame_lock['left'] = False 
            frame_lock['right'] = False
            
            if frame_left_cal is not None and frame_right_cal is not None:
                try:
                    # Convert to grayscale for brightness analysis
                    gray_left = cv2.cvtColor(frame_left_cal, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(frame_right_cal, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate average brightness
                    brightness_left = np.mean(gray_left)
                    brightness_right = np.mean(gray_right)
                    avg_brightness = (brightness_left + brightness_right) / 2
                    
                    # Target brightness for optimal hand tracking
                    target_brightness = 100  # Target grayscale value (0-255)
                    
                    # Calculate LED intensity based on brightness (inverse relationship)
                    # Brighter scene = lower LED needed
                    adjustment_factor = target_brightness / max(1, avg_brightness)
                    optimal_intensity = int(250 * adjustment_factor)
                    
                    # Clamp to valid range and quantize
                    optimal_intensity = max(150, min(450, optimal_intensity))
                    optimal_intensity = int(round(optimal_intensity / 50.0) * 50)
                    
                    logger.info(f"Image brightness: {avg_brightness:.1f}, adjustment: {adjustment_factor:.2f}")
                    logger.info(f"Calculated optimal LED intensity: {optimal_intensity}mA")
                    
                    # Set LED to calculated optimal intensity
                    try:
                        from unlook.core import MessageType
                        success, response, _ = client.send_message(MessageType.LED_SET_INTENSITY, {
                            'led1': 0,  # Always 0 for hardware safety
                            'led2': optimal_intensity  # Use optimal intensity for LED2
                        })
                        if success:
                            # Store the optimal intensity for LED2 (LED1 is always 0)
                            led2_intensity = optimal_intensity
                            logger.info(f"Auto-adjusted LED2 intensity to {led2_intensity}mA (LED1 remains at 0mA)")
                        else:
                            logger.warning(f"Could not set LED2 intensity to {optimal_intensity}mA")
                    except Exception as e:
                        logger.error(f"LED adjustment error: {e}")
                except Exception as e:
                    logger.error(f"LED calibration error: {e}")
            else:
                logger.warning("No frames available for LED calibration")
        
        while True:
            # Wait for both frames with timeout
            timeout_count = 0
            while not (frame_lock['left'] and frame_lock['right']):
                time.sleep(0.001)
                timeout_count += 1
                if timeout_count > 100:  # 100ms timeout
                    logger.debug("Frame sync timeout")
                    frame_lock['left'] = False
                    frame_lock['right'] = False
                    break
                
            # Get frames
            frame_left = frame_buffer['left']
            frame_right = frame_buffer['right']
            
            # Reset locks
            frame_lock['left'] = False
            frame_lock['right'] = False
            
            if frame_left is None or frame_right is None:
                logger.warning("Failed to get frames from cameras")
                continue
            
            # Preprocess images to improve hand detection in poor lighting conditions
            frame_left_enhanced = preprocess_image(frame_left)
            frame_right_enhanced = preprocess_image(frame_right)
            
            # Track hands in 3D using enhanced images
            results = tracker.track_hands_3d(frame_left_enhanced, frame_right_enhanced)
            
            # Debug print
            if frame_count % 30 == 0:  # Print every second
                print(f"\nFrame {frame_count}:")
                print(f"  2D left hands: {len(results['2d_left'])}, 2D right hands: {len(results['2d_right'])}")
                print(f"  3D hands: {len(results['3d_keypoints'])}, Gestures: {len(results.get('gestures', []))}")
                
                # Show details for each camera
                if results['2d_left']:
                    print(f"  Left camera:")
                    for i, hand in enumerate(results['2d_left']):
                        if 'handedness_left' in results and i < len(results['handedness_left']):
                            handedness = results['handedness_left'][i]
                        else:
                            handedness = "Unknown"
                        print(f"    Hand {i}: {handedness}, shape: {hand.shape}")
                
                if results['2d_right']:
                    print(f"  Right camera:")
                    for i, hand in enumerate(results['2d_right']):
                        if 'handedness_right' in results and i < len(results['handedness_right']):
                            handedness = results['handedness_right'][i] 
                        else:
                            handedness = "Unknown"
                        print(f"    Hand {i}: {handedness}, shape: {hand.shape}")
                
                if results['3d_keypoints']:
                    print(f"  3D matches:")
                    for i, hand in enumerate(results['3d_keypoints']):
                        print(f"    3D hand {i}: {hand.shape}")
                else:
                    print("  No 3D hands matched")
            
            # Create display - show left and right images side by side
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            # Define hand connections (MediaPipe hand landmarks)
            connections = [
                # Thumb
                [0, 1], [1, 2], [2, 3], [3, 4],
                # Index finger  
                [0, 5], [5, 6], [6, 7], [7, 8],
                # Middle finger
                [0, 9], [9, 10], [10, 11], [11, 12],
                # Ring finger
                [0, 13], [13, 14], [14, 15], [15, 16],
                # Pinky
                [0, 17], [17, 18], [18, 19], [19, 20],
                # Palm
                [5, 9], [9, 13], [13, 17]
            ]
            
            # Draw 2D detections on each image
            for i, left_kpts in enumerate(results['2d_left']):
                # Draw left camera keypoints
                h, w = display_left.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(left_kpts, w, h)
                for pt in pixel_coords:
                    cv2.circle(display_left, tuple(pt[:2].astype(int)), 3, (0, 255, 0), -1)
                
                # Draw connections
                for connection in connections:
                    pt1 = pixel_coords[connection[0]][:2].astype(int)
                    pt2 = pixel_coords[connection[1]][:2].astype(int)
                    cv2.line(display_left, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
            
            for i, right_kpts in enumerate(results['2d_right']):
                # Draw right camera keypoints  
                h, w = display_right.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(right_kpts, w, h)
                for pt in pixel_coords:
                    cv2.circle(display_right, tuple(pt[:2].astype(int)), 3, (0, 255, 0), -1)
                
                # Draw connections
                for connection in connections:
                    pt1 = pixel_coords[connection[0]][:2].astype(int)
                    pt2 = pixel_coords[connection[1]][:2].astype(int)
                    cv2.line(display_right, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
            
            # Combine images
            stereo_image = np.hstack([display_left, display_right])
            
            # Calculate FPS
            fps_count += 1
            if time.time() - fps_timer > 1.0:
                current_fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            
            # Add tracking info
            info_text = f"Frame: {frame_count} | FPS: {current_fps} | 3D Hands: {len(results['3d_keypoints'])}"
            cv2.putText(stereo_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera info
            cam_info = f"Cameras: {camera_names[0]} | {camera_names[1]}"
            cv2.putText(stereo_image, cam_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add gesture info
            if 'gestures' in results and results['gestures']:
                gesture_y = 90
                for i, gesture_info in enumerate(results['gestures']):
                    if gesture_info['type'].value != 'unknown':
                        gesture_text = f"Hand {i}: {gesture_info['name']} ({gesture_info['confidence']:.2f})"
                        cv2.putText(stereo_image, gesture_text, (10, gesture_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        gesture_y += 30
            
            cv2.imshow('UnLook Hand Tracking', stereo_image)
            
            # Show debug mode if enabled (raw camera images)
            if debug_mode:
                debug_display = np.zeros((h, w*2, 3), dtype=np.uint8)
                debug_display[:, :w] = frame_left
                debug_display[:, w:] = frame_right
                cv2.line(debug_display, (w, 0), (w, h), (255, 255, 255), 1)
                cv2.putText(debug_display, "LEFT RAW", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(debug_display, "RIGHT RAW", (w+10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Debug - Raw Cameras', debug_display)
            
            # Show 3D visualization if enabled
            if show_3d and results['3d_keypoints']:
                viz_3d = tracker.visualize_3d_hands(results)
                if viz_3d is not None:
                    cv2.imshow('3D Visualization', viz_3d)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame data
                timestamp = int(time.time())
                filename = f"unlook_hand_tracking_{timestamp}.json"
                tracker.save_tracking_data(filename)
                logger.info(f"Saved tracking data to {filename}")
            elif key == ord('v'):
                show_3d = not show_3d
                logger.info(f"3D visualization: {'ON' if show_3d else 'OFF'}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                logger.info(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                if not debug_mode:
                    cv2.destroyWindow('Debug - Raw Cameras')
            elif key == ord('r') and use_led:
                # Recalibrate LED intensity based on current image brightness
                if frame_left is not None and frame_right is not None:
                    try:
                        # Calculate brightness and optimal LED intensity
                        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
                        brightness = (np.mean(gray_left) + np.mean(gray_right)) / 2
                        
                        # Inverse relationship between brightness and needed LED intensity
                        adjustment = 100 / max(1, brightness)  # Target brightness of 100
                        intensity = int(250 * adjustment)
                        intensity = max(150, min(450, intensity))
                        intensity = int(round(intensity / 50.0) * 50)  # Step of 50mA
                        
                        # Apply new LED intensity - always set LED1 to 0
                        from unlook.core import MessageType
                        success, response, _ = client.send_message(MessageType.LED_SET_INTENSITY, {
                            'led1': 0,  # Always 0 for hardware safety
                            'led2': intensity
                        })
                        
                        if success:
                            led1_intensity = intensity
                            logger.info(f"Recalibrated LED intensity to {led1_intensity}mA")
                        else:
                            logger.warning("Failed to adjust LED intensity")
                    except Exception as e:
                        logger.error(f"LED recalibration error: {e}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during operation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save all tracking data if output file specified
        if output_file:
            tracker.save_tracking_data(output_file)
            logger.info(f"Saved all tracking data to {output_file}")
        
        # Cleanup LED if it was used
        if use_led:
            try:
                from unlook.core import MessageType
                success, response, _ = client.send_message(MessageType.LED_OFF, {})
                if success and response and response.payload.get('status') == 'success':
                    logger.info("LED flood illuminator deactivated on server")
            except Exception as e:
                logger.error(f"Failed to turn off LED on server: {e}")
        
        # Cleanup
        logger.info("Cleaning up...")
        try:
            client.stream.stop_stream(left_camera)
            client.stream.stop_stream(right_camera)
        except:
            pass
        client.disconnect()
        client.stop_discovery()
        cv2.destroyAllWindows()
        tracker.close()


def main():
    parser = argparse.ArgumentParser(description='UnLook hand pose tracking demo')
    parser.add_argument('--calibration', type=str,
                       help='Path to stereo calibration file')
    parser.add_argument('--output', type=str,
                       help='Output file for saving tracking data')
    parser.add_argument('--visualize-3d', action='store_true',
                       help='Show 3D visualization')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Discovery timeout in seconds (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    # LED control arguments
    parser.add_argument('--no-led', action='store_true',
                       help='Disable LED flood illuminator')
    parser.add_argument('--led1-intensity', type=int, default=0,
                       help='LED1 intensity in mA (always set to 0 for hardware safety, parameter kept for compatibility)' )
    parser.add_argument('--led2-intensity', type=int, default=200,
                       help='LED2 intensity in mA (0-450, default: 200)')
    parser.add_argument('--no-auto-led', action='store_true',
                       help='Disable automatic LED intensity adjustment')
    
    args = parser.parse_args()
    
    # Don't search for calibration here - let the camera client handle it
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger('unlook.client.scanning.handpose').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
    
    run_unlook_hand_tracking_demo(
        calibration_file=args.calibration,
        output_file=args.output,
        visualize_3d=args.visualize_3d,
        timeout=args.timeout,
        verbose=args.verbose,
        use_led=not args.no_led,
        led1_intensity=args.led1_intensity,
        led2_intensity=args.led2_intensity,
        auto_led_adjust=not args.no_auto_led
    )


if __name__ == '__main__':
    main()