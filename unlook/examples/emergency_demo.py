#!/usr/bin/env python3
"""Emergency Gesture Recognition Demo - Ultra-simplified version for maximum stability.

This is a bare-bones version of the enhanced gesture demo with all advanced features
stripped out for maximum stability and performance.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
import cv2
import threading
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import UnLook SDK
from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker, GestureType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
frame_buffer = {'left': None, 'right': None}
frame_lock = {'left': False, 'right': False}
running = True


def run_emergency_demo(calibration_file=None, timeout=5, downsample=8):
    """
    Run a highly simplified gesture recognition demo with maximum stability.
    
    Args:
        calibration_file: Path to stereo calibration file
        timeout: Connection timeout in seconds
        downsample: Downsampling factor for maximum performance
    """
    global running, frame_buffer, frame_lock
    
    print("\nStarting Emergency Gesture Recognition Demo...")
    print("STRIPPED DOWN VERSION FOR MAXIMUM STABILITY")
    print("Press 'q' to quit\n")
    
    # Initialize UnLook client
    try:
        print("Initializing client...")
        client = UnlookClient(auto_discover=False)
        
        # Start discovery
        client.start_discovery()
        print(f"Discovering scanners for {timeout} seconds...")
        time.sleep(timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            print("No scanners found. Check hardware connections.")
            return
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        print(f"Connecting to scanner: {scanner_info.name}")
        
        if not client.connect(scanner_info):
            print("Failed to connect to scanner.")
            return
        
        print(f"Connected to scanner: {scanner_info.name}")
        
        # Get auto-loaded calibration if none specified
        if not calibration_file:
            calibration_file = client.camera.get_calibration_file_path()
            if calibration_file:
                print(f"Using auto-loaded calibration: {calibration_file}")
            else:
                print("No calibration file available - continuing without calibration")
        
        # Initialize bare-bones hand tracker with conservative settings
        tracker = HandTracker(
            calibration_file=calibration_file, 
            max_num_hands=1,  # Only track one hand for maximum performance
            detection_confidence=0.7,  # Higher threshold for more reliable detection
            tracking_confidence=0.7,   # Higher threshold for more stable tracking
            left_camera_mirror_mode=False,
            right_camera_mirror_mode=False
        )
        
        # Configure cameras
        print("Setting up cameras...")
        cameras = client.camera.get_cameras()
        if len(cameras) < 2:
            print(f"Need at least 2 cameras, found {len(cameras)}")
            return
        
        # Use first two cameras
        left_camera = cameras[0]['id']
        right_camera = cameras[1]['id']
        print(f"Using cameras: {cameras[0]['name']} (left), {cameras[1]['name']} (right)")
        
        # Start streaming
        def frame_callback_left(frame, metadata):
            if running:
                frame_buffer['left'] = frame.copy() if frame is not None else None
                frame_lock['left'] = True
            
        def frame_callback_right(frame, metadata):
            if running:
                frame_buffer['right'] = frame.copy() if frame is not None else None
                frame_lock['right'] = True
        
        # Start streams
        print("Starting camera streams...")
        client.stream.start(left_camera, frame_callback_left, fps=15)  # Lower FPS for stability
        client.stream.start(right_camera, frame_callback_right, fps=15)  # Lower FPS for stability
        
        # Give streams time to start
        time.sleep(0.5)
        
        # Create window
        print("Setting up display...")
        window_name = 'Emergency Gesture Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Simple splash screen
        splash = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(splash, "EMERGENCY MODE", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(splash, "Starting camera feed...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(splash, "Press 'q' to quit", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow(window_name, splash)
        cv2.waitKey(10)
        
        # Main loop
        fps_timer = time.time()
        fps_counter = 0
        current_fps = 0
        last_display = None
        
        # Frame interval - process every few frames for performance
        frame_interval = 3  # Process 1 in 3 frames
        frame_count = 0
        
        print("Starting main loop...")
        while running:
            # Check for frames
            if frame_lock['left'] and frame_lock['right']:
                # Get frames
                frame_left = frame_buffer['left'] 
                frame_right = frame_buffer['right']
                
                # Reset locks
                frame_lock['left'] = False
                frame_lock['right'] = False
                
                # Skip some frames for performance
                frame_count += 1
                if frame_count % frame_interval != 0 and last_display is not None:
                    # Just update display without processing
                    cv2.imshow(window_name, last_display)
                    
                    # Handle key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                        break
                        
                    continue
                
                # Make sure we have valid frames
                if frame_left is None or frame_right is None:
                    continue
                
                # Downsample for much better performance
                try:
                    if downsample > 1:
                        h, w = frame_left.shape[:2]
                        frame_left_small = cv2.resize(frame_left, (w//downsample, h//downsample))
                        
                        h, w = frame_right.shape[:2]
                        frame_right_small = cv2.resize(frame_right, (w//downsample, h//downsample))
                    else:
                        frame_left_small = frame_left
                        frame_right_small = frame_right
                except Exception as e:
                    print(f"Error downsampling: {e}")
                    frame_left_small = frame_left
                    frame_right_small = frame_right
                
                # Track hands with conservative settings
                try:
                    results = tracker.track_hands_3d(
                        frame_left_small, 
                        frame_right_small, 
                        prioritize_left_camera=True,
                        stabilize_handedness=True
                    )
                except Exception as e:
                    print(f"Error tracking hands: {e}")
                    results = {}
                
                # Update FPS counter
                fps_counter += 1
                if time.time() - fps_timer > 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()
                
                # Create simple side-by-side display
                try:
                    # Create a simple side-by-side display
                    h1, w1 = frame_left.shape[:2]
                    h2, w2 = frame_right.shape[:2]
                    
                    # Resize to same height if needed
                    if h1 != h2:
                        new_w2 = int(w2 * (h1 / h2))
                        frame_right = cv2.resize(frame_right, (new_w2, h1))
                        h2, w2 = h1, new_w2
                    
                    # Create combined display
                    display = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
                    display[:, :w1] = frame_left
                    display[:, w1:w1+w2] = frame_right
                    
                    # Add labels
                    cv2.putText(display, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "RIGHT", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add FPS counter
                    cv2.putText(display, f"FPS: {current_fps}", (10, h1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw detected hand positions if available (very simple visualization)
                    if 'gestures' in results and results['gestures']:
                        gesture_text = f"Gesture: {results['gestures'][0]['name']}"
                        cv2.putText(display, gesture_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Draw basic hand positions if available
                    if '2d_left' in results and results['2d_left']:
                        for hand_idx, hand_kpts in enumerate(results['2d_left']):
                            # Draw wrist position
                            if len(hand_kpts) > 0:
                                x, y = int(hand_kpts[0][0] * w1), int(hand_kpts[0][1] * h1)
                                cv2.circle(display, (x, y), 10, (0, 255, 255), -1)
                    
                    # Store display for frame skipping
                    last_display = display.copy()
                    
                    # Show the display
                    cv2.imshow(window_name, display)
                except Exception as e:
                    print(f"Error creating display: {e}")
                    # Try to show just raw frames if display creation fails
                    if frame_left is not None:
                        cv2.imshow(window_name, frame_left)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break
            else:
                # No new frames, just do minimal UI updates
                if last_display is not None:
                    cv2.imshow(window_name, last_display)
                    
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    running = False
                    break
    
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Signal to stop frame callbacks
        running = False
        
        print("Cleaning up...")
        # Clean up
        try:
            if 'client' in locals():
                if hasattr(client, 'stream'):
                    if 'left_camera' in locals():
                        client.stream.stop_stream(left_camera)
                    if 'right_camera' in locals():
                        client.stream.stop_stream(right_camera)
                client.disconnect()
                client.stop_discovery()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
        # Close all windows
        cv2.destroyAllWindows()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description='Emergency Gesture Recognition Demo')
    parser.add_argument('--calibration', type=str, help='Path to stereo calibration file')
    parser.add_argument('--timeout', type=int, default=5, help='Discovery timeout in seconds (default: 5)')
    parser.add_argument('--downsample', type=int, default=8, choices=[1, 2, 4, 8], 
                      help='Downsampling factor for processing (default: 8)')
    
    args = parser.parse_args()
    run_emergency_demo(
        calibration_file=args.calibration,
        timeout=args.timeout,
        downsample=args.downsample
    )


if __name__ == '__main__':
    main()