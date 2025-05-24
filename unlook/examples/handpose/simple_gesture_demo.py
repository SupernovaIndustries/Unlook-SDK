#!/usr/bin/env python3
"""
Simple and Fast Gesture Recognition Demo for UnLook.
Optimized for speed without ML/YOLO dependencies.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import UnLook SDK
from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker, GestureType
from unlook.client.projector import LEDController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    # Initialize UnLook client
    logger.info("Initializing UnLook client...")
    client = UnlookClient(auto_discover=False)
    
    try:
        # Start discovery
        client.start_discovery()
        logger.info("Discovering scanners for 3 seconds...")
        time.sleep(3)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Check that your hardware is connected.")
            return 1
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner.")
            return 1
        
        logger.info("Successfully connected!")
        
        # Initialize LED controller
        led_controller = LEDController(client)
        led_controller.set_intensity(led1=0, led2=50)  # Moderate LED2 intensity
        time.sleep(0.5)  # Let LED stabilize
        
        # Initialize hand tracker with default calibration
        hand_tracker = HandTracker(
            detection_confidence=0.5,
            tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Camera names
        left_camera = "picamera2_0"
        right_camera = "picamera2_1"
        
        # Frame buffers
        frame_buffer = {'left': None, 'right': None}
        
        def frame_callback_left(frame, metadata):
            frame_buffer['left'] = frame
            
        def frame_callback_right(frame, metadata):
            frame_buffer['right'] = frame
        
        # Start streaming at 30fps
        logger.info("Starting camera streams...")
        client.stream.start(left_camera, frame_callback_left, fps=30)
        client.stream.start(right_camera, frame_callback_right, fps=30)
        
        # Give streams time to start
        time.sleep(0.5)
        
        logger.info("Gesture recognition ready! Press 'q' to quit")
        logger.info("Try: open palm, closed fist, pointing, peace sign, thumbs up")
        
        # Create window
        cv2.namedWindow("UnLook Gesture Demo", cv2.WINDOW_AUTOSIZE)
        
        # Main loop
        frame_count = 0
        fps_timer = time.time()
        
        while True:
            # Get frames
            left_frame = frame_buffer.get('left')
            right_frame = frame_buffer.get('right')
            
            if left_frame is None or right_frame is None:
                time.sleep(0.01)
                continue
            
            # Copy frames to avoid threading issues
            left_frame = left_frame.copy()
            right_frame = right_frame.copy()
            
            # Track hands (no downsampling!)
            results = hand_tracker.track_hands_3d(left_frame, right_frame)
            
            # Get left camera keypoints for gesture recognition
            left_keypoints = results.get('2d_left', [])
            left_handedness = results.get('handedness_left', [])
            
            # Draw hands and recognize gestures
            for i, keypoints in enumerate(left_keypoints):
                # Convert normalized to pixel coordinates
                h, w = left_frame.shape[:2]
                keypoints_px = keypoints.copy()
                keypoints_px[:, 0] *= w
                keypoints_px[:, 1] *= h
                
                # Draw hand skeleton
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]
                
                for start, end in connections:
                    pt1 = tuple(keypoints_px[start][:2].astype(int))
                    pt2 = tuple(keypoints_px[end][:2].astype(int))
                    cv2.line(left_frame, pt1, pt2, (0, 255, 0), 2)
                
                # Draw keypoints
                for pt in keypoints_px:
                    center = tuple(pt[:2].astype(int))
                    cv2.circle(left_frame, center, 4, (0, 0, 255), -1)
                
                # Recognize gesture
                handedness = left_handedness[i] if i < len(left_handedness) else "Unknown"
                gesture, confidence = hand_tracker.gesture_recognizer.recognize_gesture(keypoints, handedness)
                
                # Display gesture if confident
                if gesture != GestureType.UNKNOWN and confidence > 0.7:
                    text = f"{gesture.value.replace('_', ' ').title()} ({confidence:.0%})"
                    cv2.putText(left_frame, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    logger.info(f"Detected: {gesture.value}")
            
            # Combine images side by side
            display = np.hstack([left_frame, right_frame])
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed
                fps_timer = time.time()
                logger.info(f"FPS: {fps:.1f}")
            
            # Show FPS on screen
            if frame_count > 30:
                cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("UnLook Gesture Demo", display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"gesture_screenshot_{frame_count}.png", display)
                logger.info("Screenshot saved!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        # Stop streams
        try:
            client.stream.stop(left_camera)
            client.stream.stop(right_camera)
        except:
            pass
        
        # Turn off LED
        try:
            led_controller.set_intensity(led1=0, led2=0)
        except:
            pass
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Disconnect
        client.disconnect()
        
        logger.info("Demo finished")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())