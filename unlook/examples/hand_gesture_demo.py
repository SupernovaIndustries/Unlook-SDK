#!/usr/bin/env python3
"""Simple hand gesture recognition demo using UnLook SDK with visualization.

This example demonstrates how easy it is to add gesture recognition
to your UnLook applications with just a few lines of code.
"""

import time
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker, GestureType

def main():
    # Initialize UnLook client
    client = UnlookClient(auto_discover=True)
    
    # Start discovery and wait for scanner
    client.start_discovery()
    print("Discovering scanners...")
    time.sleep(5)
    
    # Connect to first available scanner
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("No scanners found!")
        return
    
    client.connect(scanners[0])
    print(f"Connected to: {scanners[0].name}")
    
    # Get calibration file from camera client
    calibration_file = client.camera.get_calibration_file_path()
    if calibration_file:
        print(f"Using auto-loaded calibration: {calibration_file}")
    
    # Initialize hand tracker with gesture recognition
    tracker = HandTracker(calibration_file=calibration_file, max_num_hands=2)
    
    # Get camera IDs
    cameras = client.camera.get_cameras()
    left_camera = cameras[0]['id']
    right_camera = cameras[1]['id']
    
    print("\nStarting gesture recognition demo with visualization...")
    print("Available gestures:")
    print("- Open Palm")
    print("- Closed Fist")
    print("- Pointing")
    print("- Peace Sign")
    print("- Thumbs Up/Down")
    print("- OK Sign")
    print("- Rock Sign")
    print("- Pinch")
    print("- Wave (move hand side to side)")
    print("\nPress 'q' to quit")
    print("Press 's' to save a snapshot")
    print("Press 'v' to toggle 3D visualization\n")
    
    # Define hand connections for visualization
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
    
    # Track gestures in real-time
    try:
        # Use streaming for better performance
        frame_buffer = {'left': None, 'right': None}
        frame_lock = {'left': False, 'right': False}
        
        def left_callback(frame, metadata):
            frame_buffer['left'] = frame
            frame_lock['left'] = True
            
        def right_callback(frame, metadata):
            frame_buffer['right'] = frame
            frame_lock['right'] = True
        
        client.stream.start(left_camera, left_callback, fps=30)
        client.stream.start(right_camera, right_callback, fps=30)
        
        last_gesture = None
        show_3d = False
        frame_count = 0
        fps_timer = time.time()
        fps_count = 0
        current_fps = 0
        
        while True:
            # Wait for both frames
            if not (frame_lock['left'] and frame_lock['right']):
                time.sleep(0.01)
                continue
                
            # Get frames
            left_frame = frame_buffer['left']
            right_frame = frame_buffer['right']
            
            # Reset locks
            frame_lock['left'] = False
            frame_lock['right'] = False
            
            if left_frame is None or right_frame is None:
                continue
            
            # Track hands and recognize gestures
            results = tracker.track_hands_3d(left_frame, right_frame)
            
            # Debug print to see what we're getting
            if frame_count % 30 == 0:  # Print every second
                print(f"Debug - 2D left hands: {len(results['2d_left'])}, "
                      f"2D right hands: {len(results['2d_right'])}, "
                      f"3D hands: {len(results['3d_keypoints'])}, "
                      f"Gestures: {len(results.get('gestures', []))}")
            
            # Create visualization
            display_left = left_frame.copy()
            display_right = right_frame.copy()
            
            # Draw 2D keypoints and connections on both views
            for view, keypoints_list, image in [('left', results['2d_left'], display_left), 
                                               ('right', results['2d_right'], display_right)]:
                for i, keypoints in enumerate(keypoints_list):
                    h, w = image.shape[:2]
                    pixel_coords = tracker.detector.get_2d_pixel_coordinates(keypoints, w, h)
                    
                    # Draw keypoints
                    for j, pt in enumerate(pixel_coords):
                        cv2.circle(image, tuple(pt[:2].astype(int)), 3, (0, 255, 0), -1)
                    
                    # Draw connections
                    for connection in connections:
                        pt1 = pixel_coords[connection[0]][:2].astype(int)
                        pt2 = pixel_coords[connection[1]][:2].astype(int)
                        cv2.line(image, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
                    
                    # Draw bounding box
                    if len(pixel_coords) > 0:
                        x_coords = pixel_coords[:, 0]
                        y_coords = pixel_coords[:, 1]
                        x_min = int(np.min(x_coords)) - 10
                        x_max = int(np.max(x_coords)) + 10
                        y_min = int(np.min(y_coords)) - 10
                        y_max = int(np.max(y_coords)) + 10
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
            # Combine stereo views
            stereo_image = np.hstack([display_left, display_right])
            
            # Calculate FPS
            fps_count += 1
            if time.time() - fps_timer > 1.0:
                current_fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            
            # Add info text
            info_text = f"FPS: {current_fps} | Hands: {len(results['3d_keypoints'])}"
            cv2.putText(stereo_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display gesture information
            y_offset = 60
            for i, gesture_info in enumerate(results.get('gestures', [])):
                gesture_type = gesture_info['type']
                confidence = gesture_info['confidence']
                name = gesture_info['name']
                
                # Only show confident detections
                if confidence > 0.7 and gesture_type != GestureType.UNKNOWN:
                    gesture_text = f"Hand {i}: {name} ({confidence:.2f})"
                    color = (0, 255, 255)  # Yellow for active gesture
                    
                    # Highlight current gesture
                    if gesture_type != last_gesture:
                        print(f"Detected: {gesture_text}")
                        last_gesture = gesture_type
                        color = (0, 255, 0)  # Green for new gesture
                    
                    cv2.putText(stereo_image, gesture_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 30
            
            # Add instruction text
            inst_y = stereo_image.shape[0] - 60
            cv2.putText(stereo_image, "Press 'q' to quit | 's' to save | 'v' for 3D view", 
                       (10, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show visualization
            cv2.imshow('UnLook Gesture Recognition Demo', stereo_image)
            
            # Show 3D visualization if enabled
            if show_3d and results['3d_keypoints']:
                viz_3d = tracker.visualize_3d_hands(results)
                if viz_3d is not None:
                    cv2.imshow('3D Hand Visualization', viz_3d)
            elif not show_3d:
                # Close 3D window if it exists
                cv2.destroyWindow('3D Hand Visualization')
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = int(time.time())
                filename = f"gesture_snapshot_{timestamp}.png"
                cv2.imwrite(filename, stereo_image)
                print(f"Saved snapshot to {filename}")
            elif key == ord('v'):
                show_3d = not show_3d
                print(f"3D visualization: {'ON' if show_3d else 'OFF'}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        try:
            client.stream.stop_stream(left_camera)
            client.stream.stop_stream(right_camera)
        except:
            pass
        client.disconnect()
        client.stop_discovery()
        tracker.close()
        print("Demo finished!")

if __name__ == '__main__':
    main()