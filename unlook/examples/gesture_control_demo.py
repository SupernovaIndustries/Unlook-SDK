#!/usr/bin/env python3
"""Gesture control demo - Control a virtual object with hand gestures.

This demonstrates how to use UnLook hand tracking to control applications
with simple gestures. Easy to integrate into any project!
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker, GestureType

class VirtualObject:
    """Simple virtual object that can be controlled with gestures."""
    
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = 0.0
        self.scale = 1.0
        self.color = "blue"
        self.is_selected = False
        
    def update(self, gesture_type, hand_position=None):
        """Update object based on gesture."""
        
        if gesture_type == GestureType.POINTING:
            # Move object to hand position
            if hand_position is not None:
                self.position = hand_position * 0.001  # Convert mm to meters
                print(f"Moving to: {self.position}")
                
        elif gesture_type == GestureType.PINCH:
            # Select/deselect object
            self.is_selected = not self.is_selected
            print(f"Object {'selected' if self.is_selected else 'deselected'}")
            
        elif gesture_type == GestureType.CLOSED_FIST:
            # Rotate object
            self.rotation += 10
            print(f"Rotating to: {self.rotation}°")
            
        elif gesture_type == GestureType.OPEN_PALM:
            # Reset object
            self.position = np.array([0.0, 0.0, 0.0])
            self.rotation = 0.0
            self.scale = 1.0
            print("Object reset")
            
        elif gesture_type == GestureType.THUMBS_UP:
            # Scale up
            self.scale *= 1.1
            print(f"Scaling up to: {self.scale:.2f}")
            
        elif gesture_type == GestureType.THUMBS_DOWN:
            # Scale down
            self.scale *= 0.9
            print(f"Scaling down to: {self.scale:.2f}")
            
        elif gesture_type == GestureType.PEACE:
            # Change color
            colors = ["blue", "red", "green", "yellow", "purple"]
            current_idx = colors.index(self.color)
            self.color = colors[(current_idx + 1) % len(colors)]
            print(f"Color changed to: {self.color}")


def main():
    # Initialize virtual object
    virtual_object = VirtualObject()
    
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
    
    # Initialize hand tracker
    tracker = HandTracker(calibration_file=calibration_file, max_num_hands=1)
    
    # Get camera IDs
    cameras = client.camera.get_cameras()
    left_camera = cameras[0]['id']
    right_camera = cameras[1]['id']
    
    print("\nGesture Control Demo")
    print("====================")
    print("Available controls:")
    print("- POINTING: Move object to hand position")
    print("- PINCH: Select/deselect object")
    print("- CLOSED FIST: Rotate object")
    print("- OPEN PALM: Reset object")
    print("- THUMBS UP: Scale up")
    print("- THUMBS DOWN: Scale down")
    print("- PEACE: Change color")
    print("\nPress Ctrl+C to quit\n")
    
    # Main control loop
    try:
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
        gesture_cooldown = 0
        frame_count = 0
        
        # Give streams time to start
        time.sleep(0.5)
        
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
                
            # Track hands
            results = tracker.track_hands_3d(left_frame, right_frame)
            
            # Debug info
            if frame_count % 30 == 0:
                print(f"Debug - Hands detected: {len(results['2d_left'])}, "
                      f"Gestures: {len(results.get('gestures', []))}")
                
            # Process first hand only
            if results['gestures']:
                gesture_info = results['gestures'][0]
                
                # Get hand position (3D if available, otherwise use 2D left)
                hand_center = None
                if results['3d_keypoints']:
                    hand_3d = results['3d_keypoints'][0]
                    hand_center = np.mean(hand_3d, axis=0)
                elif results['2d_left']:
                    # Use 2D position from left camera as fallback
                    left_kpts = results['2d_left'][0]
                    h, w = left_frame.shape[:2]
                    pixel_coords = tracker.detector.get_2d_pixel_coordinates(left_kpts, w, h)
                    # Use wrist position as reference
                    hand_center = np.append(pixel_coords[0], 100)  # Add dummy Z
                
                # Process gesture if confident and cooldown expired
                if gesture_info['confidence'] > 0.8 and gesture_cooldown <= 0:
                    gesture_type = gesture_info['type']
                    
                    # Only process gesture changes or continuous gestures
                    if gesture_type != last_gesture or gesture_type == GestureType.POINTING:
                        virtual_object.update(gesture_type, hand_center)
                        last_gesture = gesture_type
                        
                        # Set cooldown for non-continuous gestures
                        if gesture_type != GestureType.POINTING:
                            gesture_cooldown = 10  # 10 frames cooldown
                
                # Decrease cooldown
                if gesture_cooldown > 0:
                    gesture_cooldown -= 1
                
            # Increase frame count
            frame_count += 1
            
            time.sleep(0.03)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
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