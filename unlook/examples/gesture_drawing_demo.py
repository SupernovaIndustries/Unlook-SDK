#!/usr/bin/env python3
"""Interactive drawing application controlled by hand gestures.

Simple example showing how UnLook hand tracking can be used
to create intuitive gesture-based interfaces.
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


class DrawingCanvas:
    """Simple drawing canvas controlled by gestures."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.color = (0, 0, 255)  # Red
        self.thickness = 3
        self.drawing = False
        self.last_point = None
        
        # Color palette
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0)
        }
        self.color_names = list(self.colors.keys())
        self.current_color_idx = 0
        
    def process_gesture(self, gesture_type, hand_position):
        """Process gesture to control drawing."""
        
        # Convert 3D position to 2D canvas coordinates
        if hand_position is not None:
            # Normalize hand position (assuming typical hand range)
            x = int((hand_position[0] + 200) / 400 * self.width)  # -200 to 200 mm
            y = int((hand_position[1] + 150) / 300 * self.height)  # -150 to 150 mm
            x = np.clip(x, 0, self.width - 1)
            y = np.clip(y, 0, self.height - 1)
            current_point = (x, y)
        else:
            current_point = None
        
        # Handle different gestures
        if gesture_type == GestureType.POINTING and current_point:
            # Draw when pointing
            if self.drawing and self.last_point:
                cv2.line(self.canvas, self.last_point, current_point, 
                        self.color, self.thickness)
            self.drawing = True
            self.last_point = current_point
            
        elif gesture_type == GestureType.OPEN_PALM:
            # Stop drawing with open palm
            self.drawing = False
            self.last_point = None
            
        elif gesture_type == GestureType.CLOSED_FIST:
            # Clear canvas with closed fist
            self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            print("Canvas cleared")
            
        elif gesture_type == GestureType.PEACE:
            # Change color with peace sign
            self.current_color_idx = (self.current_color_idx + 1) % len(self.color_names)
            color_name = self.color_names[self.current_color_idx]
            self.color = self.colors[color_name]
            print(f"Color changed to {color_name}")
            
        elif gesture_type == GestureType.THUMBS_UP:
            # Increase thickness
            self.thickness = min(self.thickness + 1, 15)
            print(f"Thickness: {self.thickness}")
            
        elif gesture_type == GestureType.THUMBS_DOWN:
            # Decrease thickness
            self.thickness = max(self.thickness - 1, 1)
            print(f"Thickness: {self.thickness}")
            
        elif gesture_type == GestureType.OK:
            # Save drawing
            timestamp = int(time.time())
            filename = f"gesture_drawing_{timestamp}.png"
            cv2.imwrite(filename, self.canvas)
            print(f"Drawing saved to {filename}")
    
    def get_display(self):
        """Get canvas with UI elements."""
        display = self.canvas.copy()
        
        # Add color indicator
        cv2.rectangle(display, (10, 10), (60, 60), self.color, -1)
        cv2.rectangle(display, (10, 10), (60, 60), (0, 0, 0), 2)
        
        # Add thickness indicator
        cv2.circle(display, (35, 90), self.thickness, (0, 0, 0), -1)
        
        # Add instructions
        instructions = [
            "POINTING: Draw",
            "OPEN PALM: Stop drawing",
            "CLOSED FIST: Clear canvas",
            "PEACE: Change color",
            "THUMBS UP/DOWN: Change thickness",
            "OK: Save drawing",
            "Press 'q' to quit"
        ]
        
        y = self.height - len(instructions) * 20 - 10
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y += 20
        
        return display


def main():
    # Initialize drawing canvas
    canvas = DrawingCanvas()
    
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
    
    print("\nGesture Drawing Demo")
    print("===================")
    print("Use hand gestures to draw on the canvas!")
    print("")
    
    # Main drawing loop
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
            
            # Create camera view with hand overlay
            camera_view = np.hstack([
                cv2.resize(left_frame, (400, 300)),
                cv2.resize(right_frame, (400, 300))
            ])
            
            # Process first hand only
            if results['gestures'] and results['3d_keypoints']:
                gesture_info = results['gestures'][0]
                hand_3d = results['3d_keypoints'][0]
                
                # Get hand index finger tip position for drawing
                index_tip = hand_3d[8]  # Index finger tip
                
                # Process gesture if confident
                if gesture_info['confidence'] > 0.7:
                    gesture_type = gesture_info['type']
                    
                    # Apply cooldown for non-continuous gestures
                    if gesture_type in [GestureType.PEACE, GestureType.CLOSED_FIST, 
                                      GestureType.OK] and gesture_cooldown > 0:
                        gesture_cooldown -= 1
                    else:
                        canvas.process_gesture(gesture_type, index_tip)
                        
                        # Set cooldown for certain gestures
                        if gesture_type in [GestureType.PEACE, GestureType.CLOSED_FIST, 
                                          GestureType.OK]:
                            gesture_cooldown = 15
                        
                        # Print gesture info
                        if gesture_type != last_gesture:
                            print(f"Gesture: {gesture_info['name']}")
                            last_gesture = gesture_type
                
                # Add gesture text to camera view
                cv2.putText(camera_view, f"Gesture: {gesture_info['name']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Get canvas display
            canvas_display = canvas.get_display()
            
            # Show combined view
            # Resize canvas to match camera view height
            canvas_resized = cv2.resize(canvas_display, (800, 600))
            camera_expanded = cv2.resize(camera_view, (800, 300))
            
            combined = np.vstack([camera_expanded, canvas_resized])
            cv2.imshow('UnLook Gesture Drawing', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
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