#!/usr/bin/env python3
"""Basic test for hand pose detection to debug issues."""

import time
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlook import UnlookClient
from unlook.client.scanning.handpose import HandDetector

def test_basic_detection():
    """Test basic hand detection without gesture recognition."""
    
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
    
    # Initialize basic hand detector
    detector = HandDetector(max_num_hands=2)
    
    # Get camera IDs
    cameras = client.camera.get_cameras()
    left_camera = cameras[0]['id']
    
    print("\nStarting basic hand detection test...")
    print("Press 'q' to quit\n")
    
    try:
        frame_buffer = {'left': None}
        frame_lock = {'left': False}
        
        def left_callback(frame, metadata):
            frame_buffer['left'] = frame
            frame_lock['left'] = True
        
        client.stream.start(left_camera, left_callback, fps=30)
        
        frame_count = 0
        
        while True:
            if not frame_lock['left']:
                time.sleep(0.01)
                continue
                
            frame = frame_buffer['left']
            frame_lock['left'] = False
            
            if frame is None:
                continue
            
            # Detect hands
            results = detector.detect_hands(frame)
            
            # Debug info
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Detected {len(results['keypoints'])} hands")
                if results['keypoints']:
                    print(f"  Hand 0 wrist position: {results['keypoints'][0][0]}")
            
            # Show image with detections
            display = results['image']
            
            # Add detection count
            cv2.putText(display, f"Hands: {len(results['keypoints'])}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Hand Detection Test', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        try:
            client.stream.stop_stream(left_camera)
        except:
            pass
        client.disconnect()
        client.stop_discovery()
        detector.close()
        print("Test finished!")

if __name__ == '__main__':
    test_basic_detection()