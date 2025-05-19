#!/usr/bin/env python3
"""Demo script for hand pose detection with UnLook SDK."""

import time
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlook.client.scanning.handpose import HandTracker, HandDetector
from unlook import UnlookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define hand connections globally
HAND_CONNECTIONS = [
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


def run_unlook_hand_tracking_demo(calibration_file=None, output_file=None, visualize_3d=False, timeout=10):
    """Run 3D hand tracking demo using UnLook scanner cameras."""
    
    # Initialize UnLook client
    logger.info("Initializing UnLook client...")
    client = UnlookClient(auto_discover=False)
    tracker = None
    
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
        
        # Wait for connection to stabilize
        time.sleep(1.0)
        
        # Initialize hand tracker
        tracker = HandTracker(calibration_file=calibration_file, max_num_hands=2)
        
        # Configure cameras
        logger.info("Configuring cameras...")
        cameras = client.camera.get_cameras()
        logger.info(f"Available cameras: {cameras}")
        
        if not cameras:
            logger.error("No cameras found")
            return 1
            
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return 1
        
        # Debug output camera information
        for i, cam in enumerate(cameras):
            logger.info(f"Camera {i}: {cam}")
        
        # Use first two cameras as left and right
        left_camera = cameras[0]['id']
        right_camera = cameras[1]['id']
        camera_names = [cameras[0]['name'], cameras[1]['name']]
        logger.info(f"Using cameras: {camera_names[0]} (left), {camera_names[1]} (right)")
        logger.info(f"Left camera ID: {left_camera}")
        logger.info(f"Right camera ID: {right_camera}")
        
        print("\nStarting UnLook 3D hand tracking demo...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame data")
        print("Press 'v' to toggle 3D visualization")
        
        show_3d = visualize_3d
        frame_count = 0
        
        # Create storage for frames from streaming
        frame_buffer = {'left': None, 'right': None}
        frame_ready = {'left': False, 'right': False}
        
        def frame_callback_left(frame, metadata):
            if frame is not None and hasattr(frame, 'shape'):
                frame_buffer['left'] = frame
                frame_ready['left'] = True
            else:
                logger.warning(f"Left camera callback received invalid frame: {type(frame)}")
            
        def frame_callback_right(frame, metadata):
            if frame is not None and hasattr(frame, 'shape'):
                frame_buffer['right'] = frame
                frame_ready['right'] = True
            else:
                logger.warning(f"Right camera callback received invalid frame: {type(frame)}")
        
        # Start streaming
        logger.info("Starting camera streams...")
        logger.info(f"Starting stream for left camera: {left_camera}")
        if not client.stream.start(left_camera, frame_callback_left, fps=30):
            logger.error(f"Failed to start stream for left camera: {left_camera}")
            return 1
            
        logger.info(f"Starting stream for right camera: {right_camera}")
        if not client.stream.start(right_camera, frame_callback_right, fps=30):
            logger.error(f"Failed to start stream for right camera: {right_camera}")
            client.stream.stop_stream(left_camera)
            return 1
        
        # Give streams time to start
        logger.info("Waiting for streams to initialize...")
        time.sleep(2.0)
        
        frame_timeout_counter = 0
        
        while True:
            # Wait for both frames to be ready
            if not (frame_ready['left'] and frame_ready['right']):
                time.sleep(0.01)
                frame_timeout_counter += 1
                
                if frame_timeout_counter > 500:  # 5 seconds timeout
                    logger.warning("Timeout waiting for frames. Stream status:")
                    logger.warning(f"Left ready: {frame_ready['left']}, Right ready: {frame_ready['right']}")
                    frame_timeout_counter = 0
                continue
                
            # Get frames
            frame_left = frame_buffer['left']
            frame_right = frame_buffer['right']
            
            # Reset ready flags
            frame_ready['left'] = False
            frame_ready['right'] = False
            
            if frame_left is None or frame_right is None:
                logger.warning("Failed to get valid frames from cameras")
                continue
                
            # Validate frame formats
            if not hasattr(frame_left, 'shape') or not hasattr(frame_right, 'shape'):
                logger.warning(f"Invalid frame format - Left: {type(frame_left)}, Right: {type(frame_right)}")
                continue
            
            # Reset timeout counter when frames are received
            frame_timeout_counter = 0
            
            # Track hands in 3D
            results = tracker.track_hands_3d(frame_left, frame_right)
            
            # Create display - show left and right images side by side
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            # Draw 2D detections on each image
            for i, left_kpts in enumerate(results['2d_left']):
                # Draw left camera keypoints
                h, w = display_left.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(left_kpts, w, h)
                
                # Draw keypoints
                for pt in pixel_coords:
                    # Only use x,y coordinates (ignore z)
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(display_left, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    if connection[0] < len(pixel_coords) and connection[1] < len(pixel_coords):
                        pt1 = pixel_coords[connection[0]]
                        pt2 = pixel_coords[connection[1]]
                        cv2.line(display_left, 
                                (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), 
                                (0, 255, 0), 2)
            
            for i, right_kpts in enumerate(results['2d_right']):
                # Draw right camera keypoints  
                h, w = display_right.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(right_kpts, w, h)
                
                # Draw keypoints
                for pt in pixel_coords:
                    # Only use x,y coordinates (ignore z)
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(display_right, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    if connection[0] < len(pixel_coords) and connection[1] < len(pixel_coords):
                        pt1 = pixel_coords[connection[0]]
                        pt2 = pixel_coords[connection[1]]
                        cv2.line(display_right, 
                                (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), 
                                (0, 255, 0), 2)
            
            # Combine images
            stereo_image = np.hstack([display_left, display_right])
            
            # Add tracking info
            info_text = f"Frame: {frame_count} | 3D Hands: {len(results['3d_keypoints'])}"
            cv2.putText(stereo_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera info
            cam_info = f"Cameras: {camera_names[0]} | {camera_names[1]}"
            cv2.putText(stereo_image, cam_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('UnLook Hand Tracking', stereo_image)
            
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
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save all tracking data if output file specified
        if output_file and tracker is not None:
            tracker.save_tracking_data(output_file)
            logger.info(f"Saved all tracking data to {output_file}")
        
        # Cleanup
        logger.info("Cleaning up...")
        try:
            if hasattr(client, 'stream'):
                try:
                    client.stream.stop_stream(left_camera)
                except:
                    pass
                try:
                    client.stream.stop_stream(right_camera)
                except:
                    pass
        except Exception as e:
            logger.warning(f"Error stopping streams: {e}")
            
        try:
            client.disconnect()
        except:
            pass
            
        try:
            client.stop_discovery()
        except:
            pass
            
        cv2.destroyAllWindows()
        
        if tracker is not None:
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
    
    args = parser.parse_args()
    
    # Find calibration file if not specified
    if not args.calibration:
        # Look for calibration in standard locations
        locations = [
            Path("calibration/custom/stereo_calibration.json"),
            Path("calibration/default/default_stereo.json"),
            Path("stereo_calibration.json"),
        ]
        
        for path in locations:
            if path.exists():
                args.calibration = str(path)
                logger.info(f"Found calibration file: {path}")
                break
        
        if not args.calibration:
            logger.warning("No calibration file found. 3D reconstruction will not be available.")
    
    run_unlook_hand_tracking_demo(
        calibration_file=args.calibration,
        output_file=args.output,
        visualize_3d=args.visualize_3d,
        timeout=args.timeout
    )


if __name__ == '__main__':
    main()