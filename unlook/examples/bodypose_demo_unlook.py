#!/usr/bin/env python3
"""Body pose detection using UnLook hardware cameras."""

import time
import argparse
import logging
import numpy as np
from pathlib import Path
import cv2
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import UnLook SDK
from unlook import UnlookClient
from unlook.client.scanning.bodypose import BodyTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_unlook_body_tracking_demo(calibration_file=None, output_file=None, visualize_3d=False):
    """Run 3D body tracking demo using UnLook scanner cameras."""
    
    # Initialize UnLook client
    logger.info("Initializing UnLook client...")
    client = UnlookClient(auto_discover=False)
    
    # Start discovery
    logger.info("Starting scanner discovery...")
    client.start_discovery()
    
    # Wait for discovery
    time.sleep(10.0)
    
    # Get discovered scanners
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No UnLook scanners found!")
        client.stop_discovery()
        return
    
    # Connect to first scanner
    scanner = scanners[0]
    logger.info(f"Connecting to scanner: {scanner.name} at {scanner.endpoint}")
    
    if not client.connect(scanner):
        logger.error("Failed to connect to scanner")
        client.stop_discovery()
        return
    
    # Wait for connection
    time.sleep(1.0)
    
    # Initialize body tracker
    tracker = BodyTracker(calibration_file=calibration_file)
    
    # Configure cameras
    logger.info("Configuring cameras...")
    cameras = client.camera.get_cameras()
    if len(cameras) < 2:
        logger.error(f"Need at least 2 cameras, found {len(cameras)}")
        return
    
    # Use first two cameras as left and right
    left_camera = cameras[0]['id']
    right_camera = cameras[1]['id']
    logger.info(f"Using cameras: {left_camera} (left), {right_camera} (right)")
    
    print("\nStarting UnLook 3D body tracking demo...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame data")
    print("Press 'v' to toggle 3D visualization")
    
    show_3d = visualize_3d
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
    
    # Start streaming
    logger.info("Starting camera streams...")
    client.stream.start(left_camera, frame_callback_left, fps=30)
    client.stream.start(right_camera, frame_callback_right, fps=30)
    
    # Give streams time to start
    time.sleep(0.5)
    
    try:
        while True:
            # Wait for both frames
            if not (frame_lock['left'] and frame_lock['right']):
                time.sleep(0.01)
                continue
                
            # Get frames
            frame_left = frame_buffer['left']
            frame_right = frame_buffer['right']
            
            # Reset locks
            frame_lock['left'] = False
            frame_lock['right'] = False
            
            if frame_left is None or frame_right is None:
                logger.warning("Failed to get frames from cameras")
                continue
            
            # Track bodies in 3D
            results = tracker.track_body_3d(frame_left, frame_right)
            
            # Create display - show left and right images side by side
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            # Define body pose connections (MediaPipe pose landmarks)
            connections = [
                # Face
                [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
                # Arms
                [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
                # Torso
                [11, 23], [12, 24], [23, 24],
                # Legs
                [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30],
                [29, 31], [30, 32], [27, 31], [28, 32]
            ]
            
            # Draw 2D pose detections on each image
            for i, left_kpts in enumerate(results['2d_left']):
                # Draw left camera keypoints
                h, w = display_left.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(left_kpts, w, h)
                
                # Draw keypoints
                for j, pt in enumerate(pixel_coords):
                    cv2.circle(display_left, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
                    # Add landmark index
                    cv2.putText(display_left, str(j), tuple(pt.astype(int) + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw connections
                for connection in connections:
                    if connection[0] < len(pixel_coords) and connection[1] < len(pixel_coords):
                        pt1 = pixel_coords[connection[0]].astype(int)
                        pt2 = pixel_coords[connection[1]].astype(int)
                        cv2.line(display_left, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
            
            for i, right_kpts in enumerate(results['2d_right']):
                # Draw right camera keypoints  
                h, w = display_right.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(right_kpts, w, h)
                
                # Draw keypoints
                for j, pt in enumerate(pixel_coords):
                    cv2.circle(display_right, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
                    cv2.putText(display_right, str(j), tuple(pt.astype(int) + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw connections (same as left)
                for connection in connections:
                    if connection[0] < len(pixel_coords) and connection[1] < len(pixel_coords):
                        pt1 = pixel_coords[connection[0]].astype(int)
                        pt2 = pixel_coords[connection[1]].astype(int)
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
            info_text = f"Frame: {frame_count} | FPS: {current_fps} | 3D Bodies: {len(results['3d_keypoints'])}"
            cv2.putText(stereo_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera info
            cam_info = f"Cameras: {left_camera} | {right_camera}"
            cv2.putText(stereo_image, cam_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('UnLook Body Tracking', stereo_image)
            
            # Show 3D visualization if enabled
            if show_3d and results['3d_keypoints']:
                viz_3d = tracker.visualize_3d_bodies(results)
                if viz_3d is not None:
                    cv2.imshow('3D Visualization', viz_3d)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame data
                timestamp = int(time.time())
                filename = f"unlook_body_tracking_{timestamp}.json"
                tracker.save_tracking_data(filename)
                logger.info(f"Saved tracking data to {filename}")
            elif key == ord('v'):
                show_3d = not show_3d
                logger.info(f"3D visualization: {'ON' if show_3d else 'OFF'}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Save all tracking data if output file specified
        if output_file:
            tracker.save_tracking_data(output_file)
            logger.info(f"Saved all tracking data to {output_file}")
        
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
    parser = argparse.ArgumentParser(description='UnLook body pose tracking demo')
    parser.add_argument('--calibration', type=str,
                       help='Path to stereo calibration file')
    parser.add_argument('--output', type=str,
                       help='Output file for saving tracking data')
    parser.add_argument('--visualize-3d', action='store_true',
                       help='Show 3D visualization')
    
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
    
    run_unlook_body_tracking_demo(
        calibration_file=args.calibration,
        output_file=args.output,
        visualize_3d=args.visualize_3d
    )


if __name__ == '__main__':
    main()