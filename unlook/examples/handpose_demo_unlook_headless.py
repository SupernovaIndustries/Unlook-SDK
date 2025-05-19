#!/usr/bin/env python3
"""Hand pose detection using UnLook hardware cameras - Headless version."""

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
from unlook.client.scanning.handpose import HandTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_unlook_hand_tracking_headless(calibration_file=None, output_file=None, video_output="unlook_hand_tracking.mp4", duration=30):
    """Run 3D hand tracking demo using UnLook scanner cameras in headless mode."""
    
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
    
    # Get auto-loaded calibration if no calibration file specified
    if not calibration_file:
        calibration_file = client.camera.get_calibration_file_path()
        if calibration_file:
            logger.info(f"Using auto-loaded calibration: {calibration_file}")
    
    # Initialize hand tracker
    tracker = HandTracker(calibration_file=calibration_file, max_num_hands=2)
    
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
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    print(f"\nStarting UnLook headless hand tracking...")
    print(f"Recording for {duration} seconds to {video_output}")
    print(f"Cameras: {left_camera} (left), {right_camera} (right)")
    
    frame_count = 0
    start_time = time.time()
    elapsed_time = 0
    
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
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                logger.info(f"Recording completed after {duration} seconds")
                break
            
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
            
            # Track hands in 3D
            results = tracker.track_hands_3d(frame_left, frame_right)
            
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
            
            # Create combined display
            stereo_image = np.hstack([display_left, display_right])
            
            # Add tracking info
            info_text = f"Frame: {frame_count} | Time: {elapsed_time:.1f}s | 3D Hands: {len(results['3d_keypoints'])}"
            cv2.putText(stereo_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera info
            cam_info = f"Cameras: {left_camera} | {right_camera}"
            cv2.putText(stereo_image, cam_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add 3D info if available
            if results['3d_keypoints']:
                for i, pts_3d in enumerate(results['3d_keypoints']):
                    # Calculate hand center position
                    hand_center = np.mean(pts_3d, axis=0)
                    depth_text = f"Hand {i}: X={hand_center[0]:.1f}mm Y={hand_center[1]:.1f}mm Z={hand_center[2]:.1f}mm"
                    cv2.putText(stereo_image, depth_text, (10, 90 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Add gesture info if available
                    if 'gestures' in results and i < len(results['gestures']):
                        gesture_info = results['gestures'][i]
                        if gesture_info['type'].value != 'unknown':
                            gesture_text = f"Gesture: {gesture_info['name']} ({gesture_info['confidence']:.2f})"
                            cv2.putText(stereo_image, gesture_text, (10, 120 + i*30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Initialize video writer on first frame
            if video_writer is None:
                height, width = stereo_image.shape[:2]
                video_writer = cv2.VideoWriter(video_output, fourcc, 30.0, (width, height))
                logger.info(f"Initialized video writer: {width}x{height} @ 30fps")
            
            # Write frame to video
            video_writer.write(stereo_image)
            
            # Progress update
            if frame_count % 30 == 0:
                logger.info(f"Progress: {elapsed_time:.1f}s / {duration}s ({frame_count} frames)")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save all tracking data if output file specified
        if output_file:
            tracker.save_tracking_data(output_file)
            logger.info(f"Saved tracking data to {output_file}")
        
        # Cleanup
        logger.info("Cleaning up...")
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Video saved to {video_output}")
        
        try:
            client.stream.stop_stream(left_camera)
            client.stream.stop_stream(right_camera)
        except:
            pass
        client.disconnect()
        client.stop_discovery()
        tracker.close()
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Recorded {frame_count} frames in {elapsed_time:.1f} seconds")
        print(f"- Average FPS: {frame_count/elapsed_time:.1f}")
        print(f"- Video saved to: {video_output}")
        if output_file:
            print(f"- Tracking data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='UnLook hand pose tracking demo - Headless')
    parser.add_argument('--calibration', type=str,
                       help='Path to stereo calibration file')
    parser.add_argument('--output', type=str,
                       help='Output file for saving tracking data')
    parser.add_argument('--video-output', type=str, default='unlook_hand_tracking.mp4',
                       help='Output video file (default: unlook_hand_tracking.mp4)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Recording duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Don't search for calibration here - let the camera client handle it
    
    run_unlook_hand_tracking_headless(
        calibration_file=args.calibration,
        output_file=args.output,
        video_output=args.video_output,
        duration=args.duration
    )


if __name__ == '__main__':
    main()