#!/usr/bin/env python3
"""Body pose detection using UnLook hardware cameras - Headless version."""

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


def run_unlook_body_tracking_headless(calibration_file=None, output_file=None, 
                                     video_output="unlook_body_tracking.mp4", duration=30):
    """Run 3D body tracking demo using UnLook scanner cameras in headless mode."""
    
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
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    print(f"\nStarting UnLook headless body tracking...")
    print(f"Recording for {duration} seconds to {video_output}")
    print(f"Cameras: {left_camera} (left), {right_camera} (right)")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                logger.info(f"Recording completed after {duration} seconds")
                break
            
            # Capture frames from UnLook cameras
            frame_left = client.camera.capture(left_camera)
            frame_right = client.camera.capture(right_camera)
            
            if frame_left is None or frame_right is None:
                logger.warning("Failed to capture frames from cameras")
                continue
            
            # Track bodies in 3D
            results = tracker.track_body_3d(frame_left, frame_right)
            
            # Create display - show left and right images side by side
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            # Draw 2D pose detections on each image
            for i, left_kpts in enumerate(results['2d_left']):
                # Draw left camera keypoints
                h, w = display_left.shape[:2]
                pixel_coords = tracker.detector.get_2d_pixel_coordinates(left_kpts, w, h)
                
                # Draw keypoints
                for j, pt in enumerate(pixel_coords):
                    cv2.circle(display_left, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
                
                # Draw connections
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
                
                # Draw connections (same as left)
                for connection in connections:
                    if connection[0] < len(pixel_coords) and connection[1] < len(pixel_coords):
                        pt1 = pixel_coords[connection[0]].astype(int)
                        pt2 = pixel_coords[connection[1]].astype(int)
                        cv2.line(display_right, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
            
            # Create combined display
            stereo_image = np.hstack([display_left, display_right])
            
            # Add tracking info
            info_text = f"Frame: {frame_count} | Time: {elapsed_time:.1f}s | 3D Bodies: {len(results['3d_keypoints'])}"
            cv2.putText(stereo_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera info
            cam_info = f"Cameras: {left_camera} | {right_camera}"
            cv2.putText(stereo_image, cam_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add 3D info if available
            if results['3d_keypoints']:
                for i, pts_3d in enumerate(results['3d_keypoints']):
                    # Calculate body center position (use hip center)
                    if len(pts_3d) > 24:  # Make sure we have all landmarks
                        hip_center = (pts_3d[23] + pts_3d[24]) / 2  # Average of both hips
                        depth_text = f"Body {i}: X={hip_center[0]:.1f}mm Y={hip_center[1]:.1f}mm Z={hip_center[2]:.1f}mm"
                        cv2.putText(stereo_image, depth_text, (10, 90 + i*30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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
    parser = argparse.ArgumentParser(description='UnLook body pose tracking demo - Headless')
    parser.add_argument('--calibration', type=str,
                       help='Path to stereo calibration file')
    parser.add_argument('--output', type=str,
                       help='Output file for saving tracking data')
    parser.add_argument('--video-output', type=str, default='unlook_body_tracking.mp4',
                       help='Output video file (default: unlook_body_tracking.mp4)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Recording duration in seconds (default: 30)')
    
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
    
    run_unlook_body_tracking_headless(
        calibration_file=args.calibration,
        output_file=args.output,
        video_output=args.video_output,
        duration=args.duration
    )


if __name__ == '__main__':
    main()