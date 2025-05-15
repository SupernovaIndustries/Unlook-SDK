#!/usr/bin/env python3
"""
Camera Auto-Optimization Example

This example demonstrates the use of the camera auto-optimization
features to improve pattern visibility in challenging lighting conditions.
"""

import sys
import logging
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook import UnlookClient
from unlook.client.camera import CameraAutoOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating camera optimization."""
    parser = argparse.ArgumentParser(description="Camera Auto-Optimization Example")
    parser.add_argument("--host", default=None, help="Scanner IP address")
    parser.add_argument("--timeout", type=int, default=5, help="Discovery timeout in seconds")
    parser.add_argument("--projector", action="store_true", help="Use projector during optimization")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create client
    client = UnlookClient(auto_discover=True)
    
    try:
        # Discover scanners
        if args.host:
            logger.info(f"Connecting to scanner at {args.host}")
            # Create scanner info manually
            scanner_info = type('ScannerInfo', (), {
                'host': args.host,
                'control_port': 5555,
                'stream_port': 5556
            })()
            scanners = [scanner_info]
        else:
            logger.info("Discovering scanners...")
            client.start_discovery()
            time.sleep(args.timeout)
            scanners = client.get_discovered_scanners()
        
        if not scanners:
            logger.error("No scanners found")
            return 1
        
        # Connect to first scanner
        scanner = scanners[0]
        logger.info(f"Connecting to scanner: {scanner.name}")
        
        if not client.connect(scanner):
            logger.error("Failed to connect to scanner")
            return 1
        
        # Get available cameras
        cameras = client.camera.get_cameras()
        if not cameras:
            logger.error("No cameras found")
            return 1
        
        logger.info(f"Found {len(cameras)} cameras")
        
        # Create auto-optimizer
        optimizer = CameraAutoOptimizer(client)
        
        # Optimize first camera (or both for stereo)
        if len(cameras) >= 2:
            logger.info("Optimizing stereo camera pair")
            left_camera = cameras[0]['id']
            right_camera = cameras[1]['id']
            
            # Optimize both cameras
            logger.info(f"Optimizing left camera: {left_camera}")
            left_result = optimizer.optimize_camera_settings(left_camera, use_projector=args.projector)
            
            if left_result:
                logger.info(f"Left camera optimization result:")
                logger.info(f"  Exposure: {left_result.exposure_time}μs")
                logger.info(f"  Gain: {left_result.gain}")
                logger.info(f"  Brightness: {left_result.optimal_brightness:.3f}")
                logger.info(f"  Contrast: {left_result.optimal_contrast:.3f}")
                logger.info(f"  Score: {left_result.score:.3f}")
            
            logger.info(f"Optimizing right camera: {right_camera}")
            right_result = optimizer.optimize_camera_settings(right_camera, use_projector=args.projector)
            
            if right_result:
                logger.info(f"Right camera optimization result:")
                logger.info(f"  Exposure: {right_result.exposure_time}μs")
                logger.info(f"  Gain: {right_result.gain}")
                logger.info(f"  Brightness: {right_result.optimal_brightness:.3f}")
                logger.info(f"  Contrast: {right_result.optimal_contrast:.3f}")
                logger.info(f"  Score: {right_result.score:.3f}")
        else:
            # Optimize single camera
            camera_id = cameras[0]['id']
            logger.info(f"Optimizing camera: {camera_id}")
            
            result = optimizer.optimize_camera_settings(camera_id, use_projector=args.projector)
            
            if result:
                logger.info(f"Optimization result:")
                logger.info(f"  Exposure: {result.exposure_time}μs")
                logger.info(f"  Gain: {result.gain}")
                logger.info(f"  Brightness: {result.optimal_brightness:.3f}")
                logger.info(f"  Contrast: {result.optimal_contrast:.3f}")
                logger.info(f"  Score: {result.score:.3f}")
            else:
                logger.error("Optimization failed")
        
        # Test auto-focus if supported
        logger.info("Testing auto-focus...")
        for camera in cameras:
            camera_id = camera['id']
            success = client.camera.auto_focus(camera_id)
            if success:
                logger.info(f"Auto-focus successful for camera {camera_id}")
            else:
                logger.info(f"Auto-focus not supported or failed for camera {camera_id}")
        
        logger.info("Camera optimization complete!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()