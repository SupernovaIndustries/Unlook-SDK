#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
New Robust 3D Scanning Example

This example demonstrates how to use the redesigned 3D scanner 
implementation that properly works with the UnlookClient architecture
to perform robust structured light scanning.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("robust_scanning")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import numpy as np
    import cv2
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    logger.error("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Visualization and mesh creation will be disabled.")
    logger.warning("Install open3d for better results: pip install open3d")
    OPEN3D_AVAILABLE = False

# Import scanner and client
from unlook import UnlookClient
from unlook.client.scanner3d import create_scanner, Scanner3D, ScanConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Robust 3D Scanning Example')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for scan results')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to stereo calibration file')
    parser.add_argument('--quality', type=str, default='high',
                        choices=['fast', 'medium', 'high', 'ultra'],
                        help='Scan quality preset')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout in seconds for scanner discovery')
    parser.add_argument('--mesh', action='store_true',
                        help='Generate mesh from point cloud')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results after scanning')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--save-debug-images', action='store_true',
                        help='Save all captured and debug images (may be very slow)')
    
    return parser.parse_args()


def main():
    """Run the robust 3D scanning example."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Enable saving debug images if requested
    if args.save_debug_images:
        os.environ["UNLOOK_SAVE_DEBUG_IMAGES"] = "1"
        logger.info("Debug image saving enabled - this may slow down scanning significantly")
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"scan_{timestamp}_{args.quality}"
        output_dir = os.path.join("scans", output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Scan results will be saved to: {output_dir}")
    
    # Load calibration file path
    calibration_file = args.calibration
    if not calibration_file:
        # Use default calibration
        default_calib = "scans/calibration/default_stereo_calibration.json"
        if os.path.exists(default_calib):
            calibration_file = default_calib
            logger.info(f"Using default calibration file: {calibration_file}")
        else:
            logger.warning(f"Default calibration file not found: {default_calib}")
            logger.warning("Using built-in default calibration parameters")
    else:
        if not os.path.exists(calibration_file):
            logger.error(f"Calibration file not found: {calibration_file}")
            logger.warning("Using built-in default calibration parameters")
            calibration_file = None
        else:
            logger.info(f"Using calibration file: {calibration_file}")
    
    # Initialize client and connect to scanner
    logger.info("Initializing client and connecting to scanner...")
    try:
        # Create client with auto-discovery
        client = UnlookClient(auto_discover=True)
        
        # Start discovery
        client.start_discovery()
        logger.info(f"Discovering scanners for {args.timeout} seconds...")
        time.sleep(args.timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Please ensure scanner hardware is connected and powered on.")
            return 1
            
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return 1
            
        logger.info(f"Successfully connected to scanner: {scanner_info.name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize client and connect to scanner: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Create 3D scanner
    logger.info(f"Creating scanner with {args.quality} quality preset...")
    try:
        scanner = create_scanner(
            client=client,
            quality=args.quality,
            calibration_file=calibration_file
        )
    except Exception as e:
        logger.error(f"Failed to create scanner: {e}")
        return 1
    
    # Execute scan
    logger.info("Starting 3D scan...")
    try:
        # Perform the scan
        result = scanner.scan(
            output_dir=output_dir,
            generate_mesh=args.mesh,
            visualize=args.visualize
        )
        
        # Check if scan was successful
        if result.has_point_cloud():
            logger.info(f"Scan completed successfully with {result.num_points} points.")
            
            if args.mesh and result.has_mesh():
                logger.info(f"Mesh created with {result.num_triangles} triangles.")
        else:
            logger.warning("Scan produced no points or invalid point cloud.")
            logger.error(f"Scan error: {result.debug_info.get('error', 'Unknown error')}")
            return 1
        
        logger.info("Scan process completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
    finally:
        # Disconnect from scanner
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)