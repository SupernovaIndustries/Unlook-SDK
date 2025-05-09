#!/usr/bin/env python3
"""
Robust 3D scanner example using structured light.

This example demonstrates using the robust scanner implementation
to perform 3D scanning with more reliable results.

The robust scanner is based on techniques from SLStudio and implements
improved stereo correspondence and triangulation algorithms.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('unlook.examples.robust_scanning')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import Unlook modules
from unlook.client import UnlookClient
from unlook.client.robust_scanner import create_robust_scanner

# Try to import visualization modules
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not installed. Point cloud visualization will be disabled.")
    OPEN3D_AVAILABLE = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Robust 3D scanning example')
    
    parser.add_argument('--server', type=str, default='localhost:8080',
                        help='Server address in format host:port')
    
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to stereo calibration file')
    
    parser.add_argument('--output-dir', type=str, default='scan_output',
                        help='Directory to save scan results')
    
    parser.add_argument('--num-gray-codes', type=int, default=8,
                        help='Number of Gray code bits')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional outputs')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU acceleration if available')
    
    return parser.parse_args()

def main():
    """Run the robust scanning example."""
    # Parse command line arguments
    args = parse_args()
    
    # Determine server host and port
    if ':' in args.server:
        host, port = args.server.split(':')
        port = int(port)
    else:
        host = args.server
        port = 8080
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Connect to server
    logger.info(f"Connecting to server at {host}:{port}")
    client = UnlookClient(host=host, port=port)
    
    # Check server connection
    logger.info("Testing server connection")
    server_info = client.get_server_info()
    logger.info(f"Connected to server: {server_info}")
    
    # Initialize scanner
    logger.info("Initializing robust scanner")
    scanner = create_robust_scanner(
        client=client,
        calibration_file=args.calibration,
        num_gray_codes=args.num_gray_codes,
        use_gpu=args.gpu,
        debug_mode=args.debug
    )
    
    # Perform scan
    logger.info("Starting 3D scan")
    scan_start = time.time()
    
    point_cloud = scanner.scan(output_dir=args.output_dir)
    
    scan_duration = time.time() - scan_start
    logger.info(f"Scanning completed in {scan_duration:.2f} seconds")
    
    # Check scan result
    if point_cloud is None:
        logger.error("Scanning failed, no point cloud returned")
        return
    
    if isinstance(point_cloud, np.ndarray):
        num_points = len(point_cloud)
    else:  # Open3D point cloud
        num_points = len(point_cloud.points) if hasattr(point_cloud, 'points') else 0
    
    logger.info(f"Generated point cloud with {num_points} points")
    
    # Save results
    if OPEN3D_AVAILABLE and hasattr(point_cloud, 'points'):
        # Save as PLY file
        output_path = os.path.join(args.output_dir, "point_cloud.ply")
        o3d.io.write_point_cloud(output_path, point_cloud)
        logger.info(f"Point cloud saved to {output_path}")
        
        # Visualize if Open3D is available and we have points
        if num_points > 0:
            logger.info("Visualizing point cloud (close window to continue)")
            o3d.visualization.draw_geometries([point_cloud])
    
    logger.info("Example completed successfully")

if __name__ == "__main__":
    main()