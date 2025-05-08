#!/usr/bin/env python3
"""
Simple 3D scanning example using the high-level UnlookScanner API.

This example demonstrates how to:
1. Connect to an UnLook scanner
2. Perform a complete 3D scan
3. Create a mesh from the point cloud
4. Save the results

Usage:
    python simple_3d_scanning_example.py
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from unlook.client.scanner3d import UnlookScanner
except ImportError as e:
    logger.error(f"Error importing UnlookScanner: {e}")
    print("Please make sure you have the required dependencies installed:")
    print("  numpy, opencv-python")
    print("For full functionality, also install:")
    print("  open3d (pip install open3d)")
    print("  h5py (pip install h5py)")
    sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple 3D scanning example")
    parser.add_argument("--output", "-o", type=str, default="./scans",
                      help="Base output directory for scan results")
    parser.add_argument("--timeout", "-t", type=int, default=5,
                      help="Timeout in seconds for scanner discovery")
    parser.add_argument("--visualize", "-v", action="store_true",
                      help="Visualize the results (requires open3d)")
    parser.add_argument("--debug", "-d", action="store_true",
                      help="Enable debug mode with additional logging")
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create the scanner, connect, scan, and save results - all in one context
    try:
        logger.info("Creating scanner and connecting to hardware...")
        
        # Use the scanner as a context manager to ensure proper cleanup
        with UnlookScanner() as scanner:
            # Set the output directory
            scanner.set_output_dir(args.output)
            
            # Connect to the scanner with auto-discovery
            connected = scanner.connect(timeout=args.timeout)
            
            if not connected:
                logger.error("Failed to connect to scanner")
                return
            
            logger.info("Connected to scanner")
            
            # Perform a 3D scan with default parameters
            logger.info("Starting 3D scan...")
            point_cloud = scanner.perform_3d_scan(visualize=args.visualize)
            
            if point_cloud is None or len(point_cloud.points) == 0:
                logger.error("Scan failed - no points generated")
                return
            
            logger.info(f"Scan completed with {len(point_cloud.points)} points")
            
            # Get the scan folders
            scan_folders = scanner.get_scan_folders()
            results_dir = scan_folders.get("results_dir")
            
            if results_dir:
                # Create a mesh if we have enough points
                if len(point_cloud.points) >= 100:
                    logger.info("Creating mesh from point cloud...")
                    mesh = scanner.create_mesh(point_cloud)
                    
                    # Save the mesh
                    mesh_path = os.path.join(results_dir, "scan_mesh.obj")
                    scanner.save_mesh(mesh, mesh_path)
                    logger.info(f"Saved mesh to {mesh_path}")
                else:
                    logger.warning(f"Not enough points ({len(point_cloud.points)}) to create a mesh")
                
                logger.info(f"Scan results saved to {scan_folders.get('scan_dir')}")
            
        logger.info("Scanner disconnected and resources cleaned up")
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()