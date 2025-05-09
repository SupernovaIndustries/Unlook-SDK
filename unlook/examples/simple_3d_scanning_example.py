#!/usr/bin/env python3
"""
Simple 3D scanning example using the enhanced UnlookScanner API.

This example demonstrates how to:
1. Connect to an UnLook scanner
2. Perform a complete 3D scan with enhanced algorithms
3. Choose between different pattern types (Gray code, phase shift, or combined)
4. Set quality levels for scanning (low, medium, high, ultra)
5. Create a mesh from the point cloud
6. Save the results

Usage:
    python simple_3d_scanning_example.py [--pattern PATTERN] [--quality QUALITY]
    
Options:
    --pattern {gray_code,phase_shift,combined}
    --quality {low,medium,high,ultra}
    --visualize    Show the resulting point cloud
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
    parser.add_argument("--pattern", "-p", type=str, default="gray_code",
                      choices=["gray_code", "phase_shift", "combined"],
                      help="Type of structured light pattern to use")
    parser.add_argument("--quality", "-q", type=str, default="medium",
                      choices=["low", "medium", "high", "ultra"],
                      help="Quality setting for the scan")
    parser.add_argument("--interval", "-i", type=float, default=0.5,
                      help="Time interval between patterns (seconds)")
    parser.add_argument("--visualize", "-v", action="store_true",
                      help="Visualize the results (requires open3d)")
    parser.add_argument("--scanner", "-s", type=str, default="robust",
                      choices=["basic", "enhanced", "robust"],
                      help="Scanner type to use (robust is recommended for real objects)")
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
    
    # Log scan settings
    logger.info(f"Scan configuration:")
    logger.info(f"  Pattern type: {args.pattern}")
    logger.info(f"  Quality: {args.quality}")
    logger.info(f"  Scanner type: {args.scanner} (robust recommended for real objects)")
    logger.info(f"  Interval: {args.interval}s between patterns")
    
    # Create the scanner, connect, scan, and save results
    try:
        logger.info("Creating scanner and connecting to hardware...")
        
        # Create scanner with selected scanner type
        scanner = UnlookScanner.auto_connect(
            timeout=args.timeout,
            use_default_calibration=True,
            scanner_type=args.scanner  # Use the robust scanner by default
        )
        
        if not scanner.is_connected:
            logger.error("Failed to connect to scanner")
            return
        
        logger.info(f"Connected to scanner: {scanner.scanner_info.name}")
        
        # Set the output directory
        scanner.set_output_dir(args.output)
        
        # Create a timestamped directory for this scan
        timestamped_dir = os.path.join(
            args.output, 
            f"scan_{time.strftime('%Y%m%d_%H%M%S')}_{args.scanner}_{args.pattern}_{args.quality}"
        )
        
        # Perform a 3D scan with specified parameters
        logger.info(f"Starting 3D scan with {args.pattern} patterns at {args.quality} quality...")
        point_cloud = scanner.perform_3d_scan(
            output_dir=timestamped_dir,
            mask_threshold=5,               # Default threshold value
            interval=args.interval,         # Time between patterns
            visualize=args.visualize,       # Visualize result
            scanner_type=args.scanner,      # Scanner type (robust is best for real objects)
            scan_quality=args.quality,      # Quality setting
            pattern_type=args.pattern,      # Pattern type
            debug_output=True               # Enable debug output for troubleshooting
        )
        
        if point_cloud is None or len(point_cloud.points) == 0:
            logger.error("Scan failed - no points generated")
            scanner.disconnect()
            return
        
        logger.info(f"Scan completed with {len(point_cloud.points)} points")
        
        # Get the scan folders
        scan_folders = scanner.get_scan_folders()
        results_dir = scan_folders.get("results_dir", timestamped_dir)
        
        # Create a mesh if we have enough points
        if len(point_cloud.points) >= 100:
            logger.info("Creating mesh from point cloud...")
            
            # Adjust mesh parameters based on quality setting
            depth = 8
            smooth_iterations = 2
            
            if args.quality == "high":
                depth = 9
                smooth_iterations = 3
            elif args.quality == "ultra":
                depth = 10
                smooth_iterations = 5
            
            mesh = scanner.create_mesh(
                point_cloud,
                depth=depth,
                smooth_iterations=smooth_iterations
            )
            
            # Save the mesh
            mesh_path = os.path.join(results_dir, "scan_mesh.obj")
            scanner.save_mesh(mesh, mesh_path)
            logger.info(f"Saved mesh to {mesh_path}")
        else:
            logger.warning(f"Not enough points ({len(point_cloud.points)}) to create a mesh")
        
        logger.info(f"Scan results saved to {results_dir}")
        
        # Disconnect when done
        scanner.disconnect()
        logger.info("Scanner disconnected and resources cleaned up")
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        # Make sure to disconnect
        if 'scanner' in locals() and scanner.is_connected:
            scanner.disconnect()
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Make sure to disconnect even on error
        if 'scanner' in locals() and scanner.is_connected:
            scanner.disconnect()


if __name__ == "__main__":
    main()