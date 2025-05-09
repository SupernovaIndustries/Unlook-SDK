#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust 3D Scanning Example

This example demonstrates how to use the RobustStructuredLightScanner
to perform a complete 3D scan with a single command.
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

from unlook.client.robust_structured_light import RobustStructuredLightScanner
from unlook.client.scan_config import ScanConfig
from unlook.client.camera import StereoCamera
from unlook.client.projector import Projector


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
    parser.add_argument('--mesh', action='store_true',
                        help='Generate mesh from point cloud')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results after scanning')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Run the robust 3D scanning example."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"scan_{timestamp}_combined_{args.quality}"
        output_dir = os.path.join("scans", output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Scan results will be saved to: {output_dir}")
    
    # Load calibration
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
    
    # Initialize hardware
    logger.info("Initializing hardware...")
    try:
        camera = StereoCamera()
        projector = Projector()
    except Exception as e:
        logger.error(f"Failed to initialize hardware: {e}")
        return 1
    
    # Configure scan parameters based on quality preset
    config = ScanConfig()
    config.pattern_resolution = (1024, 768)  # Adjust based on your projector
    
    if args.quality == 'fast':
        logger.info("Using 'fast' quality preset (lower quality, faster scanning)")
        config.num_gray_codes = 8
        config.num_phase_shifts = 4
        config.phase_shift_frequencies = [1]
    elif args.quality == 'medium':
        logger.info("Using 'medium' quality preset (balanced quality and speed)")
        config.num_gray_codes = 10
        config.num_phase_shifts = 4
        config.phase_shift_frequencies = [1, 8]
    elif args.quality == 'high':
        logger.info("Using 'high' quality preset (higher quality, slower scanning)")
        config.num_gray_codes = 10
        config.num_phase_shifts = 8
        config.phase_shift_frequencies = [1, 8, 16]
    elif args.quality == 'ultra':
        logger.info("Using 'ultra' quality preset (highest quality, slowest scanning)")
        config.num_gray_codes = 12
        config.num_phase_shifts = 12
        config.phase_shift_frequencies = [1, 8, 16, 32]
    
    # Create scanner with loaded configuration
    logger.info(f"Creating scanner with {args.quality} quality preset...")
    try:
        scanner = RobustStructuredLightScanner(
            camera=camera,
            projector=projector,
            config=config,
            calibration_file=calibration_file,
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"Failed to create scanner: {e}")
        return 1
    
    # Execute scan
    logger.info("Starting 3D scan...")
    try:
        # Perform the scan and get point cloud
        point_cloud = scanner.scan(output_dir=output_dir)
        
        # Check if scan was successful
        if OPEN3D_AVAILABLE and isinstance(point_cloud, o3d.geometry.PointCloud):
            num_points = len(point_cloud.points)
            logger.info(f"Scan completed successfully with {num_points} points.")
        elif isinstance(point_cloud, np.ndarray):
            num_points = len(point_cloud)
            logger.info(f"Scan completed successfully with {num_points} points.")
        else:
            logger.warning("Scan produced no points or invalid point cloud.")
            return 1
        
        # Generate mesh if requested
        if args.mesh and OPEN3D_AVAILABLE and num_points > 0:
            logger.info("Generating mesh from point cloud...")
            try:
                mesh = scanner.create_mesh(point_cloud)
                
                if len(mesh.triangles) > 0:
                    mesh_file = os.path.join(output_dir, "results", "scan_mesh.ply")
                    o3d.io.write_triangle_mesh(mesh_file, mesh)
                    logger.info(f"Mesh with {len(mesh.triangles)} triangles saved to: {mesh_file}")
                else:
                    logger.warning("Mesh generation failed - no triangles created")
            except Exception as e:
                logger.error(f"Error during mesh creation: {e}")
        elif args.mesh and not OPEN3D_AVAILABLE:
            logger.warning("Mesh generation requires Open3D. Install with: pip install open3d")
        
        # Save point cloud
        if OPEN3D_AVAILABLE and isinstance(point_cloud, o3d.geometry.PointCloud):
            pc_file = os.path.join(output_dir, "results", "scan_point_cloud.ply")
            o3d.io.write_point_cloud(pc_file, point_cloud)
            logger.info(f"Point cloud saved to: {pc_file}")
        elif isinstance(point_cloud, np.ndarray) and len(point_cloud) > 0:
            pc_file = os.path.join(output_dir, "results", "scan_point_cloud.npy")
            np.save(pc_file, point_cloud)
            logger.info(f"Point cloud saved to: {pc_file}")
        
        # Visualize if requested
        if args.visualize and OPEN3D_AVAILABLE and num_points > 0:
            logger.info("Visualizing scan results...")
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                
                if args.mesh and 'mesh' in locals() and len(mesh.triangles) > 0:
                    vis.add_geometry(mesh)
                    logger.info("Displaying mesh. Close window to continue.")
                else:
                    vis.add_geometry(point_cloud)
                    logger.info("Displaying point cloud. Close window to continue.")
                
                # Add coordinate frame for scale reference
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
                vis.add_geometry(coord_frame)
                
                # Improve visualization settings
                opt = vis.get_render_option()
                opt.background_color = np.array([0.1, 0.1, 0.1])
                opt.point_size = 2.0
                
                vis.run()
                vis.destroy_window()
            except Exception as e:
                logger.error(f"Error during visualization: {e}")
        elif args.visualize and not OPEN3D_AVAILABLE:
            logger.warning("Visualization requires Open3D. Install with: pip install open3d")
        
        logger.info("Scan process completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())