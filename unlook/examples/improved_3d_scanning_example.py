#!/usr/bin/env python3
"""
Improved 3D Scanning Example with Enhanced Structured Light Scanner

This example demonstrates how to use the enhanced structured light scanner
with improved grayscale conversion, increased exposure/contrast, and increased
projector intensity for better scanning results.

The enhanced scanner provides better 3D reconstruction quality by:
1. Ensuring all images are properly converted to grayscale for consistent processing
2. Using increased camera exposure and contrast settings for better pattern visibility
3. Increasing projector intensity for better pattern projection
4. Combining Gray code and phase shift patterns for more accurate matching

This example requires:
- UnLook SDK (with the improvements from this PR)
- A connected UnLook scanner with stereo cameras
- Open3D (optional, for visualization)
"""

import os
import time
import logging
import sys
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import unlook module when running from examples folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook import UnlookClient
from unlook.client.scanner3d import UnlookScanner
from unlook.client.camera_config import ColorMode

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not found, visualization will be disabled")
    OPEN3D_AVAILABLE = False


def run_improved_3d_scan(output_dir=None, visualize=True, scan_quality="high", 
                        pattern_type="combined"):
    """
    Run an improved 3D scan with optimized parameters.
    
    Args:
        output_dir: Directory to save scan results (optional)
        visualize: Whether to visualize results (requires Open3D)
        scan_quality: Quality setting ("low", "medium", "high", "ultra")
        pattern_type: Pattern type ("gray_code", "phase_shift", "combined")
    
    Returns:
        Path to saved point cloud file
    """
    # Create timestamped output directory if none provided
    if not output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"scan_{timestamp}_combined_{scan_quality}"
        output_dir = os.path.join("scans", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
    logger.info(f"Starting improved 3D scan, results will be saved to: {output_dir}")
    logger.info(f"Scan quality: {scan_quality}, Pattern type: {pattern_type}")
    
    # Connect to scanner
    scanner = UnlookScanner.auto_connect(
        timeout=5,
        scanner_type="enhanced",  # Use enhanced scanner for better quality
        use_default_calibration=True
    )
    
    # Get camera client for additional configuration
    camera = scanner.client.camera
    
    # Get stereo camera pair
    left_camera, right_camera = camera.get_stereo_pair()
    if left_camera and right_camera:
        logger.info(f"Using cameras: {left_camera} (left), {right_camera} (right)")
        
        # Ensure cameras are in grayscale mode for more consistent processing
        for cam_id in [left_camera, right_camera]:
            camera.set_color_mode(cam_id, ColorMode.GRAYSCALE)
            logger.info(f"Set camera {cam_id} to grayscale mode")
    
    # Perform the scan with enhanced parameters
    try:
        # Use appropriate pattern type based on available capture capabilities
        # For some setups, the "combined" pattern might be too many patterns
        effective_pattern = pattern_type
        if pattern_type == "combined" and scan_quality in ["low", "medium"]:
            # Fall back to just gray_code for more reliable scanning in lower quality modes
            effective_pattern = "gray_code"
            logger.info(f"Using '{effective_pattern}' patterns instead of '{pattern_type}' for {scan_quality} quality")

        # For ultra quality, make sure we use a longer interval
        effective_interval = 0.75
        if scan_quality == "ultra":
            effective_interval = 1.0  # Longer interval for ultra quality

        point_cloud = scanner.perform_3d_scan(
            output_dir=output_dir,
            scanner_type="enhanced",  # Use enhanced scanner
            scan_quality=scan_quality,  # Quality setting
            pattern_type=effective_pattern,  # Adjusted pattern type
            visualize=False,  # We'll visualize manually if needed
            debug_output=True,  # Save debug information
            interval=effective_interval  # Adjusted interval for more stable patterns
        )
        
        logger.info(f"Scan completed successfully with {len(point_cloud.points) if hasattr(point_cloud, 'points') else 0} points")
        
        # Save the point cloud to a standard location
        point_cloud_path = os.path.join(output_dir, "results", "improved_scan.ply")
        scanner.save_scan(point_cloud, point_cloud_path)
        logger.info(f"Saved point cloud to {point_cloud_path}")
        
        # Create and save a mesh
        if hasattr(point_cloud, 'points') and len(point_cloud.points) > 500 and OPEN3D_AVAILABLE:
            try:
                mesh = scanner.create_mesh(point_cloud, depth=9, smooth_iterations=3)
                mesh_path = os.path.join(output_dir, "results", "improved_scan_mesh.obj")
                scanner.save_mesh(mesh, mesh_path)
                logger.info(f"Saved mesh to {mesh_path}")
            except Exception as e:
                logger.error(f"Error creating mesh: {e}")
        
        # Visualize if requested and possible
        if visualize and OPEN3D_AVAILABLE and hasattr(point_cloud, 'points') and len(point_cloud.points) > 0:
            logger.info("Visualizing 3D scan results")
            scanner.visualize_point_cloud(point_cloud)
            
        return point_cloud_path
        
    except Exception as e:
        logger.error(f"Error during 3D scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Disconnect from scanner
        scanner.disconnect()
        logger.info("Disconnected from scanner")
        

def main():
    """Main function to parse arguments and run 3D scanning."""
    parser = argparse.ArgumentParser(description="Improved 3D Scanning Example")
    parser.add_argument(
        "--output", 
        "-o", 
        help="Output directory for scan results"
    )
    parser.add_argument(
        "--visualize", 
        "-v", 
        action="store_true", 
        help="Enable result visualization"
    )
    parser.add_argument(
        "--quality", 
        "-q", 
        choices=["low", "medium", "high", "ultra"], 
        default="high",
        help="Scan quality level"
    )
    parser.add_argument(
        "--pattern", 
        "-p", 
        choices=["gray_code", "phase_shift", "combined"], 
        default="combined",
        help="Pattern type to use"
    )
    
    args = parser.parse_args()
    
    # Check if visualization is possible
    if args.visualize and not OPEN3D_AVAILABLE:
        logger.warning("Visualization requested but Open3D is not available. Installing Open3D is recommended.")
    
    # Run the scan
    result_path = run_improved_3d_scan(
        output_dir=args.output,
        visualize=args.visualize,
        scan_quality=args.quality,
        pattern_type=args.pattern
    )
    
    if result_path:
        print(f"\nScan completed successfully!")
        print(f"Results saved to: {result_path}")
    else:
        print("\nScan failed. See log for details.")

if __name__ == "__main__":
    main()