#!/usr/bin/env python3
"""
Advanced Pattern Types Example

This example demonstrates the new pattern types available for 3D scanning:
- Maze patterns
- Voronoi patterns
- Hybrid ArUco patterns
"""

import sys
import logging
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook import UnlookClient
from unlook.client.scan_config import PatternType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating advanced pattern types."""
    parser = argparse.ArgumentParser(description="Advanced Pattern Types Example")
    parser.add_argument("--pattern", choices=["maze", "voronoi", "hybrid_aruco"],
                       default="maze", help="Pattern type to test")
    parser.add_argument("--output", default="advanced_scan.ply", 
                       help="Output filename for point cloud")
    parser.add_argument("--calibration", help="Path to calibration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--timeout", type=int, default=10, 
                       help="Scanner discovery timeout")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Testing {args.pattern} pattern type")
    
    # Create client
    client = UnlookClient(auto_discover=True)
    
    try:
        # Discover scanners
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
        
        # Import scanning components
        from unlook.client.scanning import StaticScanner, StaticScanConfig
        
        # Create scan configuration with chosen pattern type
        config = StaticScanConfig(
            quality="high",
            debug=args.debug,
            save_intermediate_images=args.debug,
            save_raw_images=args.debug
        )
        
        # Set the pattern type
        pattern_map = {
            "maze": PatternType.MAZE,
            "voronoi": PatternType.VORONOI,
            "hybrid_aruco": PatternType.HYBRID_ARUCO
        }
        config.pattern_type = pattern_map[args.pattern]
        
        logger.info(f"Using {args.pattern} patterns for scanning")
        
        # Create scanner
        scanner = StaticScanner(
            client=client,
            config=config,
            calibration_file=args.calibration
        )
        
        # Print pattern information
        if args.pattern == "maze":
            logger.info("Maze patterns provide:")
            logger.info("  - Unique local topologies for robust correspondence")
            logger.info("  - Better performance in textured environments")
            logger.info("  - Reduced ambiguity in pattern matching")
        elif args.pattern == "voronoi":
            logger.info("Voronoi patterns provide:")
            logger.info("  - Dense surface reconstruction")
            logger.info("  - Rich local features")
            logger.info("  - Better performance with curved surfaces")
        elif args.pattern == "hybrid_aruco":
            logger.info("Hybrid ArUco patterns provide:")
            logger.info("  - Absolute position reference from markers")
            logger.info("  - High-resolution correspondence")
            logger.info("  - Automatic calibration and registration")
            logger.info("  - Robustness to partial occlusions")
        
        logger.info("Starting scan with advanced patterns...")
        time.sleep(2)  # Give user time to position object
        
        # Perform scan
        point_cloud = scanner.perform_scan()
        
        if point_cloud and hasattr(point_cloud, 'points') and len(point_cloud.points) > 0:
            # Save point cloud
            scanner.save_point_cloud(args.output)
            logger.info(f"Point cloud saved to: {args.output}")
            logger.info(f"Total points: {len(point_cloud.points)}")
            
            # Display statistics
            import numpy as np
            points = np.asarray(point_cloud.points)
            
            logger.info("Point cloud statistics:")
            logger.info(f"  X range: {np.min(points[:,0]):.2f} to {np.max(points[:,0]):.2f} mm")
            logger.info(f"  Y range: {np.min(points[:,1]):.2f} to {np.max(points[:,1]):.2f} mm")
            logger.info(f"  Z range: {np.min(points[:,2]):.2f} to {np.max(points[:,2]):.2f} mm")
            
            # For hybrid ArUco, show detected markers
            if args.pattern == "hybrid_aruco" and hasattr(scanner, 'detected_markers'):
                logger.info(f"Detected {len(scanner.detected_markers)} ArUco markers")
        else:
            logger.warning("No valid point cloud generated")
        
        logger.info("Scan complete!")
        
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