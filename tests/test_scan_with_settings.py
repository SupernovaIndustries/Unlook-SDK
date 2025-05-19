#!/usr/bin/env python3
"""Test scan with optimized camera settings for low ambient light."""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from unlook import UnlookClient
from unlook.client.scanning import StaticScanner, StaticScanConfig
from unlook.client.scan_config import PatternType

def main():
    """Run scan with optimized settings."""
    
    # Create client
    client = UnlookClient(auto_discover=False)
    
    # Start discovery
    client.start_discovery()
    logger.info("Discovering scanners...")
    time.sleep(5)
    
    # Get discovered scanners
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found")
        return 1
    
    # Connect to first scanner
    scanner_info = scanners[0]
    logger.info(f"Connecting to: {scanner_info.name}")
    
    if not client.connect(scanner_info):
        logger.error("Failed to connect")
        return 1
    
    # Create scanner config with enhanced processing
    config = StaticScanConfig(
        quality="high",
        pattern_type=PatternType.ENHANCED_GRAY,
        use_enhanced_processor=True,
        enhancement_level=3,
        enable_auto_optimization=False,
        debug=True,
        save_intermediate_images=True,
        save_raw_images=True
    )
    
    # Create scanner
    scanner = StaticScanner(client, config)
    
    # Apply manual camera settings for better contrast
    camera_settings = {
        'ExposureTime': 500,  # 500μs - from session notes, this worked best
        'AnalogueGain': 2.0   # Higher gain for better pattern visibility
    }
    
    try:
        logger.info(f"Setting camera exposure to {camera_settings['ExposureTime']}μs")
        scanner.set_camera_settings(camera_settings)
    except Exception as e:
        logger.warning(f"Could not set camera settings: {e}")
    
    # Perform scan
    logger.info("Starting scan...")
    point_cloud = scanner.perform_scan()
    
    if point_cloud and hasattr(point_cloud, 'points'):
        num_points = len(point_cloud.points)
        logger.info(f"Scan complete: {num_points} points")
        
        # Save point cloud
        output_file = f"test_scan_{int(time.time())}.ply"
        scanner.save_point_cloud(output_file)
        logger.info(f"Saved to: {output_file}")
    else:
        logger.error("No valid point cloud generated")
    
    # Disconnect
    client.disconnect()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())