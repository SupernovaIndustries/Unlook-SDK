#!/usr/bin/env python3
"""
Test script for the enhanced pattern processor.
Shows how to use it standalone or with the scanner.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook import UnlookClient
from unlook.client.scanning import StaticScanner, StaticScanConfig
from unlook.client.scanning.patterns.enhanced_pattern_processor import EnhancedPatternProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_processor_standalone():
    """Test the enhanced processor in standalone mode."""
    logger.info("Enhanced processor is DISABLED for demo reliability")
    print("\nEnhanced processor testing is DISABLED for demo reliability.")
    print("This test is not available during demo mode.")
    return
    
    # Create processor with different enhancement levels
    processor = EnhancedPatternProcessor(enhancement_level=2)
    
    # Example: Process an existing debug folder
    # You would replace this with actual image loading
    import cv2
    import numpy as np
    
    # Create dummy test pattern
    test_pattern = np.zeros((480, 640), dtype=np.uint8)
    test_pattern[:, ::2] = 255  # Vertical stripes
    
    # Add noise and reduce contrast
    noise = np.random.normal(0, 20, test_pattern.shape)
    noisy_pattern = np.clip(test_pattern.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Apply enhancement
    enhanced = processor.enhance_single_image(noisy_pattern, level=3)
    
    logger.info("Processed test pattern successfully")
    
    # Save results for comparison
    cv2.imwrite("test_pattern_original.png", noisy_pattern)
    cv2.imwrite("test_pattern_enhanced.png", enhanced)
    logger.info("Saved test patterns to test_pattern_original.png and test_pattern_enhanced.png")


def test_scanner_with_enhanced_processor():
    """Test the scanner with enhanced processor enabled."""
    logger.info("Testing scanner with enhanced processor")
    
    # Create config with enhanced processor
    config = StaticScanConfig(
        quality="high",
        use_enhanced_processor=True,
        enhancement_level=3,  # Maximum enhancement
        debug=True
    )
    
    # Discovery timeout
    timeout = 10
    
    try:
        # Create client
        client = UnlookClient(auto_discover=True)
        client.start_discovery()
        logger.info(f"Discovering scanners for {timeout} seconds...")
        import time
        time.sleep(timeout)
        
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found")
            return
        
        # Connect to first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to {scanner_info.name}")
        if not client.connect(scanner_info):
            logger.error("Failed to connect")
            return
        
        # Create scanner
        scanner = StaticScanner(client=client, config=config)
        
        # Perform scan
        logger.info("Starting enhanced scan...")
        point_cloud = scanner.perform_scan()
        
        if point_cloud and hasattr(point_cloud, 'points') and len(point_cloud.points) > 0:
            scanner.save_point_cloud("enhanced_scan.ply")
            logger.info(f"Saved enhanced scan with {len(point_cloud.points)} points")
        else:
            logger.warning("No points in scan result")
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Test enhanced pattern processor")
    parser.add_argument("--mode", choices=["standalone", "scanner"], default="standalone",
                       help="Test mode: standalone processor or with scanner")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" ENHANCED PATTERN PROCESSOR TEST")
    print("="*70 + "\n")
    
    if args.mode == "standalone":
        test_processor_standalone()
    else:
        test_scanner_with_enhanced_processor()
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()