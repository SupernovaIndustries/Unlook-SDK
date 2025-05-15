#!/usr/bin/env python3
"""
Simple Pattern Test - No GUI Version

Tests the new patterns by converting them to projector-compatible formats.

Usage:
    python simple_pattern_test.py
"""

import sys
import time
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook import UnlookClient
from unlook.client.patterns import (
    MazePatternGenerator, 
    VoronoiPatternGenerator,
    HybridArUcoPatternGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pattern_to_stripes(pattern, orientation='horizontal'):
    """Convert pattern to vertical or horizontal stripes for projector."""
    if orientation == 'horizontal':
        # Sample horizontal lines
        h, w = pattern.shape[:2]
        stripes = []
        step = h // 20  # Sample 20 lines
        for y in range(0, h, step):
            stripe = np.mean(pattern[y, :]) > 128
            stripes.append(stripe)
    else:
        # Sample vertical lines
        h, w = pattern.shape[:2]
        stripes = []
        step = w // 20  # Sample 20 lines
        for x in range(0, w, step):
            stripe = np.mean(pattern[:, x]) > 128
            stripes.append(stripe)
    
    return stripes


def project_pattern_as_stripes(client, pattern, name, duration=3):
    """Project pattern as horizontal/vertical stripes."""
    logger.info(f"Projecting {name} as stripes")
    
    # Convert to grayscale if needed
    if len(pattern.shape) == 3:
        import cv2
        pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2GRAY)
    
    # First show horizontal stripes
    h_stripes = pattern_to_stripes(pattern, 'horizontal')
    stripe_width = 4
    
    # Show using horizontal lines pattern
    for i, is_white in enumerate(h_stripes[:10]):  # Use first 10 stripes
        if is_white:
            client.projector.show_horizontal_lines(
                foreground_color="White",
                background_color="Black",
                foreground_width=stripe_width,
                background_width=stripe_width * (i + 1)
            )
            time.sleep(0.2)
    
    time.sleep(duration)
    
    # Then show as checkerboard approximation
    client.projector.show_checkerboard(
        horizontal_count=20,
        vertical_count=20
    )
    time.sleep(duration)


def main():
    """Main test function."""
    # Create client
    client = UnlookClient(auto_discover=True)
    
    try:
        # Start discovery
        client.start_discovery()
        logger.info("Discovering scanners for 5 seconds...")
        time.sleep(5)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found")
            return
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Test patterns
        logger.info("\n=== Pattern Projection Test ===")
        duration = 2  # seconds per pattern
        
        # 1. Simple solid colors
        logger.info("Testing solid colors...")
        colors = ["White", "Black", "Red", "Green", "Blue"]
        for color in colors:
            logger.info(f"Projecting {color}")
            client.projector.show_solid_field(color)
            time.sleep(duration)
        
        # 2. Built-in patterns
        logger.info("\nTesting built-in patterns...")
        
        # Horizontal lines
        logger.info("Horizontal lines - thin")
        client.projector.show_horizontal_lines(
            foreground_width=2, 
            background_width=10
        )
        time.sleep(duration)
        
        logger.info("Horizontal lines - thick")
        client.projector.show_horizontal_lines(
            foreground_width=10, 
            background_width=30
        )
        time.sleep(duration)
        
        # Vertical lines
        logger.info("Vertical lines")
        client.projector.show_vertical_lines(
            foreground_width=5, 
            background_width=20
        )
        time.sleep(duration)
        
        # Checkerboard
        logger.info("Checkerboard")
        client.projector.show_checkerboard(horizontal_count=20, vertical_count=20)
        time.sleep(duration)
        
        # 3. Test new pattern generators
        logger.info("\nTesting new pattern generators...")
        
        # Maze pattern
        logger.info("Generating maze pattern...")
        maze_gen = MazePatternGenerator(1280, 720)
        maze_pattern = maze_gen.generate()
        project_pattern_as_stripes(client, maze_pattern, "Maze Pattern", duration)
        
        # Voronoi pattern
        logger.info("Generating Voronoi pattern...")
        voronoi_gen = VoronoiPatternGenerator(1280, 720)
        voronoi_pattern = voronoi_gen.generate(num_points=50)
        project_pattern_as_stripes(client, voronoi_pattern, "Voronoi Pattern", duration)
        
        # Turn off projector
        logger.info("\nTurning off projector")
        client.projector.show_solid_field("Black")
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Disconnecting from scanner")
        client.disconnect()


if __name__ == "__main__":
    main()