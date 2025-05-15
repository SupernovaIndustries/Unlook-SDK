#!/usr/bin/env python3
"""
Simple Pattern Projection Test

This script projects the generated patterns using the UnLook projector.
Press SPACE to cycle through patterns, ESC to exit.

Usage:
    python simple_pattern_projection.py [--host SCANNER_IP]
"""

import sys
import time
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path

# Add parent directory to path for imports
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


def generate_patterns():
    """Generate all test patterns."""
    patterns = []
    
    # Maze patterns
    logger.info("Generating maze patterns...")
    maze_gen = MazePatternGenerator(1280, 720)
    algorithms = ["recursive_backtrack", "prim", "kruskal"]
    for algo in algorithms:
        pattern = maze_gen.generate(algorithm=algo)
        patterns.append((f"Maze - {algo}", pattern))
    
    # Voronoi patterns
    logger.info("Generating Voronoi patterns...")
    voronoi_gen = VoronoiPatternGenerator(1280, 720)
    configs = [
        (50, "grayscale"),
        (100, "binary"),
        (75, "colored")
    ]
    for num_points, color_scheme in configs:
        pattern = voronoi_gen.generate(num_points=num_points, color_scheme=color_scheme)
        patterns.append((f"Voronoi - {num_points} points ({color_scheme})", pattern))
    
    # Hybrid ArUco patterns
    logger.info("Generating hybrid ArUco patterns...")
    try:
        hybrid_gen = HybridArUcoPatternGenerator(1280, 720)
        hybrid_configs = [
            ("gray_code", 4),
            ("phase_shift", 9),
            ("checkerboard", 16)
        ]
        for base_pattern, num_markers in hybrid_configs:
            pattern, _ = hybrid_gen.generate(base_pattern=base_pattern, num_markers=num_markers)
            patterns.append((f"Hybrid - {base_pattern} ({num_markers} markers)", pattern))
    except ImportError as e:
        logger.warning(f"Cannot generate Hybrid ArUco patterns: {e}")
    
    return patterns


def main():
    """Main function to project patterns."""
    parser = argparse.ArgumentParser(description="Simple Pattern Projection Test")
    parser.add_argument("--host", help="Scanner IP address (optional)")
    parser.add_argument("--timeout", type=int, default=5, help="Discovery timeout in seconds")
    
    args = parser.parse_args()
    
    # Create client
    client = UnlookClient(auto_discover=True)
    
    try:
        # Start discovery
        client.start_discovery()
        logger.info(f"Discovering scanners for {args.timeout} seconds...")
        time.sleep(args.timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Please ensure scanner hardware is connected and powered on.")
            return
            
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Generate patterns
        patterns = generate_patterns()
        logger.info(f"Generated {len(patterns)} patterns")
        
        # Project patterns
        pattern_index = 0
        
        print("\n" + "="*50)
        print("PATTERN PROJECTION TEST")
        print("="*50)
        print("Controls:")
        print("  SPACE - Next pattern")
        print("  B     - Previous pattern")
        print("  ESC   - Exit")
        print("  W     - Project white")
        print("  K     - Project black")
        print("="*50 + "\n")
        
        while True:
            # Get current pattern
            pattern_name, pattern = patterns[pattern_index]
            
            # Convert to BGR if needed
            if len(pattern.shape) == 3:
                pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
            else:
                pattern_bgr = pattern
            
            # Project pattern as a custom pattern
            logger.info(f"Projecting: {pattern_name}")
            
            # Create pattern dict for projection
            pattern_dict = {
                "pattern_type": "custom",
                "name": pattern_name,
                "image": pattern_bgr
            }
            
            # Use project_pattern method which accepts pattern dictionaries
            client.projector.project_pattern(pattern_dict)
            
            print(f"\nCurrently projecting: {pattern_name}")
            print(f"Pattern {pattern_index + 1} of {len(patterns)}")
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - next pattern
                pattern_index = (pattern_index + 1) % len(patterns)
            elif key == ord('b') or key == ord('B'):  # B - previous pattern
                pattern_index = (pattern_index - 1) % len(patterns)
            elif key == ord('w') or key == ord('W'):  # W - white
                client.projector.show_solid_field("White")
                print("\nProjecting WHITE")
                cv2.waitKey(0)
            elif key == ord('k') or key == ord('K'):  # K - black
                client.projector.show_solid_field("Black")
                print("\nProjecting BLACK")
                cv2.waitKey(0)
        
        # Turn off projector
        logger.info("Turning off projector")
        client.projector.show_solid_field("Black")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Disconnecting from scanner")
        client.disconnect()


if __name__ == "__main__":
    main()