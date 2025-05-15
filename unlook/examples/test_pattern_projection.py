#!/usr/bin/env python3
"""
Test Pattern Projection and Capture

This script tests the new pattern generators by:
1. Generating patterns using the new pattern generators
2. Projecting them using the UnLook projector
3. Capturing images with the cameras
4. Displaying the results

Usage:
    python test_pattern_projection.py [--host SCANNER_IP]
"""

import sys
import os
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


def capture_with_pattern(client, pattern_name, pattern_image):
    """Project a pattern and capture images."""
    logger.info(f"Projecting pattern: {pattern_name}")
    
    # Convert pattern to correct format if needed
    if len(pattern_image.shape) == 3:
        # Convert RGB to BGR for OpenCV
        pattern_bgr = cv2.cvtColor(pattern_image, cv2.COLOR_RGB2BGR)
    else:
        # Grayscale pattern
        pattern_bgr = pattern_image
    
    # Project the pattern
    try:
        client.projector.project_image(pattern_bgr)
        time.sleep(0.5)  # Wait for projection to stabilize
        
        # Capture images
        left_image = client.capture(stream_index=0)
        right_image = client.capture(stream_index=1)
        
        logger.info(f"Captured images - Left: {left_image.shape}, Right: {right_image.shape}")
        
        return left_image, right_image
    except Exception as e:
        logger.error(f"Error during projection/capture: {e}")
        return None, None


def test_pattern_generators(client, output_dir="pattern_test_results"):
    """Test all pattern generators with projection and capture."""
    logger.info("Testing Pattern Generators with Projection")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test Maze Pattern
    logger.info("\n=== Testing Maze Pattern Generator ===")
    maze_gen = MazePatternGenerator(1280, 720)
    maze_pattern = maze_gen.generate(algorithm="recursive_backtrack")
    
    left_maze, right_maze = capture_with_pattern(client, "Maze Pattern", maze_pattern)
    if left_maze is not None:
        cv2.imwrite(os.path.join(output_dir, "maze_left.png"), left_maze)
        cv2.imwrite(os.path.join(output_dir, "maze_right.png"), right_maze)
        cv2.imwrite(os.path.join(output_dir, "maze_pattern.png"), maze_pattern)
        
        # Show the results
        display_results("Maze Pattern", maze_pattern, left_maze, right_maze)
    
    # Test Voronoi Pattern
    logger.info("\n=== Testing Voronoi Pattern Generator ===")
    voronoi_gen = VoronoiPatternGenerator(1280, 720)
    voronoi_pattern = voronoi_gen.generate(num_points=100, color_scheme='grayscale')
    
    left_voronoi, right_voronoi = capture_with_pattern(client, "Voronoi Pattern", voronoi_pattern)
    if left_voronoi is not None:
        cv2.imwrite(os.path.join(output_dir, "voronoi_left.png"), left_voronoi)
        cv2.imwrite(os.path.join(output_dir, "voronoi_right.png"), right_voronoi)
        cv2.imwrite(os.path.join(output_dir, "voronoi_pattern.png"), voronoi_pattern)
        
        display_results("Voronoi Pattern", voronoi_pattern, left_voronoi, right_voronoi)
    
    # Test Hybrid ArUco Pattern
    logger.info("\n=== Testing Hybrid ArUco Pattern Generator ===")
    hybrid_gen = HybridArUcoPatternGenerator(1280, 720)
    hybrid_pattern, markers_info = hybrid_gen.generate(base_pattern="gray_code", num_markers=9)
    
    left_hybrid, right_hybrid = capture_with_pattern(client, "Hybrid ArUco Pattern", hybrid_pattern)
    if left_hybrid is not None:
        cv2.imwrite(os.path.join(output_dir, "hybrid_left.png"), left_hybrid)
        cv2.imwrite(os.path.join(output_dir, "hybrid_right.png"), right_hybrid)
        cv2.imwrite(os.path.join(output_dir, "hybrid_pattern.png"), hybrid_pattern)
        
        display_results("Hybrid ArUco Pattern", hybrid_pattern, left_hybrid, right_hybrid)
    
    logger.info(f"\nAll captured images saved to: {os.path.abspath(output_dir)}")


def display_results(pattern_name, pattern, left_capture, right_capture, max_width=800):
    """Display the pattern and captured images side by side."""
    # Create a combined image showing pattern and captures
    height = max(pattern.shape[0], left_capture.shape[0], right_capture.shape[0])
    
    # Resize images if needed to fit display
    def resize_to_height(img, target_height):
        scale = target_height / img.shape[0]
        width = int(img.shape[1] * scale)
        return cv2.resize(img, (width, target_height))
    
    pattern_resized = resize_to_height(pattern, height)
    left_resized = resize_to_height(left_capture, height)
    right_resized = resize_to_height(right_capture, height)
    
    # Convert grayscale to BGR for consistent display
    if len(pattern_resized.shape) == 2:
        pattern_resized = cv2.cvtColor(pattern_resized, cv2.COLOR_GRAY2BGR)
    if len(left_resized.shape) == 2:
        left_resized = cv2.cvtColor(left_resized, cv2.COLOR_GRAY2BGR)
    if len(right_resized.shape) == 2:
        right_resized = cv2.cvtColor(right_resized, cv2.COLOR_GRAY2BGR)
    
    # Create separator lines
    separator = np.ones((height, 5, 3), dtype=np.uint8) * 255
    
    # Combine images horizontally
    combined = np.hstack([pattern_resized, separator, left_resized, separator, right_resized])
    
    # Add labels
    cv2.putText(combined, "Pattern", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Left Camera", (pattern_resized.shape[1] + 15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Right Camera", 
                (pattern_resized.shape[1] + left_resized.shape[1] + 25, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Resize if too wide for display
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        new_height = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (max_width, new_height))
    
    cv2.imshow(pattern_name, combined)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyWindow(pattern_name)


def main():
    """Main function to run pattern projection tests."""
    parser = argparse.ArgumentParser(description="Test Pattern Projection and Capture")
    parser.add_argument("--host", help="Scanner IP address (optional)")
    parser.add_argument("--timeout", type=int, default=5, help="Discovery timeout in seconds")
    parser.add_argument("--no-display", action="store_true", help="Don't display images")
    parser.add_argument("--output-dir", default="pattern_test_results", 
                       help="Output directory for captured images")
    
    args = parser.parse_args()
    
    # Global flag for display
    global DISPLAY_IMAGES
    DISPLAY_IMAGES = not args.no_display
    
    # Create client
    client = UnlookClient(auto_discover=True)
    
    try:
        # Discover or connect to scanner
        if args.host:
            logger.info(f"Connecting to scanner at {args.host}")
            scanner_info = type('ScannerInfo', (), {
                'ip': args.host,
                'port': 5000,
                'name': 'UnlookScanner',
                'model': 'Unknown'
            })()
            client.connect(scanner_info)
        else:
            logger.info("Discovering scanners...")
            discovered = client.discover_scanners(timeout=args.timeout)
            
            if not discovered:
                logger.error("No scanners found")
                return
            
            logger.info(f"Found {len(discovered)} scanner(s)")
            scanner = discovered[0]
            logger.info(f"Connecting to {scanner.name} ({scanner.ip})")
            client.connect(scanner)
        
        logger.info("Connected to scanner")
        
        # Run pattern tests
        test_pattern_generators(client, args.output_dir)
        
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