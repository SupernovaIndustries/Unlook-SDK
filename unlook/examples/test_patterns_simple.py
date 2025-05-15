#!/usr/bin/env python3
"""
Simple test for pattern generators
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np


def test_maze():
    """Test maze generation."""
    from unlook.client.patterns import MazePatternGenerator
    
    print("Testing Maze Pattern...")
    generator = MazePatternGenerator(640, 480)
    pattern = generator.generate(algorithm="recursive_backtrack")
    cv2.imwrite("simple_maze.png", pattern)
    print(f"  Saved: simple_maze.png")


def test_voronoi():
    """Test Voronoi generation."""
    from unlook.client.patterns import VoronoiPatternGenerator
    
    print("Testing Voronoi Pattern...")
    generator = VoronoiPatternGenerator(640, 480)
    pattern = generator.generate(num_points=50, color_scheme="grayscale")
    cv2.imwrite("simple_voronoi.png", pattern)
    print(f"  Saved: simple_voronoi.png")


def test_hybrid():
    """Test hybrid ArUco generation."""
    from unlook.client.patterns import HybridArUcoPatternGenerator
    
    print("Testing Hybrid ArUco Pattern...")
    generator = HybridArUcoPatternGenerator(640, 480)
    pattern, markers = generator.generate(base_pattern="checkerboard", num_markers=4)
    cv2.imwrite("simple_hybrid.png", pattern)
    print(f"  Saved: simple_hybrid.png (with {len(markers)} markers)")


def main():
    """Run simple tests."""
    print("Simple Pattern Test")
    print("=" * 30)
    
    try:
        test_maze()
        test_voronoi()
        test_hybrid()
        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()