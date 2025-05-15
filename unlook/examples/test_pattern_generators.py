#!/usr/bin/env python3
"""
Test Pattern Generators

This script tests the new pattern generators to ensure they work correctly
and generates sample patterns for visualization.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook.client.patterns import (
    MazePatternGenerator, MazePatternDecoder,
    VoronoiPatternGenerator, VoronoiPatternDecoder,
    HybridArUcoPatternGenerator, HybridArUcoPatternDecoder
)

import cv2
import numpy as np


def test_maze_patterns(output_dir="test_patterns"):
    """Test maze pattern generation."""
    print("Testing Maze Pattern Generator...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = MazePatternGenerator(1280, 720)
    algorithms = ["recursive_backtrack", "prim", "kruskal"]
    
    for i, algo in enumerate(algorithms):
        pattern = generator.generate(algorithm=algo)
        filename = os.path.join(output_dir, f"test_maze_{algo}.png")
        cv2.imwrite(filename, pattern)
        print(f"Generated {filename} (shape: {pattern.shape}, dtype: {pattern.dtype})")
    
    # Test sequence generation
    patterns = generator.generate_sequence(num_patterns=3)
    print(f"Generated sequence of {len(patterns)} maze patterns")


def test_voronoi_patterns(output_dir="test_patterns"):
    """Test Voronoi pattern generation."""
    print("\nTesting Voronoi Pattern Generator...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = VoronoiPatternGenerator(1280, 720)
    
    # Test different configurations
    configs = [
        (50, "grayscale"),
        (100, "binary"),
        (75, "colored")
    ]
    
    for num_points, color_scheme in configs:
        pattern = generator.generate(num_points=num_points, color_scheme=color_scheme)
        filename = os.path.join(output_dir, f"test_voronoi_{num_points}pts_{color_scheme}.png")
        cv2.imwrite(filename, pattern)
        print(f"Generated {filename} (shape: {pattern.shape}, dtype: {pattern.dtype})")
    
    # Test sequence generation
    patterns = generator.generate_sequence(num_patterns=4)
    print(f"Generated sequence of {len(patterns)} Voronoi patterns")


def test_hybrid_aruco_patterns(output_dir="test_patterns"):
    """Test hybrid ArUco pattern generation."""
    print("\nTesting Hybrid ArUco Pattern Generator...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = HybridArUcoPatternGenerator(1280, 720)
    
    # Test different base patterns and marker counts
    configs = [
        ("gray_code", 4),
        ("phase_shift", 9),
        ("checkerboard", 16)
    ]
    
    for base_pattern, num_markers in configs:
        pattern, markers_info = generator.generate(base_pattern=base_pattern, num_markers=num_markers)
        filename = os.path.join(output_dir, f"test_hybrid_{base_pattern}_{num_markers}markers.png")
        cv2.imwrite(filename, pattern)
        print(f"Generated {filename} with {len(markers_info)} markers")
        print(f"  Pattern shape: {pattern.shape}, dtype: {pattern.dtype}")
    
    # Generate calibration pattern
    calib_pattern, calib_markers = generator.generate_calibration_pattern()
    filename = os.path.join(output_dir, "test_hybrid_calibration.png")
    cv2.imwrite(filename, calib_pattern)
    print(f"Generated calibration pattern with {len(calib_markers)} markers")


def test_decoders():
    """Test basic decoder functionality."""
    print("\nTesting Pattern Decoders...")
    
    # Test Maze decoder
    maze_decoder = MazePatternDecoder()
    print("MazePatternDecoder initialized successfully")
    
    # Test Voronoi decoder
    voronoi_decoder = VoronoiPatternDecoder()
    print("VoronoiPatternDecoder initialized successfully")
    
    # Test Hybrid ArUco decoder
    hybrid_decoder = HybridArUcoPatternDecoder()
    print("HybridArUcoPatternDecoder initialized successfully")


def main():
    """Run all pattern generator tests."""
    print("Pattern Generator Test Suite")
    print("=" * 40)
    
    output_dir = "test_patterns"
    
    try:
        test_maze_patterns(output_dir)
        test_voronoi_patterns(output_dir)
        test_hybrid_aruco_patterns(output_dir)
        test_decoders()
        
        print("\nAll tests completed successfully!")
        print(f"Generated test patterns in: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()