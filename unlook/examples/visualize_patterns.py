#!/usr/bin/env python3
"""
Visualize Pattern Types

This script generates and displays examples of all pattern types
available in the Unlook SDK for easy comparison.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook.client.patterns import (
    MazePatternGenerator,
    VoronoiPatternGenerator,
    HybridArUcoPatternGenerator
)

# Import standard patterns too
from unlook.client.scanning.patterns.enhanced_patterns import (
    generate_multi_scale_patterns,
    generate_multi_frequency_patterns,
    generate_variable_width_gray_code
)


def create_pattern_grid():
    """Create a grid showing all pattern types."""
    # Pattern generation parameters
    width, height = 640, 480
    
    print("Generating pattern examples...")
    
    patterns = {}
    
    # Generate standard patterns
    print("  Generating multi-scale Gray code...")
    multi_scale = generate_multi_scale_patterns(width, height, num_bits=8, orientation="vertical")
    patterns["Multi-Scale Gray"] = multi_scale[2]["image"] if len(multi_scale) > 2 else np.zeros((height, width))
    
    print("  Generating multi-frequency phase shift...")
    multi_freq = generate_multi_frequency_patterns(width, height, frequencies=[1, 2, 4], steps_per_frequency=4)
    patterns["Multi-Frequency"] = multi_freq[2]["image"] if len(multi_freq) > 2 else np.zeros((height, width))
    
    print("  Generating variable-width Gray code...")
    var_width = generate_variable_width_gray_code(width, height, min_bits=4, max_bits=10, orientation="horizontal")
    patterns["Variable Width"] = var_width[2]["image"] if len(var_width) > 2 else np.zeros((height, width))
    
    # Generate new pattern types
    print("  Generating maze pattern...")
    maze_gen = MazePatternGenerator(width, height)
    patterns["Maze"] = maze_gen.generate(algorithm="recursive_backtrack")
    
    print("  Generating Voronoi pattern...")
    voronoi_gen = VoronoiPatternGenerator(width, height)
    patterns["Voronoi"] = voronoi_gen.generate(num_points=100, color_scheme="grayscale")
    
    print("  Generating hybrid ArUco pattern...")
    hybrid_gen = HybridArUcoPatternGenerator(width, height)
    hybrid_pattern, _ = hybrid_gen.generate(base_pattern="gray_code", num_markers=9)
    patterns["Hybrid ArUco"] = hybrid_pattern
    
    # Create grid
    grid_cols = 3
    grid_rows = 2
    
    # Create title cards
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    title_height = 40
    
    # Create the final grid image
    cell_width = width
    cell_height = height + title_height
    
    grid_width = cell_width * grid_cols
    grid_height = cell_height * grid_rows
    
    grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # Place patterns in grid
    pattern_names = list(patterns.keys())
    
    for i, (name, pattern) in enumerate(patterns.items()):
        row = i // grid_cols
        col = i % grid_cols
        
        x_start = col * cell_width
        y_start = row * cell_height
        
        # Add title
        title_area = np.zeros((title_height, cell_width), dtype=np.uint8)
        title_area.fill(64)  # Dark gray background
        
        # Center the text
        text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
        text_x = (cell_width - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        
        cv2.putText(title_area, name, (text_x, text_y), font, font_scale, 255, font_thickness)
        
        # Place title and pattern
        grid_image[y_start:y_start+title_height, x_start:x_start+cell_width] = title_area
        
        # Resize pattern if needed and place it
        if pattern.shape != (height, width):
            if len(pattern.shape) == 3:
                pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
            pattern = cv2.resize(pattern, (width, height))
        
        grid_image[y_start+title_height:y_start+cell_height, x_start:x_start+cell_width] = pattern
    
    return grid_image


def main():
    """Generate and display pattern comparison."""
    print("Pattern Type Visualization")
    print("=" * 40)
    
    # Create pattern grid
    grid = create_pattern_grid()
    
    # Save the grid
    output_file = "pattern_types_comparison.png"
    cv2.imwrite(output_file, grid)
    print(f"\nSaved pattern comparison to: {output_file}")
    
    # Display the grid
    window_name = "Pattern Types Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid)
    
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Also create individual high-res examples
    print("\nGenerating high-resolution individual examples...")
    
    # High-res parameters
    hires_width, hires_height = 1920, 1080
    
    # Generate high-res examples
    maze_gen = MazePatternGenerator(hires_width, hires_height)
    maze_hires = maze_gen.generate(algorithm="recursive_backtrack")
    cv2.imwrite("pattern_maze_hires.png", maze_hires)
    
    voronoi_gen = VoronoiPatternGenerator(hires_width, hires_height)
    voronoi_hires = voronoi_gen.generate(num_points=200, color_scheme="colored")
    cv2.imwrite("pattern_voronoi_hires.png", voronoi_hires)
    
    hybrid_gen = HybridArUcoPatternGenerator(hires_width, hires_height)
    hybrid_hires, markers = hybrid_gen.generate(base_pattern="checkerboard", num_markers=16)
    cv2.imwrite("pattern_hybrid_aruco_hires.png", hybrid_hires)
    
    print("High-resolution examples saved!")
    print(f"  - pattern_maze_hires.png")
    print(f"  - pattern_voronoi_hires.png")
    print(f"  - pattern_hybrid_aruco_hires.png")


if __name__ == "__main__":
    main()