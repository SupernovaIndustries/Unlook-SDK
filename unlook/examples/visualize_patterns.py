#!/usr/bin/env python3
"""
Visualize Pattern Types

This script generates and displays examples of all pattern types
available in the Unlook SDK for easy comparison.
"""

import sys
import os
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


def create_pattern_grid():
    """Create a grid showing all pattern types."""
    # Pattern generation parameters
    width, height = 640, 480
    
    print("Generating pattern examples...")
    
    patterns = {}
    
    # Generate new pattern types
    print("  Generating maze patterns...")
    maze_gen = MazePatternGenerator(width, height)
    patterns["Maze (Recursive)"] = maze_gen.generate(algorithm="recursive_backtrack")
    patterns["Maze (Prim)"] = maze_gen.generate(algorithm="prim")
    patterns["Maze (Kruskal)"] = maze_gen.generate(algorithm="kruskal")
    
    print("  Generating Voronoi patterns...")
    voronoi_gen = VoronoiPatternGenerator(width, height)
    patterns["Voronoi (Grayscale)"] = voronoi_gen.generate(num_points=50, color_scheme="grayscale")
    patterns["Voronoi (Binary)"] = voronoi_gen.generate(num_points=100, color_scheme="binary")
    
    print("  Generating hybrid ArUco patterns...")
    hybrid_gen = HybridArUcoPatternGenerator(width, height)
    hybrid_gray, _ = hybrid_gen.generate(base_pattern="gray_code", num_markers=9)
    patterns["Hybrid ArUco"] = hybrid_gray
    
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


def save_individual_patterns(output_dir="patterns"):
    """Save individual pattern examples to files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # High-res parameters
    hires_width, hires_height = 1920, 1080
    
    print(f"\nGenerating high-resolution patterns in '{output_dir}' folder...")
    
    # Generate maze patterns
    maze_gen = MazePatternGenerator(hires_width, hires_height)
    algorithms = ["recursive_backtrack", "prim", "kruskal"]
    
    for algo in algorithms:
        pattern = maze_gen.generate(algorithm=algo)
        filename = os.path.join(output_dir, f"maze_{algo}.png")
        cv2.imwrite(filename, pattern)
        print(f"  Saved: {filename}")
    
    # Generate Voronoi patterns
    voronoi_gen = VoronoiPatternGenerator(hires_width, hires_height)
    voronoi_configs = [
        (50, "grayscale"),
        (100, "binary"), 
        (75, "colored")
    ]
    
    for num_points, color_scheme in voronoi_configs:
        pattern = voronoi_gen.generate(num_points=num_points, color_scheme=color_scheme)
        filename = os.path.join(output_dir, f"voronoi_{num_points}pts_{color_scheme}.png")
        cv2.imwrite(filename, pattern)
        print(f"  Saved: {filename}")
    
    # Generate hybrid ArUco patterns
    hybrid_gen = HybridArUcoPatternGenerator(hires_width, hires_height)
    hybrid_configs = [
        ("gray_code", 4),
        ("phase_shift", 9),
        ("checkerboard", 16)
    ]
    
    for base_pattern, num_markers in hybrid_configs:
        pattern, markers = hybrid_gen.generate(base_pattern=base_pattern, num_markers=num_markers)
        filename = os.path.join(output_dir, f"hybrid_{base_pattern}_{num_markers}markers.png")
        cv2.imwrite(filename, pattern)
        print(f"  Saved: {filename} (with {len(markers)} markers)")
    
    # Generate calibration pattern
    calib_pattern, calib_markers = hybrid_gen.generate_calibration_pattern()
    filename = os.path.join(output_dir, "hybrid_calibration.png")
    cv2.imwrite(filename, calib_pattern)
    print(f"  Saved: {filename} (calibration pattern with {len(calib_markers)} markers)")


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
    
    # Save individual patterns
    save_individual_patterns()
    
    # Display the grid
    window_name = "Pattern Types Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid)
    
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()