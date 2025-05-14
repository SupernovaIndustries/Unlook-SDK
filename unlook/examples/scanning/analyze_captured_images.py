#!/usr/bin/env python3
"""
Analyze captured images to diagnose pattern decoding issues.
"""

import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns
from unlook.client.scanning.patterns.enhanced_pattern_processor import EnhancedPatternProcessor


def analyze_pattern_quality(image_dir):
    """Analyze the quality of captured patterns."""
    pattern_dir = Path(image_dir) / "01_patterns" / "raw"
    
    if not pattern_dir.exists():
        pattern_dir = Path(image_dir)
    
    print(f"\nAnalyzing patterns in: {pattern_dir}")
    
    # Load reference images
    black_left = cv2.imread(str(pattern_dir / "pattern_00_black_camera_0.png"), cv2.IMREAD_GRAYSCALE)
    white_left = cv2.imread(str(pattern_dir / "pattern_01_white_camera_0.png"), cv2.IMREAD_GRAYSCALE)
    black_right = cv2.imread(str(pattern_dir / "pattern_00_black_camera_1.png"), cv2.IMREAD_GRAYSCALE)
    white_right = cv2.imread(str(pattern_dir / "pattern_01_white_camera_1.png"), cv2.IMREAD_GRAYSCALE)
    
    if black_left is None or white_left is None:
        print("Failed to load reference images")
        return
    
    # Analyze reference images
    print("\nReference Image Analysis:")
    print(f"Black left: min={np.min(black_left)}, max={np.max(black_left)}, mean={np.mean(black_left):.1f}")
    print(f"White left: min={np.min(white_left)}, max={np.max(white_left)}, mean={np.mean(white_left):.1f}")
    print(f"Dynamic range left: {np.mean(white_left) - np.mean(black_left):.1f}")
    print(f"Black right: min={np.min(black_right)}, max={np.max(black_right)}, mean={np.mean(black_right):.1f}")
    print(f"White right: min={np.min(white_right)}, max={np.max(white_right)}, mean={np.mean(white_right):.1f}")
    print(f"Dynamic range right: {np.mean(white_right) - np.mean(black_right):.1f}")
    
    # Load a pattern image
    pattern_h_left = cv2.imread(str(pattern_dir / "pattern_02_gray_horizontal_00_camera_0.png"), cv2.IMREAD_GRAYSCALE)
    pattern_h_right = cv2.imread(str(pattern_dir / "pattern_02_gray_horizontal_00_camera_1.png"), cv2.IMREAD_GRAYSCALE)
    
    if pattern_h_left is not None:
        print(f"\nPattern image analysis:")
        print(f"Pattern left: min={np.min(pattern_h_left)}, max={np.max(pattern_h_left)}, mean={np.mean(pattern_h_left):.1f}")
        print(f"Pattern contrast: {np.std(pattern_h_left):.1f}")
    
    # Test enhanced processing
    print("\nTesting Enhanced Processing:")
    processor = EnhancedPatternProcessor(enhancement_level=3)
    
    # Load horizontal patterns
    h_patterns_left = []
    for i in range(10):
        img_path = pattern_dir / f"pattern_{i+2:02d}_gray_horizontal_{i//2:02d}_camera_0.png"
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            h_patterns_left.append(img)
    
    if h_patterns_left:
        # Process with enhancement
        enhanced = processor.preprocess_images(h_patterns_left[:2], black_left, white_left)
        
        print(f"Original pattern range: {np.min(h_patterns_left[0])}-{np.max(h_patterns_left[0])}")
        print(f"Enhanced pattern range: {np.min(enhanced[0])}-{np.max(enhanced[0])}")
        
        # Save comparison
        comparison = np.hstack([h_patterns_left[0], enhanced[0]])
        cv2.imwrite("pattern_comparison.png", comparison)
        print("Saved pattern comparison to pattern_comparison.png")
    
    # Try decoding
    print("\nTesting Pattern Decoding:")
    if len(h_patterns_left) >= 10:
        # Normal decode
        x_coord, x_conf, x_mask = decode_patterns(
            white_left, black_left, h_patterns_left[:10],
            num_bits=5, orientation="horizontal"
        )
        
        valid_count = np.sum(x_mask)
        print(f"Normal decode: {valid_count} valid pixels ({100*valid_count/x_mask.size:.1f}%)")
        
        # Enhanced decode  
        enhanced_patterns = processor.preprocess_images(h_patterns_left[:10], black_left, white_left)
        x_coord_e, x_conf_e, x_mask_e = decode_patterns(
            white_left, black_left, enhanced_patterns,
            num_bits=5, orientation="horizontal"
        )
        
        valid_count_e = np.sum(x_mask_e)
        print(f"Enhanced decode: {valid_count_e} valid pixels ({100*valid_count_e/x_mask_e.size:.1f}%)")
        
        # Save mask visualization
        mask_comparison = np.hstack([x_mask.astype(np.uint8)*255, x_mask_e.astype(np.uint8)*255])
        cv2.imwrite("mask_comparison.png", mask_comparison)
        print("Saved mask comparison to mask_comparison.png")
        
        # Save decoded coordinate visualization
        if valid_count_e > 0:
            x_vis = ((x_coord_e - np.min(x_coord_e[x_mask_e])) / 
                    (np.max(x_coord_e[x_mask_e]) - np.min(x_coord_e[x_mask_e])) * 255).astype(np.uint8)
            x_vis[~x_mask_e] = 0
            x_vis_color = cv2.applyColorMap(x_vis, cv2.COLORMAP_JET)
            cv2.imwrite("decoded_coordinates.png", x_vis_color)
            print("Saved decoded coordinates to decoded_coordinates.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze captured pattern images")
    parser.add_argument("image_dir", help="Directory containing captured images")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" PATTERN IMAGE ANALYSIS")
    print("="*70)
    
    analyze_pattern_quality(args.image_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()