#!/usr/bin/env python3
"""
Analyze captured images to diagnose pattern decoding issues.
Version 2 - Works with actual file naming convention.
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
    
    # Load reference images - using actual filenames
    black_left = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_GRAYSCALE)
    white_left = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_GRAYSCALE)
    black_right = cv2.imread(str(pattern_dir / "black_reference_right.png"), cv2.IMREAD_GRAYSCALE)
    white_right = cv2.imread(str(pattern_dir / "white_reference_right.png"), cv2.IMREAD_GRAYSCALE)
    
    if black_left is None or white_left is None:
        print("Failed to load reference images")
        print(f"Looking for: {pattern_dir / 'black_reference_left.png'}")
        return
    
    # Analyze reference images
    print("\nReference Image Analysis:")
    print(f"Black left: min={np.min(black_left)}, max={np.max(black_left)}, mean={np.mean(black_left):.1f}")
    print(f"White left: min={np.min(white_left)}, max={np.max(white_left)}, mean={np.mean(white_left):.1f}")
    print(f"Dynamic range left: {np.mean(white_left) - np.mean(black_left):.1f}")
    print(f"Black right: min={np.min(black_right)}, max={np.max(black_right)}, mean={np.mean(black_right):.1f}")
    print(f"White right: min={np.min(white_right)}, max={np.max(white_right)}, mean={np.mean(white_right):.1f}")
    print(f"Dynamic range right: {np.mean(white_right) - np.mean(black_right):.1f}")
    
    # Analyze color cast
    print("\nColor Analysis (checking for purple cast):")
    black_color = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_COLOR)
    white_color = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_COLOR)
    
    if black_color is not None:
        b, g, r = cv2.split(black_color)
        print(f"Black RGB means: R={np.mean(r):.1f}, G={np.mean(g):.1f}, B={np.mean(b):.1f}")
        b, g, r = cv2.split(white_color)
        print(f"White RGB means: R={np.mean(r):.1f}, G={np.mean(g):.1f}, B={np.mean(b):.1f}")
    
    # Load pattern images
    pattern_h_left = cv2.imread(str(pattern_dir / "gray_h_bit00_left.png"), cv2.IMREAD_GRAYSCALE)
    pattern_h_right = cv2.imread(str(pattern_dir / "gray_h_bit00_right.png"), cv2.IMREAD_GRAYSCALE)
    pattern_h_inv_left = cv2.imread(str(pattern_dir / "gray_h_bit00_inv_left.png"), cv2.IMREAD_GRAYSCALE)
    
    if pattern_h_left is not None:
        print(f"\nPattern image analysis:")
        print(f"Pattern left: min={np.min(pattern_h_left)}, max={np.max(pattern_h_left)}, mean={np.mean(pattern_h_left):.1f}")
        print(f"Pattern contrast: {np.std(pattern_h_left):.1f}")
        
        # Check difference between normal and inverted
        if pattern_h_inv_left is not None:
            diff = pattern_h_left.astype(np.float32) - pattern_h_inv_left.astype(np.float32)
            print(f"Pattern-Inverted difference: min={np.min(diff):.1f}, max={np.max(diff):.1f}, std={np.std(diff):.1f}")
    
    # Test enhanced processing
    print("\nTesting Enhanced Processing:")
    processor = EnhancedPatternProcessor(enhancement_level=3)
    
    # Load horizontal patterns
    h_patterns_left = []
    h_patterns_inv_left = []
    for i in range(5):  # 5 bits
        normal_path = pattern_dir / f"gray_h_bit{i:02d}_left.png"
        inv_path = pattern_dir / f"gray_h_bit{i:02d}_inv_left.png"
        
        if normal_path.exists() and inv_path.exists():
            normal = cv2.imread(str(normal_path), cv2.IMREAD_GRAYSCALE)
            inverted = cv2.imread(str(inv_path), cv2.IMREAD_GRAYSCALE)
            h_patterns_left.append(normal)
            h_patterns_inv_left.append(inverted)
            
            # Analyze this pattern pair
            diff = normal.astype(float) - inverted.astype(float)
            print(f"Bit {i}: diff range={np.min(diff):.1f} to {np.max(diff):.1f}, std={np.std(diff):.1f}")
    
    # Combine normal and inverted patterns as expected by decode_patterns
    all_h_patterns = []
    for i in range(len(h_patterns_left)):
        all_h_patterns.append(h_patterns_left[i])
        all_h_patterns.append(h_patterns_inv_left[i])
    
    if all_h_patterns:
        # Process with enhancement
        enhanced = processor.preprocess_images(all_h_patterns[:2], black_left, white_left)
        
        print(f"\nOriginal pattern range: {np.min(all_h_patterns[0])}-{np.max(all_h_patterns[0])}")
        print(f"Enhanced pattern range: {np.min(enhanced[0])}-{np.max(enhanced[0])}")
        
        # Save comparison
        comparison = np.hstack([all_h_patterns[0], enhanced[0]])
        cv2.imwrite("pattern_comparison.png", comparison)
        print("Saved pattern comparison to pattern_comparison.png")
    
    # Try decoding
    print("\nTesting Pattern Decoding:")
    if len(all_h_patterns) >= 10:
        # Normal decode
        x_coord, x_conf, x_mask = decode_patterns(
            white_left, black_left, all_h_patterns[:10],
            num_bits=5, orientation="horizontal"
        )
        
        valid_count = np.sum(x_mask)
        print(f"Normal decode: {valid_count} valid pixels ({100*valid_count/x_mask.size:.1f}%)")
        
        # Enhanced decode  
        enhanced_patterns = processor.preprocess_images(all_h_patterns[:10], black_left, white_left)
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
            # Normalize coordinates for visualization
            coord_range = np.max(x_coord_e[x_mask_e]) - np.min(x_coord_e[x_mask_e])
            if coord_range > 0:
                x_vis = ((x_coord_e - np.min(x_coord_e[x_mask_e])) / coord_range * 255).astype(np.uint8)
            else:
                x_vis = np.zeros_like(x_coord_e, dtype=np.uint8)
            x_vis[~x_mask_e] = 0
            x_vis_color = cv2.applyColorMap(x_vis, cv2.COLORMAP_JET)
            cv2.imwrite("decoded_coordinates.png", x_vis_color)
            print("Saved decoded coordinates to decoded_coordinates.png")
            
            # Print coordinate statistics
            print(f"\nDecoded coordinate stats:")
            print(f"Min: {np.min(x_coord_e[x_mask_e])}")
            print(f"Max: {np.max(x_coord_e[x_mask_e])}")
            print(f"Unique values: {len(np.unique(x_coord_e[x_mask_e]))}")
    
    # Save sample enhanced patterns
    if enhanced_patterns:
        for i in range(min(4, len(enhanced_patterns))):
            cv2.imwrite(f"enhanced_pattern_{i}.png", enhanced_patterns[i])
        print(f"\nSaved {min(4, len(enhanced_patterns))} enhanced patterns for inspection")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze captured pattern images")
    parser.add_argument("image_dir", help="Directory containing captured images")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" PATTERN IMAGE ANALYSIS V2")
    print("="*70)
    
    analyze_pattern_quality(args.image_dir)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()