#!/usr/bin/env python3
"""
Diagnose the pattern decoding issue - focused analysis.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def diagnose_patterns(image_dir):
    """Diagnose why patterns aren't decoding properly."""
    pattern_dir = Path(image_dir) / "01_patterns" / "raw"
    
    print(f"\nDIAGNOSTIC ANALYSIS")
    print("="*50)
    
    # Load reference images
    black_left = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_GRAYSCALE)
    white_left = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_GRAYSCALE)
    
    # Key insight: The white reference is DARKER than the black reference!
    print("\n1. REFERENCE IMAGE ISSUE:")
    print(f"Black mean: {np.mean(black_left):.1f}")
    print(f"White mean: {np.mean(white_left):.1f}")
    print(f"White - Black: {np.mean(white_left) - np.mean(black_left):.1f}")
    print("⚠️ White reference is DARKER than black reference!")
    
    # Check if references are swapped
    if np.mean(white_left) < np.mean(black_left):
        print("\n🔄 Swapping references...")
        white_left, black_left = black_left, white_left
        print(f"After swap - Black mean: {np.mean(black_left):.1f}")
        print(f"After swap - White mean: {np.mean(white_left):.1f}")
    
    # Analyze pattern differences
    print("\n2. PATTERN ANALYSIS:")
    for i in range(5):
        normal = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
        inverted = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_inv_left.png"), cv2.IMREAD_GRAYSCALE)
        
        if normal is not None and inverted is not None:
            # Check actual differences
            diff = normal.astype(float) - inverted.astype(float)
            
            # Create a threshold mask
            threshold = 10  # Minimum difference to consider valid
            valid_diff = np.abs(diff) > threshold
            
            print(f"\nBit {i}:")
            print(f"  Difference range: {np.min(diff):.1f} to {np.max(diff):.1f}")
            print(f"  Valid pixels (|diff| > {threshold}): {np.sum(valid_diff)} ({100*np.sum(valid_diff)/diff.size:.1f}%)")
            
            # Check stripe pattern
            if i == 0:  # Most significant bit - should have the widest stripes
                # Sample a horizontal line through the middle
                mid_row = normal.shape[0] // 2
                normal_line = normal[mid_row, :]
                inv_line = inverted[mid_row, :]
                diff_line = diff[mid_row, :]
                
                # Save line plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(normal_line, label='Normal')
                plt.plot(inv_line, label='Inverted')
                plt.legend()
                plt.title(f'Bit {i} - Horizontal Profile')
                plt.ylabel('Intensity')
                
                plt.subplot(2, 1, 2)
                plt.plot(diff_line)
                plt.title('Difference')
                plt.xlabel('Pixel position')
                plt.ylabel('Normal - Inverted')
                plt.tight_layout()
                plt.savefig(f'line_profile_bit{i}.png')
                plt.close()
                
                print(f"  Saved line profile to line_profile_bit{i}.png")
    
    print("\n3. PROPER DECODING TEST:")
    # Try manual decoding with fixed references
    from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns
    
    # Load all patterns
    patterns = []
    for i in range(5):
        normal = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
        inverted = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_inv_left.png"), cv2.IMREAD_GRAYSCALE)
        patterns.append(normal)
        patterns.append(inverted)
    
    # Decode with proper references
    x_coord, x_conf, x_mask = decode_patterns(
        white_left, black_left, patterns,
        num_bits=5, orientation="horizontal"
    )
    
    print(f"\nDecoding results:")
    print(f"Valid pixels: {np.sum(x_mask)} ({100*np.sum(x_mask)/x_mask.size:.1f}%)")
    if np.sum(x_mask) > 0:
        print(f"Coordinate range: {np.min(x_coord[x_mask])} to {np.max(x_coord[x_mask])}")
        print(f"Unique coordinates: {len(np.unique(x_coord[x_mask]))}")
        
        # Create proper visualization
        x_vis = np.zeros_like(x_coord, dtype=np.uint8)
        valid_coords = x_coord[x_mask]
        if len(valid_coords) > 0 and np.max(valid_coords) > np.min(valid_coords):
            x_normalized = (x_coord - np.min(valid_coords)) / (np.max(valid_coords) - np.min(valid_coords))
            x_vis[x_mask] = (x_normalized[x_mask] * 255).astype(np.uint8)
        
        x_color = cv2.applyColorMap(x_vis, cv2.COLORMAP_JET)
        cv2.imwrite('proper_decoded_coords.png', x_color)
        print("Saved visualization to proper_decoded_coords.png")
    
    print("\n4. RECOMMENDATIONS:")
    print("- Check if projector is actually changing patterns")
    print("- Verify camera exposure settings (may be over/underexposed)")
    print("- Consider adjusting projector brightness")
    print("- Check for synchronization issues between projector and camera")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose pattern decoding issues")
    parser.add_argument("image_dir", help="Directory containing captured images")
    args = parser.parse_args()
    
    diagnose_patterns(args.image_dir)
    print("\nDiagnostic complete!")


if __name__ == "__main__":
    main()