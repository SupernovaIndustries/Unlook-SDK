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
    
    # Create diagnostics directory
    diag_dir = Path(image_dir) / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    
    print(f"\nDIAGNOSTIC ANALYSIS")
    print("="*50)
    print(f"Saving diagnostics to: {diag_dir}")
    
    # Load reference images
    black_left = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_GRAYSCALE)
    white_left = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_GRAYSCALE)
    ambient_left = cv2.imread(str(pattern_dir / "ambient_left.png"), cv2.IMREAD_GRAYSCALE)
    
    # Key insight: The white reference is DARKER than the black reference!
    print("\n1. REFERENCE IMAGE ISSUE:")
    print(f"Black mean: {np.mean(black_left):.1f}")
    print(f"White mean: {np.mean(white_left):.1f}")
    print(f"Ambient mean: {np.mean(ambient_left):.1f}")
    print(f"White - Black: {np.mean(white_left) - np.mean(black_left):.1f}")
    print("WARNING: White reference is DARKER than black reference!")
    
    # This suggests the projector isn't properly displaying patterns
    # or there's strong ambient light interference
    
    # Load color version to check purple cast
    black_color = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_COLOR)
    white_color = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_COLOR)
    
    print("\n2. COLOR CAST ANALYSIS:")
    b, g, r = cv2.split(black_color)
    print(f"Black - R:{np.mean(r):.1f} G:{np.mean(g):.1f} B:{np.mean(b):.1f}")
    b, g, r = cv2.split(white_color)
    print(f"White - R:{np.mean(r):.1f} G:{np.mean(g):.1f} B:{np.mean(b):.1f}")
    print("Strong purple/blue cast detected - typical of ambient lighting")
    
    # Analyze pattern visibility
    print("\n3. PATTERN VISIBILITY TEST:")
    
    # Check first bit pattern (widest stripes)
    bit0_normal = cv2.imread(str(pattern_dir / "gray_h_bit00_left.png"), cv2.IMREAD_GRAYSCALE)
    bit0_inv = cv2.imread(str(pattern_dir / "gray_h_bit00_inv_left.png"), cv2.IMREAD_GRAYSCALE)
    
    # Calculate actual difference
    diff = bit0_normal.astype(float) - bit0_inv.astype(float)
    
    print(f"Bit 0 difference stats:")
    print(f"  Min: {np.min(diff):.1f}, Max: {np.max(diff):.1f}")
    print(f"  Mean absolute: {np.mean(np.abs(diff)):.1f}")
    print(f"  Standard deviation: {np.std(diff):.1f}")
    
    # Check if we can see stripes at all
    mid_row = bit0_normal.shape[0] // 2
    normal_line = bit0_normal[mid_row, :]
    inv_line = bit0_inv[mid_row, :]
    
    # Calculate transitions (where pattern changes)
    normal_gradient = np.abs(np.diff(normal_line))
    transitions = np.where(normal_gradient > 20)[0]
    
    print(f"\nStripe detection:")
    print(f"  Transitions found: {len(transitions)}")
    print(f"  Expected for 5-bit pattern: ~32")
    
    if len(transitions) < 10:
        print("\nCRITICAL ISSUE: No clear stripe patterns detected!")
        print("Possible causes:")
        print("- Projector not displaying patterns correctly")
        print("- Camera exposure too high/low")
        print("- Focus issues")
        print("- Synchronization problems")
    
    # Save diagnostic images
    print("\n4. SAVING DIAGNOSTIC IMAGES:")
    
    # Enhanced difference image
    diff_enhanced = np.abs(diff)
    diff_enhanced = (diff_enhanced / np.max(diff_enhanced) * 255).astype(np.uint8)
    cv2.imwrite(str(diag_dir / 'diagnostic_difference.png'), diff_enhanced)
    print(f"Saved: {diag_dir / 'diagnostic_difference.png'}")
    
    # Create side-by-side comparison
    comparison = np.hstack([bit0_normal, bit0_inv, diff_enhanced])
    cv2.imwrite(str(diag_dir / 'diagnostic_comparison.png'), comparison)
    print(f"Saved: {diag_dir / 'diagnostic_comparison.png'}")
    
    # Profile plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(normal_line, 'b-', label='Normal')
    plt.plot(inv_line, 'r-', label='Inverted')
    plt.legend()
    plt.title('Bit 0 - Middle Row Profile')
    plt.ylabel('Intensity')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(diff[mid_row, :], 'g-')
    plt.title('Difference (Normal - Inverted)')
    plt.ylabel('Difference')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(normal_gradient, 'k-')
    plt.title('Gradient (Edge Detection)')
    plt.xlabel('Pixel Position')
    plt.ylabel('Gradient')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(diag_dir / 'diagnostic_profiles.png'), dpi=150)
    plt.close()
    print(f"Saved: {diag_dir / 'diagnostic_profiles.png'}")
    
    # Test with simple threshold
    print("\n5. SIMPLE DECODING TEST:")
    
    # Try basic thresholding
    threshold = np.mean(bit0_normal)
    binary = (bit0_normal > threshold).astype(np.uint8) * 255
    cv2.imwrite(str(diag_dir / 'diagnostic_threshold.png'), binary)
    print(f"Saved: {diag_dir / 'diagnostic_threshold.png'}")
    
    # Count unique values in each pattern
    print("\nPattern intensity distributions:")
    for i in range(3):
        if i < 5:
            normal = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
            if normal is not None:
                unique_vals = len(np.unique(normal))
                print(f"Bit {i}: {unique_vals} unique intensity values")
                hist, _ = np.histogram(normal.ravel(), 256, [0, 256])
                peaks = np.where(hist > 1000)[0]
                print(f"  Major peaks at: {peaks}")
    
    print("\n6. RECOMMENDATIONS:")
    print("Based on the analysis, the main issues are:")
    print("1. White/black references are inverted or corrupted")
    print("2. Very low contrast between patterns")
    print("3. Strong ambient light interference (purple cast)")
    print("\nTo fix:")
    print("- Check projector brightness and contrast settings")
    print("- Reduce ambient lighting")
    print("- Verify projector-camera synchronization")
    print("- Consider using the enhanced processor at maximum level")
    print("- May need to adjust camera exposure settings")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose pattern decoding issues")
    parser.add_argument("image_dir", help="Directory containing captured images")
    args = parser.parse_args()
    
    diagnose_patterns(args.image_dir)
    print("\nDiagnostic complete! Check the generated images.")


if __name__ == "__main__":
    main()