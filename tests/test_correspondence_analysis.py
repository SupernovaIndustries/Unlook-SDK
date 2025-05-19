#!/usr/bin/env python3
"""Analyze correspondence finding issues in the scanning pipeline."""

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulate the correspondence finding issue
def simulate_correspondence_issue():
    """Simulate what happens during scanning when we get zero correspondences."""
    
    # Simulate decoded coordinates
    width, height = 1920, 1080
    
    # Create sample coordinate maps (simulating Gray code decoding results)
    x_coord_left = np.random.randint(0, 1024, (height, width), dtype=np.uint16)
    y_coord_left = np.random.randint(0, 768, (height, width), dtype=np.uint16)
    x_coord_right = np.random.randint(0, 1024, (height, width), dtype=np.uint16)
    y_coord_right = np.random.randint(0, 768, (height, width), dtype=np.uint16)
    
    # Create masks (some pixels are invalid)
    mask_left = np.random.rand(height, width) > 0.3  # 70% valid pixels
    mask_right = np.random.rand(height, width) > 0.3
    
    # Simulate the issue: Due to ambient light, decoded coordinates are noisy
    # In real scenarios, the same projector pixel might decode to very different values
    noise_level = 100  # High noise simulates ambient light interference
    x_coord_left = x_coord_left.astype(np.int32) + np.random.randint(-noise_level, noise_level, (height, width))
    y_coord_left = y_coord_left.astype(np.int32) + np.random.randint(-noise_level, noise_level, (height, width))
    x_coord_right = x_coord_right.astype(np.int32) + np.random.randint(-noise_level, noise_level, (height, width))
    y_coord_right = y_coord_right.astype(np.int32) + np.random.randint(-noise_level, noise_level, (height, width))
    
    # Clip to valid range
    x_coord_left = np.clip(x_coord_left, 0, 1023)
    y_coord_left = np.clip(y_coord_left, 0, 767)
    x_coord_right = np.clip(x_coord_right, 0, 1023)
    y_coord_right = np.clip(y_coord_right, 0, 767)
    
    # Simulate correspondence finding
    left_proj_to_pixel = {}
    right_proj_to_pixel = {}
    
    # Build lookup for left image
    valid_left = np.where(mask_left)
    for i in range(len(valid_left[0])):
        y, x = valid_left[0][i], valid_left[1][i]
        proj_x = int(x_coord_left[y, x])
        proj_y = int(y_coord_left[y, x])
        proj_key = (proj_x, proj_y)
        
        if proj_key not in left_proj_to_pixel:
            left_proj_to_pixel[proj_key] = []
        left_proj_to_pixel[proj_key].append((x, y))
    
    # Build lookup for right image
    valid_right = np.where(mask_right)
    for i in range(len(valid_right[0])):
        y, x = valid_right[0][i], valid_right[1][i]
        proj_x = int(x_coord_right[y, x])
        proj_y = int(y_coord_right[y, x])
        proj_key = (proj_x, proj_y)
        
        if proj_key not in right_proj_to_pixel:
            right_proj_to_pixel[proj_key] = []
        right_proj_to_pixel[proj_key].append((x, y))
    
    # Find common projector coordinates
    common_proj_coords = set(left_proj_to_pixel.keys()) & set(right_proj_to_pixel.keys())
    
    print(f"\nCorrespondence Analysis:")
    print(f"Valid pixels left: {len(valid_left[0])}")
    print(f"Valid pixels right: {len(valid_right[0])}")
    print(f"Unique projector coords left: {len(left_proj_to_pixel)}")
    print(f"Unique projector coords right: {len(right_proj_to_pixel)}")
    print(f"Common projector coordinates: {len(common_proj_coords)}")
    print(f"Common ratio: {len(common_proj_coords) / max(len(left_proj_to_pixel), len(right_proj_to_pixel)) * 100:.1f}%")
    
    # Analyze why correspondences fail
    if len(common_proj_coords) < 100:
        print("\n⚠️ ISSUE: Very few common projector coordinates found!")
        print("This happens when:")
        print("1. Gray code decoding is noisy due to ambient light")
        print("2. The same projector pixel decodes to different values in left/right")
        print("3. Pattern contrast is too low")
        
        print("\nPossible solutions:")
        print("1. Improve thresholding in Gray code decoding")
        print("2. Use more robust pattern types (phase shift + gray code)")
        print("3. Better ambient light handling")
        print("4. Hardware upgrade to VCSEL for better contrast")

# Run the analysis
if __name__ == "__main__":
    simulate_correspondence_issue()
    
    print("\n=== Actual Issue Analysis ===")
    print("Based on session notes, the real problem is:")
    print("- Dynamic range < 20 (very low)")
    print("- Zero correspondences found")
    print("- Ambient light interference (purple/blue cast)")
    print("- Pattern decoding failing completely")
    
    print("\nThe decode_patterns fix addresses the immediate bug,")
    print("but the root cause is poor pattern visibility in ambient light.")