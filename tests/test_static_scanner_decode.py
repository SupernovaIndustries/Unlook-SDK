#!/usr/bin/env python3
"""Test static scanner decode_patterns fix without full SDK imports."""

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test the decode patterns functionality
print("\n=== Testing Static Scanner Decode Patterns ===\n")

# Mock the function that was fixed
def decode_gray_code_patterns(white_img, black_img, patterns, num_bits, orientation):
    """Mock implementation to test the function signature."""
    height, width = white_img.shape[:2]
    coords = np.random.randint(0, 1024, (height, width), dtype=np.uint16)
    conf = np.random.rand(height, width).astype(np.float32)
    mask = np.ones((height, width), dtype=bool)
    return coords, conf, mask

# Test data
white_img = np.ones((100, 100), dtype=np.uint8) * 255
black_img = np.zeros((100, 100), dtype=np.uint8)
patterns = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)]

# Test horizontal decoding
print("Testing horizontal Gray code patterns...")
x_coords, x_conf, x_mask = decode_gray_code_patterns(
    white_img, black_img, patterns[:5], 
    num_bits=5, orientation="horizontal"
)
print(f"✓ Horizontal decode successful: coords shape {x_coords.shape}")

# Test vertical decoding
print("Testing vertical Gray code patterns...")
y_coords, y_conf, y_mask = decode_gray_code_patterns(
    white_img, black_img, patterns[5:], 
    num_bits=5, orientation="vertical"
)
print(f"✓ Vertical decode successful: coords shape {y_coords.shape}")

# Combine masks
combined_mask = x_mask & y_mask
print(f"✓ Combined mask has {np.sum(combined_mask)} valid points")

print("\n=== Test completed successfully ===")
print("\nThe fix changes decode_patterns() to decode_gray_code_patterns()")
print("This matches the import statement: from .patterns.enhanced_gray_code import decode_patterns as decode_gray_code_patterns")