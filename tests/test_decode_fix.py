#!/usr/bin/env python3
"""Simple test for decode_patterns fix in static scanner."""

import sys
import os
import numpy as np

# Add the SDK path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the import and function call
try:
    from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns as decode_gray_code_patterns
    print("✓ Successfully imported decode_gray_code_patterns")
    
    # Test basic functionality
    white_img = np.ones((100, 100), dtype=np.uint8) * 255
    black_img = np.zeros((100, 100), dtype=np.uint8)
    patterns = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)]
    
    print("Testing decode_gray_code_patterns function...")
    coords, conf, mask = decode_gray_code_patterns(
        white_img, black_img, patterns, 
        num_bits=5, orientation="horizontal"
    )
    
    print(f"✓ Function executed successfully")
    print(f"  - Coordinate map shape: {coords.shape}")
    print(f"  - Confidence map shape: {conf.shape}")  
    print(f"  - Mask shape: {mask.shape}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error during execution: {e}")
    import traceback
    traceback.print_exc()