#!/usr/bin/env python3
"""Direct test of decode patterns fix, no SDK imports."""

import numpy as np
import sys
sys.path.insert(0, '/mnt/g/Supernova/Prototipi/UnLook/Software/Unlook-SDK')

# Test direct import
try:
    from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns, gray_to_binary
    print("✓ Successfully imported decode_patterns and gray_to_binary")
    
    # Test gray_to_binary function
    result = gray_to_binary(5, 4)
    print(f"✓ gray_to_binary(5, 4) = {result}")
    
    # Test decode_patterns function  
    white_img = np.ones((100, 100), dtype=np.uint8) * 255
    black_img = np.zeros((100, 100), dtype=np.uint8)
    patterns = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)]
    
    print("Testing decode_patterns function...")
    coords, conf, mask = decode_patterns(
        white_img, black_img, patterns, 
        num_bits=5, orientation="horizontal"
    )
    
    print(f"✓ Function executed successfully")
    print(f"  - Coordinate map shape: {coords.shape}")
    print(f"  - Confidence map shape: {conf.shape}")  
    print(f"  - Mask shape: {mask.shape}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Error during execution: {e}")
    import traceback
    traceback.print_exc()