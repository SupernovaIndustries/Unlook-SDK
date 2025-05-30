#!/usr/bin/env python3
"""
Quick test script for modular scanner components.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing modular scanner imports...")

try:
    from unlook.client.scanning import PatternManager
    print("✅ PatternManager imported successfully")
    
    # Test pattern creation
    pm = PatternManager()
    patterns = pm.create_gray_code_patterns(num_bits=8)
    print(f"✅ Created {len(patterns)} Gray code patterns")
    
except Exception as e:
    print(f"❌ PatternManager error: {e}")
    import traceback
    traceback.print_exc()

try:
    from unlook.client.scanning import CaptureModule
    print("✅ CaptureModule imported successfully")
except Exception as e:
    print(f"❌ CaptureModule error: {e}")

try:
    from unlook.client.scanning import ReconstructionModule
    print("✅ ReconstructionModule imported successfully")
except Exception as e:
    print(f"❌ ReconstructionModule error: {e}")

print("\nAll tests completed!")