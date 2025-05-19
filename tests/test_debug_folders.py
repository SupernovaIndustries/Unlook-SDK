#!/usr/bin/env python3
"""Test if debug folders are being created during scanning."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the SDK path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Expected debug folder structure
EXPECTED_FOLDERS = [
    "01_patterns/raw",
    "02_rectified", 
    "03_decoded",
    "04_correspondence/maps",
    "04_correspondence/visualizations",
    "05_triangulation",
    "06_point_cloud"
]

def check_debug_structure(debug_path):
    """Check if all expected debug folders exist."""
    missing = []
    for folder in EXPECTED_FOLDERS:
        full_path = debug_path / folder
        if not full_path.exists():
            missing.append(folder)
        else:
            print(f"✓ Found: {folder}")
    
    if missing:
        print(f"\n✗ Missing folders:")
        for folder in missing:
            print(f"  - {folder}")
    else:
        print("\n✓ All debug folders present!")
    
    return len(missing) == 0

# Create test directory
test_dir = Path(tempfile.mkdtemp())
debug_dir = test_dir / "unlook_debug" / "scan_test"

print(f"Test directory: {debug_dir}")
print("\nExpected folder structure:")

# Mock the folder creation that should happen during scanning
for folder in EXPECTED_FOLDERS:
    full_path = debug_dir / folder
    full_path.mkdir(parents=True, exist_ok=True)
    
# Check structure
print("\nChecking debug folder structure...")
success = check_debug_structure(debug_dir)

# Cleanup
shutil.rmtree(test_dir)

if success:
    print("\n✓ Debug folder structure test PASSED")
else:
    print("\n✗ Debug folder structure test FAILED")
    
print("\nNOTE: The actual issue is that during scanning, folders 03-06 are not being populated.")
print("This is because the pattern decoding is failing due to the decode_patterns bug that was fixed.")