#!/usr/bin/env python3
"""
Check dependencies for pattern generators
"""

import sys
import importlib

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        mod = importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        
        # Check for specific submodules
        if module_name == 'cv2':
            # Check for aruco
            try:
                aruco = getattr(mod, 'aruco', None)
                if aruco:
                    print(f"  ✓ cv2.aruco available")
                else:
                    print(f"  × cv2.aruco NOT available (might need opencv-contrib-python)")
            except Exception as e:
                print(f"  × cv2.aruco error: {e}")
                
        return True
    except ImportError as e:
        print(f"× {module_name} import failed: {e}")
        return False

# Check required dependencies
print("Checking Pattern Generator Dependencies")
print("=" * 40)

dependencies = [
    'numpy',
    'cv2',
    'scipy',
    'dataclasses',
    'logging'
]

all_ok = True
for dep in dependencies:
    if not check_import(dep):
        all_ok = False

print("\n" + "=" * 40)
if all_ok:
    print("✓ All basic dependencies OK")
else:
    print("× Some dependencies missing")
    
# Check for opencv extras
print("\nChecking OpenCV extras...")
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check for contrib modules
    if hasattr(cv2, 'aruco'):
        print("✓ opencv-contrib-python is installed (ArUco available)")
    else:
        print("× opencv-contrib-python might not be installed")
        print("  Install with: pip install opencv-contrib-python")
except:
    pass

print("\nNOTE: If ArUco is missing, install opencv-contrib-python:")
print("  pip install opencv-contrib-python")