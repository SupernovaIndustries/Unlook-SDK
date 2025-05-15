#!/usr/bin/env python3
"""
Minimal test to verify camera import fix using importlib.
"""

import importlib.util
import os

# Test the exact approach used in scanner.py
try:
    # Get the path to camera.py
    camera_file = os.path.join(os.path.dirname(__file__), 'unlook', 'client', 'camera.py')
    print(f"Looking for camera.py at: {camera_file}")
    print(f"File exists: {os.path.exists(camera_file)}")
    
    # Load the module directly
    spec = importlib.util.spec_from_file_location("camera_module", camera_file)
    camera_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_module)
    
    print("Module loaded successfully!")
    print(f"Module attributes: {dir(camera_module)}")
    
    # Check if CameraClient exists
    if hasattr(camera_module, 'CameraClient'):
        print("CameraClient found in module!")
    else:
        print("ERROR: CameraClient not found in module")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()