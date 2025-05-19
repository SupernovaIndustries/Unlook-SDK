#!/usr/bin/env python3
"""
Test accessing camera through UnlookClient's property.
This mimics what happens in static_scanning_example_fixed.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Minimal imports to test the issue
import unlook.client.scanner
print(f"scanner module: {unlook.client.scanner}")

# Create a mock client to test the camera property
class MockClient:
    def __init__(self):
        self._camera = None
        
    @property
    def camera(self):
        """Lazy-loading of the camera client."""
        if self._camera is None:
            # Use importlib to directly import from camera.py file
            import importlib.util
            import os
            camera_file = os.path.join(os.path.dirname(unlook.client.scanner.__file__), 'camera.py')
            print(f"Looking for camera.py at: {camera_file}")
            print(f"File exists: {os.path.exists(camera_file)}")
            
            spec = importlib.util.spec_from_file_location("camera_module", camera_file)
            camera_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(camera_module)
            print(f"Module loaded: {camera_module}")
            print(f"Has CameraClient: {hasattr(camera_module, 'CameraClient')}")
            
            # Try to instantiate CameraClient
            self._camera = camera_module.CameraClient(self)
        return self._camera

try:
    client = MockClient()
    print("MockClient created")
    
    camera = client.camera
    print(f"Camera accessed successfully: {type(camera)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()