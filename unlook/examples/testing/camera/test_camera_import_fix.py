#!/usr/bin/env python3
"""
Test if the camera import fix using importlib works correctly.
"""

import sys
import logging
from unlook.client.scanner import UnlookClient

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Create a client instance
    client = UnlookClient(client_name="TestClient", auto_discover=False)
    print("UnlookClient created successfully")
    
    # Try to access the camera property
    print("Attempting to access camera property...")
    camera = client.camera
    print(f"Camera access successful! Camera type: {type(camera)}")
    print(f"Camera module path: {camera.__module__}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()