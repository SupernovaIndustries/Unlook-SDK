#!/usr/bin/env python3
"""
Quick script to get the actual camera IDs from the server
"""
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unlook.client.scanner.scanner import UnlookClient

def get_camera_ids():
    print("Connecting to scanner to get camera IDs...")
    
    # Create client
    client = UnlookClient("GetCameraIDs", auto_discover=True)
    time.sleep(3)
    
    # Get scanners
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("ERROR: No scanners found!")
        return
    
    # Connect
    scanner = scanners[0]
    if not client.connect(scanner):
        print("ERROR: Failed to connect!")
        return
    
    print(f"Connected to {scanner.name}")
    
    try:
        # Get camera list
        cameras = client.camera.get_cameras()
        print(f"\nAvailable cameras: {len(cameras)}")
        for i, cam in enumerate(cameras):
            print(f"  Camera {i}: {cam}")
            
        return cameras
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    finally:
        client.disconnect()

if __name__ == "__main__":
    cameras = get_camera_ids()