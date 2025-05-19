#!/usr/bin/env python3
"""
Hello UnLook - Simplest possible example to test connection.
This is like the "blink LED" example for Arduino.
"""

from unlook import UnlookClient

# Connect to scanner (auto-discovers first available)
client = UnlookClient(auto_discover=True)
print("Waiting for scanner...")

# Get first scanner
scanners = client.get_discovered_scanners()
if scanners:
    print(f"Found: {scanners[0].name}")
    client.connect(scanners[0])
    
    # Get camera list
    cameras = client.camera.get_cameras()
    print(f"Cameras: {len(cameras)}")
    
    # Disconnect
    client.disconnect()
    print("Done!")
else:
    print("No scanner found. Is it connected?")