#!/usr/bin/env python3
"""
Capture a single image from UnLook camera.
Simple as using a webcam.
"""

import cv2
from unlook import UnlookClient

# Connect
client = UnlookClient(auto_discover=True)
scanner = client.get_discovered_scanners()[0]
client.connect(scanner)

# Get first camera
cameras = client.camera.get_cameras()
camera_id = cameras[0]['id']

# Capture image
print("Capturing image...")
image = client.camera.capture(camera_id)

# Save it
cv2.imwrite("unlook_capture.jpg", image)
print("Saved as unlook_capture.jpg")

# Disconnect
client.disconnect()