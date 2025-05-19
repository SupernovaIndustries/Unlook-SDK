#!/usr/bin/env python3
"""
Super simple UnLook demo - Everything in under 20 lines.
Shows how easy it is to use UnLook for computer vision.
"""

from unlook.simple import UnlookSimple

# Connect to UnLook
unlook = UnlookSimple()
unlook.connect()

# 1. Capture an image
image = unlook.capture()
print(f"Got image: {image.shape}")

# 2. Do a 3D scan  
point_cloud = unlook.scan_3d()
print(f"Got {len(point_cloud.points) if point_cloud else 0} 3D points")

# 3. Save the results
unlook.save_scan("demo_scan.ply")

# Done!
unlook.disconnect()