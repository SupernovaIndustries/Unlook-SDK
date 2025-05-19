#!/usr/bin/env python3
"""
Perform a simple 3D scan with minimal configuration.
Everything is automatic - just like taking a photo.
"""

from unlook import UnlookClient
from unlook.client.scanner3d import create_scanner

# Connect
client = UnlookClient(auto_discover=True)
scanner = client.get_discovered_scanners()[0]
client.connect(scanner)

# Create scanner with automatic settings
scanner3d = create_scanner(client, quality="fast")

# Perform scan (everything automatic)
print("Scanning...")
result = scanner3d.scan()

# Save point cloud
if result.point_cloud:
    scanner3d.save_point_cloud("my_object.ply")
    print(f"Saved {result.num_points} points to my_object.ply")
else:
    print("No points captured")

# Cleanup
client.disconnect()