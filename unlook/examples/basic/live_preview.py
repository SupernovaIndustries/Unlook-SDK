#!/usr/bin/env python3
"""
Live preview from UnLook camera.
Press ESC to quit.
"""

import cv2
from unlook import UnlookClient

# Connect
client = UnlookClient(auto_discover=True)
scanner = client.get_discovered_scanners()[0]
client.connect(scanner)

# Get camera
camera_id = client.camera.get_cameras()[0]['id']

# Stream with simple callback
def show_frame(frame, metadata):
    cv2.imshow("UnLook Live", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        return False
    return True

# Start streaming
print("Streaming... Press ESC to quit")
client.stream.start(camera_id, show_frame)

# Cleanup
cv2.destroyAllWindows()
client.disconnect()