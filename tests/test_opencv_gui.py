#!/usr/bin/env python3
"""Test OpenCV GUI capabilities."""

import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV build info:")
print(cv2.getBuildInformation())

# Check if highgui module is available
try:
    # Try to create a simple window
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.destroyWindow("test")
    print("\n✓ GUI support is available")
except Exception as e:
    print(f"\n✗ GUI support not available: {e}")

# Try to create and display a simple image
try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Test Window", img)
    print("If you see a black window, press any key to close it")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✓ Display test successful")
except Exception as e:
    print(f"✗ Display test failed: {e}")
    print("\nYou need to reinstall OpenCV with GUI support")
    print("Run: pip uninstall opencv-python && pip install opencv-python")