#!/usr/bin/env python3
"""
3D scan in one line with UnLook.
As simple as taking a photo, but in 3D.
"""

from unlook.simple import quick_scan

# One line 3D scan!
quick_scan("my_object.ply", quality="balanced")