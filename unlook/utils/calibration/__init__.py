"""
Camera Calibration Utilities for Unlook SDK

This package contains utilities for camera calibration in the Unlook SDK.
These tools help with creating, loading, and verifying camera calibration
files, which are essential for accurate 3D reconstruction.

Key tools:
- headless_calibration.py: Command-line tool for calibrating stereo cameras
- load_calibration_images.py: Tool for creating calibration from existing images
- check_calibration.py: Tool for verifying calibration files
"""

# Import calibration tools
from .headless_calibration import main as run_headless_calibration
from .load_calibration_images import main as run_load_calibration_images
from .check_calibration import main as run_check_calibration