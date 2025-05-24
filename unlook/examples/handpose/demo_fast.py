#!/usr/bin/env python3
"""
Quick demo launcher for enhanced_gesture_demo with optimal settings.
This script runs the demo with settings optimized for speed and gesture recognition.
"""

import sys
import os

# Run enhanced_gesture_demo with optimal settings for performance
os.system(f'"{sys.executable}" enhanced_gesture_demo.py --downsample 1 --no-yolo --ip 192.168.1.92')