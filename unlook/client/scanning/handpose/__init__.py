"""Hand pose detection and tracking module for UnLook SDK.

This module provides hand pose detection, tracking, and gesture recognition
functionality for the UnLook scanner. It uses MediaPipe for 2D hand detection
and adds stereo triangulation and gesture recognition capabilities.

Classes:
    HandDetector: 2D hand detection using MediaPipe
    HandTracker: 3D hand tracking using stereo cameras
    GestureRecognizer: Gesture recognition from hand keypoints
    LEDController: LED control with automatic activation
    
Modules:
    gesture_types: Gesture enumeration and constants
    finger_detection: Finger state detection algorithms
    gesture_detectors: Individual gesture detection functions
"""

# Core classes
from .HandDetector import HandDetector
from .HandTracker import HandTracker
from .GestureRecognizer import GestureRecognizer
from .LEDController import LEDController

# Types and constants
from .gesture_types import GestureType, GESTURE_CONFIG

# Detection functions (optional imports for advanced users)
from .finger_detection import calculate_finger_states, get_partial_extensions
from .gesture_detectors import (
    detect_fist, detect_open_palm, detect_pointing, detect_peace,
    detect_thumbs_up_down, detect_ok, detect_rock, detect_pinch, detect_wave
)

__all__ = [
    # Core classes
    'HandDetector',
    'HandTracker',
    'GestureRecognizer',
    'LEDController',
    # Types
    'GestureType',
    'GESTURE_CONFIG',
    # Detection functions
    'calculate_finger_states',
    'get_partial_extensions',
    'detect_fist',
    'detect_open_palm',
    'detect_pointing',
    'detect_peace',
    'detect_thumbs_up_down',
    'detect_ok',
    'detect_rock',
    'detect_pinch',
    'detect_wave',
]