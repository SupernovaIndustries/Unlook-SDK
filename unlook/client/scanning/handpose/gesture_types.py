"""
Gesture type definitions and constants for hand tracking.

This module defines the gesture types and related constants used
throughout the handpose tracking system.
"""

from enum import Enum

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Finger base indices
THUMB_BASE = THUMB_CMC
INDEX_BASE = INDEX_FINGER_MCP
MIDDLE_BASE = MIDDLE_FINGER_MCP
RING_BASE = RING_FINGER_MCP
PINKY_BASE = PINKY_MCP

# Landmark names for debugging
LANDMARK_NAMES = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]


class GestureType(Enum):
    """
    Enumeration of supported hand gestures.
    
    Each gesture has a unique string identifier used for
    recognition and logging.
    """
    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OK = "ok"
    ROCK = "rock"
    PINCH = "pinch"
    WAVE = "wave"
    
    @classmethod
    def from_string(cls, value: str) -> 'GestureType':
        """Convert string to GestureType, returns UNKNOWN if not found."""
        for gesture in cls:
            if gesture.value == value.lower():
                return gesture
        return cls.UNKNOWN
    
    def to_display_name(self) -> str:
        """Get human-readable display name."""
        return self.value.replace('_', ' ').title()


# Gesture configuration
GESTURE_CONFIG = {
    'threshold': 0.7,              # Default confidence threshold
    'stability_frames': 2,         # Frames gesture must be stable
    'history_size': 30,           # Max gesture history for temporal analysis
    'point_history_size': 16,     # Points to track for motion gestures
    'wave_min_changes': 2,        # Min direction changes for wave
    'wave_min_range': 0.1,        # Min x-range for wave detection
}