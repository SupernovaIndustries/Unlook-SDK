"""Gesture recognition module for UnLook hand tracking.

Simple and easy-to-use gesture recognition based on hand keypoints.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Common hand gestures that can be recognized."""
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
    WAVE = "wave"  # Detected across multiple frames
    

class GestureRecognizer:
    """Simple gesture recognition from hand keypoints."""
    
    def __init__(self, gesture_threshold: float = 0.8):
        """
        Initialize gesture recognizer.
        
        Args:
            gesture_threshold: Confidence threshold for gesture recognition (0-1)
        """
        self.gesture_threshold = gesture_threshold
        self.gesture_history = []
        self.max_history = 30  # For temporal gestures like waving
        
        # MediaPipe landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        # Finger base indices
        self.THUMB_BASE = 1
        self.INDEX_BASE = 5
        self.MIDDLE_BASE = 9
        self.RING_BASE = 13
        self.PINKY_BASE = 17
        
    def recognize_gesture(self, keypoints: np.ndarray) -> Tuple[GestureType, float]:
        """
        Recognize gesture from hand keypoints.
        
        Args:
            keypoints: Hand landmarks (21x3 array)
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        if keypoints is None or len(keypoints) != 21:
            return GestureType.UNKNOWN, 0.0
        
        # Calculate finger states
        finger_states = self._calculate_finger_states(keypoints)
        
        # Check for specific gestures
        gesture, confidence = self._match_gesture(finger_states, keypoints)
        
        # Store in history for temporal gestures
        self.gesture_history.append({
            'gesture': gesture,
            'confidence': confidence,
            'keypoints': keypoints.copy()
        })
        
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        return gesture, confidence
    
    def _calculate_finger_states(self, keypoints: np.ndarray) -> Dict[str, bool]:
        """
        Calculate if each finger is extended or folded.
        
        Returns:
            Dictionary with finger states (True = extended, False = folded)
        """
        states = {}
        
        # Check if keypoints are normalized (0-1) or pixel coordinates
        max_coord = np.max(np.abs(keypoints[:, :2]))
        is_normalized = max_coord <= 1.0
        
        # Thumb: check if tip is far from base
        thumb_vec = keypoints[self.THUMB_TIP] - keypoints[self.THUMB_BASE]
        thumb_length = np.linalg.norm(thumb_vec[:2])  # Use 2D for stability
        wrist_to_base = keypoints[self.THUMB_BASE] - keypoints[self.WRIST]
        base_length = np.linalg.norm(wrist_to_base[:2])
        
        # Adjust threshold based on coordinate system
        thumb_threshold = 0.7 if not is_normalized else 1.5
        states['thumb'] = thumb_length > base_length * thumb_threshold
        
        # Other fingers: check if tip is higher than base (in y-direction)
        # and far enough from palm center
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        
        for finger, tip_idx, base_idx in [
            ('index', self.INDEX_TIP, self.INDEX_BASE),
            ('middle', self.MIDDLE_TIP, self.MIDDLE_BASE),
            ('ring', self.RING_TIP, self.RING_BASE),
            ('pinky', self.PINKY_TIP, self.PINKY_BASE)
        ]:
            # Check if tip is extended (higher than base in image coordinates)
            tip_y = keypoints[tip_idx][1]
            base_y = keypoints[base_idx][1]
            wrist_y = keypoints[self.WRIST][1]
            
            # In image coordinates, lower y means higher position
            is_extended = tip_y < base_y and tip_y < wrist_y
            
            # Also check distance from palm center
            tip_dist = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
            base_dist = np.linalg.norm(keypoints[base_idx][:2] - palm_center[:2])
            
            # Adjust distance threshold based on coordinate system
            dist_threshold = 1.2 if not is_normalized else 1.5
            is_far = tip_dist > base_dist * dist_threshold
            
            states[finger] = is_extended and is_far
        
        return states
    
    def _match_gesture(self, finger_states: Dict[str, bool], keypoints: np.ndarray) -> Tuple[GestureType, float]:
        """
        Match finger states to known gestures.
        
        Returns:
            Tuple of (gesture_type, confidence)
        """
        # Count extended fingers
        extended_count = sum(finger_states.values())
        
        # Open palm - all fingers extended
        if extended_count == 5:
            return GestureType.OPEN_PALM, 0.95
        
        # Closed fist - no fingers extended
        if extended_count == 0:
            return GestureType.CLOSED_FIST, 0.95
        
        # Pointing - only index extended
        if extended_count == 1 and finger_states['index']:
            return GestureType.POINTING, 0.9
        
        # Peace sign - index and middle extended
        if extended_count == 2 and finger_states['index'] and finger_states['middle']:
            return GestureType.PEACE, 0.9
        
        # Thumbs up - only thumb extended
        if extended_count == 1 and finger_states['thumb']:
            # Check thumb orientation
            thumb_vec = keypoints[self.THUMB_TIP] - keypoints[self.THUMB_BASE]
            if thumb_vec[1] < -0.1:  # Pointing up
                return GestureType.THUMBS_UP, 0.85
            elif thumb_vec[1] > 0.1:  # Pointing down
                return GestureType.THUMBS_DOWN, 0.85
        
        # OK sign - thumb and index forming circle
        if self._is_ok_gesture(keypoints):
            return GestureType.OK, 0.85
        
        # Rock sign - index and pinky extended
        if extended_count == 2 and finger_states['index'] and finger_states['pinky']:
            return GestureType.ROCK, 0.85
        
        # Pinch - thumb and index close together
        if self._is_pinch_gesture(keypoints):
            return GestureType.PINCH, 0.85
        
        # Check for wave gesture (temporal)
        if self._is_waving():
            return GestureType.WAVE, 0.8
        
        return GestureType.UNKNOWN, 0.5
    
    def _is_ok_gesture(self, keypoints: np.ndarray) -> bool:
        """Check if making OK gesture (thumb and index forming circle)."""
        thumb_tip = keypoints[self.THUMB_TIP]
        index_tip = keypoints[self.INDEX_TIP]
        
        # Check if keypoints are normalized
        max_coord = np.max(np.abs(keypoints[:, :2]))
        is_normalized = max_coord <= 1.0
        
        # Check if tips are close
        distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        
        # Adjust threshold based on coordinate system
        distance_threshold = 0.05 if is_normalized else 50  # pixels
        
        # Check if other fingers are extended
        middle_extended = keypoints[self.MIDDLE_TIP][1] < keypoints[self.MIDDLE_BASE][1]
        ring_extended = keypoints[self.RING_TIP][1] < keypoints[self.RING_BASE][1]
        pinky_extended = keypoints[self.PINKY_TIP][1] < keypoints[self.PINKY_BASE][1]
        
        return distance < distance_threshold and middle_extended and ring_extended and pinky_extended
    
    def _is_pinch_gesture(self, keypoints: np.ndarray) -> bool:
        """Check if making pinch gesture."""
        thumb_tip = keypoints[self.THUMB_TIP]
        index_tip = keypoints[self.INDEX_TIP]
        
        # Check if keypoints are normalized
        max_coord = np.max(np.abs(keypoints[:, :2]))
        is_normalized = max_coord <= 1.0
        
        # Check if tips are very close
        distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        
        # Adjust threshold based on coordinate system
        distance_threshold = 0.03 if is_normalized else 30  # pixels
        
        return distance < distance_threshold
    
    def _is_waving(self) -> bool:
        """Detect waving gesture by analyzing motion history."""
        if len(self.gesture_history) < 10:
            return False
        
        # Check for open palm in recent frames
        recent_gestures = [h['gesture'] for h in self.gesture_history[-10:]]
        open_palm_count = sum(1 for g in recent_gestures if g == GestureType.OPEN_PALM)
        
        if open_palm_count < 5:
            return False
        
        # Check for horizontal motion of wrist
        wrist_positions = [h['keypoints'][self.WRIST] for h in self.gesture_history[-10:]]
        x_positions = [pos[0] for pos in wrist_positions]
        
        # Calculate motion range
        x_range = max(x_positions) - min(x_positions)
        
        # Check for back-and-forth motion
        direction_changes = 0
        for i in range(1, len(x_positions) - 1):
            if (x_positions[i] - x_positions[i-1]) * (x_positions[i+1] - x_positions[i]) < 0:
                direction_changes += 1
        
        return x_range > 0.1 and direction_changes >= 2
    
    def recognize_gestures_3d(self, keypoints_3d: np.ndarray) -> Tuple[GestureType, float]:
        """
        Recognize gestures from 3D keypoints.
        
        Args:
            keypoints_3d: 3D hand landmarks (21x3 array)
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        # For now, use 2D recognition on normalized 3D points
        # More sophisticated 3D gesture recognition can be added later
        normalized_2d = keypoints_3d[:, :2] / np.max(np.abs(keypoints_3d[:, :2])) if np.max(np.abs(keypoints_3d[:, :2])) > 0 else keypoints_3d[:, :2]
        
        # Add dummy z-coordinates
        keypoints_2d = np.column_stack([normalized_2d, np.zeros(21)])
        
        return self.recognize_gesture(keypoints_2d)
    
    def recognize_gesture_2d(self, keypoints_2d: np.ndarray, image_width: int, image_height: int) -> Tuple[GestureType, float]:
        """
        Recognize gestures from 2D normalized keypoints (MediaPipe format).
        
        Args:
            keypoints_2d: 2D hand landmarks in normalized coordinates (21x3 array)
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = keypoints_2d.copy()
        pixel_coords[:, 0] *= image_width
        pixel_coords[:, 1] *= image_height
        
        # Use the pixel coordinates for gesture recognition
        return self.recognize_gesture(pixel_coords)
    
    def get_gesture_name(self, gesture_type: GestureType) -> str:
        """Get human-readable name for gesture."""
        return gesture_type.value.replace('_', ' ').title()
    
    def clear_history(self):
        """Clear gesture history."""
        self.gesture_history = []