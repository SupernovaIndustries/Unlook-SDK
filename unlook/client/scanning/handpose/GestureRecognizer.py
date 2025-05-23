"""Gesture recognition module for UnLook hand tracking.

This module provides rule-based gesture recognition from hand keypoints
without dependencies on ML models. It uses geometric and heuristic methods
to identify common hand gestures.

The recognition logic is split across multiple modules for maintainability:
- gesture_types.py: Gesture enumeration and constants
- finger_detection.py: Finger state detection algorithms
- gesture_detectors.py: Individual gesture detection functions
"""

import logging
import time
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np

from .gesture_types import GestureType, GESTURE_CONFIG
from .finger_detection import calculate_finger_states, get_partial_extensions
from .gesture_detectors import (
    detect_fist, detect_open_palm, detect_pointing, detect_peace,
    detect_thumbs_up_down, detect_ok, detect_rock, detect_pinch, detect_wave
)

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """
    Rule-based gesture recognition from hand keypoints.
    
    This class coordinates the gesture recognition process using
    detection functions from the gesture_detectors module.
    
    Attributes:
        gesture_threshold: Confidence threshold for recognition
        gesture_history: Recent gesture history for temporal analysis
        point_history: Hand movement history for motion gestures
    """
    
    def __init__(self, gesture_threshold: float = GESTURE_CONFIG['threshold']):
        """
        Initialize gesture recognizer.
        
        Args:
            gesture_threshold: Confidence threshold for gesture recognition (0-1)
        """
        self.gesture_threshold = gesture_threshold
        self.gesture_history = []
        self.max_history = GESTURE_CONFIG['history_size']
        
        # Point history for tracking movements
        self.point_history = deque(maxlen=GESTURE_CONFIG['point_history_size'])
        self.point_timestamps = deque(maxlen=GESTURE_CONFIG['point_history_size'])
        
        # Stability tracking to prevent flickering
        self.prev_gesture = GestureType.UNKNOWN
        self.prev_confidence = 0.0
        self.gesture_stability_count = 0
        self.min_stability_count = GESTURE_CONFIG['stability_frames']
        
        logger.info(f"GestureRecognizer initialized with threshold {gesture_threshold}")
        
    def recognize_gesture(self, 
                         keypoints: np.ndarray, 
                         handedness: str = "Unknown") -> Tuple[GestureType, float]:
        """
        Recognize gesture from hand keypoints using rule-based detection.
        
        Process:
        1. Validate input keypoints
        2. Update movement history
        3. Calculate finger states
        4. Match to known gestures
        5. Apply stability filtering
        
        Args:
            keypoints: Hand landmarks (21x3 array) in normalized or pixel coordinates
            handedness: Hand type ("Left", "Right", or "Unknown")
            
        Returns:
            Tuple[GestureType, float]: (detected_gesture, confidence_score)
            
        Example:
            >>> gesture, confidence = recognizer.recognize_gesture(landmarks, "Right")
            >>> if confidence > 0.7:
            ...     print(f"Detected: {gesture.value}")
        """
        # Validate input
        if keypoints is None or len(keypoints) != 21:
            return GestureType.UNKNOWN, 0.0
        
        # Update movement history
        self._update_movement_history(keypoints)
        
        # Calculate finger states
        finger_states = calculate_finger_states(keypoints, handedness)
        
        # Match to gestures
        gesture, confidence = self._match_gesture(finger_states, keypoints, handedness)
        
        # Apply stability filtering
        gesture, confidence = self._apply_stability_filter(gesture, confidence)
        
        # Update history
        self._update_gesture_history(gesture, confidence, keypoints, handedness)
        
        # Log significant detections
        if gesture != GestureType.UNKNOWN and confidence > self.gesture_threshold:
            logger.info(f"Gesture detected: {gesture.value} "
                       f"(confidence={confidence:.2f}, hand={handedness})")
        
        return gesture, confidence
    
    def _match_gesture(self, 
                      finger_states: Dict[str, bool], 
                      keypoints: np.ndarray, 
                      handedness: str = "Unknown") -> Tuple[GestureType, float]:
        """
        Match finger states to known gestures.
        
        Delegates to specific detection functions for cleaner code.
        
        Args:
            finger_states: Dictionary of finger extension states
            keypoints: Hand landmarks (21x3 array)
            handedness: Hand type ("Left", "Right", or "Unknown")
        
        Returns:
            Tuple[GestureType, float]: (matched_gesture, confidence_score)
        """
        # Try specific gestures first
        
        # OK gesture
        ok_conf = detect_ok(keypoints)
        if ok_conf > 0.8:
            return GestureType.OK, ok_conf
        
        # Pinch gesture
        pinch_conf = detect_pinch(keypoints)
        if pinch_conf > 0.8:
            return GestureType.PINCH, pinch_conf
        
        # Fist detection
        fist_conf = detect_fist(finger_states, keypoints)
        if fist_conf > 0.8:
            return GestureType.CLOSED_FIST, fist_conf
        
        # Open palm
        partial_ext = get_partial_extensions(keypoints)
        palm_conf = detect_open_palm(finger_states, partial_ext)
        if palm_conf > 0.8:
            return GestureType.OPEN_PALM, palm_conf
        
        # Pointing
        point_conf = detect_pointing(finger_states, keypoints)
        if point_conf > 0.8:
            return GestureType.POINTING, point_conf
        
        # Peace sign
        peace_conf = detect_peace(finger_states, keypoints)
        if peace_conf > 0.8:
            return GestureType.PEACE, peace_conf
        
        # Thumbs up/down
        up_conf, down_conf = detect_thumbs_up_down(finger_states, keypoints, handedness)
        if up_conf > 0.6:
            return GestureType.THUMBS_UP, up_conf
        if down_conf > 0.6:
            return GestureType.THUMBS_DOWN, down_conf
        
        # Rock sign
        rock_conf = detect_rock(finger_states)
        if rock_conf > 0.8:
            return GestureType.ROCK, rock_conf
        
        # Wave (temporal gesture)
        wave_conf = detect_wave(self.gesture_history)
        if wave_conf > 0.8:
            return GestureType.WAVE, wave_conf
        
        # No gesture detected
        return GestureType.UNKNOWN, 0.5
    
    def _update_movement_history(self, keypoints: np.ndarray) -> None:
        """Update point history for motion tracking."""
        if len(keypoints) > 0:
            from .gesture_types import WRIST
            wrist_pos = keypoints[WRIST][:2]  # X,Y only
            current_time = time.time()
            self.point_history.append(wrist_pos)
            self.point_timestamps.append(current_time)
    
    def _apply_stability_filter(self, 
                               gesture: GestureType, 
                               confidence: float) -> Tuple[GestureType, float]:
        """Apply temporal stability to prevent gesture flickering."""
        if self.prev_gesture == gesture:
            self.gesture_stability_count += 1
            # Boost confidence for stable gestures
            if self.gesture_stability_count > self.min_stability_count:
                confidence = min(1.0, confidence * 1.1)
        else:
            # Check if we should stick with previous gesture
            if (confidence < self.prev_confidence * 0.8 and 
                self.gesture_stability_count > self.min_stability_count):
                gesture = self.prev_gesture
                confidence = self.prev_confidence * 0.9
            else:
                self.gesture_stability_count = 0
        
        self.prev_gesture = gesture
        self.prev_confidence = confidence
        return gesture, confidence
    
    def _update_gesture_history(self, 
                               gesture: GestureType,
                               confidence: float,
                               keypoints: np.ndarray,
                               handedness: str) -> None:
        """Update gesture history for temporal analysis."""
        self.gesture_history.append({
            'gesture': gesture,
            'confidence': confidence,
            'keypoints': keypoints.copy(),
            'handedness': handedness,
            'timestamp': time.time()
        })
        
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
    
    
    def _get_partial_extensions(self, keypoints: np.ndarray) -> int:
        """
        Check for partially extended fingers for more lenient recognition.
        Returns count of at least partially extended fingers.
        """
        partially_extended = 0
        wrist = keypoints[self.WRIST]
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        
        # Check thumb
        thumb_tip_dist = np.linalg.norm(keypoints[self.THUMB_TIP][:2] - palm_center[:2])
        thumb_base_dist = np.linalg.norm(keypoints[self.THUMB_BASE][:2] - palm_center[:2])
        if thumb_tip_dist > thumb_base_dist:
            partially_extended += 1
            
        # Check fingers
        for tip_idx, base_idx in [
            (self.INDEX_TIP, self.INDEX_BASE),
            (self.MIDDLE_TIP, self.MIDDLE_BASE),
            (self.RING_TIP, self.RING_BASE),
            (self.PINKY_TIP, self.PINKY_BASE)
        ]:
            tip_dist = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
            base_dist = np.linalg.norm(keypoints[base_idx][:2] - palm_center[:2])
            
            # Very lenient threshold
            if tip_dist > base_dist * 1.05:
                partially_extended += 1
                
        return partially_extended
    
    def _is_waving(self) -> bool:
        Returns:
