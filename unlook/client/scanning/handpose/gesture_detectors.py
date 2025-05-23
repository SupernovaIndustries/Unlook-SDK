"""
Specific gesture detection functions.

This module contains individual detection functions for each
supported gesture type, separated from the main recognizer for
better maintainability.
"""

import numpy as np
import logging
from typing import Dict, List, Optional

from .gesture_types import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    THUMB_BASE, INDEX_BASE, MIDDLE_BASE, RING_BASE, PINKY_BASE
)

logger = logging.getLogger(__name__)


def detect_fist(finger_states: Dict[str, bool], 
                keypoints: np.ndarray) -> float:
    """
    Detect closed fist gesture with multiple validation methods.
    
    Returns confidence score (0-1).
    """
    # Method 1: Count extended fingers
    extended_count = sum(finger_states.values())
    few_extended = extended_count <= 1
    
    # Method 2: Check fingertip positions relative to palm
    palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
    fingertips_folded_count = 0
    
    for tip_idx, base_idx in [
        (INDEX_TIP, INDEX_BASE),
        (MIDDLE_TIP, MIDDLE_BASE),
        (RING_TIP, RING_BASE),
        (PINKY_TIP, PINKY_BASE)
    ]:
        tip_to_palm = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
        base_to_palm = np.linalg.norm(keypoints[base_idx][:2] - palm_center[:2])
        if tip_to_palm < base_to_palm:
            fingertips_folded_count += 1
    
    # Method 3: Check finger curling
    fingers_curled_count = 0
    for tip_idx, mid_idx in [
        (INDEX_TIP, 6),    # Index PIP
        (MIDDLE_TIP, 10),  # Middle PIP
        (RING_TIP, 14),    # Ring PIP
        (PINKY_TIP, 18)    # Pinky PIP
    ]:
        tip_to_palm = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
        mid_to_palm = np.linalg.norm(keypoints[mid_idx][:2] - palm_center[:2])
        if tip_to_palm < mid_to_palm * 0.9:
            fingers_curled_count += 1
    
    # Method 4: Fingertips within palm region
    palm_points = [0, 5, 9, 13, 17]
    palm_radius = np.mean([
        np.linalg.norm(keypoints[i, :2] - palm_center[:2]) 
        for i in palm_points
    ])
    
    fingertips_in_palm = 0
    for tip_idx in [4, 8, 12, 16, 20]:  # All fingertips
        tip_to_center = np.linalg.norm(keypoints[tip_idx, :2] - palm_center[:2])
        if tip_to_center < palm_radius * 1.1:
            fingertips_in_palm += 1
    
    # Calculate confidence based on multiple criteria
    if fingertips_folded_count >= 4 and fingers_curled_count >= 4:
        return 0.95  # Very clear fist
    elif fingertips_in_palm >= 3 and fingers_curled_count >= 3:
        return 0.90  # Clear fist
    elif (fingertips_folded_count >= 3 and fingers_curled_count >= 2) or \
         (few_extended and fingers_curled_count >= 3):
        return 0.85  # Probable fist
    
    return 0.0


def detect_open_palm(finger_states: Dict[str, bool],
                    partial_extensions: int) -> float:
    """
    Detect open palm gesture.
    
    Returns confidence score (0-1).
    """
    extended_count = sum(finger_states.values())
    
    if extended_count >= 4 or (extended_count >= 3 and partial_extensions >= 4):
        # Check if key fingers are extended
        if finger_states.get('thumb', False) and \
           finger_states.get('index', False) and \
           finger_states.get('pinky', False):
            return 0.95
        else:
            return 0.85
    
    return 0.0


def detect_pointing(finger_states: Dict[str, bool],
                   keypoints: np.ndarray) -> float:
    """
    Detect pointing gesture (index finger extended).
    
    Returns confidence score (0-1).
    """
    extended_count = sum(finger_states.values())
    
    if finger_states.get('index', False):
        if extended_count == 1:
            return 0.95  # Only index extended
        elif extended_count == 2 and not finger_states.get('middle', False):
            return 0.85  # Index + thumb/other
        elif extended_count == 2 and finger_states.get('middle', False):
            # Check if index is more extended
            index_ext = np.linalg.norm(
                keypoints[INDEX_TIP][:2] - keypoints[INDEX_BASE][:2]
            )
            middle_ext = np.linalg.norm(
                keypoints[MIDDLE_TIP][:2] - keypoints[MIDDLE_BASE][:2]
            )
            if index_ext > middle_ext * 1.1:
                return 0.8
    
    return 0.0


def detect_peace(finger_states: Dict[str, bool],
                keypoints: np.ndarray) -> float:
    """
    Detect peace sign (index and middle extended).
    
    Returns confidence score (0-1).
    """
    extended_count = sum(finger_states.values())
    
    if finger_states.get('index', False) and finger_states.get('middle', False):
        if extended_count == 2 or (extended_count == 3 and finger_states.get('thumb', False)):
            # Check finger spread
            tip_distance = np.linalg.norm(
                keypoints[INDEX_TIP][:2] - keypoints[MIDDLE_TIP][:2]
            )
            base_distance = np.linalg.norm(
                keypoints[INDEX_BASE][:2] - keypoints[MIDDLE_BASE][:2]
            )
            
            if tip_distance > base_distance * 1.2:
                return 0.9
            else:
                return 0.8
    
    return 0.0


def detect_thumbs_up_down(finger_states: Dict[str, bool],
                         keypoints: np.ndarray,
                         handedness: str) -> tuple[float, float]:
    """
    Detect thumbs up or down gesture.
    
    Returns tuple of (thumbs_up_confidence, thumbs_down_confidence).
    """
    extended_count = sum(finger_states.values())
    
    if not finger_states.get('thumb', False) or extended_count > 2:
        return 0.0, 0.0
    
    # Check if other fingers are closed
    other_fingers_closed = all(
        not finger_states.get(finger, False) 
        for finger in ['index', 'middle', 'ring', 'pinky']
    )
    
    if not other_fingers_closed:
        return 0.0, 0.0
    
    # Analyze thumb orientation
    wrist = keypoints[WRIST]
    thumb_tip = keypoints[THUMB_TIP]
    thumb_vec = thumb_tip - wrist
    
    # Calculate hand orientation
    middle_base = keypoints[MIDDLE_BASE]
    hand_direction = middle_base - wrist
    hand_angle = np.arctan2(hand_direction[1], hand_direction[0])
    is_vertical = abs(np.sin(hand_angle)) > 0.7
    
    thumbs_up_score = 0.0
    thumbs_down_score = 0.0
    
    # Orientation-based detection
    if handedness == "Right":
        if is_vertical:
            if thumb_vec[0] < 0:  # Left
                thumbs_up_score = 0.9
            elif thumb_vec[0] > 0:  # Right
                thumbs_down_score = 0.9
        else:
            if thumb_vec[1] < 0:  # Up
                thumbs_up_score = 0.9
            elif thumb_vec[1] > 0:  # Down
                thumbs_down_score = 0.9
    elif handedness == "Left":
        if is_vertical:
            if thumb_vec[0] > 0:  # Right
                thumbs_up_score = 0.9
            elif thumb_vec[0] < 0:  # Left
                thumbs_down_score = 0.9
        else:
            if thumb_vec[1] < 0:  # Up
                thumbs_up_score = 0.9
            elif thumb_vec[1] > 0:  # Down
                thumbs_down_score = 0.9
    else:
        # Unknown handedness - use simple up/down
        if thumb_vec[1] < -0.1:
            thumbs_up_score = 0.7
        elif thumb_vec[1] > 0.1:
            thumbs_down_score = 0.7
    
    return thumbs_up_score, thumbs_down_score


def detect_ok(keypoints: np.ndarray) -> float:
    """
    Detect OK gesture (thumb and index forming circle).
    
    Returns confidence score (0-1).
    """
    thumb_tip = keypoints[THUMB_TIP]
    index_tip = keypoints[INDEX_TIP]
    
    # Check distance between tips
    distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
    
    # Adaptive threshold based on hand size
    hand_size = np.mean([
        np.linalg.norm(keypoints[tip][:2] - keypoints[WRIST][:2])
        for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    ])
    
    max_coord = np.max(np.abs(keypoints[:, :2]))
    is_normalized = max_coord <= 1.0
    
    if is_normalized:
        distance_threshold = 0.1 * max(0.5, min(1.5, hand_size * 2.0))
    else:
        distance_threshold = 80 * max(0.5, min(1.5, hand_size * 0.05))
    
    tips_close = distance < distance_threshold
    
    if not tips_close:
        return 0.0
    
    # Check if other fingers are extended
    other_finger_checks = 0
    for tip, base in [
        (MIDDLE_TIP, MIDDLE_BASE),
        (RING_TIP, RING_BASE),
        (PINKY_TIP, PINKY_BASE)
    ]:
        tip_base_dist = np.linalg.norm(keypoints[tip][:2] - keypoints[base][:2])
        wrist_base_dist = np.linalg.norm(keypoints[base][:2] - keypoints[WRIST][:2])
        if tip_base_dist > wrist_base_dist * 0.5:
            other_finger_checks += 1
    
    if other_finger_checks >= 1:
        return 0.9
    
    return 0.0


def detect_rock(finger_states: Dict[str, bool]) -> float:
    """
    Detect rock gesture (index and pinky extended).
    
    Returns confidence score (0-1).
    """
    if finger_states.get('index', False) and finger_states.get('pinky', False):
        if not finger_states.get('middle', False) and not finger_states.get('ring', False):
            return 0.95
        
        extended_count = sum(finger_states.values())
        if extended_count == 3 and (finger_states.get('thumb', False) or 
                                   finger_states.get('middle', False)):
            return 0.85
    
    return 0.0


def detect_pinch(keypoints: np.ndarray) -> float:
    """
    Detect pinch gesture (thumb and index close together).
    
    Returns confidence score (0-1).
    """
    thumb_tip = keypoints[THUMB_TIP]
    index_tip = keypoints[INDEX_TIP]
    middle_tip = keypoints[MIDDLE_TIP]
    wrist = keypoints[WRIST]
    
    # Check distance
    distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
    
    # Calculate hand size
    hand_size = np.mean([
        np.linalg.norm(keypoints[INDEX_TIP][:2] - wrist[:2]),
        np.linalg.norm(keypoints[MIDDLE_TIP][:2] - wrist[:2])
    ])
    
    max_coord = np.max(np.abs(keypoints[:, :2]))
    is_normalized = max_coord <= 1.0
    
    if is_normalized:
        distance_threshold = 0.07 * max(0.5, min(1.5, hand_size * 2.0))
    else:
        distance_threshold = 50 * max(0.5, min(1.5, hand_size * 0.05))
    
    if distance >= distance_threshold:
        return 0.0
    
    # Check middle finger position
    middle_extended = np.linalg.norm(middle_tip[:2] - wrist[:2]) > \
                     np.linalg.norm(keypoints[MIDDLE_BASE][:2] - wrist[:2]) * 1.1
    
    # Check thumb/index are forward
    palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
    thumb_forward = np.linalg.norm(thumb_tip[:2] - palm_center[:2]) > 0.3 * hand_size
    index_forward = np.linalg.norm(index_tip[:2] - palm_center[:2]) > 0.3 * hand_size
    
    if (thumb_forward or index_forward) and (middle_extended or index_forward):
        return 0.9
    
    return 0.0


def detect_wave(gesture_history: List[Dict], 
                min_history: int = 10) -> float:
    """
    Detect waving gesture from motion history.
    
    Returns confidence score (0-1).
    """
    if len(gesture_history) < min_history:
        return 0.0
    
    # Check for open palm in recent frames
    recent_gestures = [h.get('gesture', '') for h in gesture_history[-min_history:]]
    from .gesture_types import GestureType
    open_palm_count = sum(1 for g in recent_gestures if g == GestureType.OPEN_PALM)
    
    if open_palm_count < 5:
        return 0.0
    
    # Check for horizontal motion
    wrist_positions = [h['keypoints'][WRIST] for h in gesture_history[-min_history:]]
    x_positions = [pos[0] for pos in wrist_positions]
    
    # Calculate motion range
    x_range = max(x_positions) - min(x_positions)
    
    # Check for direction changes
    direction_changes = 0
    for i in range(1, len(x_positions) - 1):
        if (x_positions[i] - x_positions[i-1]) * (x_positions[i+1] - x_positions[i]) < 0:
            direction_changes += 1
    
    if x_range > 0.1 and direction_changes >= 2:
        return 0.85
    
    return 0.0