"""
Finger state detection utilities for gesture recognition.

This module provides functions to detect whether fingers are extended
or folded based on hand landmarks, handling various hand orientations.
"""

import numpy as np
import logging
from typing import Dict, Tuple

from .gesture_types import (
    WRIST, THUMB_TIP, THUMB_BASE, INDEX_TIP, INDEX_BASE, 
    MIDDLE_TIP, MIDDLE_BASE, RING_TIP, RING_BASE, 
    PINKY_TIP, PINKY_BASE
)

logger = logging.getLogger(__name__)


def calculate_finger_states(keypoints: np.ndarray, 
                          handedness: str = "Unknown") -> Dict[str, bool]:
    """
    Calculate if each finger is extended or folded.
    
    Uses multiple detection methods for robustness:
    - Height comparison (tip vs base position)
    - Distance from palm center
    - Finger straightness
    - Special handling for thumb
    
    Args:
        keypoints: Hand landmarks (21x3 array)
        handedness: Hand type ("Left", "Right", or "Unknown")
        
    Returns:
        Dict[str, bool]: Finger states where True = extended
    """
    if keypoints is None or len(keypoints) != 21:
        return {}
    
    states = {}
    
    # Determine if coordinates are normalized
    max_coord = np.max(np.abs(keypoints[:, :2]))
    is_normalized = max_coord <= 1.0
    
    # Calculate hand orientation
    hand_direction = keypoints[MIDDLE_BASE] - keypoints[WRIST]
    hand_angle = np.arctan2(hand_direction[1], hand_direction[0])
    is_vertical = abs(np.sin(hand_angle)) > 0.7
    
    # Detect if hand is upside down
    palm_center = _calculate_palm_center(keypoints)
    finger_avg = np.mean(keypoints[[4, 8, 12, 16, 20]], axis=0)
    is_upside_down = finger_avg[1] > palm_center[1]
    
    # Calculate hand size for adaptive thresholds
    hand_size = _calculate_hand_size(keypoints)
    
    # Detect thumb state
    states['thumb'] = _detect_thumb_extended(
        keypoints, handedness, is_normalized, palm_center, hand_size
    )
    
    # Detect other fingers
    finger_configs = [
        ('index', INDEX_TIP, INDEX_BASE, 6),    # PIP joint at index 6
        ('middle', MIDDLE_TIP, MIDDLE_BASE, 10), # PIP joint at index 10
        ('ring', RING_TIP, RING_BASE, 14),      # PIP joint at index 14
        ('pinky', PINKY_TIP, PINKY_BASE, 18)    # PIP joint at index 18
    ]
    
    for finger_name, tip_idx, base_idx, mid_idx in finger_configs:
        states[finger_name] = _detect_finger_extended(
            keypoints, tip_idx, base_idx, mid_idx,
            palm_center, hand_size, is_vertical, is_upside_down, finger_name
        )
    
    logger.debug(f"Finger states: {', '.join([f'{k}={v}' for k, v in states.items()])}")
    return states


def _calculate_palm_center(keypoints: np.ndarray) -> np.ndarray:
    """Calculate the center of the palm from base joints."""
    return np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)


def _calculate_hand_size(keypoints: np.ndarray) -> float:
    """Calculate average hand size for adaptive thresholds."""
    wrist = keypoints[WRIST]
    distances = []
    for i in range(1, 21):
        dist = np.linalg.norm(keypoints[i, :2] - wrist[:2])
        distances.append(dist)
    return np.mean(distances)


def _detect_thumb_extended(keypoints: np.ndarray, 
                          handedness: str,
                          is_normalized: bool,
                          palm_center: np.ndarray,
                          hand_size: float) -> bool:
    """
    Detect if thumb is extended using multiple criteria.
    
    Thumb detection is complex due to its different range of motion
    compared to other fingers.
    """
    thumb_tip = keypoints[THUMB_TIP]
    thumb_base = keypoints[THUMB_BASE]
    index_base = keypoints[INDEX_BASE]
    wrist = keypoints[WRIST]
    
    # Distance checks
    thumb_length = np.linalg.norm(thumb_tip[:2] - thumb_base[:2])
    wrist_to_base = np.linalg.norm(thumb_base[:2] - wrist[:2])
    
    # Adaptive threshold
    thumb_dist_threshold = 0.5 if is_normalized else 1.1
    thumb_dist_check = thumb_length > wrist_to_base * thumb_dist_threshold
    
    # Angle check with handedness
    index_dir = index_base - wrist
    thumb_dir = thumb_tip - wrist
    cross_product = np.cross(index_dir[:2], thumb_dir[:2])
    
    # Handedness-aware angle check
    if handedness == "Left":
        thumb_angle_check = cross_product > 0
    elif handedness == "Right":
        thumb_angle_check = cross_product < 0
    else:
        thumb_angle_check = True  # Lenient if handedness unknown
    
    # Separation from palm check
    thumb_palm_dist = np.linalg.norm(thumb_tip[:2] - palm_center[:2])
    palm_size = np.linalg.norm(keypoints[INDEX_BASE][:2] - keypoints[PINKY_BASE][:2])
    thumb_separation_check = thumb_palm_dist > palm_size * 0.5
    
    # Combine checks
    return thumb_dist_check or (thumb_separation_check and abs(cross_product) > 0.01)


def _detect_finger_extended(keypoints: np.ndarray,
                           tip_idx: int,
                           base_idx: int,
                           mid_idx: int,
                           palm_center: np.ndarray,
                           hand_size: float,
                           is_vertical: bool,
                           is_upside_down: bool,
                           finger_name: str) -> bool:
    """
    Detect if a finger is extended using multiple methods.
    
    Combines height, distance, and straightness checks for robust detection
    across different hand orientations.
    """
    tip = keypoints[tip_idx]
    base = keypoints[base_idx]
    mid = keypoints[mid_idx]
    wrist = keypoints[WRIST]
    
    # Check if fingertip is inside palm (strong indicator of folded)
    inner_palm_center = np.mean(keypoints[[0, 5, 9, 13]], axis=0)
    tip_to_palm = np.linalg.norm(tip[:2] - inner_palm_center[:2])
    base_to_palm = np.linalg.norm(base[:2] - inner_palm_center[:2])
    
    if tip_to_palm < base_to_palm * 0.8:
        return False  # Clearly folded
    
    # Height check based on orientation
    if is_vertical:
        wrist_to_base_x = base[0] - wrist[0]
        tip_to_base_x = tip[0] - base[0]
        height_check = (wrist_to_base_x * tip_to_base_x) > 0
    elif is_upside_down:
        height_check = tip[1] > base[1]
    else:
        height_check = tip[1] < base[1]
    
    # Distance check
    tip_dist = np.linalg.norm(tip[:2] - palm_center[:2])
    base_dist = np.linalg.norm(base[:2] - palm_center[:2])
    
    # Adaptive threshold based on finger
    distance_factor = 1.02 if finger_name in ['index', 'pinky'] else 1.05
    distance_check = tip_dist > base_dist * distance_factor
    
    # Extension check
    base_to_tip_dist = np.linalg.norm(tip[:2] - base[:2])
    extension_check = base_to_tip_dist > hand_size * 0.3
    
    # Straightness check
    straightness_check = _check_finger_straightness(base, mid, tip)
    
    # Check if finger is curled
    mid_dist = np.linalg.norm(mid[:2] - palm_center[:2])
    curled_check = tip_dist < mid_dist
    
    # Combine checks
    if is_vertical:
        return distance_check and extension_check
    else:
        return (distance_check or height_check) and extension_check and not curled_check


def _check_finger_straightness(base: np.ndarray, 
                              mid: np.ndarray, 
                              tip: np.ndarray) -> bool:
    """Check if finger segments are relatively straight."""
    seg1 = mid - base
    seg2 = tip - mid
    
    seg1_len = np.linalg.norm(seg1[:2])
    seg2_len = np.linalg.norm(seg2[:2])
    
    if seg1_len > 0 and seg2_len > 0:
        dot_product = np.sum(seg1[:2] * seg2[:2])
        cos_angle = dot_product / (seg1_len * seg2_len)
        return cos_angle > 0.5  # ~60 degrees or straighter
    
    return False


def get_partial_extensions(keypoints: np.ndarray) -> int:
    """
    Count partially extended fingers for lenient recognition.
    
    Args:
        keypoints: Hand landmarks (21x3 array)
        
    Returns:
        int: Number of at least partially extended fingers
    """
    partially_extended = 0
    palm_center = _calculate_palm_center(keypoints)
    
    # Check thumb
    thumb_tip_dist = np.linalg.norm(keypoints[THUMB_TIP][:2] - palm_center[:2])
    thumb_base_dist = np.linalg.norm(keypoints[THUMB_BASE][:2] - palm_center[:2])
    if thumb_tip_dist > thumb_base_dist:
        partially_extended += 1
    
    # Check fingers
    for tip_idx, base_idx in [
        (INDEX_TIP, INDEX_BASE),
        (MIDDLE_TIP, MIDDLE_BASE),
        (RING_TIP, RING_BASE),
        (PINKY_TIP, PINKY_BASE)
    ]:
        tip_dist = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
        base_dist = np.linalg.norm(keypoints[base_idx][:2] - palm_center[:2])
        
        if tip_dist > base_dist * 1.05:  # Very lenient
            partially_extended += 1
    
    return partially_extended