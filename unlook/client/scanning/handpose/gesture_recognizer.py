"""Gesture recognition module for UnLook hand tracking.

Simple and easy-to-use gesture recognition based on hand keypoints.
"""

import logging
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque

# Import machine learning models if available
try:
    from .model import KeyPointClassifier, PointHistoryClassifier
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    logging.warning("Machine learning models not available for gesture recognition. Using rule-based fallback.")

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
    
    # Dynamic gestures
    SWIPE_RIGHT = "swipe_right"
    SWIPE_LEFT = "swipe_left"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    CIRCLE = "circle"
    

class GestureRecognizer:
    """Advanced gesture recognition from hand keypoints.
    
    Features:
    - Static gesture recognition (poses like fist, open hand)
    - Dynamic gesture recognition (movements like swipes, circles)
    - ML-based classification when available
    - Rule-based fallback when ML not available
    """
    
    def __init__(self, gesture_threshold: float = 0.8):
        """
        Initialize gesture recognizer.
        
        Args:
            gesture_threshold: Confidence threshold for gesture recognition (0-1)
        """
        self.gesture_threshold = gesture_threshold
        self.gesture_history = []
        self.max_history = 30  # For temporal gestures like waving
        
        # Point history buffer for tracking hand movements
        self.point_history = deque(maxlen=16)  # Store last 16 points
        self.point_timestamps = deque(maxlen=16)  # Timestamps for velocity calculation
        
        # Initialize advanced ML classifiers if available
        global ML_MODELS_AVAILABLE
        if ML_MODELS_AVAILABLE:
            try:
                self.keypoint_classifier = KeyPointClassifier(confidence_threshold=0.7)
                self.point_history_classifier = PointHistoryClassifier(confidence_threshold=0.7)
                logger.info("ML-based gesture classifiers initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing ML classifiers: {e}")
                ML_MODELS_AVAILABLE = False
        
        # Track previous gesture to prevent flickering
        self.prev_gesture = GestureType.UNKNOWN
        self.prev_confidence = 0.0
        self.gesture_stability_count = 0
        self.min_stability_count = 2  # Gestures must be stable for this many frames
        
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
        
        logger.info(f"GestureRecognizer initialized with threshold {gesture_threshold}")
        
    def recognize_gesture(self, keypoints: np.ndarray, handedness: str = "Unknown", confidence_threshold: float = None) -> Tuple[GestureType, float]:
        """
        Recognize gesture from hand keypoints using advanced ML-based classification when available.
        
        Args:
            keypoints: Hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better orientation handling
            confidence_threshold: Optional override for gesture detection threshold
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        if keypoints is None or len(keypoints) != 21:
            return GestureType.UNKNOWN, 0.0
        
        # Use provided threshold or default
        threshold = confidence_threshold if confidence_threshold is not None else self.gesture_threshold
        
        # Use ML-based classification if available
        global ML_MODELS_AVAILABLE
        if ML_MODELS_AVAILABLE and hasattr(self, 'keypoint_classifier'):
            # Convert numpy array to list for the classifier
            landmark_list = keypoints.tolist()
            
            # Update point history for dynamic gesture recognition
            if len(landmark_list) > 0:
                # Use wrist position for tracking movement
                wrist_landmark = landmark_list[0][:2]  # Use only X,Y coordinates
                current_time = time.time()
                
                # Update point history and timestamps
                self.point_history.append(wrist_landmark)
                self.point_timestamps.append(current_time)
                
                # Update point history classifier
                self.point_history_classifier.update_point_history(wrist_landmark, current_time)
            
            # Classify static hand gesture
            keypoint_class_id, keypoint_confidence = self.keypoint_classifier.predict(landmark_list)
            
            # Classify dynamic hand gesture
            history_class_id, history_confidence = self.point_history_classifier.predict()
            
            # Determine final gesture based on static and dynamic classifications
            if history_class_id > 0 and history_confidence > 0.8:  # Strong dynamic gesture
                # Map point history classifier IDs to GestureType
                dynamic_gesture_map = {
                    1: GestureType.SWIPE_RIGHT,
                    2: GestureType.SWIPE_LEFT,
                    3: GestureType.SWIPE_UP,
                    4: GestureType.SWIPE_DOWN,
                    5: GestureType.CIRCLE,
                    6: GestureType.WAVE
                }
                gesture = dynamic_gesture_map.get(history_class_id, GestureType.UNKNOWN)
                confidence = history_confidence
            elif keypoint_class_id >= 0 and keypoint_confidence > 0.7:  # Strong static gesture
                # Map keypoint classifier IDs to GestureType
                static_gesture_map = {
                    0: GestureType.OPEN_PALM,
                    1: GestureType.CLOSED_FIST,
                    2: GestureType.POINTING,
                    3: GestureType.PEACE,
                    4: GestureType.THUMBS_UP,
                    5: GestureType.OK,
                }
                gesture = static_gesture_map.get(keypoint_class_id, GestureType.UNKNOWN)
                confidence = keypoint_confidence
            else:
                # Fallback to traditional method if ML confidence is low
                # Calculate finger states with adaptive thresholds
                finger_states = self._calculate_finger_states(keypoints, handedness)
                gesture, confidence = self._match_gesture(finger_states, keypoints, handedness)
        else:
            # Use traditional method if ML is not available
            # Calculate finger states with adaptive thresholds
            finger_states = self._calculate_finger_states(keypoints, handedness)
            gesture, confidence = self._match_gesture(finger_states, keypoints, handedness)
        
        # Apply gesture stability to prevent flickering
        if self.prev_gesture == gesture:
            self.gesture_stability_count += 1
            # Boost confidence for stable gestures
            if self.gesture_stability_count > self.min_stability_count:
                confidence = min(1.0, confidence * 1.1)
        else:
            # If the new gesture has low confidence compared to previous, stick with previous
            if confidence < self.prev_confidence * 0.8 and self.gesture_stability_count > self.min_stability_count:
                gesture = self.prev_gesture
                confidence = self.prev_confidence * 0.9  # Gradually reduce confidence
            else:
                self.gesture_stability_count = 0
        
        # Store for next frame
        self.prev_gesture = gesture
        self.prev_confidence = confidence
        
        # Store in history for temporal gestures
        self.gesture_history.append({
            'gesture': gesture,
            'confidence': confidence,
            'keypoints': keypoints.copy(),
            'handedness': handedness
        })
        
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        # Log gesture recognition result
        if gesture != GestureType.UNKNOWN and confidence > threshold:
            logger.info(f"HandPose recognized: gesture={gesture.value}, confidence={confidence:.2f}, handedness={handedness}")
        else:
            logger.debug(f"Gesture below threshold: gesture={gesture.value}, confidence={confidence:.2f}")
            
        return gesture, confidence
    
    def _calculate_finger_states(self, keypoints: np.ndarray, handedness: str = "Unknown") -> Dict[str, bool]:
        """
        Calculate if each finger is extended or folded with improved detection.
        
        Args:
            keypoints: Hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better detection
            
        Returns:
            Dictionary with finger states (True = extended, False = folded)
        """
        states = {}
        
        # Check if keypoints are normalized (0-1) or pixel coordinates
        max_coord = np.max(np.abs(keypoints[:, :2]))
        is_normalized = max_coord <= 1.0
        
        # Calculate hand orientation for better detection at different angles
        # Use vector from wrist to middle finger base to define primary axis
        hand_direction_vector = keypoints[self.MIDDLE_BASE] - keypoints[self.WRIST]
        hand_angle = np.arctan2(hand_direction_vector[1], hand_direction_vector[0])
        is_vertical = abs(np.sin(hand_angle)) > 0.7  # Hand is more vertical than horizontal
        
        # Detect if hand is upside down
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        finger_avg = np.mean(keypoints[[4, 8, 12, 16, 20]], axis=0)  # Average fingertip position
        is_upside_down = finger_avg[1] > palm_center[1]  # Fingers below palm in image coordinates
        
        # Calculate hand size for adaptive thresholds
        all_dists = []
        for i in range(1, 21):  # All points except wrist
            dist = np.linalg.norm(keypoints[i, :2] - keypoints[self.WRIST, :2])
            all_dists.append(dist)
        hand_size = np.mean(all_dists)  # Average distance from wrist to all points
        
        # Thumb detection using both distance and angle
        thumb_tip = keypoints[self.THUMB_TIP]
        thumb_base = keypoints[self.THUMB_BASE]
        index_base = keypoints[self.INDEX_BASE]
        wrist = keypoints[self.WRIST]
        
        # Vector from wrist to thumb tip
        thumb_vec = thumb_tip - wrist
        
        # Distance checks
        thumb_length = np.linalg.norm(thumb_tip[:2] - thumb_base[:2])
        wrist_to_base = np.linalg.norm(thumb_base[:2] - wrist[:2])
        
        # Angle check with handedness consideration
        index_dir = index_base - wrist
        thumb_dir = thumb_tip - wrist
        cross_product = np.cross(index_dir[:2], thumb_dir[:2])
        
        # Adjust threshold based on coordinate system
        thumb_dist_threshold = 0.5 if is_normalized else 1.1  # More lenient
        thumb_dist_check = thumb_length > wrist_to_base * thumb_dist_threshold
        
        # Check if thumb is pointing away from other fingers - consider handedness
        # For left hand, negative cross product; for right hand, positive cross product
        if handedness == "Left":
            thumb_angle_check = cross_product > 0  # Left hand thumb points in opposite direction
        elif handedness == "Right":
            thumb_angle_check = cross_product < 0  # Right hand thumb
        else:
            # If handedness unknown, be more lenient - either condition can work
            thumb_angle_check = True
        
        # Check if thumb is separated enough from palm
        thumb_palm_dist = np.linalg.norm(thumb_tip[:2] - palm_center[:2])
        palm_size = np.linalg.norm(keypoints[self.INDEX_BASE][:2] - keypoints[self.PINKY_BASE][:2])
        thumb_separation_check = thumb_palm_dist > palm_size * 0.5
        
        # Combine all checks - more lenient for thumb detection
        states['thumb'] = thumb_dist_check or (thumb_separation_check and abs(cross_product) > 0.01)
        
        # Calculate palm center (weighted more toward the base of the hand)
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        
        # Check each finger with improved detection for various hand orientations
        for finger, tip_idx, base_idx, mid_idx in [
            ('index', self.INDEX_TIP, self.INDEX_BASE, 6),  # 6 is PIP joint
            ('middle', self.MIDDLE_TIP, self.MIDDLE_BASE, 10),  # 10 is PIP joint
            ('ring', self.RING_TIP, self.RING_BASE, 14),  # 14 is PIP joint
            ('pinky', self.PINKY_TIP, self.PINKY_BASE, 18)  # 18 is PIP joint
        ]:
            # Get key points
            tip = keypoints[tip_idx]
            base = keypoints[base_idx]
            mid = keypoints[mid_idx]  # Middle joint for better angle calculation
            
            # For detecting closed fist specifically, see if tips are closer to palm than bases
            # Check if fingertip is INSIDE the palm region (crucial for fist detection)
            inner_palm_center = np.mean(keypoints[[0, 5, 9, 13]], axis=0)  # Center without pinky
            tip_to_palm = np.linalg.norm(tip[:2] - inner_palm_center[:2])
            base_to_palm = np.linalg.norm(base[:2] - inner_palm_center[:2])
            
            # If tip is significantly closer to palm center than base, finger is clearly folded
            inside_palm_check = tip_to_palm < base_to_palm * 0.8  # Clear indication of folded finger
            
            # METHOD 1: HEIGHT CHECK
            # For vertical hands: check if tip is to the left/right of base (depending on hand orientation)
            # For horizontal hands: check if tip is above/below base
            if is_vertical:
                # For vertical hands, check x-coordinate relative to wrist
                wrist_to_base_x = base[0] - wrist[0]
                tip_to_base_x = tip[0] - base[0]
                # Extension is in the same direction as wrist-to-base
                height_check = (wrist_to_base_x * tip_to_base_x) > 0
            elif is_upside_down:
                # For upside-down hands, check is reversed
                tip_y = tip[1]
                base_y = base[1]
                # Extended means tip is lower than base
                height_check = tip_y > base_y
            else:
                # For horizontal hands, use traditional y-coordinate check
                # In image coordinates, lower y means higher position
                tip_y = tip[1]
                base_y = base[1]
                mid_y = mid[1]
                # Check if finger is extended (tip higher than base)
                height_check = tip_y < base_y
            
            # METHOD 2: DISTANCE CHECK
            # Check if tip is far enough from palm center relative to the base
            tip_dist = np.linalg.norm(tip[:2] - palm_center[:2])
            base_dist = np.linalg.norm(base[:2] - palm_center[:2])
            mid_dist = np.linalg.norm(mid[:2] - palm_center[:2])
            
            # Also check vector from base to tip
            base_to_tip_vec = tip[:2] - base[:2]
            base_to_tip_dist = np.linalg.norm(base_to_tip_vec)
            
            # Adjust thresholds based on finger (index/pinky need smaller threshold than middle/ring)
            if finger in ['index', 'pinky']:
                distance_factor = 1.02  # Even lower threshold for outer fingers
            else:
                distance_factor = 1.05  # Lower threshold for middle fingers
                
            # Distance check (tip farther from palm than base)
            distance_check = tip_dist > base_dist * distance_factor
            
            # Check if the finger is extended enough (base to tip distance)
            min_finger_extension = 0.3  # Minimum extension relative to hand size
            extension_check = base_to_tip_dist > hand_size * min_finger_extension
            
            # METHOD 3: STRAIGHTNESS CHECK
            # Check if finger is relatively straight using angle between segments
            seg1 = mid - base
            seg2 = tip - mid
            dot_product = np.sum(seg1[:2] * seg2[:2])
            seg1_len = np.linalg.norm(seg1[:2])
            seg2_len = np.linalg.norm(seg2[:2])
            if seg1_len > 0 and seg2_len > 0:
                cos_angle = dot_product / (seg1_len * seg2_len)
                # More lenient straightness check
                straightness_check = cos_angle > 0.5  # About 60 degrees or straighter
            else:
                straightness_check = False
            
            # Special handling for fist detection - specifically detect curled fingers
            # For a fist, tip should be closer to palm than middle joint
            curled_check = tip_dist < mid_dist
            
            # COMBINE CHECKS with more weight to distance
            # Different combinations for different scenarios
            if inside_palm_check:
                # Fingertip is clearly inside palm - definitely folded
                states[finger] = False
            elif is_vertical:
                # For vertical hands, prioritize distance check
                states[finger] = distance_check and extension_check
            else:
                # For horizontal hands, use a more lenient combination
                states[finger] = (distance_check or height_check) and extension_check and not curled_check
        
        # Log detailed finger state info at trace level
        logger.debug(f"Finger states: {', '.join([f'{k}={v}' for k, v in states.items()])}")
        
        return states
    
    def _match_gesture(self, finger_states: Dict[str, bool], keypoints: np.ndarray, handedness: str = "Unknown") -> Tuple[GestureType, float]:
        """
        Match finger states to known gestures with improved sensitivity.
        
        Args:
            finger_states: Dictionary of finger extension states
            keypoints: Hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better detection
        
        Returns:
            Tuple of (gesture_type, confidence)
        """
        # Count extended fingers
        extended_count = sum(finger_states.values())
        
        # Get some key positions for gesture analysis
        wrist = keypoints[self.WRIST]
        thumb_tip = keypoints[self.THUMB_TIP]
        thumb_base = keypoints[self.THUMB_BASE]
        index_tip = keypoints[self.INDEX_TIP]
        middle_tip = keypoints[self.MIDDLE_TIP]
        ring_tip = keypoints[self.RING_TIP]
        pinky_tip = keypoints[self.PINKY_TIP]
        
        # Calculate hand orientation
        hand_direction = keypoints[self.MIDDLE_BASE] - wrist
        hand_angle = np.arctan2(hand_direction[1], hand_direction[0])
        is_vertical = abs(np.sin(hand_angle)) > 0.7
        
        # Calculate palm center
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        
        # Special case: Check OK and pinch first as they have specific finger configurations
        if self._is_ok_gesture(keypoints, handedness):
            return GestureType.OK, 0.9
        
        if self._is_pinch_gesture(keypoints, handedness):
            return GestureType.PINCH, 0.9
        
        # Check for fist - special direct detection of closed fist for better sensitivity
        # We use multiple methods to detect a fist reliably
        
        # Method 1: Count extended fingers (should be 0-1)
        few_extended = extended_count <= 1
        
        # Method 2: Check if fingertips are closer to palm than bases
        # (This is a strong indication of a closed fist)
        fingertips_folded_count = 0
        for tip_idx, base_idx in [
            (self.INDEX_TIP, self.INDEX_BASE),
            (self.MIDDLE_TIP, self.MIDDLE_BASE),
            (self.RING_TIP, self.RING_BASE),
            (self.PINKY_TIP, self.PINKY_BASE)
        ]:
            tip_to_palm = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
            base_to_palm = np.linalg.norm(keypoints[base_idx][:2] - palm_center[:2])
            if tip_to_palm < base_to_palm:
                fingertips_folded_count += 1
        
        # Method 3: Check if fingers are curled (more reliable for fist detection)
        fingers_curled_count = 0
        for tip_idx, mid_idx in [
            (self.INDEX_TIP, 6),  # Index PIP joint
            (self.MIDDLE_TIP, 10),  # Middle PIP joint
            (self.RING_TIP, 14),  # Ring PIP joint
            (self.PINKY_TIP, 18)   # Pinky PIP joint
        ]:
            # For curled fingers, tip is closer to palm than midpoint
            tip_to_palm = np.linalg.norm(keypoints[tip_idx][:2] - palm_center[:2])
            mid_to_palm = np.linalg.norm(keypoints[mid_idx][:2] - palm_center[:2])
            if tip_to_palm < mid_to_palm * 0.9:  # Clear curling
                fingers_curled_count += 1
                
        # Method 4: Check fingertip to palm distance (more reliable for single-hand cases)
        palm_points = [0, 5, 9, 13, 17]  # Wrist and finger bases
        palm_center = np.mean(keypoints[palm_points, :2], axis=0)
        palm_radius = np.mean([np.linalg.norm(keypoints[i, :2] - palm_center) for i in palm_points])
        
        fingertips_in_palm = 0
        for tip_idx in [4, 8, 12, 16, 20]:  # All fingertips
            tip_to_palm_center = np.linalg.norm(keypoints[tip_idx, :2] - palm_center)
            if tip_to_palm_center < palm_radius * 1.1:  # Fingertip is within palm region
                fingertips_in_palm += 1
        
        fingertips_in_palm_check = fingertips_in_palm >= 3  # Most fingertips are in palm region
        
        # Stronger indication of fist: multiple detection methods agree
        is_fist = (fingertips_folded_count >= 3 and fingers_curled_count >= 2) or \
                 (few_extended and fingers_curled_count >= 3) or \
                 (fingertips_in_palm_check and fingers_curled_count >= 2)
                 
        if is_fist:
            # Higher confidence based on multiple criteria satisfaction
            if fingertips_folded_count >= 4 and fingers_curled_count >= 4:
                confidence = 0.95  # Very clear fist
            elif fingertips_in_palm_check and fingers_curled_count >= 3:
                confidence = 0.90  # Clear fist based on multiple criteria
            else:
                confidence = 0.85  # Probable fist
            return GestureType.CLOSED_FIST, confidence
        
        # Count partially extended fingers (for more lenient palm detection)
        partially_extended = self._get_partial_extensions(keypoints)
        
        # Open palm - all fingers extended or mostly extended (4-5 fingers)
        if extended_count >= 4 or (extended_count >= 3 and partially_extended >= 4):
            # More confident if thumb, index and pinky are clearly extended (full hand spread)
            if finger_states['thumb'] and finger_states['index'] and finger_states['pinky']:
                return GestureType.OPEN_PALM, 0.95
            else:
                return GestureType.OPEN_PALM, 0.85
        
        # Pointing - only index extended or index clearly more extended than others
        if finger_states['index']:
            # Strong pointing - only index fully extended
            if extended_count == 1:
                return GestureType.POINTING, 0.95
            # Weaker pointing - index extended with maybe thumb or one more finger
            elif extended_count == 2 and not finger_states['middle']:
                return GestureType.POINTING, 0.85
            # Even weaker pointing - index and middle might be extended
            elif extended_count == 2 and finger_states['middle']:
                # Check if index is more extended than middle
                index_extension = np.linalg.norm(index_tip[:2] - keypoints[self.INDEX_BASE][:2])
                middle_extension = np.linalg.norm(middle_tip[:2] - keypoints[self.MIDDLE_BASE][:2])
                if index_extension > middle_extension * 1.1:
                    return GestureType.POINTING, 0.8
        
        # Peace sign - index and middle extended, others closed
        if finger_states['index'] and finger_states['middle']:
            if extended_count == 2 or (extended_count == 3 and finger_states['thumb']):
                # Check if they're spread apart (classic peace sign)
                tip_distance = np.linalg.norm(index_tip[:2] - middle_tip[:2])
                base_distance = np.linalg.norm(keypoints[self.INDEX_BASE][:2] - keypoints[self.MIDDLE_BASE][:2])
                if tip_distance > base_distance * 1.2:  # More lenient spread check
                    return GestureType.PEACE, 0.9
                else:
                    # If fingers are close, still recognize as peace but with lower confidence
                    return GestureType.PEACE, 0.8
        
        # Thumbs up/down - greatly improved orientation detection with multiple methods
        if finger_states['thumb'] and extended_count <= 2:  # Thumb and maybe one other finger
            # Method 1: Thumb direction vector relative to wrist
            thumb_vec = thumb_tip - wrist
            
            # Method 2: Use thumb direction relative to palm
            palm_center = np.mean(keypoints[[self.INDEX_BASE, self.MIDDLE_BASE, self.RING_BASE, self.PINKY_BASE]], axis=0)
            thumb_to_palm_vec = thumb_tip - palm_center
            
            # Method 3: Calculate the thumb-to-palm angle in 3D space
            # This is more robust to different hand orientations
            palm_normal = np.cross(
                keypoints[self.INDEX_BASE] - keypoints[self.PINKY_BASE],
                keypoints[self.MIDDLE_BASE] - wrist
            )
            # Normalize palm normal
            if np.linalg.norm(palm_normal) > 0:
                palm_normal = palm_normal / np.linalg.norm(palm_normal)
            
            # Calculate dot product between thumb vector and palm normal
            thumb_normal_dot = np.dot(thumb_to_palm_vec, palm_normal)
            if np.linalg.norm(thumb_to_palm_vec) > 0:
                thumb_normal_angle = thumb_normal_dot / np.linalg.norm(thumb_to_palm_vec)
            else:
                thumb_normal_angle = 0
                
            # METHOD 4: Check if all other fingers are closed (more reliable indication of thumbs up/down)
            other_fingers_closed = (not finger_states['index'] and 
                                    not finger_states['middle'] and 
                                    not finger_states['ring'] and 
                                    not finger_states['pinky'])
            
            # Combined detection logic based on all methods
            thumbs_up_confidence = 0
            thumbs_down_confidence = 0
            
            # Add evidence from method 1 & 2 (directional vectors)
            if handedness == "Left":
                # For left hand
                if is_vertical:
                    # Vertical hand
                    if thumb_vec[0] > 0 and thumb_to_palm_vec[0] > 0:  # Pointing right
                        thumbs_up_confidence += 0.5
                    elif thumb_vec[0] < 0 and thumb_to_palm_vec[0] < 0:  # Pointing left
                        thumbs_down_confidence += 0.5
                else:
                    # Horizontal hand
                    if thumb_vec[1] < 0 and thumb_to_palm_vec[1] < 0:  # Pointing up
                        thumbs_up_confidence += 0.5
                    elif thumb_vec[1] > 0 and thumb_to_palm_vec[1] > 0:  # Pointing down
                        thumbs_down_confidence += 0.5
            elif handedness == "Right":
                # For right hand
                if is_vertical:
                    # Vertical hand
                    if thumb_vec[0] < 0 and thumb_to_palm_vec[0] < 0:  # Pointing left
                        thumbs_up_confidence += 0.5
                    elif thumb_vec[0] > 0 and thumb_to_palm_vec[0] > 0:  # Pointing right
                        thumbs_down_confidence += 0.5
                else:
                    # Horizontal hand
                    if thumb_vec[1] < 0 and thumb_to_palm_vec[1] < 0:  # Pointing up
                        thumbs_up_confidence += 0.5
                    elif thumb_vec[1] > 0 and thumb_to_palm_vec[1] > 0:  # Pointing down
                        thumbs_down_confidence += 0.5
            else:
                # If handedness unknown, give small boost based on orientation only
                if is_vertical:
                    if abs(thumb_vec[0]) > 0.1 and abs(thumb_to_palm_vec[0]) > 0.1:
                        thumbs_up_confidence += 0.3  # Less confidence without handedness
                else:
                    if thumb_vec[1] < -0.1 and thumb_to_palm_vec[1] < -0.1:  # Pointing up
                        thumbs_up_confidence += 0.3
                    elif thumb_vec[1] > 0.1 and thumb_to_palm_vec[1] > 0.1:  # Pointing down
                        thumbs_down_confidence += 0.3
            
            # Add evidence from method 3 (normal angle)
            # Thumbs up/down gesture should have thumb perpendicular to palm
            if abs(thumb_normal_angle) > 0.7:  # High perpendicularity
                if thumb_normal_angle > 0:  # Thumbs up direction
                    thumbs_up_confidence += 0.3
                else:  # Thumbs down direction
                    thumbs_down_confidence += 0.3
            
            # Add evidence from method 4 (other fingers closed)
            if other_fingers_closed:
                thumbs_up_confidence += 0.2
                thumbs_down_confidence += 0.2
            
            # Final decision
            if thumbs_up_confidence > thumbs_down_confidence and thumbs_up_confidence > 0.6:
                return GestureType.THUMBS_UP, min(0.95, thumbs_up_confidence)
            elif thumbs_down_confidence > 0.6:  # Minimum threshold for detection
                return GestureType.THUMBS_DOWN, min(0.95, thumbs_down_confidence)
            
            # Fallback for cases where thumb is clearly extended but direction is ambiguous
            if finger_states['thumb'] and other_fingers_closed and abs(thumb_normal_angle) > 0.6:
                # Default to thumbs up for clearer detection
                # This will at least detect thumb gestures more consistently
                return GestureType.THUMBS_UP, 0.7
        
        # Rock sign - index and pinky extended
        if finger_states['index'] and finger_states['pinky']:
            if not finger_states['middle'] and not finger_states['ring']:
                return GestureType.ROCK, 0.95
            elif extended_count == 3 and (finger_states['thumb'] or finger_states['middle']):
                # Allowing thumb to be extended too for more natural hand position
                return GestureType.ROCK, 0.85
        
        # Check for wave gesture (temporal)
        if self._is_waving():
            return GestureType.WAVE, 0.85
        
        # Not a recognized gesture
        return GestureType.UNKNOWN, 0.6
    
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
    
    def _is_ok_gesture(self, keypoints: np.ndarray, handedness: str = "Unknown") -> bool:
        """
        Improved OK gesture detection (thumb and index forming circle).
        Uses adaptive thresholds based on hand size, orientation, and handedness.
        
        Args:
            keypoints: Hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better detection
        """
        # Early exit for severely malformed hands
        if keypoints is None or len(keypoints) != 21:
            return False
        # Get key points
        thumb_tip = keypoints[self.THUMB_TIP]
        index_tip = keypoints[self.INDEX_TIP]
        wrist = keypoints[self.WRIST]
        
        # Check if keypoints are normalized
        max_coord = np.max(np.abs(keypoints[:, :2]))
        is_normalized = max_coord <= 1.0
        
        # Calculate hand size for adaptive thresholding
        hand_size = np.mean([
            np.linalg.norm(keypoints[self.INDEX_TIP][:2] - keypoints[self.WRIST][:2]),
            np.linalg.norm(keypoints[self.MIDDLE_TIP][:2] - keypoints[self.WRIST][:2]),
            np.linalg.norm(keypoints[self.RING_TIP][:2] - keypoints[self.WRIST][:2]),
            np.linalg.norm(keypoints[self.PINKY_TIP][:2] - keypoints[self.WRIST][:2])
        ])
        
        # Determine hand orientation
        hand_direction = keypoints[self.MIDDLE_BASE] - wrist
        hand_angle = np.arctan2(hand_direction[1], hand_direction[0])
        is_vertical = abs(np.sin(hand_angle)) > 0.7
        
        # Check if thumb and index tips are close
        distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        
        # Use more lenient threshold for more reliable detection
        if is_normalized:
            distance_threshold = 0.1 * max(0.5, min(1.5, hand_size * 2.0))
        else:
            distance_threshold = 80 * max(0.5, min(1.5, hand_size * 0.05))
        
        # Basic proximity check - are tips close?
        tips_close = distance < distance_threshold
        
        # More sophisticated checks for OK position
        # 1. Are the rest of the fingers somewhat extended?
        middle_tip = keypoints[self.MIDDLE_TIP]
        middle_base = keypoints[self.MIDDLE_BASE]
        ring_tip = keypoints[self.RING_TIP]
        ring_base = keypoints[self.RING_BASE]
        pinky_tip = keypoints[self.PINKY_TIP]
        pinky_base = keypoints[self.PINKY_BASE]
        
        # In vertical hand orientation, check differently
        if is_vertical:
            # Check if fingers are extended horizontally
            finger_checks = []
            for tip, base in [(middle_tip, middle_base), (ring_tip, ring_base), (pinky_tip, pinky_base)]:
                # Check if tip is far from base horizontally
                tip_base_dist = np.linalg.norm(tip[:2] - base[:2])
                wrist_base_dist = np.linalg.norm(base[:2] - wrist[:2])
                finger_checks.append(tip_base_dist > wrist_base_dist * 0.6)  # More lenient
        else:
            # More lenient checks for finger extension
            finger_checks = []
            for tip, base in [(middle_tip, middle_base), (ring_tip, ring_base), (pinky_tip, pinky_base)]:
                # Check distance based extension
                tip_base_dist = np.linalg.norm(tip[:2] - base[:2])
                wrist_base_dist = np.linalg.norm(base[:2] - wrist[:2])
                finger_checks.append(tip_base_dist > wrist_base_dist * 0.5)  # More lenient
        
        # 2. Calculate the elevation of thumb and index relative to other fingers
        # For OK gesture, thumb and index should be lower than other fingers
        other_tips_avg_pos = np.mean([middle_tip[:2], ring_tip[:2], pinky_tip[:2]], axis=0)
        thumb_index_avg_pos = np.mean([thumb_tip[:2], index_tip[:2]], axis=0)
        
        # Consider handedness for more accurate checks
        if handedness == "Left":
            # Left hand specifics
            if is_vertical:
                # In vertical orientation, check relative positions with handedness
                thumb_index_lower = True  # More lenient for left hand
            else:
                thumb_index_lower = thumb_index_avg_pos[1] > other_tips_avg_pos[1] - 0.05  # More lenient
        elif handedness == "Right":
            # Right hand specifics
            if is_vertical:
                thumb_index_lower = True  # More lenient for right hand
            else:
                thumb_index_lower = thumb_index_avg_pos[1] > other_tips_avg_pos[1] - 0.05  # More lenient
        else:
            # If handedness unknown, use generic check
            if is_vertical:
                # In vertical hand, check relative X position
                thumb_index_lower = abs(thumb_index_avg_pos[0] - wrist[0]) < abs(other_tips_avg_pos[0] - wrist[0]) + 0.1
            else:
                # In normal orientation, check Y position with leniency
                thumb_index_lower = thumb_index_avg_pos[1] > other_tips_avg_pos[1] - 0.1
        
        # Combine checks with more leniency
        other_fingers_extended = sum(finger_checks) >= 1  # At least 1 other finger somewhat extended
        
        # More lenient overall criteria
        return tips_close and (other_fingers_extended or thumb_index_lower)
    
    def _is_pinch_gesture(self, keypoints: np.ndarray, handedness: str = "Unknown") -> bool:
        """
        Improved pinch gesture detection with more lenient thresholds.
        Pinch is when thumb and index fingers are close, others can be in any position.
        
        Args:
            keypoints: Hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better detection
        """
        # Early exit for severely malformed hands
        if keypoints is None or len(keypoints) != 21:
            return False
        # Get key points
        thumb_tip = keypoints[self.THUMB_TIP]
        index_tip = keypoints[self.INDEX_TIP]
        wrist = keypoints[self.WRIST]
        
        # Check if keypoints are normalized
        max_coord = np.max(np.abs(keypoints[:, :2]))
        is_normalized = max_coord <= 1.0
        
        # Calculate hand size for adaptive thresholding
        hand_size = np.mean([
            np.linalg.norm(keypoints[self.INDEX_TIP][:2] - keypoints[self.WRIST][:2]),
            np.linalg.norm(keypoints[self.MIDDLE_TIP][:2] - keypoints[self.WRIST][:2])
        ])
        
        # Check if thumb and index tips are very close
        distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        
        # More lenient threshold for better detection
        if is_normalized:
            distance_threshold = 0.07 * max(0.5, min(1.5, hand_size * 2.0))
        else:
            distance_threshold = 50 * max(0.5, min(1.5, hand_size * 0.05))
        
        # Basic check - are tips close enough?
        tips_close = distance < distance_threshold
        
        # Calculate middle finger position to verify pinch
        middle_tip = keypoints[self.MIDDLE_TIP]
        
        # For pinch, middle finger is usually somewhat extended
        # This helps distinguish from fist where all fingers are curled
        middle_extended = np.linalg.norm(middle_tip[:2] - wrist[:2]) > np.linalg.norm(keypoints[self.MIDDLE_BASE][:2] - wrist[:2]) * 1.1
        
        # Make sure thumb and index are forward enough to be a pinch
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        
        # Check if thumb and index are somewhat extended (not in palm)
        thumb_forward = np.linalg.norm(thumb_tip[:2] - palm_center[:2]) > 0.3 * hand_size
        index_forward = np.linalg.norm(index_tip[:2] - palm_center[:2]) > 0.3 * hand_size
        
        # Combine all checks with more leniency
        return tips_close and (thumb_forward or index_forward) and (middle_extended or index_forward)
    
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
    
    def recognize_gestures_3d(self, keypoints_3d: np.ndarray, handedness: str = "Unknown", confidence_threshold: float = None) -> Tuple[GestureType, float]:
        """
        Recognize gestures from 3D keypoints with improved sensitivity.
        
        Args:
            keypoints_3d: 3D hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better detection
            confidence_threshold: Optional override for gesture detection threshold
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        """
        Recognize gestures from 3D keypoints with improved sensitivity.
        
        Args:
            keypoints_3d: 3D hand landmarks (21x3 array)
            handedness: Optional hand type ("Left" or "Right") for better detection
            confidence_threshold: Optional override for gesture detection threshold
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        # Normalize 3D points to use with 2D recognition
        normalized_2d = keypoints_3d[:, :2] / np.max(np.abs(keypoints_3d[:, :2])) if np.max(np.abs(keypoints_3d[:, :2])) > 0 else keypoints_3d[:, :2]
        
        # Add dummy z-coordinates
        keypoints_2d = np.column_stack([normalized_2d, np.zeros(21)])
        
        # Use a looser threshold for 3D gestures to compensate for depth uncertainty
        effective_threshold = 0.5 if confidence_threshold is None else confidence_threshold
        
        # Pass handedness information for better gesture recognition
        return self.recognize_gesture(keypoints_2d, handedness, effective_threshold)
    
    def recognize_gesture_2d(self, keypoints_2d: np.ndarray, image_width: int, image_height: int, 
                            handedness: str = "Unknown", confidence_threshold: float = None) -> Tuple[GestureType, float]:
        """
        Recognize gestures from 2D normalized keypoints (MediaPipe format).
        Enhanced to work better with single hand detection by using more lenient thresholds.
        
        Args:
            keypoints_2d: 2D hand landmarks in normalized coordinates (21x3 array)
            image_width: Width of the image
            image_height: Height of the image
            handedness: Optional hand type ("Left" or "Right") for better detection
            confidence_threshold: Optional override for gesture detection threshold
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        """
        Recognize gestures from 2D normalized keypoints (MediaPipe format).
        
        Args:
            keypoints_2d: 2D hand landmarks in normalized coordinates (21x3 array)
            image_width: Width of the image
            image_height: Height of the image
            handedness: Optional hand type ("Left" or "Right") for better detection
            confidence_threshold: Optional override for gesture detection threshold
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = keypoints_2d.copy()
        pixel_coords[:, 0] *= image_width
        pixel_coords[:, 1] *= image_height
        
        # For 2D-only gesture recognition, use a more lenient threshold if not specified
        if confidence_threshold is None:
            # More lenient threshold for 2D gestures since we have less information
            confidence_threshold = max(0.45, self.gesture_threshold - 0.1)
            
        # Use the pixel coordinates for gesture recognition with handedness info
        return self.recognize_gesture(pixel_coords, handedness, confidence_threshold)
    
    def get_gesture_name(self, gesture_type: GestureType) -> str:
        """Get human-readable name for gesture."""
        return gesture_type.value.replace('_', ' ').title()
    
    def clear_history(self):
        """Clear gesture history."""
        self.gesture_history = []
        logger.debug("Gesture history cleared")