"""3D Hand tracking using stereo cameras for UnLook SDK.

Based on handpose3d by TemugeB: https://github.com/TemugeB/handpose3d
"""

import time
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

from .hand_detector import HandDetector
from .gesture_recognizer import GestureRecognizer, GestureType

# Import dynamic gesture recognizer if available
try:
    from .dynamic_gesture_recognizer import DynamicGestureRecognizer, DynamicEvent
    DYNAMIC_GESTURES_AVAILABLE = True
except ImportError:
    DYNAMIC_GESTURES_AVAILABLE = False
    logging.warning("Dynamic gesture recognition not available - using standard gesture recognition only.")

logger = logging.getLogger(__name__)


class HandTracker:
    """3D hand tracking using stereo cameras and triangulation."""
    
    def __init__(self,
                 calibration_file: Optional[str] = None,
                 detection_confidence: float = 0.6,  # Increased for more reliable detection
                 tracking_confidence: float = 0.6,   # Increased for more reliable tracking
                 max_num_hands: int = 2,            # Reduced to focus on primary hands
                 left_camera_mirror_mode: bool = False,  # Is left camera selfie/mirrored view? Set to False for UnLook setup
                 right_camera_mirror_mode: bool = False):  # Is right camera selfie/mirrored view? Set to False for UnLook setup
        """
        Initialize the 3D hand tracker.
        
        Args:
            calibration_file: Path to stereo calibration file
            detection_confidence: Minimum confidence for hand detection (0.4 default for better sensitivity)
            tracking_confidence: Minimum confidence for hand tracking (0.4 default for better sensitivity)
            max_num_hands: Maximum number of hands to track (default 4 for better multi-hand tracking)
        """
        # Camera mirror modes determine how handedness is interpreted
        self.left_camera_mirror_mode = left_camera_mirror_mode
        self.right_camera_mirror_mode = right_camera_mirror_mode
        
        # Initialize hand detector with improved parameters
        # Left camera (typically front-facing/selfie) needs mirror mode
        self.detector_left = HandDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            max_num_hands=max_num_hands,
            hand_mirror_mode=left_camera_mirror_mode
        )
        
        # Right camera (typically world-facing) needs different mirror mode
        self.detector_right = HandDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            max_num_hands=max_num_hands,
            hand_mirror_mode=right_camera_mirror_mode
        )
        
        # Initialize gesture recognizer with more lenient threshold
        self.gesture_recognizer = GestureRecognizer(gesture_threshold=0.6)  # Lower threshold for better gesture detection
        
        # Initialize dynamic gesture recognizer if available
        self.dynamic_recognizer = None
        if DYNAMIC_GESTURES_AVAILABLE:
            try:
                self.dynamic_recognizer = DynamicGestureRecognizer()
                if self.dynamic_recognizer.models_available:
                    logger.info("Dynamic gesture recognition initialized successfully")
                else:
                    logger.warning("Dynamic gesture models not found - using standard gesture recognition only")
                    self.dynamic_recognizer = None
            except Exception as e:
                logger.error(f"Failed to initialize dynamic gesture recognition: {e}")
                self.dynamic_recognizer = None
        
        # Load calibration if provided
        self.calibration_loaded = False
        if calibration_file:
            if Path(calibration_file).exists():
                logger.info(f"Loading calibration from {calibration_file}")
                self.load_calibration(calibration_file)
            else:
                logger.warning(f"Calibration file not found: {calibration_file}")
        else:
            logger.info("No calibration file provided. 2D tracking only.")
        
        # Storage for tracked hands
        self.tracked_hands = []
        self.frame_count = 0
        
        # Configuration
        self.max_history = 1000  # Maximum frames to store
        
        # For temporal handedness stabilization
        self.prev_handedness = []
        self.handedness_history = []
        self.handedness_stability_count = 0
        
    def load_calibration(self, calibration_file: str):
        """
        Load stereo calibration parameters.
        
        Args:
            calibration_file: Path to calibration file (JSON or numpy)
        """
        try:
            if calibration_file.endswith('.json'):
                import json
                with open(calibration_file, 'r') as f:
                    calib_data = json.load(f)
                
                # Extract calibration matrices
                self.K1 = np.array(calib_data.get('camera_matrix_left', []))
                self.K2 = np.array(calib_data.get('camera_matrix_right', []))
                self.D1 = np.array(calib_data.get('dist_coeffs_left', []))
                self.D2 = np.array(calib_data.get('dist_coeffs_right', []))
                self.R = np.array(calib_data.get('R', []))
                self.T = np.array(calib_data.get('T', []))
                
                # Get projection matrices
                if 'P1' in calib_data and 'P2' in calib_data:
                    self.P1 = np.array(calib_data['P1'])
                    self.P2 = np.array(calib_data['P2'])
                else:
                    # Compute projection matrices
                    self.P1, self.P2 = self._compute_projection_matrices()
                
                self.calibration_loaded = True
                logger.info("Calibration loaded successfully")
                
            else:
                # Load numpy files
                calib_dir = Path(calibration_file).parent
                self.P1 = np.load(calib_dir / 'P1.npy')
                self.P2 = np.load(calib_dir / 'P2.npy')
                self.calibration_loaded = True
                logger.info("Projection matrices loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.calibration_loaded = False
    
    def _compute_projection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute projection matrices from calibration parameters."""
        # Create projection matrices
        # P1 = K1 * [I | 0]
        # P2 = K2 * [R | T]
        P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([self.R, self.T])
        
        return P1, P2
    
    def track_hands_3d(self, 
                      left_image: np.ndarray, 
                      right_image: np.ndarray,
                      min_detection_confidence: float = None,
                      min_tracking_confidence: float = None,
                      prioritize_left_camera: bool = True,
                      stabilize_handedness: bool = True) -> Dict[str, Any]:
        """
        Track hands in 3D using stereo images with improved single-hand support.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
            min_detection_confidence: Override the minimum confidence for detection (higher = fewer false positives)
            min_tracking_confidence: Override the minimum confidence for tracking (higher = more stable tracking)
            prioritize_left_camera: Give priority to left camera for single-hand detection
            stabilize_handedness: Apply temporal smoothing to prevent handedness flipping
        
        Returns:
            Dictionary containing:
            - '3d_keypoints': List of 3D hand keypoints
            - '2d_left': 2D keypoints in left image
            - '2d_right': 2D keypoints in right image
            - 'confidence': Tracking confidence scores
            - 'handedness': List of left/right labels
            - 'gestures': Recognized gestures
        """
        """
        Track hands in 3D using stereo images with improved single-hand support.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
            min_detection_confidence: Override the minimum confidence for detection (higher = fewer false positives)
            min_tracking_confidence: Override the minimum confidence for tracking (higher = more stable tracking)
            prioritize_left_camera: Give priority to left camera for single-hand detection
        
        Returns:
            Dictionary containing:
            - '3d_keypoints': List of 3D hand keypoints
            - '2d_left': 2D keypoints in left image
            - '2d_right': 2D keypoints in right image
            - 'confidence': Tracking confidence scores
            - 'handedness': List of left/right labels
            - 'gestures': Recognized gestures
        """
        """
        Track hands in 3D using stereo images with improved single-hand support.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
            min_detection_confidence: Override the minimum confidence for detection (higher = fewer false positives)
            min_tracking_confidence: Override the minimum confidence for tracking (higher = more stable tracking)
        
        Returns:
            Dictionary containing:
            - '3d_keypoints': List of 3D hand keypoints
            - '2d_left': 2D keypoints in left image
            - '2d_right': 2D keypoints in right image
            - 'confidence': Tracking confidence scores
            - 'handedness': List of left/right labels
            - 'gestures': Recognized gestures
        """
        # Override confidence thresholds for left detector if specified (helps filter out false positives)
        if min_detection_confidence is not None or min_tracking_confidence is not None:
            temp_left_detection_conf = self.detector_left.detection_confidence
            temp_left_tracking_conf = self.detector_left.tracking_confidence
            
            # Temporarily update confidence thresholds if specified
            if min_detection_confidence is not None:
                self.detector_left.detection_confidence = max(self.detector_left.detection_confidence, min_detection_confidence)
            if min_tracking_confidence is not None:
                self.detector_left.tracking_confidence = max(self.detector_left.tracking_confidence, min_tracking_confidence)
            
            # Do the same for right detector
            temp_right_detection_conf = self.detector_right.detection_confidence
            temp_right_tracking_conf = self.detector_right.tracking_confidence
            
            if min_detection_confidence is not None:
                self.detector_right.detection_confidence = max(self.detector_right.detection_confidence, min_detection_confidence)
            if min_tracking_confidence is not None:
                self.detector_right.tracking_confidence = max(self.detector_right.tracking_confidence, min_tracking_confidence)
                
            # Log the temporary confidence overrides
            logger.debug(f"Using temporary confidence thresholds for left camera: detection={self.detector_left.detection_confidence}, tracking={self.detector_left.tracking_confidence}")
            logger.debug(f"Using temporary confidence thresholds for right camera: detection={self.detector_right.detection_confidence}, tracking={self.detector_right.tracking_confidence}")
        
        # Detect hands in left image using the left-specific detector
        left_results = self.detector_left.detect_hands(left_image)
        
        # Detect hands in right image using the right-specific detector
        right_results = self.detector_right.detect_hands(right_image)
        
        # Combine into stereo results format
        stereo_results = {
            'left': left_results,
            'right': right_results,
            'timestamp': time.time()
        }
        
        # Restore original confidence thresholds if they were modified
        if min_detection_confidence is not None or min_tracking_confidence is not None:
            self.detector_left.detection_confidence = temp_left_detection_conf
            self.detector_left.tracking_confidence = temp_left_tracking_conf
            self.detector_right.detection_confidence = temp_right_detection_conf
            self.detector_right.tracking_confidence = temp_right_tracking_conf
        
        # Initialize output with 2D detection results
        output = {
            '3d_keypoints': [],
            '2d_left': stereo_results['left']['keypoints'],
            '2d_right': stereo_results['right']['keypoints'],
            'handedness_left': stereo_results['left']['handedness'],
            'handedness_right': stereo_results['right']['handedness'],
            'confidence': [],
            'gestures': [],  # Will hold gesture recognition results
            'dynamic_gestures': [],  # Will hold dynamic gesture recognition results
            'timestamp': time.time(),
            'frame': self.frame_count
        }
        
        # If handedness stabilization is requested, apply temporal smoothing to prevent flipping
        if stabilize_handedness and len(self.tracked_hands) > 0:
            # Get previous handedness for both cameras
            prev_frame = self.tracked_hands[-1]
            
            # Stabilize left camera handedness
            if 'handedness_left' in prev_frame and len(prev_frame['handedness_left']) > 0:
                # For each hand detected in current frame
                for i, handedness in enumerate(output.get('handedness_left', [])):
                    # Skip if we don't have enough previous hands
                    if i >= len(prev_frame['handedness_left']):
                        continue
                        
                    prev_handedness = prev_frame['handedness_left'][i]
                    curr_handedness = output['handedness_left'][i]
                    
                    # If handedness is different from previous frame but confidence is not very high,
                    # maintain the previous handedness to prevent jitter
                    curr_conf = stereo_results['left']['confidence'][i] if i < len(stereo_results['left']['confidence']) else 0.6
                    if prev_handedness != curr_handedness and curr_conf < 0.8:
                        logger.debug(f"Stabilizing left camera handedness: {curr_handedness} -> {prev_handedness} (conf={curr_conf:.2f})")
                        output['handedness_left'][i] = prev_handedness
            
            # Stabilize right camera handedness
            if 'handedness_right' in prev_frame and len(prev_frame['handedness_right']) > 0:
                # For each hand detected in current frame
                for i, handedness in enumerate(output.get('handedness_right', [])):
                    # Skip if we don't have enough previous hands
                    if i >= len(prev_frame['handedness_right']):
                        continue
                        
                    prev_handedness = prev_frame['handedness_right'][i]
                    curr_handedness = output['handedness_right'][i]
                    
                    # If handedness is different from previous frame but confidence is not very high,
                    # maintain the previous handedness to prevent jitter
                    curr_conf = stereo_results['right']['confidence'][i] if i < len(stereo_results['right']['confidence']) else 0.6
                    if prev_handedness != curr_handedness and curr_conf < 0.8:
                        logger.debug(f"Stabilizing right camera handedness: {curr_handedness} -> {prev_handedness} (conf={curr_conf:.2f})")
                        output['handedness_right'][i] = prev_handedness
        
        # Process both left and right camera hands for 2D gestures
        h, w = left_image.shape[:2]
        all_2d_gestures = []
        
        # For single-hand usage, we prioritize the left camera (usually front-facing)
        # This improves reliability when only one hand is in the field of view
        
        # Recognize gestures from left camera
        for i, keypoints in enumerate(stereo_results['left']['keypoints']):
            if i < len(stereo_results['left']['keypoints']):
                # Get handedness to help with gesture recognition
                handedness = "Unknown"
                if i < len(stereo_results['left']['handedness']):
                    handedness = stereo_results['left']['handedness'][i]
                
                # Use looser thresholds for 2D gestures (better detection)
                gesture_type, gesture_conf = self.gesture_recognizer.recognize_gesture_2d(
                    keypoints, w, h, handedness=handedness, confidence_threshold=0.45
                )
                
                # Only add if the gesture is recognized with decent confidence
                if gesture_type != GestureType.UNKNOWN or gesture_conf >= 0.45:
                    all_2d_gestures.append({
                        'type': gesture_type,
                        'confidence': gesture_conf,
                        'name': self.gesture_recognizer.get_gesture_name(gesture_type),
                        'view': 'left',  # Mark that this is from left view only
                        'is_3d': False,   # Flag that this is not from 3D matching
                        'hand_idx': i,  # Store the hand index for reference
                        'handedness': handedness,  # Store handedness
                        'camera': 'left'  # Which camera this is from
                    })
        
        # Recognize gestures from right camera (ensure both cameras can contribute)
        for i, keypoints in enumerate(stereo_results['right']['keypoints']):
            if i < len(stereo_results['right']['keypoints']):
                # Get handedness to help with gesture recognition
                handedness = "Unknown"
                if i < len(stereo_results['right']['handedness']):
                    handedness = stereo_results['right']['handedness'][i]
                
                # Use looser thresholds for 2D gestures (better detection)
                gesture_type, gesture_conf = self.gesture_recognizer.recognize_gesture_2d(
                    keypoints, w, h, handedness=handedness, confidence_threshold=0.45
                )
                
                # Only add if the gesture is recognized with decent confidence
                if gesture_type != GestureType.UNKNOWN or gesture_conf >= 0.45:
                    all_2d_gestures.append({
                        'type': gesture_type,
                        'confidence': gesture_conf,
                        'name': self.gesture_recognizer.get_gesture_name(gesture_type),
                        'view': 'right',  # Mark that this is from right view only
                        'is_3d': False,   # Flag that this is not from 3D matching
                        'hand_idx': i,  # Store the hand index for reference
                        'handedness': handedness,  # Store handedness
                        'camera': 'right'  # Which camera this is from
                    })
        
        # Sort 2D gestures by confidence and get the best one if needed
        all_2d_gestures.sort(key=lambda g: g['confidence'], reverse=True)
        
        # If we have gestures from either camera, boost their confidence slightly for single-hand use
        if prioritize_left_camera and len(all_2d_gestures) > 0:
            for gesture in all_2d_gestures:
                # Prioritize left camera gestures - they're often more reliable
                if gesture['camera'] == 'left':
                    gesture['confidence'] = min(1.0, gesture['confidence'] * 1.1)
        
        # If no calibration, just return 2D results only
        if not self.calibration_loaded or not isinstance(self.P1, np.ndarray) or not isinstance(self.P2, np.ndarray):
            logger.info("No valid calibration loaded - returning 2D tracking results only")
            # Use the 2D gestures as our output
            output['gestures'] = all_2d_gestures
            self.frame_count += 1
            return output
        
        # Match hands between left and right images for 3D analysis
        matched_hands = self._match_stereo_hands(stereo_results)
        
        # Keep track of which hands are matched (to avoid duplicate gestures)
        matched_left_indices = []
        matched_right_indices = []
        
        # Triangulate matched hands for 3D gesture detection
        for match in matched_hands:
            left_idx, right_idx = match['indices']
            matched_left_indices.append(left_idx)
            matched_right_indices.append(right_idx)
            
            # Get 2D keypoints
            left_kpts = stereo_results['left']['keypoints'][left_idx]
            right_kpts = stereo_results['right']['keypoints'][right_idx]
            
            # Convert to pixel coordinates
            h, w = left_image.shape[:2]
            # Make sure keypoints are in the right shape before converting to pixels
            if left_kpts.ndim != 2:
                left_kpts = left_kpts.reshape(21, -1)  # Reshape to 21 landmarks
            if right_kpts.ndim != 2:
                right_kpts = right_kpts.reshape(21, -1)
                
            # Use detector_left for left keypoints, detector_right for right keypoints
            left_pixels = self.detector_left.get_2d_pixel_coordinates(left_kpts, w, h)
            right_pixels = self.detector_right.get_2d_pixel_coordinates(right_kpts, w, h)
            
            # Triangulate to get 3D points
            points_3d = self._triangulate_points(left_pixels, right_pixels)
            
            output['3d_keypoints'].append(points_3d)
            output['confidence'].append(match['confidence'])
            
            # Get handedness for better 3D gesture recognition
            handedness = "Unknown"
            if left_idx < len(stereo_results['left']['handedness']):
                handedness = stereo_results['left']['handedness'][left_idx]
            
            # Recognize gesture from 3D keypoints
            gesture_type, gesture_conf = self.gesture_recognizer.recognize_gestures_3d(
                points_3d, handedness=handedness, confidence_threshold=0.5
            )
            
            # Add the 3D gesture to output
            output['gestures'].append({
                'type': gesture_type,
                'confidence': gesture_conf,
                'name': self.gesture_recognizer.get_gesture_name(gesture_type),
                'view': 'stereo',  # This is from stereo match
                'is_3d': True,     # Flag that this is from 3D matching
                'hand_idx': left_idx,  # Store the hand index for reference
                'handedness': handedness,  # Store handedness
            })
            
            # Log 3D tracking result
            gesture_value = gesture_type.value if hasattr(gesture_type, 'value') else str(gesture_type)
            logger.info(f"HandPose 3d_tracking: match_confidence={match['confidence']:.2f}, handedness={handedness}, gesture={gesture_value}")
        
        # Add 2D gestures for hands that weren't matched in 3D
        # Filter to keep only highest confidence gesture per hand
        unmatched_2d_gestures = []
        for gesture in all_2d_gestures:
            camera = gesture['camera']
            hand_idx = gesture['hand_idx']
            handedness = gesture['handedness']
            
            # Skip hands that were matched in 3D
            if (camera == 'left' and hand_idx in matched_left_indices) or \
               (camera == 'right' and hand_idx in matched_right_indices):
                continue
                
            # Add this as an unmatched gesture
            unmatched_2d_gestures.append(gesture)
            
            # Log 2D tracking
            gesture_value = gesture['type'].value if hasattr(gesture['type'], 'value') else str(gesture['type'])
            logger.info(f"HandPose 2d_tracking: camera={camera}, handedness={handedness}, gesture={gesture_value}, confidence={gesture['confidence']:.2f}")
        
        # Sort unmatched gestures by confidence
        unmatched_2d_gestures.sort(key=lambda g: g['confidence'], reverse=True)
        
        # Add unmatched 2D gestures to output (prioritize 3D)
        output['gestures'].extend(unmatched_2d_gestures)
        
        # If we have no 3D gestures but 2D ones, prioritize left camera gestures (front-facing)
        if len(output['3d_keypoints']) == 0 and len(output['gestures']) > 0:
            # Sort gestures prioritizing left camera with higher boost for single-hand scenarios
            left_boost = 1.3 if prioritize_left_camera else 1.2
            output['gestures'].sort(
                key=lambda g: (g['confidence'] * (left_boost if g['camera'] == 'left' else 1.0)), 
                reverse=True
            )
        
        # Process dynamic gestures if the recognizer is available
        if self.dynamic_recognizer is not None and self.dynamic_recognizer.models_available:
            # Process left image for dynamic gestures
            if left_image is not None:
                dynamic_left_results = self.dynamic_recognizer.process_frame(left_image)
                
                # Add any detected dynamic actions to output
                for action in dynamic_left_results.get('actions', []):
                    output['dynamic_gestures'].append({
                        'type': action['type'],
                        'name': self.dynamic_recognizer.get_action_name(action['type']),
                        'bbox': action.get('bbox'),
                        'camera': 'left',
                        'confidence': 0.9,  # Dynamic gestures typically have high confidence
                        'timestamp': time.time()
                    })
                    logger.info(f"Dynamic gesture detected from left camera: {self.dynamic_recognizer.get_action_name(action['type'])}")
            
            # Process right image for dynamic gestures if needed
            # For efficiency, we may only process one camera for dynamic gestures
            # since they are computationally intensive
            if right_image is not None and len(output['dynamic_gestures']) == 0:
                dynamic_right_results = self.dynamic_recognizer.process_frame(right_image)
                
                # Add any detected dynamic actions to output
                for action in dynamic_right_results.get('actions', []):
                    output['dynamic_gestures'].append({
                        'type': action['type'],
                        'name': self.dynamic_recognizer.get_action_name(action['type']),
                        'bbox': action.get('bbox'),
                        'camera': 'right',
                        'confidence': 0.9,  # Dynamic gestures typically have high confidence
                        'timestamp': time.time()
                    })
                    logger.info(f"Dynamic gesture detected from right camera: {self.dynamic_recognizer.get_action_name(action['type'])}")
        
        # Store results for history tracking
        self.tracked_hands.append(output)
        if len(self.tracked_hands) > self.max_history:
            self.tracked_hands.pop(0)
        
        self.frame_count += 1
        
        return output
    
    def _match_stereo_hands(self, stereo_results: Dict) -> List[Dict]:
        """
        Match detected hands between left and right images.
        
        Args:
            stereo_results: Detection results from both cameras
        
        Returns:
            List of matched hand pairs
        """
        left_hands = stereo_results['left']['keypoints']
        right_hands = stereo_results['right']['keypoints']
        
        matches = []
        
        # Get confidence values if available
        left_confidences = stereo_results['left'].get('confidence', [1.0] * len(left_hands))
        right_confidences = stereo_results['right'].get('confidence', [1.0] * len(right_hands))
        
        # *** ENHANCED DUPLICATE AND FALSE POSITIVE DETECTION ***
        # We now handle most filtering in the detector, but we'll do additional stereo-specific filtering here
        # Get the already filtered hands from each camera
        left_hand_centers = []
        right_hand_centers = []
        valid_left_indices = []
        valid_right_indices = []
        
        # Minimum confidence threshold for hands to consider in stereo matching
        # Higher threshold for stereo matching compared to single-view detection
        min_confidence_threshold = 0.5
        
        # Process left hands - enhanced filtering for duplicates and false positives
        for i, left_hand in enumerate(left_hands):
            # Check confidence if available (skip low-confidence detections that might be false positives)
            if i < len(left_confidences) and left_confidences[i] < min_confidence_threshold:
                logger.debug(f"Left camera filtering low confidence hand: {i} (confidence={left_confidences[i]:.3f})")
                continue
                
            # Calculate hand center
            wrist_pos = left_hand[0]  # Wrist is landmark 0
            center_x = np.mean(left_hand[:, 0])
            center_y = np.mean(left_hand[:, 1])
            center = np.array([center_x, center_y])
            
            # Calculate hand size (for adaptive thresholding)
            hand_width = np.max(left_hand[:, 0]) - np.min(left_hand[:, 0])
            hand_height = np.max(left_hand[:, 1]) - np.min(left_hand[:, 1])
            hand_size = max(hand_width, hand_height)
            
            # Skip hands that are extremely small or large (likely false positives)
            if hand_size < 0.02 or hand_size > 0.95:  # Normalized coordinates are 0-1
                logger.debug(f"Left camera filtering abnormal sized hand: {i} (size={hand_size:.3f})")
                continue
                
            # Check hand shape validity (wrist should be at the bottom/edge of hand)
            finger_tips_y = [left_hand[4][1], left_hand[8][1], left_hand[12][1], left_hand[16][1], left_hand[20][1]]
            wrist_y = wrist_pos[1]
            y_diff = abs(np.mean(finger_tips_y) - wrist_y)
            if y_diff < 0.05:  # Fingers and wrist at same height - likely false positive
                logger.debug(f"Left camera filtering anatomically incorrect hand: {i} (y_diff={y_diff:.3f})")
                continue
            
            # Check if this hand is too close to a previously detected hand
            is_duplicate = False
            for prev_center in left_hand_centers:
                distance = np.linalg.norm(center - prev_center)
                # Use a stricter threshold based on hand size to filter duplicates better
                duplicate_threshold = max(0.05, hand_size * 0.3)  # Stricter threshold than before
                if distance < duplicate_threshold:
                    is_duplicate = True
                    logger.debug(f"Left camera duplicate hand detected: {i} close to another hand (distance={distance:.3f})")
                    break
            
            if not is_duplicate:
                left_hand_centers.append(center)
                valid_left_indices.append(i)
        
        # Process right hands - enhanced filtering for duplicates and false positives
        for j, right_hand in enumerate(right_hands):
            # Check confidence if available (skip low-confidence detections that might be false positives)
            if j < len(right_confidences) and right_confidences[j] < min_confidence_threshold:
                logger.debug(f"Right camera filtering low confidence hand: {j} (confidence={right_confidences[j]:.3f})")
                continue
                
            # Calculate hand center
            wrist_pos = right_hand[0]  # Wrist is landmark 0
            center_x = np.mean(right_hand[:, 0])
            center_y = np.mean(right_hand[:, 1]) 
            center = np.array([center_x, center_y])
            
            # Calculate hand size (for adaptive thresholding)
            hand_width = np.max(right_hand[:, 0]) - np.min(right_hand[:, 0])
            hand_height = np.max(right_hand[:, 1]) - np.min(right_hand[:, 1])
            hand_size = max(hand_width, hand_height)
            
            # Skip hands that are extremely small or large (likely false positives)
            if hand_size < 0.02 or hand_size > 0.95:  # Normalized coordinates are 0-1
                logger.debug(f"Right camera filtering abnormal sized hand: {j} (size={hand_size:.3f})")
                continue
                
            # Check hand shape validity (wrist should be at the bottom/edge of hand)
            finger_tips_y = [right_hand[4][1], right_hand[8][1], right_hand[12][1], right_hand[16][1], right_hand[20][1]]
            wrist_y = wrist_pos[1]
            y_diff = abs(np.mean(finger_tips_y) - wrist_y)
            if y_diff < 0.05:  # Fingers and wrist at same height - likely false positive
                logger.debug(f"Right camera filtering anatomically incorrect hand: {j} (y_diff={y_diff:.3f})")
                continue
            
            # Check if this hand is too close to a previously detected hand
            is_duplicate = False
            for prev_center in right_hand_centers:
                distance = np.linalg.norm(center - prev_center)
                # Use a stricter threshold based on hand size to filter duplicates better
                duplicate_threshold = max(0.05, hand_size * 0.3)  # Stricter threshold than before
                if distance < duplicate_threshold:
                    is_duplicate = True
                    logger.debug(f"Right camera duplicate hand detected: {j} close to another hand (distance={distance:.3f})")
                    break
            
            if not is_duplicate:
                right_hand_centers.append(center)
                valid_right_indices.append(j)
        
        # Log how many valid hands we have after deduplication
        if len(valid_left_indices) < len(left_hands) or len(valid_right_indices) < len(right_hands):
            logger.info(f"Filtered duplicates: Left {len(left_hands)}->{len(valid_left_indices)}, Right {len(right_hands)}->{len(valid_right_indices)}")
        
        # *** IMPROVED MATCHING ALGORITHM ***
        # Try to match each valid left hand with a right hand
        used_right_indices = set()
        
        # If we have valid detections in both cameras
        if valid_left_indices and valid_right_indices:
            # For each left hand, find the best matching right hand
            for idx, i in enumerate(valid_left_indices):
                left_hand = left_hands[i]
                best_match = None
                best_score = float('inf')
                
                # Get key points and features for left hand
                left_wrist = left_hand[0]  # Wrist position
                left_center_x = np.mean(left_hand[:, 0])
                left_center_y = np.mean(left_hand[:, 1])
                left_center = np.array([left_center_x, left_center_y])
                
                # Get hand orientation (angle of index-pinky line relative to horizontal)
                left_index_base = left_hand[5]  # Index finger MCP
                left_pinky_base = left_hand[17]  # Pinky MCP
                left_orientation_vec = left_pinky_base - left_index_base
                left_orientation = np.arctan2(left_orientation_vec[1], left_orientation_vec[0])
                
                # Get handedness info
                left_handedness = None
                if (stereo_results['left'].get('handedness') and 
                    len(stereo_results['left']['handedness']) > i):
                    left_handedness = stereo_results['left']['handedness'][i]
                
                # Try to match with each right hand
                for j_idx, j in enumerate(valid_right_indices):
                    if j in used_right_indices:
                        continue  # Skip already matched hands
                    
                    right_hand = right_hands[j]
                    
                    # Get key points and features for right hand
                    right_wrist = right_hand[0]  # Wrist position
                    right_center_x = np.mean(right_hand[:, 0])
                    right_center_y = np.mean(right_hand[:, 1])
                    right_center = np.array([right_center_x, right_center_y])
                    
                    # Get hand orientation
                    right_index_base = right_hand[5]  # Index finger MCP
                    right_pinky_base = right_hand[17]  # Pinky MCP
                    right_orientation_vec = right_pinky_base - right_index_base
                    right_orientation = np.arctan2(right_orientation_vec[1], right_orientation_vec[0])
                    
                    # Get handedness info
                    right_handedness = None
                    if (stereo_results['right'].get('handedness') and 
                        len(stereo_results['right']['handedness']) > j):
                        right_handedness = stereo_results['right']['handedness'][j]
                    
                    # *** IMPROVED MATCHING METRICS ***
                    # 1. Y-coordinate similarity (most important in stereo)
                    y_diff = abs(left_center_y - right_center_y)
                    # More stringent y-coordinate matching (critical for stereo pairs)
                    y_threshold = 0.15  # Maximum allowed vertical difference (normalized coords)
                    if y_diff > y_threshold:
                        # Y-difference too large - these cannot be the same hand
                        y_score = 10.0  # Large penalty that will exceed the threshold
                    else:
                        y_score = y_diff * 3.0  # Weighted even more heavily
                    
                    # 2. Hand orientation similarity
                    orientation_diff = min(abs(left_orientation - right_orientation), 
                                        np.pi * 2 - abs(left_orientation - right_orientation))
                    orientation_score = orientation_diff / np.pi  # Normalize to 0-1
                    
                    # 3. X-coordinate factor (accounting for stereo disparity)
                    # In a properly configured stereo setup, the same point will have 
                    # a larger X coordinate in the left image than in the right image
                    # Calculate disparity (should be positive for correctly paired stereo points)
                    disparity = left_center_x - right_center_x
                    
                    # Disparity must be positive and within reasonable range
                    if disparity <= 0:
                        x_score = 1.0  # Strong penalty - physically impossible in standard stereo
                    elif disparity > 0.5:  # Too much disparity - unlikely to be the same hand
                        x_score = 0.8
                    else:
                        # Valid disparity range - normalize the score from 0 to 0.3
                        # Lower disparity (hand further from camera) penalized slightly
                        x_score = 0.3 * (1.0 - disparity * 2)
                    
                    # 4. Handedness matching - critical for correct pairing
                    handedness_score = 0.0
                    if left_handedness and right_handedness:
                        # In stereo, we expect opposite handedness due to mirroring
                        if (left_handedness == 'Left' and right_handedness == 'Right') or \
                           (left_handedness == 'Right' and right_handedness == 'Left'):
                            # Expected case - no penalty
                            handedness_score = 0.0
                        elif left_handedness == right_handedness:
                            # Same handedness - severe penalty (virtually impossible in stereo)
                            handedness_score = 1.0  # Effective rejection
                        else:
                            # Unknown comparison
                            handedness_score = 0.5
                    
                    # 5. Hand size similarity (real hands should have similar apparent size in both views)
                    left_hand_size = max(
                        np.max(left_hand[:, 0]) - np.min(left_hand[:, 0]),
                        np.max(left_hand[:, 1]) - np.min(left_hand[:, 1])
                    )
                    right_hand_size = max(
                        np.max(right_hand[:, 0]) - np.min(right_hand[:, 0]),
                        np.max(right_hand[:, 1]) - np.min(right_hand[:, 1])
                    )
                    
                    # Size ratio (larger / smaller) - should be close to 1.0
                    size_ratio = max(left_hand_size, right_hand_size) / max(0.001, min(left_hand_size, right_hand_size))
                    # Penalize size differences exceeding 30%
                    size_score = max(0.0, size_ratio - 1.3) * 2.0
                    
                    # 6. Overall similarity of hand shape
                    # Compute similarity based on relative positions of key landmarks
                    shape_similarity = self._compute_hand_shape_similarity(left_hand, right_hand)
                    shape_score = 1.0 - shape_similarity  # Convert to a cost (0=perfect match)
                    
                    # Combine all scores with weights - updated weights and added size similarity
                    score = (y_score * 0.4) + (orientation_score * 0.1) + \
                           (x_score * 0.15) + (handedness_score * 0.2) + \
                           (size_score * 0.05) + (shape_score * 0.1)
                    
                    # Log detailed match scores for debugging
                    logger.debug(f"Match {i}:{j} score={score:.2f} (y={y_score:.2f}, orient={orientation_score:.2f}, x={x_score:.2f}, hand={handedness_score:.2f}, shape={shape_score:.2f})")
                    
                    if score < best_score:
                        best_score = score
                        best_match = j
                
                # Add match if found and score is reasonable
                # More strict threshold to avoid incorrect matches
                threshold = 0.35  # Lower threshold for higher quality matches
                
                if best_match is not None and best_score < threshold:
                    confidence = max(0.0, 1.0 - (best_score / threshold))
                    
                    # Final check - verify handedness compatibility to avoid matching left hand to left hand
                    # or right hand to right hand if detected incorrectly
                    left_hand_type = stereo_results['left']['handedness'][i] if i < len(stereo_results['left'].get('handedness', [])) else None
                    right_hand_type = stereo_results['right']['handedness'][best_match] if best_match < len(stereo_results['right'].get('handedness', [])) else None
                    
                    # Strict handedness check - MUST have opposite handedness between views
                    if left_hand_type and right_hand_type:
                        # We're looking for opposite handedness due to mirroring in stereo
                        if (left_hand_type == 'Left' and right_hand_type == 'Right') or \
                           (left_hand_type == 'Right' and right_hand_type == 'Left'):
                            # This is the expected case - proceed with match
                            logger.debug(f"Good handedness match: left={left_hand_type}, right={right_hand_type}")
                        else:
                            # Inconsistent handedness - probably a false match
                            logger.debug(f"Rejecting match with inconsistent handedness: left={left_hand_type}, right={right_hand_type}")
                            continue
                    
                    # Create the match
                    matches.append({
                        'indices': (i, best_match),
                        'confidence': confidence,
                        'handedness': left_handedness,
                        'score': best_score  # Store score for later filtering
                    })
                    used_right_indices.add(best_match)
                    
                    # Log the match
                    logger.info(f"Matched hands: left={i} ({left_handedness}) with right={best_match} ({right_handedness}) score={best_score:.2f}")
                else:
                    logger.debug(f"No match found for left hand {i} (best score {best_score:.2f} exceeds threshold {threshold})")
            
            # Sort matches by confidence (highest first)
            matches.sort(key=lambda m: m['confidence'], reverse=True)
            
            # Take only the best match if we have multiple matches
            # This is a critical fix for the duplicate hand issue
            if len(matches) > 1:
                best_match = matches[0]
                logger.info(f"Multiple matches found ({len(matches)}), keeping only the best match with score {best_match['score']:.2f}")
                matches = [best_match]
            
            # If no matches found but we have hands in both images, try a forced match for single hand case
            # BUT only if handedness is compatible and enough confidence
            if len(matches) == 0 and len(valid_left_indices) == 1 and len(valid_right_indices) == 1:
                # For the case of a single hand in each view that wasn't matched due to threshold
                i = valid_left_indices[0]
                j = valid_right_indices[0]
                
                # Get handedness and confidence for both hands
                left_handedness = None
                right_handedness = None
                left_confidence = 0
                right_confidence = 0
                
                if stereo_results['left'].get('handedness') and len(stereo_results['left']['handedness']) > i:
                    left_handedness = stereo_results['left']['handedness'][i]
                    if 'confidence' in stereo_results['left'] and len(stereo_results['left']['confidence']) > i:
                        left_confidence = stereo_results['left']['confidence'][i]
                
                if stereo_results['right'].get('handedness') and len(stereo_results['right']['handedness']) > j:
                    right_handedness = stereo_results['right']['handedness'][j]
                    if 'confidence' in stereo_results['right'] and len(stereo_results['right']['confidence']) > j:
                        right_confidence = stereo_results['right']['confidence'][j]
                
                # Vertical position check for single-hand case
                left_hand = left_hands[i]
                right_hand = right_hands[j]
                left_center_y = np.mean(left_hand[:, 1])
                right_center_y = np.mean(right_hand[:, 1])
                y_diff = abs(left_center_y - right_center_y)
                
                # Calculate disparity (should be positive)
                left_center_x = np.mean(left_hand[:, 0])  
                right_center_x = np.mean(right_hand[:, 0])
                disparity = left_center_x - right_center_x
                
                # Only force match if all conditions are met:
                # 1. Handedness is compatible (opposite)
                # 2. Y-position is similar
                # 3. Disparity is positive and reasonable
                # 4. Both hands have decent confidence
                
                handedness_compatible = (
                    (left_handedness is None or right_handedness is None) or
                    (left_handedness == 'Left' and right_handedness == 'Right') or
                    (left_handedness == 'Right' and right_handedness == 'Left')
                )
                
                y_position_compatible = y_diff < 0.15
                disparity_compatible = disparity > 0 and disparity < 0.5
                good_confidence = left_confidence > 0.7 and right_confidence > 0.7
                
                # Perform a strict forced match
                if handedness_compatible and y_position_compatible and disparity_compatible and good_confidence:
                    logger.info(f"Forced match for single hand case: left={i} ({left_handedness}), right={j} ({right_handedness})")
                    matches.append({
                        'indices': (i, j),
                        'confidence': 0.6,  # Reasonable confidence for a clean forced match
                        'handedness': left_handedness
                    })
                else:
                    if not handedness_compatible:
                        logger.info(f"Skipping forced match due to incompatible handedness: left={left_handedness}, right={right_handedness}")
                    elif not y_position_compatible:
                        logger.info(f"Skipping forced match due to incompatible Y positions: diff={y_diff:.3f}")
                    elif not disparity_compatible:
                        logger.info(f"Skipping forced match due to incompatible disparity: {disparity:.3f}")
                    elif not good_confidence:
                        logger.info(f"Skipping forced match due to low confidence: left={left_confidence:.2f}, right={right_confidence:.2f}")
                    else:
                        logger.info(f"Skipping forced match due to unknown incompatibility")
        
        return matches
        
    def _compute_hand_shape_similarity(self, hand1: np.ndarray, hand2: np.ndarray) -> float:
        """
        Compute similarity between two hand shapes based on landmark configurations.
        
        Args:
            hand1: First hand landmarks (21x3 array)
            hand2: Second hand landmarks (21x3 array)
            
        Returns:
            Similarity score (0-1, where 1 is perfect match)
        """
        # Normalize both hands to origin at wrist and unit scale
        def normalize_hand(hand):
            # Center at wrist
            centered = hand - hand[0]
            # Scale to unit size
            scale = np.max(np.linalg.norm(centered, axis=1))
            if scale > 0:
                return centered / scale
            return centered
        
        norm_hand1 = normalize_hand(hand1)
        norm_hand2 = normalize_hand(hand2)
        
        # Calculate similarity based on corresponding landmark distances
        total_dist = 0
        for i in range(21):
            dist = np.linalg.norm(norm_hand1[i, :2] - norm_hand2[i, :2])
            total_dist += dist
        
        # Average distance per landmark, normalized to 0-1 similarity
        avg_dist = total_dist / 21
        similarity = max(0, 1.0 - (avg_dist / 2.0))  # Threshold at a distance of 2.0
        
        return similarity
    
    def detect_hands_stereo(self, 
                           left_image: np.ndarray, 
                           right_image: np.ndarray) -> Dict:
        """
        Detect hands in stereo images for 3D reconstruction using camera-specific detectors.
        Each camera's detector is configured with the appropriate mirror mode setting.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
        
        Returns:
            Dictionary containing stereo detection results
        """
        # Detect hands in both images using camera-specific detectors
        # Left camera typically has mirror mode on (front-facing view)
        left_results = self.detector_left.detect_hands(left_image)
        
        # Right camera typically has mirror mode off (world-facing view)
        right_results = self.detector_right.detect_hands(right_image)
        
        # Combine results
        stereo_results = {
            'left': left_results,
            'right': right_results,
            'timestamp': time.time()
        }
        
        # Log stereo detection
        left_count = len(left_results['keypoints'])
        right_count = len(right_results['keypoints'])
        
        # Log more detailed handedness information for debugging
        left_handedness = ", ".join([h for h in left_results.get('handedness', [])]) if left_count > 0 else ""
        right_handedness = ", ".join([h for h in right_results.get('handedness', [])]) if right_count > 0 else ""
        
        if left_count > 0 or right_count > 0:
            logger.info(f"HandPose stereo_detection: left_count={left_count} ({left_handedness}), right_count={right_count} ({right_handedness})")
        else:
            logger.debug(f"HandPose stereo_detection: left_count={left_count}, right_count={right_count}")
        
        return stereo_results
    
    def _triangulate_points(self, 
                           left_points: np.ndarray, 
                           right_points: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from corresponding 2D points.
        
        Args:
            left_points: 2D points in left image (Nx2 or Nx3)
            right_points: 2D points in right image (Nx2 or Nx3)
        
        Returns:
            3D points (Nx3)
        """
        # Ensure points are numpy arrays
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        # Handle different input shapes
        if left_points.ndim == 1:
            # Flatten array - reshape to Nx2 or Nx3 depending on size
            num_coords = len(left_points)
            if num_coords % 3 == 0:
                left_points = left_points.reshape(-1, 3)
            elif num_coords % 2 == 0:
                left_points = left_points.reshape(-1, 2)
            else:
                raise ValueError(f"Cannot reshape array of size {num_coords} into valid coordinates")
        
        if right_points.ndim == 1:
            num_coords = len(right_points)
            if num_coords % 3 == 0:
                right_points = right_points.reshape(-1, 3)
            elif num_coords % 2 == 0:
                right_points = right_points.reshape(-1, 2)
            else:
                raise ValueError(f"Cannot reshape array of size {num_coords} into valid coordinates")
            
        # Ensure points are 2D - take only x,y coordinates
        if left_points.shape[1] > 2:
            left_points = left_points[:, :2]
        if right_points.shape[1] > 2:
            right_points = right_points[:, :2]
            
        # Convert to format required by cv2.triangulatePoints (2xN)
        left_points = left_points.T
        right_points = right_points.T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, left_points, right_points)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        return points_3d
    
    def get_dynamic_gestures(self, output: Dict) -> List[Dict]:
        """
        Extract dynamic gestures from tracking output.
        
        Args:
            output: Output from track_hands_3d
            
        Returns:
            List of dynamic gesture dictionaries
        """
        return output.get('dynamic_gestures', [])
    
    def visualize_3d_hands(self, 
                          output: Dict,
                          scale: float = 1.0) -> Optional[np.ndarray]:
        """
        Create a 3D visualization of tracked hands.
        
        Args:
            output: Output from track_hands_3d
            scale: Scale factor for visualization
        
        Returns:
            Visualization image or None
        """
        if not output['3d_keypoints']:
            return None
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot each hand
            for i, points_3d in enumerate(output['3d_keypoints']):
                # Scale points
                points_3d = points_3d * scale
                
                # Get gesture info if available
                gesture_label = f'Hand {i}'
                if i < len(output.get('gestures', [])):
                    gesture_info = output['gestures'][i]
                    if gesture_info['type'] != GestureType.UNKNOWN:
                        gesture_label = f"Hand {i}: {gesture_info['name']} ({gesture_info['confidence']:.2f})"
                
                # Plot keypoints
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                          c='red', s=50, label=gesture_label)
                
                # Draw hand connections
                connections = [
                    # Thumb
                    [0, 1], [1, 2], [2, 3], [3, 4],
                    # Index finger
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    # Middle finger
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    # Ring finger
                    [0, 13], [13, 14], [14, 15], [15, 16],
                    # Pinky
                    [0, 17], [17, 18], [18, 19], [19, 20],
                    # Palm
                    [5, 9], [9, 13], [13, 17]
                ]
                
                for connection in connections:
                    pts = points_3d[connection]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', alpha=0.7)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            # Convert to image
            fig.canvas.draw()
            # Try the modern API first, fallback to old API
            try:
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape((fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4))
                img = img[:, :, :3]  # Drop alpha channel
            except AttributeError:
                # Fallback for older matplotlib versions
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return img
            
        except ImportError:
            logger.warning("Matplotlib not available for 3D visualization")
            return None
    
    def save_tracking_data(self, filename: str):
        """Save tracked hand data to file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        save_data = {
            'frames': [],
            'calibration': {
                'P1': self.P1.tolist() if hasattr(self, 'P1') else None,
                'P2': self.P2.tolist() if hasattr(self, 'P2') else None,
            }
        }
        
        for frame_data in self.tracked_hands:
            frame_entry = {
                'frame': frame_data['frame'],
                'timestamp': frame_data['timestamp'],
                '3d_keypoints': [kpts.tolist() for kpts in frame_data['3d_keypoints']],
                '2d_left': [kpts.tolist() for kpts in frame_data['2d_left']],
                '2d_right': [kpts.tolist() for kpts in frame_data['2d_right']],
                'confidence': frame_data['confidence']
            }
            save_data['frames'].append(frame_entry)
        
        # Save as JSON
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=2)
        else:
            # Save as numpy
            np.save(output_path, save_data)
        
        logger.info(f"Saved {len(self.tracked_hands)} frames of hand tracking data to {filename}")
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'detector_left'):
            logger.info("Closing left detector resources")
            self.detector_left.close()
            
        if hasattr(self, 'detector_right'):
            logger.info("Closing right detector resources")
            self.detector_right.close()
            
        # Close dynamic gesture recognizer if available
        if hasattr(self, 'dynamic_recognizer') and self.dynamic_recognizer is not None:
            logger.info("Closing dynamic gesture recognizer resources")
            # No explicit close method needed for ONNX models, but we null the reference
            self.dynamic_recognizer = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()