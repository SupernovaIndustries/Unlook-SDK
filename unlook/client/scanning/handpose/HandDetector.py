"""Hand pose detection using MediaPipe for UnLook SDK.

This module provides robust hand detection and keypoint extraction using MediaPipe,
optimized for the UnLook Scanner's stereo camera setup.
"""

import time
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)


class HandDetector:
    """Real-time hand pose detection using MediaPipe without ML dependencies."""
    
    def __init__(self, 
                 detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.6,
                 max_num_hands: int = 2,
                 hand_mirror_mode: bool = False):  # False for world-facing cameras
        """
        Initialize the hand detector.
        
        Args:
            detection_confidence: Minimum confidence for hand detection (0.6 default)
            tracking_confidence: Minimum confidence for hand tracking (0.6 default)
            max_num_hands: Maximum number of hands to detect (2 default)
            hand_mirror_mode: Whether camera is in mirrored/selfie mode (False default)
                             True for selfie/front-facing cameras, False for world-facing
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe is required for hand detection")
            raise ImportError("MediaPipe is required for hand detection. Install with: pip install mediapipe")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        logger.info(f"Initializing MediaPipe Hands with confidence: {detection_confidence}/{tracking_confidence}")
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,  # Use more accurate model (0, 1 or 2)
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        self.hand_mirror_mode = hand_mirror_mode
        logger.info(f"Hand mirror mode: {hand_mirror_mode} (True for selfie cameras, False for world facing)")
        
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Hand landmarks (21 points per hand)
        self.num_landmarks = 21
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        # Store results
        self.last_results = None
        self.last_processed_time = 0

    def detect_hands(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect hands in an image and return 2D keypoints with improved filtering.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary containing detection results:
            - 'keypoints': List of hand keypoints (normalized coordinates)
            - 'world_keypoints': List of hand keypoints in world coordinates
            - 'handedness': List of handedness (left/right) for each detected hand
            - 'confidence': List of confidence scores for each hand
            - 'image': Annotated image with hand landmarks
            - 'timestamp': Time of detection
        """
        if image is None or image.size == 0:
            return {
                'keypoints': [],
                'world_keypoints': [],
                'handedness': [],
                'confidence': [],
                'image': None,
                'timestamp': time.time()
            }
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Initialize output
        output = {
            'keypoints': [],
            'world_keypoints': [],
            'handedness': [],
            'confidence': [],  # Added confidence scores
            'image': image.copy(),
            'timestamp': time.time()
        }
        
        # Process results
        if results.multi_hand_landmarks:
            # First pass - collect all detections with additional data
            detected_hands = []
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get 2D keypoints (normalized)
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                keypoints_array = np.array(keypoints)
                
                # Get handedness and confidence
                handedness = 'Unknown'
                handedness_confidence = 0.5  # Default confidence
                
                if results.multi_handedness and idx < len(results.multi_handedness):
                    # MediaPipe assumes selfie/mirrored view by default
                    # We need to swap the label if we're using world-facing camera
                    raw_handedness = results.multi_handedness[idx].classification[0].label
                    handedness_confidence = results.multi_handedness[idx].classification[0].score
                    
                    # If not in mirror mode (world-facing camera), swap the handedness
                    if not self.hand_mirror_mode:
                        handedness = "Right" if raw_handedness == "Left" else "Left"
                        logger.debug(f"Swapping handedness from {raw_handedness} to {handedness} (world-facing camera)")
                    else:
                        handedness = raw_handedness
                
                # Calculate hand metrics for filtering
                wrist_pos = keypoints_array[0]  # Wrist is landmark 0
                center_x = np.mean(keypoints_array[:, 0])
                center_y = np.mean(keypoints_array[:, 1])
                center = np.array([center_x, center_y])
                
                # Calculate hand size
                hand_width = np.max(keypoints_array[:, 0]) - np.min(keypoints_array[:, 0])
                hand_height = np.max(keypoints_array[:, 1]) - np.min(keypoints_array[:, 1])
                hand_size = max(hand_width, hand_height)
                
                # Calculate hand volume - useful for confidence scaling for single hand detection
                hand_depth = np.max(keypoints_array[:, 2]) - np.min(keypoints_array[:, 2]) if keypoints_array.shape[1] > 2 else 0
                hand_volume = hand_width * hand_height * max(0.01, hand_depth)
                
                # Anatomical validation (finger tips should be at different height than wrist)
                finger_tips_y = [keypoints_array[4][1], keypoints_array[8][1], 
                                keypoints_array[12][1], keypoints_array[16][1], 
                                keypoints_array[20][1]]
                wrist_y = wrist_pos[1]
                y_diff = abs(np.mean(finger_tips_y) - wrist_y)
                
                # Store hand detection with metadata
                detected_hands.append({
                    'keypoints': keypoints_array,
                    'handedness': handedness,
                    'confidence': handedness_confidence,
                    'center': center,
                    'size': hand_size,
                    'y_diff': y_diff,
                    'landmarks': hand_landmarks,
                    'idx': idx  # Original index in the results
                })
            
            # Second pass - filter and deduplicate hands
            # Sort by confidence (highest first)
            detected_hands.sort(key=lambda h: h['confidence'], reverse=True)
            
            # Filter out low quality detections
            filtered_hands = []
            
            for hand in detected_hands:
                # Size-based filtering (too small or too large is invalid)
                # More lenient minimum threshold (0.01) to support single-hand use cases
                if hand['size'] < 0.01 or hand['size'] > 0.95:
                    logger.debug(f"Filtering abnormal sized hand: {hand['handedness']} (size={hand['size']:.3f})")
                    continue
                
                # Shape-based filtering (anatomically incorrect hands)
                if hand['y_diff'] < 0.05:
                    logger.debug(f"Filtering anatomically incorrect hand: {hand['handedness']} (y_diff={hand['y_diff']:.3f})")
                    continue
                    
                # Remove duplicates (hands that are too close to each other)
                is_duplicate = False
                for other_hand in filtered_hands:
                    distance = np.linalg.norm(hand['center'] - other_hand['center'])
                    duplicate_threshold = max(0.05, max(hand['size'], other_hand['size']) * 0.35)
                    
                    if distance < duplicate_threshold:
                        is_duplicate = True
                        logger.debug(f"Filtering duplicate hand: {hand['handedness']} (distance={distance:.3f})")
                        break
                        
                if not is_duplicate:
                    filtered_hands.append(hand)
            
            # Only keep at most max_num_hands (highest confidence)
            filtered_hands = filtered_hands[:self.max_num_hands]
            
            # Populate output with filtered hands
            for hand in filtered_hands:
                output['keypoints'].append(hand['keypoints'])
                output['handedness'].append(hand['handedness'])
                output['confidence'].append(hand['confidence'])
                
                # Draw landmarks on image
                self.mp_drawing.draw_landmarks(
                    output['image'], 
                    hand['landmarks'], 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Get world coordinates if available
            if results.multi_hand_world_landmarks:
                # Create a mapping from original index to filtered index
                idx_mapping = {hand['idx']: i for i, hand in enumerate(filtered_hands)}
                
                for idx, world_landmarks in enumerate(results.multi_hand_world_landmarks):
                    # Only include world landmarks for hands that weren't filtered out
                    if idx in idx_mapping:
                        world_keypoints = []
                        for landmark in world_landmarks.landmark:
                            world_keypoints.append([landmark.x, landmark.y, landmark.z])
                        output['world_keypoints'].append(np.array(world_keypoints))
        
        self.last_results = output
        self.last_processed_time = time.time()
        
        # Log detection results
        hands_count = len(output['keypoints'])
        if hands_count > 0:
            logger.info(f"HandPose detection: count={hands_count}, handedness={output['handedness']}, confidence={[f'{c:.2f}' for c in output['confidence']]}")
        else:
            logger.debug(f"HandPose detection: count={hands_count}")
        
        return output
    
    def detect_hands_stereo(self, 
                           left_image: np.ndarray, 
                           right_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect hands in stereo images for 3D reconstruction.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
        
        Returns:
            Dictionary containing stereo detection results:
            - 'left': Left camera detection results
            - 'right': Right camera detection results
            - 'timestamp': Time of detection
        """
        # Detect hands in both images
        left_results = self.detect_hands(left_image)
        right_results = self.detect_hands(right_image)
        
        # Combine results
        stereo_results = {
            'left': left_results,
            'right': right_results,
            'timestamp': time.time()
        }
        
        # Log stereo detection
        left_count = len(left_results['keypoints'])
        right_count = len(right_results['keypoints'])
        if left_count > 0 or right_count > 0:
            logger.info(f"HandPose stereo_detection: left_count={left_count}, right_count={right_count}")
        else:
            logger.debug(f"HandPose stereo_detection: left_count={left_count}, right_count={right_count}")
        
        return stereo_results
    
    def get_2d_pixel_coordinates(self, 
                                keypoints: np.ndarray, 
                                image_width: int, 
                                image_height: int) -> np.ndarray:
        """
        Convert normalized keypoints to pixel coordinates.
        
        Args:
            keypoints: Normalized keypoints from detection
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
        
        Returns:
            Keypoints in pixel coordinates
        """
        pixel_coords = keypoints.copy()
        pixel_coords[:, 0] *= image_width
        pixel_coords[:, 1] *= image_height
        
        return pixel_coords
    
    def draw_hand_bounding_box(self, 
                              image: np.ndarray, 
                              keypoints: np.ndarray,
                              color: Tuple[int, int, int] = (0, 255, 0),
                              label: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding box around detected hand.
        
        Args:
            image: Input image
            keypoints: Hand keypoints in pixel coordinates
            color: Box color (BGR)
            label: Optional label text
        
        Returns:
            Image with bounding box
        """
        if len(keypoints) == 0:
            return image
        
        if image is None:
            return None
            
        # Get bounding box from keypoints
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        # Add some padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Draw box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add label if provided
        if label is not None:
            cv2.putText(image, label, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'hands') and self.hands:
            logger.info("Closing HandDetector resources")
            try:
                self.hands.close()
            except ValueError:
                # MediaPipe sometimes raises this error on double-close
                pass
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            # Ignore cleanup errors during deletion
            pass