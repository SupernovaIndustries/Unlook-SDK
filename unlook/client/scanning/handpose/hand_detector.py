"""Hand pose detection using MediaPipe for UnLook SDK.

Based on handpose3d by TemugeB: https://github.com/TemugeB/handpose3d
"""

import time
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not installed. Hand pose detection will be disabled.")

logger = logging.getLogger(__name__)


class HandDetector:
    """Real-time hand pose detection using MediaPipe."""
    
    def __init__(self, 
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 max_num_hands: int = 2):
        """
        Initialize the hand detector.
        
        Args:
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for hand detection. Install with: pip install mediapipe")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
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
        
    def detect_hands(self, image: np.ndarray) -> Dict:
        """
        Detect hands in an image and return 2D keypoints.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary containing detection results:
            - 'keypoints': List of hand keypoints (normalized coordinates)
            - 'world_keypoints': List of hand keypoints in world coordinates
            - 'handedness': List of handedness (left/right) for each detected hand
            - 'image': Annotated image with hand landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Initialize output
        output = {
            'keypoints': [],
            'world_keypoints': [],
            'handedness': [],
            'image': image.copy(),
            'timestamp': time.time()
        }
        
        # Process results
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get 2D keypoints (normalized)
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                output['keypoints'].append(np.array(keypoints))
                
                # Get handedness
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                    output['handedness'].append(handedness)
                else:
                    output['handedness'].append('Unknown')
                
                # Draw landmarks on image
                self.mp_drawing.draw_landmarks(
                    output['image'], 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
            # Get world coordinates if available
            if results.multi_hand_world_landmarks:
                for world_landmarks in results.multi_hand_world_landmarks:
                    world_keypoints = []
                    for landmark in world_landmarks.landmark:
                        world_keypoints.append([landmark.x, landmark.y, landmark.z])
                    output['world_keypoints'].append(np.array(world_keypoints))
        
        self.last_results = output
        self.last_processed_time = time.time()
        
        return output
    
    def detect_hands_stereo(self, 
                           left_image: np.ndarray, 
                           right_image: np.ndarray) -> Dict:
        """
        Detect hands in stereo images for 3D reconstruction.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
        
        Returns:
            Dictionary containing stereo detection results
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
                              color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw bounding box around detected hand.
        
        Args:
            image: Input image
            keypoints: Hand keypoints in pixel coordinates
            color: Box color (BGR)
        
        Returns:
            Image with bounding box
        """
        if len(keypoints) == 0:
            return image
        
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
        
        return image
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'hands') and self.hands:
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