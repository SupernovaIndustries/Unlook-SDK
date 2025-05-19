"""Body pose detection using MediaPipe for UnLook SDK.

Based on bodypose3d by TemugeB: https://github.com/TemugeB/bodypose3d
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
    logging.warning("MediaPipe not installed. Body pose detection will be disabled.")

logger = logging.getLogger(__name__)


class BodyDetector:
    """Real-time body pose detection using MediaPipe."""
    
    def __init__(self, 
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True):
        """
        Initialize the body detector.
        
        Args:
            detection_confidence: Minimum confidence for pose detection
            tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
            smooth_landmarks: Whether to smooth landmarks across frames
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for body detection. Install with: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_complexity = model_complexity
        
        # Body landmarks (33 points)
        self.num_landmarks = 33
        self.landmark_names = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
        
        # Define body connections for drawing
        self.connections = self.mp_pose.POSE_CONNECTIONS
        
        # Store results
        self.last_results = None
        self.last_processed_time = 0
        
    def detect_pose(self, image: np.ndarray) -> Dict:
        """
        Detect body pose in an image and return keypoints.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary containing detection results:
            - 'keypoints': Body keypoints (normalized coordinates + visibility)
            - 'world_keypoints': Body keypoints in world coordinates  
            - 'image': Annotated image with pose landmarks
            - 'visibility': Visibility scores for each keypoint
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Initialize output
        output = {
            'keypoints': [],
            'world_keypoints': [],
            'visibility': [],
            'image': image.copy(),
            'timestamp': time.time()
        }
        
        # Process results
        if results.pose_landmarks:
            # Get 2D keypoints (normalized)
            keypoints = []
            visibility = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
                visibility.append(landmark.visibility)
            
            output['keypoints'] = np.array(keypoints)
            output['visibility'] = np.array(visibility)
            
            # Get world coordinates if available
            if results.pose_world_landmarks:
                world_keypoints = []
                for landmark in results.pose_world_landmarks.landmark:
                    world_keypoints.append([landmark.x, landmark.y, landmark.z])
                output['world_keypoints'] = np.array(world_keypoints)
            
            # Draw landmarks on image
            self.mp_drawing.draw_landmarks(
                output['image'],
                results.pose_landmarks,
                self.connections,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        self.last_results = output
        self.last_processed_time = time.time()
        
        return output
    
    def detect_pose_stereo(self, 
                          left_image: np.ndarray, 
                          right_image: np.ndarray) -> Dict:
        """
        Detect body pose in stereo images for 3D reconstruction.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
        
        Returns:
            Dictionary containing stereo detection results
        """
        # Detect pose in both images
        left_results = self.detect_pose(left_image)
        right_results = self.detect_pose(right_image)
        
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
    
    def filter_visible_keypoints(self, 
                                keypoints: np.ndarray, 
                                visibility: np.ndarray,
                                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter keypoints based on visibility threshold.
        
        Args:
            keypoints: Body keypoints
            visibility: Visibility scores
            threshold: Minimum visibility threshold
        
        Returns:
            Tuple of (filtered_keypoints, indices)
        """
        visible_indices = np.where(visibility > threshold)[0]
        filtered_keypoints = keypoints[visible_indices]
        
        return filtered_keypoints, visible_indices
    
    def draw_skeleton(self, 
                     image: np.ndarray, 
                     keypoints: np.ndarray,
                     visibility: Optional[np.ndarray] = None,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw body skeleton on image.
        
        Args:
            image: Input image
            keypoints: Body keypoints in pixel coordinates
            visibility: Optional visibility scores
            color: Line color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with skeleton overlay
        """
        output_image = image.copy()
        
        # Draw connections
        for connection in self.connections:
            start_idx, end_idx = connection
            
            # Check visibility if provided
            if visibility is not None:
                if visibility[start_idx] < 0.5 or visibility[end_idx] < 0.5:
                    continue
            
            # Get points
            start_point = tuple(keypoints[start_idx, :2].astype(int))
            end_point = tuple(keypoints[end_idx, :2].astype(int))
            
            # Draw line
            cv2.line(output_image, start_point, end_point, color, thickness)
        
        # Draw keypoints
        for i, point in enumerate(keypoints):
            if visibility is not None and visibility[i] < 0.5:
                continue
            
            center = tuple(point[:2].astype(int))
            cv2.circle(output_image, center, 5, (0, 0, 255), -1)
        
        return output_image
    
    def get_body_bbox(self, 
                     keypoints: np.ndarray,
                     visibility: Optional[np.ndarray] = None,
                     padding: int = 20) -> Tuple[int, int, int, int]:
        """
        Get bounding box around detected body.
        
        Args:
            keypoints: Body keypoints in pixel coordinates
            visibility: Optional visibility scores
            padding: Padding around the bounding box
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        # Filter by visibility if provided
        if visibility is not None:
            visible_mask = visibility > 0.5
            valid_keypoints = keypoints[visible_mask]
        else:
            valid_keypoints = keypoints
        
        if len(valid_keypoints) == 0:
            return None
        
        # Get bounding box
        x_coords = valid_keypoints[:, 0]
        y_coords = valid_keypoints[:, 1]
        
        x_min = int(np.min(x_coords)) - padding
        x_max = int(np.max(x_coords)) + padding
        y_min = int(np.min(y_coords)) - padding
        y_max = int(np.max(y_coords)) + padding
        
        return x_min, y_min, x_max, y_max
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'pose') and self.pose:
            try:
                self.pose.close()
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