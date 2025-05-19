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

logger = logging.getLogger(__name__)


class HandTracker:
    """3D hand tracking using stereo cameras and triangulation."""
    
    def __init__(self,
                 calibration_file: Optional[str] = None,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 max_num_hands: int = 2):
        """
        Initialize the 3D hand tracker.
        
        Args:
            calibration_file: Path to stereo calibration file
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to track
        """
        # Initialize hand detector
        self.detector = HandDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            max_num_hands=max_num_hands
        )
        
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer(gesture_threshold=0.7)
        
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
                      right_image: np.ndarray) -> Dict[str, Any]:
        """
        Track hands in 3D using stereo images.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
        
        Returns:
            Dictionary containing:
            - '3d_keypoints': List of 3D hand keypoints
            - '2d_left': 2D keypoints in left image
            - '2d_right': 2D keypoints in right image
            - 'confidence': Tracking confidence scores
            - 'handedness': List of left/right labels
        """
        # Detect hands in both images
        stereo_results = self.detector.detect_hands_stereo(left_image, right_image)
        
        # Log detection results
        logger.debug(f"Detection results - Left hands: {len(stereo_results['left']['keypoints'])}, Right hands: {len(stereo_results['right']['keypoints'])}")
        
        # Initialize output
        output = {
            '3d_keypoints': [],
            '2d_left': stereo_results['left']['keypoints'],
            '2d_right': stereo_results['right']['keypoints'],
            'handedness_left': stereo_results['left']['handedness'],
            'handedness_right': stereo_results['right']['handedness'],
            'confidence': [],
            'gestures': [],  # Added gesture recognition
            'timestamp': time.time(),
            'frame': self.frame_count
        }
        
        # If no calibration, add 2D gesture recognition
        if not self.calibration_loaded:
            # Recognize gestures from 2D keypoints
            for i, keypoints in enumerate(output['2d_left']):
                if i < len(output['2d_left']):
                    h, w = left_image.shape[:2]
                    gesture_type, gesture_conf = self.gesture_recognizer.recognize_gesture_2d(keypoints, w, h)
                    output['gestures'].append({
                        'type': gesture_type,
                        'confidence': gesture_conf,
                        'name': self.gesture_recognizer.get_gesture_name(gesture_type)
                    })
            self.frame_count += 1
            return output
        
        # Match hands between left and right images
        matched_hands = self._match_stereo_hands(stereo_results)
        logger.debug(f"Found {len(matched_hands)} matched hands between stereo views")
        
        # Triangulate matched hands
        for match in matched_hands:
            left_idx, right_idx = match['indices']
            
            # Get 2D keypoints
            left_kpts = stereo_results['left']['keypoints'][left_idx]
            right_kpts = stereo_results['right']['keypoints'][right_idx]
            
            # Debug shape
            logger.debug(f"Left keypoints shape: {left_kpts.shape}, type: {type(left_kpts)}")
            logger.debug(f"Right keypoints shape: {right_kpts.shape}, type: {type(right_kpts)}")
            
            # Convert to pixel coordinates
            h, w = left_image.shape[:2]
            # Make sure keypoints are in the right shape before converting to pixels
            if left_kpts.ndim != 2:
                left_kpts = left_kpts.reshape(21, -1)  # Reshape to 21 landmarks
            if right_kpts.ndim != 2:
                right_kpts = right_kpts.reshape(21, -1)
                
            left_pixels = self.detector.get_2d_pixel_coordinates(left_kpts, w, h)
            right_pixels = self.detector.get_2d_pixel_coordinates(right_kpts, w, h)
            
            # Triangulate to get 3D points
            points_3d = self._triangulate_points(left_pixels, right_pixels)
            
            output['3d_keypoints'].append(points_3d)
            output['confidence'].append(match['confidence'])
            
            # Recognize gesture from 3D keypoints
            gesture_type, gesture_conf = self.gesture_recognizer.recognize_gestures_3d(points_3d)
            output['gestures'].append({
                'type': gesture_type,
                'confidence': gesture_conf,
                'name': self.gesture_recognizer.get_gesture_name(gesture_type)
            })
        
        # Store results
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
        
        logger.debug(f"Debug - Left hands data: {len(left_hands)} hands")
        logger.debug(f"Debug - Right hands data: {len(right_hands)} hands")
        logger.debug(f"Debug - Left handedness: {stereo_results['left'].get('handedness', [])}")
        logger.debug(f"Debug - Right handedness: {stereo_results['right'].get('handedness', [])}")
        
        matches = []
        
        # Check for duplicate/overlapping detections in left camera
        # Sometimes MediaPipe detects the same hand twice
        left_hand_centers = []
        valid_left_indices = []
        
        for i, left_hand in enumerate(left_hands):
            center_x = np.mean(left_hand[:, 0])
            center_y = np.mean(left_hand[:, 1])
            center = np.array([center_x, center_y])
            
            # Check if this hand is too close to a previously detected hand
            is_duplicate = False
            for prev_center in left_hand_centers:
                distance = np.linalg.norm(center - prev_center)
                if distance < 0.15:  # Threshold for duplicate detection
                    logger.debug(f"Filtering out duplicate left hand {i} (distance: {distance:.3f})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                left_hand_centers.append(center)
                valid_left_indices.append(i)
        
        # Try to match each valid left hand with a right hand
        used_right_indices = set()
        
        for idx, i in enumerate(valid_left_indices):
            left_hand = left_hands[i]
            best_match = None
            best_score = float('inf')
            
            # Get average y-coordinate and center for left hand
            left_y = np.mean(left_hand[:, 1])
            left_x = np.mean(left_hand[:, 0])
            left_center = np.array([left_x, left_y])
            logger.debug(f"Debug - Left hand {i} center: ({left_x:.3f}, {left_y:.3f})")
            
            for j, right_hand in enumerate(right_hands):
                if j in used_right_indices:
                    continue  # Skip already matched hands
                # Get average position for right hand
                right_y = np.mean(right_hand[:, 1])
                right_x = np.mean(right_hand[:, 0])
                right_center = np.array([right_x, right_y])
                logger.debug(f"Debug - Right hand {j} center: ({right_x:.3f}, {right_y:.3f})")
                
                # Check handedness match if available
                handedness_match = True
                handedness_score = 0.0
                
                if (stereo_results['left'].get('handedness') and 
                    stereo_results['right'].get('handedness') and
                    len(stereo_results['left']['handedness']) > i and
                    len(stereo_results['right']['handedness']) > j):
                    left_type = stereo_results['left']['handedness'][i]
                    right_type = stereo_results['right']['handedness'][j]
                    logger.debug(f"Debug - Comparing handedness: left={left_type} vs right={right_type}")
                    
                    # MediaPipe labels hands from the camera's perspective
                    # In stereo setup, hands often appear with opposite labels due to viewpoint
                    # We'll use a more lenient matching approach
                    
                    # Option 1: Same handedness (rare but possible with certain angles)
                    # Option 2: Opposite handedness (common in stereo)
                    if left_type == right_type:
                        logger.debug(f"Handedness same: both are {left_type}")
                        handedness_score = 0.0  # Perfect match
                    elif (left_type == 'Left' and right_type == 'Right') or \
                         (left_type == 'Right' and right_type == 'Left'):
                        logger.debug(f"Handedness opposite (expected): left={left_type}, right={right_type}")
                        handedness_score = 0.1  # Small penalty for opposite (but expected)
                    else:
                        logger.debug(f"Handedness uncertain: left={left_type}, right={right_type}")
                        handedness_score = 0.2  # Moderate penalty
                
                # Calculate matching score based on multiple factors
                y_diff = abs(left_y - right_y)
                
                # In stereo vision, corresponding points have similar y coordinates
                # but x coordinates can vary significantly due to disparity
                # So we focus mainly on y-coordinate matching
                position_score = y_diff
                
                # Combined score
                score = position_score + handedness_score
                
                logger.debug(f"Debug - Match score between left {i} and right {j}: {score:.3f} (y_diff: {y_diff:.3f}, handedness: {handedness_score:.3f})")
                
                if score < best_score:
                    best_score = score
                    best_match = j
            
            # Add match if found and score is reasonable
            threshold = 0.4  # More relaxed threshold for better matching
            logger.debug(f"Best match score for left hand {i}: {best_score:.3f} (threshold: {threshold})")
            
            if best_match is not None and best_score < threshold:
                confidence = max(0.0, 1.0 - (best_score / threshold))
                matches.append({
                    'indices': (i, best_match),
                    'confidence': confidence,
                    'handedness': stereo_results['left'].get('handedness', [None])[i] if len(stereo_results['left'].get('handedness', [])) > i else None
                })
                used_right_indices.add(best_match)
                logger.debug(f"Matched left hand {i} with right hand {best_match} (confidence: {confidence:.2f})")
            else:
                logger.debug(f"No match found for left hand {i} (best score {best_score:.3f} exceeds threshold)")
        
        logger.debug(f"Total matches found: {len(matches)}")
        return matches
    
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
        
        # Debug shape issues
        logger.debug(f"Triangulate - left_points shape: {left_points.shape}, ndim: {left_points.ndim}")
        logger.debug(f"Triangulate - right_points shape: {right_points.shape}, ndim: {right_points.ndim}")
        
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
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()