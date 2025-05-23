"""3D Hand tracking using stereo cameras for UnLook SDK.

This module provides 3D hand tracking using stereo triangulation with two cameras.
It handles the matching of hands between camera views and reconstruction of 3D coordinates.
"""

import time
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

from .HandDetector import HandDetector

logger = logging.getLogger(__name__)


class HandTracker:
    """3D hand tracking using stereo cameras and triangulation."""
    
    def __init__(self,
                 calibration_file: Optional[str] = None,
                 detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.6,
                 max_num_hands: int = 2,            
                 left_camera_mirror_mode: bool = False,  # Is left camera selfie/mirrored view? Set to False for UnLook setup
                 right_camera_mirror_mode: bool = False):  # Is right camera selfie/mirrored view? Set to False for UnLook setup
        """
        Initialize the 3D hand tracker.
        
        Args:
            calibration_file: Path to stereo calibration file
            detection_confidence: Minimum confidence for hand detection (0.6 default)
            tracking_confidence: Minimum confidence for hand tracking (0.6 default)
            max_num_hands: Maximum number of hands to track (2 default)
            left_camera_mirror_mode: Whether left camera is in mirrored mode (False default)
            right_camera_mirror_mode: Whether right camera is in mirrored mode (False default)
        """
        # Camera mirror modes determine how handedness is interpreted
        self.left_camera_mirror_mode = left_camera_mirror_mode
        self.right_camera_mirror_mode = right_camera_mirror_mode
        logger.info(f"Camera mirror modes: left={left_camera_mirror_mode}, right={right_camera_mirror_mode}")
        
        # Initialize hand detector for left camera
        self.detector_left = HandDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            max_num_hands=max_num_hands,
            hand_mirror_mode=left_camera_mirror_mode
        )
        
        # Initialize hand detector for right camera
        self.detector_right = HandDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            max_num_hands=max_num_hands,
            hand_mirror_mode=right_camera_mirror_mode
        )
        
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
        """
        Compute projection matrices from calibration parameters.
        
        Returns:
            Tuple of projection matrices (P1, P2)
        """
        # Create projection matrices
        # P1 = K1 * [I | 0]
        # P2 = K2 * [R | T]
        P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([self.R, self.T])
        
        return P1, P2
    
    def track_hands_3d(self, 
                      left_image: np.ndarray, 
                      right_image: np.ndarray,
                      min_detection_confidence: float = None) -> Dict[str, Any]:
        """
        Track hands in 3D using stereo images.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
            min_detection_confidence: Override minimum confidence for detection
        
        Returns:
            Dictionary containing:
            - '3d_keypoints': List of 3D hand keypoints
            - '2d_left': 2D keypoints in left image
            - '2d_right': 2D keypoints in right image
            - 'handedness_left': Handedness from left camera
            - 'handedness_right': Handedness from right camera
            - 'confidence': Tracking confidence scores
            - 'timestamp': Time of tracking
            - 'frame': Frame counter
        """
        # Override confidence thresholds if specified (helps filter out false positives)
        if min_detection_confidence is not None:
            temp_left_detection_conf = self.detector_left.detection_confidence
            temp_right_detection_conf = self.detector_right.detection_confidence
            
            # Temporarily update confidence thresholds if specified
            self.detector_left.detection_confidence = max(self.detector_left.detection_confidence, 
                                                         min_detection_confidence)
            self.detector_right.detection_confidence = max(self.detector_right.detection_confidence,
                                                          min_detection_confidence)
            
            # Log the temporary confidence overrides
            logger.debug(f"Using temporary confidence threshold: detection={min_detection_confidence}")
        
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
        if min_detection_confidence is not None:
            self.detector_left.detection_confidence = temp_left_detection_conf
            self.detector_right.detection_confidence = temp_right_detection_conf
        
        # Initialize output with 2D detection results
        output = {
            '3d_keypoints': [],
            '2d_left': stereo_results['left']['keypoints'],
            '2d_right': stereo_results['right']['keypoints'],
            'handedness_left': stereo_results['left']['handedness'],
            'handedness_right': stereo_results['right']['handedness'],
            'confidence': [],
            'timestamp': time.time(),
            'frame': self.frame_count
        }
        
        # If no calibration, just return 2D results only
        if not self.calibration_loaded or not isinstance(self.P1, np.ndarray) or not isinstance(self.P2, np.ndarray):
            logger.info("No valid calibration loaded - returning 2D tracking results only")
            self.frame_count += 1
            return output
        
        # Match hands between left and right images for 3D analysis
        matched_hands = self._match_stereo_hands(stereo_results)
        
        # Keep track of which hands are matched
        matched_left_indices = []
        matched_right_indices = []
        
        # Triangulate matched hands for 3D coordinates
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
            
            # Get handedness for reference
            handedness = "Unknown"
            if left_idx < len(stereo_results['left']['handedness']):
                handedness = stereo_results['left']['handedness'][left_idx]
            
            # Log 3D tracking result
            logger.info(f"HandPose 3d_tracking: match_confidence={match['confidence']:.2f}, handedness={handedness}")
        
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
            List of matched hand pairs with confidence scores
        """
        left_hands = stereo_results['left']['keypoints']
        right_hands = stereo_results['right']['keypoints']
        
        matches = []
        
        # Get confidence values if available
        left_confidences = stereo_results['left'].get('confidence', [1.0] * len(left_hands))
        right_confidences = stereo_results['right'].get('confidence', [1.0] * len(right_hands))
        
        # Process left hands - filtering for duplicates and false positives
        left_hand_centers = []
        right_hand_centers = []
        valid_left_indices = []
        valid_right_indices = []
        
        # Minimum confidence threshold for hands to consider in stereo matching
        min_confidence_threshold = 0.5
        
        # Process left hands
        for i, left_hand in enumerate(left_hands):
            # Check confidence if available (skip low-confidence detections)
            if i < len(left_confidences) and left_confidences[i] < min_confidence_threshold:
                logger.debug(f"Left camera filtering low confidence hand: {i} (confidence={left_confidences[i]:.3f})")
                continue
                
            # Calculate hand center
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
            
            # Add this hand
            left_hand_centers.append(center)
            valid_left_indices.append(i)
        
        # Process right hands
        for j, right_hand in enumerate(right_hands):
            # Check confidence if available (skip low-confidence detections)
            if j < len(right_confidences) and right_confidences[j] < min_confidence_threshold:
                logger.debug(f"Right camera filtering low confidence hand: {j} (confidence={right_confidences[j]:.3f})")
                continue
                
            # Calculate hand center
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
            
            # Add this hand
            right_hand_centers.append(center)
            valid_right_indices.append(j)
        
        # Log how many valid hands we have after filtering
        if len(valid_left_indices) < len(left_hands) or len(valid_right_indices) < len(right_hands):
            logger.info(f"Filtered hands: Left {len(left_hands)}->{len(valid_left_indices)}, Right {len(right_hands)}->{len(valid_right_indices)}")
        
        # If we have valid detections in both cameras
        if valid_left_indices and valid_right_indices:
            # For each left hand, find the best matching right hand
            for idx, i in enumerate(valid_left_indices):
                left_hand = left_hands[i]
                best_match = None
                best_score = float('inf')
                
                # Get key points and features for left hand
                left_center_x = np.mean(left_hand[:, 0])
                left_center_y = np.mean(left_hand[:, 1])
                left_center = np.array([left_center_x, left_center_y])
                
                # Get handedness info
                left_handedness = None
                if (stereo_results['left'].get('handedness') and 
                    len(stereo_results['left']['handedness']) > i):
                    left_handedness = stereo_results['left']['handedness'][i]
                
                # Try to match with each right hand
                for j_idx, j in enumerate(valid_right_indices):
                    right_hand = right_hands[j]
                    
                    # Get key points and features for right hand
                    right_center_x = np.mean(right_hand[:, 0])
                    right_center_y = np.mean(right_hand[:, 1])
                    right_center = np.array([right_center_x, right_center_y])
                    
                    # Get handedness info
                    right_handedness = None
                    if (stereo_results['right'].get('handedness') and 
                        len(stereo_results['right']['handedness']) > j):
                        right_handedness = stereo_results['right']['handedness'][j]
                    
                    # Matching metrics
                    # 1. Y-coordinate similarity (most important in stereo)
                    y_diff = abs(left_center_y - right_center_y)
                    # More stringent y-coordinate matching
                    y_threshold = 0.15  # Maximum allowed vertical difference
                    if y_diff > y_threshold:
                        # Y-difference too large - these cannot be the same hand
                        y_score = 10.0  # Large penalty
                    else:
                        y_score = y_diff * 3.0  # Weighted heavily
                    
                    # 2. X-coordinate factor (accounting for stereo disparity)
                    # In a properly configured stereo setup, the same point will have 
                    # a larger X coordinate in the left image than in the right image
                    disparity = left_center_x - right_center_x
                    
                    # Disparity must be positive and within reasonable range
                    if disparity <= 0:
                        x_score = 1.0  # Strong penalty - physically impossible in standard stereo
                    elif disparity > 0.5:  # Too much disparity
                        x_score = 0.8
                    else:
                        # Valid disparity range
                        x_score = 0.3 * (1.0 - disparity * 2)
                    
                    # 3. Handedness matching - critical for correct pairing
                    handedness_score = 0.0
                    if left_handedness and right_handedness:
                        # In stereo, we expect opposite handedness due to mirroring
                        if (left_handedness == 'Left' and right_handedness == 'Right') or \
                           (left_handedness == 'Right' and right_handedness == 'Left'):
                            # Expected case - no penalty
                            handedness_score = 0.0
                        elif left_handedness == right_handedness:
                            # Same handedness - severe penalty
                            handedness_score = 1.0  # Effective rejection
                        else:
                            # Unknown comparison
                            handedness_score = 0.5
                    
                    # Combine all scores with weights
                    score = (y_score * 0.5) + (x_score * 0.25) + (handedness_score * 0.25)
                    
                    # Log detailed match scores for debugging
                    logger.debug(f"Match {i}:{j} score={score:.2f} (y={y_score:.2f}, x={x_score:.2f}, hand={handedness_score:.2f})")
                    
                    if score < best_score:
                        best_score = score
                        best_match = j
                
                # Add match if found and score is reasonable
                threshold = 0.35  # Threshold for match quality
                
                if best_match is not None and best_score < threshold:
                    confidence = max(0.0, 1.0 - (best_score / threshold))
                    
                    # Final check - verify handedness compatibility
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
                    
                    # Log the match
                    logger.info(f"Matched hands: left={i} ({left_handedness}) with right={best_match} ({right_handedness}) score={best_score:.2f}")
                else:
                    logger.debug(f"No match found for left hand {i} (best score {best_score:.2f} exceeds threshold {threshold})")
            
            # Sort matches by confidence (highest first)
            matches.sort(key=lambda m: m['confidence'], reverse=True)
            
            # If no matches found but we have hands in both images, try a forced match for single hand case
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
                # 1. Y-position is similar
                # 2. Disparity is positive and reasonable
                # 3. Both hands have decent confidence
                
                y_position_compatible = y_diff < 0.15
                disparity_compatible = disparity > 0 and disparity < 0.5
                good_confidence = left_confidence > 0.7 and right_confidence > 0.7
                
                # Perform a forced match if compatible
                if y_position_compatible and disparity_compatible and good_confidence:
                    logger.info(f"Forced match for single hand case: left={i} ({left_handedness}), right={j} ({right_handedness})")
                    matches.append({
                        'indices': (i, j),
                        'confidence': 0.6,  # Reasonable confidence for a forced match
                        'handedness': left_handedness
                    })
                else:
                    if not y_position_compatible:
                        logger.info(f"Skipping forced match due to incompatible Y positions: diff={y_diff:.3f}")
                    elif not disparity_compatible:
                        logger.info(f"Skipping forced match due to incompatible disparity: {disparity:.3f}")
                    elif not good_confidence:
                        logger.info(f"Skipping forced match due to low confidence: left={left_confidence:.2f}, right={right_confidence:.2f}")
        
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
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'detector_left'):
            logger.info("Closing left detector resources")
            self.detector_left.close()
            
        if hasattr(self, 'detector_right'):
            logger.info("Closing right detector resources")
            self.detector_right.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()