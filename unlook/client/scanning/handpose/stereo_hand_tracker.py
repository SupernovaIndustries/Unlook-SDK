"""Stereo Hand Tracker - CPU-optimized 3D hand tracking using stereo vision.

This module implements efficient hand tracking using stereo cameras without
requiring ML or GPU acceleration. It leverages geometric algorithms and
stereo triangulation for robust 3D hand pose estimation.

Key features:
- Real-time stereo hand tracking on CPU
- 3D triangulation of hand keypoints
- Kalman filtering for smooth tracking
- Geometric gesture recognition
- Optimized for UnLook stereo camera system
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import time
from collections import deque
from scipy.signal import savgol_filter


@dataclass
class Hand3D:
    """3D hand representation with keypoints and tracking state."""
    keypoints_3d: np.ndarray  # Shape: (21, 3) for 21 hand landmarks
    keypoints_2d_left: np.ndarray  # Shape: (21, 2) 
    keypoints_2d_right: np.ndarray  # Shape: (21, 2)
    confidence: float
    timestamp: float
    hand_id: int
    is_left_hand: bool
    
    
@dataclass
class StereoCalibration:
    """Stereo camera calibration parameters."""
    K_left: np.ndarray  # Left camera intrinsic matrix
    K_right: np.ndarray  # Right camera intrinsic matrix
    D_left: np.ndarray  # Left camera distortion coefficients
    D_right: np.ndarray  # Right camera distortion coefficients
    R: np.ndarray  # Rotation matrix between cameras
    T: np.ndarray  # Translation vector between cameras
    baseline: float  # Distance between cameras
    

class KalmanFilter3D:
    """3D Kalman filter for hand tracking."""
    
    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        """Initialize Kalman filter for 3D point tracking."""
        # State: [x, y, z, vx, vy, vz]
        self.state_size = 6
        self.measurement_size = 3
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_size)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = 1.0  # dt = 1
        
        # Measurement matrix
        self.H = np.zeros((self.measurement_size, self.state_size))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        
        # Process noise covariance
        self.Q = np.eye(self.state_size) * process_variance
        
        # Measurement noise covariance
        self.R = np.eye(self.measurement_size) * measurement_variance
        
        # Initial state
        self.x = np.zeros(self.state_size)
        self.P = np.eye(self.state_size) * 100  # High initial uncertainty
        
        self.initialized = False
        
    def predict(self, dt: float = 1.0):
        """Predict next state."""
        # Update F matrix with actual dt
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        
        # Predict state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """Update with measurement."""
        if not self.initialized:
            self.x[:3] = measurement
            self.initialized = True
            return
            
        # Innovation
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_size) - K @ self.H) @ self.P
        
    def get_position(self) -> np.ndarray:
        """Get current 3D position."""
        return self.x[:3].copy()
        
    def get_velocity(self) -> np.ndarray:
        """Get current 3D velocity."""
        return self.x[3:].copy()


class FastSkinDetector:
    """Fast CPU-optimized skin detection using color spaces."""
    
    def __init__(self):
        """Initialize skin detector with optimized parameters."""
        # Pre-compute lookup tables for color space conversions
        self._init_luts()
        
    def _init_luts(self):
        """Initialize lookup tables for fast color conversions."""
        # YCrCb thresholds for skin detection
        self.y_min, self.y_max = 0, 255
        self.cr_min, self.cr_max = 133, 173
        self.cb_min, self.cb_max = 77, 127
        
        # HSV thresholds
        self.h_min, self.h_max = 0, 20
        self.s_min, self.s_max = 20, 255
        self.v_min, self.v_max = 70, 255
        
    def detect_skin(self, image: np.ndarray, use_flood_illuminator: bool = False) -> np.ndarray:
        """Detect skin regions using optimized color space analysis.
        
        Args:
            image: BGR input image
            use_flood_illuminator: Whether flood illuminator is active (affects thresholds)
            
        Returns:
            Binary mask of skin regions
        """
        # Downscale for faster processing - balance speed vs accuracy
        scale = 0.5  # Back to 0.5 for speed, we'll compensate with better thresholds
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # Convert to YCrCb and HSV
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Adjust thresholds if flood illuminator is active
        if use_flood_illuminator:
            # Flood illuminator makes skin appear differently - wider ranges
            self.v_min = 30  # Much lower for flood illuminator
            self.s_min = 10  # Lower saturation threshold
            self.cr_min = 125  # Wider Cr range
            self.cr_max = 180
            self.cb_min = 70
            self.cb_max = 135
            
        # Create masks
        ycrcb_mask = cv2.inRange(ycrcb, 
                                (self.y_min, self.cr_min, self.cb_min),
                                (self.y_max, self.cr_max, self.cb_max))
        
        hsv_mask = cv2.inRange(hsv,
                              (self.h_min, self.s_min, self.v_min),
                              (self.h_max, self.s_max, self.v_max))
        
        # Combine masks - use OR for flood illuminator to be more inclusive
        if use_flood_illuminator:
            skin_mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)
        else:
            skin_mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)
        
        # Morphological operations to clean up - smaller kernel for speed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel2, iterations=1)
        
        # Upscale back to original size
        skin_mask = cv2.resize(skin_mask, (image.shape[1], image.shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
        
        return skin_mask


class ContourHandDetector:
    """Fast hand detection using contour analysis."""
    
    def __init__(self):
        """Initialize contour-based hand detector."""
        # Adjusted for typical hand sizes at 30-60cm distance
        self.min_hand_area = 3000  # Smaller minimum for farther hands
        self.max_hand_area = 80000  # Larger maximum for closer hands
        
    def detect_hands(self, skin_mask: np.ndarray) -> List[Dict]:
        """Detect hands from skin mask using contour analysis.
        
        Returns:
            List of hand detections with bounding boxes and contours
        """
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_hand_area < area < self.max_hand_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (hands are roughly square)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    # Compute convex hull for hand shape
                    hull = cv2.convexHull(contour)
                    
                    # Solidity check (hand vs random shape)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.6:  # Hands have high solidity
                        hands.append({
                            'bbox': (x, y, w, h),
                            'contour': contour,
                            'hull': hull,
                            'area': area,
                            'center': (x + w//2, y + h//2),
                            'solidity': solidity
                        })
        
        return hands


class StereoHandTracker:
    """Main stereo hand tracking class."""
    
    def __init__(self, calibration: StereoCalibration):
        """Initialize stereo hand tracker.
        
        Args:
            calibration: Stereo camera calibration parameters
        """
        self.calibration = calibration
        self.skin_detector = FastSkinDetector()
        self.hand_detector = ContourHandDetector()
        
        # Tracking state
        self.kalman_filters: Dict[int, List[KalmanFilter3D]] = {}  # hand_id -> list of 21 filters
        self.hand_counter = 0
        self.last_timestamp = time.time()
        
        # Performance optimization
        self.roi_tracker = {}  # hand_id -> (left_roi, right_roi)
        self.search_expansion = 50  # pixels to expand ROI
        
        # Initialize stereo matcher for depth estimation
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=96, blockSize=15)
        
    def track(self, left_image: np.ndarray, right_image: np.ndarray, 
              use_flood_illuminator: bool = False, fast_mode: bool = True) -> List[Hand3D]:
        """Track hands in stereo images.
        
        Args:
            left_image: Left camera image (BGR)
            right_image: Right camera image (BGR)
            use_flood_illuminator: Whether flood illuminator is active
            
        Returns:
            List of detected 3D hands
        """
        current_time = time.time()
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        # Detect skin in both images
        left_skin = self.skin_detector.detect_skin(left_image, use_flood_illuminator)
        right_skin = self.skin_detector.detect_skin(right_image, use_flood_illuminator)
        
        # Detect hands in left image (primary)
        left_hands = self.hand_detector.detect_hands(left_skin)
        
        detected_hands = []
        
        for left_hand in left_hands:
            # Find corresponding hand in right image using epipolar constraint
            right_hand = self._find_corresponding_hand(left_hand, right_skin, right_image)
            
            if right_hand is not None:
                # Extract hand keypoints
                left_keypoints = self._extract_keypoints(left_hand, left_image)
                right_keypoints = self._extract_keypoints(right_hand, right_image)
                
                if left_keypoints is not None and right_keypoints is not None:
                    # Triangulate 3D points
                    keypoints_3d = self._triangulate_keypoints(left_keypoints, right_keypoints)
                    
                    # Apply Kalman filtering
                    hand_id = self._assign_hand_id(keypoints_3d)
                    filtered_3d = self._apply_kalman_filter(keypoints_3d, hand_id, dt)
                    
                    # Create Hand3D object
                    hand = Hand3D(
                        keypoints_3d=filtered_3d,
                        keypoints_2d_left=left_keypoints,
                        keypoints_2d_right=right_keypoints,
                        confidence=0.9,  # High confidence for geometric detection
                        timestamp=current_time,
                        hand_id=hand_id,
                        is_left_hand=self._is_left_hand(filtered_3d)
                    )
                    
                    detected_hands.append(hand)
                    
                    # Update ROI for next frame
                    self._update_roi(hand_id, left_hand['bbox'], right_hand['bbox'])
        
        return detected_hands
    
    def _find_corresponding_hand(self, left_hand: Dict, right_skin: np.ndarray, 
                                right_image: np.ndarray) -> Optional[Dict]:
        """Find corresponding hand in right image using epipolar constraint."""
        # Get epipolar line for hand center
        left_center = np.array([left_hand['center']], dtype=np.float32)
        
        # Compute fundamental matrix from calibration
        F = self._compute_fundamental_matrix()
        
        # Compute epipolar line in right image
        epiline = cv2.computeCorrespondEpilines(left_center.reshape(-1, 1, 2), 1, F)
        epiline = epiline.reshape(-1, 3)[0]
        
        # Search along epipolar line
        height, width = right_image.shape[:2]
        search_points = []
        
        # Sample points along epipolar line
        for x in range(0, width, 10):
            y = int((-epiline[0] * x - epiline[2]) / epiline[1])
            if 0 <= y < height:
                search_points.append((x, y))
        
        # Find hand regions near epipolar line
        right_hands = self.hand_detector.detect_hands(right_skin)
        
        best_hand = None
        min_distance = float('inf')
        
        for right_hand in right_hands:
            # Compute distance from hand center to epipolar line
            center = right_hand['center']
            dist = abs(epiline[0] * center[0] + epiline[1] * center[1] + epiline[2]) / \
                   np.sqrt(epiline[0]**2 + epiline[1]**2)
            
            if dist < min_distance and dist < 30:  # 30 pixel threshold
                min_distance = dist
                best_hand = right_hand
        
        return best_hand
    
    def _extract_keypoints(self, hand: Dict, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand keypoints using geometric analysis."""
        # Get hand region
        x, y, w, h = hand['bbox']
        hand_roi = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Find fingertips using convexity defects
        contour = hand['contour'] - np.array([x, y])
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) > 3 and len(contour) > 3:
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is not None:
                # Extract fingertips and valleys
                fingertips = []
                valleys = []
                
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Fingertip candidates
                    if d > 1000:  # Significant defect
                        fingertips.append(start)
                        fingertips.append(end)
                        valleys.append(far)
                
                # Estimate palm center
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    palm_center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
                else:
                    palm_center = (w//2, h//2)
                
                # Create simplified 21-point hand model
                # For now, use a simplified approach - in production, use more sophisticated methods
                keypoints = np.zeros((21, 2))
                
                # Wrist (0)
                keypoints[0] = [palm_center[0], h - 10]
                
                # Palm center 
                keypoints[9] = palm_center
                
                # Distribute fingertips
                if len(fingertips) >= 5:
                    # Sort fingertips by angle from palm center
                    angles = []
                    for ft in fingertips[:5]:
                        angle = np.arctan2(ft[1] - palm_center[1], ft[0] - palm_center[0])
                        angles.append((angle, ft))
                    angles.sort()
                    
                    # Assign to fingers (thumb to pinky)
                    for i, (_, ft) in enumerate(angles):
                        finger_base = i * 4 + 1
                        if finger_base < 21:
                            # Fingertip
                            keypoints[finger_base + 3] = ft
                            # Interpolate intermediate joints
                            for j in range(3):
                                alpha = (j + 1) / 4.0
                                keypoints[finger_base + j] = (
                                    palm_center[0] * (1 - alpha) + ft[0] * alpha,
                                    palm_center[1] * (1 - alpha) + ft[1] * alpha
                                )
                
                # Convert back to image coordinates
                keypoints += np.array([x, y])
                
                return keypoints
        
        return None
    
    def _triangulate_keypoints(self, left_points: np.ndarray, 
                              right_points: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from stereo correspondences."""
        # Prepare projection matrices
        P1 = self.calibration.K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.calibration.K_right @ np.hstack((self.calibration.R, self.calibration.T.reshape(-1, 1)))
        
        points_3d = []
        
        for i in range(len(left_points)):
            # Triangulate each point
            point_4d = cv2.triangulatePoints(P1, P2, 
                                            left_points[i].astype(np.float32),
                                            right_points[i].astype(np.float32))
            
            # Convert to 3D
            point_3d = point_4d[:3] / point_4d[3]
            points_3d.append(point_3d.flatten())
        
        return np.array(points_3d)
    
    def _apply_kalman_filter(self, keypoints_3d: np.ndarray, hand_id: int, 
                            dt: float) -> np.ndarray:
        """Apply Kalman filtering to smooth 3D keypoints."""
        if hand_id not in self.kalman_filters:
            # Initialize filters for new hand
            self.kalman_filters[hand_id] = [
                KalmanFilter3D() for _ in range(21)
            ]
        
        filters = self.kalman_filters[hand_id]
        filtered_points = []
        
        for i, point_3d in enumerate(keypoints_3d):
            filters[i].predict(dt)
            filters[i].update(point_3d)
            filtered_points.append(filters[i].get_position())
        
        return np.array(filtered_points)
    
    def _assign_hand_id(self, keypoints_3d: np.ndarray) -> int:
        """Assign ID to tracked hand based on proximity to previous hands."""
        if not self.kalman_filters:
            # First hand
            self.hand_counter += 1
            return self.hand_counter
        
        # Find closest existing hand
        min_distance = float('inf')
        best_id = None
        
        palm_pos = keypoints_3d[9]  # Palm center
        
        for hand_id, filters in self.kalman_filters.items():
            existing_palm = filters[9].get_position()
            distance = np.linalg.norm(palm_pos - existing_palm)
            
            if distance < min_distance and distance < 100:  # 100mm threshold
                min_distance = distance
                best_id = hand_id
        
        if best_id is not None:
            return best_id
        else:
            # New hand
            self.hand_counter += 1
            return self.hand_counter
    
    def _is_left_hand(self, keypoints_3d: np.ndarray) -> bool:
        """Determine if hand is left or right based on thumb position."""
        # Simple heuristic: check if thumb is on left or right of palm
        thumb_tip = keypoints_3d[4]  # Thumb tip
        palm = keypoints_3d[9]  # Palm center
        pinky_tip = keypoints_3d[20]  # Pinky tip
        
        # Create hand coordinate system
        hand_dir = pinky_tip - thumb_tip
        
        # In camera coordinates, negative X is left
        return hand_dir[0] < 0
    
    def _compute_fundamental_matrix(self) -> np.ndarray:
        """Compute fundamental matrix from calibration."""
        # F = K_right^(-T) * E * K_left^(-1)
        # E = [T]x * R (essential matrix)
        
        # Skew-symmetric matrix of T
        T_x = np.array([
            [0, -self.calibration.T[2], self.calibration.T[1]],
            [self.calibration.T[2], 0, -self.calibration.T[0]],
            [-self.calibration.T[1], self.calibration.T[0], 0]
        ])
        
        # Essential matrix
        E = T_x @ self.calibration.R
        
        # Fundamental matrix
        F = np.linalg.inv(self.calibration.K_right).T @ E @ np.linalg.inv(self.calibration.K_left)
        
        return F
    
    def _update_roi(self, hand_id: int, left_bbox: Tuple, right_bbox: Tuple):
        """Update ROI for tracking optimization."""
        # Expand bounding boxes for next frame
        left_roi = (
            max(0, left_bbox[0] - self.search_expansion),
            max(0, left_bbox[1] - self.search_expansion),
            left_bbox[2] + 2 * self.search_expansion,
            left_bbox[3] + 2 * self.search_expansion
        )
        
        right_roi = (
            max(0, right_bbox[0] - self.search_expansion),
            max(0, right_bbox[1] - self.search_expansion),
            right_bbox[2] + 2 * self.search_expansion,
            right_bbox[3] + 2 * self.search_expansion
        )
        
        self.roi_tracker[hand_id] = (left_roi, right_roi)


class GeometricGestureRecognizer:
    """Recognize gestures using geometric features."""
    
    def __init__(self):
        """Initialize gesture recognizer."""
        self.gesture_history = deque(maxlen=10)
        self.movement_threshold = 20  # mm
        
    def recognize(self, hand: Hand3D) -> Tuple[str, float]:
        """Recognize gesture from 3D hand pose.
        
        Returns:
            Tuple of (gesture_name, confidence)
        """
        keypoints = hand.keypoints_3d
        
        # Extract geometric features
        features = self._extract_features(keypoints)
        
        # Rule-based gesture classification
        gesture, confidence = self._classify_gesture(features)
        
        # Check for dynamic gestures
        if len(self.gesture_history) > 5:
            dynamic_gesture = self._check_dynamic_gesture(hand)
            if dynamic_gesture is not None:
                return dynamic_gesture
        
        self.gesture_history.append((hand.timestamp, keypoints.copy()))
        
        return gesture, confidence
    
    def _extract_features(self, keypoints: np.ndarray) -> Dict:
        """Extract geometric features from hand keypoints."""
        features = {}
        
        # Palm center
        palm = keypoints[9]
        
        # Finger states (extended or bent)
        for i, finger_name in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
            base_idx = i * 4 + 1
            if base_idx + 3 < 21:
                tip = keypoints[base_idx + 3]
                mcp = keypoints[base_idx]  # Metacarpophalangeal joint
                
                # Distance from tip to palm
                tip_palm_dist = np.linalg.norm(tip - palm)
                mcp_palm_dist = np.linalg.norm(mcp - palm)
                
                # Finger is extended if tip is far from palm
                features[f'{finger_name}_extended'] = tip_palm_dist > mcp_palm_dist * 1.5
                
                # Finger angles
                if base_idx + 2 < 21:
                    pip = keypoints[base_idx + 2]  # Proximal interphalangeal
                    
                    # Angle at PIP joint
                    v1 = mcp - pip
                    v2 = tip - pip
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    features[f'{finger_name}_angle'] = np.degrees(angle)
        
        # Hand orientation
        # Normal vector from palm, index, and pinky
        if all(i < 21 for i in [5, 9, 17]):
            v1 = keypoints[5] - keypoints[9]   # Index MCP to palm
            v2 = keypoints[17] - keypoints[9]  # Pinky MCP to palm
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            features['palm_normal'] = normal
            
            # Palm facing camera if z-component is negative
            features['palm_facing_camera'] = normal[2] < 0
        
        # Inter-finger distances
        if all(i < 21 for i in [4, 8]):
            features['thumb_index_distance'] = np.linalg.norm(keypoints[4] - keypoints[8])
        
        return features
    
    def _classify_gesture(self, features: Dict) -> Tuple[str, float]:
        """Classify gesture based on geometric features."""
        # Count extended fingers
        extended_count = sum(1 for k, v in features.items() 
                           if k.endswith('_extended') and v)
        
        # Open palm: all fingers extended, palm facing camera
        if extended_count == 5 and features.get('palm_facing_camera', False):
            return 'open_palm', 0.9
        
        # Closed fist: no fingers extended
        if extended_count == 0:
            return 'fist', 0.9
        
        # Pointing: only index extended
        if (features.get('index_extended', False) and 
            extended_count == 1):
            return 'pointing', 0.85
        
        # Peace sign: index and middle extended
        if (features.get('index_extended', False) and 
            features.get('middle_extended', False) and 
            extended_count == 2):
            return 'peace', 0.85
        
        # Thumbs up: only thumb extended, thumb pointing up
        if (features.get('thumb_extended', False) and 
            extended_count == 1):
            return 'thumbs_up', 0.8
        
        # Pinch: thumb and index close together
        if features.get('thumb_index_distance', 100) < 30:  # 30mm threshold
            return 'pinch', 0.85
        
        # OK sign: thumb and index forming circle
        if (features.get('thumb_index_distance', 100) < 40 and
            features.get('middle_extended', False) and
            features.get('ring_extended', False) and
            features.get('pinky_extended', False)):
            return 'ok', 0.8
        
        return 'unknown', 0.0
    
    def _check_dynamic_gesture(self, current_hand: Hand3D) -> Optional[Tuple[str, float]]:
        """Check for dynamic gestures like swipes."""
        if len(self.gesture_history) < 5:
            return None
        
        # Get palm trajectory
        palm_trajectory = [pose[1][9] for pose in self.gesture_history]
        palm_trajectory.append(current_hand.keypoints_3d[9])
        
        # Smooth trajectory
        palm_trajectory = np.array(palm_trajectory)
        if len(palm_trajectory) > 3:
            # Simple moving average
            smoothed = np.convolve(palm_trajectory.mean(axis=1), 
                                  np.ones(3)/3, mode='valid')
        
        # Check for significant movement
        total_movement = np.linalg.norm(palm_trajectory[-1] - palm_trajectory[0])
        
        if total_movement > self.movement_threshold:
            # Determine direction
            movement_vector = palm_trajectory[-1] - palm_trajectory[0]
            
            # Normalize
            movement_vector = movement_vector / (np.linalg.norm(movement_vector) + 1e-6)
            
            # Classify swipe direction
            if abs(movement_vector[0]) > 0.7:  # X-axis dominant
                if movement_vector[0] > 0:
                    return 'swipe_right', 0.8
                else:
                    return 'swipe_left', 0.8
            elif abs(movement_vector[1]) > 0.7:  # Y-axis dominant
                if movement_vector[1] > 0:
                    return 'swipe_down', 0.8
                else:
                    return 'swipe_up', 0.8
        
        return None