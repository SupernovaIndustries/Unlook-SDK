"""3D Body tracking using stereo cameras for UnLook SDK.

Based on bodypose3d by TemugeB: https://github.com/TemugeB/bodypose3d
"""

import time
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

from .body_detector import BodyDetector

logger = logging.getLogger(__name__)


class BodyTracker:
    """3D body tracking using stereo cameras and triangulation."""
    
    def __init__(self,
                 calibration_file: Optional[str] = None,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Initialize the 3D body tracker.
        
        Args:
            calibration_file: Path to stereo calibration file
            detection_confidence: Minimum confidence for body detection
            tracking_confidence: Minimum confidence for body tracking
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
        """
        # Initialize body detector
        self.detector = BodyDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            model_complexity=model_complexity
        )
        
        # Load calibration if provided
        self.calibration_loaded = False
        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)
        else:
            logger.warning("No calibration file provided. 3D reconstruction will not be available.")
        
        # Storage for tracked bodies
        self.tracked_bodies = []
        self.frame_count = 0
        
        # Configuration
        self.max_history = 1000  # Maximum frames to store
        self.visibility_threshold = 0.5  # Minimum visibility for 3D reconstruction
        
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
        P1 = np.hstack([self.K1, np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([self.R, self.T])
        
        return P1, P2
    
    def track_body_3d(self, 
                     left_image: np.ndarray, 
                     right_image: np.ndarray) -> Dict[str, Any]:
        """
        Track body in 3D using stereo images.
        
        Args:
            left_image: Left camera image (BGR format)
            right_image: Right camera image (BGR format)
        
        Returns:
            Dictionary containing:
            - '3d_keypoints': 3D body keypoints
            - '2d_left': 2D keypoints in left image
            - '2d_right': 2D keypoints in right image
            - 'visibility_left': Visibility scores for left image
            - 'visibility_right': Visibility scores for right image
            - 'confidence': Overall tracking confidence
        """
        # Detect body pose in both images
        stereo_results = self.detector.detect_pose_stereo(left_image, right_image)
        
        # Initialize output
        output = {
            '3d_keypoints': None,
            '2d_left': stereo_results['left']['keypoints'],
            '2d_right': stereo_results['right']['keypoints'],
            'visibility_left': stereo_results['left']['visibility'],
            'visibility_right': stereo_results['right']['visibility'],
            'confidence': 0.0,
            'timestamp': time.time(),
            'frame': self.frame_count,
            'annotated_left': stereo_results['left']['image'],
            'annotated_right': stereo_results['right']['image']
        }
        
        # If no calibration, return 2D results only
        if not self.calibration_loaded:
            self.frame_count += 1
            return output
        
        # Check if pose detected in both images
        if (len(stereo_results['left']['keypoints']) > 0 and 
            len(stereo_results['right']['keypoints']) > 0):
            
            # Get 2D keypoints
            left_kpts = stereo_results['left']['keypoints']
            right_kpts = stereo_results['right']['keypoints']
            left_vis = stereo_results['left']['visibility']
            right_vis = stereo_results['right']['visibility']
            
            # Convert to pixel coordinates
            h, w = left_image.shape[:2]
            left_pixels = self.detector.get_2d_pixel_coordinates(left_kpts, w, h)
            right_pixels = self.detector.get_2d_pixel_coordinates(right_kpts, w, h)
            
            # Find correspondences based on visibility
            valid_indices = []
            for i in range(self.detector.num_landmarks):
                if (left_vis[i] > self.visibility_threshold and 
                    right_vis[i] > self.visibility_threshold):
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                # Triangulate visible keypoints
                points_3d = self._triangulate_keypoints(
                    left_pixels[valid_indices], 
                    right_pixels[valid_indices]
                )
                
                # Create full 3D skeleton with invalid points marked
                full_3d = np.full((self.detector.num_landmarks, 3), -1.0, dtype=np.float32)
                full_3d[valid_indices] = points_3d
                
                output['3d_keypoints'] = full_3d
                output['confidence'] = len(valid_indices) / self.detector.num_landmarks
        
        # Store results
        self.tracked_bodies.append(output)
        if len(self.tracked_bodies) > self.max_history:
            self.tracked_bodies.pop(0)
        
        self.frame_count += 1
        
        return output
    
    def _triangulate_keypoints(self, 
                              left_points: np.ndarray, 
                              right_points: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from corresponding 2D keypoints.
        
        Args:
            left_points: 2D points in left image (Nx2)
            right_points: 2D points in right image (Nx2)
        
        Returns:
            3D points (Nx3)
        """
        # Ensure points are in correct format
        left_points = left_points.reshape(-1, 2).T
        right_points = right_points.reshape(-1, 2).T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, left_points, right_points)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        return points_3d
    
    def visualize_3d_body(self, 
                         output: Dict,
                         scale: float = 1.0,
                         show_invalid: bool = False) -> Optional[np.ndarray]:
        """
        Create a 3D visualization of tracked body.
        
        Args:
            output: Output from track_body_3d
            scale: Scale factor for visualization
            show_invalid: Whether to show invalid keypoints
        
        Returns:
            Visualization image or None
        """
        if output['3d_keypoints'] is None:
            return None
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            points_3d = output['3d_keypoints'] * scale
            
            # Define body parts for coloring
            body_parts = {
                'face': list(range(0, 11)),
                'torso': [11, 12, 23, 24],
                'left_arm': [11, 13, 15, 17, 19, 21],
                'right_arm': [12, 14, 16, 18, 20, 22],
                'left_leg': [23, 25, 27, 29, 31],
                'right_leg': [24, 26, 28, 30, 32]
            }
            
            colors = {
                'face': 'blue',
                'torso': 'red',
                'left_arm': 'green',
                'right_arm': 'cyan',
                'left_leg': 'magenta',
                'right_leg': 'yellow'
            }
            
            # Plot keypoints by body part
            for part, indices in body_parts.items():
                valid_points = []
                for idx in indices:
                    if points_3d[idx, 0] != -1 or show_invalid:
                        valid_points.append(points_3d[idx])
                
                if valid_points:
                    valid_points = np.array(valid_points)
                    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                              c=colors[part], s=50, label=part, alpha=0.8)
            
            # Draw skeleton connections
            for connection in self.detector.connections:
                start_idx, end_idx = connection
                if (points_3d[start_idx, 0] != -1 and points_3d[end_idx, 0] != -1):
                    pts = np.array([points_3d[start_idx], points_3d[end_idx]])
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'gray', alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Body Pose - Frame {output["frame"]}')
            ax.legend()
            
            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return img
            
        except ImportError:
            logger.warning("Matplotlib not available for 3D visualization")
            return None
    
    def get_joint_angles(self, keypoints_3d: np.ndarray) -> Dict[str, float]:
        """
        Calculate joint angles from 3D keypoints.
        
        Args:
            keypoints_3d: 3D body keypoints
        
        Returns:
            Dictionary of joint angles in degrees
        """
        angles = {}
        
        # Define joints for angle calculation
        joints = {
            'left_elbow': (11, 13, 15),  # shoulder, elbow, wrist
            'right_elbow': (12, 14, 16),
            'left_knee': (23, 25, 27),   # hip, knee, ankle
            'right_knee': (24, 26, 28),
            'left_shoulder': (13, 11, 23),  # elbow, shoulder, hip
            'right_shoulder': (14, 12, 24),
            'left_hip': (11, 23, 25),    # shoulder, hip, knee
            'right_hip': (12, 24, 26)
        }
        
        for joint_name, (idx1, idx2, idx3) in joints.items():
            # Check if keypoints are valid
            if (keypoints_3d[idx1, 0] != -1 and 
                keypoints_3d[idx2, 0] != -1 and 
                keypoints_3d[idx3, 0] != -1):
                
                # Calculate vectors
                v1 = keypoints_3d[idx1] - keypoints_3d[idx2]
                v2 = keypoints_3d[idx3] - keypoints_3d[idx2]
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                angles[joint_name] = angle
        
        return angles
    
    def save_tracking_data(self, filename: str):
        """Save tracked body data to file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        save_data = {
            'frames': [],
            'calibration': {
                'P1': self.P1.tolist() if hasattr(self, 'P1') else None,
                'P2': self.P2.tolist() if hasattr(self, 'P2') else None,
            }
        }
        
        for frame_data in self.tracked_bodies:
            frame_entry = {
                'frame': frame_data['frame'],
                'timestamp': frame_data['timestamp'],
                '3d_keypoints': frame_data['3d_keypoints'].tolist() if frame_data['3d_keypoints'] is not None else None,
                '2d_left': frame_data['2d_left'].tolist() if isinstance(frame_data['2d_left'], np.ndarray) else [],
                '2d_right': frame_data['2d_right'].tolist() if isinstance(frame_data['2d_right'], np.ndarray) else [],
                'visibility_left': frame_data['visibility_left'].tolist() if isinstance(frame_data['visibility_left'], np.ndarray) else [],
                'visibility_right': frame_data['visibility_right'].tolist() if isinstance(frame_data['visibility_right'], np.ndarray) else [],
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
        
        logger.info(f"Saved {len(self.tracked_bodies)} frames of body tracking data to {filename}")
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()