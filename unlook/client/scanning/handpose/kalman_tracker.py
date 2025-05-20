"""Kalman filter-based object tracker for hand detection.

This module provides a simple Kalman filter-based tracker for bounding boxes.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def convert_bbox_to_z(bbox):
    """
    Convert bounding box [x1, y1, x2, y2] to [x, y, s, r] format for Kalman filter.
    Where:
    - x, y is the center of the box
    - s is the scale/area
    - r is the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # Scale is area
    r = w / float(h)  # Aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert Kalman state [x, y, s, r] to bounding box [x1, y1, x2, y2].
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    The filter uses a constant velocity model.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2, score]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], 
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1], 
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0], 
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], 
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0], 
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.  # Increase measurement uncertainty
        self.kf.P[4:, 4:] *= 1000.  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.1
        self.kf.Q[4:, 4:] *= 0.1

        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation = bbox[:4]
        self.observations = {}
        self.velocity = None
    
    def update(self, bbox):
        """
        Update the state of this tracker with observed bbox.
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2, score] or None if unmatched
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        if bbox is not None:
            # Update last observation
            self.last_observation = bbox[:4]
            self.observations[self.age] = bbox[:4]
            
            # Update Kalman filter state
            self.kf.update(convert_bbox_to_z(bbox[:4]))
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        # Calculate velocity
        if self.age > 1:
            # Look at the last two observations if available
            if self.age - 1 in self.observations and self.age - 2 in self.observations:
                prev_box = self.observations[self.age - 2]
                curr_box = self.observations[self.age - 1]
                
                prev_center = [(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2]
                curr_center = [(curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2]
                
                self.velocity = np.array(curr_center) - np.array(prev_center)
        
        return self.history[-1]
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class KalmanFilter:
    """
    Simple Kalman filter implementation for tracking bounding boxes.
    """
    
    def __init__(self, dim_x, dim_z):
        """
        Initialize Kalman filter.
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State vector
        self.x = np.zeros((dim_x, 1))
        
        # State transition matrix
        self.F = np.eye(dim_x)
        
        # Measurement function
        self.H = np.zeros((dim_z, dim_x))
        
        # Covariance matrix
        self.P = np.eye(dim_x)
        
        # Process noise covariance
        self.Q = np.eye(dim_x)
        
        # Measurement noise covariance
        self.R = np.eye(dim_z)
        
        # Identity matrix
        self.I = np.eye(dim_x)
    
    def predict(self):
        """
        Predict next state using the Kalman filter state propagation equation.
        """
        # Update state estimate
        self.x = np.dot(self.F, self.x)
        
        # Update state covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, z):
        """
        Update state estimate based on measurement.
        
        Args:
            z: Measurement vector
        """
        # Convert measurement to array
        z = np.atleast_2d(z).T if not isinstance(z, np.ndarray) else z.reshape(self.dim_z, 1)
        
        # Calculate innovation
        y = z - np.dot(self.H, self.x)
        
        # Calculate innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Calculate Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        self.x = self.x + np.dot(K, y)
        
        # Update state covariance
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)