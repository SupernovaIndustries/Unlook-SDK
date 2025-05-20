"""
Point history classifier for dynamic hand gesture recognition.

This module tracks hand movement over time and classifies dynamic gestures
like swipes, circles, and waves using either a TensorFlow Lite model or
rule-based classification as a fallback.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available for Point History Classifier, using fallback classifier")

# Default path to built-in model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'point_history_classifier.tflite')

class PointHistoryClassifier:
    """
    Dynamic gesture classifier using hand movement history.
    
    This class tracks the movement of hand landmarks over time and classifies
    dynamic gestures like swipes, circles, and waves.
    """
    
    # Dynamic gesture class names
    GESTURE_NAMES = {
        0: "No Gesture",
        1: "Swipe Right",
        2: "Swipe Left",
        3: "Swipe Up",
        4: "Swipe Down",
        5: "Circle",
        6: "Wave",
        -1: "Unknown"
    }
    
    def __init__(self, model_path: str = None, num_threads: int = 1, confidence_threshold: float = 0.5,
                 history_length: int = 16):
        """
        Initialize PointHistoryClassifier.
        
        Args:
            model_path: Path to TensorFlow Lite model
            num_threads: Number of interpreter threads
            confidence_threshold: Minimum confidence for classification
            history_length: Number of points to track for gesture recognition
        """
        self.confidence_threshold = confidence_threshold
        self.history_length = history_length
        
        # Initialize point history
        self.point_history = deque(maxlen=history_length)
        for _ in range(history_length):
            self.point_history.append([0, 0])
        
        # Initialize timestamp history for velocity calculations
        self.timestamp_history = deque(maxlen=history_length)
        for _ in range(history_length):
            self.timestamp_history.append(0)
        
        # Initialize interpreter if TensorFlow is available
        if TF_AVAILABLE:
            # Use default model path if none provided
            if model_path is None:
                model_path = DEFAULT_MODEL_PATH
                
            # Check if default model file exists, if not, we'll use fallback
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Using rule-based fallback classifier")
                self.interpreter = None
            else:
                try:
                    # Initialize TFLite interpreter
                    self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                                          num_threads=num_threads)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    
                    # Log successful initialization
                    logger.info(f"TensorFlow Lite model loaded: {model_path}")
                except Exception as e:
                    logger.error(f"Error loading TensorFlow Lite model: {e}")
                    logger.info("Using rule-based fallback classifier")
                    self.interpreter = None
        else:
            # TensorFlow not available
            self.interpreter = None
    
    def update_point_history(self, landmark: List[float], timestamp: float) -> None:
        """
        Update point history with new hand position.
        
        Args:
            landmark: Current hand landmark (using wrist or index finger)
            timestamp: Current timestamp for velocity calculation
        """
        self.point_history.append(landmark)
        self.timestamp_history.append(timestamp)
    
    def predict(self) -> Tuple[int, float]:
        """
        Classify dynamic gesture using point history.
        
        Returns:
            Tuple of (predicted class index, confidence score)
        """
        # Check if we have enough history for classification
        if len(self.point_history) < self.history_length:
            return 0, 0.0  # No gesture
        
        # Preprocess point history
        preprocessed_point_history = self._pre_process_point_history()
        
        # Use ML model if available
        if TF_AVAILABLE and self.interpreter is not None:
            # Convert to numpy array
            input_data = np.array(preprocessed_point_history, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get predicted class index and confidence
            predicted_index = int(np.argmax(output))
            confidence = float(output[0][predicted_index])
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return 0, confidence  # No gesture
            
            return predicted_index, confidence
        else:
            # Fallback to rule-based classification if ML model not available
            return self._rule_based_classify()
    
    def _pre_process_point_history(self) -> List[float]:
        """
        Preprocess point history for classification.
        
        Steps:
        1. Convert to relative coordinates
        2. Normalize coordinates
        3. Flatten to feature vector
        
        Returns:
            Preprocessed feature vector
        """
        # Convert to relative coordinates (relative to first point)
        base_x, base_y = self.point_history[0][0], self.point_history[0][1]
        
        # Ensure all points are valid numbers
        temp_point_history = []
        for point in self.point_history:
            # Handle potential NaN values
            x = point[0] if not np.isnan(point[0]) else base_x
            y = point[1] if not np.isnan(point[1]) else base_y
            
            # Calculate relative position
            x_rel = x - base_x
            y_rel = y - base_y
            
            temp_point_history.append([x_rel, y_rel])
            
        # Calculate normalization value (maximum absolute value in history)
        max_values = []
        for point in temp_point_history:
            max_values.append(max(abs(point[0]), abs(point[1])))
        
        max_value = max(max_values) if max_values else 1.0
        
        # Apply normalization (avoid division by zero)
        if max_value > 0:
            for i in range(len(temp_point_history)):
                temp_point_history[i][0] = temp_point_history[i][0] / max_value
                temp_point_history[i][1] = temp_point_history[i][1] / max_value
        
        # Flatten to feature vector
        feature_vector = []
        for point in temp_point_history:
            feature_vector.append(point[0])
            feature_vector.append(point[1])
            
        return feature_vector
    
    def _rule_based_classify(self) -> Tuple[int, float]:
        """
        Rule-based dynamic gesture classification as fallback.
        
        Analyzes the point history to detect common dynamic gestures
        like swipes, circles, and waves.
        
        Returns:
            Tuple of (predicted class index, confidence score)
        """
        # Make a copy of the point history for analysis
        points = list(self.point_history)
        timestamps = list(self.timestamp_history)
        
        # Calculate displacement between first and last points
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        dx = end_x - start_x
        dy = end_y - start_y
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate time difference
        time_diff = timestamps[-1] - timestamps[0]
        if time_diff <= 0.01:  # Avoid division by zero or very small time differences
            return 0, 0.0  # No gesture
        
        # Calculate velocity
        velocity = displacement / time_diff
        
        # Calculate trajectory consistency (how direct is the movement)
        # by comparing the actual path length to the direct distance
        path_length = 0
        for i in range(1, len(points)):
            prev_x, prev_y = points[i-1]
            curr_x, curr_y = points[i]
            segment_length = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            path_length += segment_length
        
        # Directness ratio (1.0 = perfectly straight line, lower = more curved)
        directness = displacement / path_length if path_length > 0 else 0
        
        # Calculate direction changes to identify patterns like circles and waves
        direction_changes = 0
        for i in range(2, len(points)):
            prev_dx = points[i-1][0] - points[i-2][0]
            prev_dy = points[i-1][1] - points[i-2][1]
            curr_dx = points[i][0] - points[i-1][0]
            curr_dy = points[i][1] - points[i-1][1]
            
            # Check if direction changes significantly
            if (prev_dx * curr_dx + prev_dy * curr_dy) < 0:
                direction_changes += 1
        
        # Calculate gesture confidence based on velocity and directness
        confidence = min(1.0, (velocity / 500) * directness)
        
        # Define minimum velocity and displacement thresholds
        MIN_VELOCITY = 100  # Minimum velocity to consider a gesture
        MIN_DISPLACEMENT = 0.05  # Minimum displacement to consider a gesture
        
        # Check if movement is significant enough to be a gesture
        if velocity < MIN_VELOCITY or displacement < MIN_DISPLACEMENT:
            return 0, confidence  # No gesture
        
        # Check for specific gesture patterns
        
        # Circle detection (many direction changes, low directness)
        if direction_changes >= 4 and directness < 0.5:
            return 5, confidence  # Circle
        
        # Wave detection (several direction changes, medium directness)
        if direction_changes >= 2 and directness < 0.7:
            return 6, confidence  # Wave
        
        # Swipe detection (few direction changes, high directness)
        if directness > 0.7:
            # Determine swipe direction based on displacement
            if abs(dx) > abs(dy):
                # Horizontal swipe
                if dx > 0:
                    return 1, confidence  # Swipe Right
                else:
                    return 2, confidence  # Swipe Left
            else:
                # Vertical swipe
                if dy > 0:
                    return 4, confidence  # Swipe Down
                else:
                    return 3, confidence  # Swipe Up
        
        # Default: no gesture
        return 0, confidence
    
    def get_gesture_name(self, class_id: int) -> str:
        """
        Get human-readable name for dynamic gesture class.
        
        Args:
            class_id: Gesture class index
            
        Returns:
            Gesture name as string
        """
        return self.GESTURE_NAMES.get(class_id, "Unknown")
    
    def reset(self) -> None:
        """Reset point history."""
        for i in range(len(self.point_history)):
            self.point_history[i] = [0, 0]
        
        for i in range(len(self.timestamp_history)):
            self.timestamp_history[i] = 0