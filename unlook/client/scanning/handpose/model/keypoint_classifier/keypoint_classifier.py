"""
Hand gesture keypoint classifier using machine learning.

This module provides a neural network classifier to recognize hand gestures
based on the position of hand keypoints. It's designed to work with MediaPipe
hand keypoints.
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available for Keypoint Classifier, using fallback classifier")

# Default path to built-in model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'keypoint_classifier.tflite')

class KeyPointClassifier:
    """
    Hand gesture classifier using extracted keypoints.
    
    This class provides methods to preprocess hand keypoints from MediaPipe
    and classify them into gestures using a TensorFlow Lite model.
    
    If TensorFlow is not available, a simpler rule-based classifier is used.
    """
    
    def __init__(self, model_path: str = None, num_threads: int = 1, confidence_threshold: float = 0.5):
        """
        Initialize KeyPointClassifier.
        
        Args:
            model_path: Path to TensorFlow Lite model
            num_threads: Number of interpreter threads
            confidence_threshold: Minimum confidence for classification
        """
        self.confidence_threshold = confidence_threshold
        
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
        
        # Get finger names for easier referencing
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        # Create mapping from MediaPipe hand landmarks to our structure
        self.landmark_indices = self._create_landmark_indices()
                    
    def _create_landmark_indices(self) -> Dict[str, List[int]]:
        """
        Create mapping from finger names to landmark indices.
        
        Returns:
            Dictionary mapping finger names to lists of landmark indices
        """
        # MediaPipe hand landmarks indices:
        # WRIST = 0
        # THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4
        # INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8
        # MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12
        # RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16
        # PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20
        
        # Create a dictionary mapping finger names to indices
        mapping = {
            'wrist': [0],
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        return mapping
                    
    def predict(self, landmark_list: List[List[float]]) -> Tuple[int, float]:
        """
        Classify hand gesture using keypoints.
        
        Args:
            landmark_list: List of hand landmarks from MediaPipe
            
        Returns:
            Tuple of (predicted class index, confidence score)
        """
        # Preprocess landmarks
        preprocessed_landmark_list = self._pre_process_landmark(landmark_list)
        
        # Use ML model if available
        if TF_AVAILABLE and self.interpreter is not None:
            # Convert to numpy array
            input_data = np.array(preprocessed_landmark_list, dtype=np.float32)
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
                return -1, confidence  # Unknown gesture
            
            return predicted_index, confidence
        else:
            # Fallback to rule-based classification if ML model not available
            return self._rule_based_classify(preprocessed_landmark_list)
    
    def _pre_process_landmark(self, landmark_list: List[List[float]]) -> List[float]:
        """
        Preprocess hand landmarks for classification.
        
        Steps:
        1. Convert to relative coordinates
        2. Normalize coordinates
        3. Flatten to feature vector
        
        Args:
            landmark_list: List of hand landmarks from MediaPipe
            
        Returns:
            Preprocessed feature vector
        """
        # Convert 3D array to 2D array
        temp_landmark_list = []
        for landmark in landmark_list:
            temp_landmark_list.append([landmark[0], landmark[1]])
            
        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
        for i in range(len(temp_landmark_list)):
            temp_landmark_list[i][0] = temp_landmark_list[i][0] - base_x
            temp_landmark_list[i][1] = temp_landmark_list[i][1] - base_y
            
        # Normalization
        max_value = max(
            map(lambda x: max(abs(x[0]), abs(x[1])), temp_landmark_list))
        
        if max_value == 0:
            logger.debug("Warning: Hand landmarks have 0 max value")
            return [0.0] * (len(temp_landmark_list) * 2)
            
        for i in range(len(temp_landmark_list)):
            temp_landmark_list[i][0] = temp_landmark_list[i][0] / max_value
            temp_landmark_list[i][1] = temp_landmark_list[i][1] / max_value
            
        # Flatten to feature vector
        feature_vector = []
        for landmark in temp_landmark_list:
            feature_vector.append(landmark[0])
            feature_vector.append(landmark[1])
            
        return feature_vector
    
    def _rule_based_classify(self, preprocessed_landmarks: List[float]) -> Tuple[int, float]:
        """
        Rule-based gesture classification as fallback.
        
        Args:
            preprocessed_landmarks: Preprocessed hand landmarks
            
        Returns:
            Tuple of (predicted class index, confidence score)
        """
        # Convert from flat list back to (x,y) coordinates
        landmarks = []
        for i in range(0, len(preprocessed_landmarks), 2):
            x = preprocessed_landmarks[i]
            y = preprocessed_landmarks[i+1]
            landmarks.append([x, y])
        
        # Calculate finger states (extended or folded)
        finger_states = self._calculate_finger_states(landmarks)
        
        # Recognize gestures based on finger states
        # 0: Open Palm
        if all(finger_states.values()):
            return 0, 0.9
            
        # 1: Closed Fist
        if not any(finger_states.values()):
            return 1, 0.9
            
        # 2: Pointing (index finger only)
        if finger_states['index'] and not finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']:
            return 2, 0.9
            
        # 3: Victory/Peace (index and middle finger)
        if finger_states['index'] and finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']:
            return 3, 0.9
            
        # 4: Thumbs Up
        if finger_states['thumb'] and not finger_states['index'] and not finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']:
            # Check if thumb is pointing upward
            thumb_tip = landmarks[4]  # Thumb tip
            wrist = landmarks[0]  # Wrist
            if thumb_tip[1] < wrist[1]:  # In normalized coords, smaller y is higher
                return 4, 0.9
                
        # 5: OK sign (circle with thumb and index)
        # Calculate distance between thumb tip and index tip
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        if distance < 0.1 and finger_states['middle'] and finger_states['ring'] and finger_states['pinky']:
            return 5, 0.9
        
        # Default: unknown gesture
        return -1, 0.5
        
    def _calculate_finger_states(self, landmarks: List[List[float]]) -> Dict[str, bool]:
        """
        Calculate if each finger is extended or folded.
        
        Args:
            landmarks: Hand landmarks as (x,y) coordinates
            
        Returns:
            Dictionary with finger states (True = extended, False = folded)
        """
        states = {}
        
        # Wrist and finger bases
        wrist = landmarks[0]
        finger_bases = {
            'thumb': landmarks[2],  # THUMB_MCP
            'index': landmarks[5],  # INDEX_MCP
            'middle': landmarks[9],  # MIDDLE_MCP
            'ring': landmarks[13],  # RING_MCP
            'pinky': landmarks[17]  # PINKY_MCP
        }
        
        # Finger tips
        finger_tips = {
            'thumb': landmarks[4],  # THUMB_TIP
            'index': landmarks[8],  # INDEX_TIP
            'middle': landmarks[12],  # MIDDLE_TIP
            'ring': landmarks[16],  # RING_TIP
            'pinky': landmarks[20]  # PINKY_TIP
        }
        
        # Calculate center of palm as average of finger bases
        palm_center = [0, 0]
        for base in finger_bases.values():
            palm_center[0] += base[0]
            palm_center[1] += base[1]
        palm_center[0] /= len(finger_bases)
        palm_center[1] /= len(finger_bases)
        
        # Check each finger
        for finger, tip in finger_tips.items():
            base = finger_bases[finger]
            
            # Distance from tip to palm center
            tip_to_palm = np.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            
            # Distance from base to palm center
            base_to_palm = np.sqrt((base[0] - palm_center[0])**2 + (base[1] - palm_center[1])**2)
            
            # Finger is extended if tip is further from palm than base
            states[finger] = tip_to_palm > base_to_palm * 1.1
        
        return states
    
    def get_gesture_name(self, class_id: int) -> str:
        """
        Get human-readable name for gesture class.
        
        Args:
            class_id: Gesture class index
            
        Returns:
            Gesture name as string
        """
        # Default gesture names for rule-based classifier
        gesture_names = {
            0: "Open Palm",
            1: "Fist",
            2: "Pointing",
            3: "Victory",
            4: "Thumbs Up",
            5: "OK",
            -1: "Unknown"
        }
        
        return gesture_names.get(class_id, "Unknown")