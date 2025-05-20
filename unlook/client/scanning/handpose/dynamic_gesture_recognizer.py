"""Dynamic gesture recognition for UnLook SDK.

This module provides enhanced gesture recognition using the HAGRID dataset models and tracking
techniques from the ai-forever/dynamic_gestures repository.
"""

import os
import cv2
import numpy as np
import logging
import time
import threading
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
from pathlib import Path
from functools import lru_cache

# Try importing ONNX Runtime for ONNX model support
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

# Try importing PyTorch for YOLO model support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

logger = logging.getLogger(__name__)


class HandPosition(Enum):
    """Hand position states for gesture tracking."""
    UNKNOWN = -1
    LEFT_START = 1
    RIGHT_START = 2
    LEFT_END = 3
    RIGHT_END = 4
    UP_START = 5
    UP_END = 6
    DOWN_START = 7
    DOWN_END = 8
    FAST_SWIPE_UP_START = 9
    FAST_SWIPE_UP_END = 10
    FAST_SWIPE_DOWN_START = 11
    FAST_SWIPE_DOWN_END = 12
    ZOOM_IN_START = 13
    ZOOM_IN_END = 14
    ZOOM_OUT_START = 15
    ZOOM_OUT_END = 16
    LEFT_START2 = 17
    RIGHT_START2 = 18
    LEFT_END2 = 19
    RIGHT_END2 = 20
    UP_START2 = 21
    UP_END2 = 22
    DOWN_START2 = 23
    DOWN_END2 = 24
    DRAG_START = 25
    DRAG_END = 26
    LEFT_START3 = 27
    RIGHT_START3 = 28
    LEFT_END3 = 29
    RIGHT_END3 = 30
    DOWN_START3 = 31
    DOWN_END3 = 32
    UP_START3 = 33
    UP_END3 = 34


class DynamicEvent(Enum):
    """Dynamic gesture events detected through tracking."""
    UNKNOWN = -1
    SWIPE_RIGHT = 0
    SWIPE_LEFT = 1
    SWIPE_UP = 2
    SWIPE_DOWN = 3
    DRAG = 4
    DROP = 5
    FAST_SWIPE_DOWN = 6
    FAST_SWIPE_UP = 7
    ZOOM_IN = 8
    ZOOM_OUT = 9
    SWIPE_RIGHT2 = 10
    SWIPE_LEFT2 = 11
    SWIPE_UP2 = 12
    SWIPE_DOWN2 = 13
    DOUBLE_TAP = 14
    SWIPE_RIGHT3 = 15
    SWIPE_LEFT3 = 16
    SWIPE_UP3 = 17
    SWIPE_DOWN3 = 18
    DRAG2 = 19
    DROP2 = 20
    DRAG3 = 21
    DROP3 = 22
    TAP = 23
    CIRCLE_CW = 24
    CIRCLE_CCW = 25


# HAGRID gesture labels mapping to descriptive names
GESTURE_TARGETS = [
    'hand_down',
    'hand_right',
    'hand_left',
    'thumb_index',
    'thumb_left',
    'thumb_right',
    'thumb_down',
    'half_up',
    'half_left',
    'half_right',
    'half_down',
    'part_hand_heart',
    'part_hand_heart2',
    'fist_inverted',
    'two_left',
    'two_right',
    'two_down',
    'grabbing',
    'grip',
    'point',
    'call',
    'three3',
    'little_finger',
    'middle_finger',
    'dislike',
    'fist',
    'four',
    'like',
    'mute',
    'ok',
    'one',
    'palm',
    'peace',
    'peace_inverted',
    'rock',
    'stop',
    'stop_inverted',
    'three',
    'three2',
    'two_up',
    'two_up_inverted',
    'three_gun',
    'one_left',
    'one_right', 
    'one_down'
]


class Hand:
    """Hand object for tracking and gesture recognition."""
    
    def __init__(self, bbox=None, hand_id=None, gesture=None):
        """
        Initialize Hand object.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            hand_id: Tracking ID
            gesture: Gesture class ID
        """
        self.bbox = bbox
        self.hand_id = hand_id
        self.center = None
        self.size = None
        
        if self.bbox is not None:
            self.center = self._get_center()
            self.size = self.bbox[2] - self.bbox[0]
            
        self.position = None
        self.gesture = gesture
    
    def _get_center(self):
        """Calculate center of bounding box."""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
    
    def __repr__(self):
        """String representation of hand."""
        return f"Hand(center={self.center}, size={self.size}, position={self.position}, gesture={self.gesture})"


class ActionDeque:
    """Tracks hand position and gesture history to detect dynamic gestures."""
    
    def __init__(self, maxlen=30, min_frames=15):
        """
        Initialize action tracking deque.
        
        Args:
            maxlen: Maximum length of history
            min_frames: Minimum frames for valid gestures
        """
        self.maxlen = maxlen
        self._deque = []
        self.action = None
        self.min_absolute_distance = 1.5  # Minimum distance for swipe detection
        self.min_frames = min_frames
        self.action_deque = deque(maxlen=5)
        
    def __len__(self):
        """Get length of hand history."""
        return len(self._deque)
    
    def __getitem__(self, index):
        """Get item at index."""
        return self._deque[index]
    
    def __contains__(self, item):
        """Check if position exists in history."""
        for x in self._deque:
            if x.position == item:
                return True
        return False
        
    def index_position(self, position):
        """Find index of given position in history."""
        for i in range(len(self._deque)):
            if self._deque[i].position == position:
                return i
        return -1
    
    def index_gesture(self, gesture):
        """Find index of given gesture in history."""
        for i in range(len(self._deque)):
            if self._deque[i].gesture == gesture:
                return i
        return -1
    
    def append(self, hand):
        """
        Add hand to history and check for actions.
        
        Args:
            hand: Hand object to add
        """
        if self.maxlen is not None and len(self) >= self.maxlen:
            self._deque.pop(0)
            
        self.set_hand_position(hand)
        self._deque.append(hand)
        self.check_is_action(hand)
    
    def clear(self):
        """Clear history."""
        self._deque.clear()
    
    def check_duration(self, start_index, min_frames=None):
        """
        Check if enough frames have passed since start_index.
        
        Args:
            start_index: Starting frame index
            min_frames: Minimum frames required (default: self.min_frames)
            
        Returns:
            bool: True if duration requirement met
        """
        if min_frames is None:
            min_frames = self.min_frames
            
        return len(self) - start_index >= min_frames
    
    def check_duration_max(self, start_index, max_frames=10):
        """
        Check if frames since start_index are within max_frames.
        
        Args:
            start_index: Starting frame index
            max_frames: Maximum frames allowed
            
        Returns:
            bool: True if within max duration
        """
        return len(self) - start_index <= max_frames
    
    def swipe_distance(self, first_hand, last_hand):
        """
        Check if swipe distance meets minimum threshold.
        
        Args:
            first_hand: Starting hand position
            last_hand: Ending hand position
            
        Returns:
            bool: True if swipe distance is sufficient
        """
        from scipy.spatial import distance as scipy_distance
        
        hand_dist = scipy_distance.euclidean(first_hand.center, last_hand.center)
        hand_size = (first_hand.size + last_hand.size) / 2
        return hand_dist / hand_size > self.min_absolute_distance
    
    def check_horizontal_swipe(self, start_hand, end_hand):
        """
        Check if swipe is horizontal (y-position maintained).
        
        Args:
            start_hand: Starting hand
            end_hand: Ending hand
            
        Returns:
            bool: True if horizontal swipe
        """
        boundary = [start_hand.bbox[1], start_hand.bbox[3]]
        return boundary[0] < end_hand.center[1] < boundary[1]
    
    def check_vertical_swipe(self, start_hand, end_hand):
        """
        Check if swipe is vertical (x-position maintained).
        
        Args:
            start_hand: Starting hand
            end_hand: Ending hand
            
        Returns:
            bool: True if vertical swipe
        """
        boundary = [start_hand.bbox[0], start_hand.bbox[2]]
        return boundary[0] < end_hand.center[0] < boundary[1]
    
    def set_hand_position(self, hand):
        """
        Set hand position based on gesture.
        
        Args:
            hand: Hand object to set position for
        """
        # Lookup table for gesture to position mapping
        gesture_position_mapping = {
            # Palm/stop/stop_inverted (31, 35, 36)
            31: lambda: HandPosition.UP_END if HandPosition.DOWN_START in self else HandPosition.UP_START,
            35: lambda: HandPosition.UP_END if HandPosition.DOWN_START in self else HandPosition.UP_START,
            36: lambda: HandPosition.UP_END if HandPosition.DOWN_START in self else HandPosition.UP_START,
            
            # Hand down (0)
            0: lambda: HandPosition.DOWN_END if HandPosition.UP_START in self else HandPosition.DOWN_START,
            
            # Hand right (1)
            1: lambda: HandPosition.RIGHT_END if HandPosition.LEFT_START in self else HandPosition.RIGHT_START,
            
            # Hand left (2)
            2: lambda: HandPosition.LEFT_END if HandPosition.RIGHT_START in self else HandPosition.LEFT_START,
            
            # One (30)
            30: lambda: HandPosition.FAST_SWIPE_UP_END if HandPosition.FAST_SWIPE_UP_START in self else HandPosition.FAST_SWIPE_DOWN_START,
            
            # Point (19)
            19: lambda: HandPosition.FAST_SWIPE_DOWN_END if HandPosition.FAST_SWIPE_DOWN_START in self else HandPosition.FAST_SWIPE_UP_START,
            
            # Grabbing (17)
            17: lambda: HandPosition.DRAG_START,
            
            # Fist (25)
            25: lambda: HandPosition.ZOOM_OUT_END if HandPosition.ZOOM_OUT_START in self else HandPosition.ZOOM_IN_START,
            
            # Thumb-index (3)
            3: lambda: HandPosition.ZOOM_IN_END if HandPosition.ZOOM_IN_START in self else HandPosition.ZOOM_OUT_START,
            
            # Three2 (38)
            38: lambda: HandPosition.ZOOM_IN_END if HandPosition.ZOOM_IN_START in self else HandPosition.ZOOM_OUT_START,
            
            # Thumb right (5)
            5: lambda: HandPosition.RIGHT_END2 if HandPosition.LEFT_START2 in self else HandPosition.RIGHT_START2,
            
            # Thumb left (4)
            4: lambda: HandPosition.LEFT_END2 if HandPosition.RIGHT_START2 in self else HandPosition.LEFT_START2,
            
            # Two right (15)
            15: lambda: HandPosition.RIGHT_END3 if HandPosition.LEFT_START3 in self else HandPosition.RIGHT_START3,
            
            # Two left (14)
            14: lambda: HandPosition.LEFT_END3 if HandPosition.RIGHT_START3 in self else HandPosition.LEFT_START3,
            
            # Two up (39)
            39: lambda: HandPosition.UP_END3 if HandPosition.DOWN_START3 in self else HandPosition.UP_START3,
            
            # Two down (16)
            16: lambda: HandPosition.DOWN_END3 if HandPosition.UP_START3 in self else HandPosition.DOWN_START3,
            
            # Thumb down (6)
            6: lambda: HandPosition.DOWN_END2 if HandPosition.ZOOM_OUT_START in self else HandPosition.UP_START2,
        }
        
        # Set position based on gesture
        if hand.gesture in gesture_position_mapping:
            hand.position = gesture_position_mapping[hand.gesture]()
        else:
            hand.position = HandPosition.UNKNOWN
    
    def check_is_action(self, x):
        """
        Check for dynamic gestures based on hand position history.
        
        Args:
            x: Current hand object
            
        Returns:
            bool: True if action detected
        """
        # Left swipe
        if x.position == HandPosition.LEFT_END and HandPosition.RIGHT_START in self:
            start_index = self.index_position(HandPosition.RIGHT_START)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_LEFT
                self.clear()
                return True
        
        # Right swipe
        elif x.position == HandPosition.RIGHT_END and HandPosition.LEFT_START in self:
            start_index = self.index_position(HandPosition.LEFT_START)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_RIGHT
                self.clear()
                return True
            else:
                self.clear()
        
        # Up swipe
        elif x.position == HandPosition.UP_END and HandPosition.DOWN_START in self:
            start_index = self.index_position(HandPosition.DOWN_START)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_UP
                self.clear()
                return True
            else:
                self.clear()
        
        # Down swipe
        elif x.position == HandPosition.DOWN_END and HandPosition.UP_START in self:
            start_index = self.index_position(HandPosition.UP_START)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_DOWN
                self.clear()
                return True
            else:
                self.clear()
        
        # Grip drag gesture
        elif x.gesture == 18:  # grip
            if self.action is None:
                start_index = self.index_gesture(18)
                if self.check_duration(start_index):
                    self.action = DynamicEvent.DRAG2
                    return True
        
        # Drop gesture
        elif self.action == DynamicEvent.DRAG2 and x.gesture in [11, 12]:  # hand heart
            self.action = DynamicEvent.DROP2
            self.clear()
            return True
        
        # OK drag gesture
        elif x.gesture == 29:  # ok
            if self.action is None:
                start_index = self.index_gesture(29)
                if self.check_duration(start_index):
                    self.action = DynamicEvent.DRAG3
                    return True
        
        # OK drop gesture
        elif self.action == DynamicEvent.DRAG3 and x.gesture in [11, 12]:  # hand heart
            self.action = DynamicEvent.DROP3
            self.clear()
            return True
        
        # Fast up swipe
        elif x.position == HandPosition.FAST_SWIPE_UP_END and HandPosition.FAST_SWIPE_UP_START in self:
            start_index = self.index_position(HandPosition.FAST_SWIPE_UP_START)
            if (self.check_duration(start_index, min_frames=20) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.FAST_SWIPE_UP
                self.clear()
                return True
            else:
                self.clear()
        
        # Fast down swipe
        elif x.position == HandPosition.FAST_SWIPE_DOWN_END and HandPosition.FAST_SWIPE_DOWN_START in self:
            start_index = self.index_position(HandPosition.FAST_SWIPE_DOWN_START)
            if (self.check_duration(start_index, min_frames=20) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.FAST_SWIPE_DOWN
                self.clear()
                return True
        
        # Zoom in
        elif x.position == HandPosition.ZOOM_IN_END and HandPosition.ZOOM_IN_START in self:
            start_index = self.index_position(HandPosition.ZOOM_IN_START)
            if (self.check_duration(start_index, min_frames=20) and 
                self.check_vertical_swipe(self._deque[start_index], x) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.ZOOM_IN
                self.clear()
                return True
        
        # Zoom out
        elif x.position == HandPosition.ZOOM_OUT_END and HandPosition.ZOOM_OUT_START in self:
            start_index = self.index_position(HandPosition.ZOOM_OUT_START)
            if (self.check_duration(start_index, min_frames=20) and 
                self.check_vertical_swipe(self._deque[start_index], x) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.ZOOM_OUT
                self.clear()
                return True
            else:
                self.clear()
        
        # Alternative swipe left
        elif x.position == HandPosition.LEFT_END2 and HandPosition.RIGHT_START2 in self:
            start_index = self.index_position(HandPosition.RIGHT_START2)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_LEFT2
                self.clear()
                return True
            else:
                self.clear()
        
        # Alternative swipe right
        elif x.position == HandPosition.RIGHT_END2 and HandPosition.LEFT_START2 in self:
            start_index = self.index_position(HandPosition.LEFT_START2)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_RIGHT2
                self.clear()
                return True
            else:
                self.clear()
        
        # Alternative swipe up
        elif x.position == HandPosition.UP_END2 and HandPosition.DOWN_START2 in self:
            start_index = self.index_position(HandPosition.DOWN_START2)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_UP2
                self.clear()
                return True
            else:
                self.clear()
        
        # Two finger swipe left
        elif x.position == HandPosition.LEFT_END3 and HandPosition.RIGHT_START3 in self:
            start_index = self.index_position(HandPosition.RIGHT_START3)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_LEFT3
                self.clear()
                return True
            else:
                self.clear()
        
        # Two finger swipe right
        elif x.position == HandPosition.RIGHT_END3 and HandPosition.LEFT_START3 in self:
            start_index = self.index_position(HandPosition.LEFT_START3)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_duration(start_index) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_RIGHT3
                self.clear()
                return True
            else:
                self.clear()
        
        # Two finger swipe up
        elif x.position == HandPosition.UP_END3 and HandPosition.DOWN_START3 in self:
            start_index = self.index_position(HandPosition.DOWN_START3)
            if (self.check_duration(start_index, min_frames=15) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_UP3
                self.clear()
                return True
            else:
                self.clear()
        
        # Two finger swipe down
        elif x.position == HandPosition.DOWN_END3 and HandPosition.UP_START3 in self:
            start_index = self.index_position(HandPosition.UP_START3)
            if (self.check_duration(start_index, min_frames=15) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_DOWN3
                self.clear()
                return True
            else:
                self.clear()
        
        # Drag gesture
        elif HandPosition.DRAG_START in self and x.gesture == 25:  # fist
            if self.action is None:
                start_index = self.index_gesture(17)  # grabbing
                if self.check_duration(start_index, min_frames=3):
                    self.action = DynamicEvent.DRAG
                    return True
                else:
                    self.clear()
        
        # Tap gesture 
        elif HandPosition.ZOOM_IN_START in self and x.gesture == 19:  # point
            start_index = self.index_position(HandPosition.ZOOM_IN_START)
            if (self.check_duration(start_index, min_frames=8) and 
                self.check_vertical_swipe(self._deque[start_index], x) and 
                self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.TAP
                self.clear()
                return True
            elif (self.check_duration(start_index, min_frames=2) and 
                  self.check_duration_max(start_index, max_frames=8) and 
                  self.check_vertical_swipe(self._deque[start_index], x) and 
                  self.check_horizontal_swipe(self._deque[start_index], x)):
                self.action_deque.append(DynamicEvent.TAP)
                if (len(self.action_deque) >= 2 and 
                    self.action_deque[-1] == DynamicEvent.TAP and 
                    self.action_deque[-2] == DynamicEvent.TAP):
                    self.action_deque.pop()
                    self.action_deque.pop()
                    self.action = DynamicEvent.DOUBLE_TAP
                    self.clear()
                    return True
            else:
                self.clear()
        
        # Alternative swipe down
        elif x.position == HandPosition.DOWN_END2 and HandPosition.ZOOM_OUT_START in self:
            start_index = self.index_position(HandPosition.ZOOM_OUT_START)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_DOWN2
                self.clear()
                return True
            else:
                self.clear()
        
        # Alternative swipe up
        elif x.position == HandPosition.ZOOM_OUT_START and HandPosition.UP_START2 in self:
            start_index = self.index_position(HandPosition.UP_START2)
            if (self.swipe_distance(self._deque[start_index], x) and 
                self.check_vertical_swipe(self._deque[start_index], x)):
                self.action = DynamicEvent.SWIPE_UP2
                self.clear()
                return True
            else:
                self.clear()
        
        # Drop action
        elif self.action == DynamicEvent.DRAG and x.gesture in [35, 31, 36, 17]:  # [stop, palm, stop_inverted, grabbing]
            self.action = DynamicEvent.DROP
            self.clear()
            return True
            
        return False


class OnnxModel:
    """Base class for ONNX gesture recognition models."""
    
    def __init__(self, model_path, image_size):
        """
        Initialize ONNX model.
        
        Args:
            model_path: Path to ONNX model file
            image_size: Target image size for model input
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is required but not available")
            
        self.model_path = model_path
        self.image_size = image_size
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Initialize ONNX session
        try:
            providers = ["CPUExecutionProvider"]
            # Add CUDA provider if available
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
                
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.sess = ort.InferenceSession(model_path, sess_options=options, providers=providers)
            logger.info(f"ONNX model loaded: {model_path} with providers: {self.sess.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
            
        # Get input/output details
        self._get_input_output()
    
    def _get_input_output(self):
        """Get model input and output details."""
        inputs = self.sess.get_inputs()
        self.inputs = "".join([
            f"\n {i}: {input.name}" 
            f" Shape: ({','.join(map(str, input.shape))})" 
            f" Dtype: {input.type}"
            for i, input in enumerate(inputs)
        ])
        
        outputs = self.sess.get_outputs()
        self.outputs = "".join([
            f"\n {i}: {output.name}" 
            f" Shape: ({','.join(map(str, output.shape))})" 
            f" Dtype: {output.type}"
            for i, output in enumerate(outputs)
        ])
    
    def preprocess(self, frame, fast_mode=False):
        """
        Preprocess frame for model input with optional fast mode.
        
        Args:
            frame: Input BGR image
            fast_mode: Use faster but slightly less accurate preprocessing
            
        Returns:
            Preprocessed image as numpy array
        """
        if frame is None:
            return None
            
        # Use cached preprocessing for same-sized frames when possible
        frame_id = id(frame)
        if hasattr(self, '_last_frame_id') and self._last_frame_id == frame_id:
            return self._last_preprocessed
            
        # Use faster conversion path in fast mode
        if fast_mode:
            # Single-step BGR to RGB conversion and resize
            image = cv2.cvtColor(cv2.resize(frame, self.image_size), cv2.COLOR_BGR2RGB)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to expected dimensions
            image = cv2.resize(image, self.image_size)
        
        # Normalize (using in-place operations for speed)
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        
        # Transpose to channel-first format and add batch dimension
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Ensure float32 for inference
        
        # Cache result for potential reuse
        self._last_frame_id = frame_id
        self._last_preprocessed = image
        
        return image
    
    def __repr__(self):
        """String representation of model."""
        return (
            f"Model: {os.path.basename(self.model_path)}\n"
            f"Providers: {self.sess.get_providers()}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}"
        )


class HandDetection(OnnxModel):
    """Hand detection model using ONNX."""
    
    def __init__(self, model_path, image_size=(320, 240)):
        """
        Initialize hand detection model.
        
        Args:
            model_path: Path to ONNX model file
            image_size: Target image size for model input
        """
        super().__init__(model_path, image_size)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
    
    def __call__(self, frame):
        """
        Detect hands in frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (bounding_boxes, probabilities)
        """
        # Preprocess frame
        input_tensor = self.preprocess(frame)
        
        # Run inference
        boxes, _, probs = self.sess.run(self.output_names, {self.input_name: input_tensor})
        
        # Scale boxes to image dimensions
        width, height = frame.shape[1], frame.shape[0]
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        
        return boxes.astype(np.int32), probs


class HandClassification(OnnxModel):
    """Hand gesture classification model using ONNX."""
    
    def __init__(self, model_path, image_size=(128, 128)):
        """
        Initialize hand classification model.
        
        Args:
            model_path: Path to ONNX model file
            image_size: Target image size for model input
        """
        super().__init__(model_path, image_size)
    
    @staticmethod
    def get_square(box, image):
        """
        Convert box to square with padding.
        
        Args:
            box: Bounding box coordinates (x1, y1, x2, y2)
            image: Input image for dimensions
            
        Returns:
            Square box coordinates (x1, y1, x2, y2)
        """
        height, width, _ = image.shape
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        
        # Make square by extending shorter side
        if h < w:
            y0 = y0 - int((w - h) / 2)
            y1 = y0 + w
        if h > w:
            x0 = x0 - int((h - w) / 2)
            x1 = x0 + h
        
        # Clamp to image bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width - 1, x1)
        y1 = min(height - 1, y1)
        
        return x0, y0, x1, y1
    
    def get_crops(self, frame, bboxes):
        """
        Extract hand crops from frame using bounding boxes.
        
        Args:
            frame: Input BGR image
            bboxes: List of bounding boxes
            
        Returns:
            List of cropped hand images
        """
        crops = []
        for bbox in bboxes:
            # Convert to square box
            bbox = self.get_square(bbox, frame)
            
            # Extract crop
            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Add to list
            crops.append(crop)
            
        return crops
    
    def __call__(self, image, bboxes, fast_mode=False):
        """
        Classify hand gestures in crops.
        
        Args:
            image: Input BGR image
            bboxes: List of bounding boxes
            fast_mode: Use faster processing mode
            
        Returns:
            List of class indices
        """
        # Get crops
        crops = self.get_crops(image, bboxes)
        
        # No crops, return empty result
        if not crops:
            return []
            
        # Preprocess crops in parallel using threading if multiple crops
        if len(crops) > 1:
            preprocessed_crops = []
            threads = []
            results = [None] * len(crops)
            
            def preprocess_crop(i, crop):
                results[i] = self.preprocess(crop, fast_mode=fast_mode)
            
            # Create and start threads for preprocessing
            for i, crop in enumerate(crops):
                thread = threading.Thread(target=preprocess_crop, args=(i, crop))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
            # Concatenate batches
            batch = np.concatenate(results, axis=0)
        else:
            # Only one crop, no need for threading
            preprocessed = self.preprocess(crops[0], fast_mode=fast_mode)
            batch = preprocessed
        
        # Run inference
        input_name = self.sess.get_inputs()[0].name
        outputs = self.sess.run(None, {input_name: batch})[0]
        
        # Get class indices
        labels = np.argmax(outputs, axis=1)
        
        return labels


class DynamicGestureRecognizer:
    """
    Enhanced gesture recognizer using HAGRID models or YOLOv10x for gestures and hands.
    
    This class provides improved gesture recognition using either:
    1. ONNX-based hand detection + gesture classification (44+ gestures)
    2. YOLOv10x-based gesture detection (using pre-trained YOLOv10x_gestures.pt)
    3. YOLOv10x-based hand detection (using pre-trained YOLOv10x_hands.pt)
    4. Dynamic gesture tracking (swipes, drags, etc.)
    """
    
    def __init__(self, 
                 detector_model_path=None, 
                 classifier_model_path=None,
                 yolo_model_path=None,
                 yolo_hands_model_path=None,
                 gesture_threshold=0.7,
                 maxlen=30,
                 min_frames=15,
                 model_kwargs=None,
                 parallel_inference=True):
        """
        Initialize dynamic gesture recognizer.
        
        Args:
            detector_model_path: Path to hand detector ONNX model
            classifier_model_path: Path to gesture classifier ONNX model
            gesture_threshold: Confidence threshold for gesture detection
            maxlen: Maximum length of tracking history
            min_frames: Minimum frames for valid gestures
        """
        self.gesture_threshold = gesture_threshold
        self.models_available = False
        self.using_yolo = False
        self.using_yolo_hands = False
        
        # Default model parameters
        self.model_kwargs = {
            "imgsz": 320,  # Default image size
            "verbose": False,  # Disable verbose output
            "conf": gesture_threshold,  # Use gesture_threshold as confidence threshold
            "device": self._get_optimal_device(),  # Auto-select best device
            "half": True,  # Use FP16 for faster inference
            "memory_efficient": True  # Use less memory (important for systems with limited RAM)
        }
        
        # Update with user-provided kwargs if any
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)
            
        # Parallel inference flag
        self.parallel_inference = parallel_inference
        
        # Performance optimization parameters (set by enhanced_gesture_demo.py)
        self.fast_mode = False
        self.downsample_factor = 1
        
        # Look in common locations for models
        possible_model_dirs = [
            os.path.join(os.path.dirname(__file__), "models"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "unlook", "models"),
            os.path.join(Path.home(), ".unlook", "models")
        ]
        
        # Try to find YOLOv10x gesture model if PyTorch is available
        if TORCH_AVAILABLE and yolo_model_path is None:
            for model_dir in possible_model_dirs:
                path = os.path.join(model_dir, "YOLOv10x_gestures.pt")
                if os.path.exists(path):
                    yolo_model_path = path
                    logger.info(f"Found YOLOv10x gesture model at: {yolo_model_path}")
                    break
        
        # Try to find YOLOv10x hands model if PyTorch is available
        if TORCH_AVAILABLE and yolo_hands_model_path is None:
            for model_dir in possible_model_dirs:
                path = os.path.join(model_dir, "YOLOv10x_hands.pt")
                if os.path.exists(path):
                    yolo_hands_model_path = path
                    logger.info(f"Found YOLOv10x hands model at: {yolo_hands_model_path}")
                    break
        
        # Try to load YOLOv10x models in parallel threads for faster startup
        self.model = None
        self.hands_model = None
        self.loading_complete = threading.Event()
        
        # Start model loading in separate threads
        if TORCH_AVAILABLE:
            load_threads = []
            
            # Function to load gesture model
            def load_gesture_model():
                if yolo_model_path and os.path.exists(yolo_model_path):
                    try:
                        from ultralytics import YOLO
                        model = YOLO(yolo_model_path)
                        self.model = model
                        self.models_available = True
                        self.using_yolo = True
                        logger.info(f"YOLOv10x gesture model loaded successfully from {yolo_model_path}")
                        # Set model parameters for best performance
                        if hasattr(model, 'overrides'):
                            model.overrides.update(self.model_kwargs)
                            
                        # Warmup the model with a dummy inference for faster first detection
                        try:
                            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                            model(dummy_img, **self.model_kwargs)
                        except Exception as e:
                            logger.warning(f"Model warmup failed (this is not critical): {e}")
                    except Exception as e:
                        logger.error(f"Failed to initialize YOLOv10x gesture model: {e}")
                        self.models_available = False
            
            # Function to load hands model
            def load_hands_model():
                if yolo_hands_model_path and os.path.exists(yolo_hands_model_path):
                    try:
                        from ultralytics import YOLO
                        hands_model = YOLO(yolo_hands_model_path)
                        self.hands_model = hands_model
                        self.using_yolo_hands = True
                        logger.info(f"YOLOv10x hands model loaded successfully from {yolo_hands_model_path}")
                        # Set model parameters for best performance
                        if hasattr(hands_model, 'overrides'):
                            hands_model.overrides.update(self.model_kwargs)
                            
                        # Warmup the model with a dummy inference for faster first detection
                        try:
                            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                            hands_model(dummy_img, **self.model_kwargs)
                        except Exception as e:
                            logger.warning(f"Hands model warmup failed (this is not critical): {e}")
                    except Exception as e:
                        logger.error(f"Failed to initialize YOLOv10x hands model: {e}")
                        self.using_yolo_hands = False
            
            # Start loading threads
            if self.parallel_inference and yolo_model_path and yolo_hands_model_path:
                # Load both models in parallel
                gesture_thread = threading.Thread(target=load_gesture_model)
                hands_thread = threading.Thread(target=load_hands_model)
                load_threads.extend([gesture_thread, hands_thread])
                gesture_thread.start()
                hands_thread.start()
            else:
                # Load sequentially
                load_gesture_model()
                load_hands_model()
                
            # Wait for all loading threads to complete if using parallel loading
            for thread in load_threads:
                thread.join()
                
            self.loading_complete.set()
        
        # Fall back to ONNX models if YOLO is not available
        elif ONNX_AVAILABLE:
            # Find model paths if not provided
            if detector_model_path is None or classifier_model_path is None:                
                # Try to find detector model
                if detector_model_path is None:
                    for model_dir in possible_model_dirs:
                        path = os.path.join(model_dir, "hand_detector.onnx")
                        if os.path.exists(path):
                            detector_model_path = path
                            logger.info(f"Found hand detector model at: {detector_model_path}")
                            break
                
                # Try to find classifier model
                if classifier_model_path is None:
                    for model_dir in possible_model_dirs:
                        path = os.path.join(model_dir, "crops_classifier.onnx")
                        if os.path.exists(path):
                            classifier_model_path = path
                            logger.info(f"Found gesture classifier model at: {classifier_model_path}")
                            break
            
            # Initialize ONNX models if available
            if detector_model_path and os.path.exists(detector_model_path) and \
            classifier_model_path and os.path.exists(classifier_model_path):
                try:
                    self.detection_model = HandDetection(detector_model_path)
                    self.classification_model = HandClassification(classifier_model_path)
                    self.models_available = True
                    logger.info("ONNX dynamic gesture models loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize ONNX dynamic gesture models: {e}")
                    self.models_available = False
            else:
                if detector_model_path and not os.path.exists(detector_model_path):
                    logger.warning(f"Hand detector model not found: {detector_model_path}")
                if classifier_model_path and not os.path.exists(classifier_model_path):
                    logger.warning(f"Gesture classifier model not found: {classifier_model_path}")
                self.models_available = False
        else:
            logger.warning("Neither PyTorch nor ONNX Runtime available - dynamic gesture recognition disabled")
        
        # Initialize action tracking
        self.tracks = []
        self.frame_count = 0
        
        # For tracking
        self.maxlen = maxlen
        self.min_frames = min_frames
    
    def process_frame(self, frame, fast_mode=None, downsample_factor=None):
        """
        Process a single frame for hand detection and gesture recognition.
        
        Args:
            frame: Input BGR image
            fast_mode: Use faster processing at slight accuracy cost (None = use instance default)
            downsample_factor: Factor to downsample image by (1=no downsampling, 2=half size) (None = use instance default)
            
        Returns:
            Dictionary containing:
            - bboxes: Detected hand bounding boxes
            - labels: Gesture labels for each hand
            - actions: Dynamic actions detected
            - keypoints: Hand keypoints if detected by YOLOv10x_hands model
        """
        if not self.models_available and not self.using_yolo_hands:
            return {'bboxes': [], 'labels': [], 'actions': [], 'keypoints': []}
            
        # Use provided parameters or fall back to instance variables
        fast_mode = self.fast_mode if fast_mode is None else fast_mode
        downsample_factor = self.downsample_factor if downsample_factor is None else downsample_factor
            
        self.frame_count += 1
        
        # Results container
        result = {
            'bboxes': [],
            'labels': [],
            'actions': [],
            'keypoints': [],
            'handedness': []  # Add handedness field
        }
        
        # Apply downsampling if requested (improves performance significantly)
        try:
            if downsample_factor > 1 and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                if h > 0 and w > 0:
                    new_h = max(1, h // downsample_factor)
                    new_w = max(1, w // downsample_factor)
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                else:
                    resized_frame = frame
            else:
                resized_frame = frame
        except Exception as e:
            logger.warning(f"Error in downsampling, using original frame: {e}")
            resized_frame = frame
        
        # Step 1: Process using YOLOv10x hands model if available
        if self.using_yolo_hands:
            try:
                # Run inference with YOLOv10x hands model with specified parameters
                inference_frame = resized_frame
                hands_results = self.hands_model(inference_frame, **self.model_kwargs)
                
                # Process results
                for r in hands_results:
                    boxes = r.boxes
                    keypoints = r.keypoints
                    
                    if len(boxes) == 0:
                        continue
                    
                    # Process each detected hand
                    for i, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Scale coordinates back to original size if downsampled
                        if downsample_factor > 1:
                            x1 *= downsample_factor
                            y1 *= downsample_factor
                            x2 *= downsample_factor
                            y2 *= downsample_factor
                        bbox = np.array([x1, y1, x2, y2])
                        
                        # Get confidence
                        conf = float(box.conf[0].item())
                        
                        # Get class (hand type) if available
                        if hasattr(box, 'cls') and len(box.cls) > 0:
                            cls = int(box.cls[0].item())
                            # Determine handedness (0=Left, 1=Right usually)
                            handedness = "Left" if cls == 0 else "Right" if cls == 1 else "Unknown"
                        else:
                            cls = 0
                            handedness = "Unknown"
                        
                        if conf > self.gesture_threshold:
                            result['bboxes'].append(bbox)
                            result['labels'].append(cls)
                            result['handedness'].append(handedness)
                            
                            # Get keypoints if available
                            if keypoints is not None and i < len(keypoints):
                                # Convert to numpy and reshape to match expected format
                                hand_kps = keypoints[i].data[0].cpu().numpy()
                                # YOLOv10x typically uses 21 keypoints for hands
                                if len(hand_kps) == 21:
                                    # Scale keypoints back to original size if downsampled
                                    if downsample_factor > 1:
                                        hand_kps[:, 0] *= downsample_factor
                                        hand_kps[:, 1] *= downsample_factor
                                        
                                    # Convert to normalized coordinates (0-1)
                                    h, w = frame.shape[:2]
                                    norm_kps = hand_kps.copy()
                                    norm_kps[:, 0] /= w
                                    norm_kps[:, 1] /= h
                                    # Add visibility channel if needed
                                    if norm_kps.shape[1] < 3:
                                        vis = np.ones((len(norm_kps), 1))
                                        norm_kps = np.hstack([norm_kps, vis])
                                    result['keypoints'].append(norm_kps)
                
            except Exception as e:
                logger.error(f"Error processing frame with YOLOv10x hands model: {e}")
        
        # Step 2: Process using YOLOv10x gestures model if available
        if self.using_yolo:
            try:
                # Run inference with YOLOv10x gestures model with specified parameters
                inference_frame = resized_frame
                yolo_results = self.model(inference_frame, **self.model_kwargs)
                
                # Process results
                for r in yolo_results:
                    boxes = r.boxes
                    if len(boxes) == 0:
                        continue
                        
                    # Get bounding boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Scale coordinates back to original size if downsampled
                        if downsample_factor > 1:
                            x1 *= downsample_factor
                            y1 *= downsample_factor
                            x2 *= downsample_factor
                            y2 *= downsample_factor
                        bbox = np.array([x1, y1, x2, y2])
                        
                        # Get class and confidence
                        cls = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        
                        if conf > self.gesture_threshold:
                            # Check if this bbox overlaps significantly with any hand bbox
                            # If not already in bboxes, add it
                            is_new_box = True
                            for existing_bbox in result['bboxes']:
                                iou = self._bbox_iou(tuple(bbox), tuple(existing_bbox))
                                if iou > 0.7:  # High overlap
                                    is_new_box = False
                                    break
                            
                            if is_new_box:
                                result['bboxes'].append(bbox)
                                result['labels'].append(cls)
                            
                            # Map YOLO class to dynamic gesture if confidence is high
                            if conf > 0.7:
                                # Map the YOLO class to a dynamic event
                                # This mapping will depend on your specific YOLO model training
                                # Here's a placeholder mapping based on common gestures
                                event_mapping = {
                                    0: DynamicEvent.SWIPE_RIGHT,
                                    1: DynamicEvent.SWIPE_LEFT, 
                                    2: DynamicEvent.SWIPE_UP,
                                    3: DynamicEvent.SWIPE_DOWN,
                                    4: DynamicEvent.CIRCLE_CW,
                                    5: DynamicEvent.CIRCLE_CCW,
                                    6: DynamicEvent.ZOOM_IN,
                                    7: DynamicEvent.ZOOM_OUT,
                                    8: DynamicEvent.TAP,
                                    9: DynamicEvent.DOUBLE_TAP
                                    # Add more mappings as needed
                                }
                                
                                if cls in event_mapping:
                                    result['actions'].append({
                                        'type': event_mapping[cls],
                                        'bbox': bbox,
                                        'confidence': conf
                                    })
                
            except Exception as e:
                logger.error(f"Error processing frame with YOLOv10x gestures model: {e}")
                
            # If we processed the frame with either YOLO model, return results
            if self.using_yolo or self.using_yolo_hands:
                return result
        
        # Fall back to ONNX-based processing if not using YOLO
        # Get detections
        bboxes, probs = self.detection_model(resized_frame)
        
        # Scale bounding boxes back to original size if downsampled
        if downsample_factor > 1 and len(bboxes) > 0:
            bboxes[:, 0] *= downsample_factor
            bboxes[:, 1] *= downsample_factor
            bboxes[:, 2] *= downsample_factor
            bboxes[:, 3] *= downsample_factor
        
        if len(bboxes) == 0:
            # Update tracks with empty detections
            for track in self.tracks:
                track['hands'].append(Hand(bbox=None, gesture=None))
            return result
        
        # Get classifications
        labels = self.classification_model(frame, bboxes, fast_mode=fast_mode)
        
        # Update tracks
        self._update_tracks(bboxes, probs, labels)
        
        # Collect results from active tracks
        for track in self.tracks:
            # Only include active tracks
            if track['tracker'].time_since_update < 1:
                # Get bounding box
                bbox = track['tracker'].get_state()[0]
                result['bboxes'].append(bbox)
                
                # Get latest gesture
                if len(track['hands']) > 0:
                    result['labels'].append(track['hands'][-1].gesture)
                else:
                    result['labels'].append(None)
                
                # Get dynamic action
                if track['hands'].action is not None:
                    result['actions'].append({
                        'type': track['hands'].action,
                        'bbox': bbox
                    })
        
        return result
    
    def _update_tracks(self, bboxes, probs, labels):
        """
        Update hand tracks with new detections.
        
        Args:
            bboxes: Detected bounding boxes
            probs: Detection probabilities
            labels: Gesture labels
        """
        # Combine boxes and scores
        dets = np.concatenate((bboxes, np.expand_dims(probs, axis=1)), axis=1)
        
        # For new tracks
        unmatched_dets = list(range(len(bboxes)))
        matched_track_indices = []
        
        # Match detections to existing tracks using IoU
        for i, track in enumerate(self.tracks):
            # Skip tracks that have been inactive for too long
            if track['tracker'].time_since_update > 30:  # max_age
                continue
            
            # Get track's current bbox
            track_bbox = track['tracker'].get_state()[0][:4]
            
            # Find best matching detection
            best_iou = 0
            best_det_idx = -1
            
            for j in unmatched_dets:
                # Calculate IoU - convert to tuples for caching
                det_bbox = dets[j, :4]
                track_tuple = tuple(track_bbox)
                det_tuple = tuple(det_bbox)
                iou = self._bbox_iou(track_tuple, det_tuple)
                
                # Update best match
                if iou > best_iou and iou > 0.3:  # iou_threshold
                    best_iou = iou
                    best_det_idx = j
            
            # If match found, update track
            if best_det_idx >= 0:
                # Update tracker with detection
                track['tracker'].update(dets[best_det_idx, :])
                
                # Add new hand to history
                track['hands'].append(Hand(
                    bbox=dets[best_det_idx, :4],
                    gesture=labels[best_det_idx] if best_det_idx < len(labels) else None
                ))
                
                # Mark detection and track as matched
                unmatched_dets.remove(best_det_idx)
                matched_track_indices.append(i)
        
        # Update unmatched tracks
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track['tracker'].update(None)
                track['hands'].append(Hand(bbox=None, gesture=None))
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            from .kalman_tracker import KalmanBoxTracker
            
            # Create new track
            self.tracks.append({
                'hands': ActionDeque(self.maxlen, self.min_frames),
                'tracker': KalmanBoxTracker(dets[i, :])
            })
            
            # Initialize with detection
            self.tracks[-1]['hands'].append(Hand(
                bbox=dets[i, :4],
                gesture=labels[i] if i < len(labels) else None
            ))
        
        # Remove dead tracks
        i = len(self.tracks)
        for track in reversed(self.tracks):
            i -= 1
            if track['tracker'].time_since_update > 30:  # max_age
                self.tracks.pop(i)
    
    @staticmethod
    @lru_cache(maxsize=512)  # Cache for small speedup in tracking with same boxes
    def _bbox_iou(bbox1_tuple, bbox2_tuple):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            bbox1_tuple: First bounding box (x1, y1, x2, y2) as tuple
            bbox2_tuple: Second bounding box (x1, y1, x2, y2) as tuple
            
        Returns:
            IoU score
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1_tuple
        x1_2, y1_2, x2_2, y2_2 = bbox2_tuple
        
        # Fast path for non-overlapping boxes
        if x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2:
            return 0.0
        
        # Calculate area of each box (only once)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate intersection area
        area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        
        # Calculate union area
        area_u = area1 + area2 - area_i
        
        # Return IoU (use fast division)
        return area_i / area_u
    
    def get_gesture_name(self, gesture_id):
        """
        Get human-readable name for gesture.
        
        Args:
            gesture_id: Gesture ID
            
        Returns:
            Gesture name
        """
        if gesture_id is None or gesture_id < 0 or gesture_id >= len(GESTURE_TARGETS):
            return "Unknown"
        return GESTURE_TARGETS[gesture_id]
    
    def get_action_name(self, action):
        """
        Get human-readable name for dynamic action.
        
        Args:
            action: DynamicEvent enum
            
        Returns:
            Action name
        """
        if action is None:
            return "Unknown"
        return action.name.lower().replace('_', ' ')
        
    def get_yolo_class_names(self):
        """
        Get list of class names from YOLOv10x model if available.
        
        Returns:
            List of class names or None if not using YOLO
        """
        if self.using_yolo and hasattr(self.model, 'names'):
            return self.model.names
        return None
        
    def _get_optimal_device(self):
        """Automatically determine the best available device for inference."""
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    return 0  # First CUDA device
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    return 'mps'  # Apple Metal Performance Shaders
            except Exception:
                pass
        return 'cpu'  # Fallback to CPU