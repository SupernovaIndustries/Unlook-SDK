"""Logging utilities for handpose tracking in UnLook SDK.

This module provides configurable logging functionality for hand tracking modules.
"""

import os
import logging
from datetime import datetime
import inspect
from typing import Dict, Any, Optional, List

# Global logger for this module
logger = logging.getLogger(__name__)


class HandposeLogger:
    """Specialized logger for handpose tracking with debug levels."""
    
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, name: str = "handpose", level: str = "INFO", 
                 file_logging: bool = False, log_dir: str = "./logs"):
        """
        Initialize a handpose logger with configurable options.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            file_logging: Enable logging to file
            log_dir: Directory for log files
        """
        self.name = name
        self.logger = logging.getLogger(f"unlook.client.scanning.handpose.{name}")
        
        # Set level
        self.level = self.LOG_LEVELS.get(level.upper(), logging.INFO)
        self.logger.setLevel(self.level)
        
        # Configure file logging if requested
        self.file_logging = file_logging
        if file_logging:
            self._setup_file_logging(log_dir)
    
    def _setup_file_logging(self, log_dir: str):
        """Set up file logging handler."""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            # Create timestamped log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"handpose_{self.name}_{timestamp}.log")
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.level)
            
            # Define format for file logs
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"File logging enabled: {log_file}")
        except Exception as e:
            # Fall back to console logging only
            self.logger.warning(f"Failed to setup file logging: {e}")
            self.file_logging = False
    
    def log_gesture(self, gesture_type: str, confidence: float, handedness: str = "Unknown",
                   keypoints: Optional[List] = None, is_3d: bool = False):
        """
        Log gesture recognition with optional details.
        
        Args:
            gesture_type: Type of gesture detected
            confidence: Confidence score (0-1)
            handedness: Hand orientation (Left/Right)
            keypoints: Optional keypoints used for recognition
            is_3d: Whether detection is from 3D tracking
        """
        # Basic info logging
        dimension = "3D" if is_3d else "2D"
        self.logger.info(f"Gesture[{dimension}]: {gesture_type}, confidence={confidence:.2f}, hand={handedness}")
        
        # Detailed debug logging
        if self.logger.isEnabledFor(logging.DEBUG) and keypoints is not None:
            # Get caller information for better tracing
            caller = inspect.currentframe().f_back.f_code.co_name
            self.logger.debug(f"Gesture keypoints from {caller}: {len(keypoints)} points")
    
    def log_tracking(self, tracking_data: Dict[str, Any], is_stereo: bool = True):
        """
        Log tracking results with appropriate detail level.
        
        Args:
            tracking_data: Tracking results dictionary
            is_stereo: Whether tracking is stereo or single camera
        """
        tracking_type = "Stereo" if is_stereo else "Mono"
        
        # Extract key information
        num_hands = len(tracking_data.get('3d_keypoints', []))
        gestures = [g.get('name', 'Unknown') for g in tracking_data.get('gestures', [])]
        
        # Log basic tracking info
        self.logger.info(f"{tracking_type} tracking: {num_hands} hands, gestures={gestures}")
        
        # Detailed debug logging
        if self.logger.isEnabledFor(logging.DEBUG):
            # More detailed tracking information for debugging
            confidence = tracking_data.get('confidence', [])
            self.logger.debug(f"Hand confidence: {[f'{c:.2f}' for c in confidence]}")
    
    def log_performance(self, process_time: float, frame_count: int):
        """
        Log performance metrics.
        
        Args:
            process_time: Processing time in milliseconds
            frame_count: Current frame count
        """
        if frame_count % 30 == 0:  # Log every 30 frames
            self.logger.info(f"Performance: {process_time:.1f}ms per frame")


def create_logger(name: str = "handpose", level: str = "INFO",
                 file_logging: bool = False) -> HandposeLogger:
    """
    Create a handpose logger with the specified configuration.
    
    Args:
        name: Logger name suffix
        level: Logging level
        file_logging: Enable logging to file
        
    Returns:
        Configured HandposeLogger instance
    """
    return HandposeLogger(name, level, file_logging)