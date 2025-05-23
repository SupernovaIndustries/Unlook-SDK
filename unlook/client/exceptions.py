"""
Custom exceptions for UnLook client.

This module defines all custom exceptions used throughout the client,
providing consistent error handling and clear error messages.
"""

from typing import Optional, Any


class UnlookException(Exception):
    """Base exception for all UnLook client errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        """
        Initialize UnLook exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        super().__init__(message)
        self.message = message
        self.details = details


class ConnectionError(UnlookException):
    """Raised when connection to scanner fails."""
    pass


class CameraError(UnlookException):
    """Base exception for camera-related errors."""
    pass


class CameraCaptureError(CameraError):
    """Raised when camera capture fails."""
    
    def __init__(self, camera_id: str, reason: str):
        message = f"Failed to capture from camera '{camera_id}': {reason}"
        super().__init__(message, details={'camera_id': camera_id, 'reason': reason})


class CameraNotFoundError(CameraError):
    """Raised when requested camera is not found."""
    
    def __init__(self, camera_id: str):
        message = f"Camera '{camera_id}' not found"
        super().__init__(message, details={'camera_id': camera_id})


class CalibrationError(UnlookException):
    """Base exception for calibration-related errors."""
    pass


class CalibrationNotFoundError(CalibrationError):
    """Raised when calibration data is not available."""
    
    def __init__(self, calibration_type: str = "stereo"):
        message = f"No {calibration_type} calibration data available"
        super().__init__(message, details={'type': calibration_type})


class CalibrationInvalidError(CalibrationError):
    """Raised when calibration data is invalid or corrupted."""
    
    def __init__(self, reason: str):
        message = f"Invalid calibration data: {reason}"
        super().__init__(message, details={'reason': reason})


class ProjectorError(UnlookException):
    """Base exception for projector-related errors."""
    pass


class PatternError(ProjectorError):
    """Raised when pattern generation or projection fails."""
    
    def __init__(self, pattern_type: str, reason: str):
        message = f"Pattern '{pattern_type}' error: {reason}"
        super().__init__(message, details={'pattern_type': pattern_type, 'reason': reason})


class ScanningError(UnlookException):
    """Base exception for scanning-related errors."""
    pass


class InsufficientDataError(ScanningError):
    """Raised when not enough data is available for reconstruction."""
    
    def __init__(self, data_type: str, expected: int, actual: int):
        message = f"Insufficient {data_type}: expected {expected}, got {actual}"
        super().__init__(message, details={
            'data_type': data_type,
            'expected': expected,
            'actual': actual
        })


class ReconstructionError(ScanningError):
    """Raised when 3D reconstruction fails."""
    
    def __init__(self, stage: str, reason: str):
        message = f"Reconstruction failed at {stage}: {reason}"
        super().__init__(message, details={'stage': stage, 'reason': reason})


class StreamingError(UnlookException):
    """Base exception for streaming-related errors."""
    pass


class StreamTimeoutError(StreamingError):
    """Raised when stream times out."""
    
    def __init__(self, stream_type: str, timeout: float):
        message = f"{stream_type} stream timed out after {timeout}s"
        super().__init__(message, details={'stream_type': stream_type, 'timeout': timeout})


class ConfigurationError(UnlookException):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_type: str, reason: str):
        message = f"Invalid {config_type} configuration: {reason}"
        super().__init__(message, details={'config_type': config_type, 'reason': reason})


class HardwareError(UnlookException):
    """Base exception for hardware-related errors."""
    pass


class DependencyError(UnlookException):
    """Raised when required dependency is missing."""
    
    def __init__(self, dependency: str, purpose: str):
        message = f"Missing dependency '{dependency}' required for {purpose}"
        super().__init__(message, details={'dependency': dependency, 'purpose': purpose})


# Error handling utilities
def handle_camera_error(func):
    """Decorator for consistent camera error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CameraError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            # Convert generic exceptions to camera errors
            raise CameraError(f"Camera operation failed: {str(e)}", details=e) from e
    return wrapper


def handle_scanning_error(func):
    """Decorator for consistent scanning error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ScanningError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            # Convert generic exceptions to scanning errors
            raise ScanningError(f"Scanning operation failed: {str(e)}", details=e) from e
    return wrapper