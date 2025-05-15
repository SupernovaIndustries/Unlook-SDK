"""
Definition of the communication protocol between client and server.
"""

import enum
import json
import time
import uuid
from typing import Any, Dict, Optional


class MessageType(enum.Enum):
    """Message types supported by the protocol."""
    # General messages
    HELLO = "hello"  # Connection initialization
    INFO = "info"  # Scanner information
    ERROR = "error"  # General error
    PING = "ping"  # Keep-alive/connection test

    # Projector control
    PROJECTOR_MODE = "projector_mode"  # Set projector mode
    PROJECTOR_PATTERN = "projector_pattern"  # Set projector pattern
    PROJECTOR_CONFIG = "projector_config"  # Configure projector settings
    PROJECTOR_BRIGHTNESS = "projector_brightness"  # Set projector brightness
    PROJECTOR_DLP_CONFIG = "projector_dlp_config"  # DLP-specific configuration
    PROJECTOR_PATTERN_SEQUENCE = "projector_pattern_sequence"  # Define and run a sequence of patterns
    PROJECTOR_PATTERN_SEQUENCE_STEP = "projector_pattern_sequence_step"  # Move to next pattern in sequence
    PROJECTOR_PATTERN_SEQUENCE_STOP = "projector_pattern_sequence_stop"  # Stop a running pattern sequence

    # Camera control
    CAMERA_LIST = "camera_list"  # List of available cameras
    CAMERA_CONFIG = "camera_config"  # Camera configuration
    CAMERA_APPLY_SETTINGS = "camera_apply_settings"  # Force apply settings to hardware
    CAMERA_CAPTURE = "camera_capture"  # Capture image
    CAMERA_CAPTURE_MULTI = "camera_capture_multi"  # Capture image from multiple cameras
    CAMERA_STREAM_START = "camera_stream_start"  # Start streaming
    CAMERA_STREAM_STOP = "camera_stream_stop"  # Stop streaming

    # Binary response types
    CAMERA_CAPTURE_RESPONSE = "camera_capture_response"  # Response to image capture
    CAMERA_FRAME = "camera_frame"  # Camera frame for streaming
    MULTI_CAMERA_RESPONSE = "multi_camera_response"  # Response for multi-camera capture

    # Direct streaming (new)
    CAMERA_DIRECT_STREAM_START = "camera_direct_stream_start"  # Start direct streaming
    CAMERA_DIRECT_STREAM_STOP = "camera_direct_stream_stop"  # Stop direct streaming
    DIRECT_FRAME = "direct_frame"  # Direct streaming frame
    DIRECT_STREAM_CONFIG = "direct_stream_config"  # Configure direct stream
    PROJECTOR_SYNC = "projector_sync"  # Synchronize projector with stream
    
    # Camera optimization
    CAMERA_OPTIMIZE = "camera_optimize"  # Optimize camera settings automatically
    CAMERA_AUTO_FOCUS = "camera_auto_focus"  # Perform auto-focus operation
    CAMERA_TEST_CAPTURE = "camera_test_capture"  # Capture test image for optimization

    # 3D scanning
    SCAN_START = "scan_start"  # Start scan
    SCAN_STOP = "scan_stop"  # Stop scan
    SCAN_STATUS = "scan_status"  # Scan status
    SCAN_RESULT = "scan_result"  # Scan result

    # Stereo vision
    STEREO_CALIBRATION = "stereo_calibration"  # Stereo calibration
    STEREO_RECTIFY = "stereo_rectify"  # Stereo rectify

    # Structured light patterns
    PATTERN_GENERATE = "pattern_generate"  # Generate pattern
    PATTERN_PROJECT = "pattern_project"  # Project pattern

    # Calibration
    CALIBRATION_START = "calibration_start"  # Start calibration
    CALIBRATION_CAPTURE = "calibration_capture"  # Capture calibration image
    CALIBRATION_COMPUTE = "calibration_compute"  # Compute calibration
    CALIBRATION_SAVE = "calibration_save"  # Save calibration
    CALIBRATION_LOAD = "calibration_load"  # Load calibration

    # System management
    SYSTEM_STATUS = "system_status"  # System status
    SYSTEM_RESET = "system_reset"  # System reset
    SYSTEM_SHUTDOWN = "system_shutdown"  # System shutdown


class Message:
    """Class for managing protocol messages."""

    def __init__(
        self,
        msg_type: MessageType,
        payload: Optional[Dict[str, Any]] = None,
        msg_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ):
        """
        Initialize a new message.

        Args:
            msg_type: Message type
            payload: Message content
            msg_id: Unique message ID (generated if not provided)
            reply_to: ID of the message being replied to
        """
        self.msg_type = msg_type
        self.payload = payload or {}
        self.msg_id = msg_id or str(uuid.uuid4())
        self.timestamp = time.time()
        self.reply_to = reply_to

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary."""
        return {
            "type": self.msg_type.value,
            "id": self.msg_id,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "payload": self.payload
        }

    def to_json(self) -> str:
        """Convert message to JSON."""
        return json.dumps(self.to_dict())

    def to_bytes(self) -> bytes:
        """Convert message to bytes for transmission."""
        return self.to_json().encode('utf-8')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create a message from a dictionary.

        Args:
            data: Dictionary with message data

        Returns:
            Message instance
        """
        try:
            msg_type = MessageType(data.get("type"))
            return cls(
                msg_type=msg_type,
                payload=data.get("payload", {}),
                msg_id=data.get("id"),
                reply_to=data.get("reply_to")
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid message format: {e}")

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """
        Create a message from a JSON string.

        Args:
            json_str: JSON string

        Returns:
            Message instance
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """
        Create a message from bytes.

        Args:
            data: Message bytes

        Returns:
            Message instance
        """
        try:
            return cls.from_json(data.decode('utf-8'))
        except UnicodeDecodeError as e:
            raise ValueError(f"UTF-8 decoding failed: {e}")

    @classmethod
    def create_reply(cls, original_msg: 'Message', payload: Dict[str, Any],
                    msg_type: Optional[MessageType] = None) -> 'Message':
        """
        Create a reply message.

        Args:
            original_msg: Original message
            payload: Reply payload
            msg_type: Reply type (if different from the original)

        Returns:
            Reply message
        """
        return cls(
            msg_type=msg_type or original_msg.msg_type,
            payload=payload,
            reply_to=original_msg.msg_id
        )

    @classmethod
    def create_error(cls, original_msg: 'Message', error_message: str,
                    error_code: int = 500) -> 'Message':
        """
        Create an error message.

        Args:
            original_msg: Original message
            error_message: Error message
            error_code: Error code

        Returns:
            Error message
        """
        return cls(
            msg_type=MessageType.ERROR,
            payload={
                "error_message": error_message,
                "error_code": error_code
            },
            reply_to=original_msg.msg_id if original_msg else None
        )