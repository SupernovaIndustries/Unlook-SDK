"""
Shared utilities between client and server.
"""

import json
import uuid
import socket
import logging
import platform
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def get_machine_info() -> Dict[str, Any]:
    """
    Collects information about the machine.

    Returns:
        Dictionary with machine information
    """
    try:
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "machine": platform.machine()
        }
    except Exception as e:
        logger.error(f"Error while collecting machine information: {e}")
        return {}


def generate_uuid() -> str:
    """
    Generates a unique UUID.

    Returns:
        UUID as string
    """
    return str(uuid.uuid4())


def encode_image_to_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    """
    Encodes an image to JPEG for streaming.

    Args:
        image: Numpy image
        quality: JPEG quality (0-100)

    Returns:
        JPEG image bytes
    """
    success, encoded_img = cv2.imencode(
        '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise ValueError("Error encoding image to JPEG")

    return encoded_img.tobytes()


def decode_jpeg_to_image(jpeg_data: bytes) -> np.ndarray:
    """
    Decodes a JPEG image.

    Args:
        jpeg_data: JPEG image bytes

    Returns:
        Numpy image
    """
    nparr = np.frombuffer(jpeg_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def serialize_binary_message(msg_type: str, payload: Dict,
                             binary_data: Optional[bytes] = None,
                             format_type: str = "standard") -> bytes:
    """
    Serializes a message with binary data with support for different formats.

    Args:
        msg_type: Message type
        payload: Payload as dictionary
        binary_data: Optional binary data
        format_type: Format type ("standard", "ulmc", "jpeg_direct")

    Returns:
        Serialized message as bytes
    """
    # ULMC FORMAT: If explicitly specified or if it's a multi-camera response
    if format_type == "ulmc" or (
            format_type == "auto" and msg_type == "multi_camera_response" and "cameras" in payload):
        try:
            # Check if payload contains necessary data for ULMC format
            if "cameras" not in payload:
                logger.warning("ULMC format requested but payload doesn't contain 'cameras', using standard format")
                format_type = "standard"
            else:
                # Extract camera information from payload
                cameras = payload.get("cameras", {})

                # Start with magic bytes and version
                result = bytearray(b'ULMC')  # Magic bytes
                result.append(1)  # Format version

                # Number of cameras
                result.append(len(cameras))

                # For each camera
                for camera_id, camera_data in cameras.items():
                    # Get JPEG data
                    if isinstance(camera_data, dict) and "jpeg_data" in camera_data:
                        jpeg_data = camera_data["jpeg_data"]
                    else:
                        logger.error(f"Camera {camera_id} doesn't have valid JPEG data")
                        continue

                    # Convert camera ID to bytes
                    camera_id_bytes = camera_id.encode('utf-8')

                    # Camera ID length (1 byte)
                    result.append(len(camera_id_bytes))

                    # Camera ID
                    result.extend(camera_id_bytes)

                    # JPEG size (4 bytes, little endian)
                    result.extend(len(jpeg_data).to_bytes(4, byteorder='little'))

                    # JPEG data
                    result.extend(jpeg_data)

                # Calculate and add checksum (sum of all bytes modulo 256)
                checksum = sum(result) % 256
                result.append(checksum)

                logger.debug(f"Message serialized in ULMC format: {len(result)} bytes, {len(cameras)} cameras")
                return bytes(result)

        except Exception as e:
            logger.error(f"Error during serialization in ULMC format: {e}")
            format_type = "standard"  # Fallback to standard format

    # DIRECT JPEG: To send a JPEG image directly
    if format_type == "jpeg_direct" and binary_data and len(binary_data) >= 2 and binary_data[0:2] == b'\\xff\\xd8':
        # Send JPEG data directly without header
        logger.debug(f"Message serialized in direct JPEG format: {len(binary_data)} bytes")
        return binary_data

    # STANDARD FORMAT: Default with JSON header and binary data
    try:
        # Create header, ensuring that type is always present
        header = {
            "type": msg_type,
            "payload": payload or {},
            "timestamp": time.time(),
            "has_binary": binary_data is not None,
            "binary_size": len(binary_data) if binary_data else 0,
            "protocol_version": "1.1"  # Updated protocol version
        }

        # Convert header to JSON and then to bytes
        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json)

        # Build message: [header_size (4 bytes) | header (json) | binary_data]
        message = bytearray()
        message.extend(header_size.to_bytes(4, byteorder='little'))
        message.extend(header_json)

        if binary_data:
            message.extend(binary_data)

        logger.debug(f"Message serialized in standard format: {len(message)} bytes")
        return bytes(message)

    except Exception as e:
        # Log error and recover with an error message
        logger.error(f"Error during message serialization: {e}")

        # Create a minimal fallback header
        fallback_header = {
            "type": "error",
            "payload": {"error": f"Serialization error: {str(e)}"},
            "timestamp": time.time(),
            "has_binary": False,
            "binary_size": 0,
            "protocol_version": "1.1"
        }

        fallback_json = json.dumps(fallback_header).encode('utf-8')
        fallback_size = len(fallback_json)

        fallback_message = bytearray()
        fallback_message.extend(fallback_size.to_bytes(4, byteorder='little'))
        fallback_message.extend(fallback_json)

        return bytes(fallback_message)


def deserialize_binary_message(data: bytes) -> Tuple[str, Dict, Optional[bytes]]:
    """
    Deserializes a message with binary data with support for different formats.
    Supports standard format, ULMC format, and alternative formats with fallback.

    Args:
        data: Serialized message

    Returns:
        Tuple (message_type, payload, binary_data)
    """
    # Validity checks
    if not data or len(data) < 4:
        logger.error("Insufficient binary data for deserialization")
        return "error", {"error": "Insufficient binary data"}, None

    # ULMC FORMAT: Check if it's ULMC format (UnLook MultiCamera)
    if len(data) >= 4 and data[0:4] == b'ULMC':
        try:
            # Extract version and number of cameras
            version = data[4] if len(data) > 4 else 1
            num_cameras = data[5] if len(data) > 5 else 0

            logger.debug(f"Detected ULMC format v{version} with {num_cameras} cameras")

            # Create payload with ULMC metadata
            payload = {
                "format": "ULMC",
                "version": version,
                "num_cameras": num_cameras,
                "cameras": {}
            }

            # Current position in buffer
            pos = 6
            camera_data = {}

            # For each camera
            for _ in range(num_cameras):
                if pos >= len(data):
                    break

                # Camera ID length
                id_len = data[pos] if pos < len(data) else 0
                pos += 1

                # Camera ID
                if pos + id_len > len(data):
                    break

                camera_id = data[pos:pos + id_len].decode('utf-8')
                pos += id_len

                # JPEG size
                if pos + 4 > len(data):
                    break

                jpeg_size = int.from_bytes(data[pos:pos + 4], byteorder='little')
                pos += 4

                # Save metadata
                payload["cameras"][camera_id] = {
                    "size": jpeg_size,
                    "offset": pos
                }

                # Skip JPEG data
                pos += jpeg_size

            # Verify checksum if there's space
            if pos < len(data):
                received_checksum = data[-1]
                calculated_checksum = sum(data[:-1]) % 256

                payload["checksum_valid"] = received_checksum == calculated_checksum

            # Return type, payload and complete binary data
            return "multi_camera_response", payload, data

        except Exception as e:
            logger.error(f"Error during ULMC format parsing: {e}")

    # DIRECT JPEG: Check if data starts with JPEG SOI marker (0xFF 0xD8)
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        # It's a direct JPEG image without header
        logger.debug("Detected direct JPEG without header")
        return "camera_frame", {"format": "jpeg", "direct_image": True}, data

    # STANDARD FORMAT: Attempt standard deserialization with JSON header
    try:
        # Extract header size
        header_size = int.from_bytes(data[:4], byteorder='little')

        # Check reasonable size (avoid overflow)
        if header_size <= 0 or header_size > 100000:  # Reasonable limit for JSON header
            logger.warning(f"Suspicious header size: {header_size}, could be alternative format")

            # Try to detect if it's a multi-camera message without standard header
            if len(data) >= 8:
                # Check if after 4 bytes there's a reasonable JPEG size
                img_size = int.from_bytes(data[4:8], byteorder='little')
                if img_size > 0 and img_size < len(data) - 8:
                    logger.debug(f"Detected possible multi-camera format with image size: {img_size}")
                    return "multi_camera_response", {"format": "jpeg", "alternative_format": True}, data

            # If not recognized, return raw binary data
            return "camera_capture_response", {"format": "unknown"}, data

        # Check if there's enough data for the header
        if len(data) < 4 + header_size:
            logger.error(f"Incomplete binary data: need {4 + header_size} bytes, received {len(data)}")
            return "error", {"error": "Incomplete header"}, None

        # Extract and parse header
        header_json = data[4:4 + header_size].decode('utf-8')
        header = json.loads(header_json)

        # Debug log
        logger.debug(f"Deserialized header: {header}")

        # Extract any binary data
        binary_data = None
        if header.get("has_binary", False):
            binary_data = data[4 + header_size:]

            # Verify declared size
            declared_size = header.get("binary_size", 0)
            actual_size = len(binary_data)

            if declared_size != actual_size:
                logger.warning(
                    f"Binary data size mismatch: declared {declared_size}, received {actual_size}")

        elif len(data) > 4 + header_size:
            # If there's extra data, consider it as binary even if has_binary is False
            logger.debug("Binary data present but has_binary=False, extracting anyway")
            binary_data = data[4 + header_size:]

        # Extract type and payload with safe defaults
        msg_type = header.get("type", "unknown_message_type")
        payload = header.get("payload", {})

        # Check protocol version
        protocol_version = header.get("protocol_version", "1.0")
        if protocol_version != "1.0" and protocol_version != "1.1":
            logger.warning(f"Unrecognized protocol version: {protocol_version}")

        return msg_type, payload, binary_data

    except json.JSONDecodeError as e:
        logger.debug(f"Error in JSON header decoding: {e}")
        # If it's a JSON decoding error but the beginning looks like a JPEG
        if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
            logger.debug("Detected JPEG despite JSON decoding error")
            return "camera_frame", {"format": "jpeg", "direct_image": True}, data
        return "camera_capture_response", {"format": "raw", "error": f"Invalid JSON: {str(e)}"}, data
    except Exception as e:
        logger.debug(f"Error in binary message deserialization: {e}")
        return "camera_capture_response", {"format": "raw", "error": f"Deserialization error: {str(e)}"}, data


class RateLimiter:
    """Class for limiting call frequency."""

    def __init__(self, rate: float):
        """
        Initialize a rate limiter.

        Args:
            rate: Maximum frequency in Hz
        """
        self.min_interval = 1.0 / rate
        self.last_call = 0
        self._lock = threading.Lock()

    def wait(self):
        """
        Wait the necessary time to respect the rate limit.

        Returns:
            Time actually waited in seconds
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            wait_time = max(0, self.min_interval - elapsed)

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_call = time.time()
            return wait_time