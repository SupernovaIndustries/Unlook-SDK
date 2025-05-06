"""
Modulo core di Unlook - definizioni di base che non dipendono da altri moduli.
"""

from .constants import (
    DEFAULT_CONTROL_PORT, DEFAULT_STREAM_PORT, DEFAULT_DISCOVERY_PORT,
    DEFAULT_TIMEOUT, MAX_RETRIES, SERVICE_TYPE, SERVICE_NAME,
    DEFAULT_JPEG_QUALITY, DEFAULT_STREAM_FPS, PROTOCOL_VERSION
)

from .events import EventType, EventEmitter
from .protocol import Message, MessageType
from .discovery import DiscoveryService, ScannerInfo
from .utils import (
    get_machine_info, generate_uuid, encode_image_to_jpeg,
    decode_jpeg_to_image, serialize_binary_message, deserialize_binary_message,
    RateLimiter
)

__all__ = [
    # Constants
    'DEFAULT_CONTROL_PORT', 'DEFAULT_STREAM_PORT', 'DEFAULT_DISCOVERY_PORT',
    'DEFAULT_TIMEOUT', 'MAX_RETRIES', 'SERVICE_TYPE', 'SERVICE_NAME',
    'DEFAULT_JPEG_QUALITY', 'DEFAULT_STREAM_FPS', 'PROTOCOL_VERSION',

    # Events
    'EventType', 'EventEmitter',

    # Protocol
    'Message', 'MessageType',

    # Discovery
    'DiscoveryService', 'ScannerInfo',

    # Utils
    'get_machine_info', 'generate_uuid', 'encode_image_to_jpeg',
    'decode_jpeg_to_image', 'serialize_binary_message', 'deserialize_binary_message',
    'RateLimiter'
]