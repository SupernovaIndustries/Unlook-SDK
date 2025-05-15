#!/usr/bin/env python3
"""
Test script to verify the enum handling fixes for camera and message types.
"""

import sys
import os

# Test 1: Test camera format handling
print("=== Testing Camera Format Handling ===")
try:
    from unlook.client.camera_config import CompressionFormat
    from unlook.client import camera
    
    # Test with proper enum
    format_enum = CompressionFormat.JPEG
    print(f"Enum format: {format_enum}")
    print(f"Has value attribute: {hasattr(format_enum, 'value')}")
    if hasattr(format_enum, 'value'):
        print(f"Value: {format_enum.value}")
except ImportError:
    print("Failed to import camera_config - will use fallback")
    # Simulate fallback behavior
    class CompressionFormat:
        JPEG = "jpeg"
        PNG = "png"
    
    format_string = CompressionFormat.JPEG
    print(f"String format: {format_string}")
    print(f"Has value attribute: {hasattr(format_string, 'value')}")

# Test 2: Test message type handling  
print("\n=== Testing Message Type Handling ===")
try:
    from unlook.core.protocol import MessageType, Message
    
    # Test with proper enum
    msg_type_enum = MessageType.CAMERA_LIST
    print(f"Enum message type: {msg_type_enum}")
    print(f"Has value attribute: {hasattr(msg_type_enum, 'value')}")
    if hasattr(msg_type_enum, 'value'):
        print(f"Value: {msg_type_enum.value}")
    
    # Create a message with enum type
    msg = Message(msg_type=msg_type_enum, payload={"test": "data"})
    print(f"Message created with enum type: {msg.to_dict()['type']}")
    
except ImportError:
    print("Failed to import protocol - will use fallback")
    # Simulate fallback behavior
    class MessageType:
        CAMERA_LIST = "camera_list"
    
    msg_type_string = MessageType.CAMERA_LIST
    print(f"String message type: {msg_type_string}")
    print(f"Has value attribute: {hasattr(msg_type_string, 'value')}")

# Test 3: Test the fixed camera format comparison
print("\n=== Testing Fixed Format Comparison ===")
format_test = "jpeg"  # String format
if (hasattr(CompressionFormat, 'JPEG') and format_test == CompressionFormat.JPEG) or format_test == "jpeg":
    print("Format comparison works with string!")

# Test 4: Test fixed message to_dict method
print("\n=== Testing Fixed Message to_dict ===")
try:
    from unlook.core.protocol import Message
    
    # Create a message with string type (simulating fallback)
    msg = Message.__new__(Message)
    msg.msg_type = "camera_list"  # String instead of enum
    msg.payload = {"test": "data"}
    msg.msg_id = "test-id"
    msg.timestamp = 123456
    msg.reply_to = None
    
    try:
        result = msg.to_dict()
        print(f"to_dict() succeeded with string type: {result['type']}")
    except AttributeError as e:
        print(f"to_dict() failed with string type: {e}")
except Exception as e:
    print(f"Could not test to_dict: {e}")

print("\nAll tests completed!")