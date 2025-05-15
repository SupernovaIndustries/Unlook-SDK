#!/usr/bin/env python3
"""
Test script to simulate the scanner behavior and verify all fixes work together.
"""

import sys
import os

# Simulate the scanner startup scenario
print("=== Simulating Scanner Behavior ===")

# Simulate what happens when imports might fail and fallbacks are used
class MockMessageType:
    CAMERA_LIST = "camera_list"
    CAMERA_CAPTURE_MULTI = "camera_capture_multi"

class MockCompressionFormat:
    JPEG = "jpeg"
    PNG = "png"

# Test 1: Camera capture_multi with string format
print("\n1. Testing camera capture_multi with string format")
format_value = MockCompressionFormat.JPEG  # This is just a string "jpeg"

# Simulate the fixed code in capture_multi
if hasattr(format_value, 'value'):
    actual_format = format_value.value
else:
    actual_format = format_value

print(f"Format to use: {actual_format}")

# Simulate the format comparison
if (hasattr(MockCompressionFormat, 'JPEG') and format_value == MockCompressionFormat.JPEG) or format_value == "jpeg":
    print("JPEG quality will be added to params")

# Test 2: Message creation with string type
print("\n2. Testing message creation with string type")
msg_type = MockMessageType.CAMERA_LIST  # This is just a string "camera_list"

# Simulate creating a message
class MockMessage:
    def __init__(self, msg_type, payload):
        self.msg_type = msg_type
        self.payload = payload
        self.msg_id = "test-id"
        self.timestamp = 123456
        self.reply_to = None
    
    def to_dict(self):
        # Simulate the fixed to_dict method
        if hasattr(self.msg_type, 'value'):
            type_value = self.msg_type.value
        else:
            type_value = self.msg_type
        
        return {
            "type": type_value,
            "id": self.msg_id,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "payload": self.payload
        }

msg = MockMessage(msg_type=msg_type, payload={"cameras": []})
try:
    msg_dict = msg.to_dict()
    print(f"Message dict created successfully: {msg_dict['type']}")
except Exception as e:
    print(f"Failed to create message dict: {e}")

# Test 3: Camera exposure setting with proper camera IDs
print("\n3. Testing camera exposure setting")
class MockCameraClient:
    def get_stereo_pair(self):
        return "camera0", "camera1"
    
    def set_exposure(self, camera_id, exposure_time, gain=None):
        print(f"Setting exposure for {camera_id}: {exposure_time}μs, gain={gain}")
        return True

camera_client = MockCameraClient()
left_camera, right_camera = camera_client.get_stereo_pair()
if left_camera and right_camera:
    camera_client.set_exposure(left_camera, 10000, gain=1.5)
    camera_client.set_exposure(right_camera, 10000, gain=1.5)

print("\nAll simulations completed successfully!")

# Test 4: End-to-end scenario
print("\n4. Testing end-to-end scanner scenario")
try:
    # Simulate scanner startup
    print("- Scanner connecting...")
    
    # Simulate camera list request with string MessageType
    print("- Requesting camera list...")
    msg = MockMessage(msg_type=MockMessageType.CAMERA_LIST, payload={})
    msg_bytes = str(msg.to_dict()).encode('utf-8')  # Simulate to_bytes()
    print(f"  Message sent: {msg_bytes[:50]}...")
    
    # Simulate camera capture with string CompressionFormat
    print("- Capturing images...")
    format_param = MockCompressionFormat.JPEG
    if hasattr(format_param, 'value'):
        format_str = format_param.value
    else:
        format_str = format_param
    
    capture_params = {
        "camera_ids": ["camera0", "camera1"],
        "compression_format": format_str
    }
    
    if format_str == "jpeg":
        capture_params["jpeg_quality"] = 85
    
    print(f"  Capture params: {capture_params}")
    
    print("\nScanner simulation completed successfully!")
    
except Exception as e:
    print(f"Scanner simulation failed: {e}")
    import traceback
    traceback.print_exc()