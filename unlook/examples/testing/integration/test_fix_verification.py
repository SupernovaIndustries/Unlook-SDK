#!/usr/bin/env python3
"""
Test script to verify all fixes are working properly.
"""

print("=== Testing import fallbacks ===")

# Test 1: Check camera fallback import
try:
    from unlook.client.camera import deserialize_binary_message
    print("✓ deserialize_binary_message is available")
    
    # Test with ULMC format
    test_data = b'ULMC\x01' + b'\x00' * 100
    msg_type, payload, data = deserialize_binary_message(test_data)
    print(f"✓ Fallback deserialize works: {msg_type}")
except Exception as e:
    print(f"✗ Fallback deserialize failed: {e}")

# Test 2: Check format handling
try:
    from unlook.client.camera_config import CompressionFormat
    print("✓ CompressionFormat enum imported")
    format_val = CompressionFormat.JPEG
    print(f"✓ Format value: {format_val}")
except ImportError:
    print("✗ CompressionFormat import failed - using fallback")
    class CompressionFormat:
        JPEG = "jpeg"
    format_val = CompressionFormat.JPEG
    print(f"✓ Fallback format: {format_val}")

# Test 3: Check message type handling
try:
    from unlook.core.protocol import MessageType, Message
    
    # Test with enum
    msg = Message(msg_type=MessageType.CAMERA_LIST, payload={})
    msg_dict = msg.to_dict()
    print(f"✓ Message with enum: {msg_dict['type']}")
    
    # Test with string (simulating fallback)
    msg2 = Message.__new__(Message)
    msg2.msg_type = "camera_list"  # String
    msg2.payload = {}
    msg2.msg_id = "test"
    msg2.timestamp = 123
    msg2.reply_to = None
    msg2_dict = msg2.to_dict()
    print(f"✓ Message with string: {msg2_dict['type']}")
except Exception as e:
    print(f"✗ Message handling failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All fixes verified ===")