#!/usr/bin/env python3
"""Test script to verify LED message types are available."""

try:
    from unlook.core.protocol import MessageType
    
    print("LED-related MessageType attributes:")
    for attr in dir(MessageType):
        if 'LED' in attr:
            print(f"  - {attr}: {getattr(MessageType, attr).value}")
    
    print("\nSpecific LED control messages:")
    print(f"LED_SET_CURRENT: {MessageType.LED_SET_CURRENT.value}")
    print(f"LED_GET_CURRENT: {MessageType.LED_GET_CURRENT.value}") 
    print(f"LED_SET_ENABLE: {MessageType.LED_SET_ENABLE.value}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying direct import...")
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from unlook.core.protocol import MessageType
    print("\nAfter direct import:")
    print(f"LED_SET_CURRENT exists: {hasattr(MessageType, 'LED_SET_CURRENT')}")