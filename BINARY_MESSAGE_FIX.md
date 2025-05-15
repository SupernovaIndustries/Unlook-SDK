# Binary Message Deserialization Fix

## Issue
The scanner was failing with error: `NameError: name 'deserialize_binary_message' is not defined`

## Root Cause
When the core imports fail, the code falls back to simple class definitions, but `deserialize_binary_message` wasn't included in the fallback definitions.

## Fix Applied
Added a fallback implementation of `deserialize_binary_message` in the ImportError exception handler in `/unlook/client/camera.py`:

```python
# Simple fallback for deserialize_binary_message
def deserialize_binary_message(data):
    """Simple fallback for deserializing binary messages."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Check for ULMC format v1 (multi-camera format)
    if data.startswith(b'ULMC\x01'):
        logger.debug("Detected ULMC format v1")
        try:
            pos = 5  # Skip ULMC\x01
            timestamp = int.from_bytes(data[pos:pos+8], byteorder='little')
            pos += 8
            num_cameras = data[pos]
            pos += 1
            
            payload = {
                "format": "ULMC",
                "version": 1,
                "timestamp": timestamp,
                "num_cameras": num_cameras,
                "cameras": {}
            }
            
            for i in range(num_cameras):
                # Camera ID length
                id_len = data[pos]
                pos += 1
                # Camera ID
                camera_id = data[pos:pos+id_len].decode('utf-8')
                pos += id_len
                # JPEG size
                jpeg_size = int.from_bytes(data[pos:pos+4], byteorder='little')
                pos += 4
                
                payload["cameras"][camera_id] = {
                    "size": jpeg_size,
                    "offset": pos
                }
                pos += jpeg_size
            
            return "multi_camera_response", payload, data
        except Exception as e:
            logger.error(f"Error parsing ULMC format: {e}")
    
    # Simple fallback - assume it's raw image data
    return "camera_frame", {"format": "jpeg"}, data
```

## Result
The scanner can now capture images successfully even when the core imports fail. The fallback implementation handles the ULMC multi-camera format correctly.

## From Scanner Log
Before fix:
```
ERROR - Error decoding multi-camera response: name 'deserialize_binary_message' is not defined
```

After fix:
```
INFO - Using fallback method to decode multi-camera response
INFO - Stereo capture completed successfully
```

The fix allows the scanner to continue working by providing a minimal implementation that handles the essential ULMC format used for multi-camera captures.