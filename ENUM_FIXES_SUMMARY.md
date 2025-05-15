# Enum and Camera Import Fixes Summary

## Issues Fixed

### 1. Camera Import Issue
- **Problem**: Python was importing the `camera/` directory instead of `camera.py` file
- **Solution**: Used `importlib.util` to explicitly load the camera.py file by path
- **File**: `/unlook/client/scanner.py` (camera property)

### 2. Compression Format Enum Issue
- **Problem**: Code tried to access `.value` on string when using fallback classes
- **Solution**: Added checks for `hasattr(format, 'value')` before accessing
- **Files**: `/unlook/client/camera.py` (capture and capture_multi methods)

### 3. Message Type Enum Issue  
- **Problem**: Message.to_dict() tried to access `.value` on string message types
- **Solution**: Added check for `hasattr(self.msg_type, 'value')` 
- **File**: `/unlook/core/protocol.py` (to_dict method)

### 4. Camera ID Parameter Issue
- **Problem**: set_exposure() was called without required camera_id parameter
- **Solution**: Get stereo pair IDs first, then call set_exposure for each camera
- **File**: `/unlook/client/scanning/static_scanner.py`

### 5. Non-existent set_gain Method
- **Problem**: Code tried to call set_gain() which doesn't exist
- **Solution**: Removed set_gain calls; gain is set via set_exposure method
- **File**: `/unlook/client/scanning/static_scanner.py`

## Key Code Changes

1. **Format handling in camera.py**:
```python
# Handle both enum and string format types
if hasattr(format, 'value'):
    format_value = format.value
else:
    format_value = format
```

2. **Message type handling in protocol.py**:
```python
# Handle both enum and string message types
if hasattr(self.msg_type, 'value'):
    type_value = self.msg_type.value
else:
    type_value = self.msg_type
```

3. **Camera exposure setting in static_scanner.py**:
```python
left_camera, right_camera = self.client.camera.get_stereo_pair()
if left_camera and right_camera:
    self.client.camera.set_exposure(left_camera, exposure_time, gain=gain)
    self.client.camera.set_exposure(right_camera, exposure_time, gain=gain)
```

## Result
The scanner should now properly handle both enum and string types when imports fail, and correctly manage camera operations with proper parameters.