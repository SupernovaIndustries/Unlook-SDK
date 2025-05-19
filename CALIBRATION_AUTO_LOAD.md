# Camera Calibration Auto-Load Feature

## Overview
Added automatic loading of calibration data when the UnLook client connects. This eliminates the need to manually specify calibration files in most cases.

## Changes Made

### 1. Camera Client Enhancement
In `unlook/client/camera/camera.py`:

- Added calibration data storage to CameraClient class
- Added `_load_default_calibration()` method that runs on initialization
- Added public methods:
  - `load_calibration(calibration_file)` - Load calibration from specific file
  - `get_calibration()` - Get loaded calibration data
  - `get_calibration_file_path()` - Get path to default calibration file

### 2. Default Calibration Location
The system automatically looks for calibration at:
```
unlook/calibration/default/default_stereo.json
```

### 3. Updated Handpose Examples
All handpose examples now use auto-loaded calibration:

- `handpose_demo_unlook_fixed.py`
- `handpose_demo_unlook_headless.py`
- `hand_gesture_demo.py`
- `gesture_control_demo.py`
- `gesture_drawing_demo.py`

### 4. Usage Pattern
Examples now follow this pattern:
```python
# Connect to client
client.connect(scanner)

# Get calibration file (auto-loaded by camera client)
calibration_file = client.camera.get_calibration_file_path()
if calibration_file:
    print(f"Using auto-loaded calibration: {calibration_file}")

# Initialize tracker with calibration
tracker = HandTracker(calibration_file=calibration_file, max_num_hands=2)
```

## Handpose Tracking Improvements

### 1. Fixed Array Reshaping Issues
- Enhanced `_triangulate_points()` method to handle various input shapes
- Added shape debugging to identify issues
- Better handling of 1D/2D/3D coordinate arrays
- Automatic detection of whether points are in x,y or x,y,z format

### 2. Improved Gesture Recognition
- Added support for both normalized (0-1) and pixel coordinates 
- Automatic coordinate system detection
- Enhanced thresholds for different coordinate systems
- Added 2D gesture recognition when 3D isn't available

### 3. Updated Examples
- Removed redundant calibration searching code
- Added debug output to show detection results
- Better error handling and shape debugging
- Consistent use of streaming API

### 4. Projection Matrix Fixes
- Fixed computation of projection matrices
- Better handling of calibration parameters

## Technical Details

### Shape Handling in Triangulation
The triangulation method now handles:
- 1D arrays (flattened coordinates)
- 2D arrays with shape (N, 2) or (N, 3)
- Automatic reshaping based on array size
- Debug logging for shape issues

### Coordinate Systems
The gesture recognizer now:
- Detects if coordinates are normalized (0-1) or pixels
- Adjusts thresholds automatically
- Works with both MediaPipe and pixel coordinates

## Benefits

1. **Simplified Usage**: Users don't need to manually specify calibration files
2. **Automatic Discovery**: System finds calibration automatically on connection
3. **Fallback Support**: Examples still work without calibration (2D only)
4. **Backward Compatible**: Can still manually specify calibration files
5. **Robust Shape Handling**: Works with various input formats
6. **Better Debugging**: Enhanced logging for shape issues

## Future Improvements

1. Support for multiple calibration profiles
2. Automatic calibration selection based on camera configuration
3. Remote calibration loading from server
4. Calibration validation and error checking
5. More robust hand matching between stereo cameras
6. Better handling of partial occlusions

## Related Files

- `/unlook/client/camera/camera.py` - Camera client with auto-load feature
- `/unlook/calibration/default/default_stereo.json` - Default calibration file
- `/unlook/client/scanning/handpose/hand_tracker.py` - Enhanced triangulation
- `/unlook/client/scanning/handpose/gesture_recognizer.py` - Improved recognition
- All handpose example files - Updated to use auto-loaded calibration