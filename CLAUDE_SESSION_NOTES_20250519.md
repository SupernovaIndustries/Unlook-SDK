# Claude Session Notes - January 19, 2025

## Session Summary
This session focused on debugging and fixing the 3D hand reconstruction issue where hands were being detected in 2D but not matched properly for 3D triangulation.

## Main Issues Addressed

### 1. 3D Hand Reconstruction Not Working
- **Problem**: Hands were detected in both stereo cameras but not being matched for 3D reconstruction
- **Error Messages**: "no 3d hands yet" despite successful 2D detection
- **Root Causes**:
  - Incorrect handedness matching logic (was inverting handedness between cameras)
  - Overly strict matching thresholds
  - Potential frame synchronization issues

### 2. Array Reshaping Error
- **Problem**: "ValueError: cannot reshape array of size 63 into shape (2)"
- **Solution**: Fixed array reshaping logic in triangulation function to handle different input shapes

### 3. Matplotlib API Change
- **Problem**: `tostring_rgb()` method deprecated
- **Solution**: Updated to use `buffer_rgba()` method

## Code Changes Made

### 1. Hand Tracker Improvements (`unlook/client/scanning/handpose/hand_tracker.py`)

#### Enhanced Debugging
```python
logger.debug(f"Debug - Left hands data: {len(left_hands)} hands")
logger.debug(f"Debug - Right hands data: {len(right_hands)} hands")
logger.debug(f"Debug - Left handedness: {stereo_results['left'].get('handedness', [])}")
logger.debug(f"Debug - Right handedness: {stereo_results['right'].get('handedness', [])}")
```

#### Fixed Handedness Matching
```python
# MediaPipe labels hands based on the person's perspective
# So left hand is labeled "Left" in both cameras
if left_type != right_type:
    logger.debug(f"Handedness mismatch: left={left_type}, right={right_type}")
    handedness_match = False
else:
    logger.debug(f"Handedness match: both are {left_type}")
```

#### Improved Matching Algorithm
```python
# Try to match each left hand with a right hand
used_right_indices = set()

# Calculate matching score based on multiple factors
y_diff = abs(left_y - right_y)

# In stereo vision, corresponding points have similar y coordinates
# but x coordinates can vary significantly due to disparity
position_score = y_diff

# Combined score
score = position_score + handedness_score

# Relaxed threshold
threshold = 0.3  # Previously 0.1
```

#### Fixed Triangulation
```python
def _triangulate_points(self, left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    """Fixed array reshaping issues."""
    # Handle different input shapes
    if left_points.ndim == 1:
        num_coords = len(left_points)
        if num_coords % 3 == 0:
            left_points = left_points.reshape(-1, 3)
        else:
            left_points = left_points.reshape(-1, 2)
```

### 2. HandPose Demo Updates (`unlook/examples/handpose_demo_unlook_fixed.py`)

#### Better Calibration Loading
```python
# Get auto-loaded calibration if no calibration file specified
if not calibration_file:
    calibration_file = client.camera.get_calibration_file_path()
    if calibration_file:
        logger.info(f"Using auto-loaded calibration: {calibration_file}")
    else:
        logger.warning("No calibration file available")
```

#### Improved Frame Synchronization
```python
# Enable camera synchronization
logger.info("Enabling camera synchronization...")
client.camera.set_sync_mode(enabled=True)

# Wait for both frames with timeout
timeout_count = 0
while not (frame_lock['left'] and frame_lock['right']):
    time.sleep(0.001)
    timeout_count += 1
    if timeout_count > 100:  # 100ms timeout
        logger.debug("Frame sync timeout")
        frame_lock['left'] = False
        frame_lock['right'] = False
        break
```

#### Enhanced Debug Output
```python
# Debug print
if frame_count % 30 == 0:  # Print every second
    print(f"\nFrame {frame_count}:")
    print(f"  2D left hands: {len(results['2d_left'])}, 2D right hands: {len(results['2d_right'])}")
    print(f"  3D hands: {len(results['3d_keypoints'])}, Gestures: {len(results.get('gestures', []))}")
    # ... more detailed output
```

### 3. New Test Scripts Created

#### Debug Tool (`tests/test_hand_matching_debug.py`)
- Interactive debugging tool for hand detection
- Shows detected hands in both cameras side by side
- Allows saving frames for offline analysis
- Displays handedness and position information

#### Unit Test (`tests/test_simple_hand_matching.py`)
- Simple unit test for hand matching algorithm
- Tests with mock data to verify matching logic
- Validates handedness and position-based matching

## Next Steps and TODOs

### Code Polishing Needed
1. **Remove excessive debug statements** once issues are resolved
2. **Consolidate hand tracking implementations** - merge redundant code
3. **Standardize error handling** across all modules
4. **Add comprehensive docstrings** to new methods

### Documentation Cleanup
1. **Merge redundant .md files**:
   - Consolidate multiple Claude session notes
   - Merge INSTALLATION.md variants
   - Combine fix documentation files
   - Create single comprehensive README.md

2. **Update documentation**:
   - Document the fixed handedness matching approach
   - Add troubleshooting guide for common issues
   - Update calibration documentation

### Testing and Validation
1. **Thoroughly test** hand tracking across different:
   - Lighting conditions
   - Hand positions
   - Number of hands
   - Different gestures

2. **Performance optimization**:
   - Profile the matching algorithm
   - Optimize frame synchronization
   - Consider GPU acceleration for matching

### Future Improvements
1. **Implement temporal smoothing** for more stable 3D tracking
2. **Add hand tracking prediction** for missing frames
3. **Improve gesture recognition** with more gestures
4. **Add hand mesh reconstruction** using MediaPipe's hand mesh

## Summary
This session successfully debugged and fixed the 3D hand reconstruction issue. The main problems were incorrect handedness matching logic and overly strict matching criteria. The fixes improve the robustness of the stereo hand matching algorithm and add better debugging capabilities.

The codebase now needs consolidation and cleanup, particularly in documentation and removing debug code once the solution is proven stable.

## Additional Errors Fixed

### Camera Sync Mode Error
- **Problem**: `AttributeError: 'CameraClient' object has no attribute 'set_sync_mode'`
- **Solution**: Removed the non-existent method call and added comment noting synchronization may not be available in all versions

### Test Scripts Removed
- Removed `tests/test_hand_matching_debug.py` and `tests/test_simple_hand_matching.py` as requested
- These were temporary debugging scripts created during troubleshooting

### Hand Detection Improvements
- **Problem**: Hands not being detected consistently in both cameras
- **Solutions**:
  - Lowered detection confidence thresholds from 0.5 to 0.3
  - Increased max_num_hands from 2 to 4 for better detection
  - Added debug mode ('d' key) to show raw camera images
  - Added tips for better hand detection

### Custom Calibration Support
The demo already supports custom calibration files via command line:
```bash
python unlook/examples/handpose_demo_unlook_fixed.py --calibration /path/to/calibration.json
```

### Fixed Handedness Detection Issues
- **Problem**: Handedness showing as "Unknown" and duplicate hand detections in left camera
- **Solutions**:
  - Fixed handedness array access in demo to use correct fields (`handedness_left` and `handedness_right`)
  - Added duplicate detection filtering in matching algorithm
  - Implemented distance-based filtering (threshold: 0.15) to remove overlapping detections
  - This prevents false matches when MediaPipe detects the same hand multiple times

### Fixed Stereo Handedness Matching Logic
- **Problem**: Hands detected with opposite handedness in stereo cameras (left camera shows "Right", right camera shows "Left")
- **Root Cause**: MediaPipe labels hands from each camera's perspective, not the person's perspective
- **Solution**:
  - Updated matching algorithm to expect opposite handedness as the normal case in stereo
  - Applied minimal penalty (0.1) for opposite handedness (expected behavior)
  - Applied zero penalty for same handedness (rare but possible)
  - Increased matching threshold to 0.4 for more lenient matching
  - Added debug logging to track matching failures

### Added Verbose Logging Support
- Added `--verbose` flag to enable detailed debug logging
- Helps diagnose matching issues by showing:
  - Hand detection results in each camera
  - Handedness comparisons
  - Matching scores and decisions
  - Why matches fail or succeed

### AS1170 LED Flood Illuminator Integration (Server-Side)
- **Added support for AS1170-Python library on Raspberry Pi server**
- **Architecture**: Client sends commands → Server controls LED hardware
- **Hardware Configuration**:
  - I2C Bus: 4
  - Strobe Pin: GPIO 27
- **Server Implementation**:
  - Created `unlook/server/hardware/led_controller.py` with LED control logic
  - Added LED message types to protocol: `LED_SET_INTENSITY`, `LED_ON`, `LED_OFF`, `LED_STATUS`
  - Implemented message handlers in server scanner
- **Client Features**:
  - LED control via command line arguments (`--no-led`, `--led1-intensity`, `--led2-intensity`)
  - Sends control messages to server instead of direct hardware control
  - Adaptive LED control based on hand detection
  - Toggle LED intensity with 'l' key during runtime
  - Proper cleanup on exit

### Module Structure for Future Use
- **Created modular demo class**: `HandTrackingDemo` in `unlook/client/scanning/handpose/demo.py`
- **Features**:
  - Class-based interface for integration into other applications
  - Context manager support for resource cleanup
  - Convenience function `run_demo()` for quick usage
  - Configurable LED control, calibration, and tracking parameters
  - Adaptive LED control based on detection performance

### New Example Scripts
- `unlook/examples/basic/handpose_with_led.py` - Simple usage example
- `unlook/examples/basic/handpose_advanced.py` - Advanced class-based usage

## Files Modified
- `unlook/client/scanning/handpose/hand_tracker.py`
- `unlook/client/scanning/handpose/hand_detector.py`
- `unlook/examples/handpose_demo_unlook_fixed.py`
- `unlook/client/scanning/handpose/__init__.py`
- `unlook/core/protocol.py` (added LED message types)
- `unlock/server/scanner.py` (added LED handlers)
- `server-requirements.txt` (added AS1170-Python dependency)

## Files Created
- `unlook/server/hardware/led_controller.py`
- `unlook/client/scanning/handpose/demo.py`
- `unlook/examples/basic/handpose_with_led.py`
- `unlook/examples/basic/handpose_advanced.py`

## Usage Examples

### Command Line with LED Control
```bash
# Run with default LED settings (450mA)
python unlook/examples/handpose_demo_unlook_fixed.py --calibration /path/to/calibration.json

# Run with custom LED intensity
python unlook/examples/handpose_demo_unlook_fixed.py --led1-intensity 300 --led2-intensity 350

# Run without LED
python unlook/examples/handpose_demo_unlook_fixed.py --no-led

# Run with verbose debug output
python unlook/examples/handpose_demo_unlook_fixed.py --verbose
```

### Module Usage
```python
from unlook.client.scanning.handpose import run_demo

# Quick start
run_demo(use_led=True, led1_intensity=350, led2_intensity=350)

# Or use the class interface
from unlook.client.scanning.handpose import HandTrackingDemo

with HandTrackingDemo(use_led=True) as demo:
    if demo.connect() and demo.setup_tracker():
        demo.run()
```
- `CLAUDE_SESSION_NOTES_20250519.md` (this file)