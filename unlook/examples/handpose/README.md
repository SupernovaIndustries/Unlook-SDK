# UnLook Hand Pose and Gesture Recognition Examples

This directory contains examples demonstrating the hand tracking and gesture recognition capabilities of the UnLook SDK.

## Overview

The UnLook SDK provides two approaches for hand tracking:

1. **MediaPipe-based tracking**: Uses Google's MediaPipe for 2D hand detection with stereo triangulation
2. **Stereo Vision tracking**: CPU-optimized geometric algorithms that leverage UnLook's stereo cameras for robust 3D tracking without ML/GPU requirements

## Examples

### enhanced_gesture_demo.py

The main gesture recognition demo showcasing:
- Real-time 3D hand tracking using stereo vision
- Geometric gesture recognition without ML
- Flood illuminator support (LED2 at 50mA)
- Smooth tracking with Kalman filtering
- Visual feedback with gesture overlays
- Performance monitoring

**Supported Gestures:**
- Static: Open palm, Fist, Pointing, Peace, Thumbs up, Pinch, OK
- Dynamic: Swipe (up/down/left/right)

**Usage:**
```bash
python enhanced_gesture_demo.py --ip 192.168.1.100 --flood-intensity 50 --show-fps
```

**Arguments:**
- `--ip`: IP address of UnLook scanner (auto-discovery if not specified)
- `--flood-intensity`: LED2 intensity in mA (0-200, default: 50)
- `--show-fps`: Display performance statistics
- `--record`: Record video to file

## Key Features

### Stereo Vision Advantages

1. **True 3D Tracking**: Direct triangulation of hand keypoints in 3D space
2. **Depth Information**: Native depth without specialized sensors
3. **Occlusion Handling**: Robust against partial occlusions
4. **Multi-hand Support**: Track multiple hands simultaneously
5. **No GPU Required**: Runs efficiently on CPU

### Performance Optimization

- **Color-based segmentation**: Fast HSV/YCbCr skin detection
- **ROI tracking**: Reduces search area for better performance
- **Kalman filtering**: Smooth tracking and motion prediction
- **Multi-threading**: Parallel processing of stereo images
- **Optimized algorithms**: Cache-friendly data structures

### Flood Illuminator Support

The flood illuminator (LED2) improves tracking in low-light conditions:
- Uniform illumination reduces shadows
- Better skin tone detection
- Consistent performance across lighting conditions
- Automatically adjusted thresholds when active

## Technical Details

### Stereo Hand Tracking Pipeline

1. **Image Capture**: Synchronized stereo image capture
2. **Skin Detection**: Fast color-space analysis (HSV + YCbCr)
3. **Hand Detection**: Contour analysis with geometric constraints
4. **Correspondence**: Epipolar geometry for stereo matching
5. **Triangulation**: 3D reconstruction of hand keypoints
6. **Kalman Filtering**: Temporal smoothing and prediction
7. **Gesture Recognition**: Geometric feature extraction

### Geometric Gesture Recognition

Instead of ML-based approaches, we use geometric features:
- Finger extension states
- Inter-finger distances
- Hand orientation (palm normal)
- Joint angles
- Temporal motion patterns

This approach provides:
- Fast CPU-only performance
- Deterministic results
- Easy customization
- No training data required

## Future Enhancements

When the ToF camera module is integrated:
- Enhanced depth accuracy
- Better performance in challenging lighting
- Improved gesture boundaries
- Hybrid stereo+ToF fusion

## Requirements

- UnLook SDK
- OpenCV
- NumPy
- SciPy (for Kalman filtering)

No GPU or ML frameworks required!

## Tips for Best Results

1. **Lighting**: Ensure adequate lighting or use flood illuminator
2. **Background**: Plain backgrounds work best
3. **Distance**: Keep hands 30-60cm from cameras
4. **Movement**: Smooth movements for dynamic gestures
5. **Calibration**: Ensure stereo cameras are properly calibrated

## Troubleshooting

- **Poor tracking**: Increase flood illuminator intensity
- **Jittery tracking**: Adjust Kalman filter parameters
- **Missed gestures**: Check hand is fully visible in both cameras
- **Low FPS**: Reduce image resolution or disable debug visualization