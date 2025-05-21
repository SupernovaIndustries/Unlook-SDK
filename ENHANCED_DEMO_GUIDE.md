# UnLook Enhanced Gesture Demo Guide

This guide explains how to use the Enhanced Gesture Demo effectively for investor presentations and demonstrations without requiring ML dependencies.

## Presentation Mode

The Enhanced Gesture Demo now includes a special "Presentation Mode" optimized for investor demonstrations. This mode:

- Works without any ML components or dependencies
- Provides reliable gesture recognition using the basic tracking system
- Has optimized parameters for better detection during demos
- Maintains all visual feedback features of the full version

### Running in Presentation Mode

To run the demo in presentation mode:

```bash
python unlook/examples/enhanced_gesture_demo.py --presentation-mode
```

This mode automatically:
- Disables all ML components (YOLO, dynamic gesture recognizer)
- Uses balanced performance settings for smooth operation
- Optimizes hand tracking parameters for better responsiveness
- Shows "PRESENTATION MODE" indicator on the display

### Other Useful Options

Combine presentation mode with these options for the best demo experience:

```bash
# Run with auto LED control (LED activates only when hands are detected)
python unlook/examples/enhanced_gesture_demo.py --presentation-mode

# Run with always-on LED for consistent lighting
python unlook/examples/enhanced_gesture_demo.py --presentation-mode --always-on-led

# Run with higher performance on slower machines
python unlook/examples/enhanced_gesture_demo.py --presentation-mode --downsample 4
```

## Demo Performance Tips

1. **Lighting Conditions**: Ensure good, consistent lighting for best hand detection.

2. **Hand Position**: Keep hands within 30-60cm from the cameras for best tracking.

3. **Background**: Use a simple, uncluttered background for more reliable detection.

4. **Camera Setup**: Ensure cameras are properly positioned and calibrated.

5. **LED Illumination**: The LED will activate only when hands are detected by default. Use `--always-on-led` for consistent lighting.

## Recognized Gestures

The presentation mode recognizes these basic gestures:

- Open Palm
- Closed Fist
- Pointing
- Peace Sign
- Thumbs Up/Down
- Pinch
- Wave

## Troubleshooting

If you experience any issues during presentations:

1. **Poor Detection**: Try increasing LED intensity with manual controls (press 'l' to show LED controls).

2. **Performance Issues**: Use `--downsample 4` option to increase performance on slower machines.

3. **Calibration Problems**: Ensure a valid calibration file is available or specified.

## Installation Requirements

The presentation mode works with minimal dependencies:
- OpenCV
- NumPy
- Basic UnLook SDK components

No ML frameworks (PyTorch, YOLO, etc.) are required for presentation mode.