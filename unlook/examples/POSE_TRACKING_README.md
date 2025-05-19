# UnLook Pose Tracking Examples

This directory contains examples of hand and body pose tracking using the UnLook 3D scanner hardware.

## Overview

The pose tracking examples demonstrate how to use the UnLook scanner's stereo cameras for real-time 3D hand and body pose detection. These examples integrate MediaPipe for 2D pose detection with UnLook's stereo vision capabilities for full 3D tracking.

## Features

- **Real-time 3D hand tracking**: Detects and tracks up to 2 hands in 3D space
- **Real-time 3D body tracking**: Detects and tracks full body poses with 33 landmarks
- **Stereo triangulation**: Uses calibrated stereo cameras for accurate depth estimation
- **Headless operation**: Supports both GUI and headless modes for different environments
- **Data recording**: Save tracking data for later analysis

## Examples

### Hand Pose Tracking

- `handpose_demo_unlook.py`: Interactive GUI version with real-time display
- `handpose_demo_unlook_headless.py`: Headless version that saves to video file

### Body Pose Tracking

- `bodypose_demo_unlook.py`: Interactive GUI version with real-time display  
- `bodypose_demo_unlook_headless.py`: Headless version that saves to video file

## Requirements

- UnLook scanner hardware (connected and discoverable on network)
- Python 3.7+
- MediaPipe (`pip install mediapipe`)
- OpenCV (`pip install opencv-python`)
- NumPy
- Matplotlib (optional, for 3D visualization)

## Usage

### Basic Hand Tracking
```bash
python handpose_demo_unlook.py
```

### Hand Tracking with Calibration
```bash
python handpose_demo_unlook.py --calibration path/to/stereo_calibration.json
```

### Headless Body Tracking (30 second recording)
```bash
python bodypose_demo_unlook_headless.py --duration 30 --video-output body_tracking.mp4
```

### Body Tracking with 3D Visualization
```bash
python bodypose_demo_unlook.py --visualize-3d
```

### Save Tracking Data
```bash
python handpose_demo_unlook.py --output hand_tracking_data.json
```

## Calibration

For accurate 3D reconstruction, you need a stereo calibration file. The examples will search for calibration files in these locations:

1. `calibration/custom/stereo_calibration.json`
2. `calibration/default/default_stereo.json`
3. `stereo_calibration.json` (current directory)

If no calibration is found, the demos will still run but without 3D reconstruction capabilities.

## Output

### Interactive Mode
- Live visualization of left and right camera views
- 2D pose overlays on each camera view
- Optional 3D visualization window
- Real-time tracking statistics

### Headless Mode
- Saves tracking video to MP4 file
- Records tracking data to JSON (if specified)
- Shows progress updates in console
- Summary statistics at completion

## Keyboard Controls (Interactive Mode)

- `q`: Quit the application
- `s`: Save current tracking data
- `v`: Toggle 3D visualization

## Performance Tips

1. Ensure good lighting conditions for better detection
2. Keep subjects within the camera field of view
3. Maintain appropriate distance from cameras (0.5-2m)
4. Use calibrated cameras for best 3D accuracy

## Troubleshooting

### No Scanner Found
- Check that the UnLook scanner is powered on
- Verify network connection
- Try increasing the discovery timeout

### Poor Tracking Quality
- Improve lighting conditions
- Check camera focus
- Ensure calibration is up to date
- Verify subject is within optimal range

### Missing Dependencies
```bash
pip install mediapipe opencv-python numpy matplotlib
```

## Notes

- The UnLook scanner must be connected to the same network as your computer
- First run may take longer due to MediaPipe model downloads
- 3D reconstruction requires proper stereo calibration
- Headless mode is recommended for remote or server environments