# Projector-Camera Hardware Setup Guide

## Overview
This guide explains how to configure a projector-camera structured light system using the UnLook SDK with sinusoidal phase shift patterns.

## Hardware Configuration Options

### Option 1: Single Camera + Projector (Recommended for Phase Shift)
```
[Camera] ---- 8cm ---- [Projector]
```
- **Best for**: Phase shift structured light
- **Accuracy**: Highest precision for surface reconstruction
- **Setup**: One camera captures reflected patterns from projector

### Option 2: Dual Camera + Central Projector (Your Current Setup)
```
[Camera Left] ---- 4cm ---- [Projector] ---- 4cm ---- [Camera Right]
```
- **Best for**: Hybrid stereo + structured light
- **Accuracy**: Good for both textured and textureless surfaces
- **Setup**: Projector in center, cameras symmetrically positioned

### Option 3: Stereo Cameras + External Projector
```
[Camera Left] ---- 8cm ---- [Camera Right]
                    |
                [Projector] (offset)
```
- **Best for**: Traditional stereo vision with pattern assistance
- **Accuracy**: Good for textured surfaces

## Recommended Setup for Your Hardware

Given your configuration (2 cameras at 8cm + central projector), **Option 2** is optimal:

### Physical Positioning
1. **Central Projector**: Place DLP projector in the center
2. **Left Camera**: 4cm to the left of projector
3. **Right Camera**: 4cm to the right of projector
4. **Baseline**: Total 8cm between cameras
5. **Distance to Object**: 30-60cm for optimal results

### Calibration Requirements

#### 1. Individual Camera Calibration
```bash
# Calibrate each camera separately
python unlook/examples/calibration/process_calibration.py --camera left
python unlook/examples/calibration/process_calibration.py --camera right
```

#### 2. Stereo Camera Calibration
```bash
# Calibrate stereo pair
python unlook/examples/calibration/process_calibration.py --stereo
```

#### 3. Projector Calibration (Critical for Phase Shift)
```bash
# Calibrate projector as "inverse camera" using Gray code patterns
python unlook/examples/calibration/calibrate_projector.py --camera left --projector-resolution 1920x1080
```

### Configuration Files

#### Server Configuration (unlook_config_2k.json)
```json
{
  "name": "UnLook 2K Dual Camera + Projector",
  "server": {
    "name": "UnLookScanner_2K",
    "control_port": 5555,
    "stream_port": 5556,
    "direct_stream_port": 5557
  },
  "camera": {
    "default_resolution": [2048, 1536],
    "fps": 30,
    "cameras": {
      "left": {
        "index": 0,
        "position": "left"
      },
      "right": {
        "index": 1,
        "position": "right"
      }
    }
  },
  "projector": {
    "type": "dlp342x",
    "resolution": [1920, 1080],
    "position": "center",
    "i2c_bus": 3,
    "i2c_address": "0x1b"
  },
  "scanning": {
    "method": "phase_shift",
    "pattern_type": "sinusoidal_pattern",
    "frequencies": [1, 8, 64],
    "steps_per_frequency": 4
  }
}
```

### Calibration Strategy

#### Phase 1: Basic Calibration
1. **Camera Intrinsics**: Use checkerboard patterns
2. **Stereo Extrinsics**: Calibrate camera-to-camera relationship
3. **Projector Intrinsics**: Treat projector as inverse camera

#### Phase 2: Projector-Camera Calibration
1. **Project Gray Code**: Use structured patterns for correspondence
2. **Capture with Left Camera**: Primary camera for triangulation
3. **Calculate Projector Pose**: Relative to left camera coordinate system

#### Phase 3: System Validation
1. **Test Phase Shift Patterns**: Verify pattern quality
2. **Measure Known Objects**: Validate accuracy
3. **Optimize Parameters**: Fine-tune for your setup

### Pattern Generation Strategy

For your dual-camera setup with central projector:

#### Primary Method: Projector-Left Camera
- Use left camera as primary for phase shift reconstruction
- Projector provides structured illumination
- Achieve ~100x more points than stereo alone

#### Secondary Method: Stereo Verification
- Use right camera for correspondence validation
- Cross-check phase measurements between cameras
- Improve robustness in challenging conditions

### Software Configuration

#### Server Setup
```bash
# Start server with dual camera + projector support
python unlook/server_bootstrap.py --config unlook_config_2k.json --enable-protocol-v2
```

#### Client Usage
```python
from unlook.client.scanner import Scanner3D

# Initialize with dual camera + projector config
scanner = Scanner3D()
scanner.connect()

# Capture phase shift sequence
result = scanner.scan_3d(
    pattern_type="phase_shift",
    frequencies=[1, 8, 64],
    steps_per_frequency=4,
    use_both_cameras=True  # For validation
)
```

### Expected Performance

#### Single Camera + Projector Mode
- **3D Points**: 50,000-200,000 per scan
- **Accuracy**: ±0.1mm at 50cm distance
- **Speed**: 2-5 seconds per scan

#### Dual Camera Validation Mode
- **3D Points**: 30,000-150,000 validated points
- **Accuracy**: ±0.05mm at 50cm distance (improved)
- **Speed**: 3-7 seconds per scan

### Troubleshooting

#### Low Point Count
1. Check projector focus and intensity
2. Verify calibration accuracy
3. Adjust ambient lighting
4. Optimize pattern frequencies

#### Poor Accuracy
1. Re-calibrate projector-camera system
2. Check for mechanical vibrations
3. Verify lens distortion correction
4. Test with known reference objects

#### Pattern Quality Issues
1. Increase projector brightness
2. Reduce ambient light
3. Check for surface reflectivity
4. Adjust exposure settings

### Next Steps

1. **Test Current Setup**: Run calibration with your hardware
2. **Optimize Patterns**: Tune sinusoidal pattern parameters
3. **Validate Accuracy**: Test with known objects
4. **Fine-tune Performance**: Optimize for your specific use case

This configuration leverages both the high accuracy of phase shift patterns and the robustness of stereo vision, making it ideal for professional 3D scanning applications.