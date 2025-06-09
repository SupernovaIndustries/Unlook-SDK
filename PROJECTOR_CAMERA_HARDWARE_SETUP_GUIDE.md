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

## Single Camera + Projector Setup (Option 1 Detailed)

### Physical Configuration
```
[Camera Left] -------- 8cm -------- [Projector]
```
- **Baseline**: 8cm between camera and projector
- **Distance to Object**: 30-60cm for optimal results
- **Projector Type**: DLP342x with native resolution 1280x720 (HD)
- **Camera Resolution**: 2048x1536 (2K)

### Calibration Process for Single Camera + Projector

#### Understanding "Inverse Camera" Calibration
- **Camera**: World 3D → Lens → Sensor (captures light)
- **Projector**: Pattern → Lens → World 3D (emits light)
- Mathematically equivalent with intrinsic (focal length, distortion) and extrinsic (position/rotation) parameters

#### Step 1: Camera Calibration
```bash
# First calibrate the camera using standard checkerboard images
python unlook/examples/calibration/process_calibration.py --camera left
```

#### Step 2: Projector-Camera System Calibration
The projector must project Gray code patterns ONTO the checkerboard:
1. Position physical checkerboard (printed, not on tablet)
2. Camera detects checkerboard corners
3. Projector projects Gray code patterns on the same checkerboard
4. Camera captures checkerboard + Gray code together
5. Algorithm calculates which projector pixel illuminates each corner

```bash
# Interactive calibration (recommended)
python unlook/examples/calibration/calibrate_projector_camera.py \
    --interactive \
    --num-positions 10 \
    --gray-bits 7 \
    --projector-width 1280 \
    --projector-height 720 \
    --checkerboard-rows 9 \
    --checkerboard-cols 6 \
    --square-size 25

# Or process existing images
python unlook/examples/calibration/calibrate_projector_camera.py \
    --input calibration_images/ \
    --output projector_calibration.json \
    --projector-width 1280 \
    --projector-height 720
```

#### Step 3: Verify Calibration
```bash
python unlook/examples/calibration/check_epipolar_lines.py \
    --calibration projector_calibration.json \
    --mode projector-camera
```

### Configuration Files

#### Single Camera Configuration (unlook_config_2k_projector_left.json)
```json
{
  "name": "UnLook 2K Single Camera + Projector",
  "camera": {
    "default_resolution": [2048, 1536],
    "fps": 15,
    "active_camera": "left",
    "single_camera_mode": true
  },
  "projector": {
    "type": "dlp342x",
    "resolution": [1280, 720],  // Native DLP342x resolution
    "position": "right",        // 8cm to the right of camera
    "baseline_mm": 80,
    "i2c_bus": 3,
    "i2c_address": "0x1b"
  },
  "scanning": {
    "method": "phase_shift",
    "pattern_type": "sinusoidal_pattern",
    "frequencies": [1, 8, 64],
    "steps_per_frequency": 4,
    "single_camera": true,
    "primary_camera": "left"
  }
}
```

### Checkerboard Specifications

#### Pattern Details
- **Internal corners**: 9x6 (columns x rows)
- **Total squares**: 10x7 (alternating black/white)
- **Square size**: 25-30mm per square
- **Total size on A4**: ~250x175mm (fits well)
- **Total size on A3**: ~300x210mm (better for larger FOV)

#### Important Notes
- **Must be printed** on matte paper (tablet screens cause reflections)
- **Mount on rigid surface** (cardboard or foam board) to keep flat
- **High contrast** black and white pattern required
- **Avoid glossy surfaces** that create reflections with projected patterns

### DLP342x Projector Specifications

#### Native Resolution
- **Resolution**: 1280x720 pixels (HD/720p)
- **Lumens**: 100
- **Control**: I2C interface

#### Pattern Generation Options
- **Native**: 1280x720 (recommended for calibration)
- **Upscaled**: 1920x1080 (automatically scaled by projector)
- **SDK Default**: 1024x768

### Calibration Process Overview

1. **Physical Setup**
   - Mount checkerboard on wall/stand at **40-50cm distance** (optimal range)
   - **Minimum distance**: 30cm (below this, focus issues may occur)
   - **Maximum distance**: 60cm (beyond this, pattern resolution decreases)
   - **Sweet spot**: 45cm for best calibration accuracy
   - Ensure good ambient lighting for camera calibration
   - Darken room slightly for projector calibration

2. **Camera Calibration**
   - Capture 20-30 images of checkerboard from different angles
   - No projected patterns during this phase
   - Validates camera intrinsics (focal length, distortion)

3. **Projector-Camera Calibration**
   - Interactive mode projects Gray code automatically
   - System captures checkerboard + projected patterns together
   - Creates projector↔camera correspondence map
   - Calculates relative position and orientation

4. **Validation**
   - Test with known objects
   - Verify reconstruction accuracy
   - Fine-tune if needed

### Pattern Generation Strategy for Single Camera + Projector

#### Phase Shift Method (Recommended)
- **Pattern Type**: Sinusoidal phase shift patterns
- **Frequencies**: Multi-frequency approach [1, 8, 64]
- **Steps**: 4 steps per frequency (total 12 patterns)
- **Triangulation**: Direct projector-camera ray intersection
- **Expected Points**: 50,000-200,000 per scan

#### Implementation Note
The system must use **projector-camera triangulation**, NOT stereo matching:
- Each camera pixel → phase value → projector column
- Triangulate between camera ray and projector ray
- Requires proper projector calibration as "inverse camera"

### Software Configuration

#### Server Setup
```bash
# Start server with dual camera + projector support
python unlook/server_bootstrap.py --config unlook_config_2k.json --enable-protocol-v2
```

#### Client Usage for Single Camera + Projector
```python
from unlook.client.scanner import Scanner3D

# Initialize scanner
scanner = Scanner3D()
scanner.connect()

# Load projector calibration
scanner.load_projector_calibration("projector_calibration.json")

# Capture phase shift sequence with projector-camera triangulation
result = scanner.scan_3d(
    pattern_type="phase_shift",
    frequencies=[1, 8, 64],
    steps_per_frequency=4,
    use_projector_camera_triangulation=True,  # Critical!
    single_camera_mode=True
)

print(f"3D Points generated: {len(result.point_cloud.points)}")
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