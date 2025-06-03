# MLX7502x ToF Sensor Python Module

Python driver for Melexis MLX7502x Time-of-Flight sensors (MLX75026/MLX75027).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the library

```python
from unlook.scanning_modules.ToF_module import MLX7502x, ToFConfig

# Configure sensor
config = ToFConfig(
    fps=12,
    phase_sequence=[0, 180, 90, 270],
    time_integration=[1000, 1000, 1000, 1000]
)

# Use sensor
with MLX7502x(config=config) as sensor:
    # Capture phase frames
    sensor.capture_phase_frames()
    
    # Compute depth
    magnitude, phase = sensor.compute_depth()
    
    # Continuous capture
    sensor.capture_continuous(display=True)
```

### Headless capture (no GUI)

```bash
python mlx7502x_headless.py
```

This will:
- Capture 10 sets of ToF frames
- Calculate magnitude and phase
- Save all data as .npy and .png files
- Create a timestamped output folder

## Features

- V4L2 interface for Linux
- 4-phase Time-of-Flight measurement
- I/Q demodulation for depth calculation
- Continuous and single capture modes
- Headless operation for embedded systems
- Context manager support
- 32/64-bit compatibility

## Configuration

- **Resolution**: 640×480
- **Pixel Format**: Y12P (12-bit packed)
- **Default FPS**: 12
- **Modulation Frequency**: 10 MHz
- **Phase Sequence**: [0°, 180°, 90°, 270°]
- **MIPI Lanes**: 2 or 4 (configurable)
- **MIPI Speed**: Up to 1.2 GHz per lane

### Using 2-Lane MIPI Mode

For systems with limited MIPI lanes, you can configure 2-lane mode:

```python
from unlook.scanning_modules.ToF_module import MLX7502x, ToFConfig

# Configure for 2-lane MIPI operation
config = ToFConfig(
    mipi_lanes=2,  # Use only 2 MIPI lanes
    fps=12
)

with MLX7502x(config=config) as sensor:
    # Sensor will automatically configure for 2-lane mode
    sensor.capture_continuous(display=True)
```

**Note**: 2-lane mode automatically adjusts MIPI speed to maintain bandwidth.

## Requirements

- Linux with V4L2 support
- Python 3.7+
- numpy
- opencv-python
- Access to `/dev/video*` devices

## Troubleshooting

### Permission denied
```bash
sudo usermod -a -G video $USER
# Logout and login again
```

### Driver initialization fails
Check kernel messages:
```bash
dmesg | grep mlx
```

## License

Part of Unlook SDK.