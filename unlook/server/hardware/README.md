# Unlook SDK Hardware Drivers

This directory contains hardware-specific drivers and interfaces for the Unlook scanner hardware components.

## Components

- **camera.py** - Camera hardware abstraction layer
- **projector.py** - Projector hardware abstraction layer
- **dlp342x/** - Texas Instruments DLP342X projector drivers:
  - **dlpc342x_i2c.py** - I2C communication with DLP controller
  - **packer.py** - Data packing/unpacking for DLP commands

## Camera Support

The camera module provides a unified interface to various camera types:

- **Raspberry Pi Camera Modules**:
  - PiCamera v1 (OV5647)
  - PiCamera v2 (IMX219)
  - PiCamera HQ (IMX477)
  - Supports hardware synchronization via GPIO

- **USB Cameras**:
  - V4L2-compatible USB cameras
  - OpenCV-compatible cameras

## Projector Support

The projector module supports different projector types:

- **DLP Projectors**:
  - Texas Instruments DLP342X-based projectors
  - Full pattern control via I2C
  - Internal memory pattern storage
  - Hardware synchronization

- **HDMI Projectors**:
  - Standard HDMI-connected projectors
  - Pattern display via FrameBuffer or X11

## DLP342X Driver

The specialized DLP342X driver provides low-level control:

- I2C communication with the DLPC342X controller
- Pattern sequence configuration
- Internal memory management
- Synchronization control
- Detailed status reporting

## Usage

These hardware drivers are typically used by the server implementation:

```python
from unlook.server.hardware.camera import CameraManager
from unlook.server.hardware.projector import ProjectorManager

# Initialize hardware
camera_manager = CameraManager()
projector_manager = ProjectorManager()

# Configure hardware
camera_manager.initialize_cameras()
projector_manager.initialize_projector(projector_type="dlp342x", i2c_address=0x36)

# Use hardware
camera_manager.capture_image("camera1")
projector_manager.show_pattern("solid_white")
```

## Hardware Configuration

Hardware configuration is loaded from configuration files or command-line arguments:

- **Camera configuration**: Resolution, formats, synchronization options
- **Projector configuration**: Connection type, resolution, synchronization options

## Extension

To add support for new hardware:

1. For new cameras, extend the `CameraBase` class in `camera.py`
2. For new projectors, extend the `ProjectorBase` class in `projector.py`
3. Add any required driver files in appropriate subdirectories