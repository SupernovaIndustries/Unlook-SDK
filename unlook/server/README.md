# Unlook SDK Server Module

This directory contains the server-side implementation of the Unlook SDK, designed to run on hardware connected to the scanning devices (typically a Raspberry Pi).

## Main Components

- **scanner.py** - Main server implementation for scanner control
- **hardware/** - Hardware drivers and interfaces for scanner components:
  - **camera.py** - Camera hardware interface
  - **projector.py** - Projector hardware interface
  - **dlp342x/** - DLP projector drivers

## Key Features

### Hardware Abstraction

The server module abstracts hardware-specific details, providing:

- Unified camera interfaces for various camera types
- Projector control for DLP and other projector types
- Hardware synchronization for precise timing

### Communication Endpoints

Implements all server-side communication channels:

- Control channel for commands and configuration
- Stream channels for image data
- Direct channel for low-latency control
- Event notification system

### Hardware Control

Direct control over physical devices:

- Camera configuration and capture
- Projector pattern projection
- Hardware triggering and synchronization
- I2C communication with DLP controllers

## Server Architecture

The Unlook server follows a modular architecture:

1. **Main Server Process**: Handles communication, device management, and client requests
2. **Hardware Modules**: Specialized drivers for each hardware component
3. **Streaming Pipeline**: Optimized image acquisition and streaming
4. **Control Interface**: Command API for client interaction

## Deployment

The server is designed to run on Raspberry Pi or similar embedded Linux systems:

```bash
# Run the server
python -m unlook.server_bootstrap

# Run with custom configuration
python -m unlook.server_bootstrap --name "MyScanner" --control-port 5555
```

## Hardware Support

Currently supported hardware:

- **Cameras**:
  - Raspberry Pi Camera Modules (v1, v2, HQ)
  - MIPI CSI cameras
  - USB cameras (with V4L2 support)

- **Projectors**:
  - DLP342X-based projectors (with I2C control)
  - Standard HDMI projectors (with limited control)

## Extension Points

The server module provides several extension points:

- **Custom Hardware Drivers**: Add support for new cameras/projectors in the hardware directory
- **New Scanner Types**: Implement specialized scanner types by extending the base scanner
- **Custom Protocol Extensions**: Extend the command API for new capabilities