# Unlook SDK Client Module

This directory contains the client-side implementation of the Unlook SDK, providing tools for controlling Unlook scanners, processing captured data, and performing 3D reconstruction.

## Main Components

- **camera.py** - Camera control and image capture
- **camera_calibration.py** - Stereo camera calibration utilities
- **camera_config.py** - Camera configuration and settings management
- **projector.py** - Projector control and pattern generation
- **realtime_scanner.py** - Real-time 3D scanning implementation
- **scanner.py** - Base scanner functionality
- **scanner3d.py** - 3D scanner implementation
- **structured_light.py** - Structured light pattern generation and processing
- **visualization.py** - Point cloud and 3D data visualization

## Key Features

### Camera Management

The client module provides comprehensive camera control:

- Multi-camera support
- Dynamic camera discovery and configuration
- Advanced image capture with various quality presets
- Stereo camera calibration

### Projector Control

Full control over connected projectors:

- Pattern generation (Gray code, Phase shift, custom patterns)
- Pattern sequences with timing control
- Projector-camera synchronization

### Real-time Scanning

GPU-accelerated real-time 3D scanning:

- Fast scan modes optimized for handheld operation
- Point cloud generation and processing
- Open3D integration for visualization and processing
- Optional neural network enhancement

### 3D Reconstruction

Comprehensive 3D reconstruction capabilities:

- Point cloud generation from stereo correspondences
- Point cloud filtering and optimization
- Mesh generation (with Open3D)
- 3D data export to standard formats

## Usage Examples

The client module is typically used through the main `UnlookClient` class:

```python
from unlook import UnlookClient
from unlook.client.realtime_scanner import create_realtime_scanner

client = UnlookClient(auto_discover=True)
client.connect_to_first_available()

# Access camera functions
client.camera.capture("camera_1")

# Control the projector
client.projector.show_pattern("solid_field", color="White")

# Create a real-time scanner
scanner = create_realtime_scanner(client, quality="high")
scanner.start()
```

## Extension Points

When extending the SDK, you can integrate with these components:

- **Custom Scanner Implementations**: Extend `scanner.py` for new scanning methods
- **New Structured Light Algorithms**: Add new methods to `structured_light.py`
- **Custom Visualization**: Extend `visualization.py` for specialized displays