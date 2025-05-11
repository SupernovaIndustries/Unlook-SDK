# Unlook SDK Client Module

This directory contains the client-side implementation of the Unlook SDK, providing tools for controlling Unlook scanners, processing captured data, and performing 3D reconstruction.

## Main Components

- **camera.py** - Camera control and image capture
- **camera_calibration.py** - Stereo camera calibration utilities
- **camera_config.py** - Camera configuration and settings management
- **projector.py** - Projector control and pattern generation
- **projector_adapter.py** - Adapter for different projector implementations
- **realtime_scanner.py** - CPU-optimized real-time 3D scanning implementation
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

Robust and reliable real-time 3D scanning:

- CPU-optimized implementation for reliable operation
- Configurable quality presets for different performance needs
- Comprehensive error handling and diagnostics
- Robust projection-capture synchronization
- Open3D integration for visualization and processing
- Adaptable to different hardware configurations

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
from unlook.client.realtime_scanner import create_realtime_scanner, RealTimeScanConfig

# Create client and connect to scanner
client = UnlookClient(auto_discover=True)
client.start_discovery()
time.sleep(5)  # Wait for discovery
scanner_info = client.get_discovered_scanners()[0]
client.connect(scanner_info)

# Access camera functions
left_img, right_img = client.camera.capture_stereo_pair()

# Control the projector
client.projector.show_solid_field(color="White")

# Create a robust CPU-optimized scanner
config = RealTimeScanConfig()
config.set_quality_preset("medium")  # Options: fast, medium, high
config.pattern_interval = 0.3        # Time between patterns (seconds)
config.capture_delay = 0.1           # Delay before capture (seconds)
config.epipolar_tolerance = 15.0     # Tolerance for stereo matching (pixels)

# Create the scanner
scanner = create_realtime_scanner(
    client=client,
    config=config,
    on_new_frame=lambda point_cloud, scan_count, fps: print(f"New scan #{scan_count}, points: {len(point_cloud.points)}")
)

# Set synchronization mode
scanner.pattern_sync_mode = "strict"  # strict or normal

# Start scanning with debug output
scanner.start(debug_mode=True)

# Get the resulting point cloud
point_cloud = scanner.get_current_point_cloud()

# Stop scanner when done
scanner.stop()
client.disconnect()
```

## Extension Points

When extending the SDK, you can integrate with these components:

- **Custom Scanner Implementations**: Extend `scanner.py` for new scanning methods
- **New Structured Light Algorithms**: Add new methods to `structured_light.py`
- **Custom Visualization**: Extend `visualization.py` for specialized displays