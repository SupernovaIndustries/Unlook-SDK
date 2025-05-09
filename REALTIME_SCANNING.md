# Real-time 3D Scanning with UnLook SDK

This document explains how to use the real-time 3D scanning features in the UnLook SDK.

## Overview

The real-time scanning module provides optimized scanning capabilities for faster frame rates and continuous scanning. This makes it suitable for handheld 3D scanner applications where low latency and smooth operation are essential.

Key features:
- GPU acceleration (when available)
- Neural network point cloud enhancement
- Optimized pattern generation
- Minimal pattern sets for faster scanning
- Real-time visualization
- Data recording capabilities

## Installation

For real-time scanning, additional optional dependencies are recommended:

```bash
# Basic requirements
pip install -r requirements.txt

# For real-time 3D scanning (CPU only)
pip install open3d numpy

# For GPU acceleration
pip install torch torchvision # with CUDA support
pip install opencv-contrib-python-headless # with CUDA modules
```

## Usage

### Basic Usage

```python
from unlook import UnlookClient
from unlook.client.realtime_scanner import create_realtime_scanner

# Connect to scanner
client = UnlookClient(auto_discover=True)
# ... connect to scanner ...

# Create real-time scanner
scanner = create_realtime_scanner(
    client=client,
    quality="medium",  # Options: "fast", "medium", "high", "ultra"
    calibration_file="path/to/calibration.json"  # Optional
)

# Start scanning
scanner.start()

# Get results
point_cloud = scanner.get_current_point_cloud()
fps = scanner.get_fps()

# When done
scanner.stop()
client.disconnect()
```

### With Callback Function

```python
def on_new_frame(point_cloud, scan_count, fps):
    """Handle new scan data."""
    print(f"Scan #{scan_count} | FPS: {fps:.1f} | Points: {len(point_cloud.points)}")
    # Process point cloud...

# Create scanner with callback
scanner = create_realtime_scanner(
    client=client,
    quality="high",
    on_new_frame=on_new_frame
)
```

### With Visualization

The example script `unlook/examples/realtime_scanning_example.py` includes real-time visualization:

```bash
# Run with visualization
python unlook/examples/realtime_scanning_example.py --visualize

# Run with visualization and recording
python unlook/examples/realtime_scanning_example.py --visualize --record

# Different quality presets
python unlook/examples/realtime_scanning_example.py --quality fast
python unlook/examples/realtime_scanning_example.py --quality ultra
```

## Configuration

The `RealTimeScanConfig` class provides several options to customize the real-time scanning behavior:

```python
from unlook.client.realtime_scanner import RealTimeScanConfig

# Create config
config = RealTimeScanConfig()

# Customize settings
config.use_gpu = True  # Use GPU acceleration if available
config.use_neural_network = True  # Use neural network enhancement
config.max_fps = 15  # Target maximum FPS
config.moving_average_frames = 5  # Number of frames for temporal averaging
config.downsample_voxel_size = 2.0  # Downsampling voxel size (lower = more detail, slower)
config.num_gray_codes = 6  # Number of Gray code bits
config.pattern_interval = 0.1  # Time between pattern projections

# Apply quality preset
config.set_quality_preset("high")  # fast, medium, high, ultra

# Create scanner with config
scanner = create_realtime_scanner(client=client, config=config)
```

## Performance Considerations

For best performance:

1. **Hardware**: 
   - Use a dedicated GPU with CUDA support
   - Fast SSD for recording
   - Gigabit Ethernet connection to the scanner

2. **Software**:
   - Use lower quality presets for higher frame rates
   - Disable neural network enhancement if not needed
   - Adjust downsample_voxel_size for the right balance of detail and speed
   - Disable visualization if not needed during scanning

3. **Scanning Environment**:
   - Ensure good lighting conditions
   - Matte surfaces work better than glossy ones
   - Keep the scanner steady during capture
   - Maintain optimal scanning distance (usually 30-60cm)

## GPU Acceleration

GPU acceleration is automatically enabled when supported libraries are detected:

- CUDA support (through OpenCV's CUDA modules)
- PyTorch with CUDA
- Open3D with CUDA support

You can disable GPU acceleration with:
```python
scanner.config.use_gpu = False
```

Or when using the example script:
```bash
python unlook/examples/realtime_scanning_example.py --no-gpu
```

## Troubleshooting

1. **Low frame rate**: 
   - Try a faster quality preset
   - Disable neural network enhancement
   - Increase downsample_voxel_size
   - Check system resources (CPU, GPU, memory usage)

2. **Poor point cloud quality**:
   - Try higher quality presets
   - Decrease downsample_voxel_size
   - Enable neural network enhancement if available
   - Increase moving_average_frames for smoother results
   - Ensure proper calibration

3. **Connection issues**:
   - Verify scanner is powered on and connected to network
   - Check network connectivity
   - Restart scanner hardware if needed
   - Increase discovery timeout

4. **Crashes or errors**:
   - Enable debug logging for more information
   - Check console output for error messages
   - Verify all dependencies are correctly installed
   - Check GPU memory usage if using GPU acceleration