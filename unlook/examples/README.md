# Unlook SDK Examples

This directory contains examples showcasing different features of the Unlook SDK.

## Basic Examples

- `camera_config_example.py` - Shows how to configure camera parameters
- `camera_calibration_example.py` - Shows how to calibrate the stereo camera system
- `test_client.py` - Simple test client for connecting to the scanner
- `uncertainty_visualization_example.py` - Demonstrates uncertainty visualization for ISO/ASTM 52902 compliance

## Scanning Examples

### Real-time 3D Scanning

- `real_time_scanning_example.py` - High-performance real-time 3D scanning with GPU acceleration, optimized for handheld scanning applications. Features faster frame rates, continuous scanning, and optional neural network enhancement.

```bash
# Basic usage:
python real_time_scanning_example.py

# Real-time visualization:
python real_time_scanning_example.py --visualize

# Record scan data:
python real_time_scanning_example.py --record

# Quality presets:
python real_time_scanning_example.py --quality fast
python real_time_scanning_example.py --quality high
python real_time_scanning_example.py --quality ultra

# Disable GPU acceleration:
python real_time_scanning_example.py --no-gpu

# Enable continuous scanning:
python real_time_scanning_example.py --continuous

# Debug logging:
python real_time_scanning_example.py --debug
```

See `REALTIME_SCANNING.md` in the root directory for more details on real-time scanning features.

### Pattern Sequence Example

- `pattern_sequence_example.py` - Demonstrates projector pattern sequence functionality

```bash
# Basic usage:
python pattern_sequence_example.py

# Show specific pattern types:
python pattern_sequence_example.py --pattern-type gray_code
python pattern_sequence_example.py --pattern-type phase_shift
python pattern_sequence_example.py --pattern-type combined

# Control timing:
python pattern_sequence_example.py --interval 0.5
```

### 3D Scanning Examples

- `simple_3d_scanning_example.py` - Basic 3D scanning implementation
- `enhanced_3d_scanning_example.py` - Enhanced version with additional features

```bash
# Basic usage:
python simple_3d_scanning_example.py

# With visualization:
python simple_3d_scanning_example.py --visualize

# Generate mesh:
python simple_3d_scanning_example.py --mesh

# Save to file:
python simple_3d_scanning_example.py --output ./my_scan.ply
```

### Camera Configuration Example

- `camera_configuration_demo.py` - Demonstrates camera configuration options

```bash
# Basic usage:
python camera_configuration_demo.py

# With different quality settings:
python camera_configuration_demo.py --quality high
python camera_configuration_demo.py --quality low

# With different formats:
python camera_configuration_demo.py --format jpeg
python camera_configuration_demo.py --format png
```

### Utility Scripts

- `view_scan.py` - Utility for viewing scan results
- `visualize_point_cloud.py` - Utility for visualizing point clouds

## Configuration

For hardware setup and configuration, please refer to the documentation in the `docs/` directory:

- `camera_configuration.md` - Instructions for camera setup
- `optimal_camera_spacing.md` - Guidelines for camera positioning
- `pattern_sequences.md` - Details on scanning pattern sequences

## ISO/ASTM 52902 Compliance

- `uncertainty_visualization_example.py` - Demonstrates uncertainty visualization and reporting tools for ISO/ASTM 52902 compliance

```bash
# Generate uncertainty visualizations and reports
python uncertainty_visualization_example.py
```

This example demonstrates the following ISO/ASTM 52902 compliance features:

1. Point cloud uncertainty measurement
2. Uncertainty visualization with color-coded heatmaps
3. Statistical uncertainty analysis
4. Comprehensive ISO/ASTM 52902 compliance reporting
5. Uncertainty histograms and visualization tools

## Running Examples

Most examples can be run directly with Python:

```bash
python camera_config_example.py
```

For more complex examples, command-line arguments are available:

```bash
python real_time_scanning_example.py --help
```

## Example Dependencies

These examples use the following dependencies:

- Open3D for point cloud visualization and processing
- OpenCV for image processing and display
- NumPy for numerical operations
- PyTorch for neural network enhancements (optional)