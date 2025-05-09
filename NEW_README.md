# 🔍 Unlook SDK

> **Complete SDK for Unlook - the modular open-source 3D scanning system**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/SupernovaIndustries/unlook)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Unlook SDK** is a powerful and flexible Python framework designed to control Unlook - a modular open-source 3D scanning system that supports interchangeable optics and sensors, similar to how cameras support different lenses. This SDK provides a complete solution for managing various scanning modules, from structured light systems to depth sensors, point projectors, and more.

## 🌟 Key Features

- **Modular Hardware Support**: Control multiple scanning technologies through a unified interface
- **Client-Server Architecture**: Clean separation between control clients and acquisition servers
- **Auto-Discovery**: Automatic detection of available Unlook scanners on local network
- **Comprehensive Control**:
  - DLPC342X projector interface for structured light modules
  - Multiple camera support for stereo and multi-view setups
  - Depth sensor integration
  - Point projector management
- **Advanced Video Streaming**:
  - Standard streams for visualization and monitoring
  - **✨ NEW! Low-latency direct streams** for real-time applications
  - **✨ NEW! Automated pattern sequences** for structured light scanning
  - Precise projector-camera synchronization
- **3D Scanning**:
  - **✨ NEW! Simplified 3D scanning API** - create a 3D scan in just a few lines of code
  - **✨ NEW! Real-time scanning mode** - for handheld applications with GPU acceleration
  - Stereo camera calibration and rectification
  - Point cloud generation and filtering
  - Mesh creation and export to multiple formats
- **GPU Acceleration**: Optimized processing using GPU when available
- **Neural Network Enhancement**: Point cloud filtering and enhancement using machine learning
- **Optimized Communication**: Efficient image transfer and control using ZeroMQ
- **Easy Expandability**: Modular architecture to add new hardware and algorithms

## 🧩 Scanning Module Support

Unlook is designed as a modular platform with interchangeable scanning modules, all controlled through this SDK:

| Scanning Module | Technology | Status |
|-----------------|------------|--------|
| Real-Time Scanning | GPU-accelerated scanning for handheld operation | ✅ Available |
| Depth Sensor | Time-of-flight or structured light | 🔜 Coming soon |
| Point Projector | Laser/IR dot pattern | 🔜 Coming soon |
| Custom Modules | User-created scanning solutions | 📝 Supported |

## 🚀 Getting Started

### Prerequisites

- Python 3.7+ (Python 3.9 or 3.10 recommended)
- Raspberry Pi (for server) with Raspberry Pi OS
- Unlook scanner modules (structured light, depth sensor, etc.)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/SupernovaIndustries/unlook.git
cd unlook

# Basic installation with required dependencies
pip install -r client-requirements.txt
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### GPU Acceleration Setup

For optimal performance, we recommend setting up GPU acceleration:

#### NVIDIA GPUs

```bash
# Install CUDA Toolkit from NVIDIA's website first
# https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Install CuPy (match with your CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
```

#### AMD GPUs

```bash
# Install ROCm first, following AMD's instructions
# https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

#### CPU-only Fallback

```bash
# CPU-only versions
pip install torch torchvision
pip install cupy
```

### Testing Your Installation

Verify your installation with the test script:

```bash
# Check GPU acceleration
python -m unlook.utils.check_gpu

# Run basic test
python -m unlook.examples.test_client
```

## 📊 Example Usage

### Basic Client Connection

```python
from unlook import UnlookClient, EventType

# Create client
client = UnlookClient(client_name="ExampleApp")

# Register callback for events
def on_connected(scanner):
    print(f"Connected to: {scanner.name} ({scanner.uuid})")

client.on(EventType.CONNECTED, on_connected)

# Discover available scanners
client.start_discovery()

# Connect to the first available scanner
scanners = client.get_discovered_scanners()
if scanners:
    client.connect(scanners[0])
    
    # Capture image using structured light module
    image = client.camera.capture("camera_id")
```

### Real-Time Scanning

```python
from unlook import UnlookClient
from unlook.client.realtime_scanner import create_realtime_scanner
import time

# Create client and connect to scanner
client = UnlookClient(auto_discover=True)
client.start_discovery()
time.sleep(5)  # Wait for discovery

scanners = client.get_discovered_scanners()
if scanners:
    client.connect(scanners[0])
    
    # Create real-time scanner with desired quality
    scanner = create_realtime_scanner(
        client=client,
        quality="medium",  # Options: "fast", "medium", "high", "ultra"
        calibration_file="calibration/stereo_calib.json"  # Optional
    )
    
    # Start continuous scanning
    scanner.start()
    
    # Show visualization (press ESC to exit)
    from unlook.client.visualization import ScanVisualizer
    visualizer = ScanVisualizer()
    
    try:
        while True:
            # Get the latest point cloud
            point_cloud = scanner.get_current_point_cloud()
            
            # Update visualization if available
            if point_cloud is not None:
                visualizer.update(
                    point_cloud, 
                    scanner.get_fps(), 
                    scanner.get_scan_count()
                )
                
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()
        visualizer.close()
```

### Camera Configuration

```python
from unlook.client.camera_config import CameraConfig, ColorMode, CompressionFormat

# Create a new camera configuration
config = CameraConfig()

# Configure basic settings
config.exposure_time = 20000  # in microseconds (20ms)
config.gain = 1.5
config.jpeg_quality = 90
config.color_mode = ColorMode.COLOR

# Apply configuration to a camera
client.camera.apply_camera_config("camera_1", config)
```

### Pattern Sequences for Structured Light Scanning

```python
# Define a sequence of patterns
patterns = [
    {"pattern_type": "solid_field", "color": "White"},
    {"pattern_type": "horizontal_lines", "foreground_color": "White", 
     "background_color": "Black", "foreground_width": 4, "background_width": 20},
    {"pattern_type": "vertical_lines", "foreground_color": "White", 
     "background_color": "Black", "foreground_width": 4, "background_width": 20},
    {"pattern_type": "grid", "foreground_color": "White", "background_color": "Black"}
]

# Start the sequence with 1s interval, looping
result = client.projector.start_pattern_sequence(
    patterns=patterns,
    interval=1.0,      # 1 second between patterns
    loop=True,         # Loop continuously
    sync_with_camera=True  # Enable projector-camera synchronization
)
```

## 🧩 Architecture

The SDK is structured into several main modules:

```
unlook/
├── core/              # Core components (events, protocol, discovery)
├── client/            # Client implementation
├── server/            # Server implementation
│   └── hardware/      # Drivers for specific hardware modules
└── examples/          # Example scripts
```

## 📝 Communication Protocol

Communication between client and server happens through structured messages over ZeroMQ:

- **Control channel**: REQ/REP pattern for control commands and configuration
- **Stream channel**: PUB/SUB pattern for standard video streaming
- **Direct channel**: PAIR pattern for low-latency streaming with synchronization

## 🔄 Module-Specific Features

### Real-Time Scanning Module

- Continuous scanning mode for handheld applications
- GPU acceleration using CUDA when available
- Neural network enhancement for point cloud filtering
- Real-time visualization with Open3D and OpenCV
- Optimized for low-latency operation

### Depth Sensor Module (Coming Soon)

- Depth stream acquisition
- Point cloud generation
- Filtering and processing

### Point Projector Module (Coming Soon)

- Pattern control and calibration
- Feature detection and tracking
- Sparse reconstruction

## 🛠️ Customization and Extensions

The SDK is designed to be easily extended:

- **New hardware modules**: Add support for your own scanning devices
- **Reconstruction algorithms**: Implement custom 3D scanning algorithms
- **Custom protocols**: Extend the protocol for your specific needs

## 📚 Documentation

For more detailed documentation on specific components, see:

- [Installation Guide](INSTALLATION.md) - Comprehensive installation instructions
- [Camera Configuration](docs/camera_configuration.md) - Camera settings and optimization
- [Optimal Camera Spacing](docs/optimal_camera_spacing.md) - Guidelines for camera positioning
- [Pattern Sequences](docs/pattern_sequences.md) - Custom projector pattern sequences
- [Real-time Scanning](REALTIME_SCANNING.md) - Guide to real-time scanning features
- [Examples](unlook/examples/) - Example code and usage demonstrations
- [Project Roadmap](ROADMAP.md) - Future development plans

## 📄 License

This project is released under the MIT License. See the `LICENSE` file for more details.

## 🤝 Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` for guidelines.

## 📞 Contact

For support or questions, contact us at [supernovaindustries](mailto:info@supernovaindustries.it).

---

<p align="center">
  <img src="https://via.placeholder.com/150?text=UnlookLogo" alt="Supernova" width="150">
  <br>
  <em>Developed with ❤️ by Supernova Industries</em>
</p>