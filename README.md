# 🔍 Unlook SDK

> **Complete SDK for Unlook - the modular open-source 3D scanning system**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/YourOrganization/unlook)
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
  - Gray code pattern projection and processing
  - Stereo camera calibration and rectification
  - Point cloud generation and filtering
  - Mesh creation and export to multiple formats
- **Optimized Communication**: Efficient image transfer and control using ZeroMQ
- **Easy Expandability**: Modular architecture to add new hardware and algorithms

## 🧩 Scanning Module Support

Unlook is designed as a modular platform with interchangeable scanning modules, all controlled through this SDK:

| Scanning Module | Technology | Status |
|-----------------|------------|--------|
| Structured Light | Pattern projection & triangulation | ✅ Available |
| Depth Sensor | Time-of-flight or structured light | 🔜 Coming soon |
| Point Projector | Laser/IR dot pattern | 🔜 Coming soon |
| Custom Modules | User-created scanning solutions | 📝 Supported |

## 🔄 Low-Latency Direct Streaming

The new **direct streaming** feature is designed for applications requiring minimal latency and high responsiveness:

| Feature | Standard Streaming | 🚀 **Direct Streaming** |
|---------|-------------------|-------------------------|
| Latency | 50-100ms | **5-20ms** |
| Max Framerate | 30 FPS | **60-120 FPS** |
| Sync Precision | 20-50ms | **<10ms** |
| Auto Recovery | ✓ | ✓✓ (improved) |
| CPU Overhead | Medium | Low |
| ZeroMQ Socket | PUB/SUB | PAIR (optimized) |

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Raspberry Pi (for server) with Raspberry Pi OS
- Unlook scanner modules (structured light, depth sensor, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/YourOrganization/unlook.git
cd unlook

# Basic installation with required dependencies
pip install -r requirements.txt

# For 3D scanning with advanced point cloud processing (recommended)
pip install open3d

# Development installation
pip install -e .
```

### Using as a Client

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
    
    # Start low-latency direct streaming
    client.stream.start_direct_stream(
        "camera_id",
        callback=my_frame_callback,
        fps=60,
        low_latency=True,
        sync_with_projector=True
    )
    
    # Create and start a pattern sequence
    patterns = [
        {"pattern_type": "solid_field", "color": "White"},
        {"pattern_type": "horizontal_lines", "foreground_color": "White", 
         "background_color": "Black", "foreground_width": 4, "background_width": 20},
        {"pattern_type": "grid", "foreground_color": "White", "background_color": "Black"}
    ]
    client.projector.start_pattern_sequence(patterns, interval=0.5, loop=True)
```

### Simplified 3D Scanning (Recommended)

```python
from unlook import UnlookScanner

# Create scanner with auto-detection of hardware
scanner = UnlookScanner.auto_connect()

# Perform a 3D scan with one line (uses reasonable defaults)
point_cloud = scanner.perform_3d_scan()

# Create a mesh from the point cloud
mesh = scanner.create_mesh(point_cloud)

# Save the results
scanner.save_scan(point_cloud, "my_scan.ply")
scanner.save_mesh(mesh, "my_scan_mesh.obj")

# Visualize the results
scanner.visualize_point_cloud(point_cloud)
```

### Advanced 3D Scanning Example

```python
from unlook import UnlookClient
from unlook.client.structured_light import (
    StereoStructuredLightScanner, 
    StereoCalibrator,
    create_scanning_demo
)

# Set up scanner with default calibration for testing
output_dir = "./scan_results"
scanner = create_scanning_demo(output_dir)

# Connect to scanner
client = UnlookClient()
client.start_discovery()
scanners = client.get_discovered_scanners()
if scanners:
    client.connect(scanners[0])

    # Generate scanning patterns
    patterns = scanner.generate_scan_patterns()
    
    # Set up projector and cameras
    projector = client.projector
    camera = client.camera
    
    # Get the first two cameras
    cameras = camera.get_cameras()
    if len(cameras) >= 2:
        left_camera_id = cameras[0]["id"]
        right_camera_id = cameras[1]["id"]
        
        # Capture structured light images
        left_images = []
        right_images = []
        
        for pattern in patterns:
            # Project pattern
            projector.show_pattern(pattern)
            time.sleep(0.5)  # Wait for projector to update
            
            # Capture from both cameras
            left_img = camera.capture(left_camera_id)
            right_img = camera.capture(right_camera_id)
            
            left_images.append(left_img)
            right_images.append(right_img)
        
        # Process scan to get point cloud
        point_cloud = scanner.process_scan(left_images, right_images)
        
        # Save point cloud
        scanner.save_point_cloud(point_cloud, f"{output_dir}/scan.ply")
        
        # Create and save 3D mesh
        mesh = scanner.create_mesh_from_point_cloud(point_cloud)
        scanner.save_mesh(mesh, f"{output_dir}/scan_mesh.ply")
```

### Using as a Server

```python
from unlook import UnlookServer

# Start server
server = UnlookServer(
    name="My3DScanner",
    control_port=5555,
    stream_port=5556,
    direct_stream_port=5557,
    auto_start=True
)

# The server will automatically handle client connections and requests
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

### Structured Light Module

- Pattern generation and projection
- **Automated pattern sequences** with timing control
- Camera-projector synchronization
- Gray code and phase shift pattern generation
- Stereo camera calibration
- Enhanced 3D reconstruction with point cloud filtering
- Mesh generation from point clouds

### Depth Sensor Module

- Depth stream acquisition
- Point cloud generation
- Filtering and processing

### Point Projector Module

- Pattern control and calibration
- Feature detection and tracking
- Sparse reconstruction

## 🛠️ Customization and Extensions

The SDK is designed to be easily extended:

- **New hardware modules**: Add support for your own scanning devices
- **Reconstruction algorithms**: Implement custom 3D scanning algorithms
- **Custom protocols**: Extend the protocol for your specific needs

## 📊 Performance Benchmarking

Optimize system performance with built-in benchmarking tools:

```python
# Compare performance
results = client.benchmark.compare_streaming_performance(
    camera_id="camera_id",
    duration_seconds=30,
    modes=["standard", "direct", "direct_sync"]
)

# Export results
client.benchmark.export_results("benchmark_results.json")
```

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