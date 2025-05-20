# 🔍 Unlook SDK

> **Complete SDK for Unlook - the modular open-source 3D scanning system**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/SupernovaIndustries/unlook)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/unlook/badge/?version=latest)](https://unlook.readthedocs.io/en/latest/?badge=latest)

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
  - Low-latency direct streams for real-time applications
  - Automated pattern sequences for structured light scanning
  - Precise projector-camera synchronization
- **3D Scanning**:
  - Simplified 3D scanning API - create a 3D scan in just a few lines of code
  - Real-time scanning mode - for handheld applications with GPU acceleration
  - Advanced pattern types - maze, Voronoi, and hybrid ArUco patterns
  - Camera auto-optimization - automatic exposure and gain adjustment
  - ISO/ASTM 52902 compliance - certification-ready with uncertainty quantification
  - Stereo camera calibration and rectification
  - Point cloud generation and filtering
  - Mesh creation and export to multiple formats
- **ISO/ASTM 52902 Certification Support**:
  - Uncertainty measurement - quantify accuracy for each pattern type
  - Calibration validation - using standardized test objects
  - Certification reporting - automated compliance documentation
  - Meets requirements for additive manufacturing geometric capability assessment
- **GPU Acceleration**: Optimized processing using GPU when available
- **Neural Network Enhancement**: Point cloud filtering and enhancement using machine learning
- **Advanced Pattern Systems**:
  - Maze patterns for robust correspondence matching with junction-based uncertainty
  - Voronoi patterns for dense surface reconstruction with cell boundary analysis
  - Hybrid ArUco patterns for absolute positioning with marker confidence metrics
  - All patterns include ISO-compliant uncertainty quantification
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

## 📚 Documentation

Complete documentation is available at [unlook.readthedocs.io](https://unlook.readthedocs.io/):

- Comprehensive [Installation Guide](https://unlook.readthedocs.io/en/latest/installation.html)
- Detailed [API Reference](https://unlook.readthedocs.io/en/latest/api_reference/index.html)
- Step-by-step [Tutorials](https://unlook.readthedocs.io/en/latest/examples/index.html)
- In-depth [User Guides](https://unlook.readthedocs.io/en/latest/user_guide/index.html)
- [Troubleshooting](https://unlook.readthedocs.io/en/latest/troubleshooting.html) section

## 🚀 Quick Start

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

### Simple Example

```python
from unlook import UnlookClient
import time

# Create client and auto-discover scanners
client = UnlookClient(auto_discover=True)
client.start_discovery()
time.sleep(5)  # Wait for discovery

# Connect to first available scanner
scanners = client.get_discovered_scanners()
if scanners:
    client.connect(scanners[0])
    
    # Capture an image
    image = client.camera.capture("camera_0")
    
    # Perform a 3D scan
    from unlook.client.scanning import StaticScanner
    scanner = StaticScanner(client=client)
    point_cloud = scanner.perform_scan()
    
    # Display the point cloud
    scanner.visualize_point_cloud(point_cloud)
```

## 🛠️ Advanced Configuration

### GPU Acceleration

For optimal performance, we recommend setting up GPU acceleration:

```bash
# Install CUDA Toolkit from NVIDIA's website first
# https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Advanced Pattern Support

For the advanced pattern types, install additional dependencies:

```bash
# For all pattern types
pip install scipy opencv-contrib-python
```

## 📊 Example Projects

Check the `unlook/examples/` directory for comprehensive examples:

- **Basic Examples**: Simple client connections, camera capture, and LED control
- **3D Scanning**: Real-time scanning, static scanning with different pattern types
- **Advanced Features**: Camera calibration, gesture recognition, pattern optimization
- **Integration Tests**: Verify functionality across different components

## 🏗️ Architecture

The SDK is structured into three main modules:

### Core Module (`unlook.core`)
- Communication protocol definitions
- Service discovery
- Event management system

### Client Module (`unlook.client`)
- Scanner client and connection management
- Camera and projector control
- 3D scanning algorithms
- Pattern generation and processing
- ISO compliance features

### Server Module (`unlook.server`)
- Hardware interfaces
- Device management
- Acquisition and streaming

## 🔧 Troubleshooting

If you encounter issues:

1. Ensure your virtual environment is activated
2. Verify hardware connections and permissions
3. Check for proper dependency installation
4. Enable debug logging for detailed information:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
5. Consult the [Troubleshooting Guide](https://unlook.readthedocs.io/en/latest/troubleshooting.html)

## 🤝 Contributing

Contributions are welcome! Please refer to our contribution guidelines in `CONTRIBUTING.md`.

## 📄 License

This project is released under the MIT License. See the `LICENSE` file for more details.

## 📞 Support

- Documentation: [unlook.readthedocs.io](https://unlook.readthedocs.io/)
- GitHub Issues: [github.com/SupernovaIndustries/unlook/issues](https://github.com/SupernovaIndustries/unlook/issues)
- Email: [info@supernovaindustries.it](mailto:info@supernovaindustries.it)

---

<p align="center">
  <img src="https://via.placeholder.com/150?text=UnlookLogo" alt="Supernova" width="150">
  <br>
  <em>Developed with ❤️ by Supernova Industries</em>
</p>