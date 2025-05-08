# Installation Guide for UnLook SDK

The UnLook SDK consists of two main components:
1. **Client** - For controlling the scanner, running on your development machine
2. **Server** - For operating the scanning hardware, running on a Raspberry Pi

## Prerequisites

### Client Prerequisites
- Python 3.7+ installed on your development machine
- pip (Python package manager)

### Server Prerequisites
- Raspberry Pi 4 or newer with Raspberry Pi OS
- Python 3.7+ installed on the Raspberry Pi
- DLP projector (DLPC342X-based) connected via I2C
- Raspberry Pi cameras connected and enabled

## Installation

### Client Installation

1. Clone the repository on your development machine:
```bash
git clone https://github.com/YourOrganization/unlook.git
cd unlook
```

2. Install client dependencies:
```bash
pip install -r client-requirements.txt
```

3. Install the SDK in development mode:
```bash
pip install -e .
```

#### Optional Dependencies for Mesh Generation

To enable 3D mesh generation from point clouds, install Open3D:
```bash
pip install open3d
```

### Server Installation

1. Clone the repository on your Raspberry Pi:
```bash
git clone https://github.com/YourOrganization/unlook.git
cd unlook
```

2. Install server dependencies:
```bash
pip install -r server-requirements.txt
```

3. Install the SDK in development mode:
```bash
pip install -e .
```

4. Enable I2C interface on the Raspberry Pi:
```bash
sudo raspi-config
```
Navigate to `Interfacing Options` → `I2C` → `Yes` to enable I2C.

5. Enable the Raspberry Pi cameras:
```bash
sudo raspi-config
```
Navigate to `Interfacing Options` → `Camera` → `Yes` to enable the camera interface.

6. Reboot your Raspberry Pi:
```bash
sudo reboot
```

## Running the Server

After installation, you can start the UnLook server on your Raspberry Pi:

```bash
python -m unlook.server_bootstrap
```

This will start the server with default settings. You can customize the server by creating a configuration file or by passing command-line arguments:

```bash
python -m unlook.server_bootstrap --name "MyScanner" --control-port 5555 --stream-port 5556
```

## Running the Client Examples

Several example scripts are included to demonstrate the UnLook SDK capabilities:

### Basic Connection Example

```bash
python -m unlook.examples.test_client
```

### Structured Light Pattern Sequence Example

```bash
python -m unlook.examples.pattern_sequence_example
```

### Real-Time 3D Scanning Example

```bash
python -m unlook.examples.real_time_scanning_example --quality medium --show-mesh
```

Options:
- `--continuous`: Run in continuous scanning mode
- `--quality`: Set scanning quality (low, medium, high, ultra)
- `--show-mesh`: Show the generated 3D mesh (requires Open3D)

## Troubleshooting

### Client Issues

1. **Cannot discover scanners**:
   - Ensure your client and server are on the same network
   - Check firewall settings to allow discovery on UDP port 5353 (mDNS)
   - Verify the server is running and advertising its service

2. **Cannot connect to scanner**:
   - Check network connectivity
   - Verify the server's control port is accessible
   - Check that the server is running properly

3. **Missing mesh generation**:
   - Verify Open3D is installed: `pip install open3d`
   - Check for error messages in the logs

### Server Issues

1. **I2C communication errors**:
   - Check connections between Raspberry Pi and the projector
   - Verify I2C is enabled in raspi-config
   - Run `i2cdetect -y 1` to check for connected I2C devices

2. **Camera errors**:
   - Ensure camera ribbon cables are properly connected
   - Verify the camera interface is enabled
   - Check camera permissions: `ls -la /dev/video*`

3. **Server won't start**:
   - Check for port conflicts
   - Verify all dependencies are installed
   - Check logs for specific errors

## System Requirements

### Minimum Requirements (Client)
- 8GB RAM
- Quad-core CPU
- 1GB of free disk space
- Any operating system that supports Python 3.7+

### Recommended Requirements (Client)
- 16GB RAM
- Modern multi-core CPU
- GPU (for mesh visualization and processing)
- 5GB of free disk space

### Server Requirements
- Raspberry Pi 4 with 4GB RAM (minimum)
- Raspberry Pi 4 with 8GB RAM (recommended)
- 32GB microSD card
- 5V/3A power supply

## Development Environment Setup

For contributing to the UnLook SDK, set up a development environment:

```bash
# Clone repository
git clone https://github.com/YourOrganization/unlook.git
cd unlook

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r dev-requirements.txt

# Install in development mode
pip install -e .
```