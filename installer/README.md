# Unlook SDK Installation Guide

This directory contains installer scripts and utilities for both the client and server components of the Unlook SDK.

## Installation Options

The Unlook SDK offers multiple installation options to accommodate different user needs:

1. **Standard Installation** (pip-based)
2. **Standalone Executable** (PyInstaller-based)
3. **Docker Container** (Containerized deployment)

## Client Installer

The client installer provides the Unlook SDK client components that can control the 3D scanning hardware through the server.

### Requirements

- Python 3.7+
- Windows, macOS, or Linux OS

### Standard Installation

The standard installation uses pip to install the SDK and its dependencies:

```bash
# Install the core SDK
pip install -r client-requirements.txt

# Install GPU acceleration (NVIDIA GPUs)
pip install cupy-cuda11x  # Replace with appropriate CUDA version

# Install in development mode (optional)
pip install -e .
```

### Standalone Client Executable

The standalone executable doesn't require Python installation and bundles everything needed:

1. Run the installer script:
   ```bash
   # Windows
   .\installer\build_client_installer.bat
   
   # Linux/macOS
   ./installer/build_client_installer.sh
   ```

2. Find the installer in the `installer/dist` directory:
   - Windows: `UnlookClient-Setup.exe`
   - macOS: `UnlookClient.dmg`
   - Linux: `UnlookClient.AppImage`

## Server Installer

The server installer prepares the Unlook hardware control server, typically run on a Raspberry Pi or similar device connected to the scanning hardware.

### Requirements

- Raspberry Pi 4 or higher (recommended)
- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- Connected Unlook hardware modules

### Standard Installation

```bash
# On Raspberry Pi or server device
pip install -r server-requirements.txt

# Install hardware drivers
sudo ./installer/install_server_drivers.sh
```

### Standalone Server Executable

1. Run the installer builder:
   ```bash
   # On build machine
   ./installer/build_server_installer.sh
   ```

2. Deploy to Raspberry Pi using the installer:
   ```bash
   # Copy to Raspberry Pi
   scp installer/dist/UnlookServer-Setup.sh pi@your-pi-address:/home/pi/
   
   # On Raspberry Pi
   chmod +x UnlookServer-Setup.sh
   ./UnlookServer-Setup.sh
   ```

### Docker Container

For deployment in container environments:

```bash
# Build Docker image
docker build -t unlook-server -f installer/Dockerfile.server .

# Run the container
docker run -d --privileged --name unlook-server \
    -p 5555:5555 -p 5556:5556 -p 5557:5557 \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    unlook-server
```

## Configuration

The installers include configuration utilities:

- `unlook-config-client`: GUI tool to configure client settings
- `unlook-config-server`: Server configuration utility

## Automatic Updates

The standalone installers include an auto-update mechanism:

- Client checks for updates on startup
- Server can be configured to auto-update based on schedule

## Installation Directory Structure

```
/opt/unlook/                   # Linux/macOS installation
└── bin/                       # Executable files
    ├── unlook-client          # Client executable
    ├── unlook-server          # Server executable
    └── unlook-config          # Configuration utility
└── lib/                       # Libraries
└── share/                     # Shared resources
    └── examples/              # Example scripts
    └── docs/                  # Documentation
    
C:\Program Files\Unlook\       # Windows installation
└── bin\                       # Executable files
    ├── unlook-client.exe      # Client executable
    ├── unlook-server.exe      # Server executable
    └── unlook-config.exe      # Configuration utility
└── lib\                       # Libraries
└── share\                     # Shared resources
    └── examples\              # Example scripts
    └── docs\                  # Documentation
```

## Troubleshooting

For installation issues, see the [Troubleshooting Guide](../docs/troubleshooting.md) or run:

```bash
# For client
unlook-client --doctor

# For server
unlook-server --doctor
```