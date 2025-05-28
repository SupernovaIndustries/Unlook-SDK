# UnLook Server Installation Guide

This guide explains how to install and configure the UnLook server as a systemd service on a Raspberry Pi.

## 🚀 Quick Auto-Startup Setup (Recommended for V2)

For production setups with the latest optimizations, use the new auto-startup scripts:

### 1. Install Auto-Startup Service (V2 Optimized - User Agnostic)
```bash
cd Unlook-SDK/
chmod +x install_auto_startup.sh
./install_auto_startup.sh

# Works for any user (pi, unlook, admin, etc.)
# Automatically detects current user and home directory
# Creates user-specific systemd service
```

### 2. Manual Startup with Optimizations
```bash
cd Unlook-SDK/
chmod +x unlook_startup.sh
./unlook_startup.sh
```

**V2 Features Enabled**:
- ✅ GPU-accelerated preprocessing (VideoCore VI)
- ✅ Hardware synchronization <500μs (GPIO 27)
- ✅ Protocol optimization with delta encoding
- ✅ Real-time quality metrics
- ✅ Automatic git updates on startup

### 3. Service Management
```bash
# Service is user-specific, e.g., for user 'pi':
SERVICE_NAME="unlook-auto-startup@pi.service"

# Check status
sudo systemctl status unlook-auto-startup@$(whoami)

# View logs
sudo journalctl -u unlook-auto-startup@$(whoami) -f

# Stop service
sudo systemctl stop unlook-auto-startup@$(whoami)

# Disable auto-startup
sudo systemctl disable unlook-auto-startup@$(whoami)
```

### 4. Advanced Configuration
```bash
# Custom startup with different settings
python unlook/server_bootstrap.py \
    --enable-pattern-preprocessing \
    --preprocessing-level full \
    --enable-sync \
    --sync-fps 60 \
    --log-level DEBUG
```

---

## 📜 Legacy Installation (V1)

## Prerequisites

- Raspberry Pi with Raspberry Pi OS (Bullseye or later recommended)
- Python 3.7 or higher
- Git
- UnLook Scanner hardware connected to the Raspberry Pi

## Installation Steps

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/your-username/Unlook-SDK.git
   cd Unlook-SDK
   ```

2. Run the installation script as root:
   ```bash
   sudo ./install_server_service.sh
   ```

   This script will:
   - Create a virtual environment and install dependencies
   - Create a log directory at `/var/log/unlook/`
   - Create an example configuration file if one doesn't exist
   - Install and enable the systemd service
   - Start the server

## Configuration

The server configuration is stored in `unlook/unlook_config.json`. If this file doesn't exist, the installer will create an example configuration.

You can customize the configuration file with the following settings:

```json
{
  "server": {
    "name": "UnLookScanner",
    "control_port": 5555,
    "stream_port": 5556,
    "direct_stream_port": 5557,
    "scanner_uuid": null
  },
  "hardware": {
    "projector": {
      "type": "DLP342X",
      "i2c_bus": 3,
      "i2c_address": "0x1B",
      "default_mode": "pattern"
    },
    "cameras": {
      "default_resolution": [1920, 1080],
      "default_fps": 30,
      "auto_exposure": true
    },
    "led_controller": {
      "enabled": true,
      "default_intensity": 50
    }
  },
  "streaming": {
    "default_fps": 30,
    "jpeg_quality": 85,
    "direct_streaming_fps": 60,
    "direct_jpeg_quality": 90,
    "enable_camera_sync": true
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/unlook/server.log",
    "console": true,
    "rotate_size_mb": 10,
    "keep_logs": 5
  }
}
```

## Service Management

After installation, you can manage the service with the following commands:

- **Start the server**:
  ```bash
  sudo systemctl start unlook-server
  ```

- **Stop the server**:
  ```bash
  sudo systemctl stop unlook-server
  ```

- **Restart the server**:
  ```bash
  sudo systemctl restart unlook-server
  ```

- **Check server status**:
  ```bash
  sudo systemctl status unlook-server
  ```

- **View server logs**:
  ```bash
  sudo journalctl -u unlook-server -f
  ```
  or
  ```bash
  tail -f /var/log/unlook/server.log
  ```

- **Enable auto-start on boot**:
  ```bash
  sudo systemctl enable unlook-server
  ```

- **Disable auto-start on boot**:
  ```bash
  sudo systemctl disable unlook-server
  ```

## Firewall Configuration

If you have a firewall enabled on your Raspberry Pi, you'll need to allow the following ports:

- 5555 (Control port)
- 5556 (Stream port)
- 5557 (Direct stream port)

Using UFW (Uncomplicated Firewall):
```bash
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp
sudo ufw allow 5557/tcp
```

## Troubleshooting

### Server Won't Start

Check the systemd journal for errors:
```bash
sudo journalctl -u unlook-server -n 50
```

Common issues include:
- Python dependencies not installed
- Hardware not connected properly
- Permission issues with camera or I2C devices

### Permission Issues

Make sure the `pi` user has access to the required hardware:

```bash
# For camera access
sudo usermod -a -G video pi

# For I2C access
sudo usermod -a -G i2c pi
```

### Hardware Detection

If the server cannot detect the hardware, check the connections and ensure the required modules are loaded:

```bash
# For I2C
sudo raspi-config
# Navigate to Interface Options > I2C > Enable

# List I2C devices
i2cdetect -y 1
```

## Uninstalling

To uninstall the service:

```bash
sudo systemctl stop unlook-server
sudo systemctl disable unlook-server
sudo rm /etc/systemd/system/unlook-server.service
sudo systemctl daemon-reload
```