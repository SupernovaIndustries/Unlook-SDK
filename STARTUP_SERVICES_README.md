# UnLook Scanner - Startup Services Configuration

## Overview
This document explains how to configure the UnLook scanner to start automatically on Raspberry Pi boot with optimized settings for the enhanced Protocol V2 pipeline.

## Service Features

The startup services are configured to enable:
- **Protocol V2**: Optimized bandwidth usage with delta encoding and adaptive compression
- **GPU Preprocessing**: Hardware acceleration using VideoCore VI
- **Pattern Preprocessing**: Advanced preprocessing for structured light patterns
- **Hardware Sync**: 30 FPS synchronized capture for stereo cameras
- **Auto-restart**: Automatic recovery on failure

## Installation Methods

### Method 1: User-specific Auto-startup (Recommended for Development)
```bash
./install_auto_startup.sh
```

This installs a service that:
- Runs as the current user
- Automatically pulls latest code from git
- Activates the Python virtual environment
- Starts the server with all optimizations

### Method 2: System Service (Recommended for Production)
```bash
sudo ./install_server_service.sh
```

This installs a system service that:
- Runs as the 'pi' user
- Starts immediately on boot
- Uses systemd for management
- Logs to `/var/log/unlook/server.log`

## Service Parameters

Both services start the server with these parameters:
```bash
python unlook/server_bootstrap.py \
    --enable-protocol-v2 \             # Enable Protocol V2 with compression
    --enable-pattern-preprocessing \    # Enable GPU preprocessing
    --preprocessing-level advanced \    # Use advanced preprocessing
    --enable-sync \                    # Enable hardware synchronization
    --sync-fps 30 \                   # 30 FPS capture rate
    --log-level INFO                   # Info level logging
```

## Managing the Services

### User Auto-startup Service
```bash
# Check status
sudo systemctl status unlook-auto-startup@$(whoami).service

# Start/stop/restart
sudo systemctl start unlook-auto-startup@$(whoami).service
sudo systemctl stop unlook-auto-startup@$(whoami).service
sudo systemctl restart unlook-auto-startup@$(whoami).service

# View logs
sudo journalctl -u unlook-auto-startup@$(whoami).service -f

# Disable auto-start
sudo systemctl disable unlook-auto-startup@$(whoami).service
```

### System Service
```bash
# Check status
sudo systemctl status unlook-server

# Start/stop/restart
sudo systemctl start unlook-server
sudo systemctl stop unlook-server
sudo systemctl restart unlook-server

# View logs
sudo journalctl -u unlook-server -f
# or
tail -f /var/log/unlook/server.log

# Disable auto-start
sudo systemctl disable unlook-server
```

## Protocol V2 Benefits

The startup services enable Protocol V2 which provides:

1. **Bandwidth Optimization**
   - Delta encoding for video streams
   - Adaptive compression based on content
   - Multi-camera message optimization

2. **GPU Acceleration**
   - Lens correction on VideoCore VI
   - ROI detection and cropping
   - Pattern-specific preprocessing
   - Real-time quality assessment

3. **Enhanced Performance**
   - Reduced network latency
   - Higher frame rates possible
   - Better 3D reconstruction quality
   - Lower CPU usage on client

## Troubleshooting

### Service won't start
1. Check virtual environment exists:
   ```bash
   ls -la /home/pi/Unlook-SDK/.venv
   ```

2. Check Python dependencies:
   ```bash
   source /home/pi/Unlook-SDK/.venv/bin/activate
   pip list | grep zmq
   ```

3. Check service logs:
   ```bash
   sudo journalctl -u unlook-server -n 50
   ```

### Network connectivity issues
1. Check firewall allows ports 5555-5557:
   ```bash
   sudo ufw status
   ```

2. Check ZeroMQ discovery is working:
   ```bash
   python -m unlook.core.discovery
   ```

### Performance issues
1. Verify GPU is being used:
   ```bash
   vcgencmd get_mem gpu
   ```

2. Check preprocessing is enabled in logs:
   ```
   grep "GPU preprocessing" /var/log/unlook/server.log
   ```

## Configuration

The server reads configuration from `/home/pi/Unlook-SDK/unlook/unlook_config.json`:

```json
{
  "server": {
    "name": "UnLookScanner",
    "control_port": 5555,
    "stream_port": 5556,
    "direct_stream_port": 5557
  },
  "preprocessing": {
    "gpu_enabled": true,
    "level": "advanced"
  }
}
```

## Client Connection

Clients should connect using Protocol V2 for best performance:

```python
from unlook.client.scanner.scanner import UnlookClient
from unlook.core.constants import PreprocessingVersion

# For 3D scanning (Protocol V2)
client = UnlookClient(
    preprocessing_version=PreprocessingVersion.V2_ENHANCED
)

# For handpose (Protocol V1)
client = UnlookClient(
    preprocessing_version=PreprocessingVersion.V1_LEGACY
)
```

## Performance Metrics

With Protocol V2 enabled, expect:
- **Bandwidth reduction**: 40-60% for structured light patterns
- **Capture rate**: 30 FPS synchronized stereo
- **Preprocessing latency**: <10ms per frame
- **Network latency**: <5ms on local network

## Updates

To update the service after code changes:
```bash
# For user service
sudo systemctl restart unlook-auto-startup@$(whoami).service

# For system service
sudo systemctl restart unlook-server
```

The user auto-startup service automatically pulls latest changes from git on each start.