#!/bin/bash
# UnLook Server Installer for Raspberry Pi
# This script installs the UnLook server components and configures autostart

set -e # Exit on any error

# Display banner
echo "=========================================================="
echo "   UnLook 3D Scanner Server Installer for Raspberry Pi"
echo "=========================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Define install paths
INSTALL_DIR="/opt/unlook"
CONFIG_DIR="/etc/unlook"
LOG_DIR="/var/log/unlook"
SERVICE_NAME="unlook-server"
SYSTEMD_DIR="/etc/systemd/system"
SOURCE_DIR="$(pwd)"

# Check if Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
  echo "WARNING: This installer is designed for Raspberry Pi."
  echo "Current device may not be fully compatible."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Function to update system and install dependencies
install_dependencies() {
  echo "Updating system and installing dependencies..."
  apt-get update
  apt-get upgrade -y
  
  # Install required packages
  apt-get install -y python3 python3-pip python3-dev \
                    libatlas-base-dev libjpeg-dev \
                    libhdf5-dev libopenjp2-7-dev \
                    i2c-tools \
                    libzbar0 \
                    git
  
  # Enable I2C if not already enabled
  if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt; then
    echo "Enabling I2C interface..."
    echo "dtparam=i2c_arm=on" >> /boot/config.txt
  fi

  # Enable camera if not already enabled
  if ! grep -q "^start_x=1" /boot/config.txt; then
    echo "Enabling camera interface..."
    echo "start_x=1" >> /boot/config.txt
    echo "gpu_mem=128" >> /boot/config.txt
  fi
  
  echo "Dependencies installed successfully."
}

# Function to create directories
create_directories() {
  echo "Creating installation directories..."
  mkdir -p "$INSTALL_DIR"
  mkdir -p "$CONFIG_DIR"
  mkdir -p "$LOG_DIR"
  
  # Create unlook user if it doesn't exist
  if ! id -u unlook > /dev/null 2>&1; then
    echo "Creating unlook user..."
    useradd -r -s /bin/false unlook
  fi
  
  # Set directory permissions
  chown -R unlook:unlook "$INSTALL_DIR"
  chown -R unlook:unlook "$CONFIG_DIR"
  chown -R unlook:unlook "$LOG_DIR"
  
  echo "Directories created successfully."
}

# Function to install server components
install_server() {
  echo "Installing UnLook server components..."
  
  # Copy server code to installation directory
  cp -r "$SOURCE_DIR/unlook/server" "$INSTALL_DIR/server"
  cp -r "$SOURCE_DIR/unlook/core" "$INSTALL_DIR/core"
  cp -r "$SOURCE_DIR/unlook/utils" "$INSTALL_DIR/utils"
  cp "$SOURCE_DIR/unlook/__init__.py" "$INSTALL_DIR/"
  cp "$SOURCE_DIR/unlook/server_bootstrap.py" "$INSTALL_DIR/"
  
  # Copy configuration files
  if [ -d "$SOURCE_DIR/unlook/calibration" ]; then
    cp -r "$SOURCE_DIR/unlook/calibration" "$CONFIG_DIR/"
  fi
  
  # Create log file
  touch "$LOG_DIR/unlook-server.log"
  chown unlook:unlook "$LOG_DIR/unlook-server.log"
  
  echo "Server components installed successfully."
}

# Function to install Python dependencies
install_python_dependencies() {
  echo "Installing Python dependencies..."
  
  # Create server-only requirements file
  cat > "$INSTALL_DIR/requirements.txt" << EOF
# UnLook Server Requirements
numpy>=1.19.0
opencv-python>=4.5.0
pyzmq>=21.0.0
zeroconf>=0.28.0
Pillow>=8.0.0
msgpack>=1.0.2
RPi.GPIO>=0.7.0
picamera2>=0.3.1
smbus2>=0.4.1
EOF

  # Install requirements
  pip3 install -r "$INSTALL_DIR/requirements.txt"
  
  # Create Python path file
  cat > "/etc/profile.d/unlook.sh" << EOF
#!/bin/bash
export PYTHONPATH="\$PYTHONPATH:$INSTALL_DIR"
EOF
  chmod +x "/etc/profile.d/unlook.sh"
  
  echo "Python dependencies installed successfully."
}

# Function to create systemd service
create_systemd_service() {
  echo "Creating systemd service..."
  
  # Create service file
  cat > "$SYSTEMD_DIR/$SERVICE_NAME.service" << EOF
[Unit]
Description=UnLook 3D Scanner Server
After=network.target

[Service]
Type=simple
User=unlook
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/server_bootstrap.py --server-only
Restart=always
RestartSec=5
StandardOutput=append:$LOG_DIR/unlook-server.log
StandardError=append:$LOG_DIR/unlook-server.log

[Install]
WantedBy=multi-user.target
EOF

  # Reload systemd, enable and start service
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME"
  systemctl start "$SERVICE_NAME"
  
  echo "Systemd service created and started successfully."
}

# Function to create configuration file
create_config() {
  echo "Creating default configuration..."
  
  # Create default config file
  cat > "$CONFIG_DIR/server_config.json" << EOF
{
  "server": {
    "name": "UnLook Scanner",
    "port": 5555,
    "discovery_port": 5353
  },
  "cameras": {
    "auto_detect": true,
    "default_resolution": [1280, 720],
    "default_fps": 30
  },
  "projector": {
    "default_mode": "solid",
    "default_color": "white"
  },
  "led": {
    "max_intensity": 450
  },
  "logging": {
    "level": "INFO",
    "file": "$LOG_DIR/unlook-server.log"
  }
}
EOF

  # Set permissions
  chown unlook:unlook "$CONFIG_DIR/server_config.json"
  
  echo "Configuration created successfully."
}

# Main installation sequence
main() {
  echo "Starting installation..."
  
  install_dependencies
  create_directories
  install_server
  install_python_dependencies
  create_config
  create_systemd_service
  
  echo ""
  echo "=========================================================="
  echo "     UnLook Server installed successfully!"
  echo "     Service name: $SERVICE_NAME"
  echo "     Installation directory: $INSTALL_DIR"
  echo "     Configuration directory: $CONFIG_DIR"
  echo "     Log directory: $LOG_DIR"
  echo ""
  echo "     To check status: sudo systemctl status $SERVICE_NAME"
  echo "     To view logs: sudo journalctl -u $SERVICE_NAME -f"
  echo "=========================================================="
}

# Execute main installation
main