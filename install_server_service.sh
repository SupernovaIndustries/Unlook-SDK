#!/bin/bash
#
# UnLook Server Service Installer
# This script installs the UnLook server as a systemd service
#

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}Please run as root (use sudo)${NC}"
  exit 1
fi

# Configuration variables
SERVICE_NAME="unlook-server"
SERVICE_FILE="unlook-server.service"
INSTALL_DIR=$(dirname "$(readlink -f "$0")")
TARGET_SERVICE_PATH="/etc/systemd/system/${SERVICE_FILE}"
LOG_DIR="/var/log/unlook"

echo -e "${GREEN}UnLook Server Service Installer${NC}"
echo "==============================="
echo -e "Install directory: ${YELLOW}${INSTALL_DIR}${NC}"
echo ""
echo "This will install UnLook server with:"
echo "  ✓ Protocol V2 for optimized bandwidth"
echo "  ✓ GPU preprocessing on VideoCore VI"
echo "  ✓ Hardware sync for structured light"
echo "  ✓ Auto-start on boot"
echo ""

# Create log directory
echo "Creating log directory..."
mkdir -p $LOG_DIR
chown pi:pi $LOG_DIR
chmod 755 $LOG_DIR

# Check if virtual environment exists
if [ ! -d "${INSTALL_DIR}/.venv" ]; then
  echo -e "${RED}Virtual environment not found at ${INSTALL_DIR}/.venv${NC}"
  echo "Creating virtual environment and installing dependencies..."
  
  # Create venv if it doesn't exist
  cd "${INSTALL_DIR}"
  python3 -m venv .venv
  
  # Activate and install dependencies
  source .venv/bin/activate
  pip install -e .
  pip install -r server-requirements.txt
  
  # Add systemd module for Python
  pip install systemd-python
  
  # Make sure venv belongs to pi user
  chown -R pi:pi .venv
fi

# Check if config file exists, create example if not
CONFIG_FILE="${INSTALL_DIR}/unlook/unlook_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Creating example configuration file..."
  cat > "$CONFIG_FILE" << EOL
{
  "server": {
    "name": "UnLookScanner",
    "control_port": 5555,
    "stream_port": 5556,
    "direct_stream_port": 5557
  }
}
EOL
  chown pi:pi "$CONFIG_FILE"
  chmod 644 "$CONFIG_FILE"
  echo -e "Created example config at ${YELLOW}${CONFIG_FILE}${NC}"
  echo -e "${YELLOW}Please review and modify this file as needed.${NC}"
fi

# Copy service file to systemd location
echo "Installing systemd service..."
cp "${INSTALL_DIR}/${SERVICE_FILE}" "$TARGET_SERVICE_PATH"
chmod 644 "$TARGET_SERVICE_PATH"

# Reload systemd and enable service
echo "Enabling and starting service..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME

# Check service status
echo "Checking service status..."
sleep 2
if systemctl is-active --quiet $SERVICE_NAME; then
  echo -e "${GREEN}Service is active and running!${NC}"
else
  echo -e "${RED}Service failed to start. Please check logs:${NC}"
  echo "  - journalctl -u $SERVICE_NAME"
  echo "  - $LOG_DIR/server.log"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo "Service name: $SERVICE_NAME"
echo "Configuration: $CONFIG_FILE"
echo "Logs: $LOG_DIR/server.log"
echo ""
echo "Server features enabled:"
echo "  - Protocol V2 (bandwidth optimization)"
echo "  - Pattern preprocessing (GPU acceleration)"
echo "  - Hardware sync at 30 FPS"
echo "  - Delta encoding for video streams"
echo "  - Adaptive compression"
echo ""
echo "Commands:"
echo "  - Start:   sudo systemctl start $SERVICE_NAME"
echo "  - Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  - Restart: sudo systemctl restart $SERVICE_NAME"
echo "  - Status:  sudo systemctl status $SERVICE_NAME"
echo "  - Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo -e "${YELLOW}NOTE: You may need to adjust firewall settings to allow ports 5555, 5556, and 5557.${NC}"