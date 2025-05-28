#!/bin/bash

# UnLook Auto-Startup Installation Script
# Run this script to enable automatic startup (user-agnostic)

echo "================================================"
echo "🔧 UnLook Auto-Startup Installation"
echo "================================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should NOT be run as root"
   echo "Please run as regular user: ./install_auto_startup.sh"
   exit 1
fi

# Check if we're in the right directory
if [[ ! -f "unlook_startup.sh" ]]; then
    echo "❌ Error: unlook_startup.sh not found in current directory"
    echo "Please run this script from the Unlook-SDK directory"
    exit 1
fi

# Get current user and home directory
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo ~$CURRENT_USER)

echo "👤 Installing for user: $CURRENT_USER"
echo "🏠 Home directory: $USER_HOME"

# Make startup script executable
echo "📋 Making startup script executable..."
chmod +x unlook_startup.sh

# Copy startup script to user home directory
echo "📁 Copying startup script to $USER_HOME/..."
cp unlook_startup.sh "$USER_HOME/"

# Create user-specific service name
SERVICE_NAME="unlook-auto-startup@$CURRENT_USER.service"

# Modify service file for current user
echo "⚙️  Creating user-specific systemd service..."
sed "s/%i/$CURRENT_USER/g; s|%h|$USER_HOME|g" unlook-auto-startup.service > "/tmp/$SERVICE_NAME"

# Install systemd service
echo "📦 Installing systemd service: $SERVICE_NAME"
sudo cp "/tmp/$SERVICE_NAME" "/etc/systemd/system/"
rm "/tmp/$SERVICE_NAME"

# Reload systemd and enable service
echo "🔄 Enabling auto-startup service for $CURRENT_USER..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "================================================"
echo "✅ Installation completed successfully!"
echo "================================================"
echo ""
echo "📋 Commands to manage the service:"
echo "   Start:    sudo systemctl start $SERVICE_NAME"
echo "   Stop:     sudo systemctl stop $SERVICE_NAME"
echo "   Status:   sudo systemctl status $SERVICE_NAME"
echo "   Logs:     sudo journalctl -u $SERVICE_NAME -f"
echo "   Disable:  sudo systemctl disable $SERVICE_NAME"
echo ""
echo "🔄 The service will start automatically on next boot"
echo "🚀 To start now: sudo systemctl start $SERVICE_NAME"
echo ""

# Ask if user wants to start the service now
read -p "Start the service now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting UnLook service..."
    sudo systemctl start "$SERVICE_NAME"
    sleep 2
    echo "📊 Service status:"
    sudo systemctl status "$SERVICE_NAME" --no-pager
fi

echo ""
echo "🎉 Installation complete! UnLook will start automatically on boot."
echo "👤 Service installed for user: $CURRENT_USER"