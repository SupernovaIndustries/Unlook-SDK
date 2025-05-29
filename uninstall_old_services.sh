#!/bin/bash
#
# UnLook Services Uninstaller
# This script removes old UnLook services and prepares for new installation
#

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}UnLook Services Uninstaller${NC}"
echo "================================="
echo ""
echo "This script will:"
echo "  - Stop and disable old UnLook services"
echo "  - Remove old service files"
echo "  - Clean up old configurations"
echo "  - Prepare for new Protocol V2 installation"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Check for existing services
echo ""
echo "Checking for existing services..."

# List of possible service names (old and new)
SERVICES=(
    "unlook-server"
    "unlook-auto-startup@$USER"
    "unlook-scanner"
    "unlook"
)

# Function to stop and disable a service
remove_service() {
    local service=$1
    
    # Check if service exists
    if systemctl list-unit-files | grep -q "^$service"; then
        echo -e "${YELLOW}Found service: $service${NC}"
        
        # Stop the service
        echo "  Stopping..."
        sudo systemctl stop "$service" 2>/dev/null
        
        # Disable the service
        echo "  Disabling..."
        sudo systemctl disable "$service" 2>/dev/null
        
        # Remove service file
        SERVICE_FILE="/etc/systemd/system/$service.service"
        if [ -f "$SERVICE_FILE" ]; then
            echo "  Removing service file..."
            sudo rm -f "$SERVICE_FILE"
        fi
        
        # Also check in user services
        USER_SERVICE_FILE="/home/$USER/.config/systemd/user/$service.service"
        if [ -f "$USER_SERVICE_FILE" ]; then
            echo "  Removing user service file..."
            rm -f "$USER_SERVICE_FILE"
        fi
        
        echo -e "  ${GREEN}✓ Removed${NC}"
    fi
}

# Remove each service
for service in "${SERVICES[@]}"; do
    remove_service "$service"
done

# Clean up old startup scripts
echo ""
echo "Cleaning up old startup scripts..."

OLD_SCRIPTS=(
    "/home/$USER/unlook_startup.sh"
    "/home/$USER/start_unlook.sh"
    "/home/$USER/.unlook/startup.sh"
)

for script in "${OLD_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "  Removing: ${YELLOW}$script${NC}"
        rm -f "$script"
    fi
done

# Clean up old log directories
echo ""
echo "Cleaning up log directories..."

if [ -d "/var/log/unlook" ]; then
    echo "  Found log directory: /var/log/unlook"
    read -p "  Remove log files? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo rm -rf /var/log/unlook
        echo -e "  ${GREEN}✓ Removed${NC}"
    else
        echo "  Keeping log files"
    fi
fi

# Reload systemd
echo ""
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Check for any remaining services
echo ""
echo "Checking for remaining UnLook services..."
REMAINING=$(systemctl list-unit-files | grep -i unlook | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo -e "${GREEN}✓ All UnLook services have been removed${NC}"
else
    echo -e "${YELLOW}⚠ Found $REMAINING remaining UnLook-related services:${NC}"
    systemctl list-unit-files | grep -i unlook
    echo ""
    echo "You may need to manually remove these if they're not needed."
fi

echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
echo ""
echo "You can now install the new services with:"
echo -e "  ${YELLOW}./install_auto_startup.sh${NC}     (for development)"
echo -e "  ${YELLOW}sudo ./install_server_service.sh${NC} (for production)"
echo ""
echo "The new services will include:"
echo "  ✓ Protocol V2 support"
echo "  ✓ GPU preprocessing"
echo "  ✓ Enhanced bandwidth optimization"
echo "  ✓ Hardware synchronization"