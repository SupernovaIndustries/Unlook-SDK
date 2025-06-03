#!/bin/bash
# UnLook Server 2K Startup Script

echo "🔥 Starting UnLook Server in 2K Mode"
echo "======================================"

# Export 2K configuration
export UNLOOK_CAMERA_WIDTH=2048
export UNLOOK_CAMERA_HEIGHT=1536
export UNLOOK_CAMERA_FPS=15
export UNLOOK_CAMERA_QUALITY=95
export UNLOOK_PATTERN_WIDTH=2048
export UNLOOK_PATTERN_HEIGHT=1536
export UNLOOK_QUALITY_PRESET=ultra
export UNLOOK_REALTIME_MODE=false
export UNLOOK_PRIORITY_QUALITY=true

echo "Configuration:"
echo "  Resolution: ${UNLOOK_CAMERA_WIDTH}x${UNLOOK_CAMERA_HEIGHT}"
echo "  FPS: ${UNLOOK_CAMERA_FPS}"
echo "  Quality: ${UNLOOK_CAMERA_QUALITY}%"
echo "  Mode: High-Quality (Non-Realtime)"
echo "======================================"

# Start server
python3 -m unlook.server_bootstrap --enable-2k-mode

echo "Server started with 2K configuration"
