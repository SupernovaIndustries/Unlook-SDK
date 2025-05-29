#!/bin/bash

# UnLook Scanner Auto-Startup Script
# This script automatically starts the UnLook server with optimizations
# Works from any directory where Unlook-SDK is located

echo "================================================"
echo "🚀 UnLook Scanner Startup - Optimized Version"
echo "================================================"

# Determine script directory and find Unlook-SDK
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNLOOK_DIR=""

# Look for Unlook-SDK in current script directory first
if [[ -f "$SCRIPT_DIR/unlook/server_bootstrap.py" ]]; then
    UNLOOK_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/Unlook-SDK/unlook/server_bootstrap.py" ]]; then
    UNLOOK_DIR="$SCRIPT_DIR/Unlook-SDK"
# Look in user home directory
elif [[ -f "$HOME/Unlook-SDK/unlook/server_bootstrap.py" ]]; then
    UNLOOK_DIR="$HOME/Unlook-SDK"
# Look in current working directory
elif [[ -f "$(pwd)/Unlook-SDK/unlook/server_bootstrap.py" ]]; then
    UNLOOK_DIR="$(pwd)/Unlook-SDK"
elif [[ -f "$(pwd)/unlook/server_bootstrap.py" ]]; then
    UNLOOK_DIR="$(pwd)"
else
    echo "❌ Error: Unlook-SDK directory not found!"
    echo "Searched in:"
    echo "  - $SCRIPT_DIR"
    echo "  - $SCRIPT_DIR/Unlook-SDK"
    echo "  - $HOME/Unlook-SDK"
    echo "  - $(pwd)/Unlook-SDK"
    echo "  - $(pwd)"
    echo ""
    echo "Please ensure Unlook-SDK is properly installed."
    exit 1
fi

# Change to UnLook SDK directory
cd "$UNLOOK_DIR" || {
    echo "❌ Error: Cannot access Unlook-SDK directory at $UNLOOK_DIR"
    exit 1
}

echo "📁 Current directory: $(pwd)"

# Activate virtual environment
echo "🐍 Activating Python virtual environment..."
source venv/bin/activate || {
    echo "❌ Error: Virtual environment not found or failed to activate"
    echo "Please create virtual environment: python -m venv venv"
    exit 1
}

echo "✅ Virtual environment activated"

# Pull latest changes from git
echo "📡 Pulling latest changes from git..."
git pull || {
    echo "⚠️  Warning: Git pull failed (network issue or no changes)"
    echo "Continuing with current version..."
}

echo "✅ Git pull completed"

# Wait a moment for system to be ready
echo "⏳ Waiting 3 seconds for system initialization..."
sleep 3

# Start UnLook server with full optimizations
echo "🔥 Starting UnLook server with optimizations:"
echo "   - Protocol V2: ENABLED (bandwidth optimization)"
echo "   - Pattern preprocessing: ENABLED (advanced level)"
echo "   - Hardware sync: ENABLED (30 FPS)"
echo "   - GPU acceleration: ENABLED"
echo "   - Delta encoding: ENABLED"
echo "   - Adaptive compression: ENABLED"
echo ""

python unlook/server_bootstrap.py \
    --enable-protocol-v2 \
    --enable-pattern-preprocessing \
    --preprocessing-level advanced \
    --enable-sync \
    --sync-fps 30 \
    --log-level INFO

echo ""
echo "================================================"
echo "🔴 UnLook Scanner Server Stopped"
echo "================================================"