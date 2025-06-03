#!/bin/bash
# UnLook Server Startup Script

echo "Starting UnLook Server..."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the server with 2K configuration
python3 unlook/server_bootstrap.py --config unlook_config_2k.json