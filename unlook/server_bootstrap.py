#!/usr/bin/env python3
"""
Bootstrap script for the UnLook server.
This script avoids circular import issues by starting the server
in a completely independent way, suitable for use with systemd.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import socket
from pathlib import Path

# Logger will be configured in setup_logging()
logger = logging.getLogger("ServerBoot")

# Fix: Add the root directory (Unlook-SDK) to the Python path for proper import resolution
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Set a flag in the global namespace to inform the unlook module
# that we're running in server-only mode and should not import client modules
import builtins
builtins._SERVER_ONLY_MODE = True


def get_local_ip():
    """Get the local IP address the server will be running on."""
    try:
        # Create a temporary socket to discover our IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # We don't need to actually connect to Google DNS, just use it to determine interface
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP: {e}")
        return "127.0.0.1"  # Default to localhost


def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    # Set up file handler if log file is specified
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up log file {log_file}: {e}")


def main():
    """Main function for starting the server."""
    parser = argparse.ArgumentParser(description="UnLook 3D Scanner Server")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      default="INFO", help="Set the logging level")
    parser.add_argument("--daemon", action="store_true", help="Run as a daemon process")
    parser.add_argument("--enable-pattern-preprocessing", action="store_true",
                      help="Enable GPU-accelerated pattern preprocessing on Raspberry Pi")
    parser.add_argument("--preprocessing-level", choices=["basic", "advanced", "full"],
                      default="basic", help="Level of preprocessing to perform")
    parser.add_argument("--enable-sync", action="store_true",
                      help="Enable hardware camera synchronization")
    parser.add_argument("--sync-fps", type=float, default=30.0,
                      help="FPS for software sync trigger (default: 30)")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_file, log_level)
    
    # Print system info
    local_ip = get_local_ip()
    logger.info(f"Starting UnLook server on {local_ip}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Root directory: {root_dir}")

    try:
        # Using server-only mode (set earlier), we can directly import just the server module
        # without the risk of circular imports with client modules
        from unlook.server.scanner import UnlookServer

        # Load configuration
        config_path = args.config or os.path.join(os.path.dirname(__file__), "unlook_config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            config = {
                "server": {
                    "name": "UnLookScanner",
                    "control_port": 5555,
                    "stream_port": 5556,
                    "direct_stream_port": 5557  # Default for direct streaming
                }
            }

        # Extract server configuration
        server_config = config.get("server", {})
        
        # Apply command-line overrides
        if args.enable_pattern_preprocessing:
            server_config['enable_pattern_preprocessing'] = True
            server_config['preprocessing_level'] = args.preprocessing_level
            logger.info(f"Pattern preprocessing enabled at level: {args.preprocessing_level}")
        
        if args.enable_sync:
            server_config['enable_sync'] = True
            server_config['sync_fps'] = args.sync_fps
            logger.info(f"Hardware sync enabled at {args.sync_fps} FPS")
        
        # Display server config for debugging
        logger.info(f"Server name: {server_config.get('name', 'UnLookScanner')}")
        logger.info(f"Control port: {server_config.get('control_port', 5555)}")
        logger.info(f"Stream port: {server_config.get('stream_port', 5556)}")
        logger.info(f"Direct stream port: {server_config.get('direct_stream_port', 5557)}")

        # Create and start the server
        server = UnlookServer(
            name=server_config.get("name", "UnLookScanner"),
            control_port=server_config.get("control_port", 5555),
            stream_port=server_config.get("stream_port", 5556),
            direct_stream_port=server_config.get("direct_stream_port", 5557),
            scanner_uuid=server_config.get("scanner_uuid"),
            auto_start=True,
            enable_preprocessing=server_config.get('enable_pattern_preprocessing', False),
            preprocessing_level=server_config.get('preprocessing_level', 'basic'),
            enable_sync=server_config.get('enable_sync', False),
            sync_fps=server_config.get('sync_fps', 30.0)
        )

        logger.info("Server started successfully")

        # Set up signal handler for clean shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received, terminating server...")
            server.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # If running as a daemon, notify systemd
        if args.daemon and os.environ.get('NOTIFY_SOCKET'):
            try:
                import systemd.daemon
                systemd.daemon.notify('READY=1')
                logger.info("Notified systemd that service is ready")
            except ImportError:
                logger.warning("systemd module not available, cannot notify")

        # Keep the process running
        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error during server startup: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()