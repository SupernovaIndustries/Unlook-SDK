"""
UnLook SDK - The Arduino of Computer Vision.
Simple 3D scanning and depth sensing for everyone.

Quick start:
    from unlook import quick_capture, quick_scan
    quick_capture('photo.jpg')  # Capture image
    quick_scan('object.ply')   # 3D scan
"""

# Import version from pyproject.toml to maintain single source of truth
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("unlook-sdk")
except (ImportError, importlib.metadata.PackageNotFoundError):
    # Fallback for development installs
    __version__ = "2.0.0"
__author__ = "UnLook Team"

# Core components
from .core.events import EventType
from .core.discovery import ScannerInfo, DiscoveryService

# For server-only mode detection
import builtins
import logging

# Simple API removed - use UnlookClient directly

# Client API - For advanced users
_client_imported = False
try:
    # Check if we're in server-only mode (set in server_bootstrap.py)
    _SERVER_ONLY_MODE = getattr(builtins, '_SERVER_ONLY_MODE', False)
    if not _SERVER_ONLY_MODE:
        from .client import UnlookClient
        _client_imported = True
except (ImportError, NameError) as e:
    # If builtins isn't found or client can't be imported
    logger = logging.getLogger(__name__)
    logger.debug(f"Import error: {e}")
    try:
        from .client import UnlookClient
        _client_imported = True
    except ImportError as e2:
        logger.error(f"Failed to import UnlookClient: {e2}")
        UnlookClient = None
        _client_imported = False

# High-level 3D scanning API
UnlookScanner = None
if _client_imported and not _SERVER_ONLY_MODE:
    try:
        from .client.scanner import UnlookScanner
    except ImportError:
        # Optional dependencies might not be available
        UnlookScanner = None
    
# Conditional import for server components - Only import on Raspberry Pi
import platform
import sys

if 'arm' in platform.machine():
    try:
        from .server import UnlookServer

        # Convenience function to get projector classes without direct imports
        def get_projector_classes():
            from .server.hardware.projector import (
                DLPC342XController,
                OperatingMode,
                Color,
                BorderEnable,
                TestPattern,
                DiagonalLineSpacing,
                GridLines
            )
            return {
                'DLPC342XController': DLPC342XController,
                'OperatingMode': OperatingMode,
                'Color': Color,
                'BorderEnable': BorderEnable,
                'TestPattern': TestPattern,
                'DiagonalLineSpacing': DiagonalLineSpacing,
                'GridLines': GridLines
            }
    except ImportError as e:
        # The server might not be available on all systems
        def get_projector_classes():
            raise ImportError("Server components are not available on this system")

else:
    # Not running on a Raspberry Pi, provide dummy implementations
    UnlookServer = None

    def get_projector_classes():
        raise ImportError("Server components are only available on Raspberry Pi")

# Public API
__all__ = [
    # Version
    '__version__',
    '__author__',

    # Core types
    'EventType',
    'ScannerInfo',
    'DiscoveryService',

    # Main API
    'UnlookClient',
    'UnlookServer',
    'UnlookScanner',

    # Utility functions
    'get_projector_classes'
]

# Quick help function
def hello():
    """Print quick start guide."""
    print("Welcome to UnLook - The Arduino of Computer Vision!")
    print("\nQuick start:")
    print("  from unlook.client.scanner.scanner import UnlookClient")
    print("  client = UnlookClient(auto_discover=True)")
    print("  client.connect(client.get_discovered_scanners()[0])")
    print("  image = client.camera.capture_image('cam0')")
    print("\nExamples:")
    print("  python -m unlook.examples.basic.hello_unlook")
    print("  python -m unlook.examples.test_protocol_v2_integration")
    print("\nDocumentation: https://docs.unlook.io")

# Auto-connect for interactive use
def connect():
    """Quick connect for interactive Python sessions."""
    if UnlookClient:
        try:
            client = UnlookClient(auto_discover=True)
            import time
            time.sleep(2)  # Wait for discovery
            scanners = client.get_discovered_scanners()
            if scanners and client.connect(scanners[0]):
                print("Connected! Use 'client' variable.")
                return client
            else:
                print("No scanner found. Check connection.")
                return None
        except Exception as e:
            print(f"Connection failed: {e}")
            return None
    else:
        print("UnlookClient not available. Check installation.")
        return None