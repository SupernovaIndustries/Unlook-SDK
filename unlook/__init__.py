"""
UnLook SDK - The Arduino of Computer Vision.
Simple 3D scanning and depth sensing for everyone.

Quick start:
    from unlook import quick_capture, quick_scan
    quick_capture('photo.jpg')  # Capture image
    quick_scan('object.ply')   # 3D scan
"""

__version__ = "2.0.0"
__author__ = "UnLook Team"

# Core components
from .core.events import EventType
from .core.discovery import ScannerInfo, DiscoveryService

# For server-only mode detection
import builtins
import logging

# Simple API - For most users (NEW!)
try:
    from .simple import UnlookSimple, quick_capture, quick_scan
except ImportError:
    UnlookSimple = None
    quick_capture = None
    quick_scan = None

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
if _client_imported:
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

    # Simple API (RECOMMENDED!)
    'UnlookSimple',
    'quick_capture',
    'quick_scan',

    # Core types
    'EventType',
    'ScannerInfo',
    'DiscoveryService',

    # Advanced API
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
    print("  from unlook import quick_capture, quick_scan")
    print("  quick_capture('photo.jpg')  # Capture image")
    print("  quick_scan('object.ply')    # 3D scan")
    print("\nSimple usage:")
    print("  from unlook import UnlookSimple")
    print("  scanner = UnlookSimple()")
    print("  scanner.connect()")
    print("  image = scanner.capture()")
    print("\nExamples:")
    print("  python -m unlook.examples.basic.hello_unlook")
    print("\nDocumentation: https://docs.unlook.io")

# Auto-connect for interactive use
def connect():
    """Quick connect for interactive Python sessions."""
    if UnlookSimple:
        scanner = UnlookSimple(debug=True)
        if scanner.connect():
            print("Connected! Use 'scanner' variable.")
            return scanner
        else:
            print("No scanner found. Check connection.")
            return None
    else:
        print("Simple API not available. Check installation.")
        return None