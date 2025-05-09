"""
UnLook SDK for communication and control of structured light 3D scanners.

This SDK provides a complete interface for working with UnLook scanners,
allowing to control cameras, projectors, acquire images, perform
3D scans and reconstruct 3D models.

The architecture is divided into two main parts:
- Client: for connecting to and controlling the scanner
- Server: for implementing the scanner server

See the documentation and examples to start using the SDK.
"""

__version__ = "0.1.0"
__author__ = "UnLook Team"

# Core components
from .core.events import EventType
from .core.discovery import ScannerInfo, DiscoveryService

# For server-only mode detection
import builtins

# Client API - For applications that need to connect to a scanner
# Import client modules conditionally to prevent circular imports in server-only mode
_client_imported = False
try:
    # Check if we're in server-only mode (set in server_bootstrap.py)
    _SERVER_ONLY_MODE = getattr(builtins, '_SERVER_ONLY_MODE', False)
    if not _SERVER_ONLY_MODE:
        from .client import UnlookClient
        _client_imported = True
except (ImportError, NameError):
    # If builtins isn't found or client can't be imported
    try:
        from .client import UnlookClient
        _client_imported = True
    except ImportError:
        pass

# High-level 3D scanning API
UnlookScanner = None
if _client_imported:
    try:
        from .client.scanner3d import UnlookScanner
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

    # Main classes
    'UnlookClient',
    'UnlookServer',
    'UnlookScanner',

    # Utility functions
    'get_projector_classes'
]