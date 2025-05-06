"""
Server module for the UnLook scanner.
"""

# Import the main server class
from .scanner import UnlookServer

# Import hardware components with lazy loading to avoid circular dependencies

# Function to lazily import hardware components
def _get_projector_classes():
    """Get projector-related classes."""
    from .hardware.projector import (
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

def _get_camera_classes():
    """Get camera-related classes."""
    from .hardware.camera import PiCamera2Manager
    return {
        'PiCamera2Manager': PiCamera2Manager
    }

# Export main class
__all__ = ['UnlookServer']

# Export hardware components on demand without immediate import
# This will be populated when the hardware components are actually requested