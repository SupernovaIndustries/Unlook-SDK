"""
Interface module for the DLP342X projector.
"""

# Re-export the DLPC342X components to maintain the same interface
from .dlp342x.dlpc342x_i2c import (
    DLPC342XController,
    OperatingMode,
    Color,
    BorderEnable,
    TestPattern,
    DiagonalLineSpacing,
    GridLines
)

__all__ = [
    'DLPC342XController',
    'OperatingMode',
    'Color',
    'BorderEnable',
    'TestPattern',
    'DiagonalLineSpacing',
    'GridLines'
]