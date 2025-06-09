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

# Add convenience function for sinusoidal pattern generation
def generate_sinusoidal_pattern(controller, frequency, phase_shift=0.0, 
                               orientation="vertical", amplitude=127.5, offset=127.5):
    """Convenience function to generate sinusoidal patterns.
    
    Args:
        controller: DLPC342XController instance
        frequency: Pattern frequency
        phase_shift: Phase shift in radians
        orientation: "vertical" or "horizontal"
        amplitude: Pattern amplitude
        offset: DC offset
        
    Returns:
        True if successful, False otherwise
    """
    return controller.generate_sinusoidal_pattern(
        frequency, phase_shift, orientation, amplitude, offset
    )