"""
Model definitions and integrations for Unlook SDK.

This package contains neural network models and other processing models
that can be used to enhance the Unlook SDK's scanning capabilities.

Available modules:
- mvs2d: Multi-view stereo vision models
- handpose_hagrid: HAGRID-based hand gesture recognition
"""

# Import key components for easier access
try:
    from .handpose_hagrid import HAGRIDRecognizer, HAGRIDGestureType
except ImportError:
    # This can happen if dependencies aren't installed or
    # if the module files haven't been downloaded yet
    pass