"""
Server-side placeholder for point cloud neural network modules.

This is a minimal implementation that avoids the need for heavy dependencies
like PyTorch on the server side, while still allowing client code to function.
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)

# Constants for availability
TORCH_AVAILABLE = False
OPEN3D_AVAILABLE = False
TORCH_CUDA = False


def get_point_cloud_enhancer(*args, **kwargs):
    """
    Server-side placeholder that returns None.
    The server should never need the actual neural network functionality.
    """
    logger.warning("Neural network point cloud enhancement not available on server side")
    return None