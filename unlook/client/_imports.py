"""
Standard imports for UnLook client modules.

This module provides consistent import patterns to avoid redundancy
and ensure all client modules use the same import style.
"""

# Standard library imports - sorted alphabetically
import json
import logging
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    NamedTuple, Type, TypeVar, Generic, Protocol
)

# Third-party imports - sorted alphabetically
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None
    plt = None

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for common patterns
ImageArray = np.ndarray  # Type alias for image arrays
PointCloud = np.ndarray  # Type alias for point clouds (Nx3)
CameraMatrix = np.ndarray  # Type alias for camera matrices (3x3)
ProjectionMatrix = np.ndarray  # Type alias for projection matrices (3x4)

__all__ = [
    # Standard library
    'json', 'logging', 'math', 'os', 'threading', 'time',
    'ABC', 'abstractmethod', 'dataclass', 'field', 'Enum', 'Path',
    'Any', 'Callable', 'Dict', 'List', 'Optional', 'Tuple', 'Union',
    'NamedTuple', 'Type', 'TypeVar', 'Generic', 'Protocol',
    # Third-party
    'cv2', 'CV2_AVAILABLE',
    'np', 'NUMPY_AVAILABLE',
    'o3d', 'OPEN3D_AVAILABLE',
    'matplotlib', 'plt', 'MATPLOTLIB_AVAILABLE',
    'zmq', 'ZMQ_AVAILABLE',
    'mp', 'MEDIAPIPE_AVAILABLE',
    # Utils
    'logger',
    # Type aliases
    'ImageArray', 'PointCloud', 'CameraMatrix', 'ProjectionMatrix'
]