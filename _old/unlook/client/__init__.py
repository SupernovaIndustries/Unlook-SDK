"""
Modulo client per la connessione a scanner UnLook.
"""

from .scanner import UnlookClient
from .camera import CameraClient
from .projector import ProjectorClient
from .streaming import StreamClient
from .calibration import Calibrator, CalibrationData
from .processing import (
    StructuredLightProcessor, ScanProcessor, ProcessingResult,
    PatternType, PatternDirection
)
from .stereo import (
    StereoCalibrator, StereoCalibrationData, StereoProcessor
)
from .export import ModelExporter
from .auto_calibration import AutoCalibration, CalibrationStatus

# Importa direttamente da common.events invece che da .scanner
from ..common.events import UnlookClientEvent

# Importazioni condizionali per librerie esterne
try:
    from .open3d_utils import Open3DWrapper
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from .mesh_processing import PyMeshLabProcessor, TrimeshProcessor
    MESH_PROCESSING_AVAILABLE = True
except ImportError:
    MESH_PROCESSING_AVAILABLE = False