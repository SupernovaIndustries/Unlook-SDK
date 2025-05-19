"""
Client module for connecting to UnLook scanners.
"""

# Import from new structure
from .scanner import UnlookClient, Scanner3D, ScanConfig
from .scanner.scanner3d import create_scanner, ScanResult, ScanConfig as Scanner3DConfig

# Import from new structure
from .scanning import (
    StaticScanner,
    StaticScanConfig,
    RealTimeScanner,
    RealTimeScanConfig
)
from .scanning.calibration import StereoCalibrator
from .projector import ProjectorClient
from . import scanning
from . import projector

# Import other modules and classes dynamically to avoid circular dependencies

__all__ = [
    'UnlookClient',
    'StereoCalibrator',
    'Scanner3D',
    'create_scanner',
    'ScanResult',
    'Scanner3DConfig',
    'ScanConfig',
    'StaticScanner',
    'StaticScanConfig',
    'RealTimeScanner',
    'RealTimeScanConfig',
    'ProjectorClient',
    'scanning',
    'projector'
    # Classes that will be lazy-loaded:
    # 'CameraClient', 'StreamClient',
    # 'Calibrator', 'CalibrationData',
    # 'StereoCalibrationData', 'StereoProcessor',
    # 'ModelExporter'
]

# Lazy loading function for optional modules
def _lazy_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        def _missing_module(*args, **kwargs):
            raise ImportError(f"Module {module_name} is not available")
        return _missing_module

# Dynamically create properties that will be available at module level
# but will only be imported when actually used

# Conditional imports for optional external libraries
try:
    from .open3d_utils import Open3DWrapper
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    Open3DWrapper = None

try:
    from .mesh_processing import PyMeshLabProcessor, TrimeshProcessor
    MESH_PROCESSING_AVAILABLE = True
except ImportError:
    MESH_PROCESSING_AVAILABLE = False
    PyMeshLabProcessor = None
    TrimeshProcessor = None