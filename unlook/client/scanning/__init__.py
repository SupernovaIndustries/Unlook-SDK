"""Scanning subsystem for Unlook SDK."""

from . import patterns
from . import calibration
from . import reconstruction

# Scanner implementations
from .static_scanner import StaticScanner, StaticScanConfig
# TODO: Remove after refactoring
# from .realtime_scanner import RealTimeScanner, RealTimeScanConfig

# New modular components
from .pattern_manager import PatternManager, PatternInfo
from .capture_module import CaptureModule
from .reconstruction_module import ReconstructionModule, ReconstructionConfig

__all__ = [
    "patterns", 
    "calibration", 
    "reconstruction",
    "StaticScanner",
    "StaticScanConfig",
    # "RealTimeScanner", 
    # "RealTimeScanConfig",
    # New modular components
    "PatternManager",
    "PatternInfo", 
    "CaptureModule",
    "ReconstructionModule",
    "ReconstructionConfig"
]
