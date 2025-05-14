"""Scanning examples for Unlook SDK."""

# Main examples
from .static_scanning_example_fixed import main as static_scan
from .realtime_scanning_example import main as realtime_scan
from .comprehensive_scan_debug import main as debug_scan

# Utilities
from .scan_from_images import main as scan_offline
from .debug_triangulation import main as debug_triangulation

__all__ = [
    "static_scan",
    "realtime_scan",
    "debug_scan",
    "scan_offline",
    "debug_triangulation"
]
