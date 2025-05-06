"""
Eventi generati dal client UnLook.
"""

import enum

class UnlookClientEvent(enum.Enum):
    """Eventi generati dal client UnLook."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    SCANNER_FOUND = "scanner_found"
    SCANNER_LOST = "scanner_lost"
    ERROR = "error"
    STREAM_STARTED = "stream_started"
    STREAM_STOPPED = "stream_stopped"
    SCAN_STARTED = "scan_started"
    SCAN_COMPLETED = "scan_completed"
    SCAN_ERROR = "scan_error"