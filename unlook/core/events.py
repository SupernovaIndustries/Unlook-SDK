"""
Event system for the Unlook framework.
Provides an event-based communication mechanism to avoid circular dependencies.
"""

import enum
import logging
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)


class EventType(enum.Enum):
    """Event types supported in the Unlook framework."""
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"

    # Discovery events
    SCANNER_FOUND = "scanner_found"
    SCANNER_LOST = "scanner_lost"

    # Streaming events
    STREAM_STARTED = "stream_started"
    STREAM_STOPPED = "stream_stopped"

    # Direct streaming events (new)
    DIRECT_STREAM_STARTED = "direct_stream_started"
    DIRECT_STREAM_STOPPED = "direct_stream_stopped"
    DIRECT_STREAM_SYNC = "direct_stream_sync"  # Evento di sincronizzazione proiettore-camera
    DIRECT_STREAM_ERROR = "direct_stream_error"
    PATTERN_CHANGED = "pattern_changed"  # Evento di cambio pattern

    # Scanning events
    SCAN_STARTED = "scan_started"
    SCAN_COMPLETED = "scan_completed"
    SCAN_ERROR = "scan_error"

    # Other events
    ERROR = "error"
    STATUS_CHANGED = "status_changed"
    CONFIG_CHANGED = "config_changed"


class EventEmitter:
    """
    Implements an observer pattern for event handling.
    Allows different parts of the system to communicate without direct dependencies.
    """

    def __init__(self):
        """Initialize an event emitter."""
        self._event_callbacks: Dict[EventType, List[Callable]] = {
            event: [] for event in EventType
        }

    def on(self, event_type: EventType, callback: Callable) -> None:
        """
        Register a callback for an event.

        Args:
            event_type: Event type
            callback: Function to call when the event is emitted
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []

        self._event_callbacks[event_type].append(callback)

    def off(self, event_type: EventType, callback: Callable) -> None:
        """
        Remove a callback for an event.

        Args:
            event_type: Event type
            callback: Function to remove
        """
        if event_type in self._event_callbacks and callback in self._event_callbacks[event_type]:
            self._event_callbacks[event_type].remove(callback)

    def emit(self, event_type: EventType, *args, **kwargs) -> None:
        """
        Emit an event, executing all registered callbacks.

        Args:
            event_type: Event type
            *args, **kwargs: Arguments passed to callbacks
        """
        if event_type not in self._event_callbacks:
            return

        for callback in self._event_callbacks[event_type]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for event {event_type}: {e}")