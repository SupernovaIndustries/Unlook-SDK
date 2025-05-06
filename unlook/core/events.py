"""
Sistema di eventi per il framework Unlook.
Fornisce un meccanismo di comunicazione basato su eventi per evitare dipendenze circolari.
"""

import enum
import logging
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)


class EventType(enum.Enum):
    """Tipi di eventi supportati nel framework Unlook."""
    # Eventi di connessione
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"

    # Eventi di discovery
    SCANNER_FOUND = "scanner_found"
    SCANNER_LOST = "scanner_lost"

    # Eventi di streaming
    STREAM_STARTED = "stream_started"
    STREAM_STOPPED = "stream_stopped"

    # Eventi di scansione
    SCAN_STARTED = "scan_started"
    SCAN_COMPLETED = "scan_completed"
    SCAN_ERROR = "scan_error"

    # Altri eventi
    ERROR = "error"
    STATUS_CHANGED = "status_changed"
    CONFIG_CHANGED = "config_changed"


class EventEmitter:
    """
    Implementa un pattern observer per la gestione degli eventi.
    Consente a diverse parti del sistema di comunicare senza dipendenze dirette.
    """

    def __init__(self):
        """Inizializza un emitter di eventi."""
        self._event_callbacks: Dict[EventType, List[Callable]] = {
            event: [] for event in EventType
        }

    def on(self, event_type: EventType, callback: Callable) -> None:
        """
        Registra un callback per un evento.

        Args:
            event_type: Tipo di evento
            callback: Funzione da chiamare quando l'evento viene emesso
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []

        self._event_callbacks[event_type].append(callback)

    def off(self, event_type: EventType, callback: Callable) -> None:
        """
        Rimuove un callback per un evento.

        Args:
            event_type: Tipo di evento
            callback: Funzione da rimuovere
        """
        if event_type in self._event_callbacks and callback in self._event_callbacks[event_type]:
            self._event_callbacks[event_type].remove(callback)

    def emit(self, event_type: EventType, *args, **kwargs) -> None:
        """
        Emette un evento, eseguendo tutti i callback registrati.

        Args:
            event_type: Tipo di evento
            *args, **kwargs: Argomenti passati ai callback
        """
        if event_type not in self._event_callbacks:
            return

        for callback in self._event_callbacks[event_type]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Errore nel callback per l'evento {event_type}: {e}")