"""
Server principale per lo scanner UnLook.
"""

import logging
import threading
import time
import os
import signal
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq
import numpy as np

from ..core import (
    DiscoveryService, ScannerInfo,
    Message, MessageType,
    EventType, EventEmitter,
    DEFAULT_CONTROL_PORT, DEFAULT_STREAM_PORT,
    DEFAULT_JPEG_QUALITY, DEFAULT_STREAM_FPS,
    generate_uuid, get_machine_info,
    encode_image_to_jpeg, serialize_binary_message
)

logger = logging.getLogger(__name__)


class UnlookServer(EventEmitter):
    """
    Server principale per lo scanner UnLook. Gestisce le connessioni client,
    coordina le telecamere e il proiettore, e implementa la logica di business.
    """

    def __init__(
            self,
            name: str = "UnLookScanner",
            control_port: int = DEFAULT_CONTROL_PORT,
            stream_port: int = DEFAULT_STREAM_PORT,
            scanner_uuid: Optional[str] = None,
            auto_start: bool = True
    ):
        """
        Inizializza il server UnLook.

        Args:
            name: Nome dello scanner
            control_port: Porta per i comandi di controllo
            stream_port: Porta per lo streaming video
            scanner_uuid: UUID univoco dello scanner (generato se None)
            auto_start: Avvia automaticamente il server
        """
        super().__init__()

        # Identificazione scanner
        self.name = name
        self.uuid = scanner_uuid or generate_uuid()

        # Configurazione rete
        self.control_port = control_port
        self.stream_port = stream_port

        # Stato server
        self.running = False
        self.clients = set()
        self._lock = threading.RLock()

        # Inizializza contesto ZMQ
        self.zmq_context = zmq.Context()

        # Socket per i comandi
        self.control_socket = self.zmq_context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{control_port}")

        # Socket per lo streaming
        self.stream_socket = self.zmq_context.socket(zmq.PUB)
        self.stream_socket.bind(f"tcp://*:{stream_port}")

        # Discovery network
        self.discovery = DiscoveryService()

        # Hardware - Lazy loading
        self._projector = None
        self._camera_manager = None

        # Streaming
        self.streaming_active = False
        self.active_streams = []  # Lista di stream attivi per gestire più telecamere
        self.streaming_thread = None
        self.streaming_fps = DEFAULT_STREAM_FPS
        self.jpeg_quality = DEFAULT_JPEG_QUALITY

        # Scansione
        self.scanning = False
        self.scan_thread = None

        # Thread di controllo
        self.control_thread = None

        # Callback per gestione messaggi personalizzati
        self.message_handlers: Dict[MessageType, Callable] = self._init_message_handlers()

        # Avvio automatico
        if auto_start:
            self.start()

    @property
    def projector(self):
        """Lazy-loading del controller proiettore."""
        if self._projector is None:
            try:
                # Import ritardato per evitare importazioni circolari
                from .hardware.projector import DLPC342XController
                # Bus e indirizzo hardcodati come richiesto
                self._projector = DLPC342XController(bus=3, address=0x1b)
                logger.info("Proiettore inizializzato: bus=3, address=0x1B")
            except Exception as e:
                logger.error(f"Errore durante l'inizializzazione del proiettore: {e}")
                self._projector = None
        return self._projector

    @property
    def camera_manager(self):
        """Lazy-loading del manager telecamere."""
        if self._camera_manager is None:
            try:
                # Import ritardato per evitare importazioni circolari
                from .hardware.camera import PiCamera2Manager
                self._camera_manager = PiCamera2Manager()
                logger.info("Manager telecamere inizializzato")
            except Exception as e:
                logger.error(f"Errore durante l'inizializzazione del manager telecamere: {e}")
                self._camera_manager = None
        return self._camera_manager

    def _init_message_handlers(self) -> Dict[MessageType, Callable]:
        """
        Inizializza gli handler dei messaggi.

        Returns:
            Dizionario di handler per tipo di messaggio
        """
        handlers = {
            MessageType.HELLO: self._handle_hello,
            MessageType.INFO: self._handle_info,

            # Handlers del proiettore
            MessageType.PROJECTOR_MODE: self._handle_projector_mode,
            MessageType.PROJECTOR_PATTERN: self._handle_projector_pattern,

            # Handlers della telecamera
            MessageType.CAMERA_LIST: self._handle_camera_list,
            MessageType.CAMERA_CONFIG: self._handle_camera_config,
            MessageType.CAMERA_CAPTURE: self._handle_camera_capture,
            MessageType.CAMERA_STREAM_START: self._handle_camera_stream_start,
            MessageType.CAMERA_STREAM_STOP: self._handle_camera_stream_stop,

            # Handlers per la cattura sincronizzata multicamera
            MessageType.CAMERA_CAPTURE_MULTI: self._handle_camera_capture_multi,

            # Altri handlers...
        }
        return handlers

    # Implementazione degli handler dei messaggi...