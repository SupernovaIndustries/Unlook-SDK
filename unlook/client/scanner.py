"""
Client principale per la connessione a uno scanner UnLook.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq

from ..core import (
    EventType, EventEmitter,
    Message, MessageType,
    ScannerInfo, DiscoveryService,
    generate_uuid, get_machine_info, deserialize_binary_message,
    DEFAULT_CONTROL_PORT, DEFAULT_TIMEOUT, MAX_RETRIES
)

logger = logging.getLogger(__name__)


class UnlookClient(EventEmitter):
    """
    Client principale per la connessione a uno scanner UnLook.
    Fornisce un'interfaccia unificata per tutte le funzionalità dello scanner.
    """

    def __init__(
            self,
            client_name: str = "UnlookClient",
            auto_discover: bool = True,
            discovery_callback: Optional[Callable[[ScannerInfo, bool], None]] = None
    ):
        """
        Inizializza un nuovo client UnLook.

        Args:
            client_name: Nome del client
            auto_discover: Avvia automaticamente la discovery degli scanner
            discovery_callback: Callback per la discovery degli scanner
        """
        super().__init__()

        # Identificazione client
        self.name = client_name
        self.id = generate_uuid()

        # Stato connessione
        self.connected = False
        self.scanner: Optional[ScannerInfo] = None
        self.scanner_info: Dict[str, Any] = {}

        # ZeroMQ
        self.zmq_context = zmq.Context()
        self.control_socket = None
        self.poller = zmq.Poller()

        # Lock
        self._lock = threading.RLock()

        # Discovery
        self.discovery = DiscoveryService()
        self.discovered_scanners: Dict[str, ScannerInfo] = {}

        # Blocco sottoclassi - lazy loading
        self._camera = None
        self._projector = None
        self._stream = None

        # Avvia la discovery se richiesta
        if auto_discover:
            self.start_discovery(discovery_callback)

    @property
    def camera(self):
        """Lazy-loading del client telecamera."""
        if self._camera is None:
            # Import ritardato per evitare cicli di importazione
            from .camera import CameraClient
            self._camera = CameraClient(self)
        return self._camera

    @property
    def projector(self):
        """Lazy-loading del client proiettore."""
        if self._projector is None:
            # Import ritardato per evitare cicli di importazione
            from .projector import ProjectorClient
            self._projector = ProjectorClient(self)
        return self._projector

    @property
    def stream(self):
        """Lazy-loading del client streaming."""
        if self._stream is None:
            # Import ritardato per evitare cicli di importazione
            from .streaming import StreamClient
            self._stream = StreamClient(self)
        return self._stream

    # Metodi di connessione e gestione scanner
    def start_discovery(self, callback: Optional[Callable[[ScannerInfo, bool], None]] = None):
        """
        Avvia la discovery degli scanner.

        Args:
            callback: Callback chiamato quando viene trovato o perso uno scanner
        """

        def _discovery_handler(scanner: ScannerInfo, added: bool):
            # Aggiorna la lista interna
            with self._lock:
                if added:
                    self.discovered_scanners[scanner.uuid] = scanner
                    logger.info(f"Scanner trovato: {scanner}")
                    self.emit(EventType.SCANNER_FOUND, scanner)
                else:
                    if scanner.uuid in self.discovered_scanners:
                        del self.discovered_scanners[scanner.uuid]
                    logger.info(f"Scanner perso: {scanner}")
                    self.emit(EventType.SCANNER_LOST, scanner)

            # Chiama il callback utente se fornito
            if callback:
                callback(scanner, added)

        # Avvia la discovery
        self.discovery.start_discovery(_discovery_handler)
        logger.info("Discovery scanner avviata")

    def get_discovered_scanners(self) -> List[ScannerInfo]:
        """
        Ottiene la lista degli scanner scoperti.

        Returns:
            Lista di scanner
        """
        with self._lock:
            return list(self.discovered_scanners.values())

    def connect(self, scanner_or_endpoint: Any, timeout: int = DEFAULT_TIMEOUT) -> bool:
        """
        Connette a uno scanner.

        Args:
            scanner_or_endpoint: Scanner oggetto o endpoint (tcp://host:port) o UUID
            timeout: Timeout di connessione in millisecondi

        Returns:
            True se la connessione ha successo, False altrimenti
        """
        if self.connected:
            self.disconnect()

        # Estrai endpoint
        endpoint = None
        if isinstance(scanner_or_endpoint, ScannerInfo):
            endpoint = scanner_or_endpoint.endpoint
            self.scanner = scanner_or_endpoint
        elif isinstance(scanner_or_endpoint, str):
            # Verifica se è un UUID o un endpoint
            if "://" in scanner_or_endpoint:
                # È un endpoint
                endpoint = scanner_or_endpoint
                # Crea uno scanner "dummy" per memorizzare le informazioni
                host, port = endpoint.replace("tcp://", "").split(":")
                self.scanner = ScannerInfo(
                    name="Unknown Scanner",
                    host=host,
                    port=int(port),
                    scanner_uuid=generate_uuid()
                )
            else:
                # Potrebbe essere un UUID, cerca di trovare lo scanner
                uuid = scanner_or_endpoint
                found_scanner = None

                # Cerca lo scanner nella lista di quelli scoperti
                with self._lock:
                    for scanner in self.discovered_scanners.values():
                        if scanner.uuid == uuid:
                            found_scanner = scanner
                            break

                if found_scanner:
                    self.scanner = found_scanner
                    endpoint = found_scanner.endpoint
                    logger.info(f"Trovato scanner con UUID {uuid}, endpoint: {endpoint}")
                else:
                    logger.error(f"Scanner con UUID {uuid} non trovato. Esegui prima discovery.")
                    return False
        else:
            logger.error(
                "scanner_or_endpoint deve essere un oggetto ScannerInfo, un endpoint (tcp://host:port) o un UUID")
            return False

        logger.info(f"Connessione a {endpoint}...")

        try:
            # Crea socket di controllo
            self.control_socket = self.zmq_context.socket(zmq.REQ)
            self.control_socket.setsockopt(zmq.LINGER, 0)
            self.control_socket.setsockopt(zmq.RCVTIMEO, timeout)
            self.control_socket.connect(endpoint)

            # Registra con il poller
            self.poller.register(self.control_socket, zmq.POLLIN)

            # Invia messaggio HELLO
            hello_msg = Message(
                msg_type=MessageType.HELLO,
                payload={
                    "client_info": {
                        "name": self.name,
                        "id": self.id,
                        "version": "1.0",
                        **get_machine_info()
                    }
                }
            )

            # Invia e attendi risposta
            self.control_socket.send(hello_msg.to_bytes())

            # Attendi risposta con timeout
            socks = dict(self.poller.poll(timeout))
            if self.control_socket not in socks or socks[self.control_socket] != zmq.POLLIN:
                raise TimeoutError("Timeout durante la connessione")

            # Ricevi risposta
            response_data = self.control_socket.recv()
            response = Message.from_bytes(response_data)

            # Memorizza informazioni scanner
            self.scanner_info = response.payload
            if self.scanner:
                self.scanner.info = response.payload

            # Aggiorna stato
            self.connected = True
            logger.info(f"Connesso a {self.scanner.name} ({self.scanner.uuid})")

            # Emetti evento
            self.emit(EventType.CONNECTED, self.scanner)

            return True

        except Exception as e:
            # Pulisci in caso di errore
            logger.error(f"Errore durante la connessione: {e}")
            self._cleanup_connection()
            self.emit(EventType.ERROR, str(e))
            return False

    # Retrocompatibilità con la vecchia API di eventi
    def on_event(self, event: EventType, callback: Callable):
        """
        Registra un callback per un evento (compatibilità con versioni precedenti).

        Args:
            event: Tipo di evento
            callback: Funzione da chiamare
        """
        self.on(event, callback)

class UnlookClient(EventEmitter):
    """
    Main client for connecting to an UnLook scanner.
    Provides a unified interface for all scanner functionalities.
    """

    def __init__(
            self,
            client_name: str = "UnlookClient",
            auto_discover: bool = True,
            discovery_callback: Optional[Callable[[ScannerInfo, bool], None]] = None
    ):
        """
        Initialize a new UnLook client.

        Args:
            client_name: Client name
            auto_discover: Automatically start scanner discovery
            discovery_callback: Callback for scanner discovery
        """
        super().__init__()

        # Client identification
        self.name = client_name
        self.id = generate_uuid()

        # Connection state
        self.connected = False
        self.scanner: Optional[ScannerInfo] = None
        self.scanner_info: Dict[str, Any] = {}

        # ZeroMQ
        self.zmq_context = zmq.Context()
        self.control_socket = None
        self.poller = zmq.Poller()

        # Lock
        self._lock = threading.RLock()

        # Discovery
        self.discovery = DiscoveryService()
        self.discovered_scanners: Dict[str, ScannerInfo] = {}

        # Subclasses block - lazy loading
        self._camera = None
        self._projector = None
        self._stream = None

        # Start discovery if requested
        if auto_discover:
            self.start_discovery(discovery_callback)