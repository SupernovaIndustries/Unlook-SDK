"""
Client principale per la connessione a uno scanner UnLook.
"""

import enum
import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq

from ..common.discovery import UnlookDiscovery, UnlookScanner
from ..common.protocol import Message, MessageType
from ..common.constants import (
    DEFAULT_CONTROL_PORT, DEFAULT_STREAM_PORT,
    DEFAULT_TIMEOUT, MAX_RETRIES
)
from ..common.utils import generate_uuid, get_machine_info, deserialize_binary_message

from .camera import CameraClient
from .projector import ProjectorClient
from .streaming import StreamClient
from ..common.events import UnlookClientEvent

logger = logging.getLogger(__name__)


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


class UnlookClient:
    """
    Client principale per la connessione a uno scanner UnLook.
    Fornisce un'interfaccia unificata per tutte le funzionalità dello scanner.
    """

    def __init__(
            self,
            client_name: str = "UnlookClient",
            auto_discover: bool = True,
            discovery_callback: Optional[Callable[[UnlookScanner, bool], None]] = None
    ):
        """
        Inizializza un nuovo client UnLook.

        Args:
            client_name: Nome del client
            auto_discover: Avvia automaticamente la discovery degli scanner
            discovery_callback: Callback per la discovery degli scanner
        """
        # Identificazione client
        self.name = client_name
        self.id = generate_uuid()

        # Stato connessione
        self.connected = False
        self.scanner: Optional[UnlookScanner] = None
        self.scanner_info: Dict[str, Any] = {}

        # ZeroMQ
        self.zmq_context = zmq.Context()
        self.control_socket = None
        self.poller = zmq.Poller()

        # Lock
        self._lock = threading.RLock()

        # Discovery
        self.discovery = UnlookDiscovery()
        self.discovered_scanners: Dict[str, UnlookScanner] = {}

        # Sub-clients
        self.camera = CameraClient(self)
        self.projector = ProjectorClient(self)
        self.stream = StreamClient(self)

        # Callbacks ed eventi
        self._event_callbacks: Dict[UnlookClientEvent, List[Callable]] = {
            event: [] for event in UnlookClientEvent
        }

        # Avvia la discovery se richiesta
        if auto_discover:
            self.start_discovery(discovery_callback)

    def start_discovery(self, callback: Optional[Callable[[UnlookScanner, bool], None]] = None):
        """
        Avvia la discovery degli scanner.

        Args:
            callback: Callback chiamato quando viene trovato o perso uno scanner
        """

        def _discovery_handler(scanner: UnlookScanner, added: bool):
            # Aggiorna la lista interna
            with self._lock:
                if added:
                    self.discovered_scanners[scanner.uuid] = scanner
                    logger.info(f"Scanner trovato: {scanner}")
                    self._emit_event(UnlookClientEvent.SCANNER_FOUND, scanner)
                else:
                    if scanner.uuid in self.discovered_scanners:
                        del self.discovered_scanners[scanner.uuid]
                    logger.info(f"Scanner perso: {scanner}")
                    self._emit_event(UnlookClientEvent.SCANNER_LOST, scanner)

            # Chiama il callback utente se fornito
            if callback:
                callback(scanner, added)

        # Avvia la discovery
        self.discovery.start_discovery(_discovery_handler)
        logger.info("Discovery scanner avviata")

    def stop_discovery(self):
        """Ferma la discovery degli scanner."""
        self.discovery.stop_discovery()
        logger.info("Discovery scanner fermata")

    def get_discovered_scanners(self) -> List[UnlookScanner]:
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
        if isinstance(scanner_or_endpoint, UnlookScanner):
            endpoint = scanner_or_endpoint.endpoint
            self.scanner = scanner_or_endpoint
        elif isinstance(scanner_or_endpoint, str):
            # Verifica se è un UUID o un endpoint
            if "://" in scanner_or_endpoint:
                # È un endpoint
                endpoint = scanner_or_endpoint
                # Crea uno scanner "dummy" per memorizzare le informazioni
                host, port = endpoint.replace("tcp://", "").split(":")
                self.scanner = UnlookScanner(
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
                "scanner_or_endpoint deve essere un oggetto UnlookScanner, un endpoint (tcp://host:port) o un UUID")
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

            # Nota: il client dello stream verrà connesso solo quando richiesto

            # Aggiorna stato
            self.connected = True
            logger.info(f"Connesso a {self.scanner.name} ({self.scanner.uuid})")

            # Emetti evento
            self._emit_event(UnlookClientEvent.CONNECTED, self.scanner)

            return True

        except Exception as e:
            # Pulisci in caso di errore
            logger.error(f"Errore durante la connessione: {e}")
            self._cleanup_connection()
            self._emit_event(UnlookClientEvent.ERROR, str(e))
            return False

    def disconnect(self):
        """Disconnette dallo scanner."""
        if not self.connected:
            return

        logger.info("Disconnessione dallo scanner...")

        # Ferma lo streaming se attivo
        self.stream.stop()

        # Chiudi la connessione
        self._cleanup_connection()

        # Aggiorna stato
        self.connected = False
        self._emit_event(UnlookClientEvent.DISCONNECTED)

        logger.info("Disconnesso dallo scanner")

    def _cleanup_connection(self):
        """Pulisce le risorse di connessione."""
        try:
            if self.control_socket:
                self.poller.unregister(self.control_socket)
                self.control_socket.close()
                self.control_socket = None

        except Exception as e:
            logger.error(f"Errore durante la pulizia della connessione: {e}")

    def send_message(
            self,
            msg_type: MessageType,
            payload: Dict[str, Any],
            timeout: int = DEFAULT_TIMEOUT,
            retries: int = MAX_RETRIES,
            binary_response: bool = False
    ) -> Tuple[bool, Optional[Message], Optional[bytes]]:
        """
        Invia un messaggio allo scanner.

        Args:
            msg_type: Tipo di messaggio
            payload: Payload del messaggio
            timeout: Timeout in millisecondi
            retries: Numero di tentativi in caso di errore
            binary_response: Se True, si aspetta una risposta binaria

        Returns:
            Tupla (successo, messaggio_risposta, dati_binari)
        """
        if not self.connected or not self.control_socket:
            logger.error("Impossibile inviare messaggio: non connesso")
            return False, None, None

        message = Message(msg_type=msg_type, payload=payload)
        attempt = 0

        while attempt < retries:
            try:
                # Invia il messaggio
                self.control_socket.send(message.to_bytes())

                # Attendi risposta con timeout
                socks = dict(self.poller.poll(timeout))
                if self.control_socket not in socks or socks[self.control_socket] != zmq.POLLIN:
                    logger.warning(f"Timeout durante l'attesa della risposta (tentativo {attempt + 1}/{retries})")
                    attempt += 1
                    continue

                # Ricevi risposta
                response_data = self.control_socket.recv()

                # Se ci aspettiamo una risposta binaria
                if binary_response:
                    # Deserializza la risposta binaria
                    try:
                        msg_type, payload, binary_data = deserialize_binary_message(response_data)
                        # Crea un messaggio dalla risposta
                        response = Message(
                            msg_type=MessageType(msg_type),
                            payload=payload
                        )
                        return True, response, binary_data
                    except Exception as e:
                        logger.error(f"Errore nella deserializzazione della risposta binaria: {e}")
                        return False, None, None
                else:
                    # Risposta normale
                    response = Message.from_bytes(response_data)

                    # Gestisci errori
                    if response.msg_type == MessageType.ERROR:
                        error_msg = response.payload.get("error_message", "Errore sconosciuto")
                        logger.error(f"Errore dal server: {error_msg}")
                        return False, response, None

                    return True, response, None

            except Exception as e:
                logger.error(f"Errore durante l'invio del messaggio: {e} (tentativo {attempt + 1}/{retries})")
                attempt += 1

                # Se non è l'ultimo tentativo, attendi prima di riprovare
                if attempt < retries:
                    time.sleep(0.5)

                # Verifica se il socket è ancora valido
                if attempt == retries - 1:
                    try:
                        # Prova a riconnettersi
                        logger.info("Tentativo di riconnessione...")
                        self._emit_event(UnlookClientEvent.RECONNECTING)

                        # Chiudi la connessione esistente
                        self._cleanup_connection()

                        # Ricrea il socket
                        self.control_socket = self.zmq_context.socket(zmq.REQ)
                        self.control_socket.setsockopt(zmq.LINGER, 0)
                        self.control_socket.setsockopt(zmq.RCVTIMEO, timeout)
                        self.control_socket.connect(self.scanner.endpoint)

                        # Registra con il poller
                        self.poller.register(self.control_socket, zmq.POLLIN)

                    except Exception as reconnect_error:
                        logger.error(f"Riconnessione fallita: {reconnect_error}")
                        self.connected = False
                        self._emit_event(UnlookClientEvent.DISCONNECTED)
                        return False, None, None

        # Tutti i tentativi falliti
        logger.error("Tutti i tentativi di invio del messaggio falliti")
        return False, None, None

    def get_info(self) -> Dict[str, Any]:
        """
        Ottiene informazioni sullo scanner.

        Returns:
            Dizionario con informazioni sullo scanner
        """
        success, response, _ = self.send_message(
            MessageType.INFO,
            {}
        )

        if success and response:
            return response.payload
        else:
            return {}

    def check_connection(self) -> bool:
        """
        Verifica se la connessione allo scanner è attiva.

        Returns:
            True se connesso, False altrimenti
        """
        if not self.connected:
            return False

        try:
            info = self.get_info()
            return bool(info)
        except Exception:
            return False

    def on_event(self, event: UnlookClientEvent, callback: Callable):
        """
        Registra un callback per un evento.

        Args:
            event: Tipo di evento
            callback: Funzione da chiamare
        """
        if event not in self._event_callbacks:
            raise ValueError(f"Evento non valido: {event}")

        self._event_callbacks[event].append(callback)

    def off_event(self, event: UnlookClientEvent, callback: Callable):
        """
        Rimuove un callback per un evento.

        Args:
            event: Tipo di evento
            callback: Funzione da rimuovere
        """
        if event not in self._event_callbacks:
            return

        if callback in self._event_callbacks[event]:
            self._event_callbacks[event].remove(callback)

    def _emit_event(self, event: UnlookClientEvent, *args, **kwargs):
        """
        Emette un evento.

        Args:
            event: Tipo di evento
            *args, **kwargs: Argomenti passati ai callback
        """
        if event not in self._event_callbacks:
            return

        for callback in self._event_callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Errore nel callback per l'evento {event}: {e}")

    def __del__(self):
        """Pulisce le risorse quando l'oggetto viene distrutto."""
        self.disconnect()
        self.stop_discovery()