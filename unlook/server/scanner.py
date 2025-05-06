"""
Server principale per lo scanner UnLook.
"""

import json
import logging
import threading
import time
import os
import signal
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq
import numpy as np

from ..common.discovery import UnlookDiscovery
from ..common.protocol import Message, MessageType
from ..common.constants import (
    DEFAULT_CONTROL_PORT, DEFAULT_STREAM_PORT,
    DEFAULT_JPEG_QUALITY, DEFAULT_STREAM_FPS
)
from ..common.utils import (
    generate_uuid, get_machine_info,
    encode_image_to_jpeg, serialize_binary_message
)

from .camera.picamera2 import PiCamera2Manager
from .projector.dlp342x import DLPC342XController, OperatingMode, TestPattern, Color

logger = logging.getLogger(__name__)


class UnlookServer:
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
        self.discovery = UnlookDiscovery()

        # Hardware - Proiettore hardcodato su bus I2C 3, indirizzo 0x1B
        try:
            # Bus e indirizzo hardcodati come richiesto
            self.projector = DLPC342XController(bus=3, address=0x1b)
            logger.info("Proiettore inizializzato: bus=3, address=0x1B")
        except Exception as e:
            logger.error(f"Errore durante l'inizializzazione del proiettore: {e}")
            self.projector = None

        try:
            self.camera_manager = PiCamera2Manager()
            logger.info("Manager telecamere inizializzato")
        except Exception as e:
            logger.error(f"Errore durante l'inizializzazione del manager telecamere: {e}")
            self.camera_manager = None

        # Streaming
        self.streaming_active = False
        self.streaming_camera_id = None
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

            # Handlers della scansione
            MessageType.SCAN_START: self._handle_scan_start,
            MessageType.SCAN_STOP: self._handle_scan_stop,
            MessageType.SCAN_STATUS: self._handle_scan_status,
        }
        return handlers

    def start(self):
        """Avvia il server."""
        if self.running:
            return

        logger.info(f"Avvio server UnLook: {self.name} ({self.uuid})")

        # Registra il servizio per la discovery
        scanner_info = {
            "name": self.name,
            "control_port": str(self.control_port),
            "stream_port": str(self.stream_port),
            **get_machine_info()
        }

        self.discovery.register_scanner(
            name=self.name,
            port=self.control_port,
            scanner_uuid=self.uuid,
            scanner_info=scanner_info
        )

        # Avvia il thread di controllo
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        # Gestione segnali per terminazione pulita
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Server UnLook avviato su porta {self.control_port} (streaming: {self.stream_port})")

    def stop(self):
        """Ferma il server."""
        if not self.running:
            return

        logger.info("Arresto server UnLook...")

        # Ferma lo streaming se attivo
        self.stop_streaming()

        # Ferma la scansione se attiva
        self.stop_scan()

        # Disattiva il proiettore
        if self.projector:
            try:
                self.projector.set_operating_mode(OperatingMode.Standby)
                self.projector.close()
            except Exception as e:
                logger.error(f"Errore durante la chiusura del proiettore: {e}")

        # Chiude il manager telecamere
        if self.camera_manager:
            try:
                self.camera_manager.close()
            except Exception as e:
                logger.error(f"Errore durante la chiusura del manager telecamere: {e}")

        # Ferma il thread di controllo
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        # Rimuove la registrazione dalla discovery
        self.discovery.unregister_scanner()

        # Chiude i socket ZMQ
        self.control_socket.close()
        self.stream_socket.close()
        self.zmq_context.term()

        logger.info("Server UnLook arrestato")

    def _signal_handler(self, sig, frame):
        """Gestisce i segnali di terminazione."""
        logger.info(f"Ricevuto segnale {sig}, arresto in corso...")
        self.stop()
        os._exit(0)

    def _control_loop(self):
        """Loop principale per processare i messaggi di controllo."""
        logger.info("Thread di controllo avviato")

        # Imposta timeout sul socket per permettere controlli periodici
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 secondo

        while self.running:
            try:
                # Attende un messaggio con timeout
                try:
                    data = self.control_socket.recv()
                except zmq.error.Again:
                    # Timeout, continua il loop
                    continue

                # Processa il messaggio
                try:
                    message = Message.from_bytes(data)
                    response = self._process_message(message)

                    # Se la risposta è None, significa che è già stata inviata
                    if response is not None:
                        self.control_socket.send(response.to_bytes())

                except Exception as e:
                    logger.error(f"Errore durante l'elaborazione del messaggio: {e}")
                    # Invia un messaggio di errore
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        payload={"error": str(e)}
                    )
                    self.control_socket.send(error_msg.to_bytes())

            except Exception as e:
                if self.running:  # Evita log di errori durante lo shutdown
                    logger.error(f"Errore nel loop di controllo: {e}")

        logger.info("Thread di controllo terminato")

    def _process_message(self, message: Message) -> Optional[Message]:
        """
        Processa un messaggio in arrivo.

        Args:
            message: Messaggio da processare

        Returns:
            Messaggio di risposta o None se già inviato
        """
        logger.debug(f"Ricevuto messaggio: {message.msg_type.value}")

        # Verifica se esiste un handler specifico
        handler = self.message_handlers.get(message.msg_type)
        if handler:
            try:
                response = handler(message)
                # Se la risposta è None, significa che è già stata inviata
                return response
            except Exception as e:
                logger.error(f"Errore nell'handler {message.msg_type.value}: {e}")
                return Message.create_error(
                    message,
                    f"Errore nell'elaborazione del messaggio: {e}"
                )
        else:
            logger.warning(f"Nessun handler per il messaggio: {message.msg_type.value}")
            return Message.create_error(
                message,
                f"Tipo di messaggio non supportato: {message.msg_type.value}",
                error_code=400
            )

    # HANDLERS DEI MESSAGGI

    def _handle_hello(self, message: Message) -> Message:
        """Gestisce i messaggi HELLO."""
        client_info = message.payload.get("client_info", {})
        client_id = client_info.get("id", "unknown")

        with self._lock:
            self.clients.add(client_id)

        logger.info(f"Nuovo client connesso: {client_id}")

        # Invia informazioni sullo scanner
        return Message.create_reply(
            message,
            {
                "scanner_name": self.name,
                "scanner_uuid": self.uuid,
                "control_port": self.control_port,
                "stream_port": self.stream_port,
                "capabilities": self._get_capabilities(),
                "status": {
                    "streaming": self.streaming_active,
                    "scanning": self.scanning
                }
            }
        )

    def _handle_info(self, message: Message) -> Message:
        """Gestisce i messaggi INFO."""
        return Message.create_reply(
            message,
            {
                "scanner_name": self.name,
                "scanner_uuid": self.uuid,
                "control_port": self.control_port,
                "stream_port": self.stream_port,
                "capabilities": self._get_capabilities(),
                "status": {
                    "streaming": self.streaming_active,
                    "scanning": self.scanning,
                    "projector_connected": self.projector is not None,
                    "camera_connected": self.camera_manager is not None,
                    "cameras_available": len(self.camera_manager.get_cameras()) if self.camera_manager else 0,
                },
                "system_info": get_machine_info()
            }
        )

    def _handle_projector_mode(self, message: Message) -> Message:
        """Gestisce i messaggi PROJECTOR_MODE."""
        if not self.projector:
            return Message.create_error(message, "Proiettore non disponibile")

        mode_str = message.payload.get("mode")
        if not mode_str:
            return Message.create_error(message, "Modalità non specificata")

        try:
            # Converte la stringa in enum
            mode = OperatingMode[mode_str]

            # Imposta la modalità
            success = self.projector.set_operating_mode(mode)
            if not success:
                return Message.create_error(message, f"Errore nell'impostazione della modalità {mode_str}")

            logger.info(f"Modalità proiettore impostata: {mode_str}")
            return Message.create_reply(
                message,
                {"success": True, "mode": mode_str}
            )

        except (KeyError, ValueError) as e:
            return Message.create_error(
                message,
                f"Modalità non valida: {mode_str}. Error: {str(e)}"
            )

    def _handle_projector_pattern(self, message: Message) -> Message:
        """Gestisce i messaggi PROJECTOR_PATTERN."""
        if not self.projector:
            return Message.create_error(message, "Proiettore non disponibile")

        # Prima imposta la modalità test pattern
        try:
            self.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
        except Exception as e:
            return Message.create_error(
                message,
                f"Errore nell'impostazione della modalità TestPatternGenerator: {e}"
            )

        # Ottieni il tipo di pattern
        pattern_type = message.payload.get("pattern_type")
        if not pattern_type:
            return Message.create_error(message, "Tipo di pattern non specificato")

        success = False

        try:
            if pattern_type == "solid_field":
                # Ottieni colore
                color_str = message.payload.get("color", "White")
                color = Color[color_str]

                # Genera pattern
                success = self.projector.generate_solid_field(color)

            elif pattern_type == "horizontal_lines":
                # Ottieni parametri
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                fg_width = message.payload.get("foreground_width", 4)
                bg_width = message.payload.get("background_width", 20)

                # Genera pattern
                success = self.projector.generate_horizontal_lines(
                    bg_color, fg_color, fg_width, bg_width
                )

            elif pattern_type == "vertical_lines":
                # Ottieni parametri
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                fg_width = message.payload.get("foreground_width", 4)
                bg_width = message.payload.get("background_width", 20)

                # Genera pattern
                success = self.projector.generate_vertical_lines(
                    bg_color, fg_color, fg_width, bg_width
                )

            elif pattern_type == "grid":
                # Ottieni parametri
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                h_fg_width = message.payload.get("h_foreground_width", 4)
                h_bg_width = message.payload.get("h_background_width", 20)
                v_fg_width = message.payload.get("v_foreground_width", 4)
                v_bg_width = message.payload.get("v_background_width", 20)

                # Genera pattern
                success = self.projector.generate_grid(
                    bg_color, fg_color,
                    h_fg_width, h_bg_width,
                    v_fg_width, v_bg_width
                )

            elif pattern_type == "checkerboard":
                # Ottieni parametri
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                h_count = message.payload.get("horizontal_count", 8)
                v_count = message.payload.get("vertical_count", 6)

                # Genera pattern
                success = self.projector.generate_checkerboard(
                    bg_color, fg_color, h_count, v_count
                )

            elif pattern_type == "colorbars":
                # Genera pattern
                success = self.projector.generate_colorbars()

            else:
                return Message.create_error(
                    message,
                    f"Tipo di pattern non supportato: {pattern_type}"
                )

            if success:
                logger.info(f"Pattern proiettore generato: {pattern_type}")
                return Message.create_reply(
                    message,
                    {"success": True, "pattern_type": pattern_type}
                )
            else:
                return Message.create_error(
                    message,
                    f"Errore nella generazione del pattern {pattern_type}"
                )

        except (KeyError, ValueError, Exception) as e:
            return Message.create_error(
                message,
                f"Errore nell'impostazione del pattern: {e}"
            )

    def _handle_camera_list(self, message: Message) -> Message:
        """Gestisce i messaggi CAMERA_LIST."""
        if not self.camera_manager:
            return Message.create_error(message, "Manager telecamere non disponibile")

        try:
            cameras = self.camera_manager.get_cameras()
            camera_list = []

            for cam_id, cam_info in cameras.items():
                camera_list.append({
                    "id": cam_id,
                    "name": cam_info.get("name", f"Camera {cam_id}"),
                    "resolution": cam_info.get("resolution", [1920, 1080]),
                    "fps": cam_info.get("fps", 30),
                    "capabilities": cam_info.get("capabilities", [])
                })

            return Message.create_reply(
                message,
                {"cameras": camera_list}
            )

        except Exception as e:
            return Message.create_error(
                message,
                f"Errore nell'ottenimento della lista telecamere: {e}"
            )

    def _handle_camera_config(self, message: Message) -> Message:
        """Gestisce i messaggi CAMERA_CONFIG."""
        if not self.camera_manager:
            return Message.create_error(message, "Manager telecamere non disponibile")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "ID telecamera non specificato")

        config = message.payload.get("config", {})

        try:
            success = self.camera_manager.configure_camera(camera_id, config)

            if success:
                return Message.create_reply(
                    message,
                    {"success": True, "camera_id": camera_id}
                )
            else:
                return Message.create_error(
                    message,
                    f"Errore nella configurazione della telecamera {camera_id}"
                )

        except Exception as e:
            return Message.create_error(
                message,
                f"Errore nella configurazione della telecamera: {e}"
            )

    def _handle_camera_capture(self, message: Message) -> Message:
        """Gestisce i messaggi CAMERA_CAPTURE."""
        if not self.camera_manager:
            return Message.create_error(message, "Manager telecamere non disponibile")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "ID telecamera non specificato")

        try:
            # Cattura l'immagine
            image = self.camera_manager.capture_image(camera_id)
            if image is None:
                return Message.create_error(
                    message,
                    f"Errore nella cattura dell'immagine dalla telecamera {camera_id}"
                )

            # Codifica l'immagine in JPEG
            jpeg_quality = message.payload.get("jpeg_quality", self.jpeg_quality)
            jpeg_data = encode_image_to_jpeg(image, jpeg_quality)

            # Prepara i metadati
            height, width = image.shape[:2]
            metadata = {
                "width": width,
                "height": height,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "format": "jpeg",
                "camera_id": camera_id,
                "timestamp": time.time()
            }

            # Crea un messaggio speciale di tipo stringa (non enum) per la risposta
            binary_response = serialize_binary_message(
                "camera_capture_response",  # Usa una stringa invece di un enum
                metadata,
                jpeg_data
            )

            # Invia la risposta binaria direttamente
            self.control_socket.send(binary_response)
            return None  # Ritorna None per indicare che la risposta è già stata inviata

        except Exception as e:
            logger.error(f"Errore nella cattura dell'immagine: {e}")
            return Message.create_error(
                message,
                f"Errore nella cattura dell'immagine: {e}"
            )

    def _handle_camera_capture_multi(self, message: Message) -> Message:
        """Gestisce i messaggi CAMERA_CAPTURE_MULTI per la cattura sincronizzata da più telecamere."""
        if not self.camera_manager:
            return Message.create_error(message, "Manager telecamere non disponibile")

        camera_ids = message.payload.get("camera_ids", [])
        if not camera_ids:
            return Message.create_error(message, "Lista degli ID telecamera non specificata")

        try:
            # Apri tutte le telecamere richieste se non sono già aperte
            for camera_id in camera_ids:
                if not self.camera_manager.open_camera(camera_id):
                    return Message.create_error(
                        message,
                        f"Errore nell'apertura della telecamera {camera_id}"
                    )

            # Cattura in modo sincronizzato (implementazione semplificata)
            images = {}
            jpeg_quality = message.payload.get("jpeg_quality", self.jpeg_quality)

            for camera_id in camera_ids:
                # Cattura l'immagine
                image = self.camera_manager.capture_image(camera_id)
                if image is None:
                    return Message.create_error(
                        message,
                        f"Errore nella cattura dell'immagine dalla telecamera {camera_id}"
                    )

                # Codifica l'immagine in JPEG
                jpeg_data = encode_image_to_jpeg(image, jpeg_quality)

                # Prepara i metadati
                height, width = image.shape[:2]
                metadata = {
                    "width": width,
                    "height": height,
                    "channels": image.shape[2] if len(image.shape) > 2 else 1,
                    "format": "jpeg",
                    "camera_id": camera_id,
                    "timestamp": time.time()
                }

                # Aggiungi alla lista
                images[camera_id] = {
                    "metadata": metadata,
                    "jpeg_data": jpeg_data
                }

            # Prepara la risposta
            response_payload = {
                "num_cameras": len(images),
                "camera_ids": camera_ids,
                "timestamp": time.time(),
                "type": "multi_camera_response"  # Aggiunto esplicitamente il tipo
            }

            # Creiamo un messaggio di risposta speciale che contiene tutte le immagini
            all_metadata = {camera_id: images[camera_id]["metadata"] for camera_id in camera_ids}
            response_payload["images_metadata"] = all_metadata

            # Usa la funzione di serializzazione esistente invece di implementare manualmente
            binary_message = serialize_binary_message(
                "multi_camera_response",  # Tipo esplicito
                response_payload,
                None  # Inizialmente nessun dato binario
            )

            # Preparazione del messaggio completo
            # Ora dobbiamo aggiungere manualmente i dati delle immagini dopo l'header
            header_size = binary_message[:4]  # I primi 4 byte contengono la dimensione dell'header
            header = binary_message[:4 + int.from_bytes(header_size, byteorder='little')]

            # Costruisci il messaggio completo
            message_parts = [header]

            # Aggiungi i dati binari delle immagini in sequenza
            for camera_id in camera_ids:
                jpeg_data = images[camera_id]["jpeg_data"]
                # Dimensione dei dati JPEG (4 bytes)
                message_parts.append(len(jpeg_data).to_bytes(4, byteorder='little'))
                # Dati JPEG
                message_parts.append(jpeg_data)

            # Unisci tutte le parti
            response_data = b''.join(message_parts)

            # Invia la risposta direttamente tramite il socket REP
            self.control_socket.send(response_data)

            # Restituisce None per indicare che la risposta è già stata inviata
            return None

        except Exception as e:
            logger.error(f"Errore nella cattura sincronizzata: {e}")
            return Message.create_error(
                message,
                f"Errore nella cattura sincronizzata: {e}"
            )

    def _handle_camera_stream_start(self, message: Message) -> Message:
        """Gestisce i messaggi CAMERA_STREAM_START."""
        if not self.camera_manager:
            return Message.create_error(message, "Manager telecamere non disponibile")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "ID telecamera non specificato")

        fps = message.payload.get("fps", DEFAULT_STREAM_FPS)
        jpeg_quality = message.payload.get("jpeg_quality", DEFAULT_JPEG_QUALITY)

        # Ferma lo streaming precedente se attivo
        self.stop_streaming()

        # Avvia il nuovo streaming
        try:
            # Configura lo streaming
            self.streaming_camera_id = camera_id
            self.streaming_fps = fps
            self.jpeg_quality = jpeg_quality

            # Avvia il thread di streaming
            self.streaming_active = True
            self.streaming_thread = threading.Thread(
                target=self._streaming_loop,
                daemon=True
            )
            self.streaming_thread.start()

            logger.info(f"Streaming avviato per camera {camera_id} a {fps} FPS")

            return Message.create_reply(
                message,
                {
                    "success": True,
                    "camera_id": camera_id,
                    "stream_port": self.stream_port,
                    "fps": fps
                }
            )

        except Exception as e:
            self.streaming_active = False
            return Message.create_error(
                message,
                f"Errore nell'avvio dello streaming: {e}"
            )

    def _handle_camera_stream_stop(self, message: Message) -> Message:
        """Gestisce i messaggi CAMERA_STREAM_STOP."""
        self.stop_streaming()

        logger.info("Streaming fermato")

        return Message.create_reply(
            message,
            {"success": True}
        )

    def _handle_scan_start(self, message: Message) -> Message:
        """Gestisce i messaggi SCAN_START."""
        if self.scanning:
            return Message.create_error(message, "Scansione già in corso")

        # Configura parametri scansione
        scan_config = message.payload.get("config", {})

        # Avvia la scansione
        try:
            # Imposta flag scansione
            self.scanning = True

            # Avvia thread di scansione
            self.scan_thread = threading.Thread(
                target=self._scan_process,
                args=(scan_config,),
                daemon=True
            )
            self.scan_thread.start()

            logger.info("Scansione avviata")

            return Message.create_reply(
                message,
                {"success": True, "scan_id": generate_uuid()}
            )

        except Exception as e:
            self.scanning = False
            return Message.create_error(
                message,
                f"Errore nell'avvio della scansione: {e}"
            )

    def _handle_scan_stop(self, message: Message) -> Message:
        """Gestisce i messaggi SCAN_STOP."""
        if not self.scanning:
            return Message.create_reply(
                message,
                {"success": True, "already_stopped": True}
            )

        self.stop_scan()

        logger.info("Scansione fermata")

        return Message.create_reply(
            message,
            {"success": True}
        )

    def _handle_scan_status(self, message: Message) -> Message:
        """Gestisce i messaggi SCAN_STATUS."""
        return Message.create_reply(
            message,
            {
                "scanning": self.scanning,
                "progress": 0.0,  # Da implementare il calcolo del progresso
                "status": "running" if self.scanning else "idle"
            }
        )

    # FUNZIONI DI SUPPORTO

    def _get_capabilities(self) -> Dict[str, Any]:
        """
        Restituisce le capacità dello scanner.

        Returns:
            Dizionario con le capacità
        """
        capabilities = {
            "projector": {
                "available": self.projector is not None,
                "type": "DLP342X" if self.projector else None,
                "patterns": [p.name for p in TestPattern] if self.projector else [],
                "i2c_bus": 3,
                "i2c_address": "0x1B"
            },
            "cameras": {
                "available": self.camera_manager is not None,
                "count": len(self.camera_manager.get_cameras()) if self.camera_manager else 0,
                "stereo_capable": len(self.camera_manager.get_cameras()) >= 2 if self.camera_manager else False
            },
            "streaming": {
                "available": self.camera_manager is not None,
                "max_fps": 60
            },
            "scanning": {
                "available": self.projector is not None and self.camera_manager is not None,
                "multi_camera": len(self.camera_manager.get_cameras()) >= 2 if self.camera_manager else False
            }
        }
        return capabilities

    def _streaming_loop(self):
        """Loop di streaming video."""
        logger.info(f"Thread di streaming avviato (camera: {self.streaming_camera_id})")

        # Calcola l'intervallo tra i frame
        interval = 1.0 / self.streaming_fps

        while self.streaming_active:
            start_time = time.time()

            try:
                # Cattura immagine
                image = self.camera_manager.capture_image(self.streaming_camera_id)
                if image is None:
                    logger.error(f"Errore nella cattura dell'immagine dalla camera {self.streaming_camera_id}")
                    time.sleep(interval)  # Attendi comunque l'intervallo
                    continue

                # Codifica in JPEG
                jpeg_data = encode_image_to_jpeg(image, self.jpeg_quality)

                # Prepara i metadati
                height, width = image.shape[:2]
                metadata = {
                    "width": width,
                    "height": height,
                    "channels": image.shape[2] if len(image.shape) > 2 else 1,
                    "format": "jpeg",
                    "camera_id": self.streaming_camera_id,
                    "timestamp": time.time(),
                    "frame_number": 0  # TODO: implementare contatore frame
                }

                # Crea messaggio binario
                binary_message = serialize_binary_message(
                    "camera_frame",
                    metadata,
                    jpeg_data
                )

                # Pubblica il frame
                self.stream_socket.send(binary_message)

            except Exception as e:
                logger.error(f"Errore nello streaming: {e}")

            # Calcola tempo rimanente per rispettare il frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > interval * 1.1:
                # Log solo se il ritardo è significativo
                logger.warning(f"Streaming in ritardo: {elapsed:.4f}s > {interval:.4f}s")

        logger.info("Thread di streaming terminato")

    def stop_streaming(self):
        """Ferma lo streaming video."""
        if not self.streaming_active:
            return

        # Imposta flag per terminare il thread
        self.streaming_active = False

        # Attendi la terminazione del thread
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)

        self.streaming_thread = None
        self.streaming_camera_id = None

        logger.info("Streaming video fermato")

    def _scan_process(self, config: Dict[str, Any]):
        """
        Processo di scansione 3D.

        Args:
            config: Configurazione della scansione
        """
        logger.info(f"Processo di scansione avviato con config: {config}")

        # Da implementare: algoritmo di scansione 3D
        time.sleep(5)  # Simulazione scansione

        self.scanning = False
        logger.info("Processo di scansione completato")

    def stop_scan(self):
        """Ferma il processo di scansione."""
        if not self.scanning:
            return

        # Imposta flag per terminare il thread
        self.scanning = False

        # Attendi la terminazione del thread
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=2.0)

        self.scan_thread = None

        logger.info("Processo di scansione fermato")