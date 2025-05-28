"""
Server principale per lo scanner UnLook.
"""

import logging
import threading
import time
import os
import signal
import json
import base64
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
from ..core.protocol_v2 import ProtocolOptimizer

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
            direct_stream_port: int = None,  # Nuova porta per lo streaming diretto
            scanner_uuid: Optional[str] = None,
            auto_start: bool = True,
            enable_preprocessing: bool = False,
            preprocessing_level: str = "basic",
            enable_sync: bool = False,
            sync_fps: float = 30.0,
            enable_protocol_v2: bool = True
    ):
        """
        Inizializza il server UnLook.

        Args:
            name: Nome dello scanner
            control_port: Porta per i comandi di controllo
            stream_port: Porta per lo streaming video
            direct_stream_port: Porta per lo streaming diretto (generata se None)
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
        self.direct_stream_port = direct_stream_port or (stream_port + 1)  # Default: stream_port + 1

        # Stato server
        self.running = False
        self.clients = set()
        self._lock = threading.RLock()
        
        # Preprocessing and sync settings
        self.enable_preprocessing = enable_preprocessing
        self.preprocessing_level = preprocessing_level
        self.enable_sync = enable_sync
        self.sync_fps = sync_fps
        self.enable_protocol_v2 = enable_protocol_v2

        # Inizializza contesto ZMQ
        self.zmq_context = zmq.Context()

        # Socket per i comandi
        self.control_socket = self.zmq_context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{control_port}")

        # Socket per lo streaming
        self.stream_socket = self.zmq_context.socket(zmq.PUB)
        self.stream_socket.bind(f"tcp://*:{stream_port}")

        # Socket per lo streaming diretto
        self.direct_stream_socket = None  # Inizializzato solo quando necessario

        # Discovery network
        self.discovery = DiscoveryService()

        # Hardware - Lazy loading
        self._projector = None
        self._camera_manager = None
        self._led_controller = None
        self._gpu_preprocessor = None
        self._protocol_optimizer = None

        # Streaming
        self.streaming_active = False
        self.active_streams = []  # Lista di stream attivi per gestire più telecamere
        self.streaming_thread = None
        self.streaming_fps = DEFAULT_STREAM_FPS
        self.jpeg_quality = DEFAULT_JPEG_QUALITY

        # Streaming diretto
        self.direct_streaming_active = False
        self.active_direct_streams = []  # Lista di stream diretti attivi
        self.direct_streaming_thread = None
        self.direct_streaming_fps = 60  # FPS predefinito più alto per lo streaming diretto
        self.direct_jpeg_quality = 85  # Qualità JPEG predefinita più alta

        # Scansione
        self.scanning = False
        self.scan_thread = None

        # Projector pattern sequence state
        self.pattern_sequence_active = False
        self.pattern_sequence = []
        self.current_pattern_index = 0
        self.pattern_sequence_thread = None
        self.pattern_sequence_interval = 0.0
        self.pattern_sequence_loop = False
        
        # Projector-camera synchronization
        self.projector_camera_sync_enabled = False
        self.last_pattern_change_time = 0.0

        # Thread di controllo
        self.control_thread = None
        
        # Server startup time for uptime calculation
        self._start_time = time.time()

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
            MessageType.PROJECTOR_PATTERN_SEQUENCE: self._handle_projector_pattern_sequence,
            MessageType.PROJECTOR_PATTERN_SEQUENCE_STEP: self._handle_projector_pattern_sequence_step,
            MessageType.PROJECTOR_PATTERN_SEQUENCE_STOP: self._handle_projector_pattern_sequence_stop,

            # Handlers della telecamera
            MessageType.CAMERA_LIST: self._handle_camera_list,
            MessageType.CAMERA_CONFIG: self._handle_camera_config,
            MessageType.CAMERA_CAPTURE: self._handle_camera_capture,
            MessageType.CAMERA_STREAM_START: self._handle_camera_stream_start,
            MessageType.CAMERA_STREAM_STOP: self._handle_camera_stream_stop,

            # Handlers per la cattura sincronizzata multicamera
            MessageType.CAMERA_CAPTURE_MULTI: self._handle_camera_capture_multi,

            # Nuovi handler per lo streaming diretto
            MessageType.CAMERA_DIRECT_STREAM_START: self._handle_camera_direct_stream_start,
            MessageType.CAMERA_DIRECT_STREAM_STOP: self._handle_camera_direct_stream_stop,
            
            # Camera optimization handlers
            MessageType.CAMERA_OPTIMIZE: self._handle_camera_optimize,
            MessageType.CAMERA_AUTO_FOCUS: self._handle_camera_auto_focus,
            MessageType.CAMERA_TEST_CAPTURE: self._handle_camera_test_capture,
            
            # LED control handlers
            MessageType.LED_SET_INTENSITY: self._handle_led_set_intensity,
            MessageType.LED_ON: self._handle_led_on,
            MessageType.LED_OFF: self._handle_led_off,
            MessageType.LED_STATUS: self._handle_led_status,
            MessageType.LED_SET_CURRENT: self._handle_led_set_current,
            MessageType.LED_GET_CURRENT: self._handle_led_get_current,
            MessageType.LED_SET_ENABLE: self._handle_led_set_enable,
            MessageType.SYNC_METRICS: self._handle_sync_metrics,
            MessageType.SYNC_ENABLE: self._handle_sync_enable,
            MessageType.SYSTEM_STATUS: self._handle_system_status,

            # Altri handlers...
        }
        return handlers

    def _handle_info(self, message: Message) -> Message:
        """Handle INFO messages."""
        return Message.create_reply(
            message,
            {
                "scanner_name": self.name,
                "scanner_uuid": self.uuid,
                "control_port": self.control_port,
                "stream_port": self.stream_port,
                "direct_stream_port": self.direct_stream_port,  # Aggiunta porta streaming diretto
                "capabilities": self._get_capabilities(),
                "status": {
                    "streaming": self.streaming_active,
                    "direct_streaming": self.direct_streaming_active,  # Aggiunto stato dello streaming diretto
                    "scanning": self.scanning,
                    "projector_connected": self.projector is not None,
                    "camera_connected": self.camera_manager is not None,
                    "cameras_available": len(self.camera_manager.get_cameras()) if self.camera_manager else 0,
                    "pattern_sequence_active": self.pattern_sequence_active  # Nuovo stato per sequenze pattern
                },
                "system_info": get_machine_info()
            }
        )

    def _get_capabilities(self) -> Dict[str, Any]:
        """
        Return scanner capabilities.

        Returns:
            Dictionary with capabilities
        """
        capabilities = {
            "projector": {
                "available": self.projector is not None,
                "type": "DLP342X" if self.projector else None,
                "patterns": ["SolidField", "HorizontalLines", "VerticalLines", "Grid", "Checkerboard", "Colorbars"]
                if self.projector else [],
                "i2c_bus": 3,
                "i2c_address": "0x1B",
                "pattern_sequence": True,  # Capability for pattern sequences
                "camera_sync": True  # Capability for projector-camera sync
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
            "direct_streaming": {  # Nuova capacità per lo streaming diretto
                "available": self.camera_manager is not None,
                "max_fps": 120,  # Presupponiamo un framerate massimo più alto per lo streaming diretto
                "low_latency": True,
                "sync_capabilities": self.projector is not None  # Sincronizzazione disponibile se c'è un proiettore
            },
            "scanning": {
                "available": self.projector is not None and self.camera_manager is not None,
                "multi_camera": len(self.camera_manager.get_cameras()) >= 2 if self.camera_manager else False
            }
        }
        return capabilities

    def _apply_preprocessing_config(self, client_id: str, version: str, config: Dict[str, Any]):
        """
        Apply preprocessing configuration for a specific client.
        
        Args:
            client_id: Client identifier
            version: Requested preprocessing version
            config: Preprocessing configuration
        """
        logger.info(f"Applying preprocessing config for client {client_id}: {version}")
        
        # Store current version for this client
        if not hasattr(self, 'client_preprocessing_versions'):
            self.client_preprocessing_versions = {}
        
        self.client_preprocessing_versions[client_id] = version
        
        # Apply configuration based on version
        if version == "v1_legacy":
            # V1 Legacy: Disable GPU preprocessing, use standard protocol
            self._configure_v1_preprocessing(client_id)
            logger.info(f"Client {client_id} configured for V1 Legacy preprocessing")
            
        elif version == "v2_enhanced":
            # V2 Enhanced: Enable GPU preprocessing, use protocol v2
            self._configure_v2_preprocessing(client_id)
            logger.info(f"Client {client_id} configured for V2 Enhanced preprocessing")
            
        elif version == "auto":
            # Auto: Decide based on server capabilities and client needs
            if self.enable_protocol_v2 and self.enable_preprocessing:
                self._configure_v2_preprocessing(client_id)
                logger.info(f"Client {client_id} auto-configured for V2 Enhanced preprocessing")
            else:
                self._configure_v1_preprocessing(client_id)
                logger.info(f"Client {client_id} auto-configured for V1 Legacy preprocessing")
        
        # Store the current version as server attribute for global reference
        self.current_preprocessing_version = version

    def _configure_v1_preprocessing(self, client_id: str):
        """Configure server for V1 legacy preprocessing."""
        # For V1, we want minimal preprocessing and standard protocol
        self._client_configs = getattr(self, '_client_configs', {})
        self._client_configs[client_id] = {
            'use_protocol_v2': False,
            'enable_gpu_preprocessing': False,
            'compression_level': 0,
            'delta_encoding': False
        }

    def _configure_v2_preprocessing(self, client_id: str):
        """Configure server for V2 enhanced preprocessing."""
        # For V2, we want full preprocessing and protocol v2
        self._client_configs = getattr(self, '_client_configs', {})
        self._client_configs[client_id] = {
            'use_protocol_v2': True,
            'enable_gpu_preprocessing': True,
            'compression_level': 6,
            'delta_encoding': True
        }

    def _get_client_config(self, client_id: str) -> Dict[str, Any]:
        """Get configuration for a specific client."""
        if not hasattr(self, '_client_configs'):
            return {}
        return self._client_configs.get(client_id, {})

    def _handle_camera_direct_stream_start(self, message: Message) -> Message:
        """Handle CAMERA_DIRECT_STREAM_START messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager non disponibile")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "ID telecamera non specificato")

        fps = message.payload.get("fps", self.direct_streaming_fps)
        jpeg_quality = message.payload.get("jpeg_quality", self.direct_jpeg_quality)
        sync_with_projector = message.payload.get("sync_with_projector", False)
        synchronization_pattern_interval = message.payload.get("synchronization_pattern_interval", 5)
        low_latency = message.payload.get("low_latency", True)

        # Controlla se lo stream è già attivo per questa telecamera
        for stream_info in self.active_direct_streams:
            if stream_info["camera_id"] == camera_id:
                logger.warning(f"Stream diretto già attivo per la telecamera {camera_id}, verrà riutilizzato")
                # Aggiorna configurazione
                stream_info["fps"] = fps
                stream_info["jpeg_quality"] = jpeg_quality
                stream_info["sync_with_projector"] = sync_with_projector
                stream_info["synchronization_pattern_interval"] = synchronization_pattern_interval
                stream_info["low_latency"] = low_latency
                stream_info["last_activity"] = time.time()

                return Message.create_reply(
                    message,
                    {
                        "success": True,
                        "camera_id": camera_id,
                        "direct_port": self.direct_stream_port,
                        "fps": fps,
                        "stream_id": stream_info["stream_id"]
                    }
                )

        # Inizializza il socket di streaming diretto se necessario
        if self.direct_stream_socket is None:
            try:
                # Close any existing socket first to ensure clean state
                if hasattr(self, 'direct_stream_socket') and self.direct_stream_socket:
                    try:
                        self.direct_stream_socket.close()
                    except Exception as close_err:
                        logger.warning(f"Error closing existing direct stream socket: {close_err}")
                
                # Utilizziamo un socket PAIR per comunicazione bidirezionale
                self.direct_stream_socket = self.zmq_context.socket(zmq.PAIR)

                # Configurazione per bassa latenza
                if low_latency:
                    self.direct_stream_socket.setsockopt(zmq.SNDHWM, 2)  # Buffer di invio ridotto
                    self.direct_stream_socket.setsockopt(zmq.RCVHWM, 2)  # Buffer di ricezione ridotto
                    self.direct_stream_socket.setsockopt(zmq.LINGER, 0)  # Non aspettare alla chiusura
                    self.direct_stream_socket.setsockopt(zmq.IMMEDIATE, 1)  # Non accodare messaggi
                    self.direct_stream_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout on receive
                    self.direct_stream_socket.setsockopt(zmq.SNDTIMEO, 100)  # 100ms timeout on send

                # Bind sulla porta di streaming diretto
                bind_address = f"tcp://*:{self.direct_stream_port}"
                logger.info(f"Binding direct stream socket to {bind_address}")
                self.direct_stream_socket.bind(bind_address)
                logger.info(f"Socket di streaming diretto inizializzato sulla porta {self.direct_stream_port}")
            except Exception as e:
                logger.error(f"Errore nell'inizializzazione del socket di streaming diretto: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return Message.create_error(message, f"Errore nell'inizializzazione dello streaming diretto: {e}")

        # Configura streaming
        stream_id = generate_uuid()
        stream_info = {
            "stream_id": stream_id,
            "camera_id": camera_id,
            "fps": fps,
            "jpeg_quality": jpeg_quality,
            "sync_with_projector": sync_with_projector,
            "synchronization_pattern_interval": synchronization_pattern_interval,
            "low_latency": low_latency,
            "active": True,
            "start_time": time.time(),
            "last_activity": time.time(),
            "frame_count": 0,
            "pattern_index": 0,  # Indice del pattern corrente per la sincronizzazione
            "next_pattern_time": time.time(),  # Timestamp per il prossimo cambio di pattern
            "stats": {
                "dropped_frames": 0,
                "skipped_frames": 0,
                "processing_time_ms": 0
            }
        }

        # Aggiungi alla lista di stream diretti attivi
        self.active_direct_streams.append(stream_info)

        # Avvia thread di streaming diretto se non è già attivo
        if not self.direct_streaming_active:
            self.direct_streaming_active = True
            self.direct_streaming_thread = threading.Thread(
                target=self._direct_streaming_loop,
                daemon=True
            )
            self.direct_streaming_thread.start()

        logger.info(f"Streaming diretto avviato per la telecamera {camera_id} a {fps} FPS (ID: {stream_id})")

        return Message.create_reply(
            message,
            {
                "success": True,
                "camera_id": camera_id,
                "direct_port": self.direct_stream_port,
                "fps": fps,
                "stream_id": stream_id
            }
        )

    def _handle_camera_direct_stream_stop(self, message: Message) -> Message:
        """Handle CAMERA_DIRECT_STREAM_STOP messages."""
        camera_id = message.payload.get("camera_id")
        stream_id = message.payload.get("stream_id")

        # Se né camera_id né stream_id sono specificati, interrompi tutti gli stream
        if not camera_id and not stream_id:
            self.stop_direct_streaming()
            logger.info("Tutti gli stream diretti interrotti")
            return Message.create_reply(message, {"success": True})

        # Altrimenti cerca lo stream specifico da interrompere
        for i, stream_info in enumerate(self.active_direct_streams):
            if ((camera_id and stream_info["camera_id"] == camera_id) or
                    (stream_id and stream_info["stream_id"] == stream_id)):
                # Rimuovi stream
                self.active_direct_streams.pop(i)
                logger.info(
                    f"Stream diretto interrotto per la telecamera {stream_info['camera_id']} (ID: {stream_info['stream_id']})")
                break

        # Se non ci sono più stream attivi, interrompi il thread
        if not self.active_direct_streams:
            self.stop_direct_streaming()

        return Message.create_reply(message, {"success": True})

    def stop_direct_streaming(self):
        """Interrompe lo streaming video diretto."""
        if not self.direct_streaming_active:
            return

        # Imposta flag per terminare il thread
        self.direct_streaming_active = False

        # Attendi che il thread termini
        if self.direct_streaming_thread and self.direct_streaming_thread.is_alive():
            self.direct_streaming_thread.join(timeout=2.0)

        self.direct_streaming_thread = None

        # Chiudi il socket di streaming diretto
        if self.direct_stream_socket:
            try:
                self.direct_stream_socket.close()
                self.direct_stream_socket = None
            except Exception as e:
                logger.error(f"Errore nella chiusura del socket di streaming diretto: {e}")

        logger.info("Streaming video diretto interrotto")

    def _direct_streaming_loop(self):
        """Loop di streaming video diretto ottimizzato per bassa latenza e sincronizzazione."""
        logger.info("Thread di streaming diretto avviato")

        # Dizionario per tracciare l'ultimo frame inviato per ogni telecamera
        last_frame_times = {}

        # Dati condivisi per la sincronizzazione del proiettore
        projection_patterns = [
            # Elenco di pattern predefiniti per la sincronizzazione
            {"type": "grid", "fg_color": "White", "bg_color": "Black", "h_fg_width": 4, "h_bg_width": 20,
             "v_fg_width": 4, "v_bg_width": 20},
            {"type": "horizontal_lines", "fg_color": "White", "bg_color": "Black", "fg_width": 4, "bg_width": 20},
            {"type": "vertical_lines", "fg_color": "White", "bg_color": "Black", "fg_width": 4, "bg_width": 20},
            {"type": "checkerboard", "fg_color": "White", "bg_color": "Black", "h_count": 8, "v_count": 6}
        ]

        # Ciclo principale
        while self.direct_streaming_active:
            # Leggi lista degli stream attivi (copia per sicurezza)
            current_streams = self.active_direct_streams.copy()

            # Gestione heartbeat e messaggi dal client
            try:
                if self.direct_stream_socket:
                    # Controlla se ci sono messaggi in arrivo dal client senza bloccare
                    if self.direct_stream_socket.poll(0):
                        msg_data = self.direct_stream_socket.recv(zmq.NOBLOCK)
                        try:
                            msg = json.loads(msg_data)
                            # Gestisci diversi tipi di messaggi
                            if msg.get("type") == "heartbeat":
                                # Aggiorna stato di attività per tutti gli stream
                                for stream in self.active_direct_streams:
                                    stream["last_activity"] = time.time()
                            elif msg.get("type") == "pattern_request":
                                # Richiesta di cambio pattern dal client
                                pattern_data = msg.get("pattern", {})
                                camera_id = msg.get("camera_id")

                                # Applica il pattern richiesto
                                if pattern_data and self.projector:
                                    self._apply_projector_pattern(pattern_data)

                                    # Aggiorna lo stato di tutti gli stream che usano questo pattern
                                    for stream in self.active_direct_streams:
                                        if not camera_id or stream["camera_id"] == camera_id:
                                            stream["current_pattern"] = pattern_data
                        except json.JSONDecodeError:
                            logger.warning("Ricevuto messaggio non valido dal client di streaming diretto")
            except zmq.error.Again:
                # Nessun messaggio disponibile, continua
                pass
            except Exception as e:
                logger.error(f"Errore nella gestione dei messaggi del client di streaming diretto: {e}")

            # Processa ogni stream
            for stream_info in current_streams:
                camera_id = stream_info["camera_id"]
                fps = stream_info["fps"]
                jpeg_quality = stream_info["jpeg_quality"]
                sync_with_projector = stream_info["sync_with_projector"]
                pattern_interval = stream_info["synchronization_pattern_interval"]

                # Calcola intervallo frame
                interval = 1.0 / fps

                # Controlla se è il momento di inviare un nuovo frame
                current_time = time.time()
                last_time = last_frame_times.get(camera_id, 0)

                if current_time - last_time >= interval:
                    try:
                        # Gestione della sincronizzazione con il proiettore
                        is_sync_frame = False
                        pattern_info = None

                        # Se richiesta sincronizzazione con il proiettore e il proiettore è disponibile
                        if sync_with_projector and self.projector:
                            # Controlla se è il momento di cambiare pattern
                            if current_time >= stream_info["next_pattern_time"]:
                                # Calcola il tempo per il prossimo cambio pattern
                                stream_info["next_pattern_time"] = current_time + (pattern_interval / fps)

                                # Scegli e applica il pattern successivo
                                pattern_index = stream_info["pattern_index"]
                                current_pattern = projection_patterns[pattern_index % len(projection_patterns)]

                                # Applica il pattern
                                self._apply_projector_pattern(current_pattern)

                                # Aggiorna l'indice per il prossimo pattern
                                stream_info["pattern_index"] = (pattern_index + 1) % len(projection_patterns)

                                # Marca questo come frame di sincronizzazione
                                is_sync_frame = True
                                pattern_info = {
                                    "pattern_type": current_pattern["type"],
                                    "pattern_index": pattern_index,
                                    "timestamp": current_time
                                }

                        # Tempo di inizio elaborazione per calcolo performance
                        processing_start = time.time()

                        # Cattura immagine con gestione eccezioni migliorata e logging dettagliato
                        # First check camera manager is available
                        if not self.camera_manager:
                            logger.error(f"Cannot capture image from camera {camera_id} - camera manager is None")
                            
                            # Try to initialize camera manager if needed
                            try:
                                # Import ritardato per evitare importazioni circolari
                                from .hardware.camera import PiCamera2Manager
                                self._camera_manager = PiCamera2Manager()
                                logger.info("Camera manager initialized for direct streaming")
                            except Exception as cm_init_error:
                                logger.error(f"Failed to initialize camera manager: {cm_init_error}")
                                import traceback
                                logger.error(traceback.format_exc())
                                continue
                        
                        # Now try to capture image
                        capture_attempt = 0
                        max_capture_attempts = 3
                        capture_delay = 0.05  # seconds
                        
                        while capture_attempt < max_capture_attempts:
                            try:
                                capture_attempt += 1
                                logger.info(f"Attempting to capture image from camera {camera_id} (attempt {capture_attempt}/{max_capture_attempts})")
                                
                                # Try to capture with timeout handling
                                image = self.camera_manager.capture_image(camera_id)
                                
                                if image is None:
                                    logger.error(f"Error in image capture from camera {camera_id} - returned None (attempt {capture_attempt}/{max_capture_attempts})")
                                    if capture_attempt < max_capture_attempts:
                                        time.sleep(capture_delay)
                                        continue
                                    else:
                                        break
                                        
                                # Image capture successful
                                logger.info(f"Successfully captured image from camera {camera_id}, shape: {image.shape}")
                                break
                                
                            except Exception as e:
                                logger.error(f"Exception in image capture from camera {camera_id}: {e} (attempt {capture_attempt}/{max_capture_attempts})")
                                import traceback
                                logger.error(traceback.format_exc())
                                
                                if capture_attempt < max_capture_attempts:
                                    time.sleep(capture_delay)
                                    continue
                                else:
                                    break
                        
                        # Check if image capture was successful
                        if 'image' not in locals() or image is None:
                            logger.error(f"Failed to capture image from camera {camera_id} after {max_capture_attempts} attempts")
                            continue

                        # Apply GPU preprocessing if enabled
                        if self.enable_preprocessing and self.gpu_preprocessor:
                            try:
                                preprocess_result = self.gpu_preprocessor.preprocess_frame(
                                    image, camera_id, 
                                    metadata={'pattern_type': 'streaming'}
                                )
                                
                                # Use preprocessed image if available
                                if 'compressed_frame' in preprocess_result:
                                    jpeg_data = preprocess_result['compressed_frame']
                                else:
                                    # Fallback to standard encoding
                                    processed_image = preprocess_result.get('frame', image)
                                    jpeg_data = encode_image_to_jpeg(processed_image, jpeg_quality)
                                    
                                # Add preprocessing info to metadata
                                preprocessing_info = {
                                    'preprocessing_applied': preprocess_result.get('preprocessing_applied', []),
                                    'preprocessing_time_ms': preprocess_result.get('preprocessing_time_ms', 0)
                                }
                            except Exception as e:
                                logger.error(f"Preprocessing error for camera {camera_id}: {e}")
                                # Fallback to standard encoding
                                jpeg_data = encode_image_to_jpeg(image, jpeg_quality)
                                preprocessing_info = None
                        else:
                            # Standard encoding without preprocessing
                            try:
                                jpeg_data = encode_image_to_jpeg(image, jpeg_quality)
                                preprocessing_info = None
                            except Exception as e:
                                logger.error(f"Errore nella codifica JPEG per la telecamera {camera_id}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                continue

                        # Prepara metadati
                        height, width = image.shape[:2]
                        metadata = {
                            "width": width,
                            "height": height,
                            "channels": image.shape[2] if len(image.shape) > 2 else 1,
                            "format": "jpeg",
                            "camera_id": camera_id,
                            "stream_id": stream_info["stream_id"],
                            "timestamp": current_time,
                            "frame_number": stream_info["frame_count"],
                            "fps": fps,
                            "is_direct_stream": True
                        }

                        # Add preprocessing info if available
                        if preprocessing_info:
                            metadata["preprocessing_info"] = preprocessing_info
                        
                        # Aggiungi informazioni di sincronizzazione se necessario
                        if is_sync_frame:
                            metadata["is_sync_frame"] = True

                        if pattern_info:
                            metadata["pattern_info"] = pattern_info

                        # Crea messaggio binario con ottimizzazione se disponibile
                        if self.protocol_optimizer and self.enable_protocol_v2:
                            try:
                                binary_message, compression_stats = self.protocol_optimizer.optimize_message(
                                    "direct_frame",
                                    metadata,
                                    jpeg_data,
                                    camera_id
                                )
                                # Add compression stats to metadata for next frame
                                if compression_stats:
                                    logger.debug(f"Compression ratio: {compression_stats.compression_ratio:.2f}x, "
                                               f"time: {compression_stats.compression_time_ms:.1f}ms")
                            except Exception as e:
                                logger.error(f"Protocol optimization failed: {e}")
                                # Fallback to standard serialization
                                binary_message = serialize_binary_message(
                                    "direct_frame",
                                    metadata,
                                    jpeg_data
                                )
                        else:
                            # Standard serialization
                            binary_message = serialize_binary_message(
                                "direct_frame",
                                metadata,
                                jpeg_data
                            )

                        # Pubblica frame attraverso il socket di streaming diretto
                        try:
                            if self.direct_stream_socket:
                                # Log detailed info before sending
                                logger.debug(f"Sending frame for camera {camera_id}, frame #{stream_info['frame_count']}, binary message size: {len(binary_message)} bytes")
                                
                                # Make sure we have a valid socket
                                if not self.direct_stream_socket:
                                    logger.error(f"Socket is None when trying to send frame for camera {camera_id}")
                                    continue
                                    
                                # Try to send the frame with a small timeout
                                try:
                                    self.direct_stream_socket.send(binary_message, zmq.NOBLOCK)
                                    logger.debug(f"Successfully sent frame for camera {camera_id}")
                                except zmq.error.Again:
                                    # Buffer pieno, incrementa contatore frame saltati
                                    stream_info["stats"]["skipped_frames"] += 1
                                    logger.debug(f"Frame saltato per la telecamera {camera_id} (buffer pieno)")
                                    continue
                            else:
                                logger.error(f"Direct stream socket is None for camera {camera_id}")
                                continue
                        except zmq.error.Again:
                            # Buffer pieno, incrementa contatore frame saltati
                            stream_info["stats"]["skipped_frames"] += 1
                            logger.debug(f"Frame saltato per la telecamera {camera_id} (buffer pieno)")
                            continue
                        except Exception as e:
                            logger.error(f"Errore nell'invio del frame diretto per la telecamera {camera_id}: {e}")
                            import traceback
                            logger.error(f"Send error details: {traceback.format_exc()}")
                            
                            # Try to recover the socket if there's a serious error
                            try:
                                logger.warning(f"Attempting to recover direct stream socket")
                                # Close and recreate socket
                                if self.direct_stream_socket:
                                    self.direct_stream_socket.close()
                                self.direct_stream_socket = self.zmq_context.socket(zmq.PAIR)
                                self.direct_stream_socket.setsockopt(zmq.SNDHWM, 2)
                                self.direct_stream_socket.setsockopt(zmq.RCVHWM, 2)
                                self.direct_stream_socket.setsockopt(zmq.LINGER, 0)
                                self.direct_stream_socket.setsockopt(zmq.IMMEDIATE, 1)
                                self.direct_stream_socket.bind(f"tcp://*:{self.direct_stream_port}")
                                logger.info(f"Direct stream socket recovered on port {self.direct_stream_port}")
                            except Exception as recovery_error:
                                logger.error(f"Failed to recover direct stream socket: {recovery_error}")
                                
                            continue

                        # Calcola tempo di elaborazione
                        processing_time = (time.time() - processing_start) * 1000  # ms
                        stream_info["stats"]["processing_time_ms"] = processing_time

                        # Aggiorna contatori
                        stream_info["frame_count"] += 1
                        last_frame_times[camera_id] = current_time
                        stream_info["last_activity"] = current_time

                        # Log dettagliato periodico
                        if stream_info["frame_count"] % 100 == 0:
                            logger.debug(
                                f"Camera {camera_id}: Frame #{stream_info['frame_count']}, "
                                f"Elaborazione: {processing_time:.1f}ms, "
                                f"Framerate effettivo: {1000 / max(processing_time, 1):.1f} FPS"
                            )

                    except Exception as e:
                        logger.error(f"Errore nello streaming diretto della telecamera {camera_id}: {e}")

            # Controlla inattività e rimuovi stream inattivi
            current_time = time.time()
            for i in range(len(self.active_direct_streams) - 1, -1, -1):
                if current_time - self.active_direct_streams[i]["last_activity"] > 5.0:  # 5 secondi di inattività
                    logger.warning(f"Stream diretto inattivo rimosso: {self.active_direct_streams[i]['camera_id']}")
                    self.active_direct_streams.pop(i)

            # Se non ci sono più stream attivi, esci dal loop
            if not self.active_direct_streams:
                logger.info("Nessuno stream diretto attivo, terminazione thread")
                break

            # Attendi un breve intervallo per non sovraccaricare la CPU
            # Usa un intervallo più breve per mantenere la reattività
            time.sleep(0.0005)  # 0.5ms

        logger.info("Thread di streaming diretto terminato")

    # Pattern sequence handling methods
    def _handle_projector_pattern_sequence(self, message: Message) -> Message:
        """Handle PROJECTOR_PATTERN_SEQUENCE messages."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")

        # Check if we're already running a sequence
        if self.pattern_sequence_active:
            # Stop the current sequence
            self.stop_pattern_sequence()
            logger.info("Stopping currently running pattern sequence")

        # Get sequence parameters
        sequence = message.payload.get("sequence", [])
        if not sequence:
            return Message.create_error(message, "No pattern sequence provided")

        interval = message.payload.get("interval", 0.5)  # Default 500ms between patterns
        loop = message.payload.get("loop", False)  # Default: don't loop
        start_immediately = message.payload.get("start", True)  # Default: start immediately
        sync_with_camera = message.payload.get("sync_with_camera", False)  # Default: no camera sync

        # Store sequence parameters
        self.pattern_sequence = sequence
        self.pattern_sequence_interval = interval
        self.pattern_sequence_loop = loop
        self.projector_camera_sync_enabled = sync_with_camera
        self.current_pattern_index = 0

        # Set the projector to test pattern mode
        try:
            from .hardware.projector import OperatingMode
            self.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
        except Exception as e:
            return Message.create_error(message, f"Error setting TestPatternGenerator mode: {e}")

        # Start sequence if requested
        if start_immediately:
            success = self.start_pattern_sequence()
            if not success:
                return Message.create_error(message, "Error starting pattern sequence")

        return Message.create_reply(
            message,
            {
                "success": True,
                "sequence_length": len(sequence),
                "interval": interval,
                "loop": loop,
                "started": start_immediately,
                "sync_with_camera": sync_with_camera
            }
        )

    def _handle_projector_pattern_sequence_step(self, message: Message) -> Message:
        """Handle PROJECTOR_PATTERN_SEQUENCE_STEP messages."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")

        if not self.pattern_sequence:
            return Message.create_error(message, "No pattern sequence defined")

        # Check if automatic sequence is running
        if self.pattern_sequence_active and self.pattern_sequence_thread and self.pattern_sequence_thread.is_alive():
            # If the sequence is running automatically, we can either
            # 1. Pause the sequence and step manually
            # 2. Just step to the next pattern immediately
            # Here we choose option 2
            return Message.create_reply(
                message,
                {
                    "success": False,
                    "error": "Automatic sequence is running. Stop it first to step manually."
                }
            )
        
        # Get any specific step parameters
        steps = message.payload.get("steps", 1)  # Default: advance 1 step
        
        # Move to the next pattern
        self.current_pattern_index = (self.current_pattern_index + steps) % len(self.pattern_sequence)
        
        # Project the pattern
        pattern_data = self.pattern_sequence[self.current_pattern_index]
        
        # Add detailed debug output to see what pattern is being sent
        logger.debug(f"Stepping to pattern index {self.current_pattern_index}")
        logger.debug(f"Pattern data being sent to _apply_projector_pattern: {pattern_data}")
        
        # Apply the pattern
        success = self._apply_projector_pattern(pattern_data)
        
        if not success:
            return Message.create_error(message, "Failed to apply pattern")
            
        # Record timing information
        self.last_pattern_change_time = time.time()
        
        # Emit pattern change event
        self.emit(EventType.PROJECTOR_PATTERN_CHANGED, {
            "pattern": pattern_data,
            "index": self.current_pattern_index,
            "timestamp": self.last_pattern_change_time
        })
        
        # Emit special sync event if camera sync is enabled
        if self.projector_camera_sync_enabled:
            self.emit(EventType.PROJECTOR_CAMERA_SYNC, {
                "pattern": pattern_data,
                "index": self.current_pattern_index,
                "timestamp": self.last_pattern_change_time
            })
            
        return Message.create_reply(
            message,
            {
                "success": True,
                "current_index": self.current_pattern_index,
                "pattern": pattern_data,
                "timestamp": self.last_pattern_change_time
            }
        )

    def _handle_projector_pattern_sequence_stop(self, message: Message) -> Message:
        """Handle PROJECTOR_PATTERN_SEQUENCE_STOP messages."""
        if not self.pattern_sequence_active:
            return Message.create_reply(
                message,
                {
                    "success": True,
                    "was_running": False
                }
            )
            
        # Stop the sequence
        success = self.stop_pattern_sequence()
        
        # Optionally project a specific pattern after stopping
        final_pattern = message.payload.get("final_pattern")
        if success and final_pattern:
            pattern_success = self._apply_projector_pattern(final_pattern)
            if not pattern_success:
                logger.warning(f"Failed to apply final pattern after stopping sequence")
        
        return Message.create_reply(
            message,
            {
                "success": success,
                "was_running": True,
                "final_pattern_applied": bool(final_pattern) if success else False
            }
        )

    def start_pattern_sequence(self) -> bool:
        """Start running a projector pattern sequence.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.pattern_sequence:
            logger.error("Cannot start pattern sequence: no sequence defined")
            return False
            
        if self.pattern_sequence_active:
            logger.warning("Pattern sequence already running")
            return True
            
        # Reset sequence state
        self.pattern_sequence_active = True
        self.current_pattern_index = 0
        
        # Start the sequence thread
        self.pattern_sequence_thread = threading.Thread(
            target=self._pattern_sequence_loop,
            daemon=True
        )
        self.pattern_sequence_thread.start()
        
        logger.info(f"Pattern sequence started: {len(self.pattern_sequence)} patterns, "
                   f"interval: {self.pattern_sequence_interval}s, loop: {self.pattern_sequence_loop}")
        
        # Emit sequence started event
        self.emit(EventType.PROJECTOR_SEQUENCE_STARTED, {
            "sequence_length": len(self.pattern_sequence),
            "interval": self.pattern_sequence_interval,
            "loop": self.pattern_sequence_loop
        })
        
        return True
        
    def stop_pattern_sequence(self) -> bool:
        """Stop a running projector pattern sequence.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.pattern_sequence_active:
            return True
            
        # Set stop flag
        self.pattern_sequence_active = False
        
        # Wait for thread to terminate
        if self.pattern_sequence_thread and self.pattern_sequence_thread.is_alive():
            self.pattern_sequence_thread.join(timeout=2.0)
            
        self.pattern_sequence_thread = None
        
        logger.info("Pattern sequence stopped")
        
        # Emit sequence stopped event
        self.emit(EventType.PROJECTOR_SEQUENCE_STOPPED, {
            "completed": self.current_pattern_index >= len(self.pattern_sequence) - 1
        })
        
        return True
        
    def _pattern_sequence_loop(self):
        """Loop for running projector pattern sequences."""
        logger.info("Pattern sequence thread started")
        
        try:
            sequence_length = len(self.pattern_sequence)
            
            # Project the first pattern
            if sequence_length > 0:
                initial_pattern = self.pattern_sequence[0]
                success = self._apply_projector_pattern(initial_pattern)
                if success:
                    self.last_pattern_change_time = time.time()
                    # Emit pattern changed event
                    self.emit(EventType.PROJECTOR_PATTERN_CHANGED, {
                        "pattern": initial_pattern,
                        "index": 0,
                        "timestamp": self.last_pattern_change_time
                    })
                    
                    # Emit camera sync event if needed
                    if self.projector_camera_sync_enabled:
                        self.emit(EventType.PROJECTOR_CAMERA_SYNC, {
                            "pattern": initial_pattern,
                            "index": 0,
                            "timestamp": self.last_pattern_change_time
                        })
                else:
                    logger.error("Failed to apply initial pattern in sequence")
            
            # Main sequence loop
            while self.pattern_sequence_active:
                # Wait for the next pattern interval
                time.sleep(self.pattern_sequence_interval)
                
                # Check if we should still continue
                if not self.pattern_sequence_active:
                    break
                
                # Move to the next pattern
                self.current_pattern_index += 1
                
                # Check if we've completed the sequence
                if self.current_pattern_index >= sequence_length:
                    if self.pattern_sequence_loop:
                        # Loop back to the beginning
                        self.current_pattern_index = 0
                        # Emit sequence stepped event
                        self.emit(EventType.PROJECTOR_SEQUENCE_STEPPED, {
                            "index": self.current_pattern_index,
                            "loop_completed": True
                        })
                    else:
                        # We're done
                        logger.info("Pattern sequence completed")
                        # Emit sequence completed event
                        self.emit(EventType.PROJECTOR_SEQUENCE_COMPLETED, {
                            "sequence_length": sequence_length
                        })
                        self.pattern_sequence_active = False
                        break
                else:
                    # Emit sequence stepped event
                    self.emit(EventType.PROJECTOR_SEQUENCE_STEPPED, {
                        "index": self.current_pattern_index,
                        "loop_completed": False
                    })
                
                # Project the next pattern
                current_pattern = self.pattern_sequence[self.current_pattern_index]
                success = self._apply_projector_pattern(current_pattern)
                
                if success:
                    self.last_pattern_change_time = time.time()
                    # Emit pattern changed event
                    self.emit(EventType.PROJECTOR_PATTERN_CHANGED, {
                        "pattern": current_pattern,
                        "index": self.current_pattern_index,
                        "timestamp": self.last_pattern_change_time
                    })
                    
                    # Emit camera sync event if needed
                    if self.projector_camera_sync_enabled:
                        self.emit(EventType.PROJECTOR_CAMERA_SYNC, {
                            "pattern": current_pattern,
                            "index": self.current_pattern_index,
                            "timestamp": self.last_pattern_change_time
                        })
                else:
                    logger.error(f"Failed to apply pattern {self.current_pattern_index} in sequence")
                
        except Exception as e:
            logger.error(f"Error in pattern sequence thread: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            # Make sure to mark the sequence as inactive when done
            self.pattern_sequence_active = False
            logger.info("Pattern sequence thread terminated")

    def _apply_projector_pattern(self, pattern_data):
        """
        Applica un pattern al proiettore per la sincronizzazione.

        Args:
            pattern_data: Dizionario con i parametri del pattern

        Returns:
            True se successo, False altrimenti
        """
        if not self.projector:
            logger.error("Proiettore non disponibile per applicare il pattern")
            return False

        try:
            from .hardware.projector import OperatingMode, Color, BorderEnable

            # Prima imposta la modalità test pattern
            self.projector.set_operating_mode(OperatingMode.TestPatternGenerator)

            # Ottieni il tipo di pattern e standardizza i nomi
            # Log raw pattern data for debugging
            logger.info(f"Raw pattern data received: {pattern_data}")
            
            # Extract pattern type with fallbacks
            pattern_type = pattern_data.get("pattern_type", pattern_data.get("type", "solid_field"))
            logger.info(f"Applying pattern: {pattern_type} with parameters: {pattern_data}")

            # Converti stringhe colore in oggetti Color
            def get_color(color_name):
                try:
                    return Color[color_name]
                except (KeyError, TypeError):
                    logger.warning(f"Invalid color name: {color_name}, defaulting to White")
                    return Color.White  # Default a bianco

            # Genera il pattern in base al tipo
            if pattern_type == "solid_field":
                color_name = pattern_data.get("color", "White")
                color = get_color(color_name)
                logger.info(f"Generating solid field, color: {color_name}")
                return self.projector.generate_solid_field(color)

            elif pattern_type == "horizontal_lines":
                # Get color parameters, checking both naming styles
                bg_color_name = pattern_data.get("background_color", pattern_data.get("bg_color", "Black"))
                fg_color_name = pattern_data.get("foreground_color", pattern_data.get("fg_color", "White"))
                bg_color = get_color(bg_color_name)
                fg_color = get_color(fg_color_name)
                
                # Get width parameters, checking both naming styles
                fg_width = pattern_data.get("foreground_width", pattern_data.get("fg_width", 4))
                bg_width = pattern_data.get("background_width", pattern_data.get("bg_width", 4))
                
                # Get phase shift if available
                phase_shift = pattern_data.get("phase_shift", 0)
                
                # Log pattern details
                logger.info(f"Generating horizontal lines: fg={fg_color_name}({fg_width}), bg={bg_color_name}({bg_width}), phase={phase_shift}")
                
                # Generate pattern - note that phase shift is applied at a higher level by shifting the pattern start position
                return self.projector.generate_horizontal_lines(
                    bg_color, fg_color, fg_width, bg_width
                )

            elif pattern_type == "vertical_lines":
                # Get color parameters, checking both naming styles
                bg_color_name = pattern_data.get("background_color", pattern_data.get("bg_color", "Black"))
                fg_color_name = pattern_data.get("foreground_color", pattern_data.get("fg_color", "White"))
                bg_color = get_color(bg_color_name)
                fg_color = get_color(fg_color_name)
                
                # Get width parameters, checking both naming styles
                fg_width = pattern_data.get("foreground_width", pattern_data.get("fg_width", 4))
                bg_width = pattern_data.get("background_width", pattern_data.get("bg_width", 4))
                
                # Get phase shift if available
                phase_shift = pattern_data.get("phase_shift", 0)
                
                # Log pattern details
                logger.info(f"Generating vertical lines: fg={fg_color_name}({fg_width}), bg={bg_color_name}({bg_width}), phase={phase_shift}")
                
                # Generate pattern - note that phase shift is applied at a higher level by shifting the pattern start position
                return self.projector.generate_vertical_lines(
                    bg_color, fg_color, fg_width, bg_width
                )

            elif pattern_type == "grid":
                # Get color parameters, checking both naming styles
                bg_color_name = pattern_data.get("background_color", pattern_data.get("bg_color", "Black"))
                fg_color_name = pattern_data.get("foreground_color", pattern_data.get("fg_color", "White"))
                bg_color = get_color(bg_color_name)
                fg_color = get_color(fg_color_name)
                
                # Get horizontal width parameters, checking both naming styles
                h_fg_width = pattern_data.get("h_foreground_width", pattern_data.get("h_fg_width", 4))
                h_bg_width = pattern_data.get("h_background_width", pattern_data.get("h_bg_width", 16))
                
                # Get vertical width parameters, checking both naming styles
                v_fg_width = pattern_data.get("v_foreground_width", pattern_data.get("v_fg_width", 4))
                v_bg_width = pattern_data.get("v_background_width", pattern_data.get("v_bg_width", 16))
                
                # Log pattern details
                logger.info(f"Generating grid: fg={fg_color_name}, bg={bg_color_name}, h={h_fg_width}/{h_bg_width}, v={v_fg_width}/{v_bg_width}")

                return self.projector.generate_grid(
                    bg_color, fg_color,
                    h_fg_width, h_bg_width,
                    v_fg_width, v_bg_width
                )

            elif pattern_type == "checkerboard":
                # Get color parameters, checking both naming styles
                bg_color_name = pattern_data.get("background_color", pattern_data.get("bg_color", "Black"))
                fg_color_name = pattern_data.get("foreground_color", pattern_data.get("fg_color", "White"))
                bg_color = get_color(bg_color_name)
                fg_color = get_color(fg_color_name)
                
                # Get count parameters, checking both naming styles
                h_count = pattern_data.get("horizontal_count", pattern_data.get("h_count", 8))
                v_count = pattern_data.get("vertical_count", pattern_data.get("v_count", 6))
                
                # Log pattern details
                logger.info(f"Generating checkerboard: fg={fg_color_name}, bg={bg_color_name}, h_count={h_count}, v_count={v_count}")

                return self.projector.generate_checkerboard(
                    bg_color, fg_color, h_count, v_count
                )

            elif pattern_type == "colorbars":
                # Log pattern details
                logger.info("Generating colorbars")
                return self.projector.generate_colorbars()
                
            # Add support for our enhanced pattern types
            elif pattern_type in ["gray_code", "multi_scale", "variable_width", "multi_frequency", "phase_shift"]:
                # These enhanced patterns need to be handled similarly
                logger.info(f"Processing enhanced pattern type: {pattern_type}")
                
                # For now, convert to horizontal/vertical lines based on orientation
                orientation = pattern_data.get("orientation", "horizontal")
                name = pattern_data.get("name", f"{pattern_type}_unnamed")
                
                # Log the pattern being processed
                logger.info(f"Enhanced pattern: {name}, orientation: {orientation}")
                
                # Default to horizontal lines with binary (black/white) pattern
                fg_color = get_color("White")
                bg_color = get_color("Black")
                
                # Use pattern "bit" value to determine line width if available
                bit_value = pattern_data.get("bit", pattern_data.get("width_bits", 0))
                
                # Scale line width based on bit value: 2^bit for Gray code patterns
                try:
                    width_scale = 2 ** bit_value
                    fg_width = max(1, width_scale // 2)  # Ensure at least 1 pixel width
                    bg_width = width_scale
                except (ValueError, TypeError):
                    # Default values if bit calculation fails
                    fg_width = 4
                    bg_width = 8
                
                # Process based on orientation
                if orientation == "horizontal":
                    logger.info(f"Generating horizontal pattern from {pattern_type}: width={fg_width}/{bg_width}")
                    return self.projector.generate_horizontal_lines(bg_color, fg_color, fg_width, bg_width)
                else:  # vertical
                    logger.info(f"Generating vertical pattern from {pattern_type}: width={fg_width}/{bg_width}")
                    return self.projector.generate_vertical_lines(bg_color, fg_color, fg_width, bg_width)
            
            # Add support for raw image data if available in the future
            elif pattern_type == "raw_image":
                logger.info("Processing raw image pattern")
                # Check if projector supports raw images
                if hasattr(self.projector, 'show_raw_image'):
                    image_data = pattern_data.get("image")
                    if image_data:
                        return self.projector.show_raw_image(image_data)
                    else:
                        logger.warning("Raw image pattern missing image data")
                        return False
                else:
                    logger.warning("Projector doesn't support raw images")
                    return False

            else:
                logger.warning(f"Tipo di pattern non supportato: {pattern_type}")
                return False

        except Exception as e:
            logger.error(f"Errore nell'applicazione del pattern: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def stop(self):
        """Stop the server."""
        if not self.running:
            return

        logger.info("Stopping UnLook server...")

        # Stop streaming if active
        self.stop_streaming()

        # Stop direct streaming if active
        self.stop_direct_streaming()
        
        # Stop pattern sequence if active
        if self.pattern_sequence_active:
            self.stop_pattern_sequence()

        # Stop scanning if active
        self.stop_scan()

        # Deactivate projector with improved shutdown sequence
        if self.projector:
            try:
                from .hardware.projector import OperatingMode
                
                # First, show a black pattern as fallback to prevent bright light
                try:
                    from .hardware.projector import Color
                    self.projector.generate_solid_field(Color.Black)
                    logger.info("Set projector to black during shutdown")
                    # Small delay to ensure the command is processed
                    time.sleep(0.1)
                except Exception as pattern_error:
                    logger.warning(f"Error setting black pattern during shutdown: {pattern_error}")
                
                # Then try to set standby mode
                try:
                    standby_result = self.projector.set_operating_mode(OperatingMode.Standby)
                    if standby_result:
                        logger.info("Projector set to standby mode")
                    else:
                        logger.warning("Failed to set projector to standby mode")
                    # Small delay to ensure command is processed
                    time.sleep(0.1)
                except Exception as standby_error:
                    logger.warning(f"Error setting projector to standby mode: {standby_error}")
                
                # Finally close the I2C bus
                self.projector.close()
                # Clear the reference to prevent further use
                self._projector = None
                
            except Exception as e:
                logger.error(f"Error during projector shutdown: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Close camera manager
        if self._camera_manager:
            try:
                self._camera_manager.close()
            except Exception as e:
                logger.error(f"Error closing camera manager: {e}")

        # Stop control thread
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        # Remove registration from discovery
        self.discovery.unregister_scanner()

        # Close ZMQ sockets
        self.control_socket.close()
        self.stream_socket.close()

        # Chiudi il socket di streaming diretto se esiste
        if self.direct_stream_socket:
            self.direct_stream_socket.close()

        self.zmq_context.term()

        logger.info("UnLook server stopped")

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
    
    @property
    def led_controller(self):
        """Lazy-loading del controller LED."""
        if self._led_controller is None:
            try:
                logger.info("Lazy loading LED controller...")
                # Import ritardato per evitare importazioni circolari
                from .hardware.led_controller import led_controller
                logger.info(f"Imported led_controller: {led_controller}")
                self._led_controller = led_controller
                logger.info(f"LED controller assigned: {self._led_controller}")
            except Exception as e:
                logger.error(f"Errore durante l'inizializzazione del controller LED: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._led_controller = None
        return self._led_controller
    
    @property
    def gpu_preprocessor(self):
        """Lazy-loading del GPU preprocessor."""
        if self._gpu_preprocessor is None and self.enable_preprocessing:
            try:
                logger.info("Lazy loading GPU preprocessor...")
                from .hardware.gpu_preprocessing import RaspberryProcessingV2, PreprocessingConfig
                
                # Configure preprocessing based on level
                config = PreprocessingConfig()
                if self.preprocessing_level == "basic":
                    config.pattern_preprocessing = False
                elif self.preprocessing_level == "advanced":
                    config.pattern_preprocessing = True
                    config.compression_level = "adaptive"
                elif self.preprocessing_level == "full":
                    config.pattern_preprocessing = True
                    config.compression_level = "adaptive"
                    config.roi_detection = True
                
                self._gpu_preprocessor = RaspberryProcessingV2(config)
                logger.info(f"GPU preprocessor initialized at level: {self.preprocessing_level}")
            except Exception as e:
                logger.error(f"Error initializing GPU preprocessor: {e}")
                self._gpu_preprocessor = None
        return self._gpu_preprocessor
    
    @property
    def protocol_optimizer(self):
        """Lazy-loading del protocol optimizer."""
        if self._protocol_optimizer is None:
            try:
                logger.info("Lazy loading protocol optimizer...")
                from ..core.protocol_v2 import ProtocolOptimizer
                self._protocol_optimizer = ProtocolOptimizer()
                logger.info("Protocol optimizer initialized")
            except Exception as e:
                logger.error(f"Error initializing protocol optimizer: {e}")
                self._protocol_optimizer = None
        return self._protocol_optimizer

    # Duplicate method removed to fix issue with direct streaming handlers
    # The actual handler initialization is at the beginning of the class

    # Implementazione degli handler dei messaggi...

    def _handle_hello(self, message: Message) -> Message:
        """Handle HELLO messages."""
        client_info = message.payload.get("client_info", {})
        client_id = client_info.get("id", "unknown")
        
        # Handle preprocessing configuration
        preprocessing_config = message.payload.get("preprocessing_config", {})
        if preprocessing_config:
            requested_version = preprocessing_config.get("version", "auto")
            config = preprocessing_config.get("config", {})
            
            logger.info(f"Client {client_id} requested preprocessing version: {requested_version}")
            
            # Apply preprocessing configuration
            self._apply_preprocessing_config(client_id, requested_version, config)

        with self._lock:
            self.clients.add(client_id)

        logger.info(f"New client connected: {client_id}")

        # Send scanner information
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
                },
                "preprocessing": {
                    "current_version": getattr(self, 'current_preprocessing_version', 'auto'),
                    "supported_versions": ["v1_legacy", "v2_enhanced", "auto"]
                }
            }
        )

    def _handle_projector_mode(self, message: Message) -> Message:
        """Handle PROJECTOR_MODE messages."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")

        mode_str = message.payload.get("mode")
        if not mode_str:
            return Message.create_error(message, "Mode not specified")

        try:
            # Convert string to enum
            from .hardware.projector import OperatingMode
            mode = OperatingMode[mode_str]

            # Set mode
            success = self.projector.set_operating_mode(mode)
            if not success:
                return Message.create_error(message, f"Error setting mode {mode_str}")

            logger.info(f"Projector mode set: {mode_str}")
            return Message.create_reply(
                message,
                {"success": True, "mode": mode_str}
            )

        except (KeyError, ValueError) as e:
            return Message.create_error(
                message,
                f"Invalid mode: {mode_str}. Error: {str(e)}"
            )

    def _handle_projector_pattern(self, message: Message) -> Message:
        """Handle PROJECTOR_PATTERN messages."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")

        # First set test pattern mode
        try:
            from .hardware.projector import OperatingMode
            self.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
        except Exception as e:
            return Message.create_error(
                message,
                f"Error setting TestPatternGenerator mode: {e}"
            )

        # Get pattern type
        pattern_type = message.payload.get("pattern_type")
        if not pattern_type:
            return Message.create_error(message, "Pattern type not specified")

        success = False

        try:
            from .hardware.projector import Color, BorderEnable

            if pattern_type == "solid_field":
                # Get color
                color_str = message.payload.get("color", "White")
                color = Color[color_str]

                # Generate pattern
                success = self.projector.generate_solid_field(color)

            elif pattern_type == "horizontal_lines":
                # Get parameters
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                fg_width = message.payload.get("foreground_width", 4)
                bg_width = message.payload.get("background_width", 20)

                # Generate pattern
                success = self.projector.generate_horizontal_lines(
                    bg_color, fg_color, fg_width, bg_width
                )

            elif pattern_type == "vertical_lines":
                # Get parameters
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                fg_width = message.payload.get("foreground_width", 4)
                bg_width = message.payload.get("background_width", 20)

                # Generate pattern
                success = self.projector.generate_vertical_lines(
                    bg_color, fg_color, fg_width, bg_width
                )

            elif pattern_type == "grid":
                # Get parameters
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                h_fg_width = message.payload.get("h_foreground_width", 4)
                h_bg_width = message.payload.get("h_background_width", 20)
                v_fg_width = message.payload.get("v_foreground_width", 4)
                v_bg_width = message.payload.get("v_background_width", 20)

                # Generate pattern
                success = self.projector.generate_grid(
                    bg_color, fg_color,
                    h_fg_width, h_bg_width,
                    v_fg_width, v_bg_width
                )

            elif pattern_type == "checkerboard":
                # Get parameters
                bg_color = Color[message.payload.get("background_color", "Black")]
                fg_color = Color[message.payload.get("foreground_color", "White")]
                h_count = message.payload.get("horizontal_count", 8)
                v_count = message.payload.get("vertical_count", 6)

                # Generate pattern
                success = self.projector.generate_checkerboard(
                    bg_color, fg_color, h_count, v_count
                )

            elif pattern_type == "colorbars":
                # Generate pattern
                success = self.projector.generate_colorbars()

            else:
                return Message.create_error(
                    message,
                    f"Unsupported pattern type: {pattern_type}"
                )

            if success:
                # Record pattern change time
                self.last_pattern_change_time = time.time()
                
                # Emit pattern changed event
                self.emit(EventType.PROJECTOR_PATTERN_CHANGED, {
                    "pattern_type": pattern_type,
                    "timestamp": self.last_pattern_change_time
                })
                
                logger.info(f"Projector pattern generated: {pattern_type}")
                return Message.create_reply(
                    message,
                    {
                        "success": True, 
                        "pattern_type": pattern_type,
                        "timestamp": self.last_pattern_change_time
                    }
                )
            else:
                return Message.create_error(
                    message,
                    f"Error generating pattern {pattern_type}"
                )

        except (KeyError, ValueError, Exception) as e:
            return Message.create_error(
                message,
                f"Error setting pattern: {e}"
            )

    def _handle_camera_list(self, message: Message) -> Message:
        """Handle CAMERA_LIST messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")

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
                f"Error getting camera list: {e}"
            )

    def _handle_camera_config(self, message: Message) -> Message:
        """Handle CAMERA_CONFIG messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "Camera ID not specified")

        config = message.payload.get("config", {})
        logger.info(f"Received camera config request for {camera_id}: {config}")

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
                    f"Error configuring camera {camera_id}"
                )

        except Exception as e:
            return Message.create_error(
                message,
                f"Error configuring camera: {e}"
            )

    def _handle_camera_capture(self, message: Message) -> Message:
        """Handle CAMERA_CAPTURE messages with enhanced support for different formats and configurations."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "Camera ID not specified")

        try:
            # Get optional capture parameters
            jpeg_quality = message.payload.get("jpeg_quality", self.jpeg_quality)
            compression_format = message.payload.get("compression_format", "jpeg").lower()
            resolution = message.payload.get("resolution", None)
            crop_region = message.payload.get("crop_region", None)
            
            # Configure camera before capture if needed
            config_before_capture = {}
            
            # Add resolution if specified
            if resolution:
                config_before_capture["resolution"] = resolution
                
            # Add crop region if specified
            if crop_region:
                config_before_capture["crop_region"] = crop_region
                
            # Apply temporary configuration if needed with better error handling
            if config_before_capture:
                try:
                    logger.info(f"Applying temporary camera config before capture: {config_before_capture}")
                    success = self.camera_manager.configure_camera(camera_id, config_before_capture)
                    if not success:
                        logger.warning(f"Failed to apply configuration to camera {camera_id}")
                except Exception as e:
                    logger.error(f"Error applying temporary configuration to camera {camera_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # Capture image
            image = self.camera_manager.capture_image(camera_id)
            if image is None:
                return Message.create_error(
                    message,
                    f"Error capturing image from camera {camera_id}"
                )

            # Handle different compression formats
            binary_data = None
            format_name = "jpeg"  # Default format name
            
            if compression_format == "png":
                # Encode as PNG
                import cv2
                format_name = "png"
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # Max compression
                _, binary_data = cv2.imencode('.png', image, encode_params)
                binary_data = binary_data.tobytes()
                
            elif compression_format == "raw":
                # Just use raw bytes
                format_name = "raw"
                binary_data = image.tobytes()
                
            else:
                # Default to JPEG
                binary_data = encode_image_to_jpeg(image, jpeg_quality)

            # Prepare metadata
            height, width = image.shape[:2]
            metadata = {
                "width": width,
                "height": height,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "format": format_name,
                "camera_id": camera_id,
                "compression_format": compression_format,
                "timestamp": time.time()
            }
            
            # Add jpeg_quality if applicable
            if compression_format == "jpeg":
                metadata["jpeg_quality"] = jpeg_quality
                
            # Add resolution if applicable
            if resolution:
                metadata["resolution"] = resolution
                
            # Add crop information if applicable  
            if crop_region:
                metadata["crop_region"] = crop_region

            # Create a special message of string type (not enum) for the response
            if self.protocol_optimizer and self.enable_protocol_v2:
                try:
                    binary_response, compression_stats = self.protocol_optimizer.optimize_message(
                        "camera_capture_response",
                        metadata,
                        binary_data,
                        camera_id
                    )
                    if compression_stats:
                        logger.debug(f"Capture response compression: {compression_stats.compression_ratio:.2f}x")
                except Exception as e:
                    logger.error(f"Capture response optimization failed: {e}")
                    # Fallback to standard serialization
                    binary_response = serialize_binary_message(
                        "camera_capture_response",
                        metadata,
                        binary_data
                    )
            else:
                binary_response = serialize_binary_message(
                    "camera_capture_response",
                    metadata,
                    binary_data
                )

            # Send binary response directly
            self.control_socket.send(binary_response)
            return None  # Return None to indicate response was already sent

        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return Message.create_error(
                message,
                f"Error capturing image: {e}"
            )

    def _handle_camera_capture_multi(self, message: Message) -> Message:
        """Handle CAMERA_CAPTURE_MULTI messages for synchronized capture from multiple cameras."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")

        camera_ids = message.payload.get("camera_ids", [])
        if not camera_ids:
            return Message.create_error(message, "Camera ID list not specified")

        try:
            # Open all requested cameras if they're not already open
            for camera_id in camera_ids:
                if not self.camera_manager.open_camera(camera_id):
                    return Message.create_error(
                        message,
                        f"Error opening camera {camera_id}"
                    )

            # Get capture parameters
            jpeg_quality = message.payload.get("jpeg_quality", self.jpeg_quality)
            compression_format = message.payload.get("compression_format", "jpeg").lower()
            resolution = message.payload.get("resolution", None)
            crop_region = message.payload.get("crop_region", None)
            
            # Apply temporary configuration if needed
            if resolution or crop_region:
                config_before_capture = {}
                if resolution:
                    config_before_capture["resolution"] = resolution
                if crop_region:
                    config_before_capture["crop_region"] = crop_region
                    
                # Apply to all cameras with better error handling
                for camera_id in camera_ids:
                    try:
                        logger.info(f"Applying temporary config to camera {camera_id}: {config_before_capture}")
                        success = self.camera_manager.configure_camera(camera_id, config_before_capture)
                        if not success:
                            logger.warning(f"Failed to apply configuration to camera {camera_id}")
                    except Exception as e:
                        logger.error(f"Error applying configuration to camera {camera_id}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

            # Synchronized capture
            cameras = {}

            for camera_id in camera_ids:
                # Capture image
                image = self.camera_manager.capture_image(camera_id)
                if image is None:
                    return Message.create_error(
                        message,
                        f"Error capturing image from camera {camera_id}"
                    )

                # Handle different compression formats
                binary_data = None
                format_name = "jpeg"  # Default format name
                
                if compression_format == "png":
                    # Encode as PNG
                    import cv2
                    format_name = "png"
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # Max compression
                    _, binary_data = cv2.imencode('.png', image, encode_params)
                    binary_data = binary_data.tobytes()
                    
                elif compression_format == "raw":
                    # Just use raw bytes
                    format_name = "raw"
                    binary_data = image.tobytes()
                    
                else:
                    # Default to JPEG
                    binary_data = encode_image_to_jpeg(image, jpeg_quality)

                # Prepare metadata
                height, width = image.shape[:2]
                camera_metadata = {
                    "width": width,
                    "height": height,
                    "channels": image.shape[2] if len(image.shape) > 2 else 1,
                    "format": format_name,
                    "timestamp": time.time(),
                    "compression_format": compression_format
                }
                
                # Add resolution if applicable
                if resolution:
                    camera_metadata["resolution"] = resolution
                    
                # Add crop information if applicable
                if crop_region:
                    camera_metadata["crop_region"] = crop_region
                    
                # Add jpeg_quality for JPEG format
                if compression_format == "jpeg":
                    camera_metadata["jpeg_quality"] = jpeg_quality
                
                cameras[camera_id] = {
                    "jpeg_data": binary_data,  # May not be JPEG, but we keep the field name for backwards compatibility
                    "metadata": camera_metadata
                }

            # Create complete payload
            payload = {
                "cameras": cameras,
                "timestamp": time.time(),
                "num_cameras": len(cameras),
                "format": compression_format,  # Global format for all cameras
                "ulmc_version": "2.0"  # Advanced format with multi-format support
            }
            
            # Add resolution if configured for all cameras
            if resolution:
                payload["resolution"] = resolution
                
            # Add crop_region if configured for all cameras
            if crop_region:
                payload["crop_region"] = crop_region

            # Serialize with ULMC format
            if self.protocol_optimizer and self.enable_protocol_v2:
                try:
                    # For multi-camera, we need to handle binary data specially
                    # Create metadata-only payload for JSON serialization
                    metadata_payload = {
                        "timestamp": payload["timestamp"],
                        "num_cameras": payload["num_cameras"],
                        "format": payload["format"],
                        "ulmc_version": payload["ulmc_version"]
                    }
                    if "resolution" in payload:
                        metadata_payload["resolution"] = payload["resolution"]
                    if "crop_region" in payload:
                        metadata_payload["crop_region"] = payload["crop_region"]
                    
                    # Add camera metadata (without binary data)
                    metadata_payload["cameras"] = {}
                    for cam_id, cam_data in payload["cameras"].items():
                        metadata_payload["cameras"][cam_id] = {
                            "metadata": cam_data["metadata"]
                        }
                    
                    # Serialize metadata as JSON
                    metadata_json = json.dumps(metadata_payload)
                    
                    # Combine metadata and binary data
                    # Format: [4 bytes: metadata length][metadata JSON][camera1 binary][camera2 binary]...
                    combined_parts = [
                        len(metadata_json).to_bytes(4, 'little'),
                        metadata_json.encode('utf-8')
                    ]
                    
                    # Add binary data for each camera in order
                    for cam_id in sorted(payload["cameras"].keys()):
                        binary_data = payload["cameras"][cam_id]["jpeg_data"]
                        combined_parts.extend([
                            len(binary_data).to_bytes(4, 'little'),
                            binary_data
                        ])
                    
                    combined_data = b''.join(combined_parts)
                    
                    binary_response, compression_stats = self.protocol_optimizer.optimize_message(
                        "multi_camera_response",
                        {"format_type": "ulmc", "cameras": list(cameras.keys())},
                        combined_data,
                        "multi_camera"
                    )
                    if compression_stats:
                        logger.debug(f"Multi-camera response compression: {compression_stats.compression_ratio:.2f}x")
                except Exception as e:
                    logger.error(f"Multi-camera response optimization failed: {e}")
                    # Fallback to standard serialization
                    binary_response = serialize_binary_message(
                        "multi_camera_response",
                        payload,
                        format_type="ulmc"
                    )
            else:
                binary_response = serialize_binary_message(
                    "multi_camera_response",
                    payload,
                    format_type="ulmc"
                )

            # Send response
            self.control_socket.send(binary_response)

            # Debug log
            logger.debug(
                f"Sent {len(cameras)} images in ULMC format ({compression_format}), total size: {len(binary_response)} bytes")

            return None  # Response already sent

        except Exception as e:
            logger.error(f"Error in synchronized capture: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return Message.create_error(
                message,
                f"Error in synchronized capture: {e}"
            )

    def _handle_camera_stream_start(self, message: Message) -> Message:
        """Handle CAMERA_STREAM_START messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "Camera ID not specified")

        fps = message.payload.get("fps", DEFAULT_STREAM_FPS)
        jpeg_quality = message.payload.get("jpeg_quality", DEFAULT_JPEG_QUALITY)

        # Check if stream is already active for this camera
        for stream_info in self.active_streams:
            if stream_info["camera_id"] == camera_id:
                logger.warning(f"Stream already active for camera {camera_id}, will be reused")
                # Update configuration
                stream_info["fps"] = fps
                stream_info["jpeg_quality"] = jpeg_quality
                stream_info["last_activity"] = time.time()

                return Message.create_reply(
                    message,
                    {
                        "success": True,
                        "camera_id": camera_id,
                        "stream_port": self.stream_port,
                        "fps": fps,
                        "stream_id": stream_info["stream_id"]
                    }
                )

        # Configure streaming
        stream_id = generate_uuid()
        stream_info = {
            "stream_id": stream_id,
            "camera_id": camera_id,
            "fps": fps,
            "jpeg_quality": jpeg_quality,
            "active": True,
            "start_time": time.time(),
            "last_activity": time.time(),
            "frame_count": 0
        }

        # Add to active streams list
        self.active_streams.append(stream_info)

        # Start streaming thread if not already active
        if not self.streaming_active:
            self.streaming_active = True
            self.streaming_thread = threading.Thread(
                target=self._streaming_loop,
                daemon=True
            )
            self.streaming_thread.start()

        logger.info(f"Streaming started for camera {camera_id} at {fps} FPS (ID: {stream_id})")

        return Message.create_reply(
            message,
            {
                "success": True,
                "camera_id": camera_id,
                "stream_port": self.stream_port,
                "fps": fps,
                "stream_id": stream_id
            }
        )

    # SERVER CORE METHODS

    def start(self):
        """Start the server."""
        if self.running:
            return

        logger.info(f"Starting UnLook server: {self.name} ({self.uuid})")

        # Register service for discovery
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

        # Start control thread
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        # Handle signals for clean termination
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"UnLook server started on port {self.control_port} (streaming: {self.stream_port})")

    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        os._exit(0)

    def _control_loop(self):
        """Main loop for processing control messages."""
        logger.info("Control thread started")

        # Set timeout on socket to allow periodic checks
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second

        while self.running:
            try:
                # Wait for a message with timeout
                try:
                    data = self.control_socket.recv()
                except zmq.error.Again:
                    # Timeout, continue loop
                    continue

                # Process message
                try:
                    message = Message.from_bytes(data)
                    response = self._process_message(message)

                    # If response is None, it means it was already sent
                    if response is not None:
                        self.control_socket.send(response.to_bytes())

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error message
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        payload={"error": str(e)}
                    )
                    self.control_socket.send(error_msg.to_bytes())

            except Exception as e:
                if self.running:  # Avoid error logs during shutdown
                    logger.error(f"Error in control loop: {e}")

        logger.info("Control thread terminated")

    def _process_message(self, message: Message) -> Optional[Message]:
        """
        Process an incoming message.

        Args:
            message: Message to process

        Returns:
            Response message or None if already sent
        """
        logger.debug(f"Received message: {message.msg_type.value}")

        # Check if there's a specific handler
        handler = self.message_handlers.get(message.msg_type)
        if handler:
            try:
                logger.info(f"Processing message: {message.msg_type.value} with handler {handler.__name__}")
                response = handler(message)
                # If response is None, it means it was already sent
                return response
            except Exception as e:
                logger.error(f"Error in handler {message.msg_type.value}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return Message.create_error(
                    message,
                    f"Error processing message: {e}"
                )
        else:
            logger.warning(f"No handler for message: {message.msg_type.value}")
            logger.warning(f"Available handlers: {[h.value for h in self.message_handlers.keys()]}")
            return Message.create_error(
                message,
                f"Unsupported message type: {message.msg_type.value}",
                error_code=400
            )

    def stop_streaming(self):
        """Stop video streaming."""
        if not self.streaming_active:
            return

        # Set flag to terminate thread
        self.streaming_active = False

        # Wait for thread to terminate
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)

        self.streaming_thread = None
        logger.info("Video streaming stopped")

    def _streaming_loop(self):
        """Video streaming loop."""
        logger.info("Streaming thread started")

        # Dictionary to track last frame sent for each camera
        last_frame_times = {}

        while self.streaming_active:
            # Read active streams list (copy for safety)
            current_streams = self.active_streams.copy()

            for stream_info in current_streams:
                camera_id = stream_info["camera_id"]
                fps = stream_info["fps"]
                jpeg_quality = stream_info["jpeg_quality"]

                # Calculate frame interval
                interval = 1.0 / fps

                # Check if it's time to send a new frame
                current_time = time.time()
                last_time = last_frame_times.get(camera_id, 0)

                if current_time - last_time >= interval:
                    try:
                        # Capture image
                        image = self.camera_manager.capture_image(camera_id)
                        if image is None:
                            logger.error(f"Error capturing image from camera {camera_id}")
                            continue

                        # Encode to JPEG
                        jpeg_data = encode_image_to_jpeg(image, jpeg_quality)

                        # Prepare metadata
                        height, width = image.shape[:2]
                        metadata = {
                            "width": width,
                            "height": height,
                            "channels": image.shape[2] if len(image.shape) > 2 else 1,
                            "format": "jpeg",
                            "camera_id": camera_id,
                            "stream_id": stream_info["stream_id"],
                            "timestamp": current_time,
                            "frame_number": stream_info["frame_count"],
                            "fps": fps
                        }

                        # Create binary message with optimization if available
                        if self.protocol_optimizer and self.enable_protocol_v2:
                            try:
                                binary_message, compression_stats = self.protocol_optimizer.optimize_message(
                                    "camera_frame",
                                    metadata,
                                    jpeg_data,
                                    camera_id
                                )
                                if compression_stats:
                                    logger.debug(f"Stream compression ratio: {compression_stats.compression_ratio:.2f}x, "
                                               f"time: {compression_stats.compression_time_ms:.1f}ms")
                            except Exception as e:
                                logger.error(f"Stream protocol optimization failed: {e}")
                                # Fallback to standard serialization
                                binary_message = serialize_binary_message(
                                    "camera_frame",
                                    metadata,
                                    jpeg_data
                                )
                        else:
                            # Standard serialization
                            binary_message = serialize_binary_message(
                                "camera_frame",
                                metadata,
                                jpeg_data
                            )

                        # Publish frame
                        self.stream_socket.send(binary_message)

                        # Update counters
                        stream_info["frame_count"] += 1
                        last_frame_times[camera_id] = current_time
                        stream_info["last_activity"] = current_time

                    except Exception as e:
                        logger.error(f"Error streaming camera {camera_id}: {e}")

            # Check inactivity and remove inactive streams
            current_time = time.time()
            for i in range(len(self.active_streams) - 1, -1, -1):
                if current_time - self.active_streams[i]["last_activity"] > 10.0:  # 10 seconds inactivity
                    logger.warning(f"Inactive stream removed: {self.active_streams[i]['camera_id']}")
                    self.active_streams.pop(i)

            # If there are no more active streams, exit the loop
            if not self.active_streams:
                logger.info("No active streams, terminating thread")
                break

            # Wait a short interval to not overload CPU
            # Use a shorter interval to maintain responsiveness
            time.sleep(0.001)  # 1ms

        logger.info("Streaming thread terminated")

    def stop_scan(self):
        """Stop the scanning process."""
        if not self.scanning:
            return

        # Set flag to terminate thread
        self.scanning = False

        # Wait for thread to terminate
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=2.0)

        self.scan_thread = None

        logger.info("Scan process stopped")

    def _scan_process(self, config: Dict[str, Any]):
        """
        3D scanning process.

        Args:
            config: Scan configuration
        """
        logger.info(f"Scan process started with config: {config}")

        # TODO: Implement 3D scanning algorithm
        time.sleep(5)  # Simulate scanning

        self.scanning = False
        logger.info("Scan process completed")

    def _handle_camera_stream_stop(self, message: Message) -> Message:
        """Handle CAMERA_STREAM_STOP messages."""
        camera_id = message.payload.get("camera_id")
        stream_id = message.payload.get("stream_id")

        # If neither camera_id nor stream_id is specified, stop all streams
        if not camera_id and not stream_id:
            self.stop_streaming()
            logger.info("All streams stopped")
            return Message.create_reply(message, {"success": True})

        # Otherwise look for the specific stream to stop
        for i, stream_info in enumerate(self.active_streams):
            if ((camera_id and stream_info["camera_id"] == camera_id) or
                    (stream_id and stream_info["stream_id"] == stream_id)):
                # Remove stream
                self.active_streams.pop(i)
                logger.info(f"Stream stopped for camera {stream_info['camera_id']} (ID: {stream_info['stream_id']})")
                break

        # If there are no more active streams, stop the thread
        if not self.active_streams:
            self.stop_streaming()

        return Message.create_reply(message, {"success": True})

    def _handle_scan_start(self, message: Message) -> Message:
        """Handle SCAN_START messages."""
        if self.scanning:
            return Message.create_error(message, "Scan already in progress")

        # Configure scan parameters
        scan_config = message.payload.get("config", {})

        # Start scan
        try:
            # Set scan flag
            self.scanning = True

            # Start scan thread
            self.scan_thread = threading.Thread(
                target=self._scan_process,
                args=(scan_config,),
                daemon=True
            )
            self.scan_thread.start()

            logger.info("Scan started")

            return Message.create_reply(
                message,
                {"success": True, "scan_id": generate_uuid()}
            )

        except Exception as e:
            self.scanning = False
            return Message.create_error(
                message,
                f"Error starting scan: {e}"
            )

    def _handle_scan_stop(self, message: Message) -> Message:
        """Handle SCAN_STOP messages."""
        if not self.scanning:
            return Message.create_reply(
                message,
                {"success": True, "already_stopped": True}
            )

        self.stop_scan()

        logger.info("Scan stopped")

        return Message.create_reply(
            message,
            {"success": True}
        )

    def _handle_scan_status(self, message: Message) -> Message:
        """Handle SCAN_STATUS messages."""
        return Message.create_reply(
            message,
            {
                "scanning": self.scanning,
                "progress": 0.0,  # Implement progress calculation
                "status": "running" if self.scanning else "idle"
            }
        )
    
    def _handle_camera_optimize(self, message: Message) -> Message:
        """
        Handle camera optimization request.
        
        Expects payload:
        {
            "camera_id": str,
            "target_brightness": float (optional, default 0.5),
            "target_contrast": float (optional, default 0.3)
        }
        """
        try:
            camera_id = message.payload.get("camera_id")
            if not camera_id:
                return Message.create_error(message, "camera_id is required")
            
            target_brightness = message.payload.get("target_brightness", 0.5)
            target_contrast = message.payload.get("target_contrast", 0.3)
            
            # Perform optimization
            optimized_settings = self.camera_manager.optimize_camera_settings(
                camera_id, target_brightness, target_contrast
            )
            
            if optimized_settings:
                return Message.create_reply(
                    message,
                    {
                        "success": True,
                        "optimized_settings": optimized_settings
                    }
                )
            else:
                return Message.create_error(message, "Failed to optimize camera settings")
                
        except Exception as e:
            logger.error(f"Error optimizing camera: {e}")
            return Message.create_error(message, str(e))
    
    def _handle_camera_auto_focus(self, message: Message) -> Message:
        """
        Handle auto-focus request.
        
        Expects payload:
        {
            "camera_id": str,
            "focus_region": [x, y, width, height] (optional)
        }
        """
        try:
            camera_id = message.payload.get("camera_id")
            if not camera_id:
                return Message.create_error(message, "camera_id is required")
            
            focus_region = message.payload.get("focus_region")
            
            # Perform auto-focus
            success = self.camera_manager.auto_focus(camera_id, focus_region)
            
            return Message.create_reply(
                message,
                {
                    "success": success,
                    "camera_id": camera_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error performing auto-focus: {e}")
            return Message.create_error(message, str(e))
    
    def _handle_camera_test_capture(self, message: Message) -> Message:
        """
        Handle test image capture for optimization.
        
        Expects payload:
        {
            "camera_id": str,
            "test_type": str ("underexposed", "normal", "overexposed")
        }
        """
        try:
            camera_id = message.payload.get("camera_id")
            if not camera_id:
                return Message.create_error(message, "camera_id is required")
            
            test_type = message.payload.get("test_type", "normal")
            
            # Capture test image
            image = self.camera_manager.capture_test_image(camera_id, test_type)
            
            if image is not None:
                # Encode to JPEG for response
                import cv2
                success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if success:
                    return Message.create_reply(
                        message,
                        {
                            "success": True,
                            "camera_id": camera_id,
                            "test_type": test_type,
                            "image": base64.b64encode(encoded_image).decode('utf-8')
                        }
                    )
                else:
                    return Message.create_error(message, "Failed to encode test image")
            else:
                return Message.create_error(message, "Failed to capture test image")
                
        except Exception as e:
            logger.error(f"Error capturing test image: {e}")
            return Message.create_error(message, str(e))
    
    # LED CONTROL HANDLERS
    
    def _handle_led_set_intensity(self, message: Message) -> Message:
        """Handle LED_SET_INTENSITY messages."""
        logger.info(f"LED controller status: {self.led_controller}")
        if not self.led_controller:
            logger.error("LED controller is None or False")
            return Message.create_error(message, "LED controller not available")
            
        # Check for both naming conventions (led1_mA from client, led1 for backwards compatibility)
        led1 = message.payload.get("led1_mA", message.payload.get("led1", 450))
        led2 = message.payload.get("led2_mA", message.payload.get("led2", 450))
        
        result = self.led_controller.set_intensity(led1=led1, led2=led2)
        
        if result["status"] == "success":
            return Message.create_reply(message, result)
        else:
            return Message.create_error(message, result.get("message", "Failed to set LED intensity"))
    
    def _handle_led_on(self, message: Message) -> Message:
        """Handle LED_ON messages."""
        if not self.led_controller:
            return Message.create_error(message, "LED controller not available")
            
        result = self.led_controller.turn_on()
        
        if result["status"] == "success":
            return Message.create_reply(message, result)
        else:
            return Message.create_error(message, result.get("message", "Failed to turn on LED"))
    
    def _handle_led_off(self, message: Message) -> Message:
        """Handle LED_OFF messages."""
        if not self.led_controller:
            return Message.create_error(message, "LED controller not available")
            
        result = self.led_controller.turn_off()
        
        if result["status"] == "success":
            return Message.create_reply(message, result)
        else:
            return Message.create_error(message, result.get("message", "Failed to turn off LED"))
    
    def _handle_led_status(self, message: Message) -> Message:
        """Handle LED_STATUS messages."""
        if not self.led_controller:
            return Message.create_error(message, "LED controller not available")
            
        result = self.led_controller.get_status()
        return Message.create_reply(message, result)
    
    def _handle_led_set_current(self, message: Message) -> Message:
        """Handle LED_SET_CURRENT messages for DLP projector RGB LED current control."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")
        
        # Get current values from payload (0-1023 range)
        red_current = message.payload.get("red_current", 1023)
        green_current = message.payload.get("green_current", 1023)
        blue_current = message.payload.get("blue_current", 1023)
        
        try:
            # Set LED current using the DLP controller
            success = self.projector.set_led_current(red_current, green_current, blue_current)
            
            if success:
                logger.info(f"LED current set - R:{red_current}, G:{green_current}, B:{blue_current}")
                return Message.create_reply(
                    message,
                    {
                        "success": True,
                        "red_current": red_current,
                        "green_current": green_current,
                        "blue_current": blue_current
                    }
                )
            else:
                return Message.create_error(message, "Failed to set LED current")
                
        except Exception as e:
            logger.error(f"Error setting LED current: {e}")
            return Message.create_error(message, str(e))
    
    def _handle_led_get_current(self, message: Message) -> Message:
        """Handle LED_GET_CURRENT messages to get current LED current values."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")
        
        try:
            # Get current LED values
            current_values = self.projector.get_led_current()
            
            if current_values:
                red, green, blue = current_values
                logger.info(f"Current LED values - R:{red}, G:{green}, B:{blue}")
                return Message.create_reply(
                    message,
                    {
                        "success": True,
                        "red_current": red,
                        "green_current": green,
                        "blue_current": blue
                    }
                )
            else:
                return Message.create_error(message, "Failed to get LED current values")
                
        except Exception as e:
            logger.error(f"Error getting LED current: {e}")
            return Message.create_error(message, str(e))
    
    def _handle_led_set_enable(self, message: Message) -> Message:
        """Handle LED_SET_ENABLE messages to enable/disable individual LED channels."""
        if not self.projector:
            return Message.create_error(message, "Projector not available")
        
        # Get enable flags from payload
        red_enable = message.payload.get("red_enable", True)
        green_enable = message.payload.get("green_enable", True)
        blue_enable = message.payload.get("blue_enable", True)
        
        try:
            # Set LED enable state
            success = self.projector.set_led_enable(red_enable, green_enable, blue_enable)
            
            if success:
                logger.info(f"LED enable set - R:{red_enable}, G:{green_enable}, B:{blue_enable}")
                return Message.create_reply(
                    message,
                    {
                        "success": True,
                        "red_enable": red_enable,
                        "green_enable": green_enable,
                        "blue_enable": blue_enable
                    }
                )
            else:
                return Message.create_error(message, "Failed to set LED enable state")
                
        except Exception as e:
            logger.error(f"Error setting LED enable: {e}")
            return Message.create_error(message, str(e))
    
    def _handle_sync_metrics(self, message: Message) -> Message:
        """Handle SYNC_METRICS messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")
        
        # Get sync metrics from camera manager
        sync_metrics = self.camera_manager.get_sync_metrics()
        
        if sync_metrics:
            # Convert to dict
            metrics_dict = {
                'sync_precision_us': sync_metrics.sync_precision_us,
                'frame_consistency': sync_metrics.frame_consistency,
                'average_latency_us': sync_metrics.average_latency_us,
                'jitter_us': sync_metrics.jitter_us,
                'missed_triggers': sync_metrics.missed_triggers,
                'timestamp': sync_metrics.timestamp
            }
            
            # Add protocol optimization stats if available
            if self.protocol_optimizer:
                compression_stats = self.protocol_optimizer.get_compression_stats()
                metrics_dict['compression_stats'] = compression_stats
            
            # Add preprocessing stats if available
            if self.gpu_preprocessor:
                preprocessing_stats = self.gpu_preprocessor.get_performance_stats()
                metrics_dict['preprocessing_stats'] = preprocessing_stats
            
            return Message.create_reply(message, {
                'sync_enabled': self.enable_sync,
                'sync_fps': self.sync_fps,
                'metrics': metrics_dict
            })
        else:
            return Message.create_reply(message, {
                'sync_enabled': self.enable_sync,
                'sync_fps': self.sync_fps,
                'metrics': None,
                'message': 'Synchronization not active or not available'
            })
    
    def _handle_sync_enable(self, message: Message) -> Message:
        """Handle SYNC_ENABLE messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")
        
        enable = message.payload.get('enable', True)
        fps = message.payload.get('fps', self.sync_fps)
        
        try:
            if enable:
                # Enable software sync
                self.camera_manager.enable_software_sync(fps)
                self.enable_sync = True
                self.sync_fps = fps
                return Message.create_reply(message, {
                    'success': True,
                    'sync_enabled': True,
                    'sync_fps': fps,
                    'message': f'Synchronization enabled at {fps} FPS'
                })
            else:
                # Disable sync
                self.enable_sync = False
                return Message.create_reply(message, {
                    'success': True,
                    'sync_enabled': False,
                    'message': 'Synchronization disabled'
                })
        except Exception as e:
            return Message.create_error(message, f"Failed to configure synchronization: {str(e)}")

    def _handle_system_status(self, message: Message) -> Message:
        """Handle SYSTEM_STATUS requests."""
        try:
            # Collect comprehensive system status
            status = {
                'server_info': {
                    'name': self.name,
                    'uuid': self.uuid,
                    'running': self.running,
                    'version': '2.0',
                    'uptime': time.time() - getattr(self, '_start_time', time.time())
                },
                'network': {
                    'control_port': self.control_port,
                    'stream_port': self.stream_port,
                    'direct_stream_port': self.direct_stream_port,
                    'clients_connected': len(self.clients)
                },
                'optimization_settings': {
                    'preprocessing_enabled': self.enable_preprocessing,
                    'preprocessing_level': self.preprocessing_level,
                    'sync_enabled': self.enable_sync,
                    'sync_fps': self.sync_fps,
                    'protocol_v2_enabled': self.enable_protocol_v2
                },
                'hardware_status': {
                    'cameras_available': len(self.camera_manager.cameras) if self.camera_manager else 0,
                    'projector_available': self.projector is not None,
                    'gpu_preprocessing_available': self.gpu_preprocessor is not None,
                    'led_controller_available': self.led_controller is not None
                },
                'performance_metrics': {}
            }
            
            # Add sync metrics if available
            if hasattr(self, 'camera_manager') and self.camera_manager:
                try:
                    if hasattr(self.camera_manager, 'get_sync_metrics'):
                        sync_metrics = self.camera_manager.get_sync_metrics()
                        if sync_metrics and hasattr(sync_metrics, 'to_dict'):
                            status['sync_metrics'] = sync_metrics.to_dict()
                        elif sync_metrics:
                            # Fallback for non-dataclass sync metrics
                            status['sync_metrics'] = {
                                'sync_precision_us': getattr(sync_metrics, 'sync_precision_us', 0),
                                'frame_consistency': getattr(sync_metrics, 'frame_consistency', 0),
                                'average_latency_us': getattr(sync_metrics, 'average_latency_us', 0),
                                'jitter_us': getattr(sync_metrics, 'jitter_us', 0),
                                'missed_triggers': getattr(sync_metrics, 'missed_triggers', 0),
                                'timestamp': getattr(sync_metrics, 'timestamp', time.time())
                            }
                except Exception as e:
                    logger.debug(f"Could not get sync metrics: {e}")
            
            # Add compression stats if protocol optimizer is available
            if self.protocol_optimizer:
                try:
                    compression_stats = self.protocol_optimizer.get_compression_stats()
                    status['compression_stats'] = compression_stats
                    status['performance_metrics']['protocol_v2'] = compression_stats
                except Exception as e:
                    logger.debug(f"Could not get compression stats: {e}")
            
            # Add preprocessing metrics if available
            if self.gpu_preprocessor:
                try:
                    # Would need to implement get_performance_metrics in gpu_preprocessor
                    status['performance_metrics']['preprocessing'] = {
                        'level': self.preprocessing_level,
                        'gpu_acceleration': True
                    }
                except Exception as e:
                    logger.debug(f"Could not get preprocessing metrics: {e}")
            
            return Message.create_reply(message, status)
            
        except Exception as e:
            return Message.create_error(message, f"Failed to get system status: {str(e)}")