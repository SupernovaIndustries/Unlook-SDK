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

    def _handle_hello(self, message: Message) -> Message:
        """Handle HELLO messages."""
        client_info = message.payload.get("client_info", {})
        client_id = client_info.get("id", "unknown")

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
                }
            }
        )

    def _handle_info(self, message: Message) -> Message:
        """Handle INFO messages."""
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
                logger.info(f"Projector pattern generated: {pattern_type}")
                return Message.create_reply(
                    message,
                    {"success": True, "pattern_type": pattern_type}
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
        """Handle CAMERA_CAPTURE messages."""
        if not self.camera_manager:
            return Message.create_error(message, "Camera manager not available")

        camera_id = message.payload.get("camera_id")
        if not camera_id:
            return Message.create_error(message, "Camera ID not specified")

        try:
            # Capture image
            image = self.camera_manager.capture_image(camera_id)
            if image is None:
                return Message.create_error(
                    message,
                    f"Error capturing image from camera {camera_id}"
                )

            # Encode image to JPEG
            jpeg_quality = message.payload.get("jpeg_quality", self.jpeg_quality)
            jpeg_data = encode_image_to_jpeg(image, jpeg_quality)

            # Prepare metadata
            height, width = image.shape[:2]
            metadata = {
                "width": width,
                "height": height,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "format": "jpeg",
                "camera_id": camera_id,
                "timestamp": time.time()
            }

            # Create a special message of string type (not enum) for the response
            binary_response = serialize_binary_message(
                "camera_capture_response",  # Use a string instead of an enum
                metadata,
                jpeg_data
            )

            # Send binary response directly
            self.control_socket.send(binary_response)
            return None  # Return None to indicate response was already sent

        except Exception as e:
            logger.error(f"Error capturing image: {e}")
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

            # Synchronized capture
            jpeg_quality = message.payload.get("jpeg_quality", self.jpeg_quality)
            cameras = {}

            for camera_id in camera_ids:
                # Capture image
                image = self.camera_manager.capture_image(camera_id)
                if image is None:
                    return Message.create_error(
                        message,
                        f"Error capturing image from camera {camera_id}"
                    )

                # Encode image to JPEG
                jpeg_data = encode_image_to_jpeg(image, jpeg_quality)

                # Prepare metadata
                height, width = image.shape[:2]
                cameras[camera_id] = {
                    "jpeg_data": jpeg_data,
                    "metadata": {
                        "width": width,
                        "height": height,
                        "channels": image.shape[2] if len(image.shape) > 2 else 1,
                        "format": "jpeg",
                        "timestamp": time.time()
                    }
                }

            # Create complete payload
            payload = {
                "cameras": cameras,
                "timestamp": time.time(),
                "num_cameras": len(cameras)
            }

            # Serialize with ULMC format
            binary_response = serialize_binary_message(
                "multi_camera_response",
                payload,
                format_type="ulmc"
            )

            # Send response
            self.control_socket.send(binary_response)

            # Debug log
            logger.debug(
                f"Sent {len(cameras)} images in ULMC format, total size: {len(binary_response)} bytes")

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

    def stop(self):
        """Stop the server."""
        if not self.running:
            return

        logger.info("Stopping UnLook server...")

        # Stop streaming if active
        self.stop_streaming()

        # Stop scanning if active
        self.stop_scan()

        # Deactivate projector
        if self.projector:
            try:
                from .hardware.projector import OperatingMode
                self.projector.set_operating_mode(OperatingMode.Standby)
                self.projector.close()
            except Exception as e:
                logger.error(f"Error closing projector: {e}")

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
        self.zmq_context.term()

        logger.info("UnLook server stopped")

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
                response = handler(message)
                # If response is None, it means it was already sent
                return response
            except Exception as e:
                logger.error(f"Error in handler {message.msg_type.value}: {e}")
                return Message.create_error(
                    message,
                    f"Error processing message: {e}"
                )
        else:
            logger.warning(f"No handler for message: {message.msg_type.value}")
            return Message.create_error(
                message,
                f"Unsupported message type: {message.msg_type.value}",
                error_code=400
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

                        # Create binary message
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