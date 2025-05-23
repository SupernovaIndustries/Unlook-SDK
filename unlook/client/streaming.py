"""
Client for video streaming from the UnLook scanner.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq
import numpy as np

from ..core.protocol import MessageType
from ..core.utils import decode_jpeg_to_image, deserialize_binary_message
from ..core.constants import DEFAULT_STREAM_PORT, DEFAULT_STREAM_FPS, DEFAULT_JPEG_QUALITY
from ..core.events import EventType

logger = logging.getLogger(__name__)


class StreamClient:
    """
    Client for video streaming from the UnLook scanner.
    """

    def __init__(self, parent_client):
        """
        Initialize streaming client.

        Args:
            parent_client: Main UnlookClient
        """
        self.client = parent_client

        # ZeroMQ for standard streaming
        self.stream_socket = None
        self.stream_poller = zmq.Poller()

        # ZeroMQ for direct streaming
        self.direct_socket = None
        self.direct_poller = zmq.Poller()
        self.direct_thread = None
        self.direct_watchdog_thread = None
        self.direct_streaming = False
        self.direct_streams = {}  # Dictionary of active direct streams {camera_id: stream_info}

        # State for standard streaming
        self.streaming = False
        self.streams = {}  # Dictionary of active streams {camera_id: stream_info}
        self.stream_thread = None
        self.watchdog_thread = None
        self.default_stream_fps = DEFAULT_STREAM_FPS
        self.default_jpeg_quality = DEFAULT_JPEG_QUALITY

        # Callbacks
        self.frame_callbacks = {}  # Callbacks for each stream
        self._lock = threading.RLock()

        # Metadata
        self.frame_counters = {}  # Frame counters for each stream
        self.fps_stats = {}  # FPS statistics for each stream

    def start(
            self,
            camera_id: str,
            callback: Callable[[np.ndarray, Dict[str, Any]], None],
            fps: int = DEFAULT_STREAM_FPS,
            jpeg_quality: int = DEFAULT_JPEG_QUALITY
    ) -> bool:
        """
        Start video streaming.

        Args:
            camera_id: Camera ID
            callback: Function called for each frame
            fps: Requested frame rate
            jpeg_quality: JPEG quality

        Returns:
            True if streaming started successfully, False otherwise
        """
        with self._lock:
            # Check if stream is already active
            if camera_id in self.streams:
                logger.warning(f"Stream already active for camera {camera_id}, restarting...")
                self.stop_stream(camera_id)

            # Send command to server
            success, response, _ = self.client.send_message(
                MessageType.CAMERA_STREAM_START,
                {
                    "camera_id": camera_id,
                    "fps": fps,
                    "jpeg_quality": jpeg_quality
                }
            )

            if not success or not response:
                logger.error(f"Error starting streaming from camera {camera_id}")
                return False

            # Get streaming port from server
            stream_port = response.payload.get("stream_port", DEFAULT_STREAM_PORT)

            try:
                # Create socket for streaming if needed
                if self.stream_socket is None:
                    self.stream_socket = self.client.zmq_context.socket(zmq.SUB)
                    self.stream_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
                    self.stream_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

                    # Connect to streaming port
                    stream_endpoint = f"tcp://{self.client.scanner.host}:{stream_port}"
                    self.stream_socket.connect(stream_endpoint)

                    # Register with poller
                    self.stream_poller.register(self.stream_socket, zmq.POLLIN)

                # Store parameters
                self.streams[camera_id] = {
                    "fps": fps,
                    "jpeg_quality": jpeg_quality,
                    "active": True,
                    "start_time": time.time(),
                    "last_frame_time": time.time()
                }

                self.frame_callbacks[camera_id] = callback
                self.frame_counters[camera_id] = 0
                self.fps_stats[camera_id] = {
                    "last_report_time": time.time(),
                    "frames_since_report": 0,
                    "current_fps": 0.0
                }

                # Start reception thread if not already running
                if not self.streaming:
                    self.streaming = True
                    self.stream_thread = threading.Thread(
                        target=self._stream_receiver_loop,
                        daemon=True
                    )
                    self.stream_thread.start()

                    # Also start watchdog if not already running
                    if not self.watchdog_thread or not self.watchdog_thread.is_alive():
                        self.watchdog_thread = threading.Thread(
                            target=self._stream_watchdog,
                            daemon=True
                        )
                        self.watchdog_thread.start()

                logger.info(f"Streaming started from camera {camera_id} at {fps} FPS")

                # Emit event in main client
                self.client.emit(
                    EventType.STREAM_STARTED,
                    camera_id
                )

                return True

            except Exception as e:
                logger.error(f"Error starting streaming client: {e}")
                self._cleanup_stream(camera_id)
                return False

    def stop_stream(self, camera_id: str):
        """
        Stop video streaming for a specific camera.

        Args:
            camera_id: Camera ID
        """
        with self._lock:
            if camera_id not in self.streams:
                logger.debug(f"No active stream for camera {camera_id}")
                return

            # Send command to server
            self.client.send_message(
                MessageType.CAMERA_STREAM_STOP,
                {"camera_id": camera_id}
            )

            # Remove the stream
            self._cleanup_stream(camera_id)

            logger.info(f"Streaming stopped for camera {camera_id}")

            # Emit event in main client
            self.client.emit(
                EventType.STREAM_STOPPED,
                camera_id
            )

            # If there are no more active streams, close the socket
            if not self.streams:
                self._cleanup_streaming()

    def stop(self):
        """Stop all video streams."""
        with self._lock:
            # Copy keys to avoid modifications during iteration
            camera_ids = list(self.streams.keys())

            for camera_id in camera_ids:
                self.stop_stream(camera_id)

            # Clean up remaining resources
            self._cleanup_streaming()

    def _cleanup_stream(self, camera_id: str):
        """
        Clean up resources of a specific stream.

        Args:
            camera_id: Camera ID
        """
        with self._lock:
            if camera_id in self.streams:
                self.streams[camera_id]["active"] = False
                del self.streams[camera_id]

            if camera_id in self.frame_callbacks:
                del self.frame_callbacks[camera_id]

            if camera_id in self.frame_counters:
                del self.frame_counters[camera_id]

            if camera_id in self.fps_stats:
                del self.fps_stats[camera_id]

    def _cleanup_streaming(self):
        """Clean up all streaming resources."""
        with self._lock:
            # Clean up all active streams
            for camera_id in list(self.streams.keys()):
                self._cleanup_stream(camera_id)

            # Set flag
            self.streaming = False

            # Close socket
            if self.stream_socket:
                try:
                    self.stream_poller.unregister(self.stream_socket)
                    self.stream_socket.close()
                    self.stream_socket = None
                except Exception as e:
                    logger.error(f"Error closing streaming socket: {e}")

            # Wait for threads to terminate
            if hasattr(self, 'stream_thread') and self.stream_thread and self.stream_thread.is_alive():
                if self.stream_thread != threading.current_thread():  # Avoid joining current thread
                    self.stream_thread.join(timeout=2.0)
                else:
                    logger.warning("Not joining stream_thread as it is the current thread")

            if hasattr(self, 'watchdog_thread') and self.watchdog_thread and self.watchdog_thread.is_alive():
                if self.watchdog_thread != threading.current_thread():  # Avoid joining current thread
                    self.watchdog_thread.join(timeout=2.0)
                else:
                    logger.warning("Not joining watchdog_thread as it is the current thread")

            self.stream_thread = None
            self.watchdog_thread = None

    def _stream_receiver_loop(self):
        """Video streaming reception loop."""
        logger.info("Streaming reception thread started")

        while self.streaming:
            try:
                # Wait for frame with timeout
                socks = dict(self.stream_poller.poll(1000))  # 1 second timeout

                if not self.streaming:  # Check if interrupted during wait
                    break

                if self.stream_socket in socks and socks[self.stream_socket] == zmq.POLLIN:
                    # Receive frame
                    frame_data = self.stream_socket.recv()

                    # Deserialize message
                    try:
                        msg_type, metadata, jpeg_data = deserialize_binary_message(frame_data)

                        # Extract camera_id from metadata
                        camera_id = metadata.get("camera_id", "unknown")

                        # Check if this stream is still active
                        with self._lock:
                            if camera_id not in self.streams or not self.streams[camera_id]["active"]:
                                continue

                            callback = self.frame_callbacks.get(camera_id)
                            if not callback:
                                continue

                        # Check if it's a frame
                        if (msg_type == "camera_frame" or msg_type == "binary_data") and jpeg_data:
                            # Decode image
                            image = decode_jpeg_to_image(jpeg_data)

                            # Call callback
                            if callback and image is not None:
                                callback(image, metadata)

                                # Update statistics
                                with self._lock:
                                    # Increment frame counter
                                    self.frame_counters[camera_id] += 1
                                    self.streams[camera_id]["last_frame_time"] = time.time()
                                    self.fps_stats[camera_id]["frames_since_report"] += 1

                                    # Calculate FPS every second
                                    current_time = time.time()
                                    if current_time - self.fps_stats[camera_id]["last_report_time"] >= 1.0:
                                        elapsed = current_time - self.fps_stats[camera_id]["last_report_time"]
                                        fps = self.fps_stats[camera_id]["frames_since_report"] / elapsed
                                        self.fps_stats[camera_id]["current_fps"] = fps
                                        self.fps_stats[camera_id]["frames_since_report"] = 0
                                        self.fps_stats[camera_id]["last_report_time"] = current_time
                                        logger.debug(f"Streaming FPS for camera {camera_id}: {fps:.2f}")

                    except Exception as e:
                        logger.error(f"Error decoding frame: {e}")

            except zmq.error.Again:
                # Timeout, continue loop
                pass
            except Exception as e:
                if self.streaming:  # Avoid error logs during shutdown
                    logger.error(f"Error in streaming reception loop: {e}")

        logger.info("Streaming reception thread terminated")

    def get_stats(self, camera_id: str = None) -> Dict[str, Any]:
        """
        Get streaming statistics.

        Args:
            camera_id: Specific camera ID or None for all

        Returns:
            Dictionary with streaming statistics
        """
        with self._lock:
            if camera_id:
                if camera_id not in self.streams:
                    return {}

                stats = {
                    "fps": self.fps_stats.get(camera_id, {}).get("current_fps", 0.0),
                    "frames_received": self.frame_counters.get(camera_id, 0),
                    "stream_active": camera_id in self.streams and self.streams[camera_id]["active"],
                    "stream_time": time.time() - self.streams[camera_id].get("start_time", time.time()),
                    "last_frame_time": time.time() - self.streams[camera_id].get("last_frame_time", time.time())
                }
                return stats
            else:
                # Statistics for all streams
                return {
                    "active_streams": len(self.streams),
                    "total_frames": sum(self.frame_counters.values()),
                    "streams": {
                        cam_id: {
                            "fps": self.fps_stats.get(cam_id, {}).get("current_fps", 0.0),
                            "frames": self.frame_counters.get(cam_id, 0),
                            "active": cam_id in self.streams and self.streams[cam_id]["active"]
                        } for cam_id in self.frame_counters
                    }
                }

    def start_stereo_stream(
            self,
            callback: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], None],
            fps: int = DEFAULT_STREAM_FPS,
            jpeg_quality: int = DEFAULT_JPEG_QUALITY
    ) -> bool:
        """
        Start stereo streaming (two synchronized cameras).

        Args:
            callback: Function called for each pair of frames
            fps: Requested frame rate
            jpeg_quality: JPEG quality

        Returns:
            True if streaming started successfully, False otherwise
        """
        # Get stereo pair
        left_camera_id, right_camera_id = self.client.camera.get_stereo_pair()

        if not left_camera_id or not right_camera_id:
            logger.error("Unable to find a valid stereo pair")
            return False

        # Shared state for synchronization
        stereo_state = {
            "frames": {},
            "lock": threading.Lock(),
            "callback": callback,
            "last_sync_time": time.time()
        }

        # Callback for left camera
        def left_camera_callback(image, metadata):
            with stereo_state["lock"]:
                # Save frame
                stereo_state["frames"]["left"] = (image, metadata)
                # Check if we have both frames
                _check_and_emit_stereo_frame(stereo_state)

        # Callback for right camera
        def right_camera_callback(image, metadata):
            with stereo_state["lock"]:
                # Save frame
                stereo_state["frames"]["right"] = (image, metadata)
                # Check if we have both frames
                _check_and_emit_stereo_frame(stereo_state)

        # Synchronization function
        def _check_and_emit_stereo_frame(state):
            if "left" in state["frames"] and "right" in state["frames"]:
                # We have both frames, call the callback
                left_image, left_metadata = state["frames"]["left"]
                right_image, right_metadata = state["frames"]["right"]

                # Create combined metadata
                combined_metadata = {
                    "left": left_metadata,
                    "right": right_metadata,
                    "sync_time": time.time() - state["last_sync_time"],
                    "timestamp": time.time()
                }

                # Call callback
                state["callback"](left_image, right_image, combined_metadata)

                # Reset for next cycle
                state["frames"] = {}
                state["last_sync_time"] = time.time()

        # Start individual streams
        left_success = self.start(left_camera_id, left_camera_callback, fps, jpeg_quality)
        right_success = self.start(right_camera_id, right_camera_callback, fps, jpeg_quality)

        return left_success and right_success

    def _stream_watchdog(self):
        """
        Watchdog thread that monitors active streams and detects potential issues.
        """
        while self.streaming:
            try:
                # Check inactivity
                current_time = time.time()
                with self._lock:
                    for camera_id in list(self.streams.keys()):
                        # Check if we haven't received frames for too long
                        if camera_id in self.streams and "last_frame_time" in self.streams[camera_id]:
                            last_frame_time = self.streams[camera_id]["last_frame_time"]
                            inactivity_time = current_time - last_frame_time

                            # If inactive for more than 5 seconds, log a warning
                            if inactivity_time > 5.0:
                                logger.warning(f"Stream {camera_id} inactive for {inactivity_time:.1f} seconds")

                                # If inactive for more than 15 seconds, restart the stream
                                if inactivity_time > 15.0:
                                    logger.error(f"Stream {camera_id} inactive for too long, restarting...")
                                    self._request_stream_restart(camera_id)

            except Exception as e:
                logger.error(f"Error in stream watchdog: {e}")

            # Wait 2 seconds before next check
            time.sleep(2.0)

    def _request_stream_restart(self, camera_id):
        """
        Request stream restart.

        Args:
            camera_id: Camera ID
        """
        try:
            # Save current configurations
            with self._lock:
                if camera_id not in self.streams:
                    return

                stream_config = self.streams[camera_id].copy()
                callback = self.frame_callbacks.get(camera_id)

                # Stop current stream
                self.stop_stream(camera_id)

                # Wait a bit before restarting
                time.sleep(0.5)

                # Restart stream with same configurations
                self.start(
                    camera_id,
                    callback,
                    stream_config.get("fps", self.default_stream_fps),
                    stream_config.get("jpeg_quality", self.default_jpeg_quality)
                )

        except Exception as e:
            logger.error(f"Error restarting stream {camera_id}: {e}")

    def start_direct_stream(
            self,
            camera_id: str,
            callback: Callable[[np.ndarray, Dict[str, Any]], None],
            fps: int = 60,  # Default più alto per lo streaming diretto
            jpeg_quality: int = DEFAULT_JPEG_QUALITY,  # Qualità leggermente superiore
            sync_with_projector: bool = False,
            synchronization_pattern_interval: int = 5,  # Ogni quanti frame cambiare pattern
            low_latency: bool = True
    ) -> bool:
        """
        Avvia uno streaming video diretto ottimizzato per bassa latenza.

        Args:
            camera_id: ID della telecamera
            callback: Funzione chiamata per ogni frame
            fps: Frame rate richiesto
            jpeg_quality: Qualità JPEG (0-100)
            sync_with_projector: Sincronizza lo streaming con i pattern del proiettore
            synchronization_pattern_interval: Intervallo di cambio pattern per la sincronizzazione
            low_latency: Attiva modalità a bassa latenza (buffer ridotti)

        Returns:
            True se lo streaming è stato avviato con successo, False altrimenti
        """
        with self._lock:
            # Se è già attivo uno stream diretto per questa telecamera, interrompilo
            if hasattr(self, 'direct_streams') and camera_id in self.direct_streams:
                logger.warning(f"Stream diretto già attivo per la telecamera {camera_id}, riavvio...")
                self.stop_direct_stream(camera_id)

            # Inizializza la struttura dei direct_streams se non esiste
            if not hasattr(self, 'direct_streams'):
                self.direct_streams = {}
                self.direct_socket = None
                self.direct_poller = zmq.Poller()
                self.direct_thread = None
                self.direct_streaming = False

            # Invia comando al server
            success, response, _ = self.client.send_message(
                MessageType.CAMERA_DIRECT_STREAM_START,
                {
                    "camera_id": camera_id,
                    "fps": fps,
                    "jpeg_quality": jpeg_quality,
                    "sync_with_projector": sync_with_projector,
                    "synchronization_pattern_interval": synchronization_pattern_interval,
                    "low_latency": low_latency
                }
            )

            if not success or not response:
                logger.error(f"Errore nell'avvio dello streaming diretto dalla telecamera {camera_id}")
                return False

            # Ottieni la porta e le informazioni di streaming dal server
            direct_port = response.payload.get("direct_port")
            stream_id = response.payload.get("stream_id")

            try:
                # Crea socket per lo streaming diretto se necessario
                if self.direct_socket is None:
                    # Utilizziamo un socket PAIR per comunicazione bidirezionale sincrona
                    self.direct_socket = self.client.zmq_context.socket(zmq.PAIR)
                    self.direct_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 secondo timeout

                    # Settaggi per bassa latenza
                    if low_latency:
                        self.direct_socket.setsockopt(zmq.RCVHWM, 2)  # Buffer di ricezione ridotto
                        self.direct_socket.setsockopt(zmq.LINGER, 0)  # Non aspettare alla chiusura
                        self.direct_socket.setsockopt(zmq.IMMEDIATE, 1)  # Non accodare messaggi

                    # Connettiti alla porta di streaming diretto
                    direct_endpoint = f"tcp://{self.client.scanner.host}:{direct_port}"
                    self.direct_socket.connect(direct_endpoint)

                    # Registra con il poller
                    self.direct_poller.register(self.direct_socket, zmq.POLLIN)

                # Memorizza i parametri
                self.direct_streams[camera_id] = {
                    "fps": fps,
                    "jpeg_quality": jpeg_quality,
                    "active": True,
                    "start_time": time.time(),
                    "last_frame_time": time.time(),
                    "sync_with_projector": sync_with_projector,
                    "synchronization_pattern_interval": synchronization_pattern_interval,
                    "low_latency": low_latency,
                    "stream_id": stream_id,
                    "frame_counter": 0,
                    "pattern_counter": 0,
                    "buffer": [],  # Buffer per i frame
                    "stats": {
                        "dropped_frames": 0,
                        "latency_ms": 0
                    }
                }

                self.frame_callbacks[camera_id] = callback
                self.frame_counters[camera_id] = 0
                self.fps_stats[camera_id] = {
                    "last_report_time": time.time(),
                    "frames_since_report": 0,
                    "current_fps": 0.0
                }

                # Avvia thread di ricezione se non è già in esecuzione
                try:
                    if not self.direct_streaming:
                        # Make sure we have the direct_thread attribute
                        if not hasattr(self, 'direct_thread'):
                            self.direct_thread = None
                            
                        # Make sure we have the direct_watchdog_thread attribute
                        if not hasattr(self, 'direct_watchdog_thread'):
                            self.direct_watchdog_thread = None
                            
                        self.direct_streaming = True
                        self.direct_thread = threading.Thread(
                            target=self._direct_stream_receiver_loop,
                            daemon=True
                        )
                        self.direct_thread.start()
    
                        # Avvia anche il watchdog per il direct streaming
                        if self.direct_watchdog_thread is None or not self.direct_watchdog_thread.is_alive():
                            self.direct_watchdog_thread = threading.Thread(
                                target=self._direct_stream_watchdog,
                                daemon=True
                            )
                            self.direct_watchdog_thread.start()
                except Exception as e:
                    logger.error(f"Error starting direct streaming threads: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                logger.info(f"Streaming diretto avviato dalla telecamera {camera_id} a {fps} FPS")

                # Emetti evento nel client principale
                self.client.emit(
                    EventType.DIRECT_STREAM_STARTED,
                    camera_id
                )

                return True

            except Exception as e:
                logger.error(f"Errore nell'avvio dello streaming diretto: {e}")
                self._cleanup_direct_stream(camera_id)
                return False

    def stop_direct_stream(self, camera_id: str):
        """
        Interrompe lo streaming video diretto per una specifica telecamera.

        Args:
            camera_id: ID della telecamera
        """
        try:
            if not hasattr(self, 'direct_streams'):
                logger.debug(f"Nessuno stream diretto attivo per la telecamera {camera_id}")
                return
    
            with self._lock:
                if camera_id not in self.direct_streams:
                    logger.debug(f"Nessuno stream diretto attivo per la telecamera {camera_id}")
                    return
    
                # Invia comando al server
                self.client.send_message(
                    MessageType.CAMERA_DIRECT_STREAM_STOP,
                    {"camera_id": camera_id}
                )
    
                # Rimuovi lo stream
                self._cleanup_direct_stream(camera_id)
    
                logger.info(f"Streaming diretto interrotto per la telecamera {camera_id}")
    
                # Emetti evento nel client principale
                self.client.emit(
                    EventType.DIRECT_STREAM_STOPPED,
                    camera_id
                )
    
                # Se non ci sono più stream attivi, chiudi il socket
                if not self.direct_streams:
                    self._cleanup_direct_streaming()
                    
        except Exception as e:
            logger.error(f"Error stopping direct stream for camera {camera_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _cleanup_direct_stream(self, camera_id: str):
        """
        Pulisce le risorse di uno specifico stream diretto.

        Args:
            camera_id: ID della telecamera
        """
        with self._lock:
            if hasattr(self, 'direct_streams') and camera_id in self.direct_streams:
                self.direct_streams[camera_id]["active"] = False
                del self.direct_streams[camera_id]

    def _cleanup_direct_streaming(self):
        """Pulisce tutte le risorse dello streaming diretto."""
        with self._lock:
            # Pulisci tutti gli stream attivi
            if hasattr(self, 'direct_streams'):
                for camera_id in list(self.direct_streams.keys()):
                    self._cleanup_direct_stream(camera_id)

            # Imposta il flag
            self.direct_streaming = False

            # Chiudi il socket
            if hasattr(self, 'direct_socket') and self.direct_socket:
                try:
                    self.direct_poller.unregister(self.direct_socket)
                    self.direct_socket.close()
                    self.direct_socket = None
                except Exception as e:
                    logger.error(f"Errore nella chiusura del socket di streaming diretto: {e}")

            # Attendi che i thread si terminino
            if hasattr(self, 'direct_thread') and self.direct_thread and self.direct_thread.is_alive():
                if self.direct_thread != threading.current_thread():  # Avoid joining current thread
                    self.direct_thread.join(timeout=2.0)
                else:
                    logger.warning("Not joining direct_thread as it is the current thread")

            if hasattr(self, 'direct_watchdog_thread') and self.direct_watchdog_thread and self.direct_watchdog_thread.is_alive():
                if self.direct_watchdog_thread != threading.current_thread():  # Avoid joining current thread
                    self.direct_watchdog_thread.join(timeout=2.0)
                else:
                    logger.warning("Not joining direct_watchdog_thread as it is the current thread")

            self.direct_thread = None
            self.direct_watchdog_thread = None

    def _direct_stream_receiver_loop(self):
        """Loop di ricezione dello streaming diretto."""
        logger.info("Thread di ricezione streaming diretto avviato")
        
        # Log debug info about direct streams
        with self._lock:
            if hasattr(self, 'direct_streams'):
                for cam_id, stream in self.direct_streams.items():
                    logger.info(f"Direct stream active for camera {cam_id}: {stream.get('active', False)}, "
                               f"FPS: {stream.get('fps', 0)}, Quality: {stream.get('jpeg_quality', 0)}")
            else:
                logger.warning("No direct_streams attribute found")

        last_heartbeat_time = time.time()
        heartbeat_interval = 1.0  # Invia heartbeat ogni secondo

        while self.direct_streaming:
            try:
                # Invia heartbeat per mantenere la connessione attiva
                current_time = time.time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    try:
                        if self.direct_socket:
                            # Invia un messaggio di heartbeat
                            self.direct_socket.send_json({
                                "type": "heartbeat",
                                "timestamp": current_time
                            }, zmq.NOBLOCK)
                        last_heartbeat_time = current_time
                    except zmq.error.Again:
                        # Non bloccare se il buffer è pieno
                        pass
                    except Exception as e:
                        logger.error(f"Errore nell'invio del heartbeat: {e}")

                # Attendi frame con timeout
                socks = dict(self.direct_poller.poll(100))  # Timeout più breve per reattività

                if not self.direct_streaming:  # Controlla se interrotto durante l'attesa
                    break

                if self.direct_socket in socks and socks[self.direct_socket] == zmq.POLLIN:
                    # Ricevi frame
                    frame_data = self.direct_socket.recv()

                    # Deserializza messaggio
                    try:
                        msg_type, metadata, jpeg_data = deserialize_binary_message(frame_data)

                        # Estrai camera_id dai metadati
                        camera_id = metadata.get("camera_id", "unknown")
                        timestamp = metadata.get("timestamp", time.time())
                        pattern_info = metadata.get("pattern_info", None)
                        is_sync_frame = metadata.get("is_sync_frame", False)

                        # Controlla se questo stream è ancora attivo
                        with self._lock:
                            if not hasattr(self, 'direct_streams') or camera_id not in self.direct_streams or not \
                            self.direct_streams[camera_id]["active"]:
                                continue

                            callback = self.frame_callbacks.get(camera_id)
                            if not callback:
                                continue

                            stream_info = self.direct_streams[camera_id]

                        # Controlla se è un frame
                        if (msg_type == "camera_frame" or msg_type == "direct_frame") and jpeg_data:
                            # Decodifica immagine
                            image = decode_jpeg_to_image(jpeg_data)

                            # Calcola latenza
                            latency = (time.time() - timestamp) * 1000  # in millisecondi

                            # Aggiungi informazioni al metadata
                            metadata["latency_ms"] = latency
                            metadata["is_direct_stream"] = True

                            if pattern_info:
                                metadata["pattern_info"] = pattern_info

                            if is_sync_frame:
                                metadata["is_sync_frame"] = True

                            # Chiama callback
                            if callback and image is not None:
                                callback(image, metadata)

                                # Aggiorna statistiche
                                with self._lock:
                                    if hasattr(self, 'direct_streams') and camera_id in self.direct_streams:
                                        # Incrementa contatore frame
                                        self.direct_streams[camera_id]["frame_counter"] += 1
                                        self.direct_streams[camera_id]["last_frame_time"] = time.time()

                                        # Aggiorna statistiche di latenza
                                        self.direct_streams[camera_id]["stats"]["latency_ms"] = latency

                                        # Aggiorna statistiche FPS
                                        self.fps_stats[camera_id]["frames_since_report"] += 1

                                        # Calcola FPS ogni secondo
                                        current_time = time.time()
                                        if current_time - self.fps_stats[camera_id]["last_report_time"] >= 1.0:
                                            elapsed = current_time - self.fps_stats[camera_id]["last_report_time"]
                                            fps = self.fps_stats[camera_id]["frames_since_report"] / elapsed
                                            self.fps_stats[camera_id]["current_fps"] = fps
                                            self.fps_stats[camera_id]["frames_since_report"] = 0
                                            self.fps_stats[camera_id]["last_report_time"] = current_time

                                            # Log dettagliato per lo streaming diretto
                                            logger.debug(
                                                f"Streaming diretto FPS: {fps:.2f}, Latenza: {latency:.1f}ms per camera {camera_id}")

                    except Exception as e:
                        logger.error(f"Errore nella decodifica del frame: {e}")
                        import traceback
                        logger.error(f"Stack trace: {traceback.format_exc()}")

            except zmq.error.Again:
                # Timeout, continua il loop
                pass
            except Exception as e:
                if self.direct_streaming:  # Evita log di errore durante lo shutdown
                    logger.error(f"Errore nel loop di ricezione streaming diretto: {e}")

        logger.info("Thread di ricezione streaming diretto terminato")

    def _direct_stream_watchdog(self):
        """
        Thread watchdog che monitora gli stream diretti attivi e rileva potenziali problemi.
        Include meccanismi di recupero automatico.
        """
        logger.info("Direct stream watchdog thread started")
        
        restart_attempts = {}  # Track restart attempts by camera_id
        max_restart_attempts = 3  # Maximum number of restart attempts
        restart_cooldown = 30.0  # Cooldown period between multiple restarts (seconds)
        
        try:
            # Initialize direct_streaming if not present
            if not hasattr(self, 'direct_streaming'):
                self.direct_streaming = False
                logger.warning("direct_streaming attribute wasn't initialized")
                
            # Initialize direct_streams if not present
            if not hasattr(self, 'direct_streams'):
                self.direct_streams = {}
                logger.warning("direct_streams attribute wasn't initialized")
                
            while hasattr(self, 'direct_streaming') and self.direct_streaming:
                try:
                    # Controlla inattività
                    current_time = time.time()
                    with self._lock:
                        if hasattr(self, 'direct_streams'):
                            for camera_id in list(self.direct_streams.keys()):
                                # Controlla se non abbiamo ricevuto frame per troppo tempo
                                if (camera_id in self.direct_streams and 
                                    "last_frame_time" in self.direct_streams[camera_id]):
                                    
                                    last_frame_time = self.direct_streams[camera_id]["last_frame_time"]
                                    inactivity_time = current_time - last_frame_time
    
                                    # Se inattivo per più di 2 secondi, log di avviso (più sensibile per lo streaming diretto)
                                    if inactivity_time > 2.0:
                                        logger.warning(
                                            f"Stream diretto {camera_id} inattivo per {inactivity_time:.1f} secondi")
    
                                        # Se inattivo per più di 5 secondi, considera il riavvio
                                        if inactivity_time > 5.0:
                                            # Check if we've restarted too many times
                                            if camera_id not in restart_attempts:
                                                restart_attempts[camera_id] = {
                                                    "count": 0,
                                                    "last_attempt": 0
                                                }
                                                
                                            # Check if we're in the cooldown period
                                            time_since_last_restart = current_time - restart_attempts[camera_id]["last_attempt"]
                                            if restart_attempts[camera_id]["count"] >= max_restart_attempts and time_since_last_restart < restart_cooldown:
                                                logger.warning(
                                                    f"Not restarting stream for camera {camera_id} - hit max restart attempts ({max_restart_attempts}). "
                                                    f"Cooldown: {restart_cooldown - time_since_last_restart:.1f}s remaining")
                                                continue
                                                
                                            # If cooldown period has passed, reset counter
                                            if time_since_last_restart > restart_cooldown:
                                                restart_attempts[camera_id]["count"] = 0
                                                
                                            # Attempt restart
                                            logger.error(
                                                f"Stream diretto {camera_id} inattivo troppo a lungo, riavvio... (tentativo #{restart_attempts[camera_id]['count'] + 1})")
                                            
                                            restart_attempts[camera_id]["count"] += 1
                                            restart_attempts[camera_id]["last_attempt"] = current_time
                                            
                                            self._request_direct_stream_restart(camera_id)
    
                except Exception as e:
                    logger.error(f"Errore nel watchdog dello stream diretto: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
                # Attendi 1 secondo prima del prossimo controllo (più frequente per streaming diretto)
                time.sleep(1.0)
                
            logger.info("Direct stream watchdog thread exiting normally")
            
        except Exception as e:
            logger.error(f"Fatal error in direct stream watchdog: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _request_direct_stream_restart(self, camera_id):
        """
        Richiede il riavvio dello stream diretto.

        Args:
            camera_id: ID della telecamera
        """
        try:
            # Salva le configurazioni correnti
            with self._lock:
                if not hasattr(self, 'direct_streams') or camera_id not in self.direct_streams:
                    return

                stream_config = self.direct_streams[camera_id].copy()
                callback = self.frame_callbacks.get(camera_id)

                # Interrompi lo stream corrente
                self.stop_direct_stream(camera_id)

                # Attendi un attimo prima di riavviare
                time.sleep(0.5)

                # Riavvia lo stream con le stesse configurazioni (in un thread separato per evitare errori join)
                threading.Thread(
                    target=self._delayed_restart_stream,
                    args=(camera_id, callback, stream_config),
                    daemon=True
                ).start()

        except Exception as e:
            logger.error(f"Errore nel riavvio dello stream diretto {camera_id}: {e}")
            
    def _delayed_restart_stream(self, camera_id, callback, stream_config):
        """Helper method to restart stream in a separate thread"""
        try:
            # Ensure all required attributes are properly initialized
            with self._lock:
                # Pre-cleanup: make sure no existing threads are still running
                if hasattr(self, 'direct_thread') and self.direct_thread and self.direct_thread.is_alive():
                    if self.direct_thread != threading.current_thread():
                        logger.info(f"Waiting for existing direct_thread to finish before restart")
                        try:
                            self.direct_thread.join(timeout=1.0)
                        except Exception as e:
                            logger.warning(f"Error joining direct_thread: {e}")
                    else:
                        logger.info(f"Cannot join direct_thread as it is the current thread")
                
                if hasattr(self, 'direct_watchdog_thread') and self.direct_watchdog_thread and self.direct_watchdog_thread.is_alive():
                    if self.direct_watchdog_thread != threading.current_thread():
                        logger.info(f"Waiting for existing direct_watchdog_thread to finish before restart")
                        try:
                            self.direct_watchdog_thread.join(timeout=1.0)
                        except Exception as e:
                            logger.warning(f"Error joining direct_watchdog_thread: {e}")
                    else:
                        logger.info(f"Cannot join direct_watchdog_thread as it is the current thread")
                
                # Reset direct streaming state
                self.direct_streaming = False
                self.direct_thread = None
                self.direct_watchdog_thread = None
                
                # Ensure socket is closed
                if self.direct_socket:
                    try:
                        self.direct_poller.unregister(self.direct_socket)
                    except Exception as e:
                        logger.warning(f"Error unregistering direct_socket: {e}")
                    
                    try:
                        self.direct_socket.close()
                    except Exception as e:
                        logger.warning(f"Error closing direct_socket: {e}")
                    
                    self.direct_socket = None
            
            # Wait for cleanup to complete
            time.sleep(1.0)
            
            # Now restart the stream
            logger.info(f"Restarting direct stream for camera {camera_id}")
            success = self.start_direct_stream(
                camera_id,
                callback,
                stream_config.get("fps", 60),
                stream_config.get("jpeg_quality", DEFAULT_JPEG_QUALITY),
                stream_config.get("sync_with_projector", False),
                stream_config.get("synchronization_pattern_interval", 5),
                stream_config.get("low_latency", True)
            )
            
            if not success:
                logger.error(f"Failed to restart direct stream for camera {camera_id}")
            else:
                logger.info(f"Successfully restarted direct stream for camera {camera_id}")
        except Exception as e:
            logger.error(f"Errore nel riavvio ritardato dello stream {camera_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def start_direct_stereo_stream(
            self,
            callback: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], None],
            fps: int = 60,
            jpeg_quality: int = DEFAULT_JPEG_QUALITY,
            sync_with_projector: bool = False,
            synchronization_pattern_interval: int = 5,
            low_latency: bool = True
    ) -> bool:
        """
        Avvia uno streaming stereo diretto a bassa latenza.

        Args:
            callback: Funzione chiamata per ogni coppia di frame
            fps: Frame rate richiesto
            jpeg_quality: Qualità JPEG
            sync_with_projector: Sincronizza lo streaming con i pattern del proiettore
            synchronization_pattern_interval: Intervallo di cambio pattern
            low_latency: Attiva modalità a bassa latenza

        Returns:
            True se lo streaming è stato avviato con successo, False altrimenti
        """
        # Ottieni la coppia stereo
        left_camera_id, right_camera_id = self.client.camera.get_stereo_pair()

        if not left_camera_id or not right_camera_id:
            logger.error("Impossibile trovare una coppia stereo valida")
            return False

        # Stato condiviso per la sincronizzazione
        stereo_state = {
            "frames": {},
            "lock": threading.Lock(),
            "callback": callback,
            "last_sync_time": time.time(),
            "timestamps": {},
            "pattern_info": None
        }

        # Callback per la telecamera sinistra
        def left_camera_callback(image, metadata):
            with stereo_state["lock"]:
                # Salva frame e timestamp
                stereo_state["frames"]["left"] = (image, metadata)
                stereo_state["timestamps"]["left"] = metadata.get("timestamp", time.time())

                # Salva le informazioni sul pattern se presenti
                if "pattern_info" in metadata:
                    stereo_state["pattern_info"] = metadata["pattern_info"]

                # Controlla se abbiamo entrambi i frame
                _check_and_emit_stereo_frame(stereo_state)

        # Callback per la telecamera destra
        def right_camera_callback(image, metadata):
            with stereo_state["lock"]:
                # Salva frame e timestamp
                stereo_state["frames"]["right"] = (image, metadata)
                stereo_state["timestamps"]["right"] = metadata.get("timestamp", time.time())

                # Salva le informazioni sul pattern se presenti
                if "pattern_info" in metadata:
                    stereo_state["pattern_info"] = metadata["pattern_info"]

                # Controlla se abbiamo entrambi i frame
                _check_and_emit_stereo_frame(stereo_state)

        # Funzione di sincronizzazione
        def _check_and_emit_stereo_frame(state):
            if "left" in state["frames"] and "right" in state["frames"]:
                # Abbiamo entrambi i frame, chiama la callback
                left_image, left_metadata = state["frames"]["left"]
                right_image, right_metadata = state["frames"]["right"]

                # Calcola il tempo di sincronizzazione (differenza tra i timestamp delle due immagini)
                left_ts = state["timestamps"].get("left", 0)
                right_ts = state["timestamps"].get("right", 0)
                sync_diff_ms = abs(left_ts - right_ts) * 1000  # in millisecondi

                # Crea metadati combinati
                combined_metadata = {
                    "left": left_metadata,
                    "right": right_metadata,
                    "sync_diff_ms": sync_diff_ms,
                    "sync_time": time.time() - state["last_sync_time"],
                    "timestamp": time.time(),
                    "is_direct_stream": True
                }

                # Aggiungi informazioni sul pattern se disponibili
                if state["pattern_info"]:
                    combined_metadata["pattern_info"] = state["pattern_info"]

                # Chiama callback
                state["callback"](left_image, right_image, combined_metadata)

                # Reset per il prossimo ciclo
                state["frames"] = {}
                state["timestamps"] = {}
                state["last_sync_time"] = time.time()

        # Avvia i singoli stream
        left_success = self.start_direct_stream(
            left_camera_id,
            left_camera_callback,
            fps,
            jpeg_quality,
            sync_with_projector,
            synchronization_pattern_interval,
            low_latency
        )

        right_success = self.start_direct_stream(
            right_camera_id,
            right_camera_callback,
            fps,
            jpeg_quality,
            sync_with_projector,
            synchronization_pattern_interval,
            low_latency
        )

        return left_success and right_success

    def get_direct_stream_stats(self, camera_id: str = None) -> Dict[str, Any]:
        """
        Ottiene statistiche per lo streaming diretto.

        Args:
            camera_id: ID della specifica telecamera o None per tutte

        Returns:
            Dizionario con statistiche dello streaming
        """
        with self._lock:
            if not hasattr(self, 'direct_streams'):
                return {}

            if camera_id:
                if camera_id not in self.direct_streams:
                    return {}

                stats = {
                    "fps": self.fps_stats.get(camera_id, {}).get("current_fps", 0.0),
                    "frames_received": self.direct_streams[camera_id].get("frame_counter", 0),
                    "stream_active": camera_id in self.direct_streams and self.direct_streams[camera_id]["active"],
                    "stream_time": time.time() - self.direct_streams[camera_id].get("start_time", time.time()),
                    "last_frame_time": time.time() - self.direct_streams[camera_id].get("last_frame_time",
                                                                                        time.time()),
                    "latency_ms": self.direct_streams[camera_id].get("stats", {}).get("latency_ms", 0),
                    "sync_with_projector": self.direct_streams[camera_id].get("sync_with_projector", False),
                    "dropped_frames": self.direct_streams[camera_id].get("stats", {}).get("dropped_frames", 0)
                }
                return stats
            else:
                # Statistiche per tutti gli stream
                return {
                    "active_direct_streams": len(self.direct_streams),
                    "total_frames": sum(stream.get("frame_counter", 0) for stream in self.direct_streams.values()),
                    "avg_latency_ms": sum(
                        stream.get("stats", {}).get("latency_ms", 0) for stream in self.direct_streams.values()) /
                                      max(len(self.direct_streams), 1),
                    "streams": {
                        cam_id: {
                            "fps": self.fps_stats.get(cam_id, {}).get("current_fps", 0.0),
                            "frames": stream.get("frame_counter", 0),
                            "active": stream.get("active", False),
                            "latency_ms": stream.get("stats", {}).get("latency_ms", 0),
                            "sync_with_projector": stream.get("sync_with_projector", False)
                        } for cam_id, stream in self.direct_streams.items()
                    }
                }