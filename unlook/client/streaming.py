"""
Client per lo streaming video dallo scanner UnLook.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq
import numpy as np

from ..common.protocol import MessageType
from ..common.utils import decode_jpeg_to_image, deserialize_binary_message
from ..common.constants import DEFAULT_STREAM_PORT, DEFAULT_STREAM_FPS, DEFAULT_JPEG_QUALITY

logger = logging.getLogger(__name__)


class StreamClient:
    """
    Client per lo streaming video dallo scanner UnLook.
    """

    def __init__(self, parent_client):
        """
        Inizializza il client streaming.

        Args:
            parent_client: Client principale UnlookClient
        """
        self.client = parent_client

        # ZeroMQ
        self.stream_socket = None
        self.stream_poller = zmq.Poller()

        # Stato
        self.streaming = False
        self.stream_thread = None
        self.stream_camera_id = None
        self.stream_fps = DEFAULT_STREAM_FPS
        self.jpeg_quality = DEFAULT_JPEG_QUALITY

        # Callback
        self.frame_callback = None
        self._lock = threading.RLock()

    def start(
            self,
            camera_id: str,
            callback: Callable[[np.ndarray, Dict[str, Any]], None],
            fps: int = DEFAULT_STREAM_FPS,
            jpeg_quality: int = DEFAULT_JPEG_QUALITY
    ) -> bool:
        """
        Avvia lo streaming video.

        Args:
            camera_id: ID della telecamera
            callback: Funzione chiamata per ogni frame
            fps: Frame rate richiesto
            jpeg_quality: Qualità JPEG

        Returns:
            True se lo streaming è avviato con successo, False altrimenti
        """
        if self.streaming:
            self.stop()

        # Invia comando al server
        success, response, _ = self.client.send_message(
            MessageType.CAMERA_STREAM_START,
            {
                "camera_id": camera_id,
                "fps": fps,
                "jpeg_quality": jpeg_quality
            }
        )

        if not success or not response:
            logger.error(f"Errore nell'avvio dello streaming dalla camera {camera_id}")
            return False

        # Ottieni la porta di streaming dal server
        stream_port = response.payload.get("stream_port", DEFAULT_STREAM_PORT)

        try:
            # Crea socket per lo streaming
            self.stream_socket = self.client.zmq_context.socket(zmq.SUB)
            self.stream_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe a tutti i messaggi
            self.stream_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 secondo di timeout

            # Connetti alla porta di streaming
            stream_endpoint = f"tcp://{self.client.scanner.host}:{stream_port}"
            self.stream_socket.connect(stream_endpoint)

            # Registra con il poller
            self.stream_poller.register(self.stream_socket, zmq.POLLIN)

            # Memorizza parametri
            self.stream_camera_id = camera_id
            self.stream_fps = fps
            self.jpeg_quality = jpeg_quality
            self.frame_callback = callback

            # Avvia thread di ricezione
            self.streaming = True
            self.stream_thread = threading.Thread(
                target=self._stream_receiver_loop,
                daemon=True
            )
            self.stream_thread.start()

            logger.info(f"Streaming avviato dalla camera {camera_id} a {fps} FPS")

            # Emetti evento nel client principale
            self.client._emit_event(
                self.client.UnlookClientEvent.STREAM_STARTED,
                camera_id
            )

            return True

        except Exception as e:
            logger.error(f"Errore nell'avvio del client streaming: {e}")
            self._cleanup_streaming()
            return False

    def stop(self):
        """Ferma lo streaming video."""
        if not self.streaming:
            return

        # Invia comando al server
        self.client.send_message(
            MessageType.CAMERA_STREAM_STOP,
            {}
        )

        # Ferma il thread di streaming
        self._cleanup_streaming()

        logger.info("Streaming fermato")

        # Emetti evento nel client principale
        self.client._emit_event(
            self.client.UnlookClientEvent.STREAM_STOPPED,
            self.stream_camera_id
        )

    def _cleanup_streaming(self):
        """Pulisce le risorse di streaming."""
        with self._lock:
            # Imposta flag
            self.streaming = False

            # Chiudi il socket
            if self.stream_socket:
                try:
                    self.stream_poller.unregister(self.stream_socket)
                    self.stream_socket.close()
                    self.stream_socket = None
                except Exception as e:
                    logger.error(f"Errore durante la chiusura del socket di streaming: {e}")

            # Attendi la terminazione del thread
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)

            self.stream_thread = None
            self.stream_camera_id = None
            self.frame_callback = None

    def _stream_receiver_loop(self):
        """Loop di ricezione dello streaming video."""
        logger.info("Thread di ricezione streaming avviato")

        frame_count = 0
        last_stats_time = time.time()

        while self.streaming:
            try:
                # Attendi frame con timeout
                socks = dict(self.stream_poller.poll(1000))  # 1 secondo di timeout

                if not self.streaming:  # Controllo se interrotto durante l'attesa
                    break

                if self.stream_socket in socks and socks[self.stream_socket] == zmq.POLLIN:
                    # Ricevi frame
                    frame_data = self.stream_socket.recv()

                    # Deserializza il messaggio
                    try:
                        msg_type, metadata, jpeg_data = deserialize_binary_message(frame_data)

                        # Controlla se è un frame
                        if msg_type == "camera_frame" and jpeg_data:
                            # Decodifica l'immagine
                            image = decode_jpeg_to_image(jpeg_data)

                            # Chiama il callback
                            if self.frame_callback and image is not None:
                                self.frame_callback(image, metadata)

                            # Incrementa contatore frame
                            frame_count += 1

                            # Stampa statistiche ogni secondo
                            current_time = time.time()
                            if current_time - last_stats_time >= 1.0:
                                fps = frame_count / (current_time - last_stats_time)
                                frame_count = 0
                                last_stats_time = current_time
                                logger.debug(f"Streaming FPS: {fps:.2f}")

                    except Exception as e:
                        logger.error(f"Errore nella decodifica del frame: {e}")

            except zmq.error.Again:
                # Timeout, continua il loop
                pass

            except Exception as e:
                if self.streaming:  # Evita log di errori durante lo shutdown
                    logger.error(f"Errore nel loop di ricezione streaming: {e}")

        logger.info("Thread di ricezione streaming terminato")