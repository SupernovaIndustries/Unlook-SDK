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
from ..client.scanner import UnlookClientEvent

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
        self.streams = {}  # Dizionario di stream attivi {camera_id: stream_info}
        self.stream_thread = None
        self.watchdog_thread = None
        self.default_stream_fps = DEFAULT_STREAM_FPS
        self.default_jpeg_quality = DEFAULT_JPEG_QUALITY

        # Callback
        self.frame_callbacks = {}  # Callbacks per ogni stream
        self._lock = threading.RLock()

        # Metadati
        self.frame_counters = {}  # Contatori di frame per ogni stream
        self.fps_stats = {}  # Statistiche FPS per ogni stream

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
        with self._lock:
            # Verifica se lo stream è già attivo
            if camera_id in self.streams:
                logger.warning(f"Stream già attivo per la camera {camera_id}, riavvio...")
                self.stop_stream(camera_id)

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
                # Crea socket per lo streaming se necessario
                if self.stream_socket is None:
                    self.stream_socket = self.client.zmq_context.socket(zmq.SUB)
                    self.stream_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe a tutti i messaggi
                    self.stream_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 secondo di timeout

                    # Connetti alla porta di streaming
                    stream_endpoint = f"tcp://{self.client.scanner.host}:{stream_port}"
                    self.stream_socket.connect(stream_endpoint)

                    # Registra con il poller
                    self.stream_poller.register(self.stream_socket, zmq.POLLIN)

                # Memorizza parametri
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

                # Avvia thread di ricezione se non è già in esecuzione
                if not self.streaming:
                    self.streaming = True
                    self.stream_thread = threading.Thread(
                        target=self._stream_receiver_loop,
                        daemon=True
                    )
                    self.stream_thread.start()

                    # Avvia anche il watchdog se non è già in esecuzione
                    if not self.watchdog_thread or not self.watchdog_thread.is_alive():
                        self.watchdog_thread = threading.Thread(
                            target=self._stream_watchdog,
                            daemon=True
                        )
                        self.watchdog_thread.start()

                logger.info(f"Streaming avviato dalla camera {camera_id} a {fps} FPS")

                # Emetti evento nel client principale
                self.client._emit_event(
                    UnlookClientEvent.STREAM_STARTED,
                    camera_id
                )

                return True

            except Exception as e:
                logger.error(f"Errore nell'avvio del client streaming: {e}")
                self._cleanup_stream(camera_id)
                return False

    def stop_stream(self, camera_id: str):
        """
        Ferma lo streaming video per una specifica telecamera.

        Args:
            camera_id: ID della telecamera
        """
        with self._lock:
            if camera_id not in self.streams:
                logger.debug(f"Nessuno stream attivo per la camera {camera_id}")
                return

            # Invia comando al server
            self.client.send_message(
                MessageType.CAMERA_STREAM_STOP,
                {"camera_id": camera_id}
            )

            # Rimuovi lo stream
            self._cleanup_stream(camera_id)

            logger.info(f"Streaming fermato per la camera {camera_id}")

            # Emetti evento nel client principale
            self.client._emit_event(
                UnlookClientEvent.STREAM_STOPPED,
                camera_id
            )

            # Se non ci sono più stream attivi, chiudi il socket
            if not self.streams:
                self._cleanup_streaming()

    def stop(self):
        """Ferma tutti gli stream video."""
        with self._lock:
            # Copia le chiavi per evitare modifiche durante l'iterazione
            camera_ids = list(self.streams.keys())

            for camera_id in camera_ids:
                self.stop_stream(camera_id)

            # Pulisci le risorse rimanenti
            self._cleanup_streaming()

    def _cleanup_stream(self, camera_id: str):
        """
        Pulisce le risorse di uno stream specifico.

        Args:
            camera_id: ID della telecamera
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
        """Pulisce tutte le risorse di streaming."""
        with self._lock:
            # Pulisci tutti gli stream attivi
            for camera_id in list(self.streams.keys()):
                self._cleanup_stream(camera_id)

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

            # Attendi la terminazione dei thread
            if hasattr(self, 'stream_thread') and self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)

            if hasattr(self, 'watchdog_thread') and self.watchdog_thread and self.watchdog_thread.is_alive():
                self.watchdog_thread.join(timeout=2.0)

            self.stream_thread = None
            self.watchdog_thread = None

    def _stream_receiver_loop(self):
        """Loop di ricezione dello streaming video."""
        logger.info("Thread di ricezione streaming avviato")

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

                        # Estrai il camera_id dal metadata
                        camera_id = metadata.get("camera_id", "unknown")

                        # Verifica se questo stream è ancora attivo
                        with self._lock:
                            if camera_id not in self.streams or not self.streams[camera_id]["active"]:
                                continue

                            callback = self.frame_callbacks.get(camera_id)
                            if not callback:
                                continue

                        # Controlla se è un frame
                        if (msg_type == "camera_frame" or msg_type == "binary_data") and jpeg_data:
                            # Decodifica l'immagine
                            image = decode_jpeg_to_image(jpeg_data)

                            # Chiama il callback
                            if callback and image is not None:
                                callback(image, metadata)

                                # Aggiorna statistiche
                                with self._lock:
                                    # Incrementa contatore frame
                                    self.frame_counters[camera_id] += 1
                                    self.streams[camera_id]["last_frame_time"] = time.time()
                                    self.fps_stats[camera_id]["frames_since_report"] += 1

                                    # Calcola FPS ogni secondo
                                    current_time = time.time()
                                    if current_time - self.fps_stats[camera_id]["last_report_time"] >= 1.0:
                                        elapsed = current_time - self.fps_stats[camera_id]["last_report_time"]
                                        fps = self.fps_stats[camera_id]["frames_since_report"] / elapsed
                                        self.fps_stats[camera_id]["current_fps"] = fps
                                        self.fps_stats[camera_id]["frames_since_report"] = 0
                                        self.fps_stats[camera_id]["last_report_time"] = current_time
                                        logger.debug(f"Streaming FPS per camera {camera_id}: {fps:.2f}")

                    except Exception as e:
                        logger.error(f"Errore nella decodifica del frame: {e}")

            except zmq.error.Again:
                # Timeout, continua il loop
                pass
            except Exception as e:
                if self.streaming:  # Evita log di errori durante lo shutdown
                    logger.error(f"Errore nel loop di ricezione streaming: {e}")

        logger.info("Thread di ricezione streaming terminato")

    def get_stats(self, camera_id: str = None) -> Dict[str, Any]:
        """
        Ottiene statistiche di streaming.

        Args:
            camera_id: ID della telecamera specifica o None per tutte

        Returns:
            Dizionario con statistiche di streaming
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
                # Statistiche per tutti gli stream
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
        Avvia lo streaming stereo (due telecamere sincronizzate).

        Args:
            callback: Funzione chiamata per ogni coppia di frame
            fps: Frame rate richiesto
            jpeg_quality: Qualità JPEG

        Returns:
            True se lo streaming è avviato con successo, False altrimenti
        """
        # Ottieni la coppia stereo
        left_camera_id, right_camera_id = self.client.camera.get_stereo_pair()

        if not left_camera_id or not right_camera_id:
            logger.error("Impossibile trovare una coppia stereo valida")
            return False

        # Stato condiviso per sincronizzazione
        stereo_state = {
            "frames": {},
            "lock": threading.Lock(),
            "callback": callback,
            "last_sync_time": time.time()
        }

        # Callback per la telecamera sinistra
        def left_camera_callback(image, metadata):
            with stereo_state["lock"]:
                # Salva il frame
                stereo_state["frames"]["left"] = (image, metadata)
                # Verifica se ci sono entrambi i frame
                _check_and_emit_stereo_frame(stereo_state)

        # Callback per la telecamera destra
        def right_camera_callback(image, metadata):
            with stereo_state["lock"]:
                # Salva il frame
                stereo_state["frames"]["right"] = (image, metadata)
                # Verifica se ci sono entrambi i frame
                _check_and_emit_stereo_frame(stereo_state)

        # Funzione di sincronizzazione
        def _check_and_emit_stereo_frame(state):
            if "left" in state["frames"] and "right" in state["frames"]:
                # Abbiamo entrambi i frame, chiamiamo il callback
                left_image, left_metadata = state["frames"]["left"]
                right_image, right_metadata = state["frames"]["right"]

                # Crea metadata combinato
                combined_metadata = {
                    "left": left_metadata,
                    "right": right_metadata,
                    "sync_time": time.time() - state["last_sync_time"],
                    "timestamp": time.time()
                }

                # Chiama il callback
                state["callback"](left_image, right_image, combined_metadata)

                # Reset per il prossimo ciclo
                state["frames"] = {}
                state["last_sync_time"] = time.time()

        # Avvia gli stream individuali
        left_success = self.start(left_camera_id, left_camera_callback, fps, jpeg_quality)
        right_success = self.start(right_camera_id, right_camera_callback, fps, jpeg_quality)

        return left_success and right_success

    def _stream_watchdog(self):
        """
        Thread watchdog che monitora gli stream attivi e rileva possibili problemi.
        """
        while self.streaming:
            try:
                # Verifica inattività
                current_time = time.time()
                with self._lock:
                    for camera_id in list(self.streams.keys()):
                        # Controlla se non riceviamo frame da troppo tempo
                        if camera_id in self.streams and "last_frame_time" in self.streams[camera_id]:
                            last_frame_time = self.streams[camera_id]["last_frame_time"]
                            inactivity_time = current_time - last_frame_time

                            # Se inattivo per più di 5 secondi, logga un avviso
                            if inactivity_time > 5.0:
                                logger.warning(f"Stream {camera_id} inattivo da {inactivity_time:.1f} secondi")

                                # Se inattivo per più di 15 secondi, riavvia lo stream
                                if inactivity_time > 15.0:
                                    logger.error(f"Stream {camera_id} inattivo da troppo tempo, riavvio...")
                                    self._request_stream_restart(camera_id)

            except Exception as e:
                logger.error(f"Errore nel watchdog degli stream: {e}")

            # Attendi 2 secondi prima del prossimo controllo
            time.sleep(2.0)

    def _request_stream_restart(self, camera_id):
        """
        Richiede il riavvio di uno stream.

        Args:
            camera_id: ID della telecamera
        """
        try:
            # Salva le configurazioni correnti
            with self._lock:
                if camera_id not in self.streams:
                    return

                stream_config = self.streams[camera_id].copy()
                callback = self.frame_callbacks.get(camera_id)

                # Ferma lo stream attuale
                self.stop_stream(camera_id)

                # Attendi un po' prima di riavviare
                time.sleep(0.5)

                # Riavvia lo stream con le stesse configurazioni
                self.start(
                    camera_id,
                    callback,
                    stream_config.get("fps", self.default_stream_fps),
                    stream_config.get("jpeg_quality", self.default_jpeg_quality)
                )

        except Exception as e:
            logger.error(f"Errore durante il riavvio dello stream {camera_id}: {e}")