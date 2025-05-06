"""
Modulo per la gestione delle telecamere tramite PiCamera2.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any

import numpy as np

# Importa picamera2 solo se disponibile (su Raspberry Pi)
try:
    from picamera2 import Picamera2

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logger = logging.getLogger(__name__)


class PiCamera2Manager:
    """
    Gestore delle telecamere PiCamera2.
    Fornisce un'interfaccia unificata per la gestione di più telecamere.
    """

    def __init__(self):
        """Inizializza il manager telecamere."""
        if not PICAMERA2_AVAILABLE:
            logger.warning("PiCamera2 non disponibile. Funzionalità limitate.")

        self.cameras = {}  # Dict[camera_id, camera_info]
        self.active_cameras = {}  # Dict[camera_id, Picamera2]
        self._lock = threading.RLock()

        # Rileva le telecamere disponibili
        self._discover_cameras()

    def _discover_cameras(self):
        """Rileva le telecamere disponibili."""
        if not PICAMERA2_AVAILABLE:
            logger.warning("PiCamera2 non disponibile, impossibile rilevare telecamere.")
            return

        with self._lock:
            try:
                # Ottieni la lista delle telecamere disponibili
                num_cameras = len(Picamera2.global_camera_info())
                logger.info(f"Trovate {num_cameras} telecamere")

                for i in range(num_cameras):
                    try:
                        # Crea una Picamera2 temporanea per ottenere informazioni
                        cam = Picamera2(i)

                        # Ottieni capacità e informazioni
                        capabilities = cam.camera_properties

                        # Estrai informazioni utili
                        camera_info = {
                            "name": f"Camera {i}",
                            "index": i,
                            "model": capabilities.get("Model", "Unknown"),
                            "resolution": capabilities.get("MaxResolution", [1920, 1080]),
                            "fps": 30,  # Default FPS
                            "capabilities": ["preview", "still", "video"],
                            "raw_capabilities": capabilities
                        }

                        # Aggiungi alla lista delle telecamere
                        camera_id = f"picamera2_{i}"
                        self.cameras[camera_id] = camera_info
                        logger.info(f"Telecamera rilevata: {camera_id} - {camera_info['name']}")

                        # Chiudi la telecamera
                        cam.close()

                    except Exception as e:
                        logger.error(f"Errore durante il rilevamento della telecamera {i}: {e}")

            except Exception as e:
                logger.error(f"Errore durante il rilevamento delle telecamere: {e}")

    def get_cameras(self) -> Dict[str, Dict[str, Any]]:
        """
        Ottiene la lista delle telecamere disponibili.

        Returns:
            Dizionario di telecamere {camera_id: camera_info}
        """
        with self._lock:
            return self.cameras.copy()

    def open_camera(self, camera_id: str) -> bool:
        """
        Apre una telecamera.

        Args:
            camera_id: ID della telecamera

        Returns:
            True se l'apertura ha successo, False altrimenti
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 non disponibile, impossibile aprire la telecamera.")
            return False

        with self._lock:
            # Verifica se la telecamera esiste
            if camera_id not in self.cameras:
                logger.error(f"Telecamera {camera_id} non trovata")
                return False

            # Verifica se la telecamera è già aperta
            if camera_id in self.active_cameras:
                logger.debug(f"Telecamera {camera_id} già aperta")
                return True

            try:
                # Estrai l'indice della telecamera
                camera_info = self.cameras[camera_id]
                camera_index = camera_info["index"]

                # Apri la telecamera
                camera = Picamera2(camera_index)

                # Configura la telecamera
                config = camera.create_still_configuration()
                camera.configure(config)

                # Avvia la telecamera
                camera.start()

                # Aggiungi alla lista delle telecamere attive
                self.active_cameras[camera_id] = camera

                logger.info(f"Telecamera {camera_id} aperta con successo")
                return True

            except Exception as e:
                logger.error(f"Errore durante l'apertura della telecamera {camera_id}: {e}")
                return False

    def close_camera(self, camera_id: str) -> bool:
        """
        Chiude una telecamera.

        Args:
            camera_id: ID della telecamera

        Returns:
            True se la chiusura ha successo, False altrimenti
        """
        with self._lock:
            # Verifica se la telecamera è aperta
            if camera_id not in self.active_cameras:
                logger.debug(f"Telecamera {camera_id} non è aperta")
                return True

            try:
                # Chiudi la telecamera
                camera = self.active_cameras[camera_id]
                camera.stop()
                camera.close()

                # Rimuovi dalla lista delle telecamere attive
                del self.active_cameras[camera_id]

                logger.info(f"Telecamera {camera_id} chiusa con successo")
                return True

            except Exception as e:
                logger.error(f"Errore durante la chiusura della telecamera {camera_id}: {e}")
                return False

    def configure_camera(self, camera_id: str, config: Dict[str, Any]) -> bool:
        """
        Configura una telecamera.

        Args:
            camera_id: ID della telecamera
            config: Configurazione della telecamera

        Returns:
            True se la configurazione ha successo, False altrimenti
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 non disponibile, impossibile configurare la telecamera.")
            return False

        with self._lock:
            # Verifica se la telecamera esiste
            if camera_id not in self.cameras:
                logger.error(f"Telecamera {camera_id} non trovata")
                return False

            # Apri la telecamera se non è già aperta
            if camera_id not in self.active_cameras:
                if not self.open_camera(camera_id):
                    return False

            camera = self.active_cameras[camera_id]

            try:
                # Configura la telecamera in base ai parametri

                # Risoluzione
                if "resolution" in config:
                    width, height = config["resolution"]

                    # Ferma la telecamera, riconfigura e riavvia
                    camera.stop()

                    if config.get("mode") == "video":
                        camera_config = camera.create_video_configuration(
                            main={"size": (width, height)}
                        )
                    elif config.get("mode") == "preview":
                        camera_config = camera.create_preview_configuration(
                            main={"size": (width, height)}
                        )
                    else:  # default: still
                        camera_config = camera.create_still_configuration(
                            main={"size": (width, height)}
                        )

                    camera.configure(camera_config)
                    camera.start()

                # Esposizione
                if "exposure" in config:
                    camera.set_controls({"ExposureTime": config["exposure"]})

                # Guadagno
                if "gain" in config:
                    camera.set_controls({"AnalogueGain": config["gain"]})

                # Bilanciamento del bianco
                if "awb" in config:
                    if config["awb"] == "auto":
                        camera.set_controls({"AwbEnable": True})
                    else:
                        camera.set_controls({"AwbEnable": False})
                        if "awb_gains" in config:
                            camera.set_controls({
                                "ColourGains": config["awb_gains"]
                            })

                # Aggiorna le informazioni sulla telecamera
                self.cameras[camera_id].update({
                    "configured_resolution": config.get("resolution",
                                                        self.cameras[camera_id]["resolution"]),
                    "configured_fps": config.get("fps",
                                                 self.cameras[camera_id]["fps"])
                })

                logger.info(f"Telecamera {camera_id} configurata con successo")
                return True

            except Exception as e:
                logger.error(f"Errore durante la configurazione della telecamera {camera_id}: {e}")
                return False

    def capture_image(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Cattura un'immagine da una telecamera.

        Args:
            camera_id: ID della telecamera

        Returns:
            Immagine come array numpy, None in caso di errore
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 non disponibile, impossibile catturare l'immagine.")
            return None

        with self._lock:
            # Verifica se la telecamera esiste
            if camera_id not in self.cameras:
                logger.error(f"Telecamera {camera_id} non trovata")
                return None

            # Apri la telecamera se non è già aperta
            if camera_id not in self.active_cameras:
                if not self.open_camera(camera_id):
                    return None

            camera = self.active_cameras[camera_id]

            try:
                # Cattura l'immagine
                image = camera.capture_array()

                # Converti in RGB se necessario
                if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]  # Rimuovi canale alpha

                return image

            except Exception as e:
                logger.error(f"Errore durante la cattura dell'immagine dalla telecamera {camera_id}: {e}")
                return None

    def close(self):
        """Chiude tutte le telecamere attive."""
        with self._lock:
            for camera_id in list(self.active_cameras.keys()):
                self.close_camera(camera_id)

            self.active_cameras.clear()
            logger.info("Tutte le telecamere chiuse")

    def configure_camera(self, camera_id: str, config: Dict[str, Any]) -> bool:
        """
        Configura una telecamera con supporto per opzioni avanzate.

        Args:
            camera_id: ID della telecamera
            config: Configurazione della telecamera, può includere:
                - resolution: [width, height]
                - fps: Frame rate
                - mode: Modalità di acquisizione ("video", "preview", "still")
                - exposure: Tempo di esposizione in microsecondi
                - gain: Guadagno analogico
                - awb: Bilanciamento del bianco ("auto" o "manual")
                - awb_gains: Guadagni per il bilanciamento del bianco [red_gain, blue_gain]
                - color_mode: Modalità di colore ("rgb", "bgr", "grayscale")
                - roi: Region of interest [x, y, width, height]
                - hflip: Specchia orizzontalmente (True/False)
                - vflip: Specchia verticalmente (True/False)

        Returns:
            True se la configurazione ha successo, False altrimenti
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 non disponibile, impossibile configurare la telecamera.")
            return False

        with self._lock:
            # Verifica se la telecamera esiste
            if camera_id not in self.cameras:
                logger.error(f"Telecamera {camera_id} non trovata")
                return False

            # Apri la telecamera se non è già aperta
            if camera_id not in self.active_cameras:
                if not self.open_camera(camera_id):
                    return False

            camera = self.active_cameras[camera_id]

            try:
                # Configurazione richiesta per il cambiamento di risoluzione o modalità
                needs_reconfigure = "resolution" in config or "mode" in config or "color_mode" in config

                # Se è richiesta riconfigurazione, dobbiamo fermare la telecamera
                if needs_reconfigure:
                    camera.stop()

                    # Ottieni la modalità di acquisizione
                    mode = config.get("mode", "still")  # Default: still

                    # Ottieni la risoluzione
                    width, height = config.get("resolution", self.cameras[camera_id].get("resolution", [1920, 1080]))

                    # Ottieni la modalità di colore
                    color_mode = config.get("color_mode", "rgb")

                    # Crea la configurazione appropriata
                    if mode == "video":
                        # Configurazione video con supporto per diversi formati di colore
                        if color_mode == "grayscale":
                            camera_config = camera.create_video_configuration(
                                main={"size": (width, height), "format": "YUV420"}
                            )
                        elif color_mode == "bgr":
                            camera_config = camera.create_video_configuration(
                                main={"size": (width, height), "format": "BGR888"}
                            )
                        else:  # default: rgb
                            camera_config = camera.create_video_configuration(
                                main={"size": (width, height), "format": "RGB888"}
                            )
                    elif mode == "preview":
                        # Configurazione preview con supporto per diversi formati di colore
                        if color_mode == "grayscale":
                            camera_config = camera.create_preview_configuration(
                                main={"size": (width, height), "format": "YUV420"}
                            )
                        elif color_mode == "bgr":
                            camera_config = camera.create_preview_configuration(
                                main={"size": (width, height), "format": "BGR888"}
                            )
                        else:  # default: rgb
                            camera_config = camera.create_preview_configuration(
                                main={"size": (width, height), "format": "RGB888"}
                            )
                    else:  # default: still
                        # Configurazione still con supporto per diversi formati di colore
                        if color_mode == "grayscale":
                            camera_config = camera.create_still_configuration(
                                main={"size": (width, height), "format": "YUV420"}
                            )
                        elif color_mode == "bgr":
                            camera_config = camera.create_still_configuration(
                                main={"size": (width, height), "format": "BGR888"}
                            )
                        else:  # default: rgb
                            camera_config = camera.create_still_configuration(
                                main={"size": (width, height), "format": "RGB888"}
                            )

                    # Applica la configurazione
                    camera.configure(camera_config)

                # Costruisci un dizionario di controlli da impostare
                controls = {}

                # Esposizione
                if "exposure" in config:
                    controls["ExposureTime"] = config["exposure"]
                    if "exposure_mode" in config and config["exposure_mode"] == "manual":
                        controls["AeEnable"] = False
                    else:
                        controls["AeEnable"] = True

                # Guadagno
                if "gain" in config:
                    controls["AnalogueGain"] = config["gain"]

                # Bilanciamento del bianco
                if "awb" in config:
                    if config["awb"] == "auto":
                        controls["AwbEnable"] = True
                    else:
                        controls["AwbEnable"] = False
                        if "awb_gains" in config:
                            controls["ColourGains"] = tuple(config["awb_gains"])

                # Region of interest
                if "roi" in config:
                    x, y, w, h = config["roi"]
                    controls["ScalerCrop"] = (x, y, w, h)

                # Flipping
                if "hflip" in config:
                    controls["HFlip"] = config["hflip"]
                if "vflip" in config:
                    controls["VFlip"] = config["vflip"]

                # Imposta i controlli
                if controls:
                    camera.set_controls(controls)

                # Avvia la telecamera se era stata fermata
                if needs_reconfigure:
                    camera.start()

                # Aggiorna le informazioni sulla telecamera
                updated_info = self.cameras[camera_id].copy()

                if "resolution" in config:
                    updated_info["configured_resolution"] = config["resolution"]

                if "fps" in config:
                    updated_info["configured_fps"] = config["fps"]

                if "color_mode" in config:
                    updated_info["color_mode"] = config["color_mode"]

                # Aggiorna la cache
                self.cameras[camera_id] = updated_info

                logger.info(f"Telecamera {camera_id} configurata con successo")
                return True

            except Exception as e:
                logger.error(f"Errore durante la configurazione della telecamera {camera_id}: {e}")
                return False

    def capture_image(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Cattura un'immagine da una telecamera.

        Args:
            camera_id: ID della telecamera

        Returns:
            Immagine come array numpy, None in caso di errore
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 non disponibile, impossibile catturare l'immagine.")
            return None

        with self._lock:
            # Verifica se la telecamera esiste
            if camera_id not in self.cameras:
                logger.error(f"Telecamera {camera_id} non trovata")
                return None

            # Apri la telecamera se non è già aperta
            if camera_id not in self.active_cameras:
                if not self.open_camera(camera_id):
                    return None

            camera = self.active_cameras[camera_id]

            try:
                # Cattura l'immagine
                image = camera.capture_array()

                # Verifica il formato dell'immagine
                color_mode = self.cameras[camera_id].get("color_mode", "rgb")

                # Converti l'immagine nel formato corretto se necessario
                if len(image.shape) == 3:
                    if color_mode == "grayscale" and image.shape[2] > 1:
                        # Converti in grayscale se richiesto
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    elif color_mode == "rgb" and image.shape[2] == 4:  # RGBA
                        # Rimuovi canale alpha
                        image = image[:, :, :3]
                        # Converti da BGR a RGB se necessario
                        if image.shape[2] == 3 and np.array_equal(image[0, 0], image[0, 0, ::-1]):
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif color_mode == "bgr" and image.shape[2] == 3:
                        # Assicurati che sia in formato BGR
                        if not np.array_equal(image[0, 0], image[0, 0, ::-1]):
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                return image

            except Exception as e:
                logger.error(f"Errore durante la cattura dell'immagine dalla telecamera {camera_id}: {e}")
                return None