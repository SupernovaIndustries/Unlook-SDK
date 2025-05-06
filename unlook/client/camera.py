"""
Client per la gestione delle telecamere dello scanner UnLook.
"""

import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from ..common.protocol import MessageType
from ..common.utils import decode_jpeg_to_image

logger = logging.getLogger(__name__)


class CameraClient:
    """
    Client per la gestione delle telecamere dello scanner UnLook.
    """

    def __init__(self, parent_client):
        """
        Inizializza il client telecamere.

        Args:
            parent_client: Client principale UnlookClient
        """
        self.client = parent_client
        self.cameras = {}  # Cache delle telecamere disponibili

    def get_cameras(self) -> List[Dict[str, Any]]:
        """
        Ottiene la lista delle telecamere disponibili.

        Returns:
            Lista di dizionari con informazioni sulle telecamere
        """
        success, response, _ = self.client.send_message(
            MessageType.CAMERA_LIST,
            {}
        )

        if success and response:
            cameras = response.payload.get("cameras", [])

            # Aggiorna la cache
            self.cameras = {cam["id"]: cam for cam in cameras}

            return cameras
        else:
            logger.error("Impossibile ottenere la lista delle telecamere")
            return []

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Ottiene informazioni su una telecamera.

        Args:
            camera_id: ID della telecamera

        Returns:
            Dizionario con informazioni sulla telecamera, None se non trovata
        """
        # Aggiorna la cache se necessario
        if not self.cameras:
            self.get_cameras()

        return self.cameras.get(camera_id)

    def configure(self, camera_id: str, config: Dict[str, Any]) -> bool:
        """
        Configura una telecamera.

        Args:
            camera_id: ID della telecamera
            config: Configurazione della telecamera

        Returns:
            True se la configurazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.CAMERA_CONFIG,
            {
                "camera_id": camera_id,
                "config": config
            }
        )

        if success and response:
            # Aggiorna la cache se la telecamera esiste
            if camera_id in self.cameras:
                self.cameras[camera_id].update({
                    "configured_resolution": config.get("resolution",
                                                      self.cameras[camera_id].get("resolution")),
                    "configured_fps": config.get("fps",
                                                self.cameras[camera_id].get("fps"))
                })

            logger.info(f"Telecamera {camera_id} configurata con successo")
            return True
        else:
            logger.error(f"Errore nella configurazione della telecamera {camera_id}")
            return False

    def set_exposure(self, camera_id: str, exposure_time: int, gain: Optional[float] = None) -> bool:
        """
        Imposta l'esposizione della telecamera.

        Args:
            camera_id: ID della telecamera
            exposure_time: Tempo di esposizione in microsecondi
            gain: Guadagno analogico (opzionale)

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        config = {"exposure": exposure_time}
        if gain is not None:
            config["gain"] = gain
        return self.configure(camera_id, config)

    def set_white_balance(self, camera_id: str, mode: str, red_gain: Optional[float] = None,
                          blue_gain: Optional[float] = None) -> bool:
        """
        Imposta il bilanciamento del bianco della telecamera.

        Args:
            camera_id: ID della telecamera
            mode: Modalità ("auto" o "manual")
            red_gain: Guadagno del canale rosso (necessario per "manual")
            blue_gain: Guadagno del canale blu (necessario per "manual")

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        config = {"awb": mode}
        if mode == "manual" and red_gain is not None and blue_gain is not None:
            config["awb_gains"] = [red_gain, blue_gain]
        return self.configure(camera_id, config)

    def flip_image(self, camera_id: str, horizontal: bool = False, vertical: bool = False) -> bool:
        """
        Specchia l'immagine della telecamera.

        Args:
            camera_id: ID della telecamera
            horizontal: Specchia orizzontalmente
            vertical: Specchia verticalmente

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        return self.configure(camera_id, {"hflip": horizontal, "vflip": vertical})

    def configure_all(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Configura tutte le telecamere con la stessa configurazione.

        Args:
            config: Configurazione da applicare a tutte le telecamere

        Returns:
            Dizionario {camera_id: success} con risultati
        """
        # Aggiorna la cache se necessario
        if not self.cameras:
            self.get_cameras()

        results = {}

        for camera_id in self.cameras:
            success = self.configure(camera_id, config)
            results[camera_id] = success

        return results

    def capture(self, camera_id: str, jpeg_quality: int = 80) -> Optional[np.ndarray]:
        """
        Cattura un'immagine da una telecamera.

        Args:
            camera_id: ID della telecamera
            jpeg_quality: Qualità JPEG (0-100)

        Returns:
            Immagine come array numpy, None in caso di errore
        """
        success, response, binary_data = self.client.send_message(
            MessageType.CAMERA_CAPTURE,
            {
                "camera_id": camera_id,
                "jpeg_quality": jpeg_quality
            },
            binary_response=True
        )

        if success and binary_data:
            try:
                # Decodifica l'immagine JPEG
                image = decode_jpeg_to_image(binary_data)
                return image
            except Exception as e:
                logger.error(f"Errore nella decodifica dell'immagine: {e}")
                return None
        else:
            logger.error(f"Errore nella cattura dell'immagine dalla telecamera {camera_id}")
            return None

    def capture_multi(self, camera_ids: List[str], jpeg_quality: int = 80) -> Dict[str, Optional[np.ndarray]]:
        """
        Cattura immagini sincronizzate da più telecamere.

        Args:
            camera_ids: Lista degli ID delle telecamere
            jpeg_quality: Qualità JPEG (0-100)

        Returns:
            Dizionario {camera_id: immagine} con le immagini catturate
        """
        success, response, binary_data = self.client.send_message(
            MessageType.CAMERA_CAPTURE_MULTI,
            {
                "camera_ids": camera_ids,
                "jpeg_quality": jpeg_quality
            },
            binary_response=True
        )

        if not success or not binary_data:
            logger.error("Errore nella cattura sincronizzata")
            return {}

        try:
            # Decodifica la risposta multicamera
            # Estrai la dimensione dell'header
            header_size = int.from_bytes(binary_data[:4], byteorder='little')

            # Estrai e parsa l'header
            header_json = binary_data[4:4 + header_size].decode('utf-8')
            header = json.loads(header_json)

            # Estrai i metadati delle immagini - gestisci possibili chiavi mancanti
            images_metadata = header.get("images_metadata", {})
            camera_ids_from_response = header.get("camera_ids", camera_ids)  # Fallback all'input originale

            # Dizionario per i risultati
            images = {}

            # Posizione corrente nei dati binari
            current_pos = 4 + header_size

            # Log per debug
            logger.debug(f"Header ricevuto: {header}")
            logger.debug(f"Lunghezza dati binari: {len(binary_data)}")
            logger.debug(f"Posizione dopo header: {current_pos}")

            # Estrai ogni immagine
            for camera_id in camera_ids_from_response:
                try:
                    if current_pos >= len(binary_data):
                        logger.error(f"Dati binari insufficienti per la camera {camera_id}")
                        continue

                    # Leggi la dimensione dell'immagine
                    jpeg_size = int.from_bytes(binary_data[current_pos:current_pos + 4], byteorder='little')
                    current_pos += 4

                    if current_pos + jpeg_size > len(binary_data):
                        logger.error(f"Dati JPEG incompleti per la camera {camera_id}")
                        continue

                    # Estrai i dati JPEG
                    jpeg_data = binary_data[current_pos:current_pos + jpeg_size]
                    current_pos += jpeg_size

                    # Decodifica l'immagine
                    image = decode_jpeg_to_image(jpeg_data)
                    images[camera_id] = image

                    logger.debug(
                        f"Immagine decodificata per camera {camera_id}: {image.shape if image is not None else 'None'}")
                except Exception as cam_error:
                    logger.error(f"Errore durante l'elaborazione della camera {camera_id}: {cam_error}")
                    continue  # Passa alla prossima camera

            return images

        except Exception as e:
            logger.error(f"Errore durante la decodifica della risposta multicamera: {e}")
            logger.error(f"Primi 100 byte della risposta: {binary_data[:100]}")
            return {}

    def get_stereo_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Trova una coppia di telecamere per la stereovisione.

        Returns:
            Tupla (id_camera_sinistra, id_camera_destra), None se non trovata
        """
        # Aggiorna la cache se necessario
        if not self.cameras:
            self.get_cameras()

        if len(self.cameras) < 2:
            logger.warning("Sono necessarie almeno 2 telecamere per la stereovisione")
            return None, None

        # Selezione semplificata: prendi le prime due telecamere
        # In un'implementazione più avanzata, potrebbero esserci metadati aggiuntivi
        # per identificare le telecamere sinistra e destra
        camera_ids = list(self.cameras.keys())

        # Cerca telecamere con "left" o "right" nel nome o ID
        left_camera = None
        right_camera = None

        for camera_id, camera_info in self.cameras.items():
            camera_name = camera_info.get("name", "").lower()
            if "left" in camera_name or "left" in camera_id.lower():
                left_camera = camera_id
            elif "right" in camera_name or "right" in camera_id.lower():
                right_camera = camera_id

        # Se non trovate, usa le prime due telecamere
        if left_camera is None or right_camera is None:
            left_camera = camera_ids[0]
            right_camera = camera_ids[1]
            logger.info(f"Usando telecamere di default per stereovisione: {left_camera} (sinistra), {right_camera} (destra)")
        else:
            logger.info(f"Trovata coppia stereo: {left_camera} (sinistra), {right_camera} (destra)")

        return left_camera, right_camera

    def capture_stereo_pair(self, jpeg_quality: int = 80) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Cattura un'immagine sincronizzata da una coppia stereo.

        Args:
            jpeg_quality: Qualità JPEG (0-100)

        Returns:
            Tupla (immagine_sinistra, immagine_destra), None se errore
        """
        left_camera, right_camera = self.get_stereo_pair()

        if left_camera is None or right_camera is None:
            return None, None

        # Cattura sincronizzata
        images = self.capture_multi([left_camera, right_camera], jpeg_quality)

        if not images:
            return None, None

        return images.get(left_camera), images.get(right_camera)