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

        if not success or binary_data is None:
            logger.error("Errore nella cattura sincronizzata")
            return {}

        try:
            # Analizziamo i primi byte per determinare il formato dei dati
            # Se inizia con FF D8, è direttamente un JPEG senza metadati
            # Altrimenti, assumiamo che segua il formato atteso
            images = {}

            # Log per debug a basso livello
            logger.debug(f"Ricevuti {len(binary_data)} bytes di dati binari")
            logger.debug(f"Primi byte: {binary_data[:16].hex()}")

            # Verifichiamo se riceviamo direttamente una singola immagine JPEG
            if len(binary_data) >= 2 and binary_data[0:2] == b'\xff\xd8':
                logger.debug("Rilevato JPEG senza metadati, assumo una singola immagine")
                # È solo una singola immagine JPEG, la assegniamo alla prima camera
                if camera_ids:
                    image = decode_jpeg_to_image(binary_data)
                    if image is not None:
                        images[camera_ids[0]] = image
                        logger.debug(f"Immagine decodificata: {image.shape}")
                return images

            # Formato previsto: dimensione header (4 byte) + header JSON + dati binari
            if len(binary_data) < 4:
                logger.error("Dati insufficienti per estrapolare la dimensione dell'header")
                return {}

            # Estrai la dimensione dell'header
            header_size = int.from_bytes(binary_data[:4], byteorder='little')
            logger.debug(f"Dimensione header dichiarata: {header_size} bytes")

            # Se l'header sembra errato o troppo grande, potremmo avere un formato diverso
            if header_size <= 0 or header_size > len(binary_data) - 4:
                logger.warning(f"Dimensione header sospetta: {header_size}, controllo formato alternativo")

                # Proviamo un approccio alternativo: assumiamo che riceviamo direttamente le immagini JPEG
                # senza un header JSON, ma con la dimensione di ciascuna immagine
                current_pos = 0
                for camera_id in camera_ids:
                    # Verifichiamo se abbiamo abbastanza dati
                    if current_pos + 4 > len(binary_data):
                        logger.error(f"Dati insufficienti per leggere la dimensione dell'immagine {camera_id}")
                        break

                    # Leggiamo la dimensione dell'immagine
                    img_size = int.from_bytes(binary_data[current_pos:current_pos + 4], byteorder='little')
                    current_pos += 4

                    # Verifichiamo se la dimensione è valida
                    if img_size <= 0 or current_pos + img_size > len(binary_data):
                        logger.error(f"Dimensione immagine non valida: {img_size}")
                        break

                    # Estrai i dati JPEG
                    jpeg_data = binary_data[current_pos:current_pos + img_size]
                    current_pos += img_size

                    # Decodifica l'immagine
                    image = decode_jpeg_to_image(jpeg_data)
                    if image is not None:
                        images[camera_id] = image
                        logger.debug(f"Immagine decodificata per camera {camera_id}: {image.shape}")
                    else:
                        logger.error(f"Impossibile decodificare l'immagine per la camera {camera_id}")

                return images

            # Formato standard: proviamo a decodificare l'header JSON
            try:
                header_json = binary_data[4:4 + header_size].decode('utf-8')
                header = json.loads(header_json)

                # Estrai i metadati delle immagini
                images_metadata = header.get("images_metadata", {})
                camera_ids_from_response = header.get("camera_ids", camera_ids)

                logger.debug(f"Metadati immagini: {images_metadata}")
                logger.debug(f"Camera IDs dalla risposta: {camera_ids_from_response}")

                # Posizione corrente nei dati binari
                current_pos = 4 + header_size

                # Estrai ogni immagine
                for camera_id in camera_ids_from_response:
                    # Verifichiamo se abbiamo abbastanza dati
                    if current_pos + 4 > len(binary_data):
                        logger.error(f"Dati insufficienti per leggere la dimensione dell'immagine {camera_id}")
                        break

                    # Leggiamo la dimensione dell'immagine
                    jpeg_size = int.from_bytes(binary_data[current_pos:current_pos + 4], byteorder='little')
                    current_pos += 4

                    # Verifichiamo se la dimensione è valida
                    if jpeg_size <= 0 or current_pos + jpeg_size > len(binary_data):
                        logger.error(f"Dimensione immagine non valida: {jpeg_size}")
                        break

                    # Estrai i dati JPEG
                    jpeg_data = binary_data[current_pos:current_pos + jpeg_size]
                    current_pos += jpeg_size

                    # Decodifica l'immagine
                    image = decode_jpeg_to_image(jpeg_data)
                    if image is not None:
                        images[camera_id] = image
                        logger.debug(f"Immagine decodificata per camera {camera_id}: {image.shape}")
                    else:
                        logger.error(f"Impossibile decodificare l'immagine per la camera {camera_id}")

            except UnicodeDecodeError as e:
                logger.error(f"Errore nella decodifica dell'header JSON: {e}")
                # In caso di errore nella decodifica dell'header, proviamo un approccio alternativo
                return self._fallback_decode_multi_response(binary_data, camera_ids)
            except json.JSONDecodeError as e:
                logger.error(f"Errore nella deserializzazione dell'header JSON: {e}")
                return self._fallback_decode_multi_response(binary_data, camera_ids)
            except Exception as e:
                logger.error(f"Errore generico nella decodifica: {e}")
                return self._fallback_decode_multi_response(binary_data, camera_ids)

            return images

        except Exception as e:
            logger.error(f"Errore durante la decodifica della risposta multicamera: {e}")
            logger.error(f"Primi 100 byte della risposta: {binary_data[:100]}")
            return {}

    def _fallback_decode_multi_response(self, binary_data: bytes, camera_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Metodo di fallback per decodificare la risposta nel caso in cui il formato non sia standard.

        Args:
            binary_data: Dati binari ricevuti
            camera_ids: IDs delle telecamere richieste

        Returns:
            Dizionario {camera_id: immagine}
        """
        logger.warning("Utilizzo metodo di fallback per decodificare la risposta multicamera")
        images = {}

        # Analisi di basso livello dei dati ricevuti
        if len(binary_data) < 8:
            logger.error("Dati binari insufficienti per il metodo di fallback")
            return {}

        # Controlliamo se riceviamo direttamente una sequenza di immagini JPEG con dimensioni prefissate
        current_pos = 0
        for camera_id in camera_ids:
            if current_pos + 4 > len(binary_data):
                break

            # Proviamo a leggere la dimensione come un intero little-endian a 32 bit
            img_size = int.from_bytes(binary_data[current_pos:current_pos + 4], byteorder='little')
            current_pos += 4

            # Verifichiamo se la dimensione è ragionevole
            if img_size <= 0 or img_size > 10 * 1024 * 1024 or current_pos + img_size > len(binary_data):
                # Proviamo a trovare direttamente i marker JPEG
                for i in range(current_pos, len(binary_data) - 1):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD8:
                        # Trovato inizio JPEG
                        jpeg_start = i
                        jpeg_end = None

                        # Cerca la fine del JPEG (FFD9)
                        for j in range(i + 2, len(binary_data) - 1):
                            if binary_data[j] == 0xFF and binary_data[j + 1] == 0xD9:
                                jpeg_end = j + 2
                                break

                        if jpeg_end is not None:
                            # Estrai i dati JPEG
                            jpeg_data = binary_data[jpeg_start:jpeg_end]
                            image = decode_jpeg_to_image(jpeg_data)
                            if image is not None:
                                images[camera_id] = image
                                logger.debug(f"JPEG trovato tramite rilevamento marker: {jpeg_start}-{jpeg_end}")
                                current_pos = jpeg_end
                                break

                # Se non abbiamo trovato nessun JPEG, usciamo
                if camera_id not in images:
                    logger.error(f"Impossibile trovare dati JPEG validi per la camera {camera_id}")
                    break
            else:
                # La dimensione sembra ragionevole, proviamo a estrapolare i dati JPEG
                jpeg_data = binary_data[current_pos:current_pos + img_size]
                current_pos += img_size

                # Verifichiamo se i dati iniziano con marker JPEG (FFD8)
                if len(jpeg_data) >= 2 and jpeg_data[0:2] == b'\xff\xd8':
                    image = decode_jpeg_to_image(jpeg_data)
                    if image is not None:
                        images[camera_id] = image
                        logger.debug(f"Immagine decodificata per camera {camera_id} (fallback): {image.shape}")
                    else:
                        logger.error(f"Impossibile decodificare l'immagine per la camera {camera_id} (fallback)")
                else:
                    logger.error(f"I dati estratti non sembrano un JPEG valido per la camera {camera_id}")

        return images

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
            logger.info(
                f"Usando telecamere di default per stereovisione: {left_camera} (sinistra), {right_camera} (destra)")
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
            logger.error("Impossibile trovare una coppia stereo valida")
            return None, None

        # Configura le telecamere se necessario (ad esempio, stessa risoluzione)
        try:
            # Ottieni le informazioni delle telecamere
            left_info = self.get_camera(left_camera)
            right_info = self.get_camera(right_camera)

            # Se le risoluzioni sono diverse, uniformale
            if (left_info and right_info and
                    left_info.get("configured_resolution") != right_info.get("configured_resolution")):
                # Usa la risoluzione più bassa tra le due
                left_res = left_info.get("configured_resolution", [1920, 1080])
                right_res = right_info.get("configured_resolution", [1920, 1080])

                target_res = [
                    min(left_res[0], right_res[0]),
                    min(left_res[1], right_res[1])
                ]

                logger.info(f"Uniformando risoluzione telecamere stereo a {target_res}")
                self.configure(left_camera, {"resolution": target_res})
                self.configure(right_camera, {"resolution": target_res})
        except Exception as e:
            logger.warning(f"Errore durante la configurazione delle telecamere: {e}")

        # Cattura sincronizzata con il metodo migliorato
        logger.debug("Utilizzo di capture_multi per la cattura stereo")
        images = self.capture_multi([left_camera, right_camera], jpeg_quality)

        # Verifica i risultati
        if not images:
            logger.error("Errore nella cattura delle immagini stereo")
            return None, None

        if len(images) != 2:
            logger.error(f"Numero errato di immagini ricevute ({len(images)} invece di 2)")
            return None, None

        if left_camera not in images or right_camera not in images:
            logger.error(f"Telecamere mancanti nella risposta: ricevute {list(images.keys())}")
            return None, None

        left_image = images.get(left_camera)
        right_image = images.get(right_camera)

        # Verifica finale
        if left_image is None or right_image is None:
            logger.error("Una o entrambe le immagini stereo sono nulle")
            return None, None

        # Verifica che le immagini abbiano la stessa dimensione
        if left_image.shape != right_image.shape:
            logger.warning(f"Le immagini stereo hanno dimensioni diverse: {left_image.shape} vs {right_image.shape}")
            # Potremmo ridimensionare qui, ma per ora restituiamo le immagini così come sono

        logger.info("Cattura stereo completata con successo")
        return left_image, right_image
