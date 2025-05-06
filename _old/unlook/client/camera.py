"""
Client per la gestione delle telecamere dello scanner UnLook.
"""

import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from ..common.protocol import MessageType
from ..common.utils import decode_jpeg_to_image, deserialize_binary_message

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
        Utilizza il miglioramento della funzione deserialize_binary_message per supportare diversi formati.

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
            # Log per debug
            logger.debug(f"Ricevuti {len(binary_data)} bytes di dati binari")

            # Deserializza utilizzando la funzione migliorata
            msg_type, payload, binary_data = deserialize_binary_message(binary_data)

            # Log delle informazioni sul formato
            logger.debug(f"Formato rilevato: {msg_type}, payload: {payload.get('format', 'N/A')}")

            # GESTIONE DEL FORMATO ULMC
            if msg_type == "multi_camera_response" and payload.get("format") == "ULMC":
                logger.info(f"Processamento risposta ULMC con {payload.get('num_cameras', 0)} telecamere")
                images = {}

                # Per ogni telecamera
                for camera_id, camera_info in payload.get("cameras", {}).items():
                    # Estrai e decodifica l'immagine
                    offset = camera_info.get("offset", 0)
                    size = camera_info.get("size", 0)

                    if offset > 0 and size > 0 and offset + size <= len(binary_data):
                        jpeg_data = binary_data[offset:offset + size]
                        image = decode_jpeg_to_image(jpeg_data)

                        if image is not None:
                            images[camera_id] = image
                            logger.debug(f"Decodificata immagine ULMC per camera {camera_id}: {image.shape}")
                        else:
                            logger.error(f"Impossibile decodificare l'immagine ULMC per camera {camera_id}")

                if images:
                    return images

            # GESTIONE IMMAGINE JPEG DIRETTA
            if msg_type == "camera_frame" and payload.get("direct_image", False):
                # È una singola immagine JPEG, la assegniamo alla prima camera
                if camera_ids:
                    image = decode_jpeg_to_image(binary_data)
                    if image is not None:
                        logger.warning("Ricevuta una sola immagine invece di una per ogni telecamera")
                        return {camera_ids[0]: image}

            # GESTIONE FORMATO ALTERNATIVO CON PREFISSI DIMENSIONE
            if msg_type == "multi_camera_response" and payload.get("alternative_format", False):
                logger.info("Processamento formato alternativo con prefissi dimensione")
                return self._fallback_decode_multi_response(binary_data, camera_ids)

            # GESTIONE FORMATO BINARIO GREZZO
            if msg_type == "binary_data":
                logger.info("Processamento dati binari grezzi con metodo di fallback")
                return self._fallback_decode_multi_response(binary_data, camera_ids)

            # Se siamo qui, non siamo riusciti a decodificare il formato
            logger.warning(f"Formato di risposta non riconosciuto: {msg_type}")
            return self._fallback_decode_multi_response(binary_data, camera_ids)

        except Exception as e:
            logger.error(f"Errore durante la decodifica della risposta multicamera: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Tenta il fallback come ultima risorsa
            return self._fallback_decode_multi_response(binary_data, camera_ids)

    def _fallback_decode_multi_response(self, binary_data: bytes, camera_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Metodo di fallback migliorato per decodificare la risposta multicamera.
        Implementa diverse strategie per estrarre le immagini, garantendo massima robustezza.

        Args:
            binary_data: Dati binari ricevuti
            camera_ids: IDs delle telecamere richieste

        Returns:
            Dizionario {camera_id: immagine}
        """
        logger.info("Utilizzo metodo di fallback avanzato per decodificare la risposta multicamera")
        images = {}

        # Analisi più robusta dei dati ricevuti
        if len(binary_data) < 8:
            logger.error("Dati binari insufficienti per il metodo di fallback")
            return {}

        # Stampa informazioni diagnostiche sui primi byte
        logger.debug(f"Primi 32 bytes: {binary_data[:32].hex()}")

        # STRATEGIA 1: Cerca i marker JPEG (FFD8) e fine JPEG (FFD9)
        # Questa è utile quando le immagini sono semplicemente concatenate
        jpeg_starts = []
        jpeg_ends = []

        for i in range(len(binary_data) - 1):
            if binary_data[i] == 0xFF:
                if binary_data[i + 1] == 0xD8:  # Start of JPEG
                    jpeg_starts.append(i)
                elif binary_data[i + 1] == 0xD9:  # End of JPEG
                    jpeg_ends.append(i + 1)

        # Se troviamo lo stesso numero di inizi e fini JPEG e corrisponde al numero di telecamere
        if len(jpeg_starts) == len(jpeg_ends) and len(jpeg_starts) == len(camera_ids):
            logger.info(f"STRATEGIA 1: Trovate {len(jpeg_starts)} immagini JPEG complete nel stream binario")

            # Ordiniamo le posizioni di inizio/fine
            pairs = sorted(zip(jpeg_starts, jpeg_ends))

            for idx, (start, end) in enumerate(pairs):
                jpeg_data = binary_data[start:end + 1]  # Include il marcatore di fine
                image = decode_jpeg_to_image(jpeg_data)

                if image is not None:
                    camera_id = camera_ids[idx]
                    images[camera_id] = image
                    logger.debug(f"Decodificata immagine per camera {camera_id}: {image.shape}")
                else:
                    logger.error(f"Impossibile decodificare l'immagine {idx + 1}")

            return images

        # STRATEGIA 2: Assume formato [size1(4B) | JPEG1 | size2(4B) | JPEG2 | ...]
        logger.info("STRATEGIA 2: Tentativo di decodifica basato su prefissi di dimensione")
        current_pos = 0
        strategy2_images = {}

        try:
            for idx, camera_id in enumerate(camera_ids):
                if current_pos + 4 > len(binary_data):
                    logger.error("Fine prematura dei dati")
                    break

                # Leggi la dimensione del prossimo blocco JPEG
                img_size = int.from_bytes(binary_data[current_pos:current_pos + 4], byteorder='little')
                current_pos += 4

                # Verifica se la dimensione sembra valida
                if img_size <= 0 or img_size > 10 * 1024 * 1024 or current_pos + img_size > len(binary_data):
                    logger.warning(f"Dimensione immagine non valida: {img_size}, salto alla strategia successiva")
                    break

                # Estrai i dati JPEG
                jpeg_data = binary_data[current_pos:current_pos + img_size]
                current_pos += img_size

                # Verifica che siano dati JPEG validi
                if len(jpeg_data) > 2 and jpeg_data[0:2] == b'\xff\xd8':
                    image = decode_jpeg_to_image(jpeg_data)
                    if image is not None:
                        strategy2_images[camera_id] = image
                        logger.debug(f"Decodificata immagine per camera {camera_id}: {image.shape}")
                    else:
                        logger.error(f"Impossibile decodificare l'immagine per camera {camera_id}")
                else:
                    logger.warning(f"I dati non sembrano un JPEG valido per camera {camera_id}")
                    break
        except Exception as e:
            logger.error(f"Errore nella STRATEGIA 2: {e}")

        # Se abbiamo decodificato tutte le immagini, restituisci il risultato
        if len(strategy2_images) == len(camera_ids):
            logger.info("STRATEGIA 2 completata con successo")
            return strategy2_images

        # STRATEGIA 3: Cerca marker JPEG e usa euristica per riconoscere la fine
        logger.info("STRATEGIA 3: Ricerca euristica delle immagini JPEG")
        strategy3_images = {}

        try:
            pos = 0
            for idx, camera_id in enumerate(camera_ids):
                # Cerca l'inizio di un JPEG
                jpeg_start = -1
                for i in range(pos, len(binary_data) - 1):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD8:
                        jpeg_start = i
                        break

                if jpeg_start == -1:
                    logger.error(f"Impossibile trovare l'inizio dell'immagine JPEG per camera {camera_id}")
                    break

                # Cerca la fine del JPEG o il prossimo inizio di JPEG
                jpeg_end = -1
                next_start = -1

                for i in range(jpeg_start + 2, len(binary_data) - 1):
                    if binary_data[i] == 0xFF:
                        if binary_data[i + 1] == 0xD9:  # Fine JPEG
                            jpeg_end = i + 1
                            break
                        elif binary_data[i + 1] == 0xD8 and i > jpeg_start + 100:  # Nuovo JPEG, ma non troppo vicino
                            next_start = i
                            break

                # Se abbiamo trovato la fine, estrai il JPEG
                if jpeg_end != -1:
                    jpeg_data = binary_data[jpeg_start:jpeg_end + 1]
                    pos = jpeg_end + 1
                # Altrimenti, se abbiamo trovato l'inizio di un nuovo JPEG, usa quello come fine
                elif next_start != -1:
                    jpeg_data = binary_data[jpeg_start:next_start]
                    pos = next_start
                # Altrimenti, prendi tutto fino alla fine
                else:
                    jpeg_data = binary_data[jpeg_start:]
                    pos = len(binary_data)

                # Decodifica l'immagine
                image = decode_jpeg_to_image(jpeg_data)
                if image is not None:
                    strategy3_images[camera_id] = image
                    logger.debug(f"Decodificata immagine per camera {camera_id}: {image.shape}")
                else:
                    logger.error(f"Impossibile decodificare l'immagine per camera {camera_id}")
        except Exception as e:
            logger.error(f"Errore nella STRATEGIA 3: {e}")

        # Se abbiamo decodificato tutte le immagini, restituisci il risultato
        if len(strategy3_images) == len(camera_ids):
            logger.info("STRATEGIA 3 completata con successo")
            return strategy3_images

        # STRATEGIA 4: Ultima risorsa - suddividi equamente il buffer tra le telecamere
        # Utile se le immagini hanno dimensioni simili
        if not images and len(camera_ids) > 0:
            logger.info("STRATEGIA 4: Suddivisione equidistante del buffer")

            chunk_size = len(binary_data) // len(camera_ids)
            for idx, camera_id in enumerate(camera_ids):
                start = idx * chunk_size
                end = (idx + 1) * chunk_size if idx < len(camera_ids) - 1 else len(binary_data)

                # Cerca l'inizio di un JPEG nel chunk
                jpeg_start = -1
                for i in range(start, min(start + 100, end - 1)):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD8:
                        jpeg_start = i
                        break

                if jpeg_start == -1:
                    logger.error(f"Impossibile trovare l'inizio dell'immagine JPEG per camera {camera_id}")
                    continue

                # Cerca la fine del JPEG
                jpeg_end = -1
                for i in range(end - 2, jpeg_start + 2, -1):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD9:
                        jpeg_end = i + 1
                        break

                if jpeg_end == -1:
                    logger.warning(
                        f"Impossibile trovare la fine dell'immagine JPEG per camera {camera_id}, uso fine chunk")
                    jpeg_end = end

                # Estrai e decodifica
                jpeg_data = binary_data[jpeg_start:jpeg_end + 1]
                image = decode_jpeg_to_image(jpeg_data)
                if image is not None:
                    images[camera_id] = image
                    logger.debug(f"Decodificata immagine per camera {camera_id}: {image.shape}")
                else:
                    logger.error(f"Impossibile decodificare l'immagine per camera {camera_id}")

        # Combina i risultati delle diverse strategie, dando priorità a quelle più affidabili
        final_images = {}
        final_images.update(images)  # Strategia 4 ha priorità bassa
        final_images.update(strategy3_images)  # Strategia 3 ha priorità media
        final_images.update(strategy2_images)  # Strategia 2 ha priorità alta

        logger.info(f"Decodificate {len(final_images)}/{len(camera_ids)} immagini tramite fallback")
        return final_images

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
