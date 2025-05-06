"""
Utility condivise tra client e server.
"""

import json
import uuid
import socket
import logging
import platform
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def get_machine_info() -> Dict[str, Any]:
    """
    Raccoglie informazioni sulla macchina.

    Returns:
        Dizionario con informazioni sulla macchina
    """
    try:
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "machine": platform.machine()
        }
    except Exception as e:
        logger.error(f"Errore durante la raccolta delle informazioni sulla macchina: {e}")
        return {}


def generate_uuid() -> str:
    """
    Genera un UUID univoco.

    Returns:
        UUID come stringa
    """
    return str(uuid.uuid4())


def encode_image_to_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    """
    Codifica un'immagine in JPEG per lo streaming.

    Args:
        image: Immagine numpy
        quality: Qualità JPEG (0-100)

    Returns:
        Bytes dell'immagine JPEG
    """
    success, encoded_img = cv2.imencode(
        '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise ValueError("Errore durante la codifica dell'immagine in JPEG")

    return encoded_img.tobytes()


def decode_jpeg_to_image(jpeg_data: bytes) -> np.ndarray:
    """
    Decodifica un'immagine JPEG.

    Args:
        jpeg_data: Bytes dell'immagine JPEG

    Returns:
        Immagine numpy
    """
    nparr = np.frombuffer(jpeg_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def serialize_binary_message(msg_type: str, payload: Dict,
                             binary_data: Optional[bytes] = None) -> bytes:
    """
    Serializza un messaggio con dati binari in modo più robusto.

    Args:
        msg_type: Tipo di messaggio
        payload: Payload come dizionario
        binary_data: Dati binari opzionali

    Returns:
        Messaggio serializzato come bytes
    """
    # Crea l'header, assicurandosi che il tipo sia sempre presente
    header = {
        "type": msg_type,
        "payload": payload or {},
        "timestamp": time.time(),
        "has_binary": binary_data is not None,
        "binary_size": len(binary_data) if binary_data else 0,
        "protocol_version": "1.1"  # Versione del protocollo per futuri cambiamenti
    }

    try:
        # Converte l'header in JSON e poi in bytes
        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json)

        # Costruisce il messaggio: [header_size (4 bytes) | header (json) | binary_data]
        message = bytearray()
        message.extend(header_size.to_bytes(4, byteorder='little'))
        message.extend(header_json)

        if binary_data:
            message.extend(binary_data)

        return bytes(message)
    except Exception as e:
        # Log dell'errore e ripristino con un messaggio di errore
        logger.error(f"Errore durante la serializzazione del messaggio: {e}")

        # Crea un header di fallback minimo
        fallback_header = {
            "type": "error",
            "payload": {"error": f"Errore di serializzazione: {str(e)}"},
            "timestamp": time.time(),
            "has_binary": False,
            "binary_size": 0,
            "protocol_version": "1.1"
        }

        fallback_json = json.dumps(fallback_header).encode('utf-8')
        fallback_size = len(fallback_json)

        fallback_message = bytearray()
        fallback_message.extend(fallback_size.to_bytes(4, byteorder='little'))
        fallback_message.extend(fallback_json)

        return bytes(fallback_message)


def deserialize_binary_message(data: bytes) -> Tuple[str, Dict, Optional[bytes]]:
    """
    Deserializza un messaggio con dati binari in modo più robusto.

    Args:
        data: Messaggio serializzato

    Returns:
        Tupla (tipo_messaggio, payload, dati_binari)
    """
    # Verifico se i dati iniziano con JPEG SOI marker (0xFF 0xD8)
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        # È un'immagine JPEG diretta senza header
        logger.debug("Rilevato JPEG diretto senza header")
        return "camera_frame", {"format": "jpeg", "direct_image": True}, data

    # Controlli di validità
    if not data or len(data) < 4:
        logger.error("Dati binari insufficienti per la deserializzazione")
        return "error", {"error": "Dati binari insufficienti"}, None

    try:
        # Estrae la dimensione dell'header
        header_size = int.from_bytes(data[:4], byteorder='little')

        # Verifica dimensione ragionevole (evita overflow)
        if header_size <= 0 or header_size > 100000:  # Limite ragionevole per header JSON
            logger.warning(f"Dimensione header sospetta: {header_size}, potrebbe essere formato alternativo")

            # Prova a rilevare se si tratta di un messaggio multicamera senza header standard
            if len(data) >= 8:
                # Controlla se dopo 4 byte c'è una dimensione JPEG ragionevole
                img_size = int.from_bytes(data[4:8], byteorder='little')
                if img_size > 0 and img_size < len(data) - 8:
                    logger.debug(f"Rilevato possibile formato multicamera con dimensione immagine: {img_size}")
                    return "multi_camera_response", {"format": "jpeg", "alternative_format": True}, data

            # Se non riconosciuto, restituisci dati binari grezzi
            return "binary_data", {"format": "unknown"}, data

        # Verifica che ci siano abbastanza dati per l'header
        if len(data) < 4 + header_size:
            logger.error(f"Dati binari incompleti: necessari {4 + header_size} byte, ricevuti {len(data)}")
            return "error", {"error": "Header incompleto"}, None

        # Estrae e parsa l'header
        header_json = data[4:4 + header_size].decode('utf-8')
        header = json.loads(header_json)

        # Log per debug
        logger.debug(f"Header deserializzato: {header}")

        # Estrae eventuali dati binari
        binary_data = None
        if header.get("has_binary", False):
            binary_data = data[4 + header_size:]

            # Verifica dimensione dichiarata
            declared_size = header.get("binary_size", 0)
            actual_size = len(binary_data)

            if declared_size != actual_size:
                logger.warning(
                    f"Dimensione dati binari non corrisponde: dichiarati {declared_size}, ricevuti {actual_size}")

        elif len(data) > 4 + header_size:
            # Se ci sono dati extra, li consideriamo come binari anche se has_binary è False
            logger.debug("Dati binari presenti ma has_binary=False, estrazione comunque")
            binary_data = data[4 + header_size:]

        # Estrae tipo e payload con valori di default sicuri
        msg_type = header.get("type", "unknown_message_type")
        payload = header.get("payload", {})

        # Verifica versione del protocollo
        protocol_version = header.get("protocol_version", "1.0")
        if protocol_version != "1.0" and protocol_version != "1.1":
            logger.warning(f"Versione protocollo non riconosciuta: {protocol_version}")

        return msg_type, payload, binary_data

    except json.JSONDecodeError as e:
        logger.error(f"Errore nella decodifica JSON dell'header: {e}")
        # Se è un errore di decodifica JSON ma l'inizio assomiglia a un JPEG
        if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
            logger.debug("Rilevato JPEG nonostante errore di decodifica JSON")
            return "camera_frame", {"format": "jpeg", "direct_image": True}, data
        return "error", {"error": f"JSON non valido: {str(e)}"}, None
    except Exception as e:
        logger.error(f"Errore nella deserializzazione del messaggio binario: {e}")
        return "error", {"error": f"Errore di deserializzazione: {str(e)}"}, None


class RateLimiter:
    """Classe per limitare la frequenza di chiamate."""

    def __init__(self, rate: float):
        """
        Inizializza un rate limiter.

        Args:
            rate: Frequenza massima in Hz
        """
        self.min_interval = 1.0 / rate
        self.last_call = 0
        self._lock = threading.Lock()

    def wait(self):
        """
        Attende il tempo necessario per rispettare il rate limit.

        Returns:
            Tempo effettivamente atteso in secondi
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            wait_time = max(0, self.min_interval - elapsed)

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_call = time.time()
            return wait_time