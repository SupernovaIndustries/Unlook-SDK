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
    Serializza un messaggio con dati binari.

    Args:
        msg_type: Tipo di messaggio
        payload: Payload come dizionario
        binary_data: Dati binari opzionali

    Returns:
        Messaggio serializzato come bytes
    """
    # Crea l'header
    header = {
        "type": msg_type,
        "payload": payload,
        "timestamp": time.time(),
        "has_binary": binary_data is not None,
        "binary_size": len(binary_data) if binary_data else 0
    }

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


def deserialize_binary_message(data: bytes) -> Tuple[str, Dict, Optional[bytes]]:
    """
    Deserializza un messaggio con dati binari.

    Args:
        data: Messaggio serializzato

    Returns:
        Tupla (tipo_messaggio, payload, dati_binari)
    """
    try:
        # Estrae la dimensione dell'header
        header_size = int.from_bytes(data[:4], byteorder='little')

        # Estrae e parsa l'header
        header_json = data[4:4 + header_size].decode('utf-8')
        header = json.loads(header_json)

        # Estrae eventuali dati binari
        binary_data = None
        if header.get("has_binary", False):
            binary_data = data[4 + header_size:]

        return header["type"], header["payload"], binary_data
    except Exception as e:
        logger.error(f"Errore nella deserializzazione del messaggio binario: {e}")
        raise ValueError(f"Errore nella deserializzazione: {e}")


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