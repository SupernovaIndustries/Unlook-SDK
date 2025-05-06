"""
Definizione del protocollo di comunicazione tra client e server.
"""

import enum
import json
import time
import uuid
from typing import Any, Dict, Optional


class MessageType(enum.Enum):
    """Tipi di messaggi supportati dal protocollo."""
    # Messaggi generali
    HELLO = "hello"  # Inizializzazione connessione
    INFO = "info"  # Informazioni sullo scanner
    ERROR = "error"  # Errore generale

    # Controllo proiettore
    PROJECTOR_MODE = "projector_mode"  # Imposta modalità proiettore
    PROJECTOR_PATTERN = "projector_pattern"  # Imposta pattern proiettore

    # Controllo camera
    CAMERA_LIST = "camera_list"  # Lista delle telecamere disponibili
    CAMERA_CONFIG = "camera_config"  # Configurazione telecamera
    CAMERA_CAPTURE = "camera_capture"  # Cattura immagine
    CAMERA_CAPTURE_MULTI = "camera_capture_multi"  # Cattura immagine da più telecamere
    CAMERA_STREAM_START = "camera_stream_start"  # Avvia streaming
    CAMERA_STREAM_STOP = "camera_stream_stop"  # Ferma streaming

    # Tipi di risposta binaria
    CAMERA_CAPTURE_RESPONSE = "camera_capture_response"  # Risposta alla cattura immagine
    CAMERA_FRAME = "camera_frame"  # Frame della telecamera per lo streaming
    MULTI_CAMERA_RESPONSE = "multi_camera_response"  # Risposta per cattura multi-camera

    # Scansione 3D
    SCAN_START = "scan_start"  # Avvia scansione
    SCAN_STOP = "scan_stop"  # Ferma scansione
    SCAN_STATUS = "scan_status"  # Stato scansione
    SCAN_RESULT = "scan_result"  # Risultato scansione

    # Stereo vision
    STEREO_CALIBRATION = "stereo_calibration"  # Calibrazione stereo
    STEREO_RECTIFY = "stereo_rectify"  # Rettifica stereo

    # Pattern di luce strutturata
    PATTERN_GENERATE = "pattern_generate"  # Genera pattern
    PATTERN_PROJECT = "pattern_project"  # Proietta pattern

    # Calibrazione
    CALIBRATION_START = "calibration_start"  # Avvia calibrazione
    CALIBRATION_CAPTURE = "calibration_capture"  # Cattura immagine calibrazione
    CALIBRATION_COMPUTE = "calibration_compute"  # Calcola calibrazione
    CALIBRATION_SAVE = "calibration_save"  # Salva calibrazione
    CALIBRATION_LOAD = "calibration_load"  # Carica calibrazione

    # Gestione sistema
    SYSTEM_STATUS = "system_status"  # Stato del sistema
    SYSTEM_RESET = "system_reset"  # Reset del sistema
    SYSTEM_SHUTDOWN = "system_shutdown"  # Arresto del sistema


class Message:
    """Classe per la gestione dei messaggi del protocollo."""

    def __init__(
        self,
        msg_type: MessageType,
        payload: Optional[Dict[str, Any]] = None,
        msg_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ):
        """
        Inizializza un nuovo messaggio.

        Args:
            msg_type: Tipo del messaggio
            payload: Contenuto del messaggio
            msg_id: ID univoco del messaggio (generato se non fornito)
            reply_to: ID del messaggio a cui si risponde
        """
        self.msg_type = msg_type
        self.payload = payload or {}
        self.msg_id = msg_id or str(uuid.uuid4())
        self.timestamp = time.time()
        self.reply_to = reply_to

    def to_dict(self) -> Dict[str, Any]:
        """Converte il messaggio in un dizionario."""
        return {
            "type": self.msg_type.value,
            "id": self.msg_id,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "payload": self.payload
        }

    def to_json(self) -> str:
        """Converte il messaggio in JSON."""
        return json.dumps(self.to_dict())

    def to_bytes(self) -> bytes:
        """Converte il messaggio in bytes per la trasmissione."""
        return self.to_json().encode('utf-8')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Crea un messaggio da un dizionario.

        Args:
            data: Dizionario con i dati del messaggio

        Returns:
            Istanza di Message
        """
        try:
            msg_type = MessageType(data.get("type"))
            return cls(
                msg_type=msg_type,
                payload=data.get("payload", {}),
                msg_id=data.get("id"),
                reply_to=data.get("reply_to")
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Formato messaggio non valido: {e}")

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """
        Crea un messaggio da una stringa JSON.

        Args:
            json_str: Stringa JSON

        Returns:
            Istanza di Message
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON non valido: {e}")

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """
        Crea un messaggio da bytes.

        Args:
            data: Bytes del messaggio

        Returns:
            Istanza di Message
        """
        try:
            return cls.from_json(data.decode('utf-8'))
        except UnicodeDecodeError as e:
            raise ValueError(f"Decodifica UTF-8 fallita: {e}")

    @classmethod
    def create_reply(cls, original_msg: 'Message', payload: Dict[str, Any],
                    msg_type: Optional[MessageType] = None) -> 'Message':
        """
        Crea un messaggio di risposta.

        Args:
            original_msg: Messaggio originale
            payload: Payload della risposta
            msg_type: Tipo della risposta (se diverso dall'originale)

        Returns:
            Messaggio di risposta
        """
        return cls(
            msg_type=msg_type or original_msg.msg_type,
            payload=payload,
            reply_to=original_msg.msg_id
        )

    @classmethod
    def create_error(cls, original_msg: 'Message', error_message: str,
                    error_code: int = 500) -> 'Message':
        """
        Crea un messaggio di errore.

        Args:
            original_msg: Messaggio originale
            error_message: Messaggio di errore
            error_code: Codice di errore

        Returns:
            Messaggio di errore
        """
        return cls(
            msg_type=MessageType.ERROR,
            payload={
                "error_message": error_message,
                "error_code": error_code
            },
            reply_to=original_msg.msg_id if original_msg else None
        )