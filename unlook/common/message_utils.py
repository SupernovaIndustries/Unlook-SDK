"""
Utilità per la gestione dei messaggi tra client e server.
"""

import logging
from typing import Tuple, Dict, Any, Optional

from .protocol import Message, MessageType
from .utils import serialize_binary_message, deserialize_binary_message

logger = logging.getLogger(__name__)


def create_error_response(original_message: Message, error_message: str, error_code: int = 500) -> Message:
    """
    Crea un messaggio di errore standardizzato.

    Args:
        original_message: Messaggio originale a cui rispondere
        error_message: Messaggio di errore
        error_code: Codice di errore

    Returns:
        Messaggio di errore
    """
    return Message.create_error(original_message, error_message, error_code)


def create_binary_response(msg_type: str, metadata: Dict[str, Any],
                           binary_data: bytes, format_type: str = "standard") -> bytes:
    """
    Crea una risposta binaria serializzata.

    Args:
        msg_type: Tipo di messaggio
        metadata: Metadati del messaggio
        binary_data: Dati binari da includere
        format_type: Formato di serializzazione

    Returns:
        Messaggio binario serializzato
    """
    return serialize_binary_message(msg_type, metadata, binary_data, format_type)


def process_binary_response(binary_data: bytes) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
    """
    Processa una risposta binaria.

    Args:
        binary_data: Dati binari ricevuti

    Returns:
        Tupla (tipo_messaggio, metadati, dati_binari)
    """
    return deserialize_binary_message(binary_data)