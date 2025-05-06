"""
SDK UnLook per la comunicazione e il controllo di scanner 3D a luce strutturata.

Questo SDK fornisce un'interfaccia completa per lavorare con gli scanner UnLook,
permettendo di controllare telecamere, proiettori, acquisire immagini, eseguire
scansioni 3D e ricostruire modelli 3D.

L'architettura è divisa in due parti principali:
- Client: per la connessione e controllo dello scanner
- Server: per l'implementazione del server dello scanner

Per iniziare ad utilizzare l'SDK, consulta la documentazione e gli esempi.
"""

__version__ = "0.1.0"
__author__ = "UnLook Team"

from .client import (
    UnlookClient,
    Calibrator, CalibrationData,
    StructuredLightProcessor, ScanProcessor, ProcessingResult,
    PatternType, PatternDirection,
    ModelExporter
)

from .common.events import UnlookClientEvent

from .common import (
    UnlookDiscovery, UnlookScanner,
    MessageType, Message
)

# Import condizionale per il server (solo su Raspberry Pi)
import platform
import sys

if 'arm' in platform.machine():
    try:
        from .server import (
            UnlookServer,
            DLPC342XController, OperatingMode, Color, BorderEnable
        )
    except ImportError:
        # Il server potrebbe non essere disponibile su tutti i sistemi
        pass