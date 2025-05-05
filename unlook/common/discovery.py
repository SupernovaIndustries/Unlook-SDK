"""
Modulo per la discovery automatica degli scanner UnLook sulla rete locale.
"""

import uuid
import socket
import logging
import threading
import time
from typing import Dict, List, Optional, Callable

from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser
from zeroconf.const import _TYPE_PTR

from .constants import SERVICE_TYPE, SERVICE_NAME, DEFAULT_CONTROL_PORT

logger = logging.getLogger(__name__)


class UnlookScanner:
    """Rappresenta uno scanner UnLook scoperto sulla rete."""

    def __init__(self, name: str, host: str, port: int,
                 scanner_uuid: str, scanner_info: Dict = None):
        """
        Inizializza un nuovo scanner.

        Args:
            name: Nome dello scanner
            host: Indirizzo IP dello scanner
            port: Porta di controllo dello scanner
            scanner_uuid: UUID univoco dello scanner
            scanner_info: Informazioni aggiuntive sullo scanner
        """
        self.name = name
        self.host = host
        self.port = port
        self.uuid = scanner_uuid
        self.info = scanner_info or {}
        self.last_seen = time.time()

    @property
    def endpoint(self) -> str:
        """Endpoint di connessione allo scanner."""
        return f"tcp://{self.host}:{self.port}"

    def __str__(self) -> str:
        return f"UnlookScanner(name={self.name}, uuid={self.uuid}, endpoint={self.endpoint})"

    def __repr__(self) -> str:
        return self.__str__()


class UnlookDiscovery:
    """
    Classe per la discovery degli scanner UnLook sulla rete locale.
    Supporta sia l'annuncio (server) che la scoperta (client) degli scanner.
    """

    def __init__(self):
        """Inizializza il sistema di discovery."""
        self.zeroconf = Zeroconf()
        self.browser = None
        self.service_info = None
        self.scanners: Dict[str, UnlookScanner] = {}
        self._callbacks: List[Callable[[UnlookScanner, bool], None]] = []
        self._lock = threading.RLock()

    def register_scanner(self, name: str, port: int = DEFAULT_CONTROL_PORT,
                         scanner_uuid: Optional[str] = None,
                         scanner_info: Dict = None) -> str:
        """
        Registra uno scanner sulla rete locale (lato server).

        Args:
            name: Nome dello scanner
            port: Porta di controllo dello scanner
            scanner_uuid: UUID univoco dello scanner (generato se non fornito)
            scanner_info: Informazioni aggiuntive sullo scanner

        Returns:
            UUID dello scanner
        """
        if self.service_info:
            self.unregister_scanner()

        # Genera UUID se non fornito
        scanner_uuid = scanner_uuid or str(uuid.uuid4())

        # Prepara le informazioni sul servizio
        scanner_info = scanner_info or {}
        properties = {
            'uuid': scanner_uuid,
            'version': '1.0',
        }
        properties.update(scanner_info)

        # Ottieni l'indirizzo IP locale
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        # Crea e registra il servizio
        service_name = f"{name}.{SERVICE_TYPE}"
        self.service_info = ServiceInfo(
            SERVICE_TYPE,
            service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=port,
            properties=properties,
            server=f"{hostname}.local."
        )

        self.zeroconf.register_service(self.service_info)
        logger.info(f"Scanner registrato sulla rete: {name} ({scanner_uuid}) su {local_ip}:{port}")

        return scanner_uuid

    def unregister_scanner(self):
        """Rimuove la registrazione dello scanner dalla rete."""
        if self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.service_info = None
            logger.info("Scanner rimosso dalla rete")

    def start_discovery(self, callback: Optional[Callable[[UnlookScanner, bool], None]] = None):
        """
        Avvia la discovery degli scanner sulla rete (lato client).

        Args:
            callback: Funzione da chiamare quando viene trovato o rimosso uno scanner
                     La funzione riceve (scanner, added) dove added è True se lo scanner
                     è stato aggiunto, False se rimosso.
        """
        if callback:
            self._callbacks.append(callback)

        if not self.browser:
            self.browser = ServiceBrowser(self.zeroconf, SERVICE_TYPE, self)
            logger.info("Discovery scanner avviata")

    def stop_discovery(self):
        """Ferma la discovery degli scanner."""
        if self.browser:
            self.browser.cancel()
            self.browser = None
            logger.info("Discovery scanner fermata")

    def get_scanners(self) -> List[UnlookScanner]:
        """
        Restituisce la lista degli scanner attualmente disponibili.

        Returns:
            Lista di oggetti UnlookScanner
        """
        with self._lock:
            # Rimuovi scanner non visti recentemente (> 60s)
            now = time.time()
            expired_uuids = [uuid for uuid, scanner in self.scanners.items()
                             if now - scanner.last_seen > 60]

            for uuid in expired_uuids:
                scanner = self.scanners.pop(uuid)
                logger.info(f"Scanner rimosso (timeout): {scanner}")

                for callback in self._callbacks:
                    try:
                        callback(scanner, False)
                    except Exception as e:
                        logger.error(f"Errore in callback: {e}")

            return list(self.scanners.values())

    # Metodi per ServiceBrowser

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Chiamato quando viene trovato un nuovo servizio."""
        info = zc.get_service_info(type_, name)
        if not info:
            return

        try:
            # Estrai informazioni sul servizio
            addresses = info.parsed_addresses()
            if not addresses:
                return

            host = addresses[0]
            port = info.port

            # Estrai proprietà
            properties = {}
            for key, value in info.properties.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                properties[key] = value

            scanner_uuid = properties.get('uuid')
            if not scanner_uuid:
                logger.warning(f"Servizio senza UUID ignorato: {name}")
                return

            # Estrai nome dallo service name
            scanner_name = name.replace(f".{SERVICE_TYPE}", "")

            # Crea o aggiorna scanner
            with self._lock:
                if scanner_uuid in self.scanners:
                    # Aggiorna scanner esistente
                    scanner = self.scanners[scanner_uuid]
                    scanner.host = host
                    scanner.port = port
                    scanner.name = scanner_name
                    scanner.info = properties
                    scanner.last_seen = time.time()

                    logger.debug(f"Scanner aggiornato: {scanner}")
                    is_new = False
                else:
                    # Crea nuovo scanner
                    scanner = UnlookScanner(
                        name=scanner_name,
                        host=host,
                        port=port,
                        scanner_uuid=scanner_uuid,
                        scanner_info=properties
                    )
                    self.scanners[scanner_uuid] = scanner

                    logger.info(f"Nuovo scanner trovato: {scanner}")
                    is_new = True

                # Notifica i callback
                for callback in self._callbacks:
                    try:
                        callback(scanner, is_new)
                    except Exception as e:
                        logger.error(f"Errore in callback: {e}")

        except Exception as e:
            logger.error(f"Errore durante l'analisi del servizio {name}: {e}")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Chiamato quando un servizio viene aggiornato."""
        self.add_service(zc, type_, name)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Chiamato quando un servizio viene rimosso."""
        try:
            # Cerca lo scanner con questo nome
            with self._lock:
                for uuid, scanner in list(self.scanners.items()):
                    if scanner.name == name.replace(f".{SERVICE_TYPE}", ""):
                        # Rimuovi lo scanner
                        removed_scanner = self.scanners.pop(uuid)
                        logger.info(f"Scanner rimosso: {removed_scanner}")

                        # Notifica i callback
                        for callback in self._callbacks:
                            try:
                                callback(removed_scanner, False)
                            except Exception as e:
                                logger.error(f"Errore in callback: {e}")

                        break
        except Exception as e:
            logger.error(f"Errore durante la rimozione del servizio {name}: {e}")

    def __del__(self):
        """Pulisce le risorse alla distruzione dell'oggetto."""
        self.stop_discovery()
        self.unregister_scanner()
        self.zeroconf.close()