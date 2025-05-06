"""
Module for automatic discovery of UnLook scanners on the local network.
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


class ScannerInfo:
    """Represents an UnLook scanner discovered on the network."""

    def __init__(self, name: str, host: str, port: int,
                 scanner_uuid: str, scanner_info: Dict = None):
        """
        Initialize a new scanner.

        Args:
            name: Scanner name
            host: Scanner IP address
            port: Scanner control port
            scanner_uuid: Unique UUID of the scanner
            scanner_info: Additional scanner information
        """
        self.name = name
        self.host = host
        self.port = port
        self.uuid = scanner_uuid
        self.info = scanner_info or {}
        self.last_seen = time.time()

    @property
    def endpoint(self) -> str:
        """Connection endpoint to the scanner."""
        return f"tcp://{self.host}:{self.port}"

    def __str__(self) -> str:
        return f"ScannerInfo(name={self.name}, uuid={self.uuid}, endpoint={self.endpoint})"

    def __repr__(self) -> str:
        return self.__str__()


class DiscoveryService:
    """
    Class for discovering UnLook scanners on the local network.
    Supports both advertising (server) and discovery (client) of scanners.
    """

    def __init__(self):
        """Initialize the discovery system."""
        self.zeroconf = Zeroconf()
        self.browser = None
        self.service_info = None
        self.scanners: Dict[str, ScannerInfo] = {}
        self._callbacks: List[Callable[[ScannerInfo, bool], None]] = []
        self._lock = threading.RLock()

    def register_scanner(self, name: str, port: int = DEFAULT_CONTROL_PORT,
                         scanner_uuid: Optional[str] = None,
                         scanner_info: Dict = None) -> str:
        """
        Register a scanner on the local network (server side).

        Args:
            name: Scanner name
            port: Scanner control port
            scanner_uuid: Unique scanner UUID (generated if not provided)
            scanner_info: Additional scanner information

        Returns:
            Scanner UUID
        """
        if self.service_info:
            self.unregister_scanner()

        # Generate UUID if not provided
        scanner_uuid = scanner_uuid or str(uuid.uuid4())

        # Prepare service information
        scanner_info = scanner_info or {}
        properties = {
            'uuid': scanner_uuid,
            'version': '1.0',
        }
        properties.update(scanner_info)

        # Get hostname at the beginning, will be needed anyway
        hostname = socket.gethostname()

        # Get local IP address
        local_ip = None

        # Method 1: Use a socket to determine the IP used for external connections
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))  # Use Google DNS
            local_ip = s.getsockname()[0]
            s.close()
        except Exception as e:
            logger.warning(f"Error obtaining IP via socket: {e}")

        # Method 2: If IP is still None, try with gethostbyname
        if not local_ip:
            try:
                local_ip = socket.gethostbyname(hostname)
                # If it returns a localhost IP, reject it
                if local_ip.startswith("127."):
                    local_ip = None
            except Exception as e:
                logger.warning(f"Error in gethostbyname: {e}")

        # Method 3: If we still don't have an IP, try to get it from interfaces
        if not local_ip:
            try:
                # Use subprocess instead of reimporting socket
                import subprocess

                # On Linux (Raspberry Pi)
                try:
                    output = subprocess.check_output("hostname -I", shell=True).decode('utf-8').strip()
                    ips = output.split()
                    if ips:
                        local_ip = ips[0]  # First non-localhost IP
                except:
                    pass
            except Exception as e:
                logger.warning(f"Error obtaining IP from interfaces: {e}")

        # If everything fails, use localhost
        if not local_ip:
            logger.warning("Unable to determine a non-local IP, using 0.0.0.0")
            local_ip = '0.0.0.0'  # Use "any" address which accepts connections on all interfaces

        logger.info(f"IP address used for registration: {local_ip}")

        # Create and register the service
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
        logger.info(f"Scanner registered on network: {name} ({scanner_uuid}) on {local_ip}:{port}")

        return scanner_uuid

    def unregister_scanner(self):
        """Unregister the scanner from the network."""
        if self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.service_info = None
            logger.info("Scanner removed from network")

    def start_discovery(self, callback: Optional[Callable[[ScannerInfo, bool], None]] = None):
        """
        Start discovering scanners on the network (client side).

        Args:
            callback: Function to call when a scanner is found or removed
                    The function receives (scanner, added) where added is True if the scanner
                    was added, False if removed.
        """
        if callback:
            self._callbacks.append(callback)

        if not self.browser:
            self.browser = ServiceBrowser(self.zeroconf, SERVICE_TYPE, self)
            logger.info("Scanner discovery started")

    def stop_discovery(self):
        """Stop scanner discovery."""
        if self.browser:
            self.browser.cancel()
            self.browser = None
            logger.info("Scanner discovery stopped")

    def get_scanners(self) -> List[ScannerInfo]:
        """
        Get the list of currently available scanners.

        Returns:
            List of ScannerInfo objects
        """
        with self._lock:
            # Remove scanners not seen recently (> 60s)
            now = time.time()
            expired_uuids = [uuid for uuid, scanner in self.scanners.items()
                             if now - scanner.last_seen > 60]

            for uuid in expired_uuids:
                scanner = self.scanners.pop(uuid)
                logger.info(f"Scanner removed (timeout): {scanner}")

                for callback in self._callbacks:
                    try:
                        callback(scanner, False)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")

            return list(self.scanners.values())

    # Methods for ServiceBrowser

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a new service is found."""
        info = zc.get_service_info(type_, name)
        if not info:
            return

        try:
            # Extract service information
            addresses = info.parsed_addresses()
            if not addresses:
                return

            host = addresses[0]
            port = info.port

            # Extract properties
            properties = {}
            for key, value in info.properties.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                properties[key] = value

            scanner_uuid = properties.get('uuid')
            if not scanner_uuid:
                logger.warning(f"Service without UUID ignored: {name}")
                return

            # Extract name from service name
            scanner_name = name.replace(f".{SERVICE_TYPE}", "")

            # Create or update scanner
            with self._lock:
                if scanner_uuid in self.scanners:
                    # Update existing scanner
                    scanner = self.scanners[scanner_uuid]
                    scanner.host = host
                    scanner.port = port
                    scanner.name = scanner_name
                    scanner.info = properties
                    scanner.last_seen = time.time()

                    logger.debug(f"Scanner updated: {scanner}")
                    is_new = False
                else:
                    # Create new scanner
                    scanner = ScannerInfo(
                        name=scanner_name,
                        host=host,
                        port=port,
                        scanner_uuid=scanner_uuid,
                        scanner_info=properties
                    )
                    self.scanners[scanner_uuid] = scanner

                    logger.info(f"New scanner found: {scanner}")
                    is_new = True

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(scanner, is_new)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")

        except Exception as e:
            logger.error(f"Error while analyzing service {name}: {e}")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        self.add_service(zc, type_, name)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        try:
            # Find the scanner with this name
            with self._lock:
                for uuid, scanner in list(self.scanners.items()):
                    if scanner.name == name.replace(f".{SERVICE_TYPE}", ""):
                        # Remove the scanner
                        removed_scanner = self.scanners.pop(uuid)
                        logger.info(f"Scanner removed: {removed_scanner}")

                        # Notify callbacks
                        for callback in self._callbacks:
                            try:
                                callback(removed_scanner, False)
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")

                        break
        except Exception as e:
            logger.error(f"Error while removing service {name}: {e}")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.stop_discovery()
        self.unregister_scanner()
        self.zeroconf.close()