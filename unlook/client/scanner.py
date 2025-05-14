"""
Main client for connecting to an UnLook scanner.
"""

import logging
import threading
import time
import random
from typing import Dict, List, Optional, Any, Callable, Tuple

import zmq

from ..core.events import EventType, EventEmitter
from ..core.protocol import Message, MessageType
from ..core.discovery import DiscoveryService, ScannerInfo
from ..core.constants import DEFAULT_CONTROL_PORT, DEFAULT_TIMEOUT, MAX_RETRIES
from ..core.utils import generate_uuid, get_machine_info, deserialize_binary_message

logger = logging.getLogger(__name__)


class UnlookClient(EventEmitter):
    """
    Main client for connecting to an UnLook scanner.
    Provides a unified interface for all scanner functionalities.
    """

    def __init__(
            self,
            client_name: str = "UnlookClient",
            auto_discover: bool = True,
            discovery_callback: Optional[Callable[[ScannerInfo, bool], None]] = None
    ):
        """
        Initialize a new UnLook client.

        Args:
            client_name: Client name
            auto_discover: Automatically start scanner discovery
            discovery_callback: Callback for scanner discovery
        """
        super().__init__()

        # Client identification
        self.name = client_name
        self.id = generate_uuid()

        # Connection state
        self.connected = False
        self.scanner: Optional[ScannerInfo] = None
        self.scanner_info: Dict[str, Any] = {}

        # ZeroMQ
        self.zmq_context = zmq.Context()
        self.control_socket = None
        self.poller = zmq.Poller()

        # Lock
        self._lock = threading.RLock()

        # Discovery
        self.discovery = DiscoveryService()
        self.discovered_scanners: Dict[str, ScannerInfo] = {}

        # Subclasses block - lazy loading
        self._camera = None
        self._projector = None
        self._stream = None

        # Start discovery if requested
        if auto_discover:
            self.start_discovery(discovery_callback)

    @property
    def camera(self):
        """Lazy-loading of the camera client."""
        if self._camera is None:
            # Delayed import to avoid circular imports
            from .camera import CameraClient
            self._camera = CameraClient(self)
        return self._camera

    @property
    def projector(self):
        """Lazy-loading of the projector client."""
        if self._projector is None:
            # Delayed import to avoid circular imports
            from .projector import ProjectorClient
            self._projector = ProjectorClient(self)
        return self._projector

    @property
    def stream(self):
        """Lazy-loading of the streaming client."""
        if self._stream is None:
            # Delayed import to avoid circular imports
            from .streaming import StreamClient
            self._stream = StreamClient(self)
        return self._stream

    # Connection and scanner management methods
    def start_discovery(self, callback: Optional[Callable[[ScannerInfo, bool], None]] = None):
        """
        Start scanner discovery.

        Args:
            callback: Callback called when a scanner is found or lost
        """

        def _discovery_handler(scanner: ScannerInfo, added: bool):
            # Update internal list
            with self._lock:
                if added:
                    self.discovered_scanners[scanner.uuid] = scanner
                    logger.info(f"Scanner found: {scanner}")
                    self.emit(EventType.SCANNER_FOUND, scanner)
                else:
                    if scanner.uuid in self.discovered_scanners:
                        del self.discovered_scanners[scanner.uuid]
                    logger.info(f"Scanner lost: {scanner}")
                    self.emit(EventType.SCANNER_LOST, scanner)

            # Call user callback if provided
            if callback:
                callback(scanner, added)

        # Start discovery
        self.discovery.start_discovery(_discovery_handler)
        logger.info("Scanner discovery started")

    def stop_discovery(self):
        """Stop scanner discovery."""
        self.discovery.stop_discovery()
        logger.info("Scanner discovery stopped")

    def get_discovered_scanners(self) -> List[ScannerInfo]:
        """
        Get the list of discovered scanners.

        Returns:
            List of scanners
        """
        with self._lock:
            return list(self.discovered_scanners.values())

    def connect(self, scanner_or_endpoint: Any, timeout: int = DEFAULT_TIMEOUT) -> bool:
        """
        Connect to a scanner.

        Args:
            scanner_or_endpoint: Scanner object or endpoint (tcp://host:port) or UUID
            timeout: Connection timeout in milliseconds

        Returns:
            True if connection is successful, False otherwise
        """
        if self.connected:
            self.disconnect()

        # Extract endpoint
        endpoint = None
        if isinstance(scanner_or_endpoint, ScannerInfo):
            endpoint = scanner_or_endpoint.endpoint
            self.scanner = scanner_or_endpoint
        elif isinstance(scanner_or_endpoint, str):
            # Check if it's a UUID or an endpoint
            if "://" in scanner_or_endpoint:
                # It's an endpoint
                endpoint = scanner_or_endpoint
                # Create a "dummy" scanner to store information
                host, port = endpoint.replace("tcp://", "").split(":")
                self.scanner = ScannerInfo(
                    name="Unknown Scanner",
                    host=host,
                    port=int(port),
                    scanner_uuid=generate_uuid()
                )
            else:
                # Could be a UUID, try to find the scanner
                uuid = scanner_or_endpoint
                found_scanner = None

                # Look for the scanner in the list of discovered ones
                with self._lock:
                    for scanner in self.discovered_scanners.values():
                        if scanner.uuid == uuid:
                            found_scanner = scanner
                            break

                if found_scanner:
                    self.scanner = found_scanner
                    endpoint = found_scanner.endpoint
                    logger.info(f"Found scanner with UUID {uuid}, endpoint: {endpoint}")
                else:
                    logger.error(f"Scanner with UUID {uuid} not found. Run discovery first.")
                    return False
        else:
            logger.error(
                "scanner_or_endpoint must be a ScannerInfo object, an endpoint (tcp://host:port) or a UUID")
            return False

        logger.info(f"Connecting to {endpoint}...")

        try:
            # Create control socket
            self.control_socket = self.zmq_context.socket(zmq.REQ)
            self.control_socket.setsockopt(zmq.LINGER, 0)
            self.control_socket.setsockopt(zmq.RCVTIMEO, timeout)
            self.control_socket.connect(endpoint)

            # Register with poller
            self.poller.register(self.control_socket, zmq.POLLIN)

            # Send HELLO message
            hello_msg = Message(
                msg_type=MessageType.HELLO,
                payload={
                    "client_info": {
                        "name": self.name,
                        "id": self.id,
                        "version": "1.0",
                        **get_machine_info()
                    }
                }
            )

            # Send and wait for response
            self.control_socket.send(hello_msg.to_bytes())

            # Wait for response with timeout
            socks = dict(self.poller.poll(timeout))
            if self.control_socket not in socks or socks[self.control_socket] != zmq.POLLIN:
                raise TimeoutError("Timeout during connection")

            # Receive response
            response_data = self.control_socket.recv()
            response = Message.from_bytes(response_data)

            # Store scanner information
            self.scanner_info = response.payload
            if self.scanner:
                self.scanner.info = response.payload

            # Update state
            self.connected = True
            logger.info(f"Connected to {self.scanner.name} ({self.scanner.uuid})")

            # Emit event
            self.emit(EventType.CONNECTED, self.scanner)

            return True

        except Exception as e:
            # Clean up on error
            logger.error(f"Error connecting: {e}")
            self._cleanup_connection()
            self.emit(EventType.ERROR, str(e))
            return False

    def disconnect(self):
        """Disconnect from the scanner."""
        if not self.connected:
            return

        logger.info("Disconnecting from scanner...")

        # Stop streaming if active
        self.stream.stop()

        # Close connection
        self._cleanup_connection()

        # Update state
        self.connected = False
        self.emit(EventType.DISCONNECTED)

        logger.info("Disconnected from scanner")

    def _cleanup_connection(self):
        """Clean up connection resources."""
        try:
            if self.control_socket:
                try:
                    # Only unregister if the socket is still registered with the poller
                    self.poller.unregister(self.control_socket)
                except Exception as e:
                    logger.debug(f"Error unregistering socket from poller: {e}")

                try:
                    # Force socket to close with a timeout
                    self.control_socket.setsockopt(zmq.LINGER, 0)
                    self.control_socket.close(linger=0)
                except Exception as e:
                    logger.debug(f"Error closing socket: {e}")

                # Explicitly set to None
                self.control_socket = None

        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
            # Ensure variables are reset even after errors
            self.control_socket = None

    def send_message(
            self,
            msg_type: MessageType,
            payload: Dict[str, Any],
            timeout: int = DEFAULT_TIMEOUT,
            retries: int = MAX_RETRIES,
            binary_response: bool = False
    ) -> Tuple[bool, Optional[Message], Optional[bytes]]:
        """
        Send a message to the scanner.

        Args:
            msg_type: Message type
            payload: Message payload
            timeout: Timeout in milliseconds
            retries: Number of retries in case of error
            binary_response: If True, expects a binary response

        Returns:
            Tuple (success, response_message, binary_data)
        """
        if not self.connected or not self.control_socket:
            logger.error("Cannot send message: not connected")
            if self.scanner and self.scanner.endpoint:
                # Try to reconnect once
                if self._attempt_reconnection(timeout):
                    logger.info("Reconnected successfully, proceeding with message")
                else:
                    return False, None, None
            else:
                return False, None, None

        # Lock to prevent concurrent access
        with self._lock:
            message = Message(msg_type=msg_type, payload=payload)
            attempt = 0
            max_attempts = retries + 2  # Allow more retries for important operations

            # Add jitter to avoid retry storms
            jitter = random.uniform(0.8, 1.2)

            while attempt < max_attempts:
                try:
                    # Check socket state
                    if not self.control_socket:
                        logger.error("Socket is None, reconnecting...")
                        if not self._attempt_reconnection(timeout):
                            attempt += 1
                            continue

                    # Send message
                    logger.debug(f"Sending {msg_type} message (type: {type(msg_type)}) (attempt {attempt+1}/{max_attempts})")
                    
                    try:
                        self.control_socket.send(message.to_bytes())
                    except AttributeError as e:
                        if "'str' object has no attribute 'value'" in str(e):
                            # Convert string to MessageType if needed
                            if isinstance(msg_type, str):
                                logger.warning(f"MessageType was passed as string '{msg_type}', converting to enum")
                                msg_type = MessageType(msg_type)
                                message = Message(msg_type=msg_type, payload=payload)
                                self.control_socket.send(message.to_bytes())
                            else:
                                raise

                    # Wait for response with timeout
                    socks = dict(self.poller.poll(timeout))
                    if self.control_socket not in socks or socks[self.control_socket] != zmq.POLLIN:
                        # Handle timeout with exponential backoff
                        attempt += 1
                        logger.warning(f"Timeout waiting for response (attempt {attempt}/{max_attempts})")

                        # Calculate backoff time - start at 100ms and double each retry with jitter
                        backoff_time = min(0.1 * (2 ** (attempt - 1)) * jitter, 2.0)  # Cap at 2 seconds
                        time.sleep(backoff_time)

                        # Proactive reconnection on timeout
                        if attempt % 2 == 0:  # Every other timeout, try to reconnect
                            logger.info(f"Proactive reconnection on timeout attempt {attempt}")
                            self._attempt_reconnection(timeout)

                        continue

                    # Receive response
                    response_data = self.control_socket.recv()

                    # If we expect a binary response
                    if binary_response:
                        # Deserialize binary response
                        try:
                            msg_type, payload, binary_data = deserialize_binary_message(response_data)
                            # Create a message from the response
                            response = Message(
                                msg_type=MessageType(msg_type),
                                payload=payload
                            )
                            return True, response, binary_data
                        except Exception as e:
                            logger.error(f"Error deserializing binary response: {e}")
                            attempt += 1
                            if attempt < max_attempts:
                                # Calculate backoff time for parsing errors too
                                backoff_time = min(0.1 * (2 ** (attempt - 1)) * jitter, 1.0)
                                time.sleep(backoff_time)
                                continue
                            return False, None, None
                    else:
                        # Normal response
                        try:
                            response = Message.from_bytes(response_data)

                            # Handle errors
                            if response.msg_type == MessageType.ERROR:
                                error_msg = response.payload.get("error_message", "Unknown error")
                                # Check for "Operation cannot be accomplished" error which might need reconnection
                                if "Operation cannot be accomplished" in error_msg or "not a socket" in error_msg:
                                    attempt += 1
                                    logger.error(f"Operation cannot be accomplished (attempt {attempt}/{max_attempts})")
                                    if attempt < max_attempts:
                                        # Force reconnection for these specific errors
                                        self._attempt_reconnection(timeout)
                                        # Short delay before retry
                                        time.sleep(0.2)
                                        continue

                                logger.error(f"Error from server: {error_msg}")
                                return False, response, None

                            # Success case
                            return True, response, None
                        except Exception as e:
                            logger.error(f"Error parsing response: {e}")
                            attempt += 1
                            if attempt < max_attempts:
                                time.sleep(0.1 * jitter)
                                continue
                            return False, None, None

                except Exception as e:
                    logger.error(f"Error sending message: {e} (attempt {attempt+1}/{max_attempts})")
                    attempt += 1

                    # If not the last attempt, wait before retrying with exponential backoff
                    if attempt < max_attempts:
                        backoff_time = min(0.1 * (2 ** (attempt - 1)) * jitter, 2.0)  # Cap at 2 seconds
                        time.sleep(backoff_time)

                    # Try reconnection after the first attempt for more aggressive recovery
                    if attempt >= 1 and attempt < max_attempts:
                        self._attempt_reconnection(timeout)

            # All attempts failed
            logger.error("All message sending attempts failed")
            # Try one final desperate reconnection attempt before giving up
            self._attempt_reconnection(timeout * 2)
            return False, None, None

    def _attempt_reconnection(self, timeout: int) -> bool:
        """
        Attempt to reconnect the ZMQ socket.

        Args:
            timeout: Socket timeout in milliseconds

        Returns:
            True if reconnection was successful, False otherwise
        """
        try:
            # Try to reconnect
            logger.info("Attempting to reconnect...")
            self.emit(EventType.RECONNECTING)

            # Close existing connection
            self._cleanup_connection()

            # Wait a short time to ensure socket closes properly
            time.sleep(0.1)

            try:
                # Recreate new context if needed
                if not self.zmq_context or self.zmq_context.closed:
                    logger.debug("Creating new ZMQ context")
                    self.zmq_context = zmq.Context()

                # Recreate socket with improved parameters
                self.control_socket = self.zmq_context.socket(zmq.REQ)
                self.control_socket.setsockopt(zmq.LINGER, 0)  # Don't linger on close
                self.control_socket.setsockopt(zmq.RCVTIMEO, timeout)  # Receive timeout
                self.control_socket.setsockopt(zmq.SNDTIMEO, timeout)  # Send timeout
                self.control_socket.setsockopt(zmq.IMMEDIATE, 1)  # Don't queue messages for disconnected peers
                self.control_socket.setsockopt(zmq.RECONNECT_IVL, 100)  # 100ms reconnection interval
                self.control_socket.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # 5s max reconnection interval
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)  # Enable TCP keepalive
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)  # Seconds before sending keepalives
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)  # Interval between keepalives

                # Connect to endpoint
                logger.debug(f"Connecting to endpoint: {self.scanner.endpoint}")
                self.control_socket.connect(self.scanner.endpoint)

                # Register with poller
                self.poller.register(self.control_socket, zmq.POLLIN)

                # Try simple ping to verify connection
                success = self._ping_server(timeout)
                if not success:
                    raise Exception("Ping failed after reconnection")

                logger.info("Socket reconnected successfully and verified with ping")
                return True

            except Exception as inner_e:
                logger.error(f"Error during socket recreation: {inner_e}")
                # Ensure resources are cleaned up
                if self.control_socket:
                    try:
                        self.control_socket.close(linger=0)
                    except:
                        pass
                    self.control_socket = None
                raise

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self.control_socket = None
            self.connected = False
            self.emit(EventType.DISCONNECTED)
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the scanner.

        Returns:
            Dictionary with scanner information
        """
        success, response, _ = self.send_message(
            MessageType.INFO,
            {}
        )

        if success and response:
            return response.payload
        else:
            return {}

    def _ping_server(self, timeout: int) -> bool:
        """
        Send a ping message to the server to verify connection.

        Args:
            timeout: Timeout in milliseconds

        Returns:
            True if ping successful, False otherwise
        """
        if not self.control_socket:
            return False

        try:
            # Create a simple ping message
            ping_msg = Message(
                msg_type=MessageType.PING,
                payload={"timestamp": time.time()}
            )

            # Send ping
            self.control_socket.send(ping_msg.to_bytes())

            # Wait for response with timeout
            socks = dict(self.poller.poll(timeout))
            if self.control_socket not in socks or socks[self.control_socket] != zmq.POLLIN:
                logger.warning("Ping timeout")
                return False

            # Receive response
            response_data = self.control_socket.recv()
            response = Message.from_bytes(response_data)

            # Check response type
            if response.msg_type != MessageType.PONG:
                logger.warning(f"Unexpected response to ping: {response.msg_type}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False

    def check_connection(self) -> bool:
        """
        Check if the connection to the scanner is active.

        Returns:
            True if connected, False otherwise
        """
        if not self.connected:
            return False

        try:
            # Use faster ping method first
            if self._ping_server(DEFAULT_TIMEOUT):
                return True

            # Fall back to full info check
            info = self.get_info()
            return bool(info)
        except Exception:
            return False

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            self.disconnect()
            self.stop_discovery()
            # Clean up ZMQ context
            if hasattr(self, 'zmq_context') and self.zmq_context:
                try:
                    self.zmq_context.term()
                except:
                    pass
        except:
            pass

    # Backward compatibility with old event API
    def on_event(self, event: EventType, callback: Callable):
        """
        Register a callback for an event (compatibility with previous versions).

        Args:
            event: Event type
            callback: Function to call
        """
        self.on(event, callback)

    def off_event(self, event: EventType, callback: Callable):
        """
        Remove a callback for an event (compatibility with previous versions).

        Args:
            event: Event type
            callback: Function to remove
        """
        self.off(event, callback)