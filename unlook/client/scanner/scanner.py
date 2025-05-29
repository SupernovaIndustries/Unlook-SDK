"""
Main client for connecting to an UnLook scanner.
"""

import logging
import threading
import time
import random
from typing import Dict, List, Optional, Any, Callable, Tuple, TYPE_CHECKING
import numpy as np
import cv2

import zmq

if TYPE_CHECKING:
    from ..camera import CameraClient
    from ..projector import ProjectorClient
    from ..streaming import StreamClient

from ...core.events import EventType, EventEmitter
from ...core.protocol import Message, MessageType
from ...core.discovery import DiscoveryService, ScannerInfo
from unlook.core.constants import (
    DEFAULT_CONTROL_PORT, DEFAULT_TIMEOUT, MAX_RETRIES,
    PreprocessingVersion, PREPROCESSING_CONFIGS
)
from ...core.utils import generate_uuid, get_machine_info, deserialize_binary_message

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
            discovery_callback: Optional[Callable[[ScannerInfo, bool], None]] = None,
            preprocessing_version: str = PreprocessingVersion.AUTO
    ):
        """
        Initialize a new UnLook client.

        Args:
            client_name: Client name
            auto_discover: Automatically start scanner discovery
            discovery_callback: Callback for scanner discovery
            preprocessing_version: Preprocessing pipeline version (V1_LEGACY, V2_ENHANCED, AUTO)
        """
        super().__init__()

        # Client identification
        self.name = client_name
        
        # Preprocessing version configuration
        self.preprocessing_version = preprocessing_version
        self.preprocessing_config = PREPROCESSING_CONFIGS.get(
            preprocessing_version, 
            PREPROCESSING_CONFIGS[PreprocessingVersion.AUTO]
        )
        
        logger.info(f"Client initialized with preprocessing version: {preprocessing_version}")
        logger.debug(f"Preprocessing config: {self.preprocessing_config['description']}")
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
    def camera(self) -> 'CameraClient':
        """Lazy-loading of the camera client."""
        if self._camera is None:
            # Import from reorganized structure
            from ..camera import CameraClient
            self._camera = CameraClient(self)
        return self._camera

    @property
    def projector(self) -> 'ProjectorClient':
        """Lazy-loading of the projector client."""
        if self._projector is None:
            # Delayed import to avoid circular imports
            from ..projector import ProjectorClient
            self._projector = ProjectorClient(self)
        return self._projector

    @property
    def stream(self) -> 'StreamClient':
        """Lazy-loading of the streaming client."""
        if self._stream is None:
            # Delayed import to avoid circular imports
            from ..streaming import StreamClient
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

            # Send HELLO message with preprocessing version
            hello_msg = Message(
                msg_type=MessageType.HELLO,
                payload={
                    "client_info": {
                        "name": self.name,
                        "id": self.id,
                        "version": "1.0",
                        **get_machine_info()
                    },
                    "preprocessing_config": {
                        "version": self.preprocessing_version,
                        "config": self.preprocessing_config
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
                        # For binary responses, pass raw data to the calling function
                        # Let the specific handler (like capture_multi) do the deserialization
                        # This avoids double-deserialization when V2 is used
                        
                        # Create a minimal response to indicate binary data is available
                        response = Message(
                            msg_type=MessageType.MULTI_CAMERA_RESPONSE,  # Use appropriate response type
                            payload={"binary_response": True}
                        )
                        return True, response, response_data
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

    # ============== NEW FEATURES SUPPORT ==============
    
    def get_sync_metrics(self) -> Dict[str, Any]:
        """
        Get synchronization quality metrics from the server.
        
        Returns:
            Dictionary with sync metrics like precision, frame consistency, etc.
        """
        if not self.connected:
            logger.error("Not connected to scanner")
            return {}
        
        try:
            success, response, _ = self.send_message(MessageType.SYNC_METRICS, {})
            if success and response:
                if isinstance(response, dict):
                    return response
                elif hasattr(response, 'payload'):
                    return response.payload
                else:
                    logger.warning("Failed to get sync metrics or invalid response")
                    return {}
            else:
                logger.warning("Failed to get sync metrics or invalid response")
                return {}
        except Exception as e:
            logger.error(f"Error getting sync metrics: {e}")
            return {}
    
    def enable_sync(self, enable: bool = True, fps: float = 30.0) -> bool:
        """
        Enable/disable hardware synchronization.
        
        Args:
            enable: Whether to enable sync
            fps: Target FPS for sync
            
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Not connected to scanner")
            return False
        
        try:
            success, response, _ = self.send_message(
                MessageType.SYNC_ENABLE, 
                {"enable": enable, "fps": fps}
            )
            return success and response and response.payload.get("success", False)
        except Exception as e:
            logger.error(f"Error enabling sync: {e}")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get detailed server status including optimization settings.
        
        Returns:
            Dictionary with server configuration and status
        """
        if not self.connected:
            logger.error("Not connected to scanner")
            return {}
        
        try:
            success, response, _ = self.send_message(MessageType.SYSTEM_STATUS, {})
            if success and response:
                if isinstance(response, dict):
                    return response
                elif hasattr(response, 'payload'):
                    return response.payload
                else:
                    return {}
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
            return {}
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get protocol v2 compression statistics from the server.
        
        Returns:
            Dictionary with compression performance stats
        """
        if not self.connected:
            logger.error("Not connected to scanner")
            return {}
        
        # This would need a new message type in protocol.py
        # For now, include in system status
        try:
            status = self.get_server_status()
            return status.get("compression_stats", {})
        except Exception as e:
            logger.error(f"Error getting compression stats: {e}")
            return {}
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about server preprocessing capabilities.
        
        Returns:
            Dictionary with preprocessing configuration and performance
        """
        if not self.connected:
            logger.error("Not connected to scanner")
            return {}
        
        try:
            status = self.get_server_status()
            optimization_settings = status.get("optimization_settings", {})
            hardware_status = status.get("hardware_status", {})
            return {
                "enabled": optimization_settings.get("preprocessing_enabled", False),
                "level": optimization_settings.get("preprocessing_level", "none"),
                "gpu_available": hardware_status.get("gpu_preprocessing_available", False),
                "performance_metrics": status.get("preprocessing_metrics", {})
            }
        except Exception as e:
            logger.error(f"Error getting preprocessing info: {e}")
            return {}
    
    def is_protocol_v2_enabled(self) -> bool:
        """
        Check if the server has protocol v2 optimization enabled.
        
        Returns:
            True if protocol v2 is active
        """
        try:
            status = self.get_server_status()
            optimization_settings = status.get("optimization_settings", {})
            return optimization_settings.get("protocol_v2_enabled", False)
        except Exception as e:
            logger.warning(f"Could not check protocol v2 status: {e}")
            return False


class FocusAssessment:
    """
    Real-time focus assessment for cameras and projector patterns.
    Analyzes image sharpness and pattern clarity.
    """
    
    def __init__(self):
        self.focus_history = {'left': [], 'right': [], 'projector': []}
        self.history_length = 10
        
        # Focus thresholds (adjustable based on your cameras)
        self.camera_focus_threshold = 100.0  # Laplacian variance threshold
        self.projector_focus_threshold = 0.15  # Pattern clarity threshold
        self.good_focus_threshold = 150.0  # Higher threshold for "good" focus
        
    def assess_camera_focus(self, image: np.ndarray, camera_id: str) -> Dict[str, Any]:
        """
        Assess camera focus using multiple metrics.
        
        Args:
            image: Input image (grayscale or color)
            camera_id: 'left' or 'right'
            
        Returns:
            Dictionary with focus metrics and status
        """
        if image is None or image.size == 0:
            return {'focus_score': 0, 'status': 'no_image', 'quality': 'poor'}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 1. Laplacian variance (main focus metric)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # 2. Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        # 3. High frequency content
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Calculate high frequency energy (outer 30% of spectrum)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        inner_h, inner_w = int(h * 0.35), int(w * 0.35)
        
        # Create mask for high frequencies
        y, x = np.ogrid[:h, :w]
        mask = ((y - center_h)**2 + (x - center_w)**2) > (inner_h * inner_w)
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        # Combined focus score (weighted average)
        focus_score = (
            laplacian_var * 0.5 +
            gradient_mean * 0.3 +
            high_freq_ratio * 1000 * 0.2  # Scale high freq ratio
        )
        
        # Update history
        if camera_id in self.focus_history:
            self.focus_history[camera_id].append(focus_score)
            if len(self.focus_history[camera_id]) > self.history_length:
                self.focus_history[camera_id].pop(0)
        
        # Calculate smoothed score
        smoothed_score = np.mean(self.focus_history[camera_id]) if self.focus_history[camera_id] else focus_score
        
        # Determine status
        if smoothed_score > self.good_focus_threshold:
            status = 'excellent'
            quality = 'excellent'
        elif smoothed_score > self.camera_focus_threshold:
            status = 'good'
            quality = 'good'
        elif smoothed_score > self.camera_focus_threshold * 0.5:
            status = 'fair'
            quality = 'fair'
        else:
            status = 'poor'
            quality = 'poor'
            
        return {
            'focus_score': focus_score,
            'smoothed_score': smoothed_score,
            'laplacian_var': laplacian_var,
            'gradient_mean': gradient_mean,
            'high_freq_ratio': high_freq_ratio,
            'status': status,
            'quality': quality,
            'threshold': self.camera_focus_threshold
        }
    
    def assess_projector_focus(self, image: np.ndarray, pattern_type: str = 'lines') -> Dict[str, Any]:
        """
        Assess projector focus by analyzing projected pattern clarity.
        
        Args:
            image: Image with projected pattern
            pattern_type: Type of pattern ('lines', 'grid', 'checkerboard')
            
        Returns:
            Dictionary with projector focus metrics
        """
        if image is None or image.size == 0:
            return {'focus_score': 0, 'status': 'no_image', 'quality': 'poor'}
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # For line patterns, look for clear horizontal/vertical edges
        if pattern_type == 'lines':
            # Detect horizontal and vertical edges
            edges_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            
            # Calculate edge strength
            edge_strength = np.sqrt(edges_h**2 + edges_v**2)
            
            # Look for periodic patterns (lines)
            # Use FFT to detect regularity
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Find peaks in frequency domain (indicates regular patterns)
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Look for strong peaks away from DC component
            peaks_mask = magnitude_spectrum > (np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum))
            peaks_mask[center_h-5:center_h+5, center_w-5:center_w+5] = False  # Exclude DC
            
            pattern_regularity = np.sum(magnitude_spectrum[peaks_mask]) / np.sum(magnitude_spectrum)
            
            # Calculate contrast (important for projector focus)
            min_val, max_val = np.min(gray), np.max(gray)
            contrast = (max_val - min_val) / (max_val + min_val + 1e-8)
            
            # Combined projector focus score
            focus_score = (
                np.mean(edge_strength) * 0.4 +
                pattern_regularity * 1000 * 0.3 +  # Scale pattern regularity
                contrast * 100 * 0.3  # Scale contrast
            )
            
        else:
            # Generic pattern analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Standard deviation (pattern clarity)
            pattern_clarity = np.std(gray)
            
            focus_score = edge_density * 100 + pattern_clarity * 0.5
        
        # Update history
        self.focus_history['projector'].append(focus_score)
        if len(self.focus_history['projector']) > self.history_length:
            self.focus_history['projector'].pop(0)
            
        # Calculate smoothed score
        smoothed_score = np.mean(self.focus_history['projector']) if self.focus_history['projector'] else focus_score
        
        # Determine status for projector
        if smoothed_score > self.projector_focus_threshold * 2:
            status = 'excellent'
            quality = 'excellent'
        elif smoothed_score > self.projector_focus_threshold:
            status = 'good'
            quality = 'good'
        elif smoothed_score > self.projector_focus_threshold * 0.5:
            status = 'fair'
            quality = 'fair'
        else:
            status = 'poor'
            quality = 'poor'
            
        return {
            'focus_score': focus_score,
            'smoothed_score': smoothed_score,
            'status': status,
            'quality': quality,
            'threshold': self.projector_focus_threshold,
            'contrast': locals().get('contrast', 0),
            'pattern_regularity': locals().get('pattern_regularity', 0)
        }
    
    def get_overall_focus_status(self) -> Dict[str, Any]:
        """
        Get overall focus status for the entire system.
        
        Returns:
            Dictionary with system-wide focus status
        """
        # Get latest scores
        left_score = self.focus_history['left'][-1] if self.focus_history['left'] else 0
        right_score = self.focus_history['right'][-1] if self.focus_history['right'] else 0
        proj_score = self.focus_history['projector'][-1] if self.focus_history['projector'] else 0
        
        # Determine overall status
        all_good = (
            left_score > self.camera_focus_threshold and
            right_score > self.camera_focus_threshold and
            proj_score > self.projector_focus_threshold
        )
        
        any_poor = (
            left_score < self.camera_focus_threshold * 0.5 or
            right_score < self.camera_focus_threshold * 0.5 or
            proj_score < self.projector_focus_threshold * 0.5
        )
        
        if all_good:
            overall_status = 'ready'
            message = "✅ All components in focus - Ready to scan!"
        elif any_poor:
            overall_status = 'poor'
            message = "❌ Focus adjustment needed"
        else:
            overall_status = 'adjusting'
            message = "⚠️ Fine-tuning focus..."
            
        return {
            'status': overall_status,
            'message': message,
            'left_score': left_score,
            'right_score': right_score,
            'projector_score': proj_score,
            'all_ready': all_good
        }