"""
Module for managing cameras using PiCamera2.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any

import numpy as np

# Import picamera2 only if available (on Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logger = logging.getLogger(__name__)


class PiCamera2Manager:
    """
    PiCamera2 manager.
    Provides a unified interface for managing multiple cameras.
    """

    def __init__(self):
        """Initialize camera manager."""
        if not PICAMERA2_AVAILABLE:
            logger.warning("PiCamera2 not available. Limited functionality.")

        self.cameras = {}  # Dict[camera_id, camera_info]
        self.active_cameras = {}  # Dict[camera_id, Picamera2]
        self._lock = threading.RLock()

        # Detect available cameras
        self._discover_cameras()

    def _discover_cameras(self):
        """Detect available cameras."""
        if not PICAMERA2_AVAILABLE:
            logger.warning("PiCamera2 not available, cannot detect cameras.")
            return

        with self._lock:
            try:
                # Get list of available cameras
                num_cameras = len(Picamera2.global_camera_info())
                logger.info(f"Found {num_cameras} cameras")

                for i in range(num_cameras):
                    try:
                        # Create a temporary Picamera2 to get information
                        cam = Picamera2(i)

                        # Get capabilities and information
                        capabilities = cam.camera_properties

                        # Extract useful information
                        camera_info = {
                            "name": f"Camera {i}",
                            "index": i,
                            "model": capabilities.get("Model", "Unknown"),
                            "resolution": capabilities.get("MaxResolution", [1920, 1080]),
                            "fps": 30,  # Default FPS
                            "capabilities": ["preview", "still", "video"],
                            "raw_capabilities": capabilities
                        }

                        # Add to camera list
                        camera_id = f"picamera2_{i}"
                        self.cameras[camera_id] = camera_info
                        logger.info(f"Camera detected: {camera_id} - {camera_info['name']}")

                        # Close camera
                        cam.close()

                    except Exception as e:
                        logger.error(f"Error detecting camera {i}: {e}")

            except Exception as e:
                logger.error(f"Error detecting cameras: {e}")

    def get_cameras(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available cameras.

        Returns:
            Dictionary of cameras {camera_id: camera_info}
        """
        with self._lock:
            return self.cameras.copy()

    def open_camera(self, camera_id: str) -> bool:
        """
        Open a camera.

        Args:
            camera_id: Camera ID

        Returns:
            True if successful, False otherwise
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 not available, cannot open camera.")
            return False

        with self._lock:
            # Check if camera exists
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False

            # Check if camera is already open
            if camera_id in self.active_cameras:
                logger.debug(f"Camera {camera_id} already open")
                return True

            try:
                # Extract camera index
                camera_info = self.cameras[camera_id]
                camera_index = camera_info["index"]

                # Open camera
                camera = Picamera2(camera_index)

                # Configure camera
                config = camera.create_still_configuration()
                camera.configure(config)

                # Start camera
                camera.start()

                # Add to active cameras list
                self.active_cameras[camera_id] = camera

                logger.info(f"Camera {camera_id} successfully opened")
                return True

            except Exception as e:
                logger.error(f"Error opening camera {camera_id}: {e}")
                return False

    def close_camera(self, camera_id: str) -> bool:
        """
        Close a camera.

        Args:
            camera_id: Camera ID

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Check if camera is open
            if camera_id not in self.active_cameras:
                logger.debug(f"Camera {camera_id} is not open")
                return True

            try:
                # Close camera
                camera = self.active_cameras[camera_id]
                camera.stop()
                camera.close()

                # Remove from active cameras list
                del self.active_cameras[camera_id]

                logger.info(f"Camera {camera_id} successfully closed")
                return True

            except Exception as e:
                logger.error(f"Error closing camera {camera_id}: {e}")
                return False

    def configure_camera(self, camera_id: str, config: Dict[str, Any]) -> bool:
        """
        Configure a camera with support for advanced options.

        Args:
            camera_id: Camera ID
            config: Camera configuration, can include:
                - resolution: [width, height]
                - fps: Frame rate
                - mode: Acquisition mode ("video", "preview", "still")
                - exposure: Exposure time in microseconds
                - gain: Analog gain
                - awb: White balance ("auto" or "manual")
                - awb_gains: White balance gains [red_gain, blue_gain]
                - color_mode: Color mode ("rgb", "bgr", "grayscale")
                - roi: Region of interest [x, y, width, height]
                - hflip: Flip horizontally (True/False)
                - vflip: Flip vertically (True/False)

        Returns:
            True if successful, False otherwise
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 not available, cannot configure camera.")
            return False

        with self._lock:
            # Check if camera exists
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False

            # Open camera if not already open
            if camera_id not in self.active_cameras:
                if not self.open_camera(camera_id):
                    return False

            camera = self.active_cameras[camera_id]

            try:
                # Configuration required for resolution or mode change
                needs_reconfigure = "resolution" in config or "mode" in config or "color_mode" in config

                # If reconfiguration is required, stop the camera
                if needs_reconfigure:
                    camera.stop()

                    # Get acquisition mode
                    mode = config.get("mode", "still")  # Default: still

                    # Get resolution
                    width, height = config.get("resolution", self.cameras[camera_id].get("resolution", [1920, 1080]))

                    # Get color mode
                    color_mode = config.get("color_mode", "rgb")

                    # Create appropriate configuration
                    if mode == "video":
                        # Video configuration with support for different color formats
                        if color_mode == "grayscale":
                            camera_config = camera.create_video_configuration(
                                main={"size": (width, height), "format": "YUV420"}
                            )
                        elif color_mode == "bgr":
                            camera_config = camera.create_video_configuration(
                                main={"size": (width, height), "format": "BGR888"}
                            )
                        else:  # default: rgb
                            camera_config = camera.create_video_configuration(
                                main={"size": (width, height), "format": "RGB888"}
                            )
                    elif mode == "preview":
                        # Preview configuration with support for different color formats
                        if color_mode == "grayscale":
                            camera_config = camera.create_preview_configuration(
                                main={"size": (width, height), "format": "YUV420"}
                            )
                        elif color_mode == "bgr":
                            camera_config = camera.create_preview_configuration(
                                main={"size": (width, height), "format": "BGR888"}
                            )
                        else:  # default: rgb
                            camera_config = camera.create_preview_configuration(
                                main={"size": (width, height), "format": "RGB888"}
                            )
                    else:  # default: still
                        # Still configuration with support for different color formats
                        if color_mode == "grayscale":
                            camera_config = camera.create_still_configuration(
                                main={"size": (width, height), "format": "YUV420"}
                            )
                        elif color_mode == "bgr":
                            camera_config = camera.create_still_configuration(
                                main={"size": (width, height), "format": "BGR888"}
                            )
                        else:  # default: rgb
                            camera_config = camera.create_still_configuration(
                                main={"size": (width, height), "format": "RGB888"}
                            )

                    # Apply configuration
                    camera.configure(camera_config)

                # Build a dictionary of controls to set
                controls = {}

                # Exposure
                if "exposure" in config:
                    controls["ExposureTime"] = config["exposure"]
                    if "exposure_mode" in config and config["exposure_mode"] == "manual":
                        controls["AeEnable"] = False
                    else:
                        controls["AeEnable"] = True

                # Gain
                if "gain" in config:
                    controls["AnalogueGain"] = config["gain"]

                # White balance
                if "awb" in config:
                    if config["awb"] == "auto":
                        controls["AwbEnable"] = True
                    else:
                        controls["AwbEnable"] = False
                        if "awb_gains" in config:
                            controls["ColourGains"] = tuple(config["awb_gains"])

                # Region of interest
                if "roi" in config:
                    x, y, w, h = config["roi"]
                    controls["ScalerCrop"] = (x, y, w, h)

                # Flipping
                if "hflip" in config:
                    controls["HFlip"] = config["hflip"]
                if "vflip" in config:
                    controls["VFlip"] = config["vflip"]

                # Set controls
                if controls:
                    camera.set_controls(controls)

                # Start camera if it was stopped
                if needs_reconfigure:
                    camera.start()

                # Update camera information
                updated_info = self.cameras[camera_id].copy()

                if "resolution" in config:
                    updated_info["configured_resolution"] = config["resolution"]

                if "fps" in config:
                    updated_info["configured_fps"] = config["fps"]

                if "color_mode" in config:
                    updated_info["color_mode"] = config["color_mode"]

                # Update cache
                self.cameras[camera_id] = updated_info

                logger.info(f"Camera {camera_id} successfully configured")
                return True

            except Exception as e:
                logger.error(f"Error configuring camera {camera_id}: {e}")
                return False

    def capture_image(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Capture an image from a camera.

        Args:
            camera_id: Camera ID

        Returns:
            Image as numpy array, None if error
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 not available, cannot capture image.")
            return None

        with self._lock:
            # Check if camera exists
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return None

            # Open camera if not already open
            if camera_id not in self.active_cameras:
                if not self.open_camera(camera_id):
                    return None

            camera = self.active_cameras[camera_id]

            try:
                # Capture image
                image = camera.capture_array()

                # Check image format
                color_mode = self.cameras[camera_id].get("color_mode", "rgb")

                # Convert image to correct format if necessary
                if len(image.shape) == 3:
                    if color_mode == "grayscale" and image.shape[2] > 1:
                        # Convert to grayscale if requested
                        import cv2
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    elif color_mode == "rgb" and image.shape[2] == 4:  # RGBA
                        # Remove alpha channel
                        image = image[:, :, :3]
                        # Convert from BGR to RGB if necessary
                        if image.shape[2] == 3 and np.array_equal(image[0, 0], image[0, 0, ::-1]):
                            import cv2
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif color_mode == "bgr" and image.shape[2] == 3:
                        # Make sure it's in BGR format
                        if not np.array_equal(image[0, 0], image[0, 0, ::-1]):
                            import cv2
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                return image

            except Exception as e:
                logger.error(f"Error capturing image from camera {camera_id}: {e}")
                return None

    def close(self):
        """Close all active cameras."""
        with self._lock:
            for camera_id in list(self.active_cameras.keys()):
                self.close_camera(camera_id)

            self.active_cameras.clear()
            logger.info("All cameras closed")