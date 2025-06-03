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

# Import camera sync module
try:
    from .camera_sync import get_camera_sync, SyncMetrics
    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False
    SyncMetrics = None

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
        
        # Initialize hardware sync if available
        self.sync_controller = None
        if SYNC_AVAILABLE:
            try:
                self.sync_controller = get_camera_sync()
                logger.info("Hardware camera sync initialized")
            except Exception as e:
                logger.warning(f"Could not initialize hardware sync: {e}")

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
                            "resolution": capabilities.get("MaxResolution", [2048, 1536]),  # 2K default
                            "fps": 15,  # Reduced FPS for 2K to maintain stability
                            "capabilities": ["preview", "still", "video"],
                            "raw_capabilities": capabilities,
                            "preferred_resolution": [2048, 1536],  # 2K for high quality
                            "fallback_resolution": [1920, 1080]   # Fallback if 2K not supported
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
                - exposure_time: Exposure time in microseconds
                - auto_exposure: Whether to use auto exposure (True/False)
                - exposure_compensation: EV compensation (-4.0 to 4.0)
                - gain: Analog gain value
                - auto_gain: Whether to use auto gain (True/False)
                - iso: ISO setting (camera dependent)
                - color_mode: Color mode ("color", "grayscale")
                - compression_format: Image format ("jpeg", "png", "raw")
                - jpeg_quality: JPEG compression quality (0-100)
                - crop_region: ROI crop (x, y, width, height)
                - sharpness: Sharpness adjustment (0=default, 1=max)
                - denoise: Enable noise reduction (True/False)
                - hdr_mode: Enable HDR mode (True/False)
                - stabilization: Enable image stabilization (True/False)
                - binning: Pixel binning factor (1, 2, 4)
                - framerate: Target framerate in FPS
                - contrast: Contrast enhancement factor
                - brightness: Brightness adjustment (-1.0 to 1.0)
                - saturation: Color saturation (grayscale=0, normal=1.0)
                - gamma: Gamma correction (1.0=linear)
                - hflip: Flip horizontally (True/False)
                - vflip: Flip vertically (True/False)
                - awb: White balance mode ("auto" or "manual")
                - awb_gains: White balance gains [red_gain, blue_gain]
                - backend_settings: Custom settings for specific backends

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
                # Log configuration for debugging
                logger.info(f"Configuring camera {camera_id} with settings: {config}")
                
                # Handle configuration requiring camera restart
                needs_reconfigure = any(key in config for key in [
                    "resolution", "mode", "color_mode", "compression_format", "crop_region"
                ])

                # If reconfiguration is required, stop the camera
                if needs_reconfigure:
                    camera.stop()

                    # Get acquisition mode
                    mode = config.get("mode", "still")  # Default: still

                    # Get resolution with better error handling
                    resolution = config.get("resolution")
                    if resolution is None:
                        # Try to get from camera info
                        resolution = self.cameras[camera_id].get("resolution")
                    
                    # Default to 1920x1080 if neither is available
                    if resolution is None or not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
                        resolution = [1920, 1080]
                        
                    width, height = resolution

                    # Get color mode - support for both strings and EnumNamespace.ColorMode values
                    color_mode_raw = config.get("color_mode", "color")
                    # Handle if color_mode is an enum value object with .value attribute
                    color_mode = color_mode_raw.lower() if isinstance(color_mode_raw, str) else str(color_mode_raw).lower()
                    
                    # Map client color modes to PiCamera color formats
                    if color_mode == "grayscale":
                        format_name = "YUV420"
                    elif color_mode == "bgr" or color_mode == "color.bgr":
                        format_name = "BGR888"
                    else:  # default: rgb/color
                        format_name = "RGB888"

                    # Create appropriate configuration based on mode
                    if mode == "video":
                        camera_config = camera.create_video_configuration(
                            main={"size": (width, height), "format": format_name}
                        )
                    elif mode == "preview":
                        camera_config = camera.create_preview_configuration(
                            main={"size": (width, height), "format": format_name}
                        )
                    else:  # default: still
                        camera_config = camera.create_still_configuration(
                            main={"size": (width, height), "format": format_name}
                        )

                    # Apply configuration
                    camera.configure(camera_config)

                # Build a dictionary of camera controls to set
                controls = {}

                # Exposure settings
                if "exposure_time" in config:
                    controls["ExposureTime"] = int(config["exposure_time"])
                
                if "auto_exposure" in config:
                    controls["AeEnable"] = bool(config["auto_exposure"])
                elif "exposure_mode" in config:  # Legacy support
                    controls["AeEnable"] = config["exposure_mode"] != "manual"
                
                if "exposure_compensation" in config:
                    controls["ExposureValue"] = float(config["exposure_compensation"])

                # Gain settings
                if "gain" in config:
                    controls["AnalogueGain"] = float(config["gain"])
                
                if "auto_gain" in config:
                    # Some cameras may have separate gain control
                    # This might need to be mapped to a camera-specific control
                    pass  # PiCamera doesn't have separate auto gain control

                # Image adjustments
                if "brightness" in config:
                    controls["Brightness"] = float(config["brightness"])
                
                if "contrast" in config:
                    controls["Contrast"] = float(config["contrast"])
                
                if "saturation" in config:
                    controls["Saturation"] = float(config["saturation"])
                
                if "sharpness" in config:
                    controls["Sharpness"] = float(config["sharpness"])

                # Denoise and HDR
                if "denoise" in config:
                    # Map to camera-specific noise reduction mode if available
                    noiseReductionMode = 2 if config["denoise"] else 0  # Example mapping
                    controls["NoiseReductionMode"] = noiseReductionMode
                
                if "hdr_mode" in config:
                    # Map to camera-specific HDR mode if available
                    # This is highly camera-dependent
                    pass

                # White balance
                if "awb" in config:
                    if config["awb"] == "auto":
                        controls["AwbEnable"] = True
                    else:
                        controls["AwbEnable"] = False
                        if "awb_gains" in config:
                            controls["ColourGains"] = tuple(config["awb_gains"])

                # Region of interest / Crop with better error handling
                if "crop_region" in config:
                    crop_region = config["crop_region"]
                    try:
                        if isinstance(crop_region, (list, tuple)) and len(crop_region) == 4:
                            x, y, w, h = crop_region
                            controls["ScalerCrop"] = (int(x), int(y), int(w), int(h))
                            logger.info(f"Setting crop region to {controls['ScalerCrop']}")
                        else:
                            logger.warning(f"Invalid crop_region format, expected list/tuple of 4 elements, got: {crop_region}")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error setting crop_region: {e}")
                        
                elif "roi" in config:  # Legacy support
                    roi = config["roi"]
                    try:
                        if isinstance(roi, (list, tuple)) and len(roi) == 4:
                            x, y, w, h = roi
                            controls["ScalerCrop"] = (int(x), int(y), int(w), int(h))
                        else:
                            logger.warning(f"Invalid roi format, expected list/tuple of 4 elements, got: {roi}")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error setting roi: {e}")

                # Flipping
                if "hflip" in config:
                    controls["HFlip"] = bool(config["hflip"])
                if "vflip" in config:
                    controls["VFlip"] = bool(config["vflip"])

                # Apply all controls
                if controls:
                    logger.info(f"Setting camera controls: {controls}")
                    try:
                        camera.set_controls(controls)
                        logger.info(f"Successfully applied controls to camera {camera_id}")
                    except Exception as e:
                        logger.error(f"Failed to apply controls to camera {camera_id}: {e}")
                        raise

                # Start camera if it was stopped
                if needs_reconfigure:
                    camera.start()

                # Update camera information in our cache
                updated_info = self.cameras[camera_id].copy()

                # Update resolution info
                if "resolution" in config:
                    updated_info["configured_resolution"] = config["resolution"]

                # Update framerate info
                if "framerate" in config or "fps" in config:
                    fps = config.get("framerate", config.get("fps", 30))
                    updated_info["configured_fps"] = fps

                # Update color mode
                if "color_mode" in config:
                    updated_info["color_mode"] = color_mode

                # Update compression format
                if "compression_format" in config:
                    updated_info["compression_format"] = config["compression_format"]

                # Update jpeg quality
                if "jpeg_quality" in config:
                    updated_info["jpeg_quality"] = config["jpeg_quality"]

                # Update cache
                self.cameras[camera_id] = updated_info

                logger.info(f"Camera {camera_id} successfully configured")
                return True

            except Exception as e:
                logger.error(f"Error configuring camera {camera_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
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
                logger.info(f"Opening camera {camera_id} for capture")
                if not self.open_camera(camera_id):
                    logger.error(f"Failed to open camera {camera_id}")
                    return None

            camera = self.active_cameras[camera_id]

            try:
                # Check if camera is still valid
                if camera is None:
                    logger.error(f"Camera {camera_id} is None in active_cameras")
                    return None
                    
                # Check camera state
                try:
                    # Try different ways to check if camera is open based on API version
                    camera_open = False
                    
                    # Check if is_open is an attribute
                    if hasattr(camera, 'is_open'):
                        # Check if it's a property or method
                        if callable(camera.is_open):
                            camera_open = camera.is_open()
                        else:
                            camera_open = camera.is_open
                    # Otherwise check if camera is None or has stop method
                    elif camera and hasattr(camera, 'stop'):
                        camera_open = True
                    
                    if not camera_open:
                        logger.warning(f"Camera {camera_id} was closed, reopening")
                        self.close_camera(camera_id)
                        if not self.open_camera(camera_id):
                            logger.error(f"Failed to reopen camera {camera_id}")
                            return None
                        camera = self.active_cameras[camera_id]
                except Exception as check_error:
                    logger.warning(f"Error checking camera state: {check_error}, assuming camera needs reopening")
                    try:
                        self.close_camera(camera_id)
                        if not self.open_camera(camera_id):
                            logger.error(f"Failed to reopen camera {camera_id}")
                            return None
                        camera = self.active_cameras[camera_id]
                    except Exception as reopen_error:
                        logger.error(f"Failed to reopen camera after state check error: {reopen_error}")
                        return None
                
                # Capture image with timeout handling
                logger.debug(f"Capturing array from camera {camera_id}")
                try:
                    # Use a shorter timeout for direct streaming
                    image = camera.capture_array()
                except Exception as capture_error:
                    logger.error(f"Error in camera.capture_array() for {camera_id}: {capture_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None

                # Verify image is valid
                if image is None:
                    logger.error(f"Capture from camera {camera_id} returned None")
                    return None
                    
                if not isinstance(image, np.ndarray):
                    logger.error(f"Capture from camera {camera_id} returned non-ndarray: {type(image)}")
                    return None
                    
                if image.size == 0:
                    logger.error(f"Capture from camera {camera_id} returned empty array")
                    return None

                # Check image format
                color_mode = self.cameras[camera_id].get("color_mode", "rgb")
                logger.debug(f"Image shape: {image.shape}, color mode: {color_mode}")

                # Convert image to correct format if necessary
                try:
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
                except Exception as conversion_error:
                    logger.error(f"Error converting image format for camera {camera_id}: {conversion_error}")
                    # Return original image if conversion fails
                    return image

                return image

            except Exception as e:
                logger.error(f"Error capturing image from camera {camera_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None

    def capture_test_image(self, camera_id: str, test_type: str = "normal") -> Optional[np.ndarray]:
        """
        Capture a test image for auto-optimization.
        
        Args:
            camera_id: Camera ID
            test_type: Type of test image ("underexposed", "normal", "overexposed")
            
        Returns:
            Test image as numpy array, None if error
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 not available, cannot capture test image.")
            return None
            
        with self._lock:
            if camera_id not in self.active_cameras:
                logger.error(f"Camera {camera_id} not available")
                return None
                
            camera = self.active_cameras[camera_id]
            
            try:
                # Store current settings
                original_controls = camera.capture_metadata()
                
                # Set test parameters based on type
                test_controls = {}
                if test_type == "underexposed":
                    test_controls["ExposureTime"] = 5000  # 5ms
                    test_controls["AnalogueGain"] = 1.0
                elif test_type == "overexposed":
                    test_controls["ExposureTime"] = 50000  # 50ms
                    test_controls["AnalogueGain"] = 8.0
                else:  # normal
                    test_controls["ExposureTime"] = 20000  # 20ms
                    test_controls["AnalogueGain"] = 2.0
                
                # Apply test settings
                camera.set_controls(test_controls)
                
                # Wait for settings to take effect
                time.sleep(0.5)
                
                # Capture test image
                image = self.capture_image(camera_id)
                
                # Restore original settings
                restore_controls = {
                    "ExposureTime": original_controls.get("ExposureTime", 20000),
                    "AnalogueGain": original_controls.get("AnalogueGain", 1.0)
                }
                camera.set_controls(restore_controls)
                
                return image
                
            except Exception as e:
                logger.error(f"Error capturing test image: {e}")
                return None
    
    def optimize_camera_settings(self, camera_id: str, target_brightness: float = 0.5, 
                               target_contrast: float = 0.3) -> Dict[str, Any]:
        """
        Automatically optimize camera settings for pattern visibility.
        
        Args:
            camera_id: Camera ID
            target_brightness: Target mean brightness (0-1)
            target_contrast: Target contrast level (0-1)
            
        Returns:
            Optimized settings dictionary
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 not available, cannot optimize settings.")
            return {}
            
        logger.info(f"Starting camera optimization for {camera_id}")
        
        # Capture test images
        test_types = ["underexposed", "normal", "overexposed"]
        test_images = {}
        
        for test_type in test_types:
            image = self.capture_test_image(camera_id, test_type)
            if image is not None:
                test_images[test_type] = image
        
        if len(test_images) < 3:
            logger.error("Failed to capture enough test images for optimization")
            return {}
        
        # Analyze images
        best_settings = None
        best_score = float('inf')
        
        # Try different exposure and gain combinations
        exposure_times = [5000, 10000, 15000, 20000, 30000, 40000]
        gains = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        
        for exposure in exposure_times:
            for gain in gains:
                # Test these settings
                test_config = {
                    "exposure_time": exposure,
                    "gain": gain,
                    "auto_exposure": False,
                    "auto_gain": False
                }
                
                # Configure and capture
                self.configure_camera(camera_id, test_config)
                time.sleep(0.3)  # Let settings stabilize
                
                image = self.capture_image(camera_id)
                if image is None:
                    continue
                
                # Convert to grayscale for analysis
                if len(image.shape) == 3:
                    import cv2
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Calculate metrics
                mean_brightness = np.mean(gray) / 255.0
                std_contrast = np.std(gray) / 255.0
                
                # Score based on distance from targets
                brightness_error = abs(mean_brightness - target_brightness)
                contrast_error = abs(std_contrast - target_contrast)
                score = brightness_error + contrast_error
                
                logger.debug(f"Exposure={exposure}, Gain={gain}, Brightness={mean_brightness:.3f}, "
                           f"Contrast={std_contrast:.3f}, Score={score:.3f}")
                
                if score < best_score:
                    best_score = score
                    best_settings = {
                        "exposure_time": exposure,
                        "gain": gain,
                        "auto_exposure": False,
                        "auto_gain": False,
                        "brightness": mean_brightness,
                        "contrast": std_contrast
                    }
        
        if best_settings:
            logger.info(f"Optimization complete. Best settings: {best_settings}")
            # Apply best settings
            self.configure_camera(camera_id, best_settings)
            return best_settings
        else:
            logger.error("Failed to find optimal settings")
            return {}
    
    def auto_focus(self, camera_id: str, focus_region: Optional[List[int]] = None) -> bool:
        """
        Perform auto-focus operation.
        
        Args:
            camera_id: Camera ID
            focus_region: Optional region of interest for focus [x, y, width, height]
            
        Returns:
            True if successful, False otherwise
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("PiCamera2 not available, cannot perform auto-focus.")
            return False
            
        with self._lock:
            if camera_id not in self.active_cameras:
                logger.error(f"Camera {camera_id} not available")
                return False
                
            camera = self.active_cameras[camera_id]
            
            try:
                # Check if camera supports autofocus
                controls = camera.camera_controls
                if "AfMode" not in controls:
                    logger.warning(f"Camera {camera_id} does not support autofocus")
                    return False
                
                # Set focus region if provided
                if focus_region and len(focus_region) == 4:
                    x, y, w, h = focus_region
                    # Set the AF window for focus
                    camera.set_controls({
                        "AfWindows": [(x, y, w, h)],
                        "AfMode": 1  # Auto mode
                    })
                else:
                    # Use auto mode without specific region
                    camera.set_controls({"AfMode": 1})
                
                # Trigger autofocus
                camera.set_controls({"AfTrigger": 0})  # Start AF
                
                # Wait for focus to complete (with timeout)
                start_time = time.time()
                timeout = 5.0  # 5 second timeout
                
                while time.time() - start_time < timeout:
                    metadata = camera.capture_metadata()
                    af_state = metadata.get("AfState", 0)
                    
                    # Check if focus is complete (state depends on camera)
                    if af_state == 2:  # Focused
                        logger.info(f"Autofocus successful for camera {camera_id}")
                        return True
                    elif af_state == 0:  # Idle
                        time.sleep(0.1)
                        continue
                    elif af_state == 1:  # Scanning
                        time.sleep(0.1)
                        continue
                    else:  # Failed or unknown state
                        break
                
                logger.warning(f"Autofocus timeout or failed for camera {camera_id}")
                return False
                
            except Exception as e:
                logger.error(f"Error during autofocus: {e}")
                return False

    def close(self):
        """Close all active cameras."""
        with self._lock:
            for camera_id in list(self.active_cameras.keys()):
                self.close_camera(camera_id)

            self.active_cameras.clear()
            logger.info("All cameras closed")
    
    def capture_synchronized(self, camera_ids: List[str], timeout: float = 1.0) -> Dict[str, Any]:
        """
        Capture synchronized images from multiple cameras.
        
        Args:
            camera_ids: List of camera IDs to capture from
            timeout: Maximum time to wait for sync trigger
            
        Returns:
            Dictionary with camera_id -> capture result mapping
        """
        import queue
        
        if not self.sync_controller:
            logger.warning("No sync controller, falling back to sequential capture")
            results = {}
            for cam_id in camera_ids:
                image = self.capture_image(cam_id)
                results[cam_id] = {
                    'image': image,
                    'timestamp': time.time(),
                    'synchronized': False
                }
            return results
        
        results = {}
        result_queue = queue.Queue()
        
        def capture_on_trigger(cam_id: str, trigger_time: float):
            """Capture callback for synchronized acquisition."""
            try:
                # Capture with minimal latency
                image = self.capture_image(cam_id)
                if image is not None:
                    result = {
                        'image': image,
                        'trigger_timestamp': trigger_time,
                        'capture_timestamp': time.time(),
                        'capture_latency_us': (time.time() - trigger_time) * 1_000_000,
                        'synchronized': True
                    }
                else:
                    result = None
                result_queue.put((cam_id, result))
            except Exception as e:
                logger.error(f"Error in synchronized capture for {cam_id}: {e}")
                result_queue.put((cam_id, None))
        
        # Register callbacks for all cameras
        callbacks = []
        for cam_id in camera_ids:
            callback = lambda t, cid=cam_id: capture_on_trigger(cid, t)
            self.sync_controller.register_callback(callback)
            callbacks.append(callback)
        
        # Wait for trigger
        trigger_time = self.sync_controller.wait_for_trigger(timeout)
        
        if trigger_time:
            # Collect results
            for _ in camera_ids:
                try:
                    cam_id, result = result_queue.get(timeout=0.5)
                    results[cam_id] = result
                except:
                    pass
        
        # Cleanup callbacks
        for callback in callbacks:
            try:
                self.sync_controller.callbacks.remove(callback)
            except:
                pass
        
        return results
    
    def get_sync_metrics(self) -> Optional[SyncMetrics]:
        """
        Get synchronization quality metrics.
        
        Returns:
            SyncMetrics object or None if sync not available
        """
        if self.sync_controller:
            return self.sync_controller.get_sync_metrics()
        return None
    
    def enable_software_sync(self, fps: float = 30.0):
        """
        Enable software-based synchronization.
        
        Args:
            fps: Frames per second for sync trigger
        """
        if self.sync_controller:
            self.sync_controller.enable_software_trigger(fps)
            logger.info(f"Software sync enabled at {fps} FPS")
        else:
            logger.warning("No sync controller available")