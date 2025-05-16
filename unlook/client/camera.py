"""
Client for managing UnLook scanner cameras.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

try:
    from ..core.protocol import MessageType
    from ..core.utils import decode_jpeg_to_image, deserialize_binary_message
    from ..core.constants import DEFAULT_JPEG_QUALITY
    from ..core.events import EventType
    from .camera_config import CameraConfig, ColorMode, CompressionFormat, ImageQualityPreset
except ImportError:
    # Define fallback classes for when the core modules are not available
    class MessageType:
        CAMERA_LIST = "camera_list"
        CAMERA_CONFIG = "camera_config"
        CAMERA_CAPTURE = "camera_capture"
        CAMERA_CAPTURE_MULTI = "camera_capture_multi"
    
    class EventType:
        pass
    
    class ColorMode:
        COLOR = "color"
        GRAYSCALE = "grayscale"
    
    class CompressionFormat:
        JPEG = "jpeg"
        PNG = "png"
        RAW = "raw"
    
    class ImageQualityPreset:
        LOWEST = "lowest"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        HIGHEST = "highest"
    
    class CameraConfig:
        def __init__(self):
            pass
    
    # Simple fallback for deserialize_binary_message
    def deserialize_binary_message(data):
        """Simple fallback for deserializing binary messages."""
        import logging
        import struct
        logger = logging.getLogger(__name__)
        
        # Check for ULMC format v1 (multi-camera format)
        if data.startswith(b'ULMC\x01'):
            logger.debug("Detected ULMC format v1")
            
            try:
                # Parse ULMC header
                # Format: 'ULMC' (4 bytes) + version (1 byte) + reserved (3 bytes) + num_cameras (4 bytes)
                if len(data) < 12:
                    logger.error("ULMC data too short for header")
                    return "binary_data", {}, data
                
                header = data[:12]
                num_cameras = struct.unpack(">I", header[8:12])[0]
                logger.debug(f"ULMC header indicates {num_cameras} cameras")
                
                # Parse camera entries
                offset = 12
                cameras = {}
                
                for i in range(num_cameras):
                    if offset + 4 > len(data):
                        logger.error(f"ULMC data too short for camera {i} name length")
                        break
                        
                    # Read camera name length
                    name_len = struct.unpack(">I", data[offset:offset+4])[0]
                    offset += 4
                    
                    if offset + name_len > len(data):
                        logger.error(f"ULMC data too short for camera {i} name")
                        break
                        
                    # Read camera name
                    camera_name = data[offset:offset+name_len].decode('utf-8')
                    offset += name_len
                    
                    if offset + 16 > len(data):
                        logger.error(f"ULMC data too short for camera {i} info")
                        break
                        
                    # Read camera info (timestamp + offset + size)
                    timestamp = struct.unpack(">Q", data[offset:offset+8])[0]
                    img_offset = struct.unpack(">I", data[offset+8:offset+12])[0]
                    img_size = struct.unpack(">I", data[offset+12:offset+16])[0]
                    offset += 16
                    
                    cameras[camera_name] = {
                        "timestamp": timestamp,
                        "offset": img_offset, 
                        "size": img_size
                    }
                    
                    logger.debug(f"Camera {camera_name}: offset={img_offset}, size={img_size}")
                
                payload = {
                    "format": "ULMC",
                    "version": 1,
                    "num_cameras": num_cameras,
                    "cameras": cameras
                }
                
                return "multi_camera_response", payload, data
                
            except Exception as e:
                logger.error(f"Error parsing ULMC format: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return "binary_data", {}, data
        
        # For other formats, return as camera frame
        return "camera_frame", {"format": "jpeg"}, data
    
    DEFAULT_JPEG_QUALITY = 85

logger = logging.getLogger(__name__)


class StereoCamera:
    """
    Class for stereo camera operations in the UnLook scanner.
    """
    
    def __init__(self, camera_ids: List[str] = None):
        """
        Initialize a stereo camera.
        
        Args:
            camera_ids: Optional list of camera IDs to use (left, right)
        """
        self.camera_ids = camera_ids if camera_ids else ["left", "right"]
        self.client = None  # Will be set when connected to a real server

        # Use real hardware mode by default
        self._is_simulation = False
        self.image_size = (1280, 720)
        
        logger.info(f"Initialized stereo camera with IDs: {self.camera_ids}")
    
    def capture_stereo_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture a synchronized image from the stereo pair.

        Returns:
            Tuple (left_image, right_image)
        """
        # If we have a real client, use it to capture
        if self.client:
            return self.client.capture_stereo_pair()

        # If we don't have a client, this is an error - we want to use real hardware only
        logger.error("No camera client available. Real hardware is required.")
        # As a fallback for development, we'll still generate test images
        # but we should treat this as an error case
        return self._generate_simulated_images()
    
    def _generate_simulated_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated stereo images for testing.
        
        Returns:
            Tuple (left_image, right_image)
        """
        # Create a simulated checkerboard pattern
        height, width = self.image_size[1], self.image_size[0]
        
        # Generate a simple pattern
        pattern = np.zeros((height, width), dtype=np.uint8)
        square_size = 50
        
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    pattern[i:i+square_size, j:j+square_size] = 255
        
        # Convert to RGB
        left_img = np.stack([pattern, pattern, pattern], axis=2)
        
        # Create right image with slight offset
        offset = 10  # Simulated disparity
        right_img = np.zeros_like(left_img)
        right_img[:, :-offset] = left_img[:, offset:]
        
        return left_img, right_img
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set the resolution for both cameras.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            True if successful, False otherwise
        """
        self.image_size = (width, height)
        
        # If we have a real client, configure both cameras
        if self.client and not self._is_simulation:
            left_id, right_id = self.camera_ids
            success_left = self.client.configure(left_id, {"resolution": (width, height)})
            success_right = self.client.configure(right_id, {"resolution": (width, height)})
            return success_left and success_right
        
        return True  # Simulated success
    
    def set_exposure(self, exposure_time: int, gain: float = 1.0) -> bool:
        """
        Set exposure settings for both cameras.
        
        Args:
            exposure_time: Exposure time in microseconds
            gain: Camera gain value
            
        Returns:
            True if successful, False otherwise
        """
        # If we have a real client, configure both cameras
        if self.client and not self._is_simulation:
            left_id, right_id = self.camera_ids
            success_left = self.client.set_exposure(left_id, exposure_time, gain)
            success_right = self.client.set_exposure(right_id, exposure_time, gain)
            return success_left and success_right
        
        return True  # Simulated success


class CameraClient:
    """
    Client for managing UnLook scanner cameras.
    """

    def __init__(self, parent_client):
        """
        Initialize camera client.

        Args:
            parent_client: Main UnlookClient
        """
        self.client = parent_client
        self.cameras = {}  # Cache of available cameras

        # Define focus quality thresholds
        self.focus_thresholds = {
            'poor': 50,
            'moderate': 150,
            'good': 300,
            'excellent': 500
        }

    def get_cameras(self) -> List[Dict[str, Any]]:
        """
        Get the list of available cameras.

        Returns:
            List of dictionaries with camera information
        """
        success, response, _ = self.client.send_message(
            MessageType.CAMERA_LIST,
            {}
        )

        if success and response:
            cameras = response.payload.get("cameras", [])

            # Update cache
            self.cameras = {cam["id"]: cam for cam in cameras}

            return cameras
        else:
            logger.error("Unable to get camera list")
            return []

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a camera.

        Args:
            camera_id: Camera ID

        Returns:
            Dictionary with camera information, None if not found
        """
        # Update cache if necessary
        if not self.cameras:
            self.get_cameras()

        return self.cameras.get(camera_id)

    def configure(self, camera_id: str, config: Union[Dict[str, Any], CameraConfig]) -> bool:
        """
        Configure a camera.

        Args:
            camera_id: Camera ID
            config: Camera configuration (dict or CameraConfig object)

        Returns:
            True if configuration was successful, False otherwise
        """
        # Convert CameraConfig to dict if needed
        if isinstance(config, CameraConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        # Log what we're doing
        logger.info(f"Configuring camera {camera_id} with: {config_dict}")
            
        success, response, _ = self.client.send_message(
            MessageType.CAMERA_CONFIG,
            {
                "camera_id": camera_id,
                "config": config_dict
            }
        )

        if success and response:
            # Update cache if camera exists
            if camera_id in self.cameras:
                self.cameras[camera_id].update({
                    "configured_resolution": config_dict.get("resolution",
                                                      self.cameras[camera_id].get("resolution")),
                    "configured_fps": config_dict.get("framerate",
                                                self.cameras[camera_id].get("fps"))
                })

            logger.info(f"Camera {camera_id} successfully configured")
            return True
        else:
            logger.error(f"Error configuring camera {camera_id}")
            return False
            
    def apply_camera_config(self, camera_id: str, config: CameraConfig) -> bool:
        """
        Apply a complete camera configuration using the CameraConfig object.
        
        Args:
            camera_id: Camera ID
            config: Complete camera configuration
            
        Returns:
            True if configuration was successful, False otherwise
        """
        return self.configure(camera_id, config)

    def set_exposure(self, camera_id: str, exposure_time: int, gain: Optional[float] = None, auto_exposure: bool = False, auto_gain: bool = False) -> bool:
        """
        Set camera exposure and gain settings.

        Args:
            camera_id: Camera ID
            exposure_time: Exposure time in microseconds
            gain: Analog gain (optional)
            auto_exposure: Whether to use auto exposure (default: False)
            auto_gain: Whether to use auto gain (default: False)

        Returns:
            True if successful, False otherwise
        """
        config = {
            "exposure": exposure_time,
            "auto_exposure": auto_exposure
        }
        
        if gain is not None:
            config["gain"] = gain
            config["auto_gain"] = auto_gain
            
        return self.configure(camera_id, config)
        
    def set_color_mode(self, camera_id: str, color_mode: ColorMode) -> bool:
        """
        Set the camera color mode (color or grayscale).
        
        Args:
            camera_id: Camera ID
            color_mode: Color mode to use
            
        Returns:
            True if successful, False otherwise
        """
        config = {"color_mode": color_mode.value}
        return self.configure(camera_id, config)
        
    def set_image_quality(self, camera_id: str, preset: ImageQualityPreset) -> bool:
        """
        Apply an image quality preset to the camera.
        
        Args:
            camera_id: Camera ID
            preset: Quality preset to apply
            
        Returns:
            True if successful, False otherwise
        """
        # Create new config with the preset
        config = CameraConfig()
        config.set_quality_preset(preset)
        
        # Apply the configuration
        return self.configure(camera_id, config)
        
    def set_compression(self, camera_id: str, format: CompressionFormat, jpeg_quality: int = None) -> bool:
        """
        Set the image compression settings.
        
        Args:
            camera_id: Camera ID
            format: Compression format (JPEG, PNG, RAW)
            jpeg_quality: JPEG quality (0-100), only applicable for JPEG format
            
        Returns:
            True if successful, False otherwise
        """
        config = {"compression_format": format.value}
        
        if format == CompressionFormat.JPEG and jpeg_quality is not None:
            config["jpeg_quality"] = max(0, min(100, jpeg_quality))
            
        return self.configure(camera_id, config)
        
    def set_crop_region(self, camera_id: str, x: int, y: int, width: int, height: int) -> bool:
        """
        Set a region of interest (ROI) crop for the camera.
        
        Args:
            camera_id: Camera ID
            x: Left coordinate
            y: Top coordinate
            width: ROI width
            height: ROI height
            
        Returns:
            True if successful, False otherwise
        """
        config = {"crop_region": (x, y, width, height)}
        return self.configure(camera_id, config)
        
    def reset_crop_region(self, camera_id: str) -> bool:
        """
        Reset to use the full camera frame (no cropping).
        
        Args:
            camera_id: Camera ID
            
        Returns:
            True if successful, False otherwise
        """
        config = {"crop_region": None}
        return self.configure(camera_id, config)
        
    def set_image_adjustments(self, camera_id: str, 
                             brightness: float = None,
                             contrast: float = None,
                             saturation: float = None,
                             sharpness: float = None,
                             gamma: float = None) -> bool:
        """
        Set image adjustment parameters.
        
        Args:
            camera_id: Camera ID
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast enhancement factor
            saturation: Color saturation
            sharpness: Sharpness adjustment (0=default, 1=max)
            gamma: Gamma correction (1.0=linear)
            
        Returns:
            True if successful, False otherwise
        """
        config = {}
        
        if brightness is not None:
            config["brightness"] = brightness
        
        if contrast is not None:
            config["contrast"] = contrast
        
        if saturation is not None:
            config["saturation"] = saturation
            
        if sharpness is not None:
            config["sharpness"] = sharpness
            
        if gamma is not None:
            config["gamma"] = gamma
            
        if not config:
            # Nothing to set
            return True
            
        return self.configure(camera_id, config)
        
    def set_image_processing(self, camera_id: str,
                           denoise: bool = None,
                           hdr_mode: bool = None,
                           stabilization: bool = None) -> bool:
        """
        Set image processing options.
        
        Args:
            camera_id: Camera ID
            denoise: Enable noise reduction
            hdr_mode: Enable High Dynamic Range mode
            stabilization: Enable image stabilization
            
        Returns:
            True if successful, False otherwise
        """
        config = {}
        
        if denoise is not None:
            config["denoise"] = denoise
            
        if hdr_mode is not None:
            config["hdr_mode"] = hdr_mode
            
        if stabilization is not None:
            config["stabilization"] = stabilization
            
        if not config:
            # Nothing to set
            return True
            
        return self.configure(camera_id, config)

    def set_white_balance(self, camera_id: str, mode: str, red_gain: Optional[float] = None,
                          blue_gain: Optional[float] = None) -> bool:
        """
        Set camera white balance.

        Args:
            camera_id: Camera ID
            mode: Mode ("auto" or "manual")
            red_gain: Red channel gain (required for "manual")
            blue_gain: Blue channel gain (required for "manual")

        Returns:
            True if successful, False otherwise
        """
        config = {"awb": mode}
        if mode == "manual" and red_gain is not None and blue_gain is not None:
            config["awb_gains"] = [red_gain, blue_gain]
        return self.configure(camera_id, config)

    def flip_image(self, camera_id: str, horizontal: bool = False, vertical: bool = False) -> bool:
        """
        Flip camera image.

        Args:
            camera_id: Camera ID
            horizontal: Flip horizontally
            vertical: Flip vertically

        Returns:
            True if successful, False otherwise
        """
        return self.configure(camera_id, {"hflip": horizontal, "vflip": vertical})

    def configure_all(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Configure all cameras with the same configuration.

        Args:
            config: Configuration to apply to all cameras

        Returns:
            Dictionary {camera_id: success} with results
        """
        # Update cache if necessary
        if not self.cameras:
            self.get_cameras()

        results = {}

        for camera_id in self.cameras:
            success = self.configure(camera_id, config)
            results[camera_id] = success

        return results

    def capture(self, 
              camera_id: str, 
              jpeg_quality: int = DEFAULT_JPEG_QUALITY,
              format: CompressionFormat = CompressionFormat.JPEG,
              resolution: Tuple[int, int] = None,
              crop_region: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Capture an image from a camera with advanced configuration options.

        Args:
            camera_id: Camera ID
            jpeg_quality: JPEG quality (0-100)
            format: Image compression format
            resolution: Optional resolution override (width, height)
            crop_region: Optional crop region (x, y, width, height)

        Returns:
            Image as numpy array, None if error
        """
        # Prepare capture request parameters
        # Handle both enum and string format types
        if hasattr(format, 'value'):
            format_value = format.value
        else:
            format_value = format
            
        params = {
            "camera_id": camera_id,
            "compression_format": format_value
        }
        
        # Add JPEG quality if needed
        if (hasattr(CompressionFormat, 'JPEG') and format == CompressionFormat.JPEG) or format == "jpeg":
            params["jpeg_quality"] = jpeg_quality
        
        # Add resolution if specified
        if resolution is not None:
            params["resolution"] = resolution
        
        # Add crop region if specified
        if crop_region is not None:
            params["crop_region"] = crop_region
        
        # Send capture request
        success, response, binary_data = self.client.send_message(
            MessageType.CAMERA_CAPTURE,
            params,
            binary_response=True
        )

        if success and binary_data:
            try:
                # Decode image based on format
                if (hasattr(CompressionFormat, 'JPEG') and format == CompressionFormat.JPEG) or format == "jpeg":
                    image = decode_jpeg_to_image(binary_data)
                elif format == CompressionFormat.PNG:
                    # Decode PNG using OpenCV
                    import cv2
                    import numpy as np
                    nparr = np.frombuffer(binary_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                elif format == CompressionFormat.RAW:
                    # Process raw data based on response metadata
                    if response and hasattr(response, 'payload'):
                        metadata = response.payload
                        width = metadata.get('width', 0)
                        height = metadata.get('height', 0)
                        channels = metadata.get('channels', 1)
                        
                        if width > 0 and height > 0:
                            # Convert raw bytes to numpy array
                            image = np.frombuffer(binary_data, dtype=np.uint8)
                            image = image.reshape((height, width, channels))
                        else:
                            logger.error("Invalid raw image dimensions")
                            return None
                    else:
                        logger.error("Missing metadata for raw image format")
                        return None
                else:
                    logger.error(f"Unsupported image format: {format}")
                    return None
                    
                return image
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        else:
            logger.error(f"Error capturing image from camera {camera_id}")
            return None

    def capture_multi(self, 
                  camera_ids: List[str], 
                  jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                  format: CompressionFormat = CompressionFormat.JPEG,
                  resolution: Tuple[int, int] = None,
                  crop_region: Tuple[int, int, int, int] = None) -> Dict[str, Optional[np.ndarray]]:
        """
        Capture synchronized images from multiple cameras with advanced configuration options.
        Uses the improved deserialize_binary_message function to support different formats.

        Args:
            camera_ids: List of camera IDs
            jpeg_quality: JPEG quality (0-100)
            format: Image compression format
            resolution: Optional resolution override (width, height)
            crop_region: Optional crop region (x, y, width, height)

        Returns:
            Dictionary {camera_id: image} with captured images
        """
        # Prepare capture request parameters
        # Handle both enum and string format types
        if hasattr(format, 'value'):
            format_value = format.value
        else:
            format_value = format
            
        params = {
            "camera_ids": camera_ids,
            "compression_format": format_value
        }
        
        # Add JPEG quality if needed
        if (hasattr(CompressionFormat, 'JPEG') and format == CompressionFormat.JPEG) or format == "jpeg":
            params["jpeg_quality"] = jpeg_quality
        
        # Add resolution if specified
        if resolution is not None:
            params["resolution"] = resolution
        
        # Add crop region if specified
        if crop_region is not None:
            params["crop_region"] = crop_region
            
        # Send synchronized capture request
        success, response, binary_data = self.client.send_message(
            MessageType.CAMERA_CAPTURE_MULTI,
            params,
            binary_response=True
        )

        if not success or binary_data is None:
            logger.error("Error in synchronized capture")
            return {}

        try:
            # Debug log
            logger.debug(f"Received {len(binary_data)} bytes of binary data")

            # Deserialize using the improved function
            msg_type, payload, binary_data = deserialize_binary_message(binary_data)

            # Log format information
            logger.debug(f"Detected format: {msg_type}, payload: {payload.get('format', 'N/A')}")

            # ULMC FORMAT HANDLING
            if msg_type == "multi_camera_response" and payload.get("format") == "ULMC":
                logger.info(f"Processing ULMC response with {payload.get('num_cameras', 0)} cameras")
                logger.debug(f"Full payload: {payload}")
                logger.debug(f"Camera info: {payload.get('cameras', {})}")
                images = {}

                # Check if cameras dict exists
                cameras_dict = payload.get("cameras", {})
                if not cameras_dict:
                    logger.error("No cameras information in ULMC payload")
                    # Fall through to fallback method
                else:
                    # For each camera
                    for camera_id, camera_info in cameras_dict.items():
                        logger.debug(f"Processing camera {camera_id} with info: {camera_info}")
                        
                        # Extract and decode image
                        offset = camera_info.get("offset", 0)
                        size = camera_info.get("size", 0)

                        if offset >= 0 and size > 0 and offset + size <= len(binary_data):
                            jpeg_data = binary_data[offset:offset + size]
                            image = decode_jpeg_to_image(jpeg_data)

                            if image is not None:
                                images[camera_id] = image
                                logger.debug(f"Decoded ULMC image for camera {camera_id}: {image.shape}")
                            else:
                                logger.error(f"Unable to decode ULMC image for camera {camera_id}")
                        else:
                            logger.error(f"Invalid offset/size for camera {camera_id}: offset={offset}, size={size}, data_len={len(binary_data)}")

                    if images:
                        return images
                    else:
                        logger.warning("No images decoded from ULMC format, falling through")

            # DIRECT JPEG HANDLING
            if msg_type == "camera_frame" and payload.get("direct_image", False):
                # It's a single JPEG image, assign it to the first camera
                if camera_ids:
                    image = decode_jpeg_to_image(binary_data)
                    if image is not None:
                        logger.warning("Received only one image instead of one for each camera")
                        return {camera_ids[0]: image}

            # ALTERNATIVE FORMAT HANDLING WITH SIZE PREFIXES
            if msg_type == "multi_camera_response" and payload.get("alternative_format", False):
                logger.info("Processing alternative format with size prefixes")
                return self._fallback_decode_multi_response(binary_data, camera_ids)

            # RAW BINARY DATA HANDLING
            if msg_type == "binary_data":
                logger.info("Processing raw binary data with fallback method")
                return self._fallback_decode_multi_response(binary_data, camera_ids)

            # If we're here, we couldn't decode the format
            logger.warning(f"Unrecognized response format: {msg_type}")
            return self._fallback_decode_multi_response(binary_data, camera_ids)

        except Exception as e:
            logger.error(f"Error decoding multi-camera response: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Attempt fallback as last resort
            return self._fallback_decode_multi_response(binary_data, camera_ids)

    def _fallback_decode_multi_response(self, binary_data: bytes, camera_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Fallback method to decode multi-camera response when format is unknown.
        Attempts to extract actual camera data instead of generating synthetic patterns.
        """
        try:
            logger.info("Using fallback method to decode multi-camera response")
            images = {}
            
            # Try to extract JPEG images from the binary data
            # Look for JPEG markers in the data
            jpeg_starts = []
            jpeg_ends = []
            
            # JPEG SOI marker (Start of Image)
            soi_marker = b'\xff\xd8'
            # JPEG EOI marker (End of Image) 
            eoi_marker = b'\xff\xd9'
            
            # Find all JPEG images in the data
            offset = 0
            while offset < len(binary_data) - 1:
                if binary_data[offset:offset+2] == soi_marker:
                    jpeg_starts.append(offset)
                elif binary_data[offset:offset+2] == eoi_marker:
                    jpeg_ends.append(offset + 2)
                offset += 1
            
            # Extract and decode images
            num_images_found = min(len(jpeg_starts), len(jpeg_ends))
            logger.info(f"Found {num_images_found} JPEG images in binary data")
            
            if num_images_found > 0:
                for i in range(min(num_images_found, len(camera_ids))):
                    try:
                        # Extract JPEG data
                        jpeg_data = binary_data[jpeg_starts[i]:jpeg_ends[i]]
                        
                        # Decode JPEG to image
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            logger.info(f"Successfully decoded image {i+1} of size {image.shape}")
                            images[camera_ids[i]] = image
                        else:
                            logger.warning(f"Failed to decode JPEG image {i+1}")
                            
                    except Exception as e:
                        logger.error(f"Error decoding image {i+1}: {e}")
            else:
                logger.warning("No JPEG images found in binary data, attempting raw decode")
                
                # If no JPEG markers found, try to decode as raw data
                # This could be raw RGB or other format
                try:
                    # Assume raw RGB data for now
                    expected_size = 1280 * 720 * 3  # width * height * channels
                    
                    for i, camera_id in enumerate(camera_ids):
                        offset = i * expected_size
                        if offset + expected_size <= len(binary_data):
                            raw_data = binary_data[offset:offset + expected_size]
                            image = np.frombuffer(raw_data, dtype=np.uint8).reshape(720, 1280, 3)
                            images[camera_id] = image
                            logger.info(f"Decoded raw RGB image for camera {camera_id}")
                except Exception as e:
                    logger.error(f"Failed to decode as raw data: {e}")
            
            # If still no images, return empty dict rather than synthetic patterns
            if not images:
                logger.error("Could not extract any real camera images from binary data")
                logger.error(f"Binary data size: {len(binary_data)} bytes")
                logger.error(f"First 100 bytes: {binary_data[:100].hex()}")
                
            return images
            
        except Exception as e:
            logger.error(f"Error in fallback method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_stereo_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find a camera pair for stereovision.

        Returns:
            Tuple (left_camera_id, right_camera_id), None if not found
        """
        # Update cache if necessary
        if not self.cameras:
            self.get_cameras()

        if len(self.cameras) < 2:
            logger.warning("At least 2 cameras are needed for stereovision")
            return None, None

        # Simplified selection: take the first two cameras
        # In a more advanced implementation, there might be additional metadata
        # to identify left and right cameras
        camera_ids = list(self.cameras.keys())

        # Look for cameras with "left" or "right" in the name or ID
        left_camera = None
        right_camera = None

        for camera_id, camera_info in self.cameras.items():
            camera_name = camera_info.get("name", "").lower()
            if "left" in camera_name or "left" in camera_id.lower():
                left_camera = camera_id
            elif "right" in camera_name or "right" in camera_id.lower():
                right_camera = camera_id

        # If not found, use the first two cameras
        if left_camera is None or right_camera is None:
            left_camera = camera_ids[0]
            right_camera = camera_ids[1]
            logger.info(
                f"Using default cameras for stereovision: {left_camera} (left), {right_camera} (right)")
        else:
            logger.info(f"Found stereo pair: {left_camera} (left), {right_camera} (right)")

        return left_camera, right_camera

    def capture_stereo_pair(self, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture a synchronized image from a stereo pair.

        Args:
            jpeg_quality: JPEG quality (0-100)

        Returns:
            Tuple (left_image, right_image), None if error
        """
        left_camera, right_camera = self.get_stereo_pair()

        if left_camera is None or right_camera is None:
            logger.error("Unable to find a valid stereo pair")
            return None, None

        # Configure cameras if necessary (e.g., same resolution)
        try:
            # Get camera information
            left_info = self.get_camera(left_camera)
            right_info = self.get_camera(right_camera)

            # If resolutions are different, make them uniform
            if (left_info and right_info and
                    left_info.get("configured_resolution") != right_info.get("configured_resolution")):
                # Use the lowest resolution of the two
                left_res = left_info.get("configured_resolution", [1920, 1080])
                right_res = right_info.get("configured_resolution", [1920, 1080])

                target_res = [
                    min(left_res[0], right_res[0]),
                    min(left_res[1], right_res[1])
                ]

                logger.info(f"Uniforming stereo camera resolution to {target_res}")
                self.configure(left_camera, {"resolution": target_res})
                self.configure(right_camera, {"resolution": target_res})
        except Exception as e:
            logger.warning(f"Error configuring cameras: {e}")

        # Synchronized capture
        images = self.capture_multi([left_camera, right_camera], jpeg_quality)

        # Check results
        if not images:
            logger.error("Error capturing stereo images")
            return None, None

        if len(images) != 2:
            logger.error(f"Wrong number of images received ({len(images)} instead of 2)")
            return None, None

        if left_camera not in images or right_camera not in images:
            logger.error(f"Missing cameras in response: received {list(images.keys())}")
            return None, None

        left_image = images.get(left_camera)
        right_image = images.get(right_camera)

        # Final check
        if left_image is None or right_image is None:
            logger.error("One or both stereo images are null")
            return None, None

        # Check that images have the same size
        if left_image.shape != right_image.shape:
            logger.warning(f"Stereo images have different sizes: {left_image.shape} vs {right_image.shape}")
            # We could resize here, but for now we return the images as they are

        logger.info("Stereo capture completed successfully")
        return left_image, right_image

    def _calculate_focus_score(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Calculate the focus score of an image using the Laplacian variance method.

        Args:
            image: Input image (grayscale or color)
            roi: Optional region of interest (x, y, width, height) to analyze

        Returns:
            Focus score (higher is better focused)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply ROI if specified
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # Use Laplacian for focus measure - higher variance means sharper image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_score = laplacian.var()

        return focus_score

    def check_focus(self, camera_id: str, num_samples: int = 3,
                   roi: Optional[Tuple[int, int, int, int]] = None,
                   threshold_for_good: float = None) -> Tuple[float, str, np.ndarray]:
        """
        Check the focus quality of a camera.

        Args:
            camera_id: Camera ID
            num_samples: Number of images to capture and average
            roi: Optional region of interest (x, y, width, height) to analyze
            threshold_for_good: Optional custom threshold to determine good focus

        Returns:
            Tuple of (focus_score, quality_level, latest_image)
        """
        focus_scores = []
        latest_image = None

        # Capture multiple samples for better accuracy
        for _ in range(num_samples):
            image = self.capture(camera_id)
            if image is None:
                logger.error(f"Failed to capture image from camera {camera_id}")
                continue

            latest_image = image
            score = self._calculate_focus_score(image, roi)
            focus_scores.append(score)
            time.sleep(0.1)  # Brief pause between captures

        # Calculate average score
        if not focus_scores:
            logger.error(f"No valid focus measurements for camera {camera_id}")
            return 0, "unknown", latest_image

        avg_score = sum(focus_scores) / len(focus_scores)

        # Determine quality level
        if threshold_for_good is None:
            if avg_score < self.focus_thresholds['poor']:
                quality = "poor"
            elif avg_score < self.focus_thresholds['moderate']:
                quality = "moderate"
            elif avg_score < self.focus_thresholds['good']:
                quality = "good"
            else:
                quality = "excellent"
        else:
            quality = "good" if avg_score >= threshold_for_good else "poor"

        logger.info(f"Camera {camera_id} focus score: {avg_score:.2f} - Quality: {quality}")
        return avg_score, quality, latest_image

    def check_stereo_focus(self, left_camera_id: str = None, right_camera_id: str = None,
                         num_samples: int = 3, roi: Optional[Tuple[int, int, int, int]] = None,
                         threshold_for_good: float = None) -> Tuple[Dict[str, Tuple[float, str]], Dict[str, np.ndarray]]:
        """
        Check the focus quality of a stereo camera pair.

        Args:
            left_camera_id: ID of the left camera, if None uses first found
            right_camera_id: ID of the right camera, if None uses second found
            num_samples: Number of images to capture and average
            roi: Optional region of interest (x, y, width, height) to analyze
            threshold_for_good: Optional custom threshold to determine good focus

        Returns:
            Tuple of (
                {camera_id: (focus_score, quality_level)},
                {camera_id: latest_image}
            )
        """
        left_camera, right_camera = self.get_stereo_pair()

        if left_camera is None or right_camera is None:
            logger.error("Need at least 2 cameras for stereo focus check")
            return ({}, {})

        # Override camera IDs if specified
        if left_camera_id is not None:
            left_camera = left_camera_id
        if right_camera_id is not None:
            right_camera = right_camera_id

        # Check focus for each camera
        left_score, left_quality, left_image = self.check_focus(
            left_camera, num_samples, roi, threshold_for_good)
        right_score, right_quality, right_image = self.check_focus(
            right_camera, num_samples, roi, threshold_for_good)

        results = {
            left_camera: (left_score, left_quality),
            right_camera: (right_score, right_quality)
        }

        images = {
            left_camera: left_image,
            right_camera: right_image
        }

        return results, images

    def interactive_focus_check(self, camera_id: str = None,
                               interval: float = 0.5,
                               roi: Optional[Tuple[int, int, int, int]] = None,
                               callback: Optional[Callable] = None,
                               threshold_for_good: float = None) -> Tuple[float, str, np.ndarray]:
        """
        Run an interactive focus check with continuous feedback.

        Args:
            camera_id: Camera ID, if None uses first found
            interval: Seconds between focus checks
            roi: Optional region of interest (x, y, width, height) to analyze
            callback: Optional callback function(score, quality, image) to process results
            threshold_for_good: Optional custom threshold to determine good focus

        Returns:
            Final (focus_score, quality_level, latest_image)
        """
        cameras = self.get_cameras()
        if not cameras:
            logger.error("No cameras available for focus check")
            return (0, "unknown", None)

        # If camera ID is not specified, use first one
        if camera_id is None:
            camera_id = cameras[0]["id"]

        logger.info(f"Starting interactive focus check for camera {camera_id}")
        logger.info("Press Ctrl+C to stop when focus is good")

        try:
            while True:
                score, quality, image = self.check_focus(
                    camera_id, num_samples=1, roi=roi, threshold_for_good=threshold_for_good)

                if callback:
                    callback(score, quality, image)
                else:
                    # Default feedback
                    if quality == "poor":
                        direction = "ADJUST FOCUS - Image is very blurry"
                    elif quality == "moderate":
                        direction = "KEEP ADJUSTING - Focus is improving"
                    elif quality == "good":
                        direction = "ALMOST THERE - Focus is good"
                    else:  # excellent
                        direction = "PERFECT - Focus is excellent"

                    logger.info(f"Focus score: {score:.2f} - {direction}")

                    # Visual feedback by showing the image - only if OpenCV display is available
                    try:
                        if image is not None:
                            # Draw focus information on image
                            feedback_img = image.copy()
                            feedback_text = f"Focus: {score:.1f} - {quality.upper()}"
                            cv2.putText(feedback_img, feedback_text, (30, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Show ROI if specified
                            if roi:
                                x, y, w, h = roi
                                cv2.rectangle(feedback_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                            cv2.imshow("Focus Check", feedback_img)
                            key = cv2.waitKey(1)
                            if key == 27:  # ESC key
                                break
                    except Exception as e:
                        # Skip visual feedback if there are display issues
                        logger.debug(f"Could not show visual feedback: {e}")

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Focus check interrupted by user")
        finally:
            try:
                cv2.destroyWindow("Focus Check")
            except:
                pass

        # Final check with more samples for accurate result
        final_score, final_quality, final_image = self.check_focus(
            camera_id, num_samples=3, roi=roi, threshold_for_good=threshold_for_good)

        logger.info(f"Final focus score: {final_score:.2f} - Quality: {final_quality}")
        return final_score, final_quality, final_image

    def interactive_stereo_focus_check(self, left_camera_id: str = None, right_camera_id: str = None,
                                    interval: float = 0.5,
                                    roi: Optional[Tuple[int, int, int, int]] = None,
                                    threshold_for_good: float = None) -> Tuple[Dict[str, Tuple[float, str]], Dict[str, np.ndarray]]:
        """
        Run an interactive focus check for stereo cameras with continuous feedback.

        Args:
            left_camera_id: ID of the left camera, if None uses first found
            right_camera_id: ID of the right camera, if None uses second found
            interval: Seconds between focus checks
            roi: Optional region of interest (x, y, width, height) to analyze
            threshold_for_good: Optional custom threshold to determine good focus

        Returns:
            Final (
                {camera_id: (focus_score, quality_level)},
                {camera_id: latest_image}
            )
        """
        left_camera, right_camera = self.get_stereo_pair()

        if left_camera is None or right_camera is None:
            logger.error("Need at least 2 cameras for stereo focus check")
            return ({}, {})

        # Override camera IDs if specified
        if left_camera_id is not None:
            left_camera = left_camera_id
        if right_camera_id is not None:
            right_camera = right_camera_id

        logger.info(f"Starting interactive stereo focus check for cameras {left_camera} and {right_camera}")
        logger.info("Press Ctrl+C to stop when both cameras are in focus")

        try:
            while True:
                results, images = self.check_stereo_focus(
                    left_camera, right_camera, num_samples=1, roi=roi, threshold_for_good=threshold_for_good)

                if not results:
                    logger.error("Failed to check stereo focus")
                    time.sleep(interval)
                    continue

                # Visual feedback
                try:
                    left_score, left_quality = results[left_camera]
                    right_score, right_quality = results[right_camera]
                    left_image = images[left_camera]
                    right_image = images[right_camera]

                    # Create combined feedback image
                    if left_image is not None and right_image is not None:
                        # Ensure images are the same size
                        if left_image.shape != right_image.shape:
                            # Resize smaller image to match larger one
                            if left_image.shape[0] * left_image.shape[1] < right_image.shape[0] * right_image.shape[1]:
                                left_image = cv2.resize(left_image, (right_image.shape[1], right_image.shape[0]))
                            else:
                                right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

                        # Draw focus information on images
                        left_feedback = left_image.copy()
                        right_feedback = right_image.copy()

                        left_text = f"Left: {left_score:.1f} - {left_quality.upper()}"
                        right_text = f"Right: {right_score:.1f} - {right_quality.upper()}"

                        cv2.putText(left_feedback, left_text, (30, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(right_feedback, right_text, (30, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Show ROI if specified
                        if roi:
                            x, y, w, h = roi
                            cv2.rectangle(left_feedback, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.rectangle(right_feedback, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # Combine images horizontally
                        combined = np.hstack((left_feedback, right_feedback))

                        # Add combined status text
                        if left_quality == "excellent" and right_quality == "excellent":
                            status = "BOTH CAMERAS IN PERFECT FOCUS"
                        elif left_quality in ["good", "excellent"] and right_quality in ["good", "excellent"]:
                            status = "GOOD FOCUS - Press Ctrl+C to continue"
                        else:
                            status = "ADJUSTING FOCUS NEEDED"

                        cv2.putText(combined, status, (combined.shape[1]//2 - 200, combined.shape[0] - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                        # Show combined image
                        cv2.imshow("Stereo Focus Check", combined)
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC key
                            break
                except Exception as e:
                    # Skip visual feedback if there are display issues
                    logger.debug(f"Could not show visual feedback: {e}")

                # Print feedback
                left_score, left_quality = results[left_camera]
                right_score, right_quality = results[right_camera]

                logger.info(f"Left camera: {left_score:.2f} ({left_quality}) | Right camera: {right_score:.2f} ({right_quality})")

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Stereo focus check interrupted by user")
        finally:
            try:
                cv2.destroyWindow("Stereo Focus Check")
            except:
                pass

        # Final check with more samples for accurate result
        final_results, final_images = self.check_stereo_focus(
            left_camera, right_camera, num_samples=3, roi=roi, threshold_for_good=threshold_for_good)

        logger.info(f"Final focus results: Left: {final_results[left_camera][0]:.2f} ({final_results[left_camera][1]}), "
                  f"Right: {final_results[right_camera][0]:.2f} ({final_results[right_camera][1]})")

        return final_results, final_images
    
    def optimize_camera_settings(self, camera_id: str, target_brightness: float = 0.5, 
                               target_contrast: float = 0.3) -> Dict[str, Any]:
        """
        Automatically optimize camera settings for pattern visibility.
        
        Args:
            camera_id: Camera ID to optimize
            target_brightness: Target mean brightness (0-1, default 0.5)
            target_contrast: Target contrast level (0-1, default 0.3)
            
        Returns:
            Dictionary with optimized settings
        """
        request = Message(
            msg_type=MessageType.CAMERA_OPTIMIZE,
            payload={
                "camera_id": camera_id,
                "target_brightness": target_brightness,
                "target_contrast": target_contrast
            }
        )
        
        response = self.client.send_request(request)
        if response.payload.get("success"):
            return response.payload.get("optimized_settings", {})
        else:
            logger.error(f"Camera optimization failed: {response.payload.get('error')}")
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
        payload = {"camera_id": camera_id}
        if focus_region:
            payload["focus_region"] = focus_region
            
        request = Message(
            msg_type=MessageType.CAMERA_AUTO_FOCUS,
            payload=payload
        )
        
        response = self.client.send_request(request)
        return response.payload.get("success", False)
    
    def capture_test_image(self, camera_id: str, test_type: str = "normal") -> Optional[np.ndarray]:
        """
        Capture a test image for optimization.
        
        Args:
            camera_id: Camera ID
            test_type: Type of test image ("underexposed", "normal", "overexposed")
            
        Returns:
            Test image as numpy array, None if error
        """
        request = Message(
            msg_type=MessageType.CAMERA_TEST_CAPTURE,
            payload={
                "camera_id": camera_id,
                "test_type": test_type
            }
        )
        
        response = self.client.send_request(request)
        
        if response.payload.get("success"):
            # Decode base64 image
            import base64
            image_data = base64.b64decode(response.payload["image"])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        else:
            logger.error(f"Failed to capture test image: {response.payload.get('error')}")
            return None