"""
Client for managing UnLook scanner cameras.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np

from ..core.protocol import MessageType
from ..core.utils import decode_jpeg_to_image, deserialize_binary_message
from ..core.constants import DEFAULT_JPEG_QUALITY
from ..core.events import EventType
from .camera_config import CameraConfig, ColorMode, CompressionFormat, ImageQualityPreset

logger = logging.getLogger(__name__)


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
        params = {
            "camera_id": camera_id,
            "compression_format": format.value
        }
        
        # Add JPEG quality if needed
        if format == CompressionFormat.JPEG:
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
                if format == CompressionFormat.JPEG:
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
        params = {
            "camera_ids": camera_ids,
            "compression_format": format.value
        }
        
        # Add JPEG quality if needed
        if format == CompressionFormat.JPEG:
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
                images = {}

                # For each camera
                for camera_id, camera_info in payload.get("cameras", {}).items():
                    # Extract and decode image
                    offset = camera_info.get("offset", 0)
                    size = camera_info.get("size", 0)

                    if offset > 0 and size > 0 and offset + size <= len(binary_data):
                        jpeg_data = binary_data[offset:offset + size]
                        image = decode_jpeg_to_image(jpeg_data)

                        if image is not None:
                            images[camera_id] = image
                            logger.debug(f"Decoded ULMC image for camera {camera_id}: {image.shape}")
                        else:
                            logger.error(f"Unable to decode ULMC image for camera {camera_id}")

                if images:
                    return images

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
        Improved fallback method to decode multi-camera response.
        Implements different strategies to extract images, ensuring maximum robustness.

        Args:
            binary_data: Received binary data
            camera_ids: Requested camera IDs

        Returns:
            Dictionary {camera_id: image}
        """
        logger.info("Using advanced fallback method to decode multi-camera response")
        images = {}

        # More robust analysis of received data
        if len(binary_data) < 8:
            logger.error("Insufficient binary data for fallback method")
            return {}

        # Print diagnostic information on first bytes
        logger.debug(f"First 32 bytes: {binary_data[:32].hex()}")

        # STRATEGY 1: Look for JPEG markers (FFD8) and JPEG end (FFD9)
        # This is useful when images are simply concatenated
        jpeg_starts = []
        jpeg_ends = []

        for i in range(len(binary_data) - 1):
            if binary_data[i] == 0xFF:
                if binary_data[i + 1] == 0xD8:  # Start of JPEG
                    jpeg_starts.append(i)
                elif binary_data[i + 1] == 0xD9:  # End of JPEG
                    jpeg_ends.append(i + 1)

        # If we find the same number of JPEG starts and ends and it matches the number of cameras
        if len(jpeg_starts) == len(jpeg_ends) and len(jpeg_starts) == len(camera_ids):
            logger.info(f"STRATEGY 1: Found {len(jpeg_starts)} complete JPEG images in binary stream")

            # Sort start/end positions
            pairs = sorted(zip(jpeg_starts, jpeg_ends))

            for idx, (start, end) in enumerate(pairs):
                jpeg_data = binary_data[start:end + 1]  # Include end marker
                image = decode_jpeg_to_image(jpeg_data)

                if image is not None:
                    camera_id = camera_ids[idx]
                    images[camera_id] = image
                    logger.debug(f"Decoded image for camera {camera_id}: {image.shape}")
                else:
                    logger.error(f"Unable to decode image {idx + 1}")

            return images

        # STRATEGY 2: Assume format [size1(4B) | JPEG1 | size2(4B) | JPEG2 | ...]
        logger.info("STRATEGY 2: Attempting decoding based on size prefixes")
        current_pos = 0
        strategy2_images = {}

        try:
            for idx, camera_id in enumerate(camera_ids):
                if current_pos + 4 > len(binary_data):
                    logger.error("Premature end of data")
                    break

                # Read size of next JPEG block
                img_size = int.from_bytes(binary_data[current_pos:current_pos + 4], byteorder='little')
                current_pos += 4

                # Check if size seems valid
                if img_size <= 0 or img_size > 10 * 1024 * 1024 or current_pos + img_size > len(binary_data):
                    logger.warning(f"Invalid image size: {img_size}, skipping to next strategy")
                    break

                # Extract JPEG data
                jpeg_data = binary_data[current_pos:current_pos + img_size]
                current_pos += img_size

                # Verify it's valid JPEG data
                if len(jpeg_data) > 2 and jpeg_data[0:2] == b'\\xff\\xd8':
                    image = decode_jpeg_to_image(jpeg_data)
                    if image is not None:
                        strategy2_images[camera_id] = image
                        logger.debug(f"Decoded image for camera {camera_id}: {image.shape}")
                    else:
                        logger.error(f"Unable to decode image for camera {camera_id}")
                else:
                    logger.warning(f"Data doesn't appear to be valid JPEG for camera {camera_id}")
                    break
        except Exception as e:
            logger.error(f"Error in STRATEGY 2: {e}")

        # If we decoded all images, return the result
        if len(strategy2_images) == len(camera_ids):
            logger.info("STRATEGY 2 completed successfully")
            return strategy2_images

        # STRATEGY 3: Look for JPEG markers and use heuristics to recognize the end
        logger.info("STRATEGY 3: Heuristic search for JPEG images")
        strategy3_images = {}

        try:
            pos = 0
            for idx, camera_id in enumerate(camera_ids):
                # Look for start of a JPEG
                jpeg_start = -1
                for i in range(pos, len(binary_data) - 1):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD8:
                        jpeg_start = i
                        break

                if jpeg_start == -1:
                    logger.error(f"Unable to find JPEG start for camera {camera_id}")
                    break

                # Look for end of JPEG or next JPEG start
                jpeg_end = -1
                next_start = -1

                for i in range(jpeg_start + 2, len(binary_data) - 1):
                    if binary_data[i] == 0xFF:
                        if binary_data[i + 1] == 0xD9:  # JPEG end
                            jpeg_end = i + 1
                            break
                        elif binary_data[i + 1] == 0xD8 and i > jpeg_start + 100:  # New JPEG, but not too close
                            next_start = i
                            break

                # If we found the end, extract JPEG
                if jpeg_end != -1:
                    jpeg_data = binary_data[jpeg_start:jpeg_end + 1]
                    pos = jpeg_end + 1
                # Otherwise, if we found the start of a new JPEG, use that as end
                elif next_start != -1:
                    jpeg_data = binary_data[jpeg_start:next_start]
                    pos = next_start
                # Otherwise, take everything to the end
                else:
                    jpeg_data = binary_data[jpeg_start:]
                    pos = len(binary_data)

                # Decode image
                image = decode_jpeg_to_image(jpeg_data)
                if image is not None:
                    strategy3_images[camera_id] = image
                    logger.debug(f"Decoded image for camera {camera_id}: {image.shape}")
                else:
                    logger.error(f"Unable to decode image for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error in STRATEGY 3: {e}")

        # If we decoded all images, return the result
        if len(strategy3_images) == len(camera_ids):
            logger.info("STRATEGY 3 completed successfully")
            return strategy3_images

        # STRATEGY 4: Last resort - divide buffer equally between cameras
        # Useful if images have similar sizes
        if not images and len(camera_ids) > 0:
            logger.info("STRATEGY 4: Equidistant buffer division")

            chunk_size = len(binary_data) // len(camera_ids)
            for idx, camera_id in enumerate(camera_ids):
                start = idx * chunk_size
                end = (idx + 1) * chunk_size if idx < len(camera_ids) - 1 else len(binary_data)

                # Look for a JPEG start in the chunk
                jpeg_start = -1
                for i in range(start, min(start + 100, end - 1)):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD8:
                        jpeg_start = i
                        break

                if jpeg_start == -1:
                    logger.error(f"Unable to find JPEG start for camera {camera_id}")
                    continue

                # Look for JPEG end
                jpeg_end = -1
                for i in range(end - 2, jpeg_start + 2, -1):
                    if binary_data[i] == 0xFF and binary_data[i + 1] == 0xD9:
                        jpeg_end = i + 1
                        break

                if jpeg_end == -1:
                    logger.warning(
                        f"Unable to find JPEG end for camera {camera_id}, using chunk end")
                    jpeg_end = end

                # Extract and decode
                jpeg_data = binary_data[jpeg_start:jpeg_end + 1]
                image = decode_jpeg_to_image(jpeg_data)
                if image is not None:
                    images[camera_id] = image
                    logger.debug(f"Decoded image for camera {camera_id}: {image.shape}")
                else:
                    logger.error(f"Unable to decode image for camera {camera_id}")

        # Combine results from different strategies, giving priority to more reliable ones
        final_images = {}
        final_images.update(images)  # Strategy 4 has low priority
        final_images.update(strategy3_images)  # Strategy 3 has medium priority
        final_images.update(strategy2_images)  # Strategy 2 has high priority

        logger.info(f"Decoded {len(final_images)}/{len(camera_ids)} images via fallback")
        return final_images

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

        # Synchronized capture with improved method
        logger.debug("Using capture_multi for stereo capture")
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