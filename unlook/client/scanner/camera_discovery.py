"""
Camera discovery and mapping module for UnLook client.

This module handles dynamic camera ID discovery and mapping between logical
camera names (left, right, primary) and hardware camera IDs (picamera2_0, etc.).
It ensures backward compatibility while fixing the "Camera left not found" issues.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


class CameraDiscovery:
    """
    Manages camera discovery and dynamic ID mapping on the client side.
    """
    
    # Default logical camera names
    LOGICAL_NAMES = {
        "stereo": ["left", "right"],
        "single": ["primary"],
        "multi": ["camera0", "camera1", "camera2", "camera3"]
    }
    
    def __init__(self, client):
        """
        Initialize camera discovery.
        
        Args:
            client: UnlookClient instance for communication
        """
        self.client = client
        self._camera_list = []
        self._camera_mapping = {}
        self._module_config = None
        self._initialized = False
        
    def discover_cameras(self) -> List[str]:
        """
        Discover available cameras from the server.
        
        Returns:
            List of camera IDs (e.g., ["picamera2_0", "picamera2_1"])
        """
        try:
            # Get camera list from server
            cameras = self.client.camera.get_cameras()
            
            # Extract camera IDs from the list
            if isinstance(cameras, dict):
                # If it's a dict, get the keys
                self._camera_list = list(cameras.keys())
            elif isinstance(cameras, list):
                # If it's a list of dicts, extract the 'id' field
                self._camera_list = []
                for cam in cameras:
                    if isinstance(cam, dict):
                        cam_id = cam.get('id') or cam.get('camera_id')
                        if cam_id:
                            self._camera_list.append(cam_id)
                    else:
                        self._camera_list.append(str(cam))
            else:
                self._camera_list = []
            
            logger.info(f"Discovered {len(self._camera_list)} cameras: {self._camera_list}")
            
            # Get module configuration if available
            self._get_module_config()
            
            # Build camera mapping
            self._build_camera_mapping()
            
            self._initialized = True
            return self._camera_list
            
        except Exception as e:
            logger.error(f"Error discovering cameras: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _get_module_config(self):
        """Get scanning module configuration from server."""
        try:
            # Try to get scanner configuration
            response = self.client.send_message({
                "type": "SCANNER_CONFIG",
                "action": "get"
            })
            
            if response and response.get("status") == "success":
                self._module_config = response.get("data", {})
                logger.debug(f"Got module config: {self._module_config}")
        except Exception as e:
            logger.debug(f"Could not get module config: {e}")
            # Not critical - we can still work without it
    
    def _build_camera_mapping(self):
        """Build mapping between logical names and camera IDs."""
        num_cameras = len(self._camera_list)
        
        # Clear existing mapping
        self._camera_mapping = {}
        
        # Get camera mapping from module config if available
        if self._module_config and "camera_mapping" in self._module_config:
            # Use server-provided mapping
            self._camera_mapping.update(self._module_config["camera_mapping"])
            logger.info(f"Using server-provided camera mapping: {self._camera_mapping}")
        else:
            # Build default mapping based on camera count
            if num_cameras == 0:
                logger.warning("No cameras found for mapping")
                
            elif num_cameras == 1:
                # Single camera setup
                camera_id = self._camera_list[0]
                self._camera_mapping = {
                    "primary": camera_id,
                    "left": camera_id,  # Map left to primary for compatibility
                    "right": camera_id,  # Map right to primary for compatibility
                    "camera0": camera_id
                }
                logger.info(f"Single camera setup - mapped all logical names to '{camera_id}'")
                
            elif num_cameras == 2:
                # Stereo setup
                self._camera_mapping = {
                    "left": self._camera_list[0],
                    "right": self._camera_list[1],
                    "primary": self._camera_list[0],
                    "secondary": self._camera_list[1],
                    "camera0": self._camera_list[0],
                    "camera1": self._camera_list[1]
                }
                logger.info(f"Stereo setup - mapped left={self._camera_list[0]}, right={self._camera_list[1]}")
                
            else:
                # Multi-camera setup
                for i, camera_id in enumerate(self._camera_list):
                    self._camera_mapping[f"camera{i}"] = camera_id
                    
                # Also map common names
                self._camera_mapping["primary"] = self._camera_list[0]
                self._camera_mapping["left"] = self._camera_list[0]
                if num_cameras > 1:
                    self._camera_mapping["right"] = self._camera_list[1]
                    self._camera_mapping["secondary"] = self._camera_list[1]
                    
                logger.info(f"Multi-camera setup with {num_cameras} cameras")
        
        # Add direct mappings for hardware IDs (identity mapping)
        for camera_id in self._camera_list:
            self._camera_mapping[camera_id] = camera_id
            
        logger.debug(f"Final camera mapping: {self._camera_mapping}")
    
    @lru_cache(maxsize=32)
    def resolve_camera_id(self, camera_name: str) -> Optional[str]:
        """
        Resolve a logical camera name to hardware ID.
        
        Args:
            camera_name: Logical name (e.g., "left", "right") or hardware ID
            
        Returns:
            Hardware camera ID (e.g., "picamera2_0") or None if not found
        """
        # Ensure cameras are discovered
        if not self._initialized:
            self.discover_cameras()
        
        # Direct lookup
        if camera_name in self._camera_mapping:
            return self._camera_mapping[camera_name]
        
        # Case-insensitive lookup
        camera_name_lower = camera_name.lower()
        for key, value in self._camera_mapping.items():
            if key.lower() == camera_name_lower:
                return value
        
        # Check if it's already a hardware ID
        if camera_name in self._camera_list:
            return camera_name
        
        # Try partial matching (e.g., "cam0" -> "camera0")
        for key, value in self._camera_mapping.items():
            if camera_name_lower in key.lower() or key.lower() in camera_name_lower:
                logger.debug(f"Partial match: {camera_name} -> {key} -> {value}")
                return value
        
        logger.warning(f"Could not resolve camera name '{camera_name}'")
        return None
    
    def get_stereo_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the stereo camera pair IDs.
        
        Returns:
            Tuple of (left_camera_id, right_camera_id) or (None, None)
        """
        left = self.resolve_camera_id("left")
        right = self.resolve_camera_id("right")
        
        # If standard mapping doesn't work, try alternatives
        if left is None or right is None:
            if len(self._camera_list) >= 2:
                left = self._camera_list[0]
                right = self._camera_list[1]
                logger.info(f"Using first two cameras as stereo pair: {left}, {right}")
            else:
                logger.warning("Not enough cameras for stereo pair")
                return None, None
                
        return left, right
    
    def get_primary_camera(self) -> Optional[str]:
        """
        Get the primary camera ID.
        
        Returns:
            Primary camera ID or None
        """
        primary = self.resolve_camera_id("primary")
        
        # Fallback to first camera
        if primary is None and self._camera_list:
            primary = self._camera_list[0]
            logger.info(f"Using first camera as primary: {primary}")
            
        return primary
    
    def get_camera_count(self) -> int:
        """Get the number of available cameras."""
        return len(self._camera_list)
    
    def get_camera_list(self) -> List[str]:
        """Get list of all camera IDs."""
        if not self._initialized:
            self.discover_cameras()
        return self._camera_list.copy()
    
    def get_mapping(self) -> Dict[str, str]:
        """Get the complete camera mapping dictionary."""
        if not self._initialized:
            self.discover_cameras()
        return self._camera_mapping.copy()
    
    def validate_camera_names(self, camera_names: List[str]) -> List[str]:
        """
        Validate and resolve a list of camera names.
        
        Args:
            camera_names: List of logical names or hardware IDs
            
        Returns:
            List of valid hardware IDs
        """
        valid_ids = []
        
        for name in camera_names:
            hw_id = self.resolve_camera_id(name)
            if hw_id:
                valid_ids.append(hw_id)
            else:
                logger.warning(f"Invalid camera name: {name}")
                
        return valid_ids
    
    def refresh(self):
        """Refresh camera discovery and mapping."""
        logger.info("Refreshing camera discovery...")
        self._initialized = False
        self.resolve_camera_id.cache_clear()
        self.discover_cameras()


class CameraMapper:
    """
    Simplified camera mapper for backward compatibility.
    Can be used as a drop-in replacement in existing code.
    """
    
    def __init__(self, camera_list: List[str]):
        """
        Initialize mapper with a list of camera IDs.
        
        Args:
            camera_list: List of hardware camera IDs
        """
        self.discovery = None  # Simulate discovery without client
        self._camera_list = camera_list
        self._camera_mapping = {}
        self._build_static_mapping()
        
    def _build_static_mapping(self):
        """Build static mapping from camera list."""
        num_cameras = len(self._camera_list)
        
        if num_cameras == 1:
            camera_id = self._camera_list[0]
            self._camera_mapping = {
                "primary": camera_id,
                "left": camera_id,
                "right": camera_id,
                camera_id: camera_id
            }
        elif num_cameras >= 2:
            self._camera_mapping = {
                "left": self._camera_list[0],
                "right": self._camera_list[1],
                "primary": self._camera_list[0],
                self._camera_list[0]: self._camera_list[0],
                self._camera_list[1]: self._camera_list[1]
            }
            
    def resolve(self, camera_name: str) -> Optional[str]:
        """Resolve camera name to hardware ID."""
        return self._camera_mapping.get(camera_name)
        
    def get_stereo_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """Get stereo pair."""
        return self._camera_mapping.get("left"), self._camera_mapping.get("right")


# Convenience functions
def create_camera_discovery(client) -> CameraDiscovery:
    """Create a camera discovery instance."""
    return CameraDiscovery(client)


def create_static_mapper(camera_list: List[str]) -> CameraMapper:
    """Create a static camera mapper for testing."""
    return CameraMapper(camera_list)