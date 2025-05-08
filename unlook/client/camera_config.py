"""
Camera configuration module for the UnLook scanner.
Provides classes and utilities for configuring camera parameters.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union


class ColorMode(Enum):
    """Camera color mode options."""
    COLOR = "color"
    GRAYSCALE = "grayscale"


class CameraConfig:
    """
    Configuration class for camera settings.
    This allows fine-grained control of camera parameters.
    """
    
    def __init__(self):
        """Initialize camera configuration with default values."""
        # Exposure settings
        self.exposure_time = 10000  # in microseconds
        self.auto_exposure = False  # Whether to use auto exposure
        
        # Gain settings
        self.gain = 1.0            # Camera gain value
        self.auto_gain = False     # Whether to use auto gain
        
        # Image quality and format
        self.color_mode = ColorMode.COLOR  # Color or grayscale
        self.jpeg_quality = 85     # JPEG compression quality (0-100)
        self.resolution = None     # Resolution (width, height) or None for default
        
        # Advanced settings
        self.binning = 1           # Pixel binning factor (1, 2, 4)
        self.framerate = 30        # Target framerate in FPS
        self.contrast = 1.0        # Contrast enhancement factor
        self.brightness = 0.0      # Brightness adjustment (-1.0 to 1.0)
        self.saturation = 1.0      # Color saturation (grayscale=0, normal=1.0)
        
        # Backend-specific settings
        self.backend_settings = {}  # Custom settings for specific camera backends
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'CameraConfig':
        """
        Create a configuration preset for specific scanning scenarios.
        
        Args:
            preset_name: Preset name ('scanning', 'streaming', 'high_quality', 'low_light')
            
        Returns:
            Configuration object with preset values
        """
        config = cls()
        
        if preset_name == "scanning":
            # Optimized for structured light scanning
            config.exposure_time = 10000  # 10ms exposure
            config.gain = 1.0
            config.auto_exposure = False
            config.auto_gain = False
            config.color_mode = ColorMode.GRAYSCALE
            config.jpeg_quality = 90
            config.contrast = 1.2
            
        elif preset_name == "streaming":
            # Optimized for low-latency streaming
            config.exposure_time = 16000  # ~16ms (60fps)
            config.gain = 1.0
            config.auto_exposure = True
            config.auto_gain = True
            config.color_mode = ColorMode.COLOR
            config.jpeg_quality = 75
            config.framerate = 60
            
        elif preset_name == "high_quality":
            # Optimized for high-quality captures
            config.exposure_time = 30000  # 30ms
            config.gain = 1.0
            config.auto_exposure = False
            config.auto_gain = False
            config.color_mode = ColorMode.COLOR
            config.jpeg_quality = 95
            config.framerate = 15
            
        elif preset_name == "low_light":
            # Optimized for low-light conditions
            config.exposure_time = 60000  # 60ms
            config.gain = 2.0
            config.auto_exposure = True
            config.auto_gain = True
            config.color_mode = ColorMode.COLOR
            config.jpeg_quality = 85
            config.brightness = 0.1
            config.contrast = 1.1
            
        return config
    
    def set_exposure(self, exposure_time: int, auto_exposure: bool = False) -> 'CameraConfig':
        """
        Set exposure settings.
        
        Args:
            exposure_time: Exposure time in microseconds
            auto_exposure: Whether to enable auto exposure
            
        Returns:
            Self for method chaining
        """
        self.exposure_time = exposure_time
        self.auto_exposure = auto_exposure
        return self
    
    def set_gain(self, gain: float, auto_gain: bool = False) -> 'CameraConfig':
        """
        Set gain settings.
        
        Args:
            gain: Camera gain value
            auto_gain: Whether to enable auto gain
            
        Returns:
            Self for method chaining
        """
        self.gain = gain
        self.auto_gain = auto_gain
        return self
    
    def set_color_mode(self, color_mode: ColorMode) -> 'CameraConfig':
        """
        Set the color mode.
        
        Args:
            color_mode: Color mode to use (COLOR or GRAYSCALE)
            
        Returns:
            Self for method chaining
        """
        self.color_mode = color_mode
        return self
    
    def set_resolution(self, width: int, height: int) -> 'CameraConfig':
        """
        Set the camera resolution.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Self for method chaining
        """
        self.resolution = (width, height)
        return self
    
    def set_framerate(self, fps: int) -> 'CameraConfig':
        """
        Set the target framerate.
        
        Args:
            fps: Frames per second
            
        Returns:
            Self for method chaining
        """
        self.framerate = fps
        return self
    
    def set_image_adjustments(self, 
                            brightness: float = None,
                            contrast: float = None,
                            saturation: float = None) -> 'CameraConfig':
        """
        Set image adjustment parameters.
        
        Args:
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast enhancement factor
            saturation: Color saturation
            
        Returns:
            Self for method chaining
        """
        if brightness is not None:
            self.brightness = brightness
        
        if contrast is not None:
            self.contrast = contrast
        
        if saturation is not None:
            self.saturation = saturation
        
        return self
    
    def add_backend_setting(self, key: str, value: Any) -> 'CameraConfig':
        """
        Add a backend-specific setting.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            Self for method chaining
        """
        self.backend_settings[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "exposure_time": self.exposure_time,
            "auto_exposure": self.auto_exposure,
            "gain": self.gain,
            "auto_gain": self.auto_gain,
            "color_mode": self.color_mode.value,
            "jpeg_quality": self.jpeg_quality,
            "resolution": self.resolution,
            "binning": self.binning,
            "framerate": self.framerate,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "saturation": self.saturation,
            "backend_settings": self.backend_settings
        }