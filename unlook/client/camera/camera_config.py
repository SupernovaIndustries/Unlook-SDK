"""
Camera configuration module for the UnLook scanner.
Provides classes and utilities for configuring camera parameters.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Tuple


class ColorMode(Enum):
    """Camera color mode options."""
    COLOR = "color"
    GRAYSCALE = "grayscale"
    
    
class CompressionFormat(Enum):
    """Image compression format options."""
    JPEG = "jpeg"
    PNG = "png"
    RAW = "raw"  # Uncompressed
    
    
class ImageQualityPreset(Enum):
    """Image quality presets."""
    LOWEST = "lowest"     # Maximum compression, lowest quality
    LOW = "low"           # High compression, lower quality
    MEDIUM = "medium"     # Balanced compression/quality
    HIGH = "high"         # Low compression, higher quality
    HIGHEST = "highest"   # Minimal compression, highest quality
    LOSSLESS = "lossless" # No quality loss (PNG)


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
        self.exposure_compensation = 0  # EV compensation (-4.0 to 4.0)
        
        # Gain settings
        self.gain = 1.0            # Camera gain value
        self.auto_gain = False     # Whether to use auto gain
        self.iso = None            # ISO setting (camera dependent)
        
        # Image quality and format
        self.color_mode = ColorMode.COLOR  # Color or grayscale
        self.compression_format = CompressionFormat.JPEG  # Image format
        self.jpeg_quality = 85     # JPEG compression quality (0-100)
        self.resolution = None     # Resolution (width, height) or None for default
        self.crop_region = None    # ROI crop (x, y, width, height) or None for full frame
        
        # Image processing settings
        self.sharpness = 0         # Sharpness adjustment (0 = default, 1 = max)
        self.denoise = False       # Noise reduction 
        self.hdr_mode = False      # High Dynamic Range mode
        self.stabilization = False # Image stabilization (if supported)
        
        # Advanced settings
        self.binning = 1           # Pixel binning factor (1, 2, 4)
        self.framerate = 30        # Target framerate in FPS
        self.contrast = 1.0        # Contrast enhancement factor
        self.brightness = 0.0      # Brightness adjustment (-1.0 to 1.0)
        self.saturation = 1.0      # Color saturation (grayscale=0, normal=1.0)
        self.gamma = 1.0           # Gamma correction (1.0 = linear)
        
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
                            saturation: float = None,
                            sharpness: float = None,
                            gamma: float = None) -> 'CameraConfig':
        """
        Set image adjustment parameters.
        
        Args:
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast enhancement factor
            saturation: Color saturation
            sharpness: Sharpness adjustment (0=default, 1=max)
            gamma: Gamma correction (1.0=linear)
            
        Returns:
            Self for method chaining
        """
        if brightness is not None:
            self.brightness = brightness
        
        if contrast is not None:
            self.contrast = contrast
        
        if saturation is not None:
            self.saturation = saturation
            
        if sharpness is not None:
            self.sharpness = sharpness
            
        if gamma is not None:
            self.gamma = gamma
        
        return self
        
    def set_quality_preset(self, preset: ImageQualityPreset) -> 'CameraConfig':
        """
        Apply a quality preset to the camera configuration.
        
        Args:
            preset: Quality preset to apply
            
        Returns:
            Self for method chaining
        """
        if preset == ImageQualityPreset.LOWEST:
            self.compression_format = CompressionFormat.JPEG
            self.jpeg_quality = 40
            self.denoise = True
            
        elif preset == ImageQualityPreset.LOW:
            self.compression_format = CompressionFormat.JPEG
            self.jpeg_quality = 60
            self.denoise = True
            
        elif preset == ImageQualityPreset.MEDIUM:
            self.compression_format = CompressionFormat.JPEG
            self.jpeg_quality = 75
            self.denoise = False
            
        elif preset == ImageQualityPreset.HIGH:
            self.compression_format = CompressionFormat.JPEG
            self.jpeg_quality = 90
            self.denoise = False
            
        elif preset == ImageQualityPreset.HIGHEST:
            self.compression_format = CompressionFormat.JPEG
            self.jpeg_quality = 98
            self.denoise = False
            
        elif preset == ImageQualityPreset.LOSSLESS:
            self.compression_format = CompressionFormat.PNG
            self.denoise = False
            
        return self
        
    def set_compression(self, 
                      format: CompressionFormat,
                      jpeg_quality: int = None) -> 'CameraConfig':
        """
        Set the image compression settings.
        
        Args:
            format: Compression format (JPEG, PNG, RAW)
            jpeg_quality: JPEG quality (0-100), only applicable for JPEG format
            
        Returns:
            Self for method chaining
        """
        self.compression_format = format
        
        if format == CompressionFormat.JPEG and jpeg_quality is not None:
            self.jpeg_quality = max(0, min(100, jpeg_quality))
            
        return self
        
    def set_image_processing(self,
                           denoise: bool = None,
                           hdr_mode: bool = None,
                           stabilization: bool = None) -> 'CameraConfig':
        """
        Set image processing options.
        
        Args:
            denoise: Enable noise reduction
            hdr_mode: Enable High Dynamic Range mode
            stabilization: Enable image stabilization
            
        Returns:
            Self for method chaining
        """
        if denoise is not None:
            self.denoise = denoise
            
        if hdr_mode is not None:
            self.hdr_mode = hdr_mode
            
        if stabilization is not None:
            self.stabilization = stabilization
            
        return self
        
    def set_crop_region(self, x: int, y: int, width: int, height: int) -> 'CameraConfig':
        """
        Set a region of interest (ROI) crop.
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: ROI width
            height: ROI height
            
        Returns:
            Self for method chaining
        """
        self.crop_region = (x, y, width, height)
        return self
        
    def reset_crop_region(self) -> 'CameraConfig':
        """
        Reset to use the full camera frame (no cropping).
        
        Returns:
            Self for method chaining
        """
        self.crop_region = None
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
            # Exposure settings
            "exposure_time": self.exposure_time,
            "auto_exposure": self.auto_exposure,
            "exposure_compensation": self.exposure_compensation,
            
            # Gain settings
            "gain": self.gain,
            "auto_gain": self.auto_gain,
            "iso": self.iso,
            
            # Image quality and format
            "color_mode": self.color_mode.value,
            "compression_format": self.compression_format.value,
            "jpeg_quality": self.jpeg_quality,
            "resolution": self.resolution,
            "crop_region": self.crop_region,
            
            # Image processing settings
            "sharpness": self.sharpness,
            "denoise": self.denoise,
            "hdr_mode": self.hdr_mode,
            "stabilization": self.stabilization,
            
            # Advanced settings
            "binning": self.binning,
            "framerate": self.framerate,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "saturation": self.saturation,
            "gamma": self.gamma,
            
            # Backend-specific settings
            "backend_settings": self.backend_settings
        }