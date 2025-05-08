"""
Configuration class for real-time 3D scanning.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum


class PatternType(Enum):
    """Types of projection patterns supported for 3D scanning."""
    HORIZONTAL_LINES = "horizontal_lines"
    VERTICAL_LINES = "vertical_lines"
    GRAY_CODE = "gray_code"
    PHASE_SHIFT = "phase_shift"


class ScanningQuality(Enum):
    """Scanning quality presets."""
    LOW = "low"          # Faster, lower quality
    MEDIUM = "medium"    # Balanced speed and quality
    HIGH = "high"        # Slower, higher quality
    ULTRA = "ultra"      # Very slow, maximum quality


class ScanningMode(Enum):
    """Scanning modes."""
    SINGLE = "single"          # One-time scan
    CONTINUOUS = "continuous"  # Continuous scanning
    TRIGGERED = "triggered"    # Scan on trigger


class RealTimeScannerConfig:
    """
    Configuration class for real-time 3D scanning.
    This class provides settings for 3D scanning with the UnLook scanner.
    """

    def __init__(self):
        """Initialize scanner configuration with default values."""
        # Scanning mode and quality
        self.mode = ScanningMode.SINGLE
        self.quality = ScanningQuality.MEDIUM
        
        # Pattern projection settings
        self.pattern_type = PatternType.PHASE_SHIFT
        self.pattern_steps = 8  # Number of phase shifts
        self.pattern_interval = 0.1  # Time between patterns in seconds
        
        # Camera settings
        self.exposure_time = 10000  # in microseconds
        self.gain = 1.0
        self.jpeg_quality = 85
        
        # Scan settings
        self.use_stereo = True  # Use stereo cameras if available
        self.real_time_processing = True  # Process during scanning
        self.save_raw_images = False  # Save raw captured images
        self.output_directory = "scan_output"  # Directory for saving results
        
        # Advanced settings
        self.custom_pattern_sequence = None  # Custom pattern sequence
        self.custom_camera_ids = None  # Specific cameras to use
    
    @classmethod
    def create_preset(cls, quality: ScanningQuality) -> 'RealTimeScannerConfig':
        """
        Create a configuration preset based on quality setting.
        
        Args:
            quality: Quality preset to use
            
        Returns:
            Configuration object with preset values
        """
        config = cls()
        config.quality = quality
        
        # Adjust settings based on quality preset
        if quality == ScanningQuality.LOW:
            config.pattern_steps = 4
            config.pattern_interval = 0.05
            config.jpeg_quality = 75
        
        elif quality == ScanningQuality.MEDIUM:
            config.pattern_steps = 8
            config.pattern_interval = 0.1
            config.jpeg_quality = 85
        
        elif quality == ScanningQuality.HIGH:
            config.pattern_steps = 12
            config.pattern_interval = 0.15
            config.jpeg_quality = 92
        
        elif quality == ScanningQuality.ULTRA:
            config.pattern_steps = 16
            config.pattern_interval = 0.2
            config.jpeg_quality = 98
        
        return config
    
    def set_pattern_type(self, pattern_type: PatternType, steps: int = None) -> 'RealTimeScannerConfig':
        """
        Set the pattern type for scanning.
        
        Args:
            pattern_type: Type of pattern to use
            steps: Number of pattern steps/phases (optional)
            
        Returns:
            Self for method chaining
        """
        self.pattern_type = pattern_type
        
        if steps is not None:
            self.pattern_steps = steps
        
        return self
    
    def set_camera_settings(self, exposure_time: int = None, gain: float = None) -> 'RealTimeScannerConfig':
        """
        Set camera exposure and gain settings.
        
        Args:
            exposure_time: Exposure time in microseconds
            gain: Camera gain value
            
        Returns:
            Self for method chaining
        """
        if exposure_time is not None:
            self.exposure_time = exposure_time
        
        if gain is not None:
            self.gain = gain
        
        return self
    
    def set_mode(self, mode: ScanningMode) -> 'RealTimeScannerConfig':
        """
        Set the scanning mode.
        
        Args:
            mode: Scanning mode to use
            
        Returns:
            Self for method chaining
        """
        self.mode = mode
        return self
    
    def use_specific_cameras(self, camera_ids: List[str]) -> 'RealTimeScannerConfig':
        """
        Specify which cameras to use for scanning.
        
        Args:
            camera_ids: List of camera IDs to use
            
        Returns:
            Self for method chaining
        """
        self.custom_camera_ids = camera_ids
        return self
    
    def set_custom_pattern_sequence(self, patterns: List[Dict[str, Any]]) -> 'RealTimeScannerConfig':
        """
        Set a custom pattern sequence for scanning.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Self for method chaining
        """
        self.custom_pattern_sequence = patterns
        return self
    
    def enable_save_raw_images(self, enable: bool = True, output_dir: str = None) -> 'RealTimeScannerConfig':
        """
        Enable saving of raw captured images.
        
        Args:
            enable: Whether to save raw images
            output_dir: Directory to save images to
            
        Returns:
            Self for method chaining
        """
        self.save_raw_images = enable
        
        if output_dir:
            self.output_directory = output_dir
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "mode": self.mode.value,
            "quality": self.quality.value,
            "pattern_type": self.pattern_type.value,
            "pattern_steps": self.pattern_steps,
            "pattern_interval": self.pattern_interval,
            "exposure_time": self.exposure_time,
            "gain": self.gain,
            "jpeg_quality": self.jpeg_quality,
            "use_stereo": self.use_stereo,
            "real_time_processing": self.real_time_processing,
            "save_raw_images": self.save_raw_images,
            "output_directory": self.output_directory,
            "custom_pattern_sequence": self.custom_pattern_sequence,
            "custom_camera_ids": self.custom_camera_ids
        }