"""
Configuration classes for 3D scanning.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from .camera_config import ColorMode, CompressionFormat, ImageQualityPreset


class PatternType(Enum):
    """Types of projection patterns supported for 3D scanning."""
    HORIZONTAL_LINES = "horizontal_lines"
    VERTICAL_LINES = "vertical_lines"
    GRAY_CODE = "gray_code"
    PHASE_SHIFT = "phase_shift"
    ADVANCED_GRAY_CODE = "advanced_gray_code"  # Advanced Gray code from structured-light-stereo
    ADVANCED_PHASE_SHIFT = "advanced_phase_shift"  # Advanced phase shift from structured-light-stereo
    MULTI_SCALE = "multi_scale"  # Multi-scale Gray code patterns with varying line widths
    MULTI_FREQUENCY = "multi_frequency"  # Multi-frequency phase shift patterns
    VARIABLE_WIDTH = "variable_width"  # Variable width Gray code patterns


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


class ScanConfig:
    """
    Configuration class for 3D scanning with the robust structured light scanner.
    """
    
    def __init__(self):
        """Initialize scanner configuration with default values."""
        # Pattern generation settings
        self.pattern_resolution = (1920, 1080)
        self.num_gray_codes = 10
        self.num_phase_shifts = 8
        self.phase_shift_frequencies = [1, 8, 16]
        self.min_bit_width = 4  # Minimum bit width for variable width patterns
        self.max_bit_width = 10  # Maximum bit width for variable width patterns
        
        # Scanning settings
        self.quality = ScanningQuality.HIGH
        self.mode = ScanningMode.SINGLE
        self.pattern_type = PatternType.MULTI_SCALE  # Use the new multi-scale patterns by default
        
        # Image acquisition settings
        self.exposure_time = 10000  # in microseconds
        self.gain = 1.0
        self.image_size = (1280, 720)
        self.color_mode = ColorMode.GRAYSCALE
        
        # Output settings
        self.save_raw_images = True
        self.output_directory = "scan_output"
        self.save_debug_info = True
        
        # Processing settings
        self.filter_points = True
        self.mesh_quality = 9  # Poisson reconstruction depth
        self.smoothing_iterations = 5
        
        # Advanced settings
        self.custom_camera_ids = None
        self.custom_pattern_sequence = None
    
    @classmethod
    def create_preset(cls, quality: ScanningQuality) -> 'ScanConfig':
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
            config.num_gray_codes = 8
            config.num_phase_shifts = 4
            config.phase_shift_frequencies = [1]
            config.mesh_quality = 7
            config.smoothing_iterations = 2
            config.pattern_type = PatternType.MULTI_SCALE
            config.min_bit_width = 3
            config.max_bit_width = 8
        
        elif quality == ScanningQuality.MEDIUM:
            config.num_gray_codes = 10
            config.num_phase_shifts = 4
            config.phase_shift_frequencies = [1, 8]
            config.mesh_quality = 8
            config.smoothing_iterations = 3
            config.pattern_type = PatternType.MULTI_SCALE
            config.min_bit_width = 4
            config.max_bit_width = 10
        
        elif quality == ScanningQuality.HIGH:
            config.num_gray_codes = 10
            config.num_phase_shifts = 8
            config.phase_shift_frequencies = [1, 8, 16]
            config.mesh_quality = 9
            config.smoothing_iterations = 5
            config.pattern_type = PatternType.VARIABLE_WIDTH
            config.min_bit_width = 4
            config.max_bit_width = 10
        
        elif quality == ScanningQuality.ULTRA:
            config.num_gray_codes = 12
            config.num_phase_shifts = 12
            config.phase_shift_frequencies = [1, 8, 16, 32, 64]
            config.mesh_quality = 10
            config.smoothing_iterations = 7
            # Use both variable width and multi-frequency patterns for ultra quality
            config.pattern_type = PatternType.MULTI_FREQUENCY
        
        return config
    
    def set_image_size(self, width: int, height: int) -> 'ScanConfig':
        """
        Set the camera image size.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Self for method chaining
        """
        self.image_size = (width, height)
        return self
    
    def set_pattern_resolution(self, width: int, height: int) -> 'ScanConfig':
        """
        Set the projector pattern resolution.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
            
        Returns:
            Self for method chaining
        """
        self.pattern_resolution = (width, height)
        return self
    
    def set_camera_settings(self, exposure_time: int = None, gain: float = None) -> 'ScanConfig':
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
    
    def set_gray_code_settings(self, num_codes: int) -> 'ScanConfig':
        """
        Set Gray code settings.
        
        Args:
            num_codes: Number of Gray code patterns
            
        Returns:
            Self for method chaining
        """
        self.num_gray_codes = num_codes
        return self
    
    def set_phase_shift_settings(self, num_shifts: int, frequencies: List[int]) -> 'ScanConfig':
        """
        Set phase shift settings.
        
        Args:
            num_shifts: Number of phase shifts per frequency
            frequencies: List of frequencies to use
            
        Returns:
            Self for method chaining
        """
        self.num_phase_shifts = num_shifts
        self.phase_shift_frequencies = frequencies
        return self
        
    def set_variable_width_settings(self, min_bits: int, max_bits: int) -> 'ScanConfig':
        """
        Set variable width Gray code pattern settings.
        
        Args:
            min_bits: Minimum number of bits (larger stripes)
            max_bits: Maximum number of bits (finer stripes)
            
        Returns:
            Self for method chaining
        """
        self.min_bit_width = min_bits
        self.max_bit_width = max_bits
        return self
    
    def set_mesh_settings(self, quality: int = None, smoothing: int = None) -> 'ScanConfig':
        """
        Set mesh generation settings.
        
        Args:
            quality: Mesh quality (Poisson reconstruction depth)
            smoothing: Number of smoothing iterations
            
        Returns:
            Self for method chaining
        """
        if quality is not None:
            self.mesh_quality = quality
        
        if smoothing is not None:
            self.smoothing_iterations = smoothing
        
        return self
    
    def set_output_settings(self, 
                         save_raw_images: bool = None, 
                         save_debug_info: bool = None,
                         output_directory: str = None) -> 'ScanConfig':
        """
        Set output settings.
        
        Args:
            save_raw_images: Whether to save raw captured images
            save_debug_info: Whether to save debug information
            output_directory: Directory for saving results
            
        Returns:
            Self for method chaining
        """
        if save_raw_images is not None:
            self.save_raw_images = save_raw_images
        
        if save_debug_info is not None:
            self.save_debug_info = save_debug_info
        
        if output_directory is not None:
            self.output_directory = output_directory
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            "pattern_resolution": self.pattern_resolution,
            "num_gray_codes": self.num_gray_codes,
            "num_phase_shifts": self.num_phase_shifts,
            "phase_shift_frequencies": self.phase_shift_frequencies,
            "min_bit_width": self.min_bit_width,
            "max_bit_width": self.max_bit_width,
            "quality": self.quality.value,
            "mode": self.mode.value,
            "pattern_type": self.pattern_type.value,
            "exposure_time": self.exposure_time,
            "gain": self.gain,
            "image_size": self.image_size,
            "save_raw_images": self.save_raw_images,
            "output_directory": self.output_directory,
            "save_debug_info": self.save_debug_info,
            "filter_points": self.filter_points,
            "mesh_quality": self.mesh_quality,
            "smoothing_iterations": self.smoothing_iterations,
        }
        
        # Add enum values correctly
        if self.color_mode is not None:
            config_dict["color_mode"] = self.color_mode.value
        
        # Add conditionally present settings
        if self.custom_camera_ids is not None:
            config_dict["custom_camera_ids"] = self.custom_camera_ids
        
        if self.custom_pattern_sequence is not None:
            config_dict["custom_pattern_sequence"] = self.custom_pattern_sequence
        
        return config_dict


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
        self.color_mode = None  # Uses default based on pattern type
        self.use_hdr = False  # High Dynamic Range mode
        self.image_quality_preset = None  # Quality preset (None = manual settings)
        self.image_format = None  # Defaults to JPEG

        # Image processing
        self.use_denoise = False  # Noise reduction
        self.sharpness = 0.0  # Default sharpness
        self.contrast = 1.0  # Default contrast
        self.brightness = 0.0  # Default brightness

        # Resolution and cropping
        self.custom_resolution = None  # Custom resolution (width, height)
        self.crop_region = None  # ROI crop (x, y, width, height)

        # Scan settings
        self.use_stereo = True  # Use stereo cameras if available
        self.real_time_processing = True  # Process during scanning
        self.save_raw_images = False  # Save raw captured images
        self.output_directory = "scan_output"  # Directory for saving results

        # GPU and acceleration settings
        self.use_gpu = True  # Main GPU flag (controls all GPU features)
        self.opencv_cuda_enabled = True  # Specific flag for OpenCV CUDA operations
        self.use_neural_network = True  # Controls neural network enhancement
        self.nn_denoise_strength = 0.5  # Neural network denoising strength (0-1)
        self.nn_upsample = False  # Whether to upsample point clouds with neural network
        self.nn_target_points = None  # Target number of points for upsampling

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
        
    def set_image_quality(self, preset: ImageQualityPreset) -> 'RealTimeScannerConfig':
        """
        Set image quality preset for scanning.
        
        Args:
            preset: Image quality preset
            
        Returns:
            Self for method chaining
        """
        self.image_quality_preset = preset
        
        # Update JPEG quality based on preset
        if preset == ImageQualityPreset.LOWEST:
            self.jpeg_quality = 40
        elif preset == ImageQualityPreset.LOW:
            self.jpeg_quality = 60
        elif preset == ImageQualityPreset.MEDIUM:
            self.jpeg_quality = 75
        elif preset == ImageQualityPreset.HIGH:
            self.jpeg_quality = 90
        elif preset == ImageQualityPreset.HIGHEST:
            self.jpeg_quality = 98
        
        return self
        
    def set_image_format(self, format: CompressionFormat) -> 'RealTimeScannerConfig':
        """
        Set image format for scanning.
        
        Args:
            format: Image compression format
            
        Returns:
            Self for method chaining
        """
        self.image_format = format
        return self
        
    def set_color_mode(self, mode: ColorMode) -> 'RealTimeScannerConfig':
        """
        Set color mode for scanning.
        
        Args:
            mode: Color mode (COLOR or GRAYSCALE)
            
        Returns:
            Self for method chaining
        """
        self.color_mode = mode
        return self
        
    def set_image_processing(self, 
                           denoise: bool = None,
                           hdr: bool = None,
                           sharpness: float = None,
                           contrast: float = None,
                           brightness: float = None) -> 'RealTimeScannerConfig':
        """
        Set image processing options.
        
        Args:
            denoise: Enable noise reduction
            hdr: Enable High Dynamic Range mode
            sharpness: Sharpness adjustment (0=default, 1=max)
            contrast: Contrast enhancement factor
            brightness: Brightness adjustment (-1.0 to 1.0)
            
        Returns:
            Self for method chaining
        """
        if denoise is not None:
            self.use_denoise = denoise
            
        if hdr is not None:
            self.use_hdr = hdr
            
        if sharpness is not None:
            self.sharpness = sharpness
            
        if contrast is not None:
            self.contrast = contrast
            
        if brightness is not None:
            self.brightness = brightness
            
        return self
        
    def set_resolution(self, width: int, height: int) -> 'RealTimeScannerConfig':
        """
        Set custom resolution for scanning.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Self for method chaining
        """
        self.custom_resolution = (width, height)
        return self
        
    def set_crop_region(self, x: int, y: int, width: int, height: int) -> 'RealTimeScannerConfig':
        """
        Set region of interest (ROI) crop for scanning.
        
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
        
    def reset_crop_region(self) -> 'RealTimeScannerConfig':
        """
        Reset to use the full camera frame (no cropping).
        
        Returns:
            Self for method chaining
        """
        self.crop_region = None
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
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
            "custom_camera_ids": self.custom_camera_ids,
            
            # Advanced image settings
            "use_denoise": self.use_denoise,
            "use_hdr": self.use_hdr,
            "sharpness": self.sharpness,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "custom_resolution": self.custom_resolution,
            "crop_region": self.crop_region,

            # GPU and acceleration settings
            "use_gpu": self.use_gpu,
            "opencv_cuda_enabled": self.opencv_cuda_enabled,
            "use_neural_network": self.use_neural_network,
            "nn_denoise_strength": self.nn_denoise_strength,
            "nn_upsample": self.nn_upsample,
            "nn_target_points": self.nn_target_points
        }
        
        # Add enum values correctly
        if self.color_mode is not None:
            config_dict["color_mode"] = self.color_mode.value
            
        if self.image_format is not None:
            config_dict["image_format"] = self.image_format.value
            
        if self.image_quality_preset is not None:
            config_dict["image_quality_preset"] = self.image_quality_preset.value
            
        return config_dict