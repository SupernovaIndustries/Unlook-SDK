"""
Scanning configuration management for UnLook SDK.

This module provides centralized configuration for different scanning modules,
including pattern generation parameters, calibration settings, and processing
options. It works with the module selector to provide optimal configurations
for each hardware setup.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    """Configuration for pattern generation."""
    type: str  # "sinusoidal", "gray_code", "random_dot", etc.
    frequencies: list = None  # For sinusoidal patterns
    phase_steps: int = 4  # Number of phase shifts
    bit_depth: int = 8  # Pattern bit depth
    use_gamma_correction: bool = True
    gamma_value: float = 2.2
    brightness: int = 80  # 0-100
    contrast: int = 90  # 0-100


@dataclass
class CalibrationConfig:
    """Configuration for calibration parameters."""
    mode: str  # "stereo", "projector_camera", "single_camera"
    checkerboard_size: tuple = (9, 6)
    square_size: float = 25.0  # mm
    min_calibration_images: int = 20
    calibration_flags: int = 0
    use_rational_model: bool = True
    fix_principal_point: bool = False
    fix_aspect_ratio: bool = False


@dataclass
class ProcessingConfig:
    """Configuration for 3D processing parameters."""
    use_gpu: bool = True
    point_cloud_colorized: bool = True
    downsample_voxel_size: float = 2.0  # mm
    statistical_outlier_neighbors: int = 20
    statistical_outlier_std_ratio: float = 2.0
    normal_estimation_radius: float = 10.0  # mm
    poisson_depth: int = 9
    confidence_threshold: float = 0.5


@dataclass
class LEDConfig:
    """Configuration for LED controller."""
    led1_current: int = 0  # mA (0-450)
    led2_current: int = 250  # mA (0-450)
    led1_enabled: bool = True
    led2_enabled: bool = True
    pulse_mode: bool = False
    pulse_duration: int = 100  # ms


class ScanningConfig:
    """
    Manages scanning configurations for different modules.
    """
    
    # Default configurations for each module type
    DEFAULT_CONFIGS = {
        "phase_shift_structured_light": {
            "pattern": PatternConfig(
                type="sinusoidal",
                frequencies=[1, 8, 64],
                phase_steps=4,
                brightness=80,
                contrast=90
            ),
            "calibration": CalibrationConfig(
                mode="projector_camera",
                checkerboard_size=(9, 6),
                square_size=25.0,
                min_calibration_images=25
            ),
            "processing": ProcessingConfig(
                use_gpu=True,
                downsample_voxel_size=1.5,
                confidence_threshold=0.6
            ),
            "led": LEDConfig(
                led1_current=0,
                led2_current=250,
                led1_enabled=False,
                led2_enabled=True
            )
        },
        "stereo_vision": {
            "pattern": PatternConfig(
                type="random_dot",
                brightness=100,
                contrast=100
            ),
            "calibration": CalibrationConfig(
                mode="stereo",
                checkerboard_size=(9, 6),
                square_size=25.0,
                min_calibration_images=30,
                use_rational_model=True
            ),
            "processing": ProcessingConfig(
                use_gpu=True,
                downsample_voxel_size=2.0,
                confidence_threshold=0.7
            ),
            "led": LEDConfig(
                led1_current=450,
                led2_current=150,
                led1_enabled=True,
                led2_enabled=True
            )
        },
        "hybrid_stereo_structured": {
            "pattern": PatternConfig(
                type="hybrid_sinusoidal",
                frequencies=[1, 8, 32, 64],
                phase_steps=4,
                brightness=75,
                contrast=85
            ),
            "calibration": CalibrationConfig(
                mode="stereo",
                checkerboard_size=(11, 8),
                square_size=20.0,
                min_calibration_images=40,
                use_rational_model=True
            ),
            "processing": ProcessingConfig(
                use_gpu=True,
                downsample_voxel_size=1.0,
                confidence_threshold=0.5,
                poisson_depth=10
            ),
            "led": LEDConfig(
                led1_current=350,
                led2_current=200,
                led1_enabled=True,
                led2_enabled=True
            )
        }
    }
    
    def __init__(self, module_name: str, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize scanning configuration.
        
        Args:
            module_name: Name of the scanning module
            custom_config: Optional custom configuration to override defaults
        """
        self.module_name = module_name
        self.config = self._load_default_config(module_name)
        
        # Apply custom configuration if provided
        if custom_config:
            self._apply_custom_config(custom_config)
            
        # Load calibration data if available
        self.calibration_data = self._load_calibration_data()
        
    def _load_default_config(self, module_name: str) -> Dict[str, Any]:
        """Load default configuration for the module."""
        if module_name in self.DEFAULT_CONFIGS:
            # Convert dataclasses to dicts
            config = {}
            for key, value in self.DEFAULT_CONFIGS[module_name].items():
                config[key] = asdict(value)
            return config
        else:
            logger.warning(f"No default config for module {module_name}, using phase_shift defaults")
            return self._load_default_config("phase_shift_structured_light")
    
    def _apply_custom_config(self, custom_config: Dict[str, Any]):
        """Apply custom configuration overrides."""
        for section, params in custom_config.items():
            if section in self.config and isinstance(params, dict):
                self.config[section].update(params)
            else:
                self.config[section] = params
                
        logger.info(f"Applied custom configuration for {self.module_name}")
    
    def _load_calibration_data(self) -> Optional[Dict[str, Any]]:
        """Load existing calibration data if available."""
        # Check standard calibration locations
        calibration_paths = [
            Path("calibration_2k_fixed.json"),
            Path("calibration_2k.json"),
            Path("unlook/calibration/custom/stereo_calibration.json"),
            Path("unlook/calibration/custom/enhanced_stereo_calibration.json")
        ]
        
        for cal_path in calibration_paths:
            if cal_path.exists():
                try:
                    with open(cal_path, 'r') as f:
                        cal_data = json.load(f)
                    logger.info(f"Loaded calibration data from {cal_path}")
                    return cal_data
                except Exception as e:
                    logger.error(f"Error loading calibration from {cal_path}: {e}")
                    
        logger.warning("No calibration data found")
        return None
    
    def get_pattern_config(self) -> PatternConfig:
        """Get pattern configuration as PatternConfig object."""
        return PatternConfig(**self.config.get("pattern", {}))
    
    def get_calibration_config(self) -> CalibrationConfig:
        """Get calibration configuration as CalibrationConfig object."""
        return CalibrationConfig(**self.config.get("calibration", {}))
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration as ProcessingConfig object."""
        return ProcessingConfig(**self.config.get("processing", {}))
    
    def get_led_config(self) -> LEDConfig:
        """Get LED configuration as LEDConfig object."""
        return LEDConfig(**self.config.get("led", {}))
    
    def get_scanner_params(self) -> Dict[str, Any]:
        """
        Get complete scanner parameters for the module.
        
        Returns:
            Dict with all parameters needed for scanning
        """
        params = {
            "module_name": self.module_name,
            "pattern": self.config.get("pattern", {}),
            "calibration": self.config.get("calibration", {}),
            "processing": self.config.get("processing", {}),
            "led": self.config.get("led", {}),
            "has_calibration": self.calibration_data is not None
        }
        
        # Add calibration matrices if available
        if self.calibration_data:
            params["calibration_data"] = {
                "camera_matrix_left": self.calibration_data.get("camera_matrix_left"),
                "camera_matrix_right": self.calibration_data.get("camera_matrix_right"),
                "dist_coeffs_left": self.calibration_data.get("dist_coeffs_left"),
                "dist_coeffs_right": self.calibration_data.get("dist_coeffs_right"),
                "R": self.calibration_data.get("R"),
                "T": self.calibration_data.get("T"),
                "Q": self.calibration_data.get("Q")
            }
            
        return params
    
    def save_config(self, filepath: Path):
        """Save current configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def update_led_config(self, led1: Optional[int] = None, led2: Optional[int] = None):
        """
        Update LED configuration.
        
        Args:
            led1: LED1 current in mA (0-450)
            led2: LED2 current in mA (0-450)
        """
        if led1 is not None:
            self.config["led"]["led1_current"] = max(0, min(450, led1))
        if led2 is not None:
            self.config["led"]["led2_current"] = max(0, min(450, led2))
            
        logger.info(f"Updated LED config: LED1={self.config['led']['led1_current']}mA, LED2={self.config['led']['led2_current']}mA")
    
    def optimize_for_material(self, material_type: str):
        """
        Optimize configuration for specific material types.
        
        Args:
            material_type: "matte", "glossy", "transparent", "dark", "metallic"
        """
        material_adjustments = {
            "matte": {
                "pattern": {"brightness": 80, "contrast": 90},
                "led": {"led2_current": 250}
            },
            "glossy": {
                "pattern": {"brightness": 60, "contrast": 70},
                "led": {"led2_current": 150},
                "processing": {"confidence_threshold": 0.7}
            },
            "transparent": {
                "pattern": {"brightness": 100, "contrast": 100},
                "led": {"led1_current": 450, "led2_current": 300}
            },
            "dark": {
                "pattern": {"brightness": 100, "contrast": 100},
                "led": {"led2_current": 400}
            },
            "metallic": {
                "pattern": {"brightness": 50, "contrast": 60},
                "led": {"led2_current": 100},
                "processing": {"statistical_outlier_std_ratio": 1.5}
            }
        }
        
        if material_type in material_adjustments:
            adjustments = material_adjustments[material_type]
            for section, params in adjustments.items():
                if section in self.config:
                    self.config[section].update(params)
                    
            logger.info(f"Optimized configuration for {material_type} material")
        else:
            logger.warning(f"Unknown material type: {material_type}")


def create_scanning_config(module_name: str, 
                          hardware_config: Dict[str, Any],
                          custom_params: Optional[Dict[str, Any]] = None) -> ScanningConfig:
    """
    Create a scanning configuration for the given module and hardware.
    
    Args:
        module_name: Name of the scanning module
        hardware_config: Hardware configuration from detector
        custom_params: Optional custom parameters
        
    Returns:
        ScanningConfig instance
    """
    # Auto-adjust some parameters based on hardware
    auto_adjustments = {}
    
    # Adjust for camera resolution
    cameras = hardware_config.get("cameras", [])
    if cameras:
        # Assume 2K cameras if detected
        auto_adjustments["processing"] = {
            "downsample_voxel_size": 1.5  # Finer voxel size for 2K
        }
    
    # Merge auto adjustments with custom params
    if custom_params:
        for key, value in auto_adjustments.items():
            if key in custom_params:
                custom_params[key].update(value)
            else:
                custom_params[key] = value
    else:
        custom_params = auto_adjustments
    
    return ScanningConfig(module_name, custom_params)


if __name__ == "__main__":
    # Test configuration creation
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Test different module configurations
    modules = [
        "phase_shift_structured_light",
        "stereo_vision",
        "hybrid_stereo_structured"
    ]
    
    for module in modules:
        print(f"\n\nConfiguration for {module}:")
        print("-" * 60)
        
        config = ScanningConfig(module)
        params = config.get_scanner_params()
        
        print(json.dumps(params, indent=2))
        
        # Test material optimization
        print(f"\nOptimizing for glossy material...")
        config.optimize_for_material("glossy")
        print(f"LED config after optimization: {config.get_led_config()}")