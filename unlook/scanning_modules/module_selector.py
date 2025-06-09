"""
Scanning module selection logic for UnLook SDK.

This module automatically selects the appropriate scanning module based on
detected hardware configuration. It determines whether to use:
- Phase Shift Structured Light (single camera + projector)
- Stereo Vision (dual cameras)
- Hybrid modes (dual cameras + projector)
- Fallback modes for limited hardware

The selection ensures optimal performance for the available hardware.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ScanningModule(Enum):
    """Available scanning modules."""
    PHASE_SHIFT_STRUCTURED_LIGHT = "phase_shift_structured_light"
    STEREO_VISION = "stereo_vision"
    HYBRID_STEREO_STRUCTURED = "hybrid_stereo_structured"
    SINGLE_CAMERA_STRUCTURED = "single_camera_structured"
    FALLBACK_MONO = "fallback_mono"
    TOF_ENHANCED = "tof_enhanced"


class ModuleSelector:
    """
    Selects the appropriate scanning module based on hardware configuration.
    """
    
    def __init__(self, hardware_config: Dict[str, Any]):
        """
        Initialize the module selector.
        
        Args:
            hardware_config: Hardware configuration from HardwareDetector
        """
        self.hardware_config = hardware_config
        self.selected_module = None
        self.camera_mapping = {}
        self.module_config = {}
        
    def select_module(self) -> Dict[str, Any]:
        """
        Select the optimal scanning module based on hardware.
        
        Returns:
            Dict containing:
            {
                "module": ScanningModule enum value,
                "camera_mapping": {"primary": "picamera2_0", ...},
                "calibration_mode": "projector_camera" or "stereo",
                "pattern_type": "sinusoidal" or "gray_code",
                "features": ["high_resolution", "real_time", ...],
                "configuration": {module-specific config}
            }
        """
        num_cameras = len(self.hardware_config.get("cameras", []))
        has_projector = self.hardware_config.get("projector") is not None
        has_as1170 = self.hardware_config.get("as1170", False)
        has_tof = self.hardware_config.get("tof_sensor") is not None
        
        logger.info(f"Selecting module: Cameras={num_cameras}, Projector={has_projector}, AS1170={has_as1170}, TOF={has_tof}")
        
        # Decision tree for module selection
        if num_cameras == 1 and has_projector and has_as1170:
            # Single camera + projector + LED = Phase Shift Structured Light
            self._select_phase_shift_module()
            
        elif num_cameras == 2 and has_projector and has_as1170:
            # Dual cameras + projector = Hybrid mode (best quality)
            self._select_hybrid_stereo_structured_module()
            
        elif num_cameras == 2 and has_as1170:
            # Dual cameras + LED = Stereo Vision
            self._select_stereo_vision_module()
            
        elif num_cameras == 1 and has_projector:
            # Single camera + projector (no LED) = Basic structured light
            self._select_single_camera_structured_module()
            
        elif num_cameras >= 1 and has_tof:
            # Any camera + TOF sensor = TOF enhanced mode
            self._select_tof_enhanced_module()
            
        else:
            # Fallback mode for limited hardware
            self._select_fallback_mono_module()
            
        # Build the complete configuration
        config = {
            "module": self.selected_module.value,
            "camera_mapping": self.camera_mapping,
            "calibration_mode": self._get_calibration_mode(),
            "pattern_type": self._get_pattern_type(),
            "features": self._get_module_features(),
            "configuration": self.module_config,
            "hardware_summary": self._get_hardware_summary()
        }
        
        self._log_selection_summary(config)
        
        return config
    
    def _select_phase_shift_module(self):
        """Configure for Phase Shift Structured Light scanning."""
        self.selected_module = ScanningModule.PHASE_SHIFT_STRUCTURED_LIGHT
        
        # Map single camera as primary
        cameras = self.hardware_config.get("cameras", [])
        if cameras:
            self.camera_mapping = {"primary": cameras[0]}
        
        # Phase shift specific configuration
        self.module_config = {
            "pattern_frequencies": [1, 8, 64],  # Multi-frequency approach
            "phase_steps": 4,  # 4-step phase shifting
            "use_gray_code": True,  # Gray code for absolute phase
            "led_intensity": {
                "led1": 0,  # Point projection off
                "led2": 250  # Flood illumination for texture
            },
            "projector_brightness": 80,
            "expected_points": 150000,  # High density point cloud
            "scanning_distance": {
                "min": 300,  # mm
                "optimal": 450,
                "max": 600
            }
        }
        
        logger.info("Selected Phase Shift Structured Light module (single camera + projector)")
    
    def _select_hybrid_stereo_structured_module(self):
        """Configure for Hybrid Stereo + Structured Light scanning."""
        self.selected_module = ScanningModule.HYBRID_STEREO_STRUCTURED
        
        # Map dual cameras
        cameras = self.hardware_config.get("cameras", [])
        if len(cameras) >= 2:
            self.camera_mapping = {
                "left": cameras[0],
                "right": cameras[1],
                "primary": cameras[0]  # For structured light
            }
        
        # Hybrid configuration
        self.module_config = {
            "stereo_baseline": 40,  # mm between cameras
            "use_structured_light": True,
            "pattern_frequencies": [1, 8, 64],
            "phase_steps": 4,
            "fusion_mode": "adaptive",  # Fuse stereo and structured light
            "led_intensity": {
                "led1": 450,  # Point projection for feature matching
                "led2": 200   # Flood for texture
            },
            "expected_points": 250000,  # Very high density
            "scanning_modes": ["fast_preview", "high_quality", "texture_enhanced"]
        }
        
        logger.info("Selected Hybrid Stereo + Structured Light module (best quality)")
    
    def _select_stereo_vision_module(self):
        """Configure for Stereo Vision scanning."""
        self.selected_module = ScanningModule.STEREO_VISION
        
        # Map dual cameras
        cameras = self.hardware_config.get("cameras", [])
        if len(cameras) >= 2:
            self.camera_mapping = {
                "left": cameras[0],
                "right": cameras[1]
            }
        
        # Stereo vision configuration
        self.module_config = {
            "stereo_baseline": 40,  # mm
            "matcher": "SGBM",  # Semi-Global Block Matching
            "disparity_range": [0, 128],
            "block_size": 11,
            "led_intensity": {
                "led1": 450,  # Strong point projection for features
                "led2": 150   # Moderate flood
            },
            "use_pattern_projection": True,  # Random dot pattern via LED1
            "expected_points": 50000,
            "post_processing": ["wls_filter", "confidence_filter"]
        }
        
        logger.info("Selected Stereo Vision module (dual cameras)")
    
    def _select_single_camera_structured_module(self):
        """Configure for basic single camera structured light."""
        self.selected_module = ScanningModule.SINGLE_CAMERA_STRUCTURED
        
        cameras = self.hardware_config.get("cameras", [])
        if cameras:
            self.camera_mapping = {"primary": cameras[0]}
        
        self.module_config = {
            "pattern_type": "gray_code",  # Simple gray code patterns
            "pattern_count": 10,
            "use_phase_shift": False,  # No phase shifting
            "projector_brightness": 100,
            "expected_points": 30000,
            "requires_led_controller": False
        }
        
        logger.info("Selected Single Camera Structured Light module (basic)")
    
    def _select_tof_enhanced_module(self):
        """Configure for TOF sensor enhanced scanning."""
        self.selected_module = ScanningModule.TOF_ENHANCED
        
        cameras = self.hardware_config.get("cameras", [])
        if cameras:
            self.camera_mapping = {"primary": cameras[0]}
            if len(cameras) >= 2:
                self.camera_mapping["secondary"] = cameras[1]
        
        tof_info = self.hardware_config.get("tof_sensor", {})
        
        self.module_config = {
            "tof_sensor": tof_info.get("type", "Unknown"),
            "use_tof_for_scaling": True,
            "auto_focus": True,
            "depth_fusion": True,
            "led_intensity": {
                "led1": 200,
                "led2": 200
            },
            "scanning_modes": ["tof_guided", "hybrid_depth"]
        }
        
        logger.info(f"Selected TOF Enhanced module with {tof_info.get('type', 'Unknown')} sensor")
    
    def _select_fallback_mono_module(self):
        """Configure fallback mode for limited hardware."""
        self.selected_module = ScanningModule.FALLBACK_MONO
        
        cameras = self.hardware_config.get("cameras", [])
        if cameras:
            self.camera_mapping = {"primary": cameras[0]}
        else:
            self.camera_mapping = {}
            
        self.module_config = {
            "mode": "photogrammetry",
            "requires_manual_positioning": True,
            "expected_points": 10000,
            "limitations": ["low_accuracy", "manual_process", "no_real_time"]
        }
        
        logger.warning("Selected Fallback Mono module (limited hardware)")
    
    def _get_calibration_mode(self) -> str:
        """Get the required calibration mode for the selected module."""
        if self.selected_module in [ScanningModule.PHASE_SHIFT_STRUCTURED_LIGHT,
                                    ScanningModule.SINGLE_CAMERA_STRUCTURED]:
            return "projector_camera"
        elif self.selected_module in [ScanningModule.STEREO_VISION,
                                      ScanningModule.HYBRID_STEREO_STRUCTURED]:
            return "stereo"
        else:
            return "single_camera"
    
    def _get_pattern_type(self) -> str:
        """Get the optimal pattern type for the selected module."""
        if self.selected_module == ScanningModule.PHASE_SHIFT_STRUCTURED_LIGHT:
            return "sinusoidal"
        elif self.selected_module == ScanningModule.HYBRID_STEREO_STRUCTURED:
            return "hybrid_sinusoidal"
        elif self.selected_module == ScanningModule.SINGLE_CAMERA_STRUCTURED:
            return "gray_code"
        elif self.selected_module == ScanningModule.STEREO_VISION:
            return "random_dot"
        else:
            return "none"
    
    def _get_module_features(self) -> List[str]:
        """Get feature list for the selected module."""
        features = {
            ScanningModule.PHASE_SHIFT_STRUCTURED_LIGHT: [
                "high_resolution", "accurate_depth", "texture_capture",
                "real_time_preview", "gpu_accelerated"
            ],
            ScanningModule.HYBRID_STEREO_STRUCTURED: [
                "ultra_high_resolution", "best_accuracy", "multi_mode",
                "texture_enhanced", "gpu_accelerated", "professional_grade"
            ],
            ScanningModule.STEREO_VISION: [
                "real_time", "passive_scanning", "outdoor_capable",
                "moderate_resolution", "fast_capture"
            ],
            ScanningModule.SINGLE_CAMERA_STRUCTURED: [
                "basic_3d", "simple_setup", "low_cost",
                "limited_accuracy"
            ],
            ScanningModule.TOF_ENHANCED: [
                "auto_focus", "distance_guided", "hybrid_depth",
                "real_time_feedback"
            ],
            ScanningModule.FALLBACK_MONO: [
                "basic_capture", "manual_process", "minimal_hardware"
            ]
        }
        
        return features.get(self.selected_module, [])
    
    def _get_hardware_summary(self) -> str:
        """Generate a human-readable hardware summary."""
        parts = []
        
        num_cameras = len(self.hardware_config.get("cameras", []))
        if num_cameras == 1:
            parts.append("1 camera")
        elif num_cameras > 1:
            parts.append(f"{num_cameras} cameras")
            
        if self.hardware_config.get("projector"):
            parts.append("DLP projector")
            
        if self.hardware_config.get("as1170"):
            parts.append("LED controller")
            
        if self.hardware_config.get("tof_sensor"):
            tof_type = self.hardware_config["tof_sensor"].get("type", "TOF")
            parts.append(f"{tof_type} sensor")
            
        return " + ".join(parts) if parts else "No hardware detected"
    
    def _log_selection_summary(self, config: Dict[str, Any]):
        """Log a summary of the module selection."""
        logger.info("=" * 60)
        logger.info("Module Selection Summary:")
        logger.info(f"  Selected Module: {self.selected_module.value}")
        logger.info(f"  Hardware: {config['hardware_summary']}")
        logger.info(f"  Calibration Mode: {config['calibration_mode']}")
        logger.info(f"  Pattern Type: {config['pattern_type']}")
        logger.info(f"  Camera Mapping: {config['camera_mapping']}")
        logger.info(f"  Features: {', '.join(config['features'])}")
        logger.info("=" * 60)


def select_scanning_module(hardware_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to select scanning module.
    
    Args:
        hardware_config: Hardware configuration from detect_hardware()
        
    Returns:
        Module configuration dictionary
    """
    selector = ModuleSelector(hardware_config)
    return selector.select_module()


if __name__ == "__main__":
    # Test module selection with different hardware configs
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Test configurations
    test_configs = [
        {
            "name": "Phase Shift Setup (Current)",
            "config": {
                "cameras": ["picamera2_0"],
                "as1170": True,
                "projector": {"type": "DLP342x", "address": "0x1b"},
                "tof_sensor": None
            }
        },
        {
            "name": "Stereo Vision Setup",
            "config": {
                "cameras": ["picamera2_0", "picamera2_1"],
                "as1170": True,
                "projector": None,
                "tof_sensor": None
            }
        },
        {
            "name": "Hybrid Setup (Best)",
            "config": {
                "cameras": ["picamera2_0", "picamera2_1"],
                "as1170": True,
                "projector": {"type": "DLP342x", "address": "0x1b"},
                "tof_sensor": None
            }
        }
    ]
    
    for test in test_configs:
        print(f"\n\nTesting: {test['name']}")
        print("-" * 40)
        
        module_config = select_scanning_module(test['config'])
        print(json.dumps(module_config, indent=2))