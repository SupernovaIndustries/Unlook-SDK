"""
Unlook Scanning Modules

This package manages specifications and configurations for different Unlook 3D scanning modules.
It provides utilities for loading module specifications, validating configurations against module
capabilities, and supporting auto-discovery of scanning hardware.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)


class ScanningModule:
    """
    Representation of a scanning module with its specifications and capabilities.
    """
    
    def __init__(self, spec_data: Dict[str, Any]):
        """
        Initialize a scanning module from specification data.
        
        Args:
            spec_data: Dictionary containing the module specifications
        """
        self.spec_data = spec_data
        self.module_id = spec_data.get("module_id", "unknown")
        self.module_name = spec_data.get("module_name", "Unknown Module")
        self.module_type = spec_data.get("type", "unknown")
        self.version = spec_data.get("version", "0.0.0")
        
    @property
    def min_distance(self) -> float:
        """Minimum scanning distance in mm"""
        return self.spec_data.get("scanning_capabilities", {}).get("working_volume", {}).get("min_distance_mm", 0.0)
    
    @property
    def max_distance(self) -> float:
        """Maximum scanning distance in mm"""
        return self.spec_data.get("scanning_capabilities", {}).get("working_volume", {}).get("max_distance_mm", 0.0)
    
    @property
    def optimal_distance(self) -> float:
        """Optimal scanning distance in mm"""
        return self.spec_data.get("scanning_capabilities", {}).get("working_volume", {}).get("optimal_distance_mm", 0.0)
    
    @property
    def min_disparity(self) -> float:
        """Minimum disparity in pixels"""
        return self.spec_data.get("scanning_capabilities", {}).get("correspondence", {}).get("min_disparity_px", 0.0)
    
    @property
    def max_disparity(self) -> float:
        """Maximum disparity in pixels"""
        return self.spec_data.get("scanning_capabilities", {}).get("correspondence", {}).get("max_disparity_px", 0.0)
    
    @property
    def epipolar_tolerance(self) -> float:
        """Epipolar tolerance in pixels"""
        return self.spec_data.get("scanning_capabilities", {}).get("correspondence", {}).get("epipolar_tolerance_px", 1.0)
    
    @property
    def focal_length_px(self) -> float:
        """Camera focal length in pixels"""
        return self.spec_data.get("hardware", {}).get("cameras", {}).get("focal_length_px", 0.0)
    
    @property
    def baseline_mm(self) -> float:
        """Stereo baseline in mm"""
        return self.spec_data.get("hardware", {}).get("cameras", {}).get("baseline_mm", 0.0)
    
    @property
    def limitations(self) -> List[str]:
        """Known limitations and issues"""
        return self.spec_data.get("limitations", {}).get("known_issues", [])
    
    def get_resolution_at_distance(self, distance_mm: float) -> float:
        """
        Calculate the depth resolution at a given distance.
        
        Args:
            distance_mm: Distance to object in mm
            
        Returns:
            Estimated depth resolution in mm
        """
        # Use provided values if available
        res_at_distances = self.spec_data.get("scanning_capabilities", {}).get("resolution", {}).get("resolution_at_distances_mm", {})
        distance_str = f"{int(distance_mm)}mm"
        if distance_str in res_at_distances:
            return res_at_distances[distance_str]
        
        # Otherwise calculate using the theoretical formula
        # dZ = Z²/(f*B*dD)
        f = self.focal_length_px
        b = self.baseline_mm
        min_disparity_diff = self.spec_data.get("scanning_capabilities", {}).get("resolution", {}).get("min_detectable_disparity_px", 0.1)
        
        if f <= 0 or b <= 0 or min_disparity_diff <= 0:
            logger.warning("Cannot calculate resolution: invalid camera parameters")
            return 0.0
        
        return (distance_mm ** 2) / (f * b * min_disparity_diff)
    
    def is_parameter_valid(self, parameter_name: str, value: Union[float, int, str]) -> bool:
        """
        Check if a parameter value is valid for this module.
        
        Args:
            parameter_name: Name of the parameter to check
            value: Value to validate
            
        Returns:
            True if the parameter value is valid, False otherwise
        """
        # Implementation depends on parameter types
        if parameter_name == "distance_mm":
            return self.min_distance <= float(value) <= self.max_distance
        elif parameter_name == "disparity_px":
            return self.min_disparity <= float(value) <= self.max_disparity
        elif parameter_name == "epipolar_tolerance_px":
            return 0 < float(value) <= self.epipolar_tolerance * 2
        
        # For other parameters, assume valid
        return True
    
    def get_scan_mode_parameters(self, mode: str) -> Dict[str, Any]:
        """
        Get the parameters for a specific scan mode.
        
        Args:
            mode: Scan mode name (e.g., "high_quality", "balanced", "fast", "real_time")
            
        Returns:
            Dictionary of scan parameters for the mode
        """
        modes = self.spec_data.get("scanning_capabilities", {}).get("performance", {}).get("capture_modes", {})
        return modes.get(mode, {})
    
    def __str__(self) -> str:
        """String representation of the module"""
        return f"{self.module_name} (v{self.version}, ID: {self.module_id})"
    
    def __repr__(self) -> str:
        """Representation of the module"""
        return f"ScanningModule({self.module_id}, {self.module_type}, v{self.version})"


def load_module_specs(module_type: Optional[str] = None) -> List[ScanningModule]:
    """
    Load specifications for all available scanning modules.
    
    Args:
        module_type: Optional filter for module type
        
    Returns:
        List of ScanningModule objects
    """
    modules = []
    
    # Get the directory containing module specifications
    spec_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all JSON files in the directory
    for filename in os.listdir(spec_dir):
        if filename.endswith("_specs.json"):
            try:
                with open(os.path.join(spec_dir, filename), 'r') as f:
                    spec_data = json.load(f)
                    
                    # Filter by module type if specified
                    if module_type is None or spec_data.get("type") == module_type:
                        modules.append(ScanningModule(spec_data))
            except Exception as e:
                logger.error(f"Error loading module spec from {filename}: {e}")
    
    return modules


def get_module_by_id(module_id: str) -> Optional[ScanningModule]:
    """
    Get a scanning module by its ID.
    
    Args:
        module_id: Module identifier
        
    Returns:
        ScanningModule if found, None otherwise
    """
    for module in load_module_specs():
        if module.module_id == module_id:
            return module
    return None


def get_default_structured_light_module() -> Optional[ScanningModule]:
    """
    Get the default structured light module.
    
    Returns:
        Default structured light module if available, None otherwise
    """
    modules = load_module_specs("structured_light")
    return modules[0] if modules else None