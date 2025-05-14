#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scanning Module Example

This example demonstrates how to use the scanning module specifications
to validate scan parameters and provide feedback about scanning capabilities.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the scanning module utilities
from unlook.scanning_modules import (
    load_module_specs,
    get_module_by_id,
    get_default_structured_light_module,
    ScanningModule
)


def print_module_capabilities(module: ScanningModule):
    """Print the capabilities of a scanning module."""
    print(f"\n{'-'*80}")
    print(f"Module: {module.module_name} (v{module.version})")
    print(f"Type: {module.module_type}")
    print(f"ID: {module.module_id}")
    print(f"{'-'*80}")
    
    print("\nWorking Volume:")
    print(f"  Min Distance: {module.min_distance} mm")
    print(f"  Max Distance: {module.max_distance} mm")
    print(f"  Optimal Distance: {module.optimal_distance} mm")
    
    print("\nStereo Parameters:")
    print(f"  Baseline: {module.baseline_mm} mm")
    print(f"  Focal Length: {module.focal_length_px} pixels")
    print(f"  Min Disparity: {module.min_disparity} pixels")
    print(f"  Max Disparity: {module.max_disparity} pixels")
    print(f"  Epipolar Tolerance: {module.epipolar_tolerance} pixels")
    
    print("\nDepth Resolution at Different Distances:")
    distances = [200, 400, 600, 800, 1000]
    for dist in distances:
        res = module.get_resolution_at_distance(dist)
        print(f"  At {dist} mm: {res:.3f} mm")
    
    print("\nCapture Modes:")
    for mode_name in ["high_quality", "balanced", "fast", "real_time"]:
        mode_params = module.get_scan_mode_parameters(mode_name)
        if mode_params:
            print(f"  {mode_name.replace('_', ' ').title()}:")
            print(f"    Pattern Count: {mode_params.get('pattern_count')}")
            print(f"    Capture Time: {mode_params.get('capture_time_sec')} seconds")
            print(f"    Processing Time: {mode_params.get('processing_time_sec')} seconds")
    
    print("\nKnown Limitations:")
    for i, limitation in enumerate(module.limitations, 1):
        print(f"  {i}. {limitation}")
    
    print(f"{'-'*80}")


def validate_scan_parameters(module: ScanningModule, params: dict):
    """Validate scan parameters against module capabilities."""
    print("\nValidating Scan Parameters:")
    
    all_valid = True
    
    for param_name, param_value in params.items():
        valid = module.is_parameter_valid(param_name, param_value)
        status = "✓ Valid" if valid else "✗ Invalid"
        
        if not valid:
            all_valid = False
        
        if param_name == "distance_mm":
            valid_range = f"Valid range: {module.min_distance}-{module.max_distance} mm"
            print(f"  {param_name}: {param_value} mm - {status} ({valid_range})")
        elif param_name == "disparity_px":
            valid_range = f"Valid range: {module.min_disparity}-{module.max_disparity} px"
            print(f"  {param_name}: {param_value} px - {status} ({valid_range})")
        elif param_name == "epipolar_tolerance_px":
            valid_range = f"Valid range: 0-{module.epipolar_tolerance*2} px"
            print(f"  {param_name}: {param_value} px - {status} ({valid_range})")
        else:
            print(f"  {param_name}: {param_value} - {status}")
    
    print(f"\nOverall validation: {'Passed' if all_valid else 'Failed'}")
    
    return all_valid


def calculate_optimal_parameters(module: ScanningModule, distance_mm: float):
    """Calculate optimal scanning parameters for a given distance."""
    print(f"\nOptimal Parameters for {distance_mm} mm Distance:")
    
    # Calculate depth resolution
    resolution = module.get_resolution_at_distance(distance_mm)
    
    # Calculate appropriate disparity range
    focal_length = module.focal_length_px
    baseline = module.baseline_mm
    
    # Estimate disparity for this distance: disparity = (baseline * focal) / distance
    estimated_disparity = (baseline * focal_length) / distance_mm
    
    # Calculate distance-appropriate parameters
    if distance_mm < 300:
        mode = "high_quality"
        exposure = "low"
        gain = "medium"
    elif distance_mm < 600:
        mode = "balanced"
        exposure = "medium"
        gain = "low"
    else:
        mode = "fast"
        exposure = "high"
        gain = "medium"
    
    # Get mode parameters
    mode_params = module.get_scan_mode_parameters(mode)
    
    print(f"  Recommended Scan Mode: {mode.replace('_', ' ').title()}")
    print(f"  Expected Depth Resolution: {resolution:.3f} mm")
    print(f"  Estimated Disparity: {estimated_disparity:.1f} pixels")
    print(f"  Recommended Pattern Count: {mode_params.get('pattern_count')}")
    print(f"  Recommended Exposure: {exposure.title()}")
    print(f"  Recommended Gain: {gain.title()}")
    print(f"  Estimated Capture Time: {mode_params.get('capture_time_sec')} seconds")
    print(f"  Estimated Processing Time: {mode_params.get('processing_time_sec')} seconds")
    
    # Resolution disclaimer
    if resolution > 1.0:
        print(f"\n  ⚠️  Warning: At {distance_mm} mm, resolution is {resolution:.2f} mm,")
        print(f"      which may not be sufficient for detecting small features.")
        print(f"      Consider moving closer to the object if possible.")


def main():
    """Main function demonstrating scanning module usage."""
    print("Unlook Scanning Module Example")
    print("==============================")
    
    # Load all available scanning modules
    modules = load_module_specs()
    print(f"Found {len(modules)} scanning module specifications")
    
    # Get the default structured light module
    sl_module = get_default_structured_light_module()
    
    if not sl_module:
        print("No structured light module found!")
        return
    
    # Display module capabilities
    print_module_capabilities(sl_module)
    
    # Validate some example scan parameters
    example_params = {
        "distance_mm": 400,
        "disparity_px": 100,
        "epipolar_tolerance_px": 3
    }
    
    valid = validate_scan_parameters(sl_module, example_params)
    
    # Calculate optimal parameters for different distances
    distances = [250, 400, 700, 900]
    for dist in distances:
        calculate_optimal_parameters(sl_module, dist)


if __name__ == "__main__":
    main()