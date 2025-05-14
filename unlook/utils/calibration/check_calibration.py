#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check Calibration File

This script loads and displays the parameters from a calibration file
to verify that it was saved correctly and has all the right parameters.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

def load_calibration(json_path):
    """Load calibration file and convert lists to numpy arrays"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays
    for key in data:
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
    
    return data

def check_calibration(json_path):
    """Check and display calibration parameters"""
    print(f"\nChecking calibration file: {json_path}")
    print("=" * 80)
    
    if not os.path.exists(json_path):
        print(f"Error: Calibration file not found at {json_path}")
        return False
    
    try:
        # Load calibration data
        calib = load_calibration(json_path)
        
        # Display key parameters
        print("\nKey Calibration Parameters:")
        
        # Check essential matrices
        for key in ['P1', 'P2', 'R', 'T', 'E', 'F']:
            if key in calib:
                if isinstance(calib[key], np.ndarray):
                    print(f"\n{key} matrix:")
                    print(calib[key])
                    
                    # Special check for P2[0,3] which encodes the baseline
                    if key == 'P2' and calib[key].shape[0] >= 1 and calib[key].shape[1] >= 4:
                        p2_03 = calib[key][0, 3]
                        print(f"\nP2[0,3] = {p2_03}")
                        
                        # Calculate baseline if P1 is available for the focal length
                        if 'P1' in calib and isinstance(calib['P1'], np.ndarray) and calib['P1'].shape[0] >= 1:
                            fx = calib['P1'][0, 0]
                            baseline_mm = abs(p2_03) * 1000.0 / fx
                            print(f"Implied baseline: {baseline_mm:.2f}mm")
                else:
                    print(f"\n{key}: {calib[key]}")
            else:
                print(f"\n{key}: Not found in calibration file")
        
        # Check camera matrices and distortion coefficients
        for key in ['K1', 'K2', 'D1', 'D2']:
            if key in calib:
                print(f"\n{key}:")
                print(calib[key])
        
        # Check essential parameters for 3D reconstruction
        if 'P1' in calib and 'P2' in calib:
            print("\nCalibration contains both projection matrices - Good for 3D reconstruction!")
        else:
            print("\nWARNING: Missing projection matrices!")
        
        # Check for translation vector (baseline)
        if 'T' in calib and isinstance(calib['T'], np.ndarray):
            print(f"\nBaseline from T vector: {abs(calib['T'][0, 0]):.2f}mm")
            
            # Check if baseline is reasonable
            if abs(calib['T'][0, 0]) < 40 or abs(calib['T'][0, 0]) > 120:
                print("WARNING: Baseline seems unusual! Should be around 80mm.")
            else:
                print("Baseline looks reasonable.")
        
        print("\nCalibration successfully loaded and verified!")
        return True
    
    except Exception as e:
        print(f"Error checking calibration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Check calibration file")
    parser.add_argument("--calibration", type=str, default="stereo_calibration.json",
                        help="Path to calibration file")
    args = parser.parse_args()
    
    # Get absolute path to calibration file
    if not os.path.isabs(args.calibration):
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.calibration)
    else:
        calib_path = args.calibration
    
    # Check calibration
    success = check_calibration(calib_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())