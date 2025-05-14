"""
Camera calibration utilities for the UnLook SDK.

This module provides helper functions for finding, loading, and managing
camera calibration files across the system. It supports automatic calibration
search and standardized storage paths.
"""

import os
import glob
import logging
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

# Configure logger
logger = logging.getLogger(__name__)

# Define standard calibration directories
SDK_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_ROOT = SDK_ROOT / "calibration"
DEFAULT_CALIBRATION_DIR = CALIBRATION_ROOT / "default"
CUSTOM_CALIBRATION_DIR = CALIBRATION_ROOT / "custom"
CALIBRATION_DIR = str(CALIBRATION_ROOT)  # For backward compatibility

def load_calibration(calibration_file: Optional[str] = None):
    """
    Load a calibration file and return a StereoCalibrator instance with validation.
    
    This enhanced function loads calibration data and automatically validates
    the projection matrices, correcting any scaling issues to ensure consistent
    real-world measurements.
    
    Args:
        calibration_file: Path to calibration file
        
    Returns:
        StereoCalibrator instance or None if failed
    """
    try:
        # Find the calibration file
        if calibration_file is None:
            calib_path = find_most_recent_calibration()
        else:
            calib_path = find_calibration_file(calibration_file)
            
        if not calib_path or not os.path.exists(calib_path):
            logger.warning(f"Could not find calibration file: {calibration_file}")
            return None
            
        # Import here to avoid circular imports
        from .camera_calibration import StereoCalibrator
        
        # Create a calibrator first with default parameters
        calibrator = StereoCalibrator()
        
        # Then load the calibration data manually
        logger.info(f"Loading calibration data from: {calib_path}")
        load_success = False
        
        if calib_path.endswith('.json'):
            if calibrator.load_calibration(calib_path):
                logger.info("Successfully loaded calibration data from JSON file")
                load_success = True
        elif calib_path.endswith('.npy') or calib_path.endswith('.npz'):
            if calibrator.load_calibration_npy(calib_path):
                logger.info("Successfully loaded calibration data from NPY/NPZ file")
                load_success = True
        
        if not load_success:
            logger.warning(f"Failed to load calibration data from {calib_path}")
            return None
        
        # Important: Validate and correct projection matrices
        if hasattr(calibrator, 'validate_projection_matrices'):
            try:
                logger.info("Validating projection matrices...")
                calibrator.validate_projection_matrices()
                
                # Get baseline value and log it for reference
                if hasattr(calibrator, 'P2') and calibrator.P2 is not None:
                    try:
                        tx = calibrator.P2[0, 3]
                        fx = calibrator.P2[0, 0]
                        baseline_mm = -tx / fx * 1000.0  # Convert to mm
                        
                        # Store for later access
                        calibrator.baseline_mm = baseline_mm
                        
                        logger.info(f"Final baseline value after validation: {baseline_mm:.2f}mm")
                    except Exception as be:
                        logger.warning(f"Could not extract final baseline: {be}")
            except Exception as ve:
                logger.warning(f"Error validating projection matrices: {ve}")
        else:
            logger.warning("Calibrator doesn't support matrix validation")
        
        return calibrator
    except Exception as e:
        logger.error(f"Error loading calibration: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def find_calibration_file(calibration_file: Optional[str] = None,
                          camera_name: Optional[str] = None) -> Optional[str]:
    """
    Find a valid calibration file, searching multiple locations.
    
    Search order:
    1. Explicitly provided path
    2. Custom directory with camera_name
    3. Default directory with camera_name
    4. Most recent calibration in custom dir
    5. Most recent calibration in default dir
    
    Args:
        calibration_file: Explicit calibration file path
        camera_name: Name of camera model to search for
        
    Returns:
        Path to a valid calibration file, or None if none found
    """
    # 1. If explicit calibration file is provided, verify it exists
    if calibration_file:
        if os.path.exists(calibration_file):
            logger.info(f"Using explicit calibration file: {calibration_file}")
            return calibration_file
        else:
            logger.warning(f"Specified calibration file not found: {calibration_file}")
    
    # 2. Try camera-specific calibration in custom directory
    if camera_name:
        custom_patterns = [
            str(CUSTOM_CALIBRATION_DIR / f"{camera_name.lower()}.json"),
            str(CUSTOM_CALIBRATION_DIR / f"{camera_name.lower()}_calibration.json"),
            str(CUSTOM_CALIBRATION_DIR / f"{camera_name.lower()}.npy")
        ]
        
        for pattern in custom_patterns:
            matching_files = glob.glob(pattern)
            if matching_files:
                logger.info(f"Using custom camera-specific calibration: {matching_files[0]}")
                return matching_files[0]
    
    # 3. Try camera-specific calibration in default directory
    if camera_name:
        default_patterns = [
            str(DEFAULT_CALIBRATION_DIR / f"{camera_name.lower()}.json"),
            str(DEFAULT_CALIBRATION_DIR / f"{camera_name.lower()}_calibration.json"),
            str(DEFAULT_CALIBRATION_DIR / f"{camera_name.lower()}.npy")
        ]
        
        for pattern in default_patterns:
            matching_files = glob.glob(pattern)
            if matching_files:
                logger.info(f"Using default camera-specific calibration: {matching_files[0]}")
                return matching_files[0]
    
    # 4. Try any calibration file in custom directory, sorted by modification time
    custom_files = list(CUSTOM_CALIBRATION_DIR.glob("*.json")) + list(CUSTOM_CALIBRATION_DIR.glob("*.npy"))
    if custom_files:
        # Sort by modification time, newest first
        custom_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        logger.info(f"Using most recent custom calibration: {custom_files[0]}")
        return str(custom_files[0])
    
    # 5. Try any calibration file in default directory, sorted by modification time
    default_files = list(DEFAULT_CALIBRATION_DIR.glob("*.json")) + list(DEFAULT_CALIBRATION_DIR.glob("*.npy"))
    if default_files:
        # Sort by modification time, newest first
        default_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        logger.info(f"Using most recent default calibration: {default_files[0]}")
        return str(default_files[0])
    
    # No calibration file found
    logger.warning("No calibration file found in any location")
    return None

def save_calibration_to_standard_location(
    calibration_data: Dict[str, Any],
    camera_name: Optional[str] = None,
    is_default: bool = False,
    custom_id: Optional[str] = None
) -> str:
    """
    Save calibration data to a standardized location.
    
    Args:
        calibration_data: Calibration parameter dictionary
        camera_name: Name of camera (used in filename)
        is_default: Whether to save as a default calibration
        custom_id: Optional custom identifier for the calibration
        
    Returns:
        Path to the saved calibration file
    """
    # Ensure directories exist
    os.makedirs(DEFAULT_CALIBRATION_DIR, exist_ok=True)
    os.makedirs(CUSTOM_CALIBRATION_DIR, exist_ok=True)
    
    # Determine file path
    target_dir = DEFAULT_CALIBRATION_DIR if is_default else CUSTOM_CALIBRATION_DIR
    
    # Create filename
    timestamp = int(time.time())
    if camera_name:
        if custom_id:
            filename = f"{camera_name.lower()}_{custom_id}.json"
        else:
            filename = f"{camera_name.lower()}.json"
    else:
        if custom_id:
            filename = f"calibration_{custom_id}.json"
        else:
            filename = f"calibration_{timestamp}.json"
    
    file_path = os.path.join(target_dir, filename)
    
    # Save as JSON
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in calibration_data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
        logger.info(f"Calibration saved to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save calibration: {e}")
        return ""

def find_most_recent_calibration():
    """
    Find the most recent calibration file.
    
    Returns:
        Most recent calibration file path or None if not found
    """
    # First, check for customized calibration
    custom_path = os.path.join(CALIBRATION_DIR, "custom")
    if os.path.isdir(custom_path):
        calibration_files = glob.glob(os.path.join(custom_path, "*.json"))
        if calibration_files:
            # Sort by modification time, newest first
            newest_file = max(calibration_files, key=os.path.getmtime)
            logger.info(f"Using most recent custom calibration: {newest_file}")
            return newest_file
    
    # Check for stereo_calibration.json in the current directory
    current_dir_file = os.path.join(os.getcwd(), "stereo_calibration.json")
    if os.path.isfile(current_dir_file):
        logger.info(f"Using calibration file from current directory: {current_dir_file}")
        return current_dir_file
        
    # Also check in examples directory
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
    examples_file = os.path.join(examples_dir, "stereo_calibration.json")
    if os.path.isfile(examples_file):
        logger.info(f"Using calibration file from examples directory: {examples_file}")
        return examples_file
            
    # Next, check for per-device calibration based on serial number
    # This would require knowing the device serial number
    
    # Finally, fall back to default calibration
    default_file = os.path.join(CALIBRATION_DIR, "default", "default_stereo.json")
    if os.path.isfile(default_file):
        logger.info(f"Using default calibration: {default_file}")
        return default_file
        
    return None

def extract_baseline_from_calibration(calibration_file: str) -> float:
    """
    Extract the baseline value from a calibration file with robust scaling correction.
    
    This enhanced function implements multiple methods to extract the baseline value
    and applies scale correction for common calibration issues, ensuring consistent
    real-world scaling for 3D reconstruction.
    
    Args:
        calibration_file: Path to calibration file (JSON or NPY)
        
    Returns:
        Baseline value in millimeters, or 75.0 if not found
    """
    # Define reasonable baseline ranges (most stereo cameras use 40-120mm baselines)
    MIN_REASONABLE_BASELINE = 20.0  # mm - smallest practical baseline
    MAX_REASONABLE_BASELINE = 200.0  # mm - largest practical baseline
    DEFAULT_BASELINE = 75.0         # mm - typical baseline for consumer stereo cameras
    
    try:
        # First find the calibration file through the standard search
        if calibration_file is None:
            calib_path = find_most_recent_calibration()
        else:
            calib_path = find_calibration_file(calibration_file)
        
        if not calib_path:
            logger.error(f"No calibration file found matching: {calibration_file}")
            return DEFAULT_BASELINE
            
        logger.info(f"Extracting baseline from calibration file: {calib_path}")
        
        # Store all potential baseline values for later validation
        potential_values = []
        
        # Determine file type and load data
        data = None
        if calib_path.endswith('.json'):
            # Load JSON calibration
            with open(calib_path, 'r') as f:
                data = json.load(f)
        elif calib_path.endswith('.npy') or calib_path.endswith('.npz'):
            # Load NumPy calibration
            try:
                data_npz = np.load(calib_path, allow_pickle=True)
                
                # Handle different NPZ file formats
                if isinstance(data_npz, np.lib.npyio.NpzFile):
                    # Regular NPZ file with multiple arrays
                    data = {}
                    for key in data_npz.files:
                        data[key] = data_npz[key]
                    
                    # Check for metadata in special format
                    if 'metadata' in data and hasattr(data['metadata'], 'item'):
                        metadata = data['metadata'].item()
                        if isinstance(metadata, dict):
                            # Add metadata items to data dict
                            for k, v in metadata.items():
                                if k not in data:  # Don't overwrite existing keys
                                    data[k] = v
                else:
                    # Single .npy file with a dict
                    data = data_npz.item() if hasattr(data_npz, 'item') else dict(data_npz)
            except Exception as e:
                logger.warning(f"Error loading NPY/NPZ file, trying alternative methods: {e}")
                try:
                    # Try loading as a simple .npy file with a dictionary
                    data = dict(np.load(calib_path, allow_pickle=True).item())
                except:
                    logger.error(f"Failed to load NumPy calibration file: {calib_path}")
                    return DEFAULT_BASELINE
        
        if data is None:
            logger.error(f"Failed to load calibration data from: {calib_path}")
            return DEFAULT_BASELINE
        
        # APPROACH 1: Check for direct baseline value in metadata (most reliable)
        if 'baseline_mm' in data:
            baseline = float(data['baseline_mm'])
            
            # Validate the baseline is in a reasonable range
            if MIN_REASONABLE_BASELINE <= baseline <= MAX_REASONABLE_BASELINE:
                logger.info(f"Using baseline from calibration metadata: {baseline:.2f}mm")
                return baseline
            elif baseline > 1000.0:
                # This looks like a scaling issue (meters stored as millimeters)
                corrected = baseline / 1000.0
                if MIN_REASONABLE_BASELINE <= corrected <= MAX_REASONABLE_BASELINE:
                    logger.warning(f"Corrected baseline from {baseline:.2f}mm to {corrected:.2f}mm (scaling issue)")
                    return corrected
            
            # Store for later comparison even if out of range
            potential_values.append(("metadata", baseline))
        
        # APPROACH 2: Try to calculate from projection matrix
        p2_baseline = None
        if 'P2' in data:
            P2 = np.array(data['P2']) if not isinstance(data['P2'], np.ndarray) else data['P2']
            
            # Validate P2 shape
            if P2.shape == (3, 4):
                try:
                    fx = P2[0, 0]
                    tx = P2[0, 3]
                    
                    if fx != 0:  # Avoid division by zero
                        # Calculate baseline
                        p2_baseline = -tx / fx * 1000.0  # Convert to mm
                        
                        # Check for scaling issues
                        if p2_baseline > 1000.0:
                            # This is likely a scaling issue
                            corrected = p2_baseline / 1000.0
                            if MIN_REASONABLE_BASELINE <= corrected <= MAX_REASONABLE_BASELINE:
                                logger.warning(f"Corrected P2 baseline from {p2_baseline:.2f}mm to {corrected:.2f}mm (scaling issue)")
                                p2_baseline = corrected
                        
                        # Store for later comparison
                        potential_values.append(("P2", p2_baseline))
                        
                        # Use immediately if in reasonable range
                        if MIN_REASONABLE_BASELINE <= p2_baseline <= MAX_REASONABLE_BASELINE:
                            logger.info(f"Using baseline calculated from P2 matrix: {p2_baseline:.2f}mm")
                            return p2_baseline
                except Exception as e:
                    logger.warning(f"Error calculating baseline from P2 matrix: {e}")
        
        # APPROACH 3: Try to calculate from translation vector
        t_baseline = None
        if 'T' in data:
            try:
                T = np.array(data['T']) if not isinstance(data['T'], np.ndarray) else data['T']
                
                # Calculate baseline from translation vector norm
                t_baseline = float(np.linalg.norm(T) * 1000.0)  # Convert to mm
                
                # Check for scaling issues
                if t_baseline > 1000.0:
                    # This is likely a scaling issue
                    corrected = t_baseline / 1000.0
                    if MIN_REASONABLE_BASELINE <= corrected <= MAX_REASONABLE_BASELINE:
                        logger.warning(f"Corrected T baseline from {t_baseline:.2f}mm to {corrected:.2f}mm (scaling issue)")
                        t_baseline = corrected
                
                # Store for later comparison
                potential_values.append(("T", t_baseline))
                
                # Use immediately if in reasonable range
                if MIN_REASONABLE_BASELINE <= t_baseline <= MAX_REASONABLE_BASELINE:
                    logger.info(f"Using baseline calculated from T vector: {t_baseline:.2f}mm")
                    return t_baseline
            except Exception as e:
                logger.warning(f"Error calculating baseline from T vector: {e}")
        
        # APPROACH 4: Try other sources in the calibration data
        for field in ['stereo_baseline', 'extrinsics_baseline', 'camera_baseline', 'calibration_baseline']:
            if field in data:
                try:
                    value = float(data[field])
                    if MIN_REASONABLE_BASELINE <= value <= MAX_REASONABLE_BASELINE:
                        logger.info(f"Using baseline from {field}: {value:.2f}mm")
                        return value
                    potential_values.append((field, value))
                except (ValueError, TypeError):
                    pass
        
        # APPROACH 5: Make intelligent choice from gathered baselines
        if potential_values:
            logger.info(f"Comparing potential baseline values: {potential_values}")
            
            # First, check if we have any values in reasonable range
            reasonable_values = [(source, value) for source, value in potential_values 
                                if MIN_REASONABLE_BASELINE <= value <= MAX_REASONABLE_BASELINE]
            
            if reasonable_values:
                # If we have reasonable values, take their average
                avg_baseline = sum(value for _, value in reasonable_values) / len(reasonable_values)
                logger.info(f"Using average of reasonable baseline values: {avg_baseline:.2f}mm")
                return avg_baseline
            
            # If no reasonable values, check for correctable values (scale factor of 1000)
            correctable_values = [(source, value/1000.0) for source, value in potential_values 
                                 if value > 1000.0 and MIN_REASONABLE_BASELINE <= value/1000.0 <= MAX_REASONABLE_BASELINE]
            
            if correctable_values:
                # Take average of corrected values
                avg_corrected = sum(value for _, value in correctable_values) / len(correctable_values)
                logger.warning(f"Using average of scale-corrected baseline values: {avg_corrected:.2f}mm")
                return avg_corrected
            
            # If we still have values but none are reasonable or correctable
            # Take the closest one to the default value
            if potential_values:
                closest = min(potential_values, key=lambda x: abs(x[1] - DEFAULT_BASELINE) 
                             if not np.isnan(x[1]) and not np.isinf(x[1]) else float('inf'))
                closest_value = closest[1]
                
                # Apply scaling correction if needed
                if closest_value > 1000.0:
                    closest_value /= 1000.0
                
                logger.warning(f"Using closest baseline to default value: {closest_value:.2f}mm (from {closest[0]})")
                return closest_value
        
        # APPROACH 6: Last resort - use default value
        logger.warning(f"Could not extract reliable baseline from calibration, using default value of {DEFAULT_BASELINE}mm")
        return DEFAULT_BASELINE
            
    except Exception as e:
        logger.error(f"Error extracting baseline from calibration: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        logger.info(f"Using default baseline of {DEFAULT_BASELINE}mm")
        return DEFAULT_BASELINE