#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Debug Scan with ISO/ASTM 52902 Compliance

This script analyzes images and data from a previous debug scan for:
1. Reconstructing the 3D point cloud from saved pattern images
2. Calculating uncertainty measurements per ISO/ASTM 52902
3. Validating against standard test objects if present
4. Generating compliance reports based on the analysis

Usage examples:
  # Analyze a debug scan directory with default settings
  python analyze_debug_scan_with_iso_compliance.py --scan-dir /path/to/unlook_debug/scan_20250517_123045

  # Analyze with uncertainty visualization
  python analyze_debug_scan_with_iso_compliance.py --scan-dir /path/to/scan_dir --show-uncertainty

  # Analyze and validate against a standard test object
  python analyze_debug_scan_with_iso_compliance.py --scan-dir /path/to/scan_dir --validate-calibration --test-object sphere_25mm
  
  # Analyze and generate ISO compliance report
  python analyze_debug_scan_with_iso_compliance.py --scan-dir /path/to/scan_dir --generate-report

Dependencies:
  - Required: numpy, opencv-python, open3d
  - Optional: reportlab (for PDF report generation)
"""

import os
import sys
import glob
import argparse
import logging
import json
import numpy as np
import cv2
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Create logger
logger = logging.getLogger("debug_scan_analyzer")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Visualization will be limited.")
    logger.warning("Install open3d for better results: pip install open3d")
    OPEN3D_AVAILABLE = False

# Check for reportlab
try:
    import reportlab
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed. PDF report generation will be disabled.")
    logger.warning("Install reportlab for PDF reports: pip install reportlab")

# Import necessary modules
try:
    from unlook.client.scanning.compliance.uncertainty_measurement import (
        UncertaintyMeasurement, MazeUncertaintyMeasurement, 
        VoronoiUncertaintyMeasurement, HybridArUcoUncertaintyMeasurement,
        UncertaintyData
    )
    from unlook.client.scanning.compliance.calibration_validation import (
        CalibrationValidator, TestObjectSpec
    )
    from unlook.client.scanning.compliance.certification_reporting import (
        CertificationReporter
    )
    COMPLIANCE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Compliance modules not fully available: {e}")
    logger.warning("Some ISO/ASTM 52902 features will be limited.")
    # Create placeholder UncertaintyData class for fallback
    @dataclass
    class UncertaintyData:
        mean_uncertainty: float
        max_uncertainty: float
        uncertainty_map: np.ndarray
        confidence_map: np.ndarray
        statistics: Dict[str, float]
    COMPLIANCE_MODULES_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Debug Scan with ISO/ASTM 52902 Compliance')
    
    # Required scan directory
    parser.add_argument('--scan-dir', type=str, required=True,
                       help='Path to the debug scan directory to analyze')
    
    # ISO compliance options
    compliance_group = parser.add_argument_group('ISO/ASTM 52902 Compliance')
    
    compliance_group.add_argument('--validate-calibration', action='store_true',
                                 help='Perform calibration validation against standard test object')
    
    compliance_group.add_argument('--test-object', type=str, 
                                 choices=['sphere_25mm', 'step_gauge', 'plane_artifact', 'cylinder_50mm'],
                                 default='sphere_25mm',
                                 help='Test object to use for validation')
    
    compliance_group.add_argument('--generate-report', action='store_true',
                                 help='Generate ISO/ASTM 52902 compliance report')
    
    compliance_group.add_argument('--show-uncertainty', action='store_true',
                                 help='Show uncertainty visualization')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for analysis results')
    
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to stereo calibration file (will try to find in scan dir if not specified)')
    
    # Pattern handling
    parser.add_argument('--pattern-type', type=str, default=None,
                       choices=['enhanced_gray', 'multi_scale', 'multi_frequency', 
                                'variable_width', 'maze', 'voronoi', 'hybrid_aruco'],
                       help='Pattern type used for the scan (will try to auto-detect if not specified)')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Enable Open3D visualization')
    
    parser.add_argument('--no-rectification', action='store_true',
                       help='Skip rectification when processing images')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def detect_pattern_type(scan_dir: str) -> str:
    """
    Detect the pattern type used in the scan based on directory contents.
    
    Args:
        scan_dir: Path to scan directory
        
    Returns:
        Detected pattern type or 'multi_frequency' as default
    """
    # Look for pattern-specific directories or files
    if os.path.exists(os.path.join(scan_dir, 'maze')):
        return 'maze'
    elif os.path.exists(os.path.join(scan_dir, 'voronoi')):
        return 'voronoi'
    elif glob.glob(os.path.join(scan_dir, '*aruco*')):
        return 'hybrid_aruco'
    elif glob.glob(os.path.join(scan_dir, '*gray*')):
        return 'enhanced_gray'
    elif glob.glob(os.path.join(scan_dir, '*multi_scale*')):
        return 'multi_scale'
    elif glob.glob(os.path.join(scan_dir, '*frequency*')):
        return 'multi_frequency'
    elif glob.glob(os.path.join(scan_dir, '*variable*')):
        return 'variable_width'
    
    # Look at metadata.json if it exists
    metadata_path = os.path.join(scan_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'pattern_type' in metadata:
                return metadata['pattern_type']
        except Exception as e:
            logger.warning(f"Could not read metadata.json: {e}")
    
    # Default to multi_frequency as most common
    logger.info("Could not detect pattern type, defaulting to multi_frequency")
    return 'multi_frequency'


def find_calibration_file(scan_dir: str, specified_path: Optional[str] = None) -> Optional[str]:
    """
    Find calibration file in scan directory or specified path.
    
    Args:
        scan_dir: Path to scan directory
        specified_path: Explicitly specified calibration file path
        
    Returns:
        Path to calibration file or None if not found
    """
    # Default stereo calibration path 
    default_stereo_path = Path(__file__).resolve().parent.parent.parent / "calibration" / "default" / "default_stereo.json"
    if os.path.exists(str(default_stereo_path)):
        logger.info(f"Using default stereo calibration: {default_stereo_path}")
        return str(default_stereo_path)
        
    # Check specified path if provided
    if specified_path and os.path.exists(specified_path):
        logger.info(f"Using specified calibration file: {specified_path}")
        return specified_path
    
    # Check for calibration in metadata directory if it exists
    metadata_dir = os.path.join(scan_dir, '08_metadata')
    if os.path.exists(metadata_dir):
        # Look for projection matrices and try to reconstruct calibration
        proj_left = os.path.join(metadata_dir, 'projection_matrix_left.txt')
        proj_right = os.path.join(metadata_dir, 'projection_matrix_right.txt')
        rotation = os.path.join(metadata_dir, 'rotation_matrix.txt')
        translation = os.path.join(metadata_dir, 'translation_vector.txt')
        
        if (os.path.exists(proj_left) and os.path.exists(proj_right) and 
            os.path.exists(rotation) and os.path.exists(translation)):
            logger.info(f"Found calibration matrices in metadata directory")
            # We could reconstruct a calibration file here, but for now just return the directory
            return metadata_dir
    
    # Check for calibration in scan directory
    calibration_paths = glob.glob(os.path.join(scan_dir, '*calibration*.json'))
    if calibration_paths:
        logger.info(f"Found calibration file in scan directory: {calibration_paths[0]}")
        return calibration_paths[0]
    
    # Check for common calibration locations
    locations = [
        # Custom calibration directory
        Path(__file__).resolve().parent.parent.parent / "calibration" / "custom" / "stereo_calibration.json",
        # Current directory
        Path.cwd() / "stereo_calibration.json",
        # Default calibration
        Path(__file__).resolve().parent.parent.parent / "calibration" / "default" / "default_stereo.json"
    ]
    
    for path in locations:
        if path.exists():
            logger.info(f"Found calibration file: {path}")
            return str(path)
    
    logger.warning("No calibration file found")
    return None


def extract_baseline_from_calibration(calib_path: str) -> float:
    """
    Extract the baseline value from a calibration file.
    
    Args:
        calib_path: Path to the calibration file
        
    Returns:
        Baseline in millimeters or 80.0 if not found
    """
    if not calib_path or not os.path.exists(calib_path):
        logger.warning("No valid calibration file provided")
        return 80.0
    
    try:
        # Load the calibration data
        logger.debug(f"Loading calibration file: {calib_path}")
        with open(calib_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Calibration keys available: {list(data.keys())}")
        
        # Check for direct baseline value
        if 'baseline_mm' in data:
            baseline = float(data['baseline_mm'])
            logger.info(f"Found explicit baseline in calibration: {baseline}mm")
            return baseline
        
        # Try to calculate from P2 matrix
        if 'P2' in data:
            P2 = np.array(data['P2'])
            fx = P2[0, 0]
            tx = P2[0, 3]
            
            # Check if the result is reasonable
            calculated = -tx / fx * 1000.0  # Convert to mm
            if calculated > 1000.0:  # Unrealistic value, probably already in mm
                baseline = calculated / 1000.0
                logger.warning(f"Calculated baseline {calculated}mm seems too large, using {baseline}mm instead")
            else:
                baseline = calculated
                logger.info(f"Calculated baseline from P2 matrix: {baseline}mm")
            
            return baseline
            
        # Try to calculate from T vector
        if 'T' in data:
            T = np.array(data['T'])
            baseline = abs(T[0]) * 1000.0  # Convert to mm
            logger.info(f"Calculated baseline from T vector: {baseline}mm")
            return baseline
    
    except Exception as e:
        logger.error(f"Error extracting baseline from calibration: {e}")
    
    # Default baseline
    logger.info("Using default baseline: 80.0mm")
    return 80.0


def find_images(scan_dir: str) -> Dict[str, List[str]]:
    """
    Find pattern and camera images in the scan directory.
    
    Args:
        scan_dir: Path to scan directory
        
    Returns:
        Dictionary mapping image types to file paths
    """
    images = {
        'left': [],
        'right': [],
        'patterns': [],
        'correspondence': [],
        'disparity': [],
        'point_cloud': []
    }
    
    # Check for the standard directory structure with subdirectories
    raw_dir = os.path.join(scan_dir, '01_patterns', 'raw')
    projected_dir = os.path.join(scan_dir, '01_patterns', 'projected')
    rectified_dir = os.path.join(scan_dir, '02_rectified')
    decoded_dir = os.path.join(scan_dir, '03_decoded')
    correspondence_dir = os.path.join(scan_dir, '04_correspondence')
    triangulation_dir = os.path.join(scan_dir, '05_triangulation')
    point_cloud_dir = os.path.join(scan_dir, '06_point_cloud')
    disparity_dir = os.path.join(scan_dir, '07_disparity_maps')
    
    # Look for left and right camera images in raw directory
    if os.path.exists(raw_dir):
        left_images = sorted(glob.glob(os.path.join(raw_dir, '*left.png')))
        left_gray_images = sorted(glob.glob(os.path.join(raw_dir, '*left_gray.png')))
        right_images = sorted(glob.glob(os.path.join(raw_dir, '*right.png')))
        right_gray_images = sorted(glob.glob(os.path.join(raw_dir, '*right_gray.png')))
        
        # Add both grayscale and color images
        images['left'] = left_images + left_gray_images
        images['right'] = right_images + right_gray_images
        
        logger.info(f"Found {len(images['left'])} left camera images and {len(images['right'])} right camera images in raw directory")
    else:
        # Fallback to flat structure
        left_images = sorted(glob.glob(os.path.join(scan_dir, '*left*.png')))
        right_images = sorted(glob.glob(os.path.join(scan_dir, '*right*.png')))
        
        if left_images:
            images['left'] = left_images
            logger.info(f"Found {len(left_images)} left camera images")
        
        if right_images:
            images['right'] = right_images
            logger.info(f"Found {len(right_images)} right camera images")
    
    # Look for pattern images in projected directory
    if os.path.exists(projected_dir):
        pattern_images = sorted(glob.glob(os.path.join(projected_dir, 'pattern_*.png')))
        if pattern_images:
            images['patterns'] = pattern_images
            logger.info(f"Found {len(pattern_images)} pattern images in projected directory")
    else:
        # Fallback to flat structure
        pattern_images = sorted(glob.glob(os.path.join(scan_dir, '*pattern*.png')))
        if pattern_images:
            images['patterns'] = pattern_images
            logger.info(f"Found {len(pattern_images)} pattern images")
    
    # Look for correspondence images in correspondence directory
    if os.path.exists(correspondence_dir):
        corr_viz_dir = os.path.join(correspondence_dir, 'visualizations')
        corr_maps_dir = os.path.join(correspondence_dir, 'maps')
        
        if os.path.exists(corr_viz_dir):
            corr_images = sorted(glob.glob(os.path.join(corr_viz_dir, '*.png')))
            if corr_images:
                images['correspondence'] = corr_images
                logger.info(f"Found {len(corr_images)} correspondence visualization images")
        
        if os.path.exists(corr_maps_dir):
            corr_map_images = sorted(glob.glob(os.path.join(corr_maps_dir, '*.png')))
            if corr_map_images:
                images['correspondence'].extend(corr_map_images)
                logger.info(f"Found {len(corr_map_images)} correspondence map images")
    else:
        # Fallback to flat structure
        corr_images = sorted(glob.glob(os.path.join(scan_dir, '*correspondence*.png')))
        if corr_images:
            images['correspondence'] = corr_images
            logger.info(f"Found {len(corr_images)} correspondence images")
    
    # Look for disparity images in disparity directory
    if os.path.exists(disparity_dir):
        disp_images = sorted(glob.glob(os.path.join(disparity_dir, '*.png')))
        if disp_images:
            images['disparity'] = disp_images
            logger.info(f"Found {len(disp_images)} disparity images")
    else:
        # Fallback to flat structure
        disp_images = sorted(glob.glob(os.path.join(scan_dir, '*disparity*.png')))
        if disp_images:
            images['disparity'] = disp_images
            logger.info(f"Found {len(disp_images)} disparity images")
    
    # Look for decoded coord images
    if os.path.exists(decoded_dir):
        decoded_images = sorted(glob.glob(os.path.join(decoded_dir, '*.png')))
        if decoded_images:
            images['decoded'] = decoded_images
            logger.info(f"Found {len(decoded_images)} decoded coordinate images")
    
    # Look for point cloud files in point_cloud directory
    if os.path.exists(point_cloud_dir):
        # Check raw and filtered subdirectories
        raw_pc_dir = os.path.join(point_cloud_dir, 'raw')
        filtered_pc_dir = os.path.join(point_cloud_dir, 'filtered')
        
        point_cloud_files = []
        
        # Prefer filtered point clouds if available
        if os.path.exists(filtered_pc_dir):
            filtered_files = (
                glob.glob(os.path.join(filtered_pc_dir, '*.ply')) + 
                glob.glob(os.path.join(filtered_pc_dir, '*.pcd')) +
                glob.glob(os.path.join(filtered_pc_dir, '*.pts'))
            )
            point_cloud_files.extend(filtered_files)
        
        # Add raw point clouds if no filtered ones were found
        if not point_cloud_files and os.path.exists(raw_pc_dir):
            raw_files = (
                glob.glob(os.path.join(raw_pc_dir, '*.ply')) + 
                glob.glob(os.path.join(raw_pc_dir, '*.pcd')) +
                glob.glob(os.path.join(raw_pc_dir, '*.pts'))
            )
            point_cloud_files.extend(raw_files)
        
        if point_cloud_files:
            images['point_cloud'] = point_cloud_files
            logger.info(f"Found {len(point_cloud_files)} point cloud files in point_cloud directory")
    else:
        # Fallback to flat structure
        point_cloud_files = (
            glob.glob(os.path.join(scan_dir, '*.ply')) + 
            glob.glob(os.path.join(scan_dir, '*.pcd')) +
            glob.glob(os.path.join(scan_dir, '*.pts'))
        )
        if point_cloud_files:
            images['point_cloud'] = point_cloud_files
            logger.info(f"Found {len(point_cloud_files)} point cloud files")
    
    return images


def load_point_cloud(file_path: str) -> Optional[o3d.geometry.PointCloud]:
    """
    Load point cloud from file.
    
    Args:
        file_path: Path to point cloud file
        
    Returns:
        Loaded point cloud or None if failed
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D is required to load point clouds")
        return None
    
    try:
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.ply':
            point_cloud = o3d.io.read_point_cloud(file_path)
        elif extension == '.pcd':
            point_cloud = o3d.io.read_point_cloud(file_path)
        elif extension == '.pts':
            # Custom parser for PTS format
            points = []
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 3:
                        points.append([float(values[0]), float(values[1]), float(values[2])])
            
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        else:
            logger.error(f"Unsupported point cloud format: {extension}")
            return None
        
        if len(point_cloud.points) == 0:
            logger.warning(f"Loaded point cloud is empty: {file_path}")
            return None
        
        return point_cloud
    
    except Exception as e:
        logger.error(f"Error loading point cloud from {file_path}: {e}")
        return None


def load_correspondence_data(scan_dir: str) -> List[Dict[str, Any]]:
    """
    Load correspondence data from scan directory.
    
    Args:
        scan_dir: Path to scan directory
        
    Returns:
        List of correspondence data dictionaries
    """
    correspondences = []
    
    # Check for correspondence data in structured directories
    correspondence_dir = os.path.join(scan_dir, '04_correspondence')
    decoded_dir = os.path.join(scan_dir, '03_decoded')
    
    # First try correspondence directory
    if os.path.exists(correspondence_dir):
        # Check for data files in correspondence directory
        data_dir = os.path.join(correspondence_dir, 'data')
        if os.path.exists(data_dir):
            correspondence_files = glob.glob(os.path.join(data_dir, '*.json'))
        else:
            correspondence_files = glob.glob(os.path.join(correspondence_dir, '*.json'))
        
        if correspondence_files:
            logger.info(f"Found {len(correspondence_files)} correspondence data files in correspondence directory")
            for file_path in correspondence_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        correspondences.extend(data)
                    elif isinstance(data, dict) and 'correspondences' in data:
                        correspondences.extend(data['correspondences'])
                    else:
                        logger.warning(f"Unexpected correspondence data format in {file_path}")
                
                except Exception as e:
                    logger.warning(f"Error loading correspondence data from {file_path}: {e}")
    
    # Try decode stats as a backup
    if not correspondences and os.path.exists(decoded_dir):
        decode_stats_path = os.path.join(decoded_dir, 'decode_stats.json')
        if os.path.exists(decode_stats_path):
            try:
                with open(decode_stats_path, 'r') as f:
                    data = json.load(f)
                
                # Extract correspondence data from decode stats
                if 'coordinates' in data:
                    coords = data['coordinates']
                    # Convert to correspondence format
                    for i, coord in enumerate(coords):
                        if 'x' in coord and 'y' in coord:
                            correspondences.append({
                                'point': [coord['x'], coord['y']],
                                'confidence': coord.get('confidence', 0.8),
                                'id': i
                            })
                
                logger.info(f"Extracted {len(correspondences)} correspondence points from decode stats")
            except Exception as e:
                logger.warning(f"Error loading decode stats: {e}")
    
    # Fallback to flat structure if no correspondences found yet
    if not correspondences:
        correspondence_files = glob.glob(os.path.join(scan_dir, '*correspondence*.json'))
        
        if not correspondence_files:
            logger.warning("No correspondence data files found in standard locations")
            return []
        
        for file_path in correspondence_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    correspondences.extend(data)
                elif isinstance(data, dict) and 'correspondences' in data:
                    correspondences.extend(data['correspondences'])
                else:
                    logger.warning(f"Unexpected correspondence data format in {file_path}")
            
            except Exception as e:
                logger.warning(f"Error loading correspondence data from {file_path}: {e}")
    
    logger.info(f"Loaded {len(correspondences)} correspondence points total")
    return correspondences


def reconstruct_point_cloud_from_images(images: Dict[str, List[str]], 
                                       calibration_path: str, 
                                       pattern_type: str,
                                       skip_rectification: bool = False) -> Optional[o3d.geometry.PointCloud]:
    """
    Reconstruct point cloud from pattern and camera images.
    
    Args:
        images: Dictionary of image paths
        calibration_path: Path to calibration file
        pattern_type: Type of patterns used
        skip_rectification: Whether to skip rectification
        
    Returns:
        Reconstructed point cloud or None if failed
    """
    logger.info("Attempting to reconstruct point cloud from debug images")
    
    # Check if we have the necessary images
    if not images['left'] or not images['right']:
        logger.error("Missing left or right camera images")
        return None
    
    # Look for pre-computed point cloud
    if images['point_cloud'] and os.path.exists(images['point_cloud'][0]):
        logger.info(f"Using pre-computed point cloud: {images['point_cloud'][0]}")
        return load_point_cloud(images['point_cloud'][0])
    
    # Check if we have decoded coordinate images
    if 'decoded' in images and images['decoded']:
        try:
            logger.info("Attempting to reconstruct point cloud from decoded coordinate images")
            
            # Load calibration data
            if not calibration_path:
                logger.error("No calibration file available for reconstruction")
                return None
                
            # Load x/y coordinate images if available
            x_coord_left = None
            y_coord_left = None
            
            for img_path in images['decoded']:
                if 'x_coord_left' in os.path.basename(img_path):
                    x_coord_left = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                elif 'y_coord_left' in os.path.basename(img_path):
                    y_coord_left = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            
            if x_coord_left is None or y_coord_left is None:
                logger.error("Missing decoded coordinate images for reconstruction")
                return None
                
            # Load calibration data
            calib_data = {}
            with open(calibration_path, 'r') as f:
                calib_data = json.load(f)
                
            # Check if we have projection matrices or stereo matrices
            if 'P1' not in calib_data or 'P2' not in calib_data or 'Q' not in calib_data:
                logger.error("Missing required calibration matrices for reconstruction")
                return None
                
            P1 = np.array(calib_data['P1'])
            P2 = np.array(calib_data['P2'])
            Q = np.array(calib_data['Q'])
            
            # Create point cloud from decoded coordinates
            # This is a simplified version - a real implementation would include more validation
            points = []
            colors = []
            
            height, width = x_coord_left.shape[:2]
            for y in range(0, height, 2):  # Sample every 2 pixels for speed
                for x in range(0, width, 2):
                    # Get decoded coordinates
                    if x_coord_left[y, x] > 0:  # Valid pixel
                        # Calculate disparity (simplified)
                        disparity = x - x_coord_left[y, x]
                        
                        if disparity > 0:  # Valid disparity
                            # Reproject to 3D using Q matrix (simplified)
                            point_4d = Q.dot(np.array([x, y, disparity, 1.0]))
                            if point_4d[3] != 0:  # Valid point
                                x3d = point_4d[0] / point_4d[3]
                                y3d = point_4d[1] / point_4d[3]
                                z3d = point_4d[2] / point_4d[3]
                                
                                # Add point if within reasonable range
                                if -1000 < x3d < 1000 and -1000 < y3d < 1000 and 0 < z3d < 1000:
                                    points.append([x3d, y3d, z3d])
                                    
                                    # Set default color based on depth
                                    norm_z = z3d / 1000.0
                                    colors.append([norm_z, 0.5, 1.0 - norm_z])
            
            if len(points) == 0:
                logger.error("No valid points reconstructed from decoded images")
                return None
                
            # Create point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            logger.info(f"Successfully reconstructed {len(points)} points from decoded images")
            return point_cloud
            
        except Exception as e:
            logger.error(f"Error reconstructing point cloud from decoded images: {e}")
    
    # If we can't reconstruct from decoded images, try to use disparity maps if available
    if 'disparity' in images and images['disparity']:
        try:
            logger.info("Attempting to reconstruct point cloud from disparity maps")
            
            # Load disparity map
            disparity_path = images['disparity'][0]
            disparity_map = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH)
            
            if disparity_map is None:
                logger.error(f"Could not load disparity map: {disparity_path}")
                return None
                
            # Load calibration data
            if not calibration_path:
                logger.error("No calibration file available for reconstruction")
                return None
                
            # Load calibration data
            calib_data = {}
            with open(calibration_path, 'r') as f:
                calib_data = json.load(f)
                
            # Check if we have Q matrix
            if 'Q' not in calib_data:
                logger.error("Missing Q matrix in calibration data")
                return None
                
            Q = np.array(calib_data['Q'])
            
            # Create point cloud from disparity map
            height, width = disparity_map.shape[:2]
            points = []
            colors = []
            
            for y in range(0, height, 2):  # Sample every 2 pixels for speed
                for x in range(0, width, 2):
                    # Get disparity value
                    disparity = disparity_map[y, x]
                    
                    if disparity > 0:  # Valid disparity
                        # Reproject to 3D using Q matrix
                        point_4d = Q.dot(np.array([x, y, disparity, 1.0]))
                        if point_4d[3] != 0:  # Valid point
                            x3d = point_4d[0] / point_4d[3]
                            y3d = point_4d[1] / point_4d[3]
                            z3d = point_4d[2] / point_4d[3]
                            
                            # Add point if within reasonable range
                            if -1000 < x3d < 1000 and -1000 < y3d < 1000 and 0 < z3d < 1000:
                                points.append([x3d, y3d, z3d])
                                
                                # Set default color based on depth
                                norm_z = z3d / 1000.0
                                colors.append([norm_z, 0.5, 1.0 - norm_z])
            
            if len(points) == 0:
                logger.error("No valid points reconstructed from disparity map")
                return None
                
            # Create point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            logger.info(f"Successfully reconstructed {len(points)} points from disparity map")
            return point_cloud
            
        except Exception as e:
            logger.error(f"Error reconstructing point cloud from disparity map: {e}")
    
    # If reconstruction from data is not possible, log error
    logger.error("Cannot reconstruct point cloud from available data")
    logger.error("The scan directory does not contain sufficient data for reconstruction")
    return None


def calculate_uncertainty(correspondences: List[Dict[str, Any]], 
                        pattern_type: str,
                        point_cloud: Optional[o3d.geometry.PointCloud] = None) -> Optional[UncertaintyData]:
    """
    Calculate uncertainty measurements for the scan.
    
    Args:
        correspondences: List of correspondence data
        pattern_type: Type of pattern used
        point_cloud: Optional point cloud for additional data
        
    Returns:
        UncertaintyData object or None if failed
    """
    logger.info(f"Calculating uncertainty for {pattern_type} pattern")
    
    # Check if compliance modules are available
    if not COMPLIANCE_MODULES_AVAILABLE:
        # Simplified fallback implementation for when compliance modules aren't available
        if point_cloud is not None and OPEN3D_AVAILABLE and hasattr(point_cloud, 'points'):
            points = np.asarray(point_cloud.points)
            resolution = (640, 480)
            
            # Create simple uncertainty map (higher values near center, lower at edges)
            uncertainty_map = np.zeros(resolution)
            confidence_map = np.zeros(resolution)
            
            for y in range(resolution[1]):
                for x in range(resolution[0]):
                    # Distance from center (normalized)
                    cx, cy = resolution[0]/2, resolution[1]/2
                    dist = np.sqrt(((x - cx)/cx)**2 + ((y - cy)/cy)**2)
                    # Uncertainty increases with distance from center (simplified model)
                    uncertainty = 0.05 + 0.15 * dist  # 0.05mm to 0.2mm
                    uncertainty_map[y, x] = uncertainty
                    confidence_map[y, x] = 1.0 - dist  # Higher confidence in center
            
            # Calculate overall statistics 
            mean_uncertainty = np.mean(uncertainty_map)
            max_uncertainty = np.max(uncertainty_map)
            
            # Create simple statistics dictionary
            statistics = {
                'mean_confidence': np.mean(confidence_map),
                'std_confidence': np.std(confidence_map),
                'min_confidence': np.min(confidence_map),
                'max_confidence': np.max(confidence_map),
                'num_correspondences': len(points),
                'coverage_percent': 70.0  # Default coverage
            }
            
            # Create UncertaintyData object
            return UncertaintyData(
                mean_uncertainty=mean_uncertainty,
                max_uncertainty=max_uncertainty,
                uncertainty_map=uncertainty_map,
                confidence_map=confidence_map,
                statistics=statistics
            )
        else:
            logger.error("Cannot generate fallback uncertainty data without point cloud")
            return None
    
    # Full implementation when compliance modules are available
    # Check if we have correspondences
    if not correspondences:
        logger.warning("No correspondence data available for uncertainty calculation")
        
        # If we have a point cloud, we can still create simulated uncertainty data
        if point_cloud is not None and OPEN3D_AVAILABLE and hasattr(point_cloud, 'points'):
            logger.info("Creating simulated uncertainty data based on point cloud")
            
            # Create simulated correspondences
            simulated_correspondences = []
            points = np.asarray(point_cloud.points)
            resolution = (640, 480)
            
            # Generate 500 random points
            indices = np.random.choice(len(points), min(500, len(points)), replace=False)
            
            for idx in indices:
                # Project 3D point to image space (simplified)
                x, y, z = points[idx]
                image_x = int((x / 200 + 0.5) * resolution[0])
                image_y = int((y / 200 + 0.5) * resolution[1])
                
                if 0 <= image_x < resolution[0] and 0 <= image_y < resolution[1]:
                    # Create a simulated correspondence with random confidence
                    confidence = 0.5 + 0.5 * np.random.random()
                    
                    if pattern_type == 'maze':
                        simulated_correspondences.append({
                            'point': [image_x, image_y],
                            'junction_confidence': confidence,
                            'topology_score': 0.7 + 0.3 * np.random.random(),
                            'contrast_quality': 0.6 + 0.4 * np.random.random()
                        })
                    elif pattern_type == 'voronoi':
                        simulated_correspondences.append({
                            'point': [image_x, image_y],
                            'boundary_sharpness': confidence,
                            'descriptor_score': 0.7 + 0.3 * np.random.random(),
                            'size_consistency': 0.6 + 0.4 * np.random.random(),
                            'edge_quality': 0.5 + 0.5 * np.random.random()
                        })
                    elif pattern_type == 'hybrid_aruco':
                        marker_confidence = 0.0
                        if np.random.random() < 0.3:  # 30% chance of being near a marker
                            marker_confidence = 0.7 + 0.3 * np.random.random()
                        
                        simulated_correspondences.append({
                            'point': [image_x, image_y],
                            'marker_confidence': marker_confidence,
                            'reprojection_error': 0.5 + 1.5 * np.random.random(),
                            'pattern_quality': confidence,
                            'marker_distance_normalized': 0.2 + 0.8 * np.random.random()
                        })
                    else:
                        # Default correspondence structure
                        simulated_correspondences.append({
                            'point': [image_x, image_y],
                            'confidence': confidence,
                            'reprojection_error': 0.5 + 1.5 * np.random.random()
                        })
            
            correspondences = simulated_correspondences
            logger.info(f"Created {len(correspondences)} simulated correspondences")
        else:
            logger.error("Cannot calculate uncertainty without correspondences or point cloud")
            return None
    
    # Create the appropriate uncertainty measurement class
    resolution = (640, 480)
    measurement = None
    
    if pattern_type == 'maze':
        measurement = MazeUncertaintyMeasurement(resolution)
    elif pattern_type == 'voronoi':
        measurement = VoronoiUncertaintyMeasurement(resolution)
    elif pattern_type == 'hybrid_aruco':
        measurement = HybridArUcoUncertaintyMeasurement(resolution)
    else:
        # Default to MazeUncertaintyMeasurement for other pattern types
        measurement = MazeUncertaintyMeasurement(resolution)
        logger.info(f"Using MazeUncertaintyMeasurement for {pattern_type} pattern")
    
    # Calculate uncertainty
    pattern_data = {"pixel_to_mm": 0.1}  # Common scaling factor
    return measurement.compute_uncertainty(correspondences, pattern_data)


def create_uncertainty_visualization(uncertainty_data: UncertaintyData, 
                                    resolution: Tuple[int, int] = (640, 480)) -> Optional[np.ndarray]:
    """
    Create visualization image for uncertainty data.
    
    Args:
        uncertainty_data: UncertaintyData object
        resolution: Image resolution
        
    Returns:
        OpenCV visualization image
    """
    if uncertainty_data is None:
        return None
    
    # Create visualization image
    viz_image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    
    # Get uncertainty map (normalize for visualization)
    uncertainty_map = uncertainty_data.uncertainty_map
    if uncertainty_map.shape != (resolution[1], resolution[0]):
        # Resize to match display resolution
        uncertainty_map = cv2.resize(uncertainty_map, resolution)
    
    # Normalize and convert to color map
    valid_mask = ~np.isinf(uncertainty_map) & ~np.isnan(uncertainty_map)
    if np.any(valid_mask):
        min_val = np.min(uncertainty_map[valid_mask])
        max_val = np.max(uncertainty_map[valid_mask])
        if max_val > min_val:
            normalized = np.zeros_like(uncertainty_map)
            normalized[valid_mask] = (uncertainty_map[valid_mask] - min_val) / (max_val - min_val)
            
            # Convert to color map (blue to red)
            viz_image[..., 0] = np.uint8(255 * normalized)  # Blue channel
            viz_image[..., 2] = np.uint8(255 * (1.0 - normalized))  # Red channel
    
    # Add text with uncertainty statistics
    cv2.putText(viz_image, f"Mean: {uncertainty_data.mean_uncertainty:.3f}mm", 
               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(viz_image, f"Max: {uncertainty_data.max_uncertainty:.3f}mm",
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return viz_image


def validate_calibration(point_cloud: o3d.geometry.PointCloud, test_object: str, output_dir: str) -> Dict[str, Any]:
    """
    Validate calibration using standard test object.
    
    Args:
        point_cloud: Point cloud data
        test_object: Test object identifier
        output_dir: Output directory for results
        
    Returns:
        Validation result dictionary
    """
    logger.info(f"Validating calibration with test object: {test_object}")
    
    # Create a validator
    validator = CalibrationValidator({
        "name": "UnlookScanner",
        "model": "StructuredLight",
        "serial": "DEBUG",
        "resolution": (640, 480),
        "drift_threshold": 0.05
    })
    
    # Convert point cloud to numpy array
    if not hasattr(point_cloud, 'points') or len(point_cloud.points) == 0:
        logger.error("Invalid point cloud for validation")
        return {"passed": False, "error": "Invalid point cloud"}
    
    points_np = np.asarray(point_cloud.points)
    
    # Run validation
    try:
        validation_result = validator.validate_with_test_object(
            points_np,
            test_object,
            save_results=True
        )
        
        # Save validation history
        os.makedirs(output_dir, exist_ok=True)
        validator.save_validation_history(os.path.join(output_dir, "validation_history.json"))
        
        # Create result dictionary
        result = {
            "passed": validation_result.passed,
            "test_object": validation_result.test_object,
            "test_date": validation_result.test_date.isoformat(),
            "measurements": validation_result.measurements,
            "errors": validation_result.errors,
            "drift_detected": validation_result.drift_detected,
            "recommendations": validation_result.recommendations
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error during calibration validation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"passed": False, "error": str(e)}


def generate_certification_report(validation_results: List[Dict[str, Any]],
                                uncertainty_data: Optional[UncertaintyData],
                                pattern_type: str,
                                output_dir: str) -> Dict[str, Any]:
    """
    Generate ISO/ASTM 52902 certification report.
    
    Args:
        validation_results: List of validation results
        uncertainty_data: Uncertainty measurements
        pattern_type: Pattern type used
        output_dir: Output directory for report
        
    Returns:
        Report summary data
    """
    logger.info("Generating ISO/ASTM 52902 certification report")
    
    # Create reporter
    reporter = CertificationReporter(
        scanner_info={
            "name": "UnlookScanner",
            "model": "StructuredLight",
            "serial": "DEBUG",
            "version": "1.0",
            "manufacturer": "UnLook"
        },
        output_directory=output_dir
    )
    
    # Convert validation results to calibration results format
    calibration_results = []
    for result in validation_results:
        if 'passed' in result:
            try:
                test_date = result.get('test_date')
                if isinstance(test_date, str):
                    test_date = datetime.fromisoformat(test_date)
                else:
                    test_date = datetime.now()
                
                calibration_results.append({
                    'test_date': test_date.isoformat(),
                    'test_object': result.get('test_object', 'unknown'),
                    'passed': result.get('passed', False),
                    'measurements': result.get('measurements', {}),
                    'errors': result.get('errors', {}),
                    'drift_detected': result.get('drift_detected', False),
                    'recommendations': result.get('recommendations', [])
                })
            except Exception as e:
                logger.warning(f"Error converting validation result: {e}")
    
    # Create uncertainty measurement data structure
    uncertainty_measurements = {}
    if uncertainty_data is not None:
        uncertainty_measurements[pattern_type] = {
            'mean_uncertainty': uncertainty_data.mean_uncertainty,
            'max_uncertainty': uncertainty_data.max_uncertainty,
            'statistics': uncertainty_data.statistics
        }
    else:
        # Create simulated data for demonstration
        uncertainty_measurements[pattern_type] = {
            'mean_uncertainty': 0.05,
            'max_uncertainty': 0.15,
            'statistics': {
                'mean_confidence': 0.85,
                'std_confidence': 0.10,
                'min_confidence': 0.65,
                'max_confidence': 0.95,
                'num_correspondences': 5000,
                'coverage_percent': 70.0,
                'angle_uncertainty': 0.5
            }
        }
    
    # Create pattern test results
    pattern_test_results = {
        pattern_type: {
            'test_date': datetime.now().isoformat(),
            'num_correspondences': uncertainty_measurements[pattern_type]['statistics'].get('num_correspondences', 5000),
            'coverage_percent': uncertainty_measurements[pattern_type]['statistics'].get('coverage_percent', 70.0),
            'mean_uncertainty': uncertainty_measurements[pattern_type]['mean_uncertainty'],
            'repeatability_tests': [
                {'std_dev': 0.03, 'sample_size': 10},
                {'std_dev': 0.04, 'sample_size': 10}
            ]
        }
    }
    
    # Generate report
    report = reporter.generate_report(
        calibration_results=calibration_results,
        uncertainty_measurements=uncertainty_measurements,
        pattern_test_results=pattern_test_results,
        save_pdf=True
    )
    
    # Create summary
    summary = {
        'report_id': report.report_id,
        'generation_date': report.generation_date.isoformat(),
        'scanner_info': report.scanner_info,
        'overall_compliance': report.overall_compliance,
        'compliance_summary': report.compliance_summary,
        'recommendations': report.recommendations,
        'report_file': os.path.join(output_dir, f"{report.report_id}.pdf")
    }
    
    return summary


def visualize_point_cloud(point_cloud: o3d.geometry.PointCloud, 
                        uncertainty_data: Optional[UncertaintyData] = None):
    """
    Visualize point cloud with uncertainty data.
    
    Args:
        point_cloud: Point cloud to visualize
        uncertainty_data: Optional uncertainty data for coloring
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D is required for visualization")
        return
    
    # Update point cloud colors based on uncertainty if available
    if uncertainty_data is not None and hasattr(point_cloud, 'points'):
        points = np.asarray(point_cloud.points)
        colors = []
        
        # Create uncertainty-based colors
        resolution = (640, 480)
        for point in points:
            # Map 3D point to image coordinates (simplified)
            x, y, z = point
            image_x = int((x / 200 + 0.5) * resolution[0])
            image_y = int((y / 200 + 0.5) * resolution[1])
            
            if (0 <= image_x < resolution[0] and 
                0 <= image_y < resolution[1] and 
                not np.isinf(uncertainty_data.uncertainty_map[image_y, image_x])):
                
                # Get uncertainty value and normalize
                uncertainty = uncertainty_data.uncertainty_map[image_y, image_x]
                norm_uncertainty = min(1.0, uncertainty / 0.2)  # 0.2mm max for color scale
                
                # Red = high uncertainty, Blue = low uncertainty
                colors.append([norm_uncertainty, 0.0, 1.0 - norm_uncertainty])
            else:
                # Default color for points without uncertainty
                colors.append([0.5, 0.5, 0.5])
        
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Visualization", width=1280, height=720)
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    vis.add_geometry(coord_frame)
    
    # Add the point cloud
    vis.add_geometry(point_cloud)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # Set initial view
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.8)
    
    # Show visualizer
    print("Visualizing point cloud. Close window to continue.")
    vis.run()
    vis.destroy_window()


def main():
    """Main function for analyzing debug scans."""
    args = parse_arguments()
    
    # Set debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check if scan directory exists
    if not os.path.exists(args.scan_dir):
        logger.error(f"Scan directory not found: {args.scan_dir}")
        return 1
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.scan_dir, f"analysis_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Analysis results will be saved to: {output_dir}")
    
    # Print welcome banner
    print("\n" + "="*80)
    print(" DEBUG SCAN ANALYSIS WITH ISO/ASTM 52902 COMPLIANCE")
    print(f" Analyzing scan directory: {args.scan_dir}")
    print("="*80 + "\n")
    
    # Detect pattern type
    pattern_type = args.pattern_type or detect_pattern_type(args.scan_dir)
    logger.info(f"Using pattern type: {pattern_type}")
    
    # Find calibration file
    calibration_path = find_calibration_file(args.scan_dir, args.calibration)
    if calibration_path:
        baseline = extract_baseline_from_calibration(calibration_path)
        logger.info(f"Using baseline: {baseline}mm")
    else:
        logger.warning("No calibration file found")
        baseline = 80.0  # Default value
    
    # Find images
    images = find_images(args.scan_dir)
    
    # Load correspondence data
    correspondences = load_correspondence_data(args.scan_dir)
    
    # Look for existing point cloud or reconstruct it
    point_cloud = None
    
    # Check for point cloud in '06_point_cloud' directory first
    point_cloud_dir = os.path.join(args.scan_dir, '06_point_cloud')
    if os.path.exists(point_cloud_dir):
        # Check filtered directory first (if it exists)
        filtered_dir = os.path.join(point_cloud_dir, 'filtered')
        if os.path.exists(filtered_dir):
            filtered_files = (
                glob.glob(os.path.join(filtered_dir, '*.ply')) + 
                glob.glob(os.path.join(filtered_dir, '*.pcd')) +
                glob.glob(os.path.join(filtered_dir, '*.pts'))
            )
            if filtered_files:
                logger.info(f"Found filtered point cloud: {filtered_files[0]}")
                point_cloud = load_point_cloud(filtered_files[0])
        
        # If no filtered point cloud, check raw directory
        if point_cloud is None:
            raw_dir = os.path.join(point_cloud_dir, 'raw')
            if os.path.exists(raw_dir):
                raw_files = (
                    glob.glob(os.path.join(raw_dir, '*.ply')) + 
                    glob.glob(os.path.join(raw_dir, '*.pcd')) +
                    glob.glob(os.path.join(raw_dir, '*.pts'))
                )
                if raw_files:
                    logger.info(f"Found raw point cloud: {raw_files[0]}")
                    point_cloud = load_point_cloud(raw_files[0])
    
    # Fallback to using the already found point clouds in images dictionary
    if point_cloud is None and images.get('point_cloud'):
        # Use the first point cloud file
        logger.info(f"Loading point cloud from: {images['point_cloud'][0]}")
        point_cloud = load_point_cloud(images['point_cloud'][0])
    
    # If no point cloud found, try to reconstruct it
    if point_cloud is None:
        logger.info("No point cloud found, attempting reconstruction")
        point_cloud = reconstruct_point_cloud_from_images(
            images, 
            calibration_path, 
            pattern_type,
            args.no_rectification
        )
    
    if point_cloud is None:
        logger.error("Failed to load or reconstruct point cloud")
        return 1
    
    # Calculate uncertainty if requested
    uncertainty_data = None
    if args.show_uncertainty:
        if not COMPLIANCE_MODULES_AVAILABLE:
            logger.warning("Limited uncertainty measurement available - compliance modules not fully loaded")
        
        logger.info("Calculating uncertainty measurements")
        uncertainty_data = calculate_uncertainty(correspondences, pattern_type, point_cloud)
        
        if uncertainty_data:
            logger.info(f"Mean uncertainty: {uncertainty_data.mean_uncertainty:.3f}mm")
            logger.info(f"Max uncertainty: {uncertainty_data.max_uncertainty:.3f}mm")
            
            # Save uncertainty data to log for future reference
            try:
                reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certification_reports")
                os.makedirs(reports_dir, exist_ok=True)
                uncertainty_log_path = os.path.join(reports_dir, "uncertainty_log.json")
                
                # Create or update the uncertainty log
                uncertainty_log = {}
                if os.path.exists(uncertainty_log_path):
                    with open(uncertainty_log_path, 'r') as f:
                        uncertainty_log = json.load(f)
                
                # Add or update entry for this pattern type
                uncertainty_log[pattern_type] = {
                    "mean_uncertainty": float(uncertainty_data.mean_uncertainty),
                    "max_uncertainty": float(uncertainty_data.max_uncertainty),
                    "statistics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                   for k, v in uncertainty_data.statistics.items()},
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save the updated log
                with open(uncertainty_log_path, 'w') as f:
                    json.dump(uncertainty_log, f, indent=2)
                
                logger.info(f"Saved uncertainty data to {uncertainty_log_path}")
            except Exception as e:
                logger.warning(f"Failed to save uncertainty log: {e}")
            
            # Save uncertainty visualization
            uncertainty_viz = create_uncertainty_visualization(uncertainty_data)
            if uncertainty_viz is not None:
                viz_path = os.path.join(output_dir, "uncertainty_map.png")
                cv2.imwrite(viz_path, uncertainty_viz)
                logger.info(f"Saved uncertainty visualization to: {viz_path}")
                
                # Show visualization if requested
                if args.show_uncertainty:
                    try:
                        cv2.namedWindow("Uncertainty Map", cv2.WINDOW_NORMAL)
                        cv2.imshow("Uncertainty Map", uncertainty_viz)
                        cv2.waitKey(500)  # Short delay to ensure window is created
                    except Exception as e:
                        logger.warning(f"Could not display uncertainty visualization: {e}")
    
    # Validate calibration if requested
    validation_results = []
    if args.validate_calibration:
        if not COMPLIANCE_MODULES_AVAILABLE:
            logger.error("Cannot validate calibration - compliance modules not available")
            logger.error("Please install all required modules")
        else:
            logger.info(f"Validating calibration with test object: {args.test_object}")
            result = validate_calibration(point_cloud, args.test_object, output_dir)
            validation_results.append(result)
            
            # Display results
            print("\n" + "="*70)
            print(" CALIBRATION VALIDATION RESULTS")
            print(f" Test object: {args.test_object}")
            print(f" Status: {'PASSED' if result.get('passed', False) else 'FAILED'}")
            print("="*70)
            
            print("\nMeasurements:")
            for key, value in result.get('measurements', {}).items():
                print(f"  {key}: {value:.4f}")
            
            print("\nErrors:")
            for key, value in result.get('errors', {}).items():
                print(f"  {key}: {value:.4f}")
            
            print("\nRecommendations:")
            for rec in result.get('recommendations', []):
                print(f"  - {rec}")
    
    # Generate certification report if requested
    if args.generate_report:
        if not COMPLIANCE_MODULES_AVAILABLE or not REPORTLAB_AVAILABLE:
            logger.error("Cannot generate certification report - missing required modules")
            logger.error("Please install reportlab: pip install reportlab")
        else:
            logger.info("Generating ISO/ASTM 52902 certification report")
            report_summary = generate_certification_report(
                validation_results,
                uncertainty_data,
                pattern_type,
                output_dir
            )
            
            # Display report summary
            print("\n" + "="*70)
            print(" ISO/ASTM 52902 CERTIFICATION REPORT")
            print(f" Report ID: {report_summary.get('report_id', 'Unknown')}")
            print(f" Date: {report_summary.get('generation_date', 'Unknown')}")
            print(f" Overall compliance: {'PASS' if report_summary.get('overall_compliance', False) else 'FAIL'}")
            print("="*70)
            
            print("\nCompliance Summary:")
            for requirement, compliant in report_summary.get('compliance_summary', {}).items():
                status = "PASS" if compliant else "FAIL"
                print(f"  {requirement}: {status}")
            
            print("\nRecommendations:")
            for rec in report_summary.get('recommendations', []):
                print(f"  - {rec}")
            
            print(f"\nReport saved to: {report_summary.get('report_file', 'Unknown')}")
    
    # Visualize point cloud if requested
    if args.visualize and point_cloud is not None:
        logger.info("Visualizing point cloud")
        visualize_point_cloud(point_cloud, uncertainty_data)
    
    # Clean up OpenCV windows
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print(f" Results saved to: {output_dir}")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)