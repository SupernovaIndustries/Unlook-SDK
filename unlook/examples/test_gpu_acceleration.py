#!/usr/bin/env python3
"""
GPU Acceleration Test Script for Unlook SDK

This script tests the GPU acceleration capabilities of the Unlook SDK
by running the triangulation and correspondence matching operations
on sample data from the unlook_debug folder.

Usage:
    python test_gpu_acceleration.py [--scan-folder FOLDER] [--cpu-only]
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import cv2
from pathlib import Path
import json

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import Unlook SDK modules
from unlook.client.gpu_utils import get_gpu_accelerator, is_gpu_available, diagnose_gpu
from unlook.utils.cuda_setup import setup_cuda_env, is_cuda_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_acceleration_test")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GPU Acceleration Test for Unlook SDK')
    
    parser.add_argument('--scan-folder', type=str, 
                      default=None,
                      help='Folder containing scan data (default: most recent scan in unlook_debug)')
    parser.add_argument('--cpu-only', action='store_true',
                      help='Force CPU-only processing (for comparison)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def find_most_recent_scan_folder():
    """Find the most recent scan folder in the unlook_debug directory."""
    unlook_debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlook_debug")
    
    if not os.path.exists(unlook_debug_dir):
        logger.error(f"Unlook debug directory not found: {unlook_debug_dir}")
        return None
    
    # Find all scan folders
    scan_folders = []
    for item in os.listdir(unlook_debug_dir):
        item_path = os.path.join(unlook_debug_dir, item)
        if os.path.isdir(item_path) and item.startswith("static_scan_"):
            scan_folders.append(item_path)
    
    if not scan_folders:
        logger.error("No scan folders found in unlook_debug directory")
        return None
    
    # Sort by modification time (most recent first)
    scan_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return scan_folders[0]

def get_projection_matrices(scan_folder):
    """
    Try to find or create projection matrices from the scan folder.
    
    Returns:
        Tuple of (P1, P2) projection matrices
    """
    # Default projection matrices if we can't find calibration data
    P1 = np.array([
        [800, 0, 640, 0],
        [0, 800, 360, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    P2 = np.array([
        [800, 0, 640, -80000],
        [0, 800, 360, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    # Try to find calibration data in the scan folder
    calib_file = os.path.join(scan_folder, "calibration.json")
    if os.path.exists(calib_file):
        try:
            with open(calib_file, "r") as f:
                calib_data = json.load(f)
            
            # Check if the calibration data has projection matrices
            if "P1" in calib_data and "P2" in calib_data:
                P1 = np.array(calib_data["P1"], dtype=np.float32)
                P2 = np.array(calib_data["P2"], dtype=np.float32)
                logger.info("Using projection matrices from calibration file")
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
    
    return P1, P2

def load_sample_data(scan_folder):
    """
    Load sample data from the scan folder.
    
    Returns:
        Dictionary with sample data
    """
    logger.info(f"Loading sample data from {scan_folder}")
    
    # Initialize data dictionary
    data = {
        "left_images": [],
        "right_images": [],
        "left_coords": None,
        "right_coords": None,
        "mask_left": None,
        "mask_right": None,
        "points_left": None,
        "points_right": None,
        "P1": None,
        "P2": None
    }
    
    # Load rectified images
    rectified_dir = os.path.join(scan_folder, "rectified")
    if os.path.exists(rectified_dir):
        # Count number of image pairs
        left_prefix = "rect_left_"
        right_prefix = "rect_right_"
        
        # Get all left and right image files
        left_files = sorted([f for f in os.listdir(rectified_dir) if f.startswith(left_prefix)])
        right_files = sorted([f for f in os.listdir(rectified_dir) if f.startswith(right_prefix)])
        
        # Load images
        for left_file, right_file in zip(left_files, right_files):
            left_path = os.path.join(rectified_dir, left_file)
            right_path = os.path.join(rectified_dir, right_file)
            
            left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            
            if left_img is not None and right_img is not None:
                data["left_images"].append(left_img)
                data["right_images"].append(right_img)
    
    # Load masks
    masks_dir = os.path.join(scan_folder, "masks")
    if os.path.exists(masks_dir):
        mask_left_path = os.path.join(masks_dir, "mask_left.png")
        mask_right_path = os.path.join(masks_dir, "mask_right.png")
        
        if os.path.exists(mask_left_path) and os.path.exists(mask_right_path):
            data["mask_left"] = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE) > 0
            data["mask_right"] = cv2.imread(mask_right_path, cv2.IMREAD_GRAYSCALE) > 0
            
            logger.info(f"Loaded masks: {data['mask_left'].shape}, {data['mask_right'].shape}")
        
        # Try to load coordinate maps
        coord_map_left_path = os.path.join(masks_dir, "coordinate_map_left.png")
        coord_map_right_path = os.path.join(masks_dir, "coordinate_map_right.png")
        
        if os.path.exists(coord_map_left_path) and os.path.exists(coord_map_right_path):
            coord_map_left = cv2.imread(coord_map_left_path, cv2.IMREAD_GRAYSCALE)
            coord_map_right = cv2.imread(coord_map_right_path, cv2.IMREAD_GRAYSCALE)
            
            if coord_map_left is not None and coord_map_right is not None:
                # Convert from 8-bit image to coordinates
                h, w = coord_map_left.shape
                data["left_coords"] = np.zeros((h, w, 2), dtype=np.float32)
                data["right_coords"] = np.zeros((h, w, 2), dtype=np.float32)
                
                data["left_coords"][:, :, 0] = coord_map_left.astype(np.float32)
                data["right_coords"][:, :, 0] = coord_map_right.astype(np.float32)
                
                logger.info(f"Loaded coordinate maps: {data['left_coords'].shape}, {data['right_coords'].shape}")
        
        # Try to load correspondence points
        disparity_path = os.path.join(masks_dir, "disparity.png")
        if os.path.exists(disparity_path):
            # We need to create some sample correspondence points for testing
            # This won't be the actual correspondences but will work for testing
            h, w = data["mask_left"].shape if data["mask_left"] is not None else (480, 640)
            
            # Create sample points (50 random points)
            num_points = 1000
            points_left = np.zeros((num_points, 1, 2), dtype=np.float32)
            points_right = np.zeros((num_points, 1, 2), dtype=np.float32)
            
            # Generate random points within the valid mask area
            if data["mask_left"] is not None and data["mask_right"] is not None:
                valid_y, valid_x = np.where(data["mask_left"])
                if len(valid_y) > 0:
                    indices = np.random.choice(len(valid_y), min(num_points, len(valid_y)), replace=False)
                    
                    for i, idx in enumerate(indices):
                        y, x = valid_y[idx], valid_x[idx]
                        points_left[i, 0, 0] = x
                        points_left[i, 0, 1] = y
                        
                        # For right points, add some disparity (shift left)
                        disparity = np.random.randint(20, 50)  # Random disparity between 20-50 pixels
                        points_right[i, 0, 0] = max(0, x - disparity)
                        points_right[i, 0, 1] = y
            else:
                # Generate random points if no masks are available
                for i in range(num_points):
                    x = np.random.randint(0, w)
                    y = np.random.randint(0, h)
                    
                    points_left[i, 0, 0] = x
                    points_left[i, 0, 1] = y
                    
                    # For right points, add some disparity (shift left)
                    disparity = np.random.randint(20, 50)
                    points_right[i, 0, 0] = max(0, x - disparity)
                    points_right[i, 0, 1] = y
            
            data["points_left"] = points_left
            data["points_right"] = points_right
            
            logger.info(f"Created {num_points} sample correspondence points")
    
    # Get projection matrices
    data["P1"], data["P2"] = get_projection_matrices(scan_folder)
    
    return data

def test_triangulation(gpu_accelerator, data, use_gpu=True):
    """
    Test triangulation performance.
    
    Args:
        gpu_accelerator: GPU accelerator instance
        data: Dictionary with sample data
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with performance results
    """
    logger.info(f"Testing triangulation performance (GPU: {use_gpu})")
    
    if data["points_left"] is None or data["points_right"] is None:
        logger.error("No correspondence points available for triangulation test")
        return None
    
    # Make sure we have at least 100 points for a meaningful test
    num_points = min(len(data["points_left"]), 1000)
    if num_points < 100:
        logger.warning(f"Only {num_points} correspondence points available. Test results may not be reliable.")
    
    # Extract points and convert to the format expected by triangulation function
    points_left = data["points_left"][:num_points].reshape(-1, 2).T  # Convert to 2xN
    points_right = data["points_right"][:num_points].reshape(-1, 2).T  # Convert to 2xN
    
    # Set up projection matrices
    P1 = data["P1"]
    P2 = data["P2"]
    
    # Run triangulation benchmark
    num_runs = 5
    cpu_times = []
    gpu_times = []
    
    # First run CPU triangulation
    for i in range(num_runs):
        start_time = time.time()
        points_3d_cpu = gpu_accelerator._triangulate_points_cpu(P1, P2, points_left, points_right)
        cpu_times.append(time.time() - start_time)
        
        logger.info(f"CPU triangulation run {i+1}/{num_runs}: {cpu_times[-1]:.4f} seconds")
    
    # Then run GPU triangulation if requested
    if use_gpu and gpu_accelerator.gpu_available:
        for i in range(num_runs):
            start_time = time.time()
            points_3d_gpu = gpu_accelerator.triangulate_points_gpu(P1, P2, points_left, points_right)
            gpu_times.append(time.time() - start_time)
            
            logger.info(f"GPU triangulation run {i+1}/{num_runs}: {gpu_times[-1]:.4f} seconds")
    
    # Calculate average times
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else None
    
    # Calculate speedup
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time else None
    
    results = {
        "test": "triangulation",
        "num_points": num_points,
        "avg_cpu_time": avg_cpu_time,
        "avg_gpu_time": avg_gpu_time,
        "speedup": speedup,
        "cpu_times": cpu_times,
        "gpu_times": gpu_times
    }
    
    logger.info(f"Triangulation test results:")
    logger.info(f"  Number of points: {num_points}")
    logger.info(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
    
    if avg_gpu_time is not None:
        logger.info(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
    else:
        logger.info("  GPU triangulation not tested")
    
    return results

def test_correspondence_matching(gpu_accelerator, data, use_gpu=True):
    """
    Test correspondence matching performance.
    
    Args:
        gpu_accelerator: GPU accelerator instance
        data: Dictionary with sample data
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with performance results
    """
    logger.info(f"Testing correspondence matching performance (GPU: {use_gpu})")
    
    if data["left_coords"] is None or data["right_coords"] is None or data["mask_left"] is None or data["mask_right"] is None:
        logger.error("Required data not available for correspondence matching test")
        return None
    
    # Get sample data
    left_coords = data["left_coords"]
    right_coords = data["right_coords"]
    mask_left = data["mask_left"]
    mask_right = data["mask_right"]
    
    # Run correspondence matching benchmark
    # Only 1 run because this operation is slow
    cpu_times = []
    gpu_times = []
    
    # Define a simplified CPU correspondence matching function for testing
    def find_correspondences_cpu(left_coords, right_coords, mask_left, mask_right):
        logger.info("Running CPU correspondence matching (simplified version)...")
        
        start_time = time.time()
        h, w = mask_left.shape
        
        # Only process a subset of points for faster testing
        step = 10  # Process every 10th pixel
        points_left = []
        points_right = []
        
        # Find valid points in left image
        valid_y, valid_x = np.where(mask_left)
        
        # Process a random subset (maximum 1000 points)
        max_points = min(1000, len(valid_y))
        indices = np.random.choice(len(valid_y), max_points, replace=False)
        
        for idx in indices:
            y, x = valid_y[idx], valid_x[idx]
            
            # Get projector coordinate
            proj_coord = left_coords[y, x, 0]
            if proj_coord == 0:
                continue
            
            # Define epipolar search region
            epipolar_tolerance = 3
            min_y = max(0, y - epipolar_tolerance)
            max_y = min(h - 1, y + epipolar_tolerance)
            
            best_match_x = -1
            best_match_y = -1
            best_match_dist = float('inf')
            
            # Search along epipolar line
            for y_right in range(min_y, max_y + 1):
                # Search up to current x position (for expected disparity)
                search_range = min(x, w - 1)
                
                for x_right in range(search_range):
                    if not mask_right[y_right, x_right]:
                        continue
                    
                    # Get projector coordinate in right image
                    right_proj_coord = right_coords[y_right, x_right, 0]
                    
                    # Calculate coordinate difference
                    diff = abs(proj_coord - right_proj_coord)
                    
                    # Update best match
                    if diff < best_match_dist and diff < 15:  # Threshold for matching
                        best_match_dist = diff
                        best_match_x = x_right
                        best_match_y = y_right
            
            # If a match was found, add to correspondences
            if best_match_x >= 0:
                points_left.append([x, y])
                points_right.append([best_match_x, best_match_y])
        
        # Convert to numpy arrays
        points_left = np.array(points_left).reshape(-1, 1, 2).astype(np.float32)
        points_right = np.array(points_right).reshape(-1, 1, 2).astype(np.float32)
        
        elapsed = time.time() - start_time
        logger.info(f"CPU correspondence matching found {len(points_left)} matches in {elapsed:.2f} seconds")
        
        return points_left, points_right, elapsed
    
    # Run CPU correspondence matching
    points_left_cpu, points_right_cpu, cpu_time = find_correspondences_cpu(
        left_coords, right_coords, mask_left, mask_right
    )
    cpu_times.append(cpu_time)
    
    # Run GPU correspondence matching if available
    if use_gpu and gpu_accelerator.gpu_available:
        start_time = time.time()
        points_left_gpu, points_right_gpu = gpu_accelerator.find_correspondences_gpu(
            left_coords, right_coords, mask_left, mask_right,
            epipolar_tolerance=3.0, min_disparity=5, max_disparity=100, gray_code_threshold=15
        )
        gpu_time = time.time() - start_time
        
        if points_left_gpu is not None and points_right_gpu is not None:
            gpu_times.append(gpu_time)
            logger.info(f"GPU correspondence matching found {len(points_left_gpu)} matches in {gpu_time:.2f} seconds")
        else:
            logger.warning("GPU correspondence matching failed")
    
    # Calculate average times
    avg_cpu_time = sum(cpu_times) / len(cpu_times) if cpu_times else None
    avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else None
    
    # Calculate speedup
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time and avg_cpu_time else None
    
    results = {
        "test": "correspondence_matching",
        "num_cpu_matches": len(points_left_cpu) if points_left_cpu is not None else 0,
        "num_gpu_matches": len(points_left_gpu) if 'points_left_gpu' in locals() and points_left_gpu is not None else 0,
        "avg_cpu_time": avg_cpu_time,
        "avg_gpu_time": avg_gpu_time,
        "speedup": speedup
    }
    
    logger.info(f"Correspondence matching test results:")
    logger.info(f"  CPU matches: {results['num_cpu_matches']}")
    logger.info(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
    
    if avg_gpu_time is not None:
        logger.info(f"  GPU matches: {results['num_gpu_matches']}")
        logger.info(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
    else:
        logger.info("  GPU correspondence matching not tested or failed")
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CUDA environment
    logger.info("Initializing CUDA environment...")
    if setup_cuda_env():
        logger.info("CUDA environment set up successfully")
        
        # Check CUDA availability
        cuda_available = is_cuda_available()
        logger.info(f"CUDA available: {cuda_available}")
    else:
        logger.warning("Failed to set up CUDA environment")
    
    # Check GPU availability
    gpu_available = is_gpu_available()
    if gpu_available:
        logger.info("GPU acceleration is available")
        
        # Get detailed GPU information
        gpu_info = diagnose_gpu()
        logger.info(f"GPU information:")
        for k, v in gpu_info.items():
            logger.info(f"  {k}: {v}")
    else:
        logger.warning("GPU acceleration is not available")
    
    # Override GPU acceleration if requested
    use_gpu = gpu_available and not args.cpu_only
    
    # Initialize GPU accelerator
    gpu_accelerator = get_gpu_accelerator(enable_gpu=use_gpu)
    
    # Find scan folder
    scan_folder = args.scan_folder
    if scan_folder is None:
        scan_folder = find_most_recent_scan_folder()
        if scan_folder is None:
            logger.error("Could not find a scan folder to use. Please specify one with --scan-folder.")
            return
    
    # Load sample data
    data = load_sample_data(scan_folder)
    
    # Run tests
    triangulation_results = test_triangulation(gpu_accelerator, data, use_gpu=use_gpu)
    correspondence_results = test_correspondence_matching(gpu_accelerator, data, use_gpu=use_gpu)
    
    # Summary
    logger.info("\nTest Summary:")
    logger.info(f"Scan folder: {scan_folder}")
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    logger.info(f"GPU available: {gpu_available}")
    
    # Triangulation summary
    if triangulation_results:
        logger.info("\nTriangulation:")
        logger.info(f"  Points: {triangulation_results['num_points']}")
        logger.info(f"  CPU time: {triangulation_results['avg_cpu_time']:.4f} seconds")
        if triangulation_results['avg_gpu_time'] is not None:
            logger.info(f"  GPU time: {triangulation_results['avg_gpu_time']:.4f} seconds")
            logger.info(f"  Speedup: {triangulation_results['speedup']:.2f}x")
        else:
            logger.info("  GPU triangulation not tested")
    
    # Correspondence matching summary
    if correspondence_results:
        logger.info("\nCorrespondence Matching:")
        logger.info(f"  CPU matches: {correspondence_results['num_cpu_matches']}")
        logger.info(f"  CPU time: {correspondence_results['avg_cpu_time']:.4f} seconds")
        if correspondence_results['avg_gpu_time'] is not None:
            logger.info(f"  GPU matches: {correspondence_results['num_gpu_matches']}")
            logger.info(f"  GPU time: {correspondence_results['avg_gpu_time']:.4f} seconds")
            logger.info(f"  Speedup: {correspondence_results['speedup']:.2f}x")
        else:
            logger.info("  GPU correspondence matching not tested or failed")

if __name__ == "__main__":
    main()