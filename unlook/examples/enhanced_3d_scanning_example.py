#!/usr/bin/env python3
"""
Example script demonstrating the enhanced 3D scanning capabilities of the UnLook SDK.

This example demonstrates:
1. Setting up the stereo structured light scanner with calibration
2. Generating scanning patterns
3. Capturing structured light images
4. Processing the images to create a 3D point cloud
5. Creating a 3D mesh from the point cloud
6. Saving results to disk

Usage:
    python enhanced_3d_scanning_example.py
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import argparse
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Try to import optional dependencies
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("open3d is not installed. Visualization and mesh creation will be disabled.")
    print("To enable all features, install open3d: pip install open3d")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from unlook import UnlookClient
try:
    from unlook.client.structured_light import (
        StereoStructuredLightScanner, 
        StereoCalibrator,
        StereoCameraParameters,
        create_scanning_demo
    )
except ImportError as e:
    logger.error(f"Error importing structured_light module: {e}")
    print("Please make sure you have the required dependencies installed:")
    print("  numpy, opencv-python")
    print("For full functionality, also install:")
    print("  open3d (pip install open3d)")
    sys.exit(1)


def setup_scanner(client: UnlookClient, output_dir: str) -> StereoStructuredLightScanner:
    """
    Set up the stereo structured light scanner.
    
    Args:
        client: Connected UnlookClient
        output_dir: Output directory for calibration and results
        
    Returns:
        Configured StereoStructuredLightScanner
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # In a production environment, you'd use real calibration data
    # For this example, we create a demo scanner with default parameters
    logger.info("Setting up scanner with default calibration")
    return create_scanning_demo(output_dir)


def capture_structured_light_images(client: UnlookClient, scanner: StereoStructuredLightScanner,
                                  output_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Capture structured light images using the pattern sequence from the scanner.
    
    Note: Currently, the ProjectorClient doesn't support raw image display directly.
    This example uses an approximation with standard patterns based on pattern names.
    Future SDK versions will add proper raw image support for precise structured light patterns.
    
    Args:
        client: Connected UnlookClient
        scanner: Configured StereoStructuredLightScanner
        output_dir: Directory to save captured images
        
    Returns:
        Tuple of (left_images, right_images)
    """
    # Ensure we have a clean output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving captured images to {output_dir}")
    
    # Get the pattern sequence from the scanner
    patterns = scanner.generate_scan_patterns()
    logger.info(f"Generated {len(patterns)} structured light patterns")
    
    # Save pattern information for debugging
    pattern_info_path = os.path.join(output_dir, "pattern_info.json")
    try:
        # Create a simplified version of patterns without binary data for debugging
        pattern_info = []
        for idx, p in enumerate(patterns):
            info = {
                "index": idx,
                "name": p.get("name", f"pattern_{idx}"),
                "type": p.get("pattern_type", "unknown"),
            }
            # Add other keys except 'image' which is binary
            for k, v in p.items():
                if k != "image" and k != "name" and k != "pattern_type":
                    info[k] = str(v)
            pattern_info.append(info)
            
        with open(pattern_info_path, 'w') as f:
            json.dump(pattern_info, f, indent=2)
        logger.info(f"Saved pattern metadata to {pattern_info_path}")
    except Exception as e:
        logger.warning(f"Could not save pattern info: {e}")
    
    # Set up projector and cameras
    projector = client.projector
    camera = client.camera
    
    # Get available cameras and configure them for optimal capture
    cameras = camera.get_cameras()
    
    if len(cameras) < 2:
        logger.error(f"Need at least 2 cameras, found {len(cameras)}")
        # For demo purposes, we'll generate synthetic test images
        return generate_test_images(patterns, output_dir)
    
    # Select the first two cameras
    left_camera_id = cameras[0]["id"]
    right_camera_id = cameras[1]["id"]
    
    # Configure cameras for optimal scanning
    try:
        # Set exposure and other parameters for better scanning
        camera_config = {
            "exposure": 100,  # Set a fixed exposure to avoid fluctuations
            "auto_exposure": False,  # Turn off auto exposure for consistency
            "format": "png",  # Use PNG for lossless image saving
            "quality": 100    # Use maximum quality
        }
        
        # Apply configuration to both cameras
        for cam_id in [left_camera_id, right_camera_id]:
            camera.configure_camera(cam_id, camera_config)
            logger.info(f"Configured camera {cam_id} for scanning")
    except Exception as e:
        logger.warning(f"Could not configure cameras: {e}")
        logger.warning("Using default camera settings")
    
    # Capture images for each pattern
    left_images = []
    right_images = []
    
    logger.info("Starting structured light capture sequence")
    
    # Display a black pattern first to prepare and calibrate exposure
    logger.info("Showing black calibration pattern")
    projector.show_solid_field("Black")
    time.sleep(1.0)  # Wait for the projector to update
    
    # Take a test shot to warm up cameras
    try:
        camera.capture(left_camera_id)
        camera.capture(right_camera_id)
        logger.info("Test capture completed")
    except Exception as e:
        logger.warning(f"Test capture failed: {e}")
    
    # Prepare pattern display and capture functions for better organization
    def display_pattern(pattern_obj, idx):
        """Display a pattern on the projector"""
        try:
            logger.info(f"Projecting pattern {idx+1}/{len(patterns)}: {pattern_obj.get('name', f'pattern_{idx}')}")
            
            if pattern_obj["pattern_type"] == "raw_image":
                # For raw images, adapt based on pattern name
                name = pattern_obj.get('name', '').lower()
                # Scale factor for pattern frequency - more patterns = higher frequency
                scale = max(1, int((idx % 5) + 1)) if 'gray_code' in name else 1
                
                if 'white' in name:
                    return projector.show_solid_field("White")
                elif 'black' in name:
                    return projector.show_solid_field("Black")
                elif 'gray_code_x' in name or 'gray_code_0' in name or 'gray_code_2' in name:
                    # Use horizontal lines with varying width for X patterns
                    # Alternating width helps create better structured light patterns
                    width = max(1, int(20 / (1 + (idx % 8)))) if idx > 0 else 4
                    
                    return projector.show_horizontal_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=scale * width,
                        background_width=scale * width
                    )
                elif 'gray_code_y' in name or 'gray_code_1' in name or 'gray_code_3' in name:
                    # Use vertical lines with varying width for Y patterns
                    width = max(1, int(20 / (1 + (idx % 8)))) if idx > 0 else 4
                    
                    return projector.show_vertical_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=scale * width,
                        background_width=scale * width
                    )
                else:
                    # Create a wider variety of patterns for better coverage
                    pattern_idx = idx % 4
                    if pattern_idx == 0:
                        return projector.show_grid(
                            foreground_color="White",
                            background_color="Black",
                            h_foreground_width=2,
                            h_background_width=6,
                            v_foreground_width=2,
                            v_background_width=6
                        )
                    elif pattern_idx == 1:
                        return projector.show_checkerboard(
                            foreground_color="White",
                            background_color="Black",
                            horizontal_count=6 + (idx % 10),
                            vertical_count=4 + (idx % 10)
                        )
                    elif pattern_idx == 2:
                        return projector.show_horizontal_lines(
                            foreground_color="White",
                            background_color="Black",
                            foreground_width=1 + (idx % 8),
                            background_width=1 + (idx % 8)
                        )
                    else:
                        return projector.show_vertical_lines(
                            foreground_color="White",
                            background_color="Black",
                            foreground_width=1 + (idx % 8),
                            background_width=1 + (idx % 8)
                        )
            else:
                # For predefined patterns, dispatch to the appropriate method
                pattern_type = pattern_obj.get("pattern_type", "")
                
                if pattern_type == "solid_field":
                    return projector.show_solid_field(pattern_obj.get("color", "White"))
                elif pattern_type == "horizontal_lines":
                    return projector.show_horizontal_lines(
                        foreground_color=pattern_obj.get("foreground_color", "White"),
                        background_color=pattern_obj.get("background_color", "Black"),
                        foreground_width=pattern_obj.get("foreground_width", 4),
                        background_width=pattern_obj.get("background_width", 20)
                    )
                elif pattern_type == "vertical_lines":
                    return projector.show_vertical_lines(
                        foreground_color=pattern_obj.get("foreground_color", "White"),
                        background_color=pattern_obj.get("background_color", "Black"),
                        foreground_width=pattern_obj.get("foreground_width", 4),
                        background_width=pattern_obj.get("background_width", 20)
                    )
                elif pattern_type == "grid":
                    return projector.show_grid(
                        foreground_color=pattern_obj.get("foreground_color", "White"),
                        background_color=pattern_obj.get("background_color", "Black"),
                        h_foreground_width=pattern_obj.get("h_foreground_width", 4),
                        h_background_width=pattern_obj.get("h_background_width", 20),
                        v_foreground_width=pattern_obj.get("v_foreground_width", 4),
                        v_background_width=pattern_obj.get("v_background_width", 20)
                    )
                elif pattern_type == "checkerboard":
                    return projector.show_checkerboard(
                        foreground_color=pattern_obj.get("foreground_color", "White"),
                        background_color=pattern_obj.get("background_color", "Black"),
                        horizontal_count=pattern_obj.get("horizontal_count", 8),
                        vertical_count=pattern_obj.get("vertical_count", 6)
                    )
                elif pattern_type == "colorbars":
                    return projector.show_colorbars()
                else:
                    # Default to solid white if pattern type not recognized
                    logger.warning(f"Unrecognized pattern type: {pattern_type}, defaulting to white")
                    return projector.show_solid_field("White")
        except Exception as e:
            logger.error(f"Error displaying pattern {idx+1}: {e}")
            return False
    
    def capture_and_save(idx, retry_count=0):
        """Capture images from both cameras and save them"""
        max_retries = 3
        
        try:
            if retry_count > 0:
                logger.info(f"Retry attempt {retry_count}/{max_retries} for pattern {idx+1}")
            
            # Capture from left camera
            left_img = camera.capture(left_camera_id)
            # Capture from right camera
            right_img = camera.capture(right_camera_id)
            
            # Check if images are valid
            if left_img is None or left_img.size == 0 or right_img is None or right_img.size == 0:
                error_msg = "Captured images are empty or invalid"
                if retry_count < max_retries:
                    logger.warning(f"{error_msg}. Retrying...")
                    time.sleep(0.5)  # Wait a bit before retry
                    return capture_and_save(idx, retry_count + 1)
                else:
                    logger.error(f"{error_msg}. Max retries reached.")
                    return None, None
            
            # Check image dimensions
            if left_img.shape[0] < 10 or left_img.shape[1] < 10 or right_img.shape[0] < 10 or right_img.shape[1] < 10:
                error_msg = f"Images too small: left={left_img.shape}, right={right_img.shape}"
                if retry_count < max_retries:
                    logger.warning(f"{error_msg}. Retrying...")
                    time.sleep(0.5)  # Wait a bit before retry
                    return capture_and_save(idx, retry_count + 1)
                else:
                    logger.error(f"{error_msg}. Max retries reached.")
                    return None, None
            
            # Create debug info overlay
            left_debug = left_img.copy()
            right_debug = right_img.copy()
            
            # Add pattern number and information
            cv2.putText(
                left_debug, 
                f"Pattern {idx+1}: {patterns[idx].get('name', '')}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                right_debug, 
                f"Pattern {idx+1}: {patterns[idx].get('name', '')}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # File paths
            left_path = os.path.join(output_dir, f"left_{idx:03d}.png")
            right_path = os.path.join(output_dir, f"right_{idx:03d}.png")
            left_debug_path = os.path.join(output_dir, f"left_{idx:03d}_debug.jpg")
            right_debug_path = os.path.join(output_dir, f"right_{idx:03d}_debug.jpg")
            
            # Save the captured images in high quality PNG format
            if not cv2.imwrite(left_path, left_img):
                logger.error(f"Failed to save left image to {left_path}")
            
            if not cv2.imwrite(right_path, right_img):
                logger.error(f"Failed to save right image to {right_path}")
            
            # Save the debug images in JPEG format
            cv2.imwrite(left_debug_path, left_debug, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(right_debug_path, right_debug, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            logger.info(f"Saved image pair {idx+1} to {left_path} and {right_path}")
            
            return left_img, right_img
            
        except Exception as e:
            if retry_count < max_retries:
                logger.warning(f"Error capturing images for pattern {idx+1}: {e}. Retrying...")
                time.sleep(0.5)  # Wait a bit before retry
                return capture_and_save(idx, retry_count + 1)
            else:
                logger.error(f"Error capturing images for pattern {idx+1}: {e}. Max retries reached.")
                return None, None
    
    # Capture images for each pattern
    success_count = 0
    failure_count = 0
    
    for i, pattern in enumerate(patterns):
        # Display pattern
        display_success = display_pattern(pattern, i)
        
        if not display_success:
            logger.warning(f"Failed to display pattern {i+1}, skipping capture")
            continue
        
        # Wait for the projector to update - longer wait for better stability
        wait_time = 0.8 if i < 2 else 0.5  # Longer for first patterns
        time.sleep(wait_time)
        
        # Capture and save images
        logger.info(f"Capturing image pair for pattern {i+1}/{len(patterns)}")
        left_img, right_img = capture_and_save(i)
        
        if left_img is not None and right_img is not None:
            # Add to our lists
            left_images.append(left_img)
            right_images.append(right_img)
            success_count += 1
        else:
            logger.warning(f"Failed to capture valid image pair for pattern {i+1}")
            failure_count += 1
            
        # Every 10 patterns, show progress
        if (i + 1) % 10 == 0 or i == len(patterns) - 1:
            logger.info(f"Progress: {i+1}/{len(patterns)} patterns ({success_count} successful, {failure_count} failed)")
    
    # Reset projector to black
    projector.show_solid_field("Black")
    
    # Save capture statistics
    stats = {
        "total_patterns": len(patterns),
        "successful_captures": success_count,
        "failed_captures": failure_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    stats_path = os.path.join(output_dir, "capture_stats.json")
    try:
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save capture stats: {e}")
    
    logger.info(f"Captured {len(left_images)} image pairs for structured light scanning")
    
    return left_images, right_images


def generate_test_images(patterns: List[Dict[str, Any]], output_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate synthetic test images for structured light scanning.
    This is used when real cameras are not available.
    
    Args:
        patterns: List of pattern dictionaries
        output_dir: Directory to save generated images
        
    Returns:
        Tuple of (left_images, right_images)
    """
    logger.info("Generating synthetic test images for structured light scanning")
    
    # Create synthetic camera data
    left_images = []
    right_images = []
    
    # Size of the test images
    width, height = 1280, 720
    
    # Create a simple test scene with a sphere in the middle
    sphere_center = (width // 2, height // 2)
    sphere_radius = 200
    
    # Create background images
    background_left = np.zeros((height, width), dtype=np.uint8)
    background_right = np.zeros((height, width), dtype=np.uint8)
    
    # Add some ambient shading
    y, x = np.ogrid[:height, :width]
    background_left += 30 * np.exp(-((x - width//2)**2 + (y - height//2)**2) / (2 * 300**2))
    background_right += 30 * np.exp(-((x - width//2)**2 + (y - height//2)**2) / (2 * 300**2))
    
    # Create a sphere mask (a bit offset in the right image to simulate stereo disparity)
    sphere_mask_left = ((x - sphere_center[0])**2 + (y - sphere_center[1])**2 <= sphere_radius**2)
    sphere_mask_right = ((x - (sphere_center[0]-30))**2 + (y - sphere_center[1])**2 <= sphere_radius**2)
    
    # Process each pattern
    for i, pattern in enumerate(patterns):
        if i % 10 == 0:
            logger.info(f"Generating synthetic images for pattern {i+1}/{len(patterns)}")
        
        # Create base images
        left_img = background_left.copy()
        right_img = background_right.copy()
        
        # Pattern name for different processing
        pattern_name = pattern.get('name', '')
        
        if 'white' in pattern_name.lower():
            # Bright white pattern
            left_img[sphere_mask_left] = 200
            right_img[sphere_mask_right] = 200
        elif 'black' in pattern_name.lower():
            # Dark pattern
            left_img[sphere_mask_left] = 10
            right_img[sphere_mask_right] = 10
        elif 'gray_code' in pattern_name.lower():
            # Structured pattern (simple stripes for example)
            pattern_num = int(pattern_name.split('_')[-1])
            period = max(3, 10 + pattern_num * 5)  # Vary the stripe width
            
            # Create striped pattern
            stripe_pattern = np.zeros((height, width), dtype=np.uint8)
            stripe_pattern[:, ::period] = 255
            
            # Apply pattern to sphere with some attenuation
            left_img[sphere_mask_left] = np.minimum(background_left[sphere_mask_left] + 
                                                  stripe_pattern[sphere_mask_left] * 0.8, 255).astype(np.uint8)
            right_img[sphere_mask_right] = np.minimum(background_right[sphere_mask_right] + 
                                                    stripe_pattern[sphere_mask_right] * 0.8, 255).astype(np.uint8)
        else:
            # Generic pattern
            left_img[sphere_mask_left] = 150
            right_img[sphere_mask_right] = 150
        
        # Convert to 3-channel if needed
        if len(left_img.shape) == 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        
        # Save the images
        left_path = os.path.join(output_dir, f"synthetic_left_{i:03d}.png")
        right_path = os.path.join(output_dir, f"synthetic_right_{i:03d}.png")
        
        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)
        
        # Add to our lists
        left_images.append(left_img)
        right_images.append(right_img)
    
    logger.info(f"Generated {len(left_images)} synthetic image pairs")
    return left_images, right_images


def process_scans(scanner: StereoStructuredLightScanner, left_images: List[np.ndarray], 
                right_images: List[np.ndarray], output_dir: str) -> None:
    """
    Process the captured structured light images to create a 3D point cloud and mesh.
    
    Args:
        scanner: Configured StereoStructuredLightScanner
        left_images: List of left camera images
        right_images: List of right camera images
        output_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Processing structured light scans")
    logger.info(f"Processing {len(left_images)} image pairs")
    
    # Save processing metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_image_pairs": len(left_images),
        "image_dimensions": str(left_images[0].shape) if left_images else "None",
        "camera_parameters": {
            "left_resolution": list(scanner.stereo_params.left.resolution),
            "right_resolution": list(scanner.stereo_params.right.resolution),
        }
    }
    
    try:
        with open(os.path.join(output_dir, "processing_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save processing metadata: {e}")
    
    # Verify images look reasonable before processing
    if len(left_images) > 0 and len(right_images) > 0:
        # Create a comparative image of first image pair
        try:
            left_sample = left_images[0].copy()
            right_sample = right_images[0].copy()
            
            # Resize if needed
            if left_sample.shape[1] > 800:
                scale = 800 / left_sample.shape[1]
                left_sample = cv2.resize(left_sample, (0, 0), fx=scale, fy=scale)
                right_sample = cv2.resize(right_sample, (0, 0), fx=scale, fy=scale)
            
            # Ensure images are color for visualization
            if len(left_sample.shape) == 2:
                left_sample = cv2.cvtColor(left_sample, cv2.COLOR_GRAY2BGR)
                right_sample = cv2.cvtColor(right_sample, cv2.COLOR_GRAY2BGR)
            
            # Create a side-by-side comparison
            comparison = np.hstack((left_sample, right_sample))
            cv2.putText(comparison, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Right Camera", (left_sample.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw a center line
            cv2.line(comparison, 
                     (left_sample.shape[1], 0), 
                     (left_sample.shape[1], comparison.shape[0]), 
                     (0, 0, 255), 2)
            
            # Save comparison
            cv2.imwrite(os.path.join(output_dir, "camera_comparison.jpg"), comparison)
            logger.info(f"Saved camera comparison image to {os.path.join(output_dir, 'camera_comparison.jpg')}")
        except Exception as e:
            logger.warning(f"Could not create camera comparison image: {e}")
    
    # Process the scan to get a point cloud with additional filtering for better results
    try:
        # Process scan with more robust parameters
        logger.info("Processing scan with standard parameters")
        pcd = scanner.process_scan(left_images, right_images, mask_threshold=10)
        
        # If the point cloud has too few points, try with lower thresholds
        if len(pcd.points) < 100:
            logger.warning(f"Initial scan produced only {len(pcd.points)} points. Trying with lower thresholds...")
            pcd = scanner.process_scan(left_images, right_images, mask_threshold=5)
            
        if len(pcd.points) < 50:
            logger.warning(f"Still only {len(pcd.points)} points. Trying with minimal thresholds...")
            pcd = scanner.process_scan(left_images, right_images, mask_threshold=3)
    except Exception as e:
        logger.error(f"Error during scan processing: {e}")
        logger.error("Trying again with more lenient parameters...")
        
        try:
            # One more attempt with more lenient parameters
            pcd = scanner.process_scan(left_images, right_images, mask_threshold=2)
        except Exception as e2:
            logger.error(f"Failed again during scan processing: {e2}")
            return
    
    # Check if we have a valid point cloud
    if pcd is None or len(pcd.points) == 0:
        logger.error("No valid points were generated from the scans")
        return
    
    logger.info(f"Generated initial point cloud with {len(pcd.points)} points")
    
    # Apply additional filtering for better results if we have enough points
    if len(pcd.points) >= 10:
        try:
            # Apply statistical outlier removal
            filtered_pcd = scanner.filter_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0)
            if len(filtered_pcd.points) > 0:
                logger.info(f"Filtered point cloud from {len(pcd.points)} to {len(filtered_pcd.points)} points")
                pcd = filtered_pcd
        except Exception as e:
            logger.warning(f"Could not apply additional filtering: {e}")
    
    # Save the point cloud in multiple formats for compatibility
    if len(pcd.points) > 0:
        # Save in PLY format
        point_cloud_path = os.path.join(output_dir, "scan_point_cloud.ply")
        scanner.save_point_cloud(pcd, point_cloud_path)
        logger.info(f"Saved point cloud to {point_cloud_path}")
        
        # Also save in other formats if available
        scanner.save_point_cloud(pcd, os.path.join(output_dir, "scan_point_cloud.xyz"))
        
        # Now try to create a mesh
        if len(pcd.points) >= 100:
            try:
                logger.info("Creating mesh from point cloud with low detail (better for sparse data)")
                mesh = scanner.create_mesh_from_point_cloud(pcd, depth=8, smooth_iterations=5)
                
                # Save the mesh in different formats
                mesh_path = os.path.join(output_dir, "scan_mesh.ply")
                scanner.save_mesh(mesh, mesh_path)
                logger.info(f"Saved mesh to {mesh_path}")
                
                # Also save in OBJ format
                scanner.save_mesh(mesh, os.path.join(output_dir, "scan_mesh.obj"))
            except Exception as e:
                logger.error(f"Failed to create mesh: {e}")
        else:
            logger.warning(f"Not enough points ({len(pcd.points)}) to create a mesh, skipping mesh generation")
    
    # Save point statistics
    stats = {
        "num_points": len(pcd.points) if pcd is not None else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_pairs_used": len(left_images)
    }
    try:
        with open(os.path.join(output_dir, "point_cloud_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save point cloud stats: {e}")
    
    # Visualize results if open3d is available and we have points
    if OPEN3D_AVAILABLE and pcd is not None and len(pcd.points) > 0:
        try:
            logger.info("Visualizing results")
            o3d.visualization.draw_geometries([pcd], window_name="Structured Light Point Cloud")
        except Exception as e:
            logger.error(f"Failed to visualize results: {e}")
    else:
        logger.warning(f"Skipping visualization: open3d available={OPEN3D_AVAILABLE}, points={len(pcd.points) if pcd is not None else 0}")


def create_timestamped_scan_folder(base_dir: str) -> Tuple[str, str, str]:
    """
    Create a timestamped scan folder structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Tuple of (scan_dir, captures_dir, results_dir)
    """
    # Create timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scan_dir = os.path.join(base_dir, f"Scan_{timestamp}")
    
    # Create scan directory
    os.makedirs(scan_dir, exist_ok=True)
    
    # Create subdirectories
    captures_dir = os.path.join(scan_dir, "captures")
    results_dir = os.path.join(scan_dir, "results")
    
    os.makedirs(captures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Created scan folder structure at {scan_dir}")
    return scan_dir, captures_dir, results_dir

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced 3D scanning example")
    parser.add_argument("--output", "-o", type=str, default="./scans",
                        help="Base output directory for scan results")
    parser.add_argument("--synthetic", "-s", action="store_true",
                        help="Use synthetic test data instead of real cameras")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode with additional logging and diagnostics")
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create timestamped output directories
    scan_dir, captures_dir, results_dir = create_timestamped_scan_folder(args.output)
    
    # Create UnLook client
    client = UnlookClient()
    
    try:
        # Discover scanners and connect
        client.start_discovery()
        logger.info("Waiting for scanners to be discovered...")
        time.sleep(3.0)
        
        scanners = client.get_discovered_scanners()
        
        if not scanners and not args.synthetic:
            logger.error("No scanner found. Please make sure the UnLook scanner server is running.")
            logger.info("Falling back to synthetic mode for demonstration purposes.")
            args.synthetic = True
        elif not args.synthetic:
            # Connect to the first scanner
            scanner_info = scanners[0]
            logger.info(f"Connecting to scanner: {scanner_info.name} at {scanner_info.endpoint}")
            
            if not client.connect(scanner_info):
                logger.error("Failed to connect to scanner")
                return
            
            logger.info("Connected to scanner")
        
        # Set up the scanner
        structured_light_scanner = setup_scanner(client, scan_dir)
        
        # Capture structured light images (or generate synthetic ones)
        if args.synthetic:
            logger.info("Using synthetic test data")
            patterns = structured_light_scanner.generate_scan_patterns()
            left_images, right_images = generate_test_images(patterns, captures_dir)
        else:
            logger.info("Capturing structured light images")
            left_images, right_images = capture_structured_light_images(
                client, structured_light_scanner, captures_dir
            )
        
        # Process the scans
        process_scans(structured_light_scanner, left_images, right_images, results_dir)
        
        logger.info(f"3D scanning complete. Results saved to {scan_dir}")
        
    except Exception as e:
        logger.error(f"Error in enhanced 3D scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if client.connected:
            client.disconnect()
            logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()