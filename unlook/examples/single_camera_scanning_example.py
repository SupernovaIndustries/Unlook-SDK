#!/usr/bin/env python3
"""
Example of 3D scanning using a single camera and projector setup.

This example demonstrates how to calibrate a single camera and projector system
and perform 3D scanning using structured light patterns.

Usage:
    python single_camera_scanning_example.py [--calibrate] [--pattern gray_code|phase_shift|combined] [--quality low|medium|high]
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import Open3D for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Install with 'pip install open3d' for visualization.")
    OPEN3D_AVAILABLE = False

# Import our single_camera_scanner module
try:
    from unlook import UnlookClient
    from unlook.client.single_camera_scanner import SingleCameraCalibrator, SingleCameraStructuredLight
except ImportError as e:
    logger.error(f"Error importing UnLook modules: {e}")
    print("Make sure you're running this script from the UnLook-SDK directory structure.")
    sys.exit(1)


def check_directories():
    """Ensure necessary directories exist."""
    os.makedirs("./scans", exist_ok=True)
    os.makedirs("./calibration", exist_ok=True)


def calibrate_system(client: UnlookClient):
    """
    Calibrate the camera-projector system.
    
    Args:
        client: Connected UnlookClient
    """
    print("\n===== Calibrating Camera-Projector System =====")
    print("This process requires a checkerboard pattern.")
    print("You will be prompted to position the checkerboard in different orientations.")
    
    # Create calibrator
    calibrator = SingleCameraCalibrator(checkerboard_size=(9, 6), checkerboard_square_size=25.0)
    
    # Get camera and projector
    camera = client.camera
    projector = client.projector
    
    # Get available cameras
    cameras = camera.get_cameras()
    if not cameras:
        print("Error: No cameras found.")
        return False
    
    # Use the first camera
    cam_id = cameras[0]["id"]
    print(f"Using camera: {cameras[0]['name']} ({cam_id})")
    
    # Configure camera for calibration
    try:
        camera_config = {
            "exposure": 100,
            "auto_exposure": False,
            "format": "png",
            "quality": 100
        }
        camera.configure(cam_id, camera_config)
    except Exception as e:
        print(f"Warning: Camera configuration failed: {e}")
    
    # Step 1: Camera Calibration
    print("\nStep 1: Camera Calibration")
    print("We will capture multiple images of the checkerboard for camera calibration.")
    print("Position the checkerboard at different angles and distances.")
    
    camera_images = []
    num_images = 10
    
    # Show white screen for better checkerboard visibility
    projector.show_solid_field("White")
    
    for i in range(num_images):
        input(f"Press Enter to capture image {i+1}/{num_images}...")
        
        # Capture image
        img = camera.capture(cam_id)
        
        # Save image
        cv2.imwrite(f"./calibration/camera_calib_{i:02d}.png", img)
        camera_images.append(img)
        
        print(f"Captured image {i+1}/{num_images}")
    
    # Calibrate camera
    print("\nCalibrating camera...")
    camera_success = calibrator.calibrate_camera(camera_images)
    
    if not camera_success:
        print("Error: Camera calibration failed. Please try again.")
        return False
    
    print("Camera calibration successful!")
    
    # Step 2: Projector Calibration
    print("\nStep 2: Projector Calibration")
    print("We will project patterns onto the checkerboard for projector calibration.")
    print("Keep the checkerboard in a fixed position for each image pair.")
    
    # For actual projector calibration, you'd use Gray code or other structured light patterns
    # Here we'll use a simplified approach with a checkerboard pattern
    
    projector_width = 1920
    projector_height = 1080
    
    calib_images = []
    proj_patterns = []
    num_patterns = 5
    
    for i in range(num_patterns):
        # Generate a calibration pattern
        pattern = calibrator.generate_calibration_pattern(projector_width, projector_height)
        
        # Display the pattern
        # In a real implementation, you'd use a more sophisticated method to display patterns
        # For example, show_raw_image() if available
        projector.show_checkerboard(
            foreground_color="White",
            background_color="Black",
            horizontal_count=8 + i,
            vertical_count=6 + i
        )
        
        # Wait for projector to update
        time.sleep(1.0)
        
        # Capture image
        img = camera.capture(cam_id)
        
        # Save image pair
        cv2.imwrite(f"./calibration/proj_calib_img_{i:02d}.png", img)
        cv2.imwrite(f"./calibration/proj_pattern_{i:02d}.png", pattern)
        
        calib_images.append(img)
        proj_patterns.append(pattern)
        
        print(f"Captured calibration pair {i+1}/{num_patterns}")
        
        input(f"Press Enter to continue to the next pattern...")
    
    # Calibrate projector
    print("\nCalibrating projector...")
    projector_success = calibrator.calibrate_projector(calib_images, proj_patterns)
    
    if not projector_success:
        print("Error: Projector calibration failed. Please try again.")
        return False
    
    print("Projector calibration successful!")
    
    # Save calibration data
    calib_file = "./calibration/camera_projector_calibration.npz"
    calibrator.save_calibration(calib_file)
    print(f"Calibration data saved to {calib_file}")
    
    # Reset projector
    projector.show_solid_field("Black")
    
    return True


def perform_scan(client: UnlookClient, pattern_type: str = "gray_code", 
                quality: str = "medium", visualize: bool = True):
    """
    Perform a 3D scan using structured light.
    
    Args:
        client: Connected UnlookClient
        pattern_type: Type of structured light pattern to use
        quality: Quality setting for the scan
        visualize: Whether to visualize the result
    """
    print(f"\n===== Performing 3D Scan with {pattern_type} patterns ({quality} quality) =====")
    
    # Load calibration data
    calib_file = "./calibration/camera_projector_calibration.npz"
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file not found: {calib_file}")
        print("Please run calibration first with --calibrate")
        return
    
    # Create scanner
    scanner = SingleCameraStructuredLight.from_calibration_file(
        calib_file, 
        projector_width=1920, 
        projector_height=1080
    )
    
    # Get camera and projector
    camera = client.camera
    projector = client.projector
    
    # Get available cameras
    cameras = camera.get_cameras()
    if not cameras:
        print("Error: No cameras found.")
        return
    
    # Use the first camera
    cam_id = cameras[0]["id"]
    print(f"Using camera: {cameras[0]['name']} ({cam_id})")
    
    # Create a timestamped directory for this scan
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scan_dir = os.path.join("./scans", f"single_camera_scan_{timestamp}_{pattern_type}_{quality}")
    os.makedirs(scan_dir, exist_ok=True)
    
    # Create subdirectories
    captures_dir = os.path.join(scan_dir, "captures")
    results_dir = os.path.join(scan_dir, "results")
    debug_dir = os.path.join(scan_dir, "debug")
    
    os.makedirs(captures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Generate structured light patterns
    if pattern_type == "phase_shift":
        patterns = scanner.generate_phase_shift_patterns()
    elif pattern_type == "combined":
        patterns = scanner.generate_gray_code_patterns() + scanner.generate_phase_shift_patterns()
    else:  # default to gray_code
        patterns = scanner.generate_gray_code_patterns()
    
    # Adjust patterns based on quality setting
    if quality == "low":
        # Use fewer patterns for faster scanning
        if pattern_type == "combined":
            # Just use Gray code
            patterns = scanner.generate_gray_code_patterns()
        elif pattern_type == "phase_shift":
            # Use fewer frequencies
            patterns = scanner.generate_phase_shift_patterns(
                frequencies=[16]
            )
        else:
            # Skip some patterns
            patterns = patterns[::2]
            
    elif quality == "high":
        # Use more patterns for higher quality
        if pattern_type == "phase_shift":
            # Use more frequencies
            patterns = scanner.generate_phase_shift_patterns(
                frequencies=[8, 16, 32, 64]
            )
    
    print(f"Generated {len(patterns)} structured light patterns")
    
    # Capture images
    captured_images = []
    
    # Set projector to black to start
    projector.show_solid_field("Black")
    time.sleep(0.5)
    
    # Configure camera
    try:
        camera_config = {
            "exposure": 100,
            "auto_exposure": False,
            "format": "png",
            "quality": 100
        }
        camera.configure(cam_id, camera_config)
    except Exception as e:
        print(f"Warning: Camera configuration failed: {e}")
    
    # Function to display pattern
    def display_pattern(pattern, idx):
        """Helper to display a pattern."""
        pattern_name = pattern.get('name', '').lower()
        pattern_type = pattern.get('pattern_type', '')
        
        if pattern_type == "raw_image":
            # For raw images with binary data, try to use direct image display if available
            if 'image' in pattern and hasattr(projector, 'show_raw_image'):
                try:
                    projector.show_raw_image(pattern['image'])
                    return
                except Exception:
                    pass  # Fall back to approximation
            
            # Approximate with built-in patterns
            if 'white' in pattern_name:
                projector.show_solid_field("White")
            elif 'black' in pattern_name:
                projector.show_solid_field("Black")
            elif 'gray_code_x' in pattern_name or 'horizontal' in pattern_name:
                # For horizontal Gray code patterns
                projector.show_horizontal_lines(
                    foreground_color="White",
                    background_color="Black",
                    foreground_width=max(1, 4 - (idx % 4)),
                    background_width=max(1, 4 - (idx % 4))
                )
            elif 'gray_code_y' in pattern_name or 'vertical' in pattern_name:
                # For vertical Gray code patterns
                projector.show_vertical_lines(
                    foreground_color="White",
                    background_color="Black",
                    foreground_width=max(1, 4 - (idx % 4)),
                    background_width=max(1, 4 - (idx % 4))
                )
            elif 'phase' in pattern_name:
                # For phase patterns
                if 'h_phase' in pattern_name:
                    # Horizontal sinusoidal pattern
                    projector.show_horizontal_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=2,
                        background_width=2
                    )
                else:
                    # Vertical sinusoidal pattern
                    projector.show_vertical_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=2,
                        background_width=2
                    )
            else:
                # Fallback
                projector.show_checkerboard(
                    foreground_color="White",
                    background_color="Black",
                    horizontal_count=8,
                    vertical_count=6
                )
    
    # Display patterns and capture images
    print("\nCapturing structured light patterns...")
    print("Please ensure the object is positioned in the scan area.")
    input("Press Enter to begin capturing...")
    
    for i, pattern in enumerate(patterns):
        print(f"Projecting pattern {i+1}/{len(patterns)}")
        
        # Display pattern
        display_pattern(pattern, i)
        
        # Wait for projector to update and for image to stabilize
        time.sleep(0.5)
        
        # Capture image
        img = camera.capture(cam_id)
        
        # Save image
        img_path = os.path.join(captures_dir, f"capture_{i:03d}.png")
        cv2.imwrite(img_path, img)
        
        # Add to list
        captured_images.append(img)
    
    # Reset projector
    projector.show_solid_field("Black")
    
    print(f"Captured {len(captured_images)} images")
    
    # Process scan
    print("\nProcessing scan data...")
    
    use_gray_code = pattern_type in ["gray_code", "combined"]
    use_phase_shift = pattern_type in ["phase_shift", "combined"]
    
    # Set mask threshold based on quality
    if quality == "low":
        mask_threshold = 5
    elif quality == "medium":
        mask_threshold = 10
    else:  # high
        mask_threshold = 15
    
    # Process the scan
    point_cloud = scanner.process_scan(
        captured_images,
        use_gray_code=use_gray_code,
        use_phase_shift=use_phase_shift,
        mask_threshold=mask_threshold,
        output_dir=debug_dir
    )
    
    if len(point_cloud.points) == 0:
        print("Error: No points generated in the scan.")
        return
    
    print(f"Generated point cloud with {len(point_cloud.points)} points")
    
    # Save point cloud
    pcd_path = os.path.join(results_dir, "scan_point_cloud.ply")
    scanner.save_point_cloud(point_cloud, pcd_path)
    print(f"Saved point cloud to {pcd_path}")
    
    # Create mesh if we have enough points
    if len(point_cloud.points) >= 100:
        print("\nCreating mesh from point cloud...")
        
        # Adjust mesh parameters based on quality
        if quality == "low":
            depth = 6
            smoothing = 1
        elif quality == "medium":
            depth = 8
            smoothing = 2
        else:  # high
            depth = 9
            smoothing = 3
        
        mesh = scanner.create_mesh(
            point_cloud,
            depth=depth,
            smoothing=smoothing
        )
        
        # Save mesh
        if len(mesh.triangles) > 0:
            mesh_path = os.path.join(results_dir, "scan_mesh.obj")
            scanner.save_mesh(mesh, mesh_path)
            print(f"Saved mesh with {len(mesh.triangles)} triangles to {mesh_path}")
            
            # Visualize mesh if requested
            if visualize and OPEN3D_AVAILABLE:
                print("\nVisualizing mesh...")
                o3d.visualization.draw_geometries([mesh], window_name="3D Scan Mesh")
        else:
            print("Warning: Could not create mesh from point cloud")
    
    # Visualize point cloud if requested
    if visualize and OPEN3D_AVAILABLE:
        print("\nVisualizing point cloud...")
        o3d.visualization.draw_geometries([point_cloud], window_name="3D Scan Point Cloud")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Single Camera Structured Light 3D Scanning")
    parser.add_argument("--calibrate", "-c", action="store_true", 
                       help="Run camera-projector calibration")
    parser.add_argument("--pattern", "-p", type=str, default="gray_code",
                       choices=["gray_code", "phase_shift", "combined"],
                       help="Type of structured light pattern to use")
    parser.add_argument("--quality", "-q", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Quality setting for the scan")
    parser.add_argument("--no-visualize", "-n", action="store_true",
                       help="Skip visualization of results")
    args = parser.parse_args()
    
    # Check for Open3D
    if not OPEN3D_AVAILABLE and not args.no_visualize:
        print("Warning: open3d not installed. Visualization will be disabled.")
        print("Install with: pip install open3d")
    
    # Ensure directories exist
    check_directories()
    
    # Connect to UnLook client
    try:
        client = UnlookClient()
    except Exception as e:
        print(f"Error connecting to UnLook client: {e}")
        return
    
    try:
        # Start client discovery
        client.start_discovery()
        print("Discovering UnLook scanners...")
        time.sleep(3)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            print("No scanners found. Make sure your scanner is connected and powered on.")
            return
        
        # Connect to first scanner
        scanner = scanners[0]
        print(f"Connecting to scanner: {scanner.name} ({scanner.uuid})")
        
        if not client.connect(scanner):
            print("Failed to connect to scanner")
            return
        
        print(f"Connected to scanner: {scanner.name}")
        
        # Run requested operation
        if args.calibrate:
            calibrate_system(client)
        else:
            perform_scan(
                client, 
                pattern_type=args.pattern, 
                quality=args.quality, 
                visualize=not args.no_visualize
            )
            
        # Disconnect
        client.disconnect()
        print("Disconnected from scanner")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        if client:
            client.disconnect()
            print("Disconnected from scanner")
    except Exception as e:
        print(f"Error during operation: {e}")
        if client:
            client.disconnect()
            print("Disconnected from scanner")


if __name__ == "__main__":
    main()