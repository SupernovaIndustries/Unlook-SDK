"""
High-level API for simplified 3D scanning with UnLook SDK.

This module provides a simplified API for performing 3D scans with the UnLook SDK.
It abstracts the complexity of structured light scanning, pattern generation,
camera control, and point cloud processing into a simple interface.
"""

import os
import time
import logging
import numpy as np
import json
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.info("open3d not installed. 3D mesh visualization and advanced filtering will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for o3dg when open3d is not available
    class PlaceholderO3DG:
        class PointCloud:
            pass
        class TriangleMesh:
            pass
    o3dg = PlaceholderO3DG

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    logger.info("h5py not installed. HDF5 file format support will be limited.")
    H5PY_AVAILABLE = False

from .. import UnlookClient
from .structured_light import (
    StereoStructuredLightScanner,
    StereoCalibrator,
    StereoCameraParameters,
    create_scanning_demo
)


class UnlookScanner:
    """
    High-level 3D scanning API for the UnLook SDK.
    
    This class provides a simplified interface for 3D scanning, abstracting away the
    complexity of structured light scanning, pattern generation, camera control,
    and point cloud processing.
    
    Example:
        ```python
        # Create scanner with auto-detection of hardware
        scanner = UnlookScanner.auto_connect()
        
        # Perform a scan with default parameters
        point_cloud = scanner.perform_3d_scan()
        
        # Save the result
        scanner.save_scan(point_cloud, "my_scan.ply")
        
        # Optionally create and save a mesh
        mesh = scanner.create_mesh(point_cloud)
        scanner.save_mesh(mesh, "my_scan_mesh.obj")
        ```
    """
    
    def __init__(self, client: Optional[UnlookClient] = None, use_default_calibration: bool = True):
        """
        Initialize the UnlookScanner.
        
        Args:
            client: An existing UnlookClient instance, or None to create a new one
            use_default_calibration: Whether to use default calibration parameters
        """
        self.client = client or UnlookClient()
        self.is_connected = False
        self.scanner_info = None
        self.structured_light_scanner = None
        self.output_dir = os.path.join(os.getcwd(), "scans")
        self.use_default_calibration = use_default_calibration
        self.scan_folders = {}  # Map of scan_id -> folders
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    @classmethod
    def auto_connect(cls, timeout: int = 5, use_default_calibration: bool = True) -> 'UnlookScanner':
        """
        Create a scanner and automatically connect to the first available UnLook scanner.
        
        Args:
            timeout: Timeout in seconds for scanner discovery
            use_default_calibration: Whether to use default calibration parameters
            
        Returns:
            Connected UnlookScanner instance
        """
        scanner = cls(use_default_calibration=use_default_calibration)
        scanner.connect(timeout=timeout)
        return scanner
    
    def connect(self, scanner_id: Optional[str] = None, timeout: int = 5) -> bool:
        """
        Connect to an UnLook scanner.
        
        Args:
            scanner_id: Optional scanner ID to connect to. If None, connects to the first found scanner
            timeout: Timeout in seconds for scanner discovery
            
        Returns:
            True if connection was successful, False otherwise
        """
        if self.is_connected:
            logger.info("Already connected to a scanner")
            return True
            
        # Start discovery
        self.client.start_discovery()
        logger.info(f"Discovering scanners for {timeout} seconds...")
        time.sleep(timeout)
        
        # Get discovered scanners
        scanners = self.client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found")
            return False
            
        # Select scanner
        if scanner_id:
            selected_scanner = None
            for scanner in scanners:
                if scanner.uuid == scanner_id:
                    selected_scanner = scanner
                    break
                    
            if not selected_scanner:
                logger.error(f"Scanner with ID {scanner_id} not found")
                return False
        else:
            selected_scanner = scanners[0]
            
        # Connect to scanner
        logger.info(f"Connecting to scanner: {selected_scanner.name} ({selected_scanner.uuid})")
        if not self.client.connect(selected_scanner):
            logger.error("Failed to connect to scanner")
            return False
            
        self.is_connected = True
        self.scanner_info = selected_scanner
        
        # Initialize structured light scanner
        self._initialize_scanner()
        
        return True
        
    def _initialize_scanner(self) -> None:
        """
        Initialize the structured light scanner with the appropriate calibration.
        """
        # Create a calibration directory
        calib_dir = os.path.join(self.output_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)
        
        # Create a scanner with default calibration for now
        if self.use_default_calibration:
            logger.info("Using default stereo calibration parameters")
            calib_file = os.path.join(calib_dir, "default_stereo_calibration.json")
            if os.path.exists(calib_file):
                # Load existing calibration file
                try:
                    logger.info(f"Loading existing calibration from {calib_file}")
                    stereo_params = StereoCameraParameters.load(calib_file)
                    self.structured_light_scanner = StereoStructuredLightScanner(stereo_params)
                except Exception as e:
                    logger.error(f"Error loading calibration file: {e}")
                    logger.info("Creating new default calibration")
                    self.structured_light_scanner = create_scanning_demo(calib_dir)
            else:
                # Create new calibration
                logger.info("Creating new default calibration")
                self.structured_light_scanner = create_scanning_demo(calib_dir)
        else:
            # TODO: Add support for custom calibration with actual images
            logger.warning("Custom calibration not yet implemented, using default")
            self.structured_light_scanner = create_scanning_demo(calib_dir)
            
        logger.info("Structured light scanner initialized")
        
    def disconnect(self) -> None:
        """Disconnect from the scanner."""
        if self.is_connected:
            self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from scanner")
        
    def perform_3d_scan(
            self, 
            output_dir: Optional[str] = None,
            mask_threshold: int = 5,
            interval: float = 0.5,
            visualize: bool = False
        ) -> o3dg.PointCloud:
        """
        Perform a complete 3D scan using structured light patterns.
        
        Args:
            output_dir: Optional output directory for scan results
            mask_threshold: Threshold for shadow/valid pixel detection
            interval: Time interval between pattern projections in seconds
            visualize: Whether to visualize the results (requires open3d)
            
        Returns:
            3D point cloud
        """
        if not self.is_connected:
            logger.error("Not connected to a scanner")
            return o3dg.PointCloud()
            
        if not self.structured_light_scanner:
            logger.error("Structured light scanner not initialized")
            return o3dg.PointCloud()
            
        # Create a timestamped scan folder if output_dir is not specified
        if output_dir:
            scan_dir = output_dir
            os.makedirs(scan_dir, exist_ok=True)
            captures_dir = os.path.join(scan_dir, "captures")
            results_dir = os.path.join(scan_dir, "results")
            os.makedirs(captures_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        else:
            scan_dir, captures_dir, results_dir = self._create_timestamped_folder()
            
        # Store the scan folders
        scan_id = os.path.basename(scan_dir)
        self.scan_folders[scan_id] = {
            "scan_dir": scan_dir,
            "captures_dir": captures_dir,
            "results_dir": results_dir
        }
        
        logger.info(f"Starting 3D scan, saving results to {scan_dir}")
        
        # Get structured light patterns
        patterns = self.structured_light_scanner.generate_scan_patterns()
        logger.info(f"Generated {len(patterns)} structured light patterns")
        
        # Set up projector
        projector = self.client.projector
        
        # Set projector to black to start
        projector.show_solid_field("Black")
        
        # Set up cameras
        camera = self.client.camera
        cameras = camera.get_cameras()
        
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return o3dg.PointCloud()
        
        # Select the first two cameras
        left_camera_id = cameras[0]["id"]
        right_camera_id = cameras[1]["id"]
        
        # Configure cameras for optimal scanning
        try:
            camera_config = {
                "exposure": 100,
                "auto_exposure": False,
                "format": "png",
                "quality": 100
            }
            
            for cam_id in [left_camera_id, right_camera_id]:
                camera.configure(cam_id, camera_config)
        except Exception as e:
            logger.warning(f"Camera configuration failed: {e}")
            
        # Prepare arrays for images
        left_images = []
        right_images = []
        
        # Display and capture for each pattern
        for i, pattern in enumerate(patterns):
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}")
            self._display_pattern(projector, pattern, i)
            
            # Wait for projector to update
            time.sleep(interval)
            
            # Capture images
            try:
                # Capture from cameras
                logger.info(f"Capturing image pair {i+1}")
                left_img = camera.capture(left_camera_id)
                right_img = camera.capture(right_camera_id)
                
                # Save images
                left_path = os.path.join(captures_dir, f"left_{i:03d}.png")
                right_path = os.path.join(captures_dir, f"right_{i:03d}.png")
                
                cv2.imwrite(left_path, left_img)
                cv2.imwrite(right_path, right_img)
                
                # Add to arrays
                left_images.append(left_img)
                right_images.append(right_img)
                
            except Exception as e:
                logger.error(f"Error capturing images: {e}")
                continue
                
        # Reset projector to black
        projector.show_solid_field("Black")
        
        # Process the scan
        if len(left_images) < 2 or len(right_images) < 2:
            logger.error("Not enough images captured for scanning")
            return o3dg.PointCloud()
            
        logger.info(f"Processing scan with {len(left_images)} image pairs")
        pcd = self.structured_light_scanner.process_scan(left_images, right_images, mask_threshold=mask_threshold)
        
        # If the point cloud has few points, try with lower thresholds
        if len(pcd.points) < 100:
            logger.warning(f"Initial scan produced only {len(pcd.points)} points. Trying with lower thresholds...")
            pcd = self.structured_light_scanner.process_scan(left_images, right_images, mask_threshold=3)
            
        logger.info(f"Generated point cloud with {len(pcd.points)} points")
        
        # Save point cloud
        if len(pcd.points) > 0:
            point_cloud_path = os.path.join(results_dir, "scan_point_cloud.ply")
            self.structured_light_scanner.save_point_cloud(pcd, point_cloud_path)
            logger.info(f"Saved point cloud to {point_cloud_path}")
            
        # Visualize if requested
        if visualize and OPEN3D_AVAILABLE and len(pcd.points) > 0:
            try:
                logger.info("Visualizing results")
                o3d.visualization.draw_geometries([pcd], window_name="3D Scan Results")
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
                
        return pcd
    
    def _display_pattern(self, projector, pattern, idx):
        """Helper method to display a pattern on the projector."""
        try:
            pattern_name = pattern.get('name', '').lower()
            pattern_type = pattern.get('pattern_type', '')
            
            if pattern_type == "raw_image":
                # For raw images, we can't directly set them, so we approximate
                if 'white' in pattern_name:
                    projector.show_solid_field("White")
                elif 'black' in pattern_name:
                    projector.show_solid_field("Black")
                elif 'gray_code_x' in pattern_name or 'horizontal' in pattern_name:
                    # Use horizontal lines
                    width = max(1, 4 - (idx % 4))
                    projector.show_horizontal_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=width,
                        background_width=width
                    )
                elif 'gray_code_y' in pattern_name or 'vertical' in pattern_name:
                    # Use vertical lines
                    width = max(1, 4 - (idx % 4))
                    projector.show_vertical_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=width,
                        background_width=width
                    )
                else:
                    # Fallback to a checkerboard pattern
                    projector.show_checkerboard(
                        foreground_color="White",
                        background_color="Black",
                        horizontal_count=6 + (idx % 6),
                        vertical_count=4 + (idx % 4)
                    )
            else:
                # For predefined patterns, use the appropriate method
                if pattern_type == "solid_field":
                    projector.show_solid_field(pattern.get("color", "White"))
                elif pattern_type == "horizontal_lines":
                    projector.show_horizontal_lines(
                        foreground_color=pattern.get("foreground_color", "White"),
                        background_color=pattern.get("background_color", "Black"),
                        foreground_width=pattern.get("foreground_width", 4),
                        background_width=pattern.get("background_width", 20)
                    )
                elif pattern_type == "vertical_lines":
                    projector.show_vertical_lines(
                        foreground_color=pattern.get("foreground_color", "White"),
                        background_color=pattern.get("background_color", "Black"),
                        foreground_width=pattern.get("foreground_width", 4),
                        background_width=pattern.get("background_width", 20)
                    )
                else:
                    # Default to a grid for other types
                    projector.show_grid(
                        foreground_color="White",
                        background_color="Black",
                        h_foreground_width=4,
                        h_background_width=20,
                        v_foreground_width=4,
                        v_background_width=20
                    )
        except Exception as e:
            logger.error(f"Error displaying pattern: {e}")
    
    def _create_timestamped_folder(self) -> Tuple[str, str, str]:
        """
        Create a timestamped folder structure for a scan.
        
        Returns:
            Tuple of (scan_dir, captures_dir, results_dir)
        """
        # Create timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(self.output_dir, f"Scan_{timestamp}")
        
        # Create scan directory
        os.makedirs(scan_dir, exist_ok=True)
        
        # Create subdirectories
        captures_dir = os.path.join(scan_dir, "captures")
        results_dir = os.path.join(scan_dir, "results")
        
        os.makedirs(captures_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        return scan_dir, captures_dir, results_dir
    
    def create_mesh(
            self, 
            point_cloud: o3dg.PointCloud, 
            depth: int = 9, 
            smooth_iterations: int = 5
        ) -> o3dg.TriangleMesh:
        """
        Create a triangle mesh from a point cloud.
        
        Args:
            point_cloud: Input point cloud
            depth: Depth parameter for Poisson reconstruction (higher = more detail)
            smooth_iterations: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Cannot create mesh from point cloud")
            return o3dg.TriangleMesh()
            
        return self.structured_light_scanner.create_mesh_from_point_cloud(
            point_cloud, 
            depth=depth, 
            smooth_iterations=smooth_iterations
        )
    
    def save_scan(self, point_cloud: o3dg.PointCloud, filepath: str) -> None:
        """
        Save a point cloud to a file.
        
        Args:
            point_cloud: Point cloud to save
            filepath: Output file path (supported formats: .ply, .pcd, .obj)
        """
        return self.structured_light_scanner.save_point_cloud(point_cloud, filepath)
    
    def save_mesh(self, mesh: o3dg.TriangleMesh, filepath: str) -> None:
        """
        Save a triangle mesh to a file.
        
        Args:
            mesh: Triangle mesh to save
            filepath: Output file path (supported formats: .ply, .obj, .off, .gltf)
        """
        return self.structured_light_scanner.save_mesh(mesh, filepath)
    
    def get_latest_scan_id(self) -> Optional[str]:
        """
        Get the ID of the most recent scan.
        
        Returns:
            Scan ID or None if no scans have been performed
        """
        if not self.scan_folders:
            return None
            
        return list(self.scan_folders.keys())[-1]
    
    def get_scan_folders(self, scan_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get the folder paths for a scan.
        
        Args:
            scan_id: Scan ID, or None to use the most recent scan
            
        Returns:
            Dictionary of folder paths (scan_dir, captures_dir, results_dir)
        """
        if not scan_id:
            scan_id = self.get_latest_scan_id()
            
        if not scan_id or scan_id not in self.scan_folders:
            return {}
            
        return self.scan_folders[scan_id]
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the base output directory for all scans.
        
        Args:
            output_dir: Path to the base output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Set output directory to {output_dir}")

    def visualize_point_cloud(self, point_cloud: o3dg.PointCloud) -> None:
        """
        Visualize a point cloud (requires open3d).
        
        Args:
            point_cloud: Point cloud to visualize
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available. Cannot visualize point cloud")
            return
            
        if len(point_cloud.points) == 0:
            logger.warning("Point cloud is empty, nothing to visualize")
            return
            
        try:
            o3d.visualization.draw_geometries([point_cloud], window_name="UnLook 3D Scan")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically disconnect from scanner."""
        self.disconnect()