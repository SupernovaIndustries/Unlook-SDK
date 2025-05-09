"""
Robust structured light scanning implementation for the UnLook SDK.

This module provides a complete structured light scanning implementation 
combining Gray code and Phase Shift techniques for reliable 3D reconstruction 
in various lighting conditions.
"""

import os
import time
import logging
import numpy as np
import cv2
import json
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
    logger.warning("open3d not installed. 3D mesh visualization and advanced filtering will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for o3dg when open3d is not available
    class PlaceholderO3DG:
        class PointCloud:
            pass
        class TriangleMesh:
            pass
    o3dg = PlaceholderO3DG

# Import scanner functionality from robust_scan.py
from .robust_scan import (
    StructuredLightPatternGenerator,
    StructuredLightDecoder,
    StereoTriangulator,
    RobustStructuredLightScanner as BaseRobustScanner
)


class RobustStructuredLightScanner:
    """
    Complete robust structured light scanning system combining Gray code and Phase Shift
    techniques for reliable 3D reconstruction in various lighting conditions.
    """
    
    def __init__(self, camera, projector, config, calibration_file=None, debug=False):
        """
        Initialize the robust structured light scanner.

        Args:
            camera: Stereo camera instance
            projector: Projector instance
            config: Scan configuration
            calibration_file: Path to stereo calibration file (optional)
            debug: Enable detailed debug logging (optional)
        """
        self.camera = camera
        self.projector = projector
        self.config = config
        self._debug = debug
        
        # Set up image size
        self.image_size = (1280, 720)  # Default
        if hasattr(config, 'image_size'):
            self.image_size = config.image_size
            
        # Set up projector resolution
        self.projector_width = 1920
        self.projector_height = 1080
        if hasattr(config, 'pattern_resolution'):
            self.projector_width, self.projector_height = config.pattern_resolution
            
        # Get pattern settings
        self.num_gray_codes = 10  # Default
        self.num_phase_shifts = 8  # Default
        self.phase_shift_frequencies = [1, 8, 16]  # Default
        
        if hasattr(config, 'num_gray_codes'):
            self.num_gray_codes = config.num_gray_codes
        if hasattr(config, 'num_phase_shifts'):
            self.num_phase_shifts = config.num_phase_shifts
        if hasattr(config, 'phase_shift_frequencies'):
            self.phase_shift_frequencies = config.phase_shift_frequencies
        
        # Load calibration
        if calibration_file:
            logger.info(f"Loading calibration from {calibration_file}")
            self.scanner = self._load_scanner_from_calibration(calibration_file)
        else:
            logger.info("Using default calibration")
            self.scanner = BaseRobustScanner.from_default_calibration(
                image_size=self.image_size,
                projector_width=self.projector_width,
                projector_height=self.projector_height
            )
    
    def _load_scanner_from_calibration(self, calibration_file):
        """
        Load calibration data and create scanner.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            BaseRobustScanner instance
        """
        try:
            # Try JSON format first
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
                
            # Extract calibration parameters
            camera_matrix_left = np.array(calib_data.get('camera_matrix_left', []))
            dist_coeffs_left = np.array(calib_data.get('dist_coeffs_left', []))
            camera_matrix_right = np.array(calib_data.get('camera_matrix_right', []))
            dist_coeffs_right = np.array(calib_data.get('dist_coeffs_right', []))
            R = np.array(calib_data.get('R', []))
            T = np.array(calib_data.get('T', [])).reshape(3, 1)
            
            # Check if we have all required data
            if not all([len(camera_matrix_left), len(dist_coeffs_left), 
                        len(camera_matrix_right), len(dist_coeffs_right), 
                        len(R), len(T)]):
                raise ValueError("Missing calibration parameters")
                
            # Create scanner
            return BaseRobustScanner(
                camera_matrix_left=camera_matrix_left,
                dist_coeffs_left=dist_coeffs_left,
                camera_matrix_right=camera_matrix_right,
                dist_coeffs_right=dist_coeffs_right,
                R=R,
                T=T,
                image_size=self.image_size,
                projector_width=self.projector_width,
                projector_height=self.projector_height
            )
            
        except Exception as e:
            logger.warning(f"Failed to load calibration from JSON: {e}")
            
            try:
                # Try loading using the built-in method from BaseRobustScanner
                return BaseRobustScanner.from_calibration_file(
                    calibration_file,
                    self.image_size,
                    self.projector_width,
                    self.projector_height
                )
            except Exception as e2:
                logger.error(f"Failed to load calibration: {e2}")
                
                # Fall back to default calibration
                logger.warning("Using default calibration instead")
                return BaseRobustScanner.from_default_calibration(
                    self.image_size,
                    self.projector_width,
                    self.projector_height
                )
    
    def generate_patterns(self):
        """
        Generate structured light patterns.

        Returns:
            List of pattern dictionaries
        """
        # Generate the base patterns from the scanner
        patterns = self.scanner.generate_patterns()

        # Ensure all raw_image patterns include the image data field
        for pattern in patterns:
            if pattern.get("pattern_type") == "raw_image" and "image" not in pattern:
                # Create a default image for the pattern
                if "orientation" in pattern and "frequency" in pattern:
                    # Phase shift pattern
                    height = self.projector_height
                    width = self.projector_width
                    img = np.zeros((height, width), dtype=np.uint8)

                    if pattern.get("orientation") == "horizontal":
                        freq = pattern.get("frequency", 16)
                        step = pattern.get("step", 0)
                        steps = self.num_phase_shifts
                        phase_offset = 2 * np.pi * step / steps

                        for x in range(width):
                            val = 127.5 + 127.5 * np.cos(2 * np.pi * x / freq + phase_offset)
                            img[:, x] = val
                    else:  # vertical
                        freq = pattern.get("frequency", 16)
                        step = pattern.get("step", 0)
                        steps = self.num_phase_shifts
                        phase_offset = 2 * np.pi * step / steps

                        for y in range(height):
                            val = 127.5 + 127.5 * np.cos(2 * np.pi * y / freq + phase_offset)
                            img[y, :] = val

                elif "orientation" in pattern and "bit_position" in pattern:
                    # Gray code pattern
                    height = self.projector_height
                    width = self.projector_width
                    img = np.zeros((height, width), dtype=np.uint8)

                    # Simple placeholder striped pattern
                    bit_pos = pattern.get("bit_position", 0)
                    is_inverse = pattern.get("is_inverse", False)

                    # Create stripe width based on bit position
                    stripe_width = 2 ** bit_pos

                    if pattern.get("orientation") == "horizontal":
                        for x in range(width):
                            if (x // stripe_width) % 2 == 0:
                                img[:, x] = 255 if not is_inverse else 0
                            else:
                                img[:, x] = 0 if not is_inverse else 255
                    else:  # vertical
                        for y in range(height):
                            if (y // stripe_width) % 2 == 0:
                                img[y, :] = 255 if not is_inverse else 0
                            else:
                                img[y, :] = 0 if not is_inverse else 255

                else:
                    # Default solid pattern
                    img = np.ones((self.projector_height, self.projector_width), dtype=np.uint8) * 128

                # Add the image to the pattern
                success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success:
                    pattern["image"] = encoded.tobytes()
                    if self._debug:
                        logger.debug(f"Added image data to pattern: {pattern.get('name', 'unnamed')}")

        if self._debug:
            logger.debug(f"Generated {len(patterns)} patterns with image data")

        return patterns
    
    def scan(self, output_dir=None):
        """
        Perform a complete 3D scan.
        
        Args:
            output_dir: Directory to save scan results and debug info (optional)
            
        Returns:
            PointCloud object (Open3D or numpy)
        """
        # Create output directories
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            captures_dir = os.path.join(output_dir, "captures")
            debug_dir = os.path.join(output_dir, "debug")
            results_dir = os.path.join(output_dir, "results")
            
            os.makedirs(captures_dir, exist_ok=True)
            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
        else:
            captures_dir = None
            debug_dir = None
            results_dir = None
        
        # Generate patterns
        logger.info("Generating structured light patterns...")
        patterns = self.generate_patterns()
        
        # Project patterns and capture images
        logger.info(f"Projecting {len(patterns)} patterns and capturing images...")
        left_images = []
        right_images = []
        
        for i, pattern in enumerate(patterns):
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern.get('name', f'pattern_{i}')}")
            
            # Project pattern with additional debug logging
            if self._debug:
                pattern_type = pattern.get("pattern_type", "unknown")
                pattern_name = pattern.get("name", f"pattern_{i}")
                has_image = "image" in pattern
                img_size = len(pattern.get("image", b"")) if has_image else 0
                logger.debug(f"Pattern {i+1}/{len(patterns)}: type={pattern_type}, name={pattern_name}, has_image={has_image}, image_size={img_size} bytes")

            # Ensure projector client is properly set for server communication
            if hasattr(self.projector, "client") and self.projector.client is None and not self.projector._is_simulation:
                logger.warning("Projector client not properly initialized - trying to force simulation mode")
                self.projector._is_simulation = True

            success = self.projector.project_pattern(pattern)
            if not success:
                logger.warning(f"Failed to project pattern {i+1}/{len(patterns)}: {pattern.get('name', 'unnamed')}")
                # Attempt recovery
                if "image" not in pattern and pattern.get("pattern_type") == "raw_image":
                    logger.warning("Adding missing image data and retrying projection")
                    # Add a simple image
                    img = np.ones((self.projector_height, self.projector_width), dtype=np.uint8) * 128
                    success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if success:
                        pattern["image"] = encoded.tobytes()
                        success = self.projector.project_pattern(pattern)
                        if success:
                            logger.info("Recovered from missing image data")

            # Capture stereo pair
            left_img, right_img = self.camera.capture_stereo_pair()
            
            left_images.append(left_img)
            right_images.append(right_img)
            
            # Save captured images
            if captures_dir:
                left_path = os.path.join(captures_dir, f"left_{i:03d}.png")
                right_path = os.path.join(captures_dir, f"right_{i:03d}.png")
                
                cv2.imwrite(left_path, left_img)
                cv2.imwrite(right_path, right_img)
        
        # Process scan
        logger.info("Processing scan data to generate point cloud...")
        point_cloud = self.scanner.process_scan(
            left_images,
            right_images,
            output_dir=debug_dir
        )
        
        # Save point cloud
        if results_dir and OPEN3D_AVAILABLE:
            pc_path = os.path.join(results_dir, "scan_point_cloud.ply")
            
            if isinstance(point_cloud, o3dg.PointCloud):
                o3d.io.write_point_cloud(pc_path, point_cloud)
            elif isinstance(point_cloud, np.ndarray) and len(point_cloud) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                o3d.io.write_point_cloud(pc_path, pcd)
                point_cloud = pcd
                
            logger.info(f"Saved point cloud to {pc_path}")
        
        return point_cloud
    
    def create_mesh(self, point_cloud, depth=9, smoothing=5):
        """
        Create a triangle mesh from point cloud.
        
        Args:
            point_cloud: Point cloud (Open3D PointCloud or numpy array)
            depth: Depth parameter for Poisson reconstruction
            smoothing: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        return self.scanner.create_mesh(point_cloud, depth, smoothing)
    
    def save_point_cloud(self, point_cloud, filepath):
        """
        Save point cloud to file.
        
        Args:
            point_cloud: Point cloud to save
            filepath: Output filepath
        """
        self.scanner.save_point_cloud(point_cloud, filepath)
    
    def save_mesh(self, mesh, filepath):
        """
        Save mesh to file.
        
        Args:
            mesh: Triangle mesh to save
            filepath: Output filepath
        """
        self.scanner.save_mesh(mesh, filepath)