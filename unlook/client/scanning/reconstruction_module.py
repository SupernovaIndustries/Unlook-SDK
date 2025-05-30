"""
Reconstruction Module for Offline 3D Processing.

This module processes previously captured images to generate 3D point clouds.
It loads captured data from disk and performs pattern decoding, correspondence
matching, and triangulation without requiring a live scanner connection.
"""

import os
import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import reconstruction components - use the existing modules
from .reconstruction import (
    Integrated3DPipeline,
    ScanningResult,
    ImprovedCorrespondenceMatcher,
    Triangulator
)
RECONSTRUCTION_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction processing."""
    enable_uncertainty: bool = True
    enable_filtering: bool = True
    enable_optimization: bool = True
    save_intermediate: bool = False
    output_format: str = "ply"  # ply, obj, or npy
    downsample_voxel_size: Optional[float] = None
    statistical_outlier_neighbors: int = 20
    statistical_outlier_std_ratio: float = 2.0


class ReconstructionModule:
    """
    Processes captured images offline to generate 3D reconstructions.
    
    This module loads previously captured pattern sequences and
    performs all 3D reconstruction steps without requiring a
    live connection to the scanner.
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize the reconstruction module.
        
        Args:
            config: Reconstruction configuration options
        """
        self.config = config or ReconstructionConfig()
        self.pipeline = None
        self.session_metadata = None
        self.calibration_data = None
        
    def load_session(self, session_path: str) -> bool:
        """
        Load a captured session for processing.
        
        Args:
            session_path: Path to session directory
            
        Returns:
            True if session loaded successfully
        """
        session_dir = Path(session_path)
        if not session_dir.exists():
            logger.error(f"Session directory not found: {session_path}")
            return False
        
        # Load metadata
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                self.session_metadata = json.load(f)
            logger.info(f"Loaded session metadata from {metadata_path}")
            
            # Load calibration
            calib_file = self.session_metadata['capture'].get('calibration_file')
            if calib_file:
                calib_path = session_dir / calib_file
                if calib_path.exists():
                    with open(calib_path, 'r') as f:
                        self.calibration_data = json.load(f)
                    logger.info(f"Loaded calibration from {calib_path}")
                else:
                    logger.warning(f"Calibration file not found: {calib_path}")
                    # Try default calibration
                    self._load_default_calibration()
            else:
                logger.warning("No calibration file specified in metadata")
                self._load_default_calibration()
            
            # Initialize pipeline
            if RECONSTRUCTION_AVAILABLE and self.calibration_data:
                self.pipeline = Integrated3DPipeline(self.calibration_data)
                self.pipeline.enable_uncertainty_quantification = self.config.enable_uncertainty
                logger.info("Initialized reconstruction pipeline")
            else:
                logger.warning("Reconstruction pipeline not available or no calibration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def _load_default_calibration(self):
        """Try to load default calibration file."""
        default_paths = [
            Path("unlook/calibration/custom/stereo_calibration_fixed.json"),
            Path("unlook/calibration/default/default_stereo.json")
        ]
        
        for path in default_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        self.calibration_data = json.load(f)
                    logger.info(f"Loaded default calibration from {path}")
                    return
                except Exception as e:
                    logger.error(f"Error loading calibration from {path}: {e}")
        
        logger.error("No calibration file found")
    
    def process_captured_data(self, session_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all captured data in a session.
        
        Args:
            session_path: Path to captured session
            output_dir: Optional output directory (defaults to session_path/reconstruction)
            
        Returns:
            Dictionary with processing results
        """
        # Load session
        if not self.load_session(session_path):
            return {"success": False, "error": "Failed to load session"}
        
        session_dir = Path(session_path)
        if output_dir is None:
            output_dir = session_dir / "reconstruction"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get pattern type from metadata
        pattern_info = self.session_metadata['capture']['pattern_info']
        pattern_types = pattern_info.get('pattern_types', [])
        
        results = {
            "success": False,
            "session_path": str(session_path),
            "output_dir": str(output_dir),
            "pattern_types": pattern_types,
            "processing_results": {}
        }
        
        # Process based on pattern type
        if 'gray_code' in pattern_types:
            logger.info("Processing Gray code patterns...")
            gray_result = self._process_gray_code_patterns(session_dir, output_dir)
            results['processing_results']['gray_code'] = gray_result
            results['success'] = gray_result.get('success', False)
            
        if 'phase_shift' in pattern_types:
            logger.info("Processing phase shift patterns...")
            phase_result = self._process_phase_shift_patterns(session_dir, output_dir)
            results['processing_results']['phase_shift'] = phase_result
            
        # Save processing results
        self._save_results(output_dir, results)
        
        return results
    
    def _process_gray_code_patterns(self, session_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process Gray code pattern sequence."""
        start_time = time.time()
        
        result = {
            "success": False,
            "num_points": 0,
            "processing_time": 0,
            "error": None
        }
        
        try:
            # Load captured images
            captured_images = self.session_metadata['capture']['captured_images']
            left_images = []
            right_images = []
            
            logger.info(f"Loading {len(captured_images)} captured image pairs...")
            
            for img_info in captured_images:
                # Load left image
                left_path = session_dir / img_info['left_image']
                left_img = cv2.imread(str(left_path))
                if left_img is None:
                    logger.error(f"Failed to load left image: {left_path}")
                    continue
                left_images.append(left_img)
                
                # Load right image
                right_path = session_dir / img_info['right_image']
                right_img = cv2.imread(str(right_path))
                if right_img is None:
                    logger.error(f"Failed to load right image: {right_path}")
                    continue
                right_images.append(right_img)
            
            logger.info(f"Loaded {len(left_images)} image pairs")
            
            if len(left_images) < 4:  # Need at least reference + some patterns
                raise ValueError("Not enough images for Gray code processing")
            
            # Process with pipeline
            if self.pipeline:
                # Get pattern info
                pattern_info = {
                    'num_bits': self.session_metadata['capture']['pattern_info'].get('gray_code_bits', 8),
                    'orientation': 'vertical',
                    'color': 'blue' if self.session_metadata['capture']['pattern_info'].get('uses_blue_channel', False) else 'white'
                }
                
                logger.info("Starting Gray code reconstruction...")
                scan_result = self.pipeline.process_gray_code_scan(
                    left_images, right_images,
                    pattern_info=pattern_info
                )
                
                if scan_result.success and scan_result.points_3d is not None:
                    points_3d = scan_result.points_3d
                    result['num_points'] = len(points_3d)
                    
                    # Save point cloud
                    self._save_point_cloud(points_3d, output_dir / "point_cloud", self.config.output_format)
                    
                    # Apply filtering if enabled
                    if self.config.enable_filtering and len(points_3d) > 0:
                        filtered_points = self._filter_point_cloud(points_3d)
                        self._save_point_cloud(filtered_points, output_dir / "point_cloud_filtered", self.config.output_format)
                        result['num_points_filtered'] = len(filtered_points)
                    
                    result['success'] = True
                    logger.info(f"✅ Reconstruction successful: {result['num_points']} points")
                else:
                    result['error'] = scan_result.error_message if hasattr(scan_result, 'error_message') else "Unknown error"
                    logger.error(f"❌ Reconstruction failed: {result['error']}")
            else:
                result['error'] = "Pipeline not available"
                logger.error("Pipeline not initialized")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing Gray code patterns: {e}")
            import traceback
            traceback.print_exc()
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _process_phase_shift_patterns(self, session_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process phase shift pattern sequence."""
        # TODO: Implement phase shift processing
        logger.warning("Phase shift processing not yet implemented")
        return {
            "success": False,
            "error": "Not implemented"
        }
    
    def _save_point_cloud(self, points: np.ndarray, filename: Path, format: str):
        """Save point cloud in specified format."""
        try:
            if format == 'npy':
                np.save(f"{filename}.npy", points)
                logger.info(f"Saved {len(points)} points to {filename}.npy")
                
            elif format == 'ply':
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    o3d.io.write_point_cloud(f"{filename}.ply", pcd)
                    logger.info(f"Saved {len(points)} points to {filename}.ply")
                except ImportError:
                    logger.warning("Open3D not available, saving as numpy instead")
                    np.save(f"{filename}.npy", points)
                    
            elif format == 'obj':
                # Simple OBJ format (vertices only)
                with open(f"{filename}.obj", 'w') as f:
                    for point in points:
                        f.write(f"v {point[0]} {point[1]} {point[2]}\n")
                logger.info(f"Saved {len(points)} points to {filename}.obj")
                
        except Exception as e:
            logger.error(f"Error saving point cloud: {e}")
    
    def _filter_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Apply filtering to point cloud."""
        try:
            import open3d as o3d
            
            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Statistical outlier removal
            if self.config.statistical_outlier_neighbors > 0:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=self.config.statistical_outlier_neighbors,
                    std_ratio=self.config.statistical_outlier_std_ratio
                )
                logger.info(f"Statistical outlier removal: {len(points)} → {len(pcd.points)} points")
            
            # Voxel downsampling
            if self.config.downsample_voxel_size:
                pcd = pcd.voxel_down_sample(voxel_size=self.config.downsample_voxel_size)
                logger.info(f"Voxel downsampling: → {len(pcd.points)} points")
            
            return np.asarray(pcd.points)
            
        except ImportError:
            logger.warning("Open3D not available for filtering")
            return points
    
    def _save_results(self, output_dir: Path, results: Dict[str, Any]):
        """Save processing results to JSON."""
        results_path = output_dir / "processing_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved processing results to {results_path}")
    
    def process_quick(self, session_path: str) -> Tuple[bool, int]:
        """
        Quick processing method that returns success and number of points.
        
        Args:
            session_path: Path to captured session
            
        Returns:
            Tuple of (success, num_points)
        """
        results = self.process_captured_data(session_path)
        
        if results['success']:
            gray_result = results['processing_results'].get('gray_code', {})
            num_points = gray_result.get('num_points', 0)
            return True, num_points
        else:
            return False, 0