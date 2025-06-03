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
from datetime import datetime
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
    save_debug_steps: bool = True  # For ISO/ASTM 52902 compliance
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
                
                # Verify rectification is working
                if hasattr(self.pipeline, 'rectifier') and self.pipeline.rectifier.is_initialized:
                    logger.info(f"[OK] Rectification initialized: {self.pipeline.rectifier.image_size}")
                else:
                    logger.warning("❌ Rectification NOT initialized")
                
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
            Path("unlook/calibration/custom/corrected_stereo.json"),  # Use corrected calibration first!
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
            # Update overall success if phase shift succeeded
            if phase_result.get('success', False):
                results['success'] = True
            
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
            
            # Debug step: Save loaded images info
            self._save_debug_step(session_dir, "01_loaded_images", {
                'num_left_images': len(left_images),
                'num_right_images': len(right_images),
                'image_shapes': [img.shape for img in left_images[:3]]  # First 3 shapes
            }, f"Loaded {len(left_images)} image pairs for Gray code processing")
            
            if len(left_images) < 4:  # Need at least reference + some patterns
                raise ValueError("Not enough images for Gray code processing")
            
            # Debug step: Save reference images and analyze  
            if len(left_images) >= 2:
                self._save_debug_step(session_dir, "02_reference_left_white", left_images[0], "Reference white pattern - left camera")
                self._save_debug_step(session_dir, "02_reference_right_white", right_images[0], "Reference white pattern - right camera")
                self._save_debug_step(session_dir, "03_reference_left_black", left_images[1], "Reference black pattern - left camera")
                self._save_debug_step(session_dir, "03_reference_right_black", right_images[1], "Reference black pattern - right camera")
                
                # Additional analysis for correspondence debugging
                left_white_stats = {
                    'mean': float(left_images[0].mean()),
                    'std': float(left_images[0].std()),
                    'min': int(left_images[0].min()),
                    'max': int(left_images[0].max())
                }
                right_white_stats = {
                    'mean': float(right_images[0].mean()),
                    'std': float(right_images[0].std()),
                    'min': int(right_images[0].min()),
                    'max': int(right_images[0].max())
                }
                
                self._save_debug_step(session_dir, "03a_image_statistics", {
                    'left_white': left_white_stats,
                    'right_white': right_white_stats,
                    'brightness_diff': abs(left_white_stats['mean'] - right_white_stats['mean']),
                    'possible_sync_issue': abs(left_white_stats['mean'] - right_white_stats['mean']) > 50
                }, "Image statistics to detect sync/exposure issues")
                
                # Focus analysis from capture metadata if available
                focus_stats = self.session_metadata['capture'].get('focus_statistics', {})
                if focus_stats:
                    self._save_debug_step(session_dir, "03b_focus_analysis", focus_stats, "Focus quality analysis from capture session")
            
            # Process with pipeline
            if self.pipeline:
                # Get pattern info
                pattern_info = {
                    'num_bits': self.session_metadata['capture']['pattern_info'].get('gray_code_bits', 8),
                    'orientation': 'vertical',
                    'color': 'blue' if self.session_metadata['capture']['pattern_info'].get('uses_blue_channel', False) else 'white'
                }
                
                # Debug step: Save pattern info
                self._save_debug_step(session_dir, "04_pattern_info", pattern_info, "Pattern configuration for Gray code reconstruction")
                
                logger.info("Starting Gray code reconstruction...")
                scan_result = self.pipeline.process_gray_code_scan(
                    left_images, right_images,
                    pattern_info=pattern_info
                )
                
                if scan_result.success and scan_result.points_3d is not None:
                    points_3d = scan_result.points_3d
                    result['num_points'] = len(points_3d)
                    
                    # Debug step: Save reconstruction results
                    self._save_debug_step(session_dir, "05_reconstruction_success", {
                        'num_points': len(points_3d),
                        'points_shape': points_3d.shape,
                        'points_range_x': [float(points_3d[:, 0].min()), float(points_3d[:, 0].max())],
                        'points_range_y': [float(points_3d[:, 1].min()), float(points_3d[:, 1].max())],
                        'points_range_z': [float(points_3d[:, 2].min()), float(points_3d[:, 2].max())]
                    }, f"Reconstruction successful: {len(points_3d)} points generated")
                    
                    # Debug step: Save raw point cloud
                    self._save_debug_step(session_dir, "06_raw_point_cloud", points_3d, "Raw 3D point cloud before filtering")
                    
                    # Save point cloud
                    self._save_point_cloud(points_3d, output_dir / "point_cloud", self.config.output_format)
                    
                    # Apply filtering if enabled
                    if self.config.enable_filtering and len(points_3d) > 0:
                        filtered_points = self._filter_point_cloud(points_3d)
                        self._save_point_cloud(filtered_points, output_dir / "point_cloud_filtered", self.config.output_format)
                        result['num_points_filtered'] = len(filtered_points)
                        
                        # Debug step: Save filtered point cloud
                        self._save_debug_step(session_dir, "07_filtered_point_cloud", filtered_points, f"Filtered point cloud: {len(points_3d)} → {len(filtered_points)} points")
                    
                    result['success'] = True
                    logger.info(f"[SUCCESS] Reconstruction successful: {result['num_points']} points")
                else:
                    # Debug step: Save error details
                    error_msg = scan_result.error_message if hasattr(scan_result, 'error_message') else "Unknown error"
                    self._save_debug_step(session_dir, "05_reconstruction_failed", {
                        'error': error_msg,
                        'scan_result_success': scan_result.success,
                        'has_points': scan_result.points_3d is not None if hasattr(scan_result, 'points_3d') else False
                    }, f"Reconstruction failed: {error_msg}")
                    
                    result['error'] = error_msg
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
            
            logger.info(f"Loading {len(captured_images)} captured image pairs for phase shift...")
            
            # For demo speed: Use key patterns but include more for better reconstruction
            demo_mode = True  # Set to True for demo speed
            if demo_mode:
                # Use references + low and medium frequency patterns (skip highest frequency)
                selected_images = captured_images[:10]  # First 10 images (references + 2 frequencies)
                logger.info(f"DEMO MODE: Using {len(selected_images)} patterns (balanced speed/quality)")
                captured_images = selected_images
            
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
                raise ValueError("Not enough images for phase shift processing")
            
            # Apply stereo rectification to fix point cloud distortions
            if (self.pipeline and hasattr(self.pipeline, 'rectifier') and 
                self.pipeline.enable_rectification and self.pipeline.rectifier.is_initialized):
                logger.info("[OK] Applying stereo rectification to images...")
                logger.info(f"Original image size: {left_images[0].shape} vs calibration: {self.pipeline.rectifier.image_size}")
                
                rectified_pairs = []
                for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
                    try:
                        rect_left, rect_right = self.pipeline.rectifier.rectify_images(
                            left_img, right_img, use_opencl=True
                        )
                        rectified_pairs.append((rect_left, rect_right))
                        
                        if (i + 1) % 5 == 0:
                            logger.info(f"Rectified {i + 1}/{len(left_images)} image pairs")
                            
                    except Exception as e:
                        logger.error(f"Failed to rectify image pair {i}: {e}")
                        rectified_pairs.append((left_img, right_img))  # Fallback to original
                
                # Update image lists with rectified images
                left_images = [pair[0] for pair in rectified_pairs]
                right_images = [pair[1] for pair in rectified_pairs]
                
                logger.info("Stereo rectification completed")
                
                # Save rectified images for debug
                if self.config.save_debug_steps:
                    debug_dir = session_dir / "debug_rectification"
                    debug_dir.mkdir(exist_ok=True)
                    
                    logger.info("Saving rectified images for debug...")
                    for i, (orig_left, orig_right, rect_left, rect_right) in enumerate(zip(
                        [cv2.imread(str(session_dir / img_info['left_image'])) for img_info in captured_images],
                        [cv2.imread(str(session_dir / img_info['right_image'])) for img_info in captured_images],
                        left_images, right_images
                    )):
                        # Save original and rectified side by side
                        if i < 5:  # Save first 5 pairs only
                            try:
                                cv2.imwrite(str(debug_dir / f"original_left_{i:02d}.jpg"), orig_left)
                                cv2.imwrite(str(debug_dir / f"original_right_{i:02d}.jpg"), orig_right)
                                cv2.imwrite(str(debug_dir / f"rectified_left_{i:02d}.jpg"), rect_left)
                                cv2.imwrite(str(debug_dir / f"rectified_right_{i:02d}.jpg"), rect_right)
                                
                                # Create comparison images - resize to match dimensions
                                h_orig, w_orig = orig_left.shape[:2]
                                h_rect, w_rect = rect_left.shape[:2]
                                
                                if h_orig != h_rect or w_orig != w_rect:
                                    # Resize rectified to match original for comparison
                                    rect_left_resized = cv2.resize(rect_left, (w_orig, h_orig))
                                    rect_right_resized = cv2.resize(rect_right, (w_orig, h_orig))
                                    comparison_left = np.hstack([orig_left, rect_left_resized])
                                    comparison_right = np.hstack([orig_right, rect_right_resized])
                                else:
                                    comparison_left = np.hstack([orig_left, rect_left])
                                    comparison_right = np.hstack([orig_right, rect_right])
                                    
                                cv2.imwrite(str(debug_dir / f"comparison_left_{i:02d}.jpg"), comparison_left)
                                cv2.imwrite(str(debug_dir / f"comparison_right_{i:02d}.jpg"), comparison_right)
                            except Exception as e:
                                logger.error(f"Failed to save debug image {i}: {e}")
                    
                    logger.info(f"Debug images saved to {debug_dir}")
                    
                    # Save enhanced debug visualizations
                    try:
                        logger.info("Generating enhanced debug visualizations...")
                        
                        # Generate disparity map for first rectified pair
                        rect_left_gray = cv2.cvtColor(left_images[0], cv2.COLOR_BGR2GRAY) if len(left_images[0].shape) == 3 else left_images[0]
                        rect_right_gray = cv2.cvtColor(right_images[0], cv2.COLOR_BGR2GRAY) if len(right_images[0].shape) == 3 else right_images[0]
                        
                        # Use StereoBM for disparity map
                        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
                        disparity = stereo.compute(rect_left_gray, rect_right_gray)
                        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        cv2.imwrite(str(debug_dir / "disparity_map.jpg"), disparity_normalized)
                        
                        # Apply colormap for better visualization
                        disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
                        cv2.imwrite(str(debug_dir / "disparity_map_colored.jpg"), disparity_colored)
                        
                        # Save depth map visualization if Q matrix available
                        if self.calibration_data and 'Q' in self.calibration_data:
                            Q = np.array(self.calibration_data['Q'])
                            points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16.0, Q)
                            depth = points_3d[:, :, 2]
                            
                            # Mask invalid depth values
                            valid_depth = (depth > 0) & (depth < 2000)  # Reasonable depth range in mm
                            depth_viz = np.zeros_like(depth, dtype=np.uint8)
                            
                            # Fix the normalization - handle array dimensions correctly
                            if np.any(valid_depth):
                                valid_depth_values = depth[valid_depth]
                                # Ensure 1D input and output for boolean indexing
                                if valid_depth_values.size > 0:
                                    min_val, max_val = np.min(valid_depth_values), np.max(valid_depth_values)
                                    if max_val > min_val:
                                        normalized_values = ((valid_depth_values - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                                    else:
                                        normalized_values = np.full_like(valid_depth_values, 128, dtype=np.uint8)
                                    depth_viz[valid_depth] = normalized_values
                            
                            cv2.imwrite(str(debug_dir / "depth_map.jpg"), depth_viz)
                            depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_PLASMA)
                            cv2.imwrite(str(debug_dir / "depth_map_colored.jpg"), depth_colored)
                        
                        # Save epipolar line visualization
                        epi_img = self._draw_epipolar_lines(left_images[0], right_images[0])
                        cv2.imwrite(str(debug_dir / "epipolar_lines.jpg"), epi_img)
                        
                        logger.info("Enhanced debug visualizations saved")
                        
                    except Exception as e:
                        logger.error(f"Enhanced debug visualization failed: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.warning("Stereo rectification not available - point cloud may be distorted")
            
            # Process with pipeline using phase shift mode
            if self.pipeline:
                # Get pattern info
                pattern_info = {
                    'num_steps': self.session_metadata['capture']['pattern_info'].get('phase_shift_steps', 4),
                    'frequencies': self.session_metadata['capture']['pattern_info'].get('phase_frequencies', [1]),
                    'color': 'blue' if self.session_metadata['capture']['pattern_info'].get('uses_blue_channel', False) else 'white'
                }
                
                logger.info("Starting phase shift reconstruction...")
                
                # IMPORTANT: Update pipeline calibration to use rectified parameters if rectification was applied
                if (hasattr(self.pipeline, 'rectifier') and self.pipeline.rectifier.is_initialized and 
                    len(left_images) > 0):
                    # Use rectified calibration for triangulation
                    rectified_calib = self.pipeline.rectifier.get_rectified_calibration()
                    if rectified_calib:
                        self.pipeline.triangulator.calibration_data = rectified_calib
                        logger.info("[OK] Updated triangulator with rectified calibration parameters")
                
                # ADVANCED MODE: Use state-of-the-art stereo algorithms
                logger.info("USING ADVANCED STEREO RECONSTRUCTION MODE")
                
                scan_result = self._advanced_stereo_reconstruction(left_images, right_images, session_path=session_dir)
                
                # Fallback to complex pipeline if emergency mode fails
                if not scan_result.success:
                    logger.warning("Emergency mode failed, trying complex pipeline...")
                    scan_result = self.pipeline.process_gray_code_scan(
                        left_images, right_images,
                        pattern_info={'num_bits': 6, 'orientation': 'vertical', 'color': pattern_info['color']}
                    )
                
                if scan_result.success and scan_result.points_3d is not None:
                    points_3d = scan_result.points_3d
                    result['num_points'] = len(points_3d)
                    
                    # Save point cloud
                    self._save_point_cloud(points_3d, output_dir / "point_cloud_phase", self.config.output_format)
                    
                    # Apply filtering if enabled
                    if self.config.enable_filtering and len(points_3d) > 0:
                        filtered_points = self._filter_point_cloud(points_3d)
                        self._save_point_cloud(filtered_points, output_dir / "point_cloud_phase_filtered", self.config.output_format)
                        result['num_points_filtered'] = len(filtered_points)
                    
                    result['success'] = True
                    logger.info(f"[SUCCESS] Phase shift reconstruction successful: {result['num_points']} points")
                else:
                    result['error'] = scan_result.error_message if hasattr(scan_result, 'error_message') else "Unknown error"
                    logger.error(f"❌ Phase shift reconstruction failed: {result['error']}")
            else:
                result['error'] = "Pipeline not available"
                logger.error("Pipeline not initialized")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing phase shift patterns: {e}")
            import traceback
            traceback.print_exc()
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _save_debug_step(self, session_dir: Path, step_name: str, data: Any, description: str = ""):
        """Save debug step for ISO/ASTM 52902 compliance tracking."""
        if not self.config.save_debug_steps:
            return
        
        try:
            debug_dir = session_dir / "debug_steps"
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%H%M%S")
            step_file = debug_dir / f"{timestamp}_{step_name}"
            
            if isinstance(data, np.ndarray):
                if data.dtype == np.uint8 or len(data.shape) == 3:  # Image
                    cv2.imwrite(f"{step_file}.jpg", data)
                    logger.info(f"🔍 Debug: Saved image {step_name}.jpg")
                else:  # Point cloud or array
                    np.save(f"{step_file}.npy", data)
                    logger.info(f"🔍 Debug: Saved array {step_name}.npy ({data.shape})")
            elif isinstance(data, dict):
                import json
                with open(f"{step_file}.json", 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"🔍 Debug: Saved metadata {step_name}.json")
            elif isinstance(data, str):
                with open(f"{step_file}.txt", 'w') as f:
                    f.write(f"{description}\n\n{data}")
                logger.info(f"🔍 Debug: Saved text {step_name}.txt")
            
            # Log to certification file
            cert_file = debug_dir / "iso_astm_52902_trace.log"
            with open(cert_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {step_name}: {description}\n")
                
        except Exception as e:
            logger.warning(f"Failed to save debug step {step_name}: {e}")
    
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
    
    def _draw_epipolar_lines(self, left_img: np.ndarray, right_img: np.ndarray, num_lines: int = 10) -> np.ndarray:
        """Draw epipolar lines to verify rectification quality."""
        h, w = left_img.shape[:2]
        
        # Create side-by-side image
        combined = np.hstack([left_img, right_img])
        
        # Draw horizontal lines at regular intervals
        for i in range(0, h, h // num_lines):
            cv2.line(combined, (0, i), (w * 2, i), (0, 255, 0), 1)
            
        return combined
    
    def _draw_correspondences(self, left_img: np.ndarray, right_img: np.ndarray, 
                             left_pts: np.ndarray, right_pts: np.ndarray, 
                             confidence: np.ndarray, max_display: int = 100) -> np.ndarray:
        """Draw correspondence matches between stereo images."""
        # Create side-by-side image
        h, w = left_img.shape[:2]
        combined = np.hstack([left_img, right_img])
        
        # Sort by confidence and take top matches
        if len(confidence) > max_display:
            top_indices = np.argsort(confidence)[-max_display:]
            left_pts = left_pts[top_indices]
            right_pts = right_pts[top_indices]
            confidence = confidence[top_indices]
        
        # Draw matches with color coding based on confidence
        for i, (conf, lpt, rpt) in enumerate(zip(confidence, left_pts, right_pts)):
            # Color from red (low confidence) to green (high confidence)
            color = (0, int(255 * conf), int(255 * (1 - conf)))
            
            # Draw points
            cv2.circle(combined, (int(lpt[0]), int(lpt[1])), 3, color, -1)
            cv2.circle(combined, (int(rpt[0] + w), int(rpt[1])), 3, color, -1)
            
            # Draw connection line
            cv2.line(combined, 
                    (int(lpt[0]), int(lpt[1])), 
                    (int(rpt[0] + w), int(rpt[1])), 
                    color, 1)
        
        return combined
    
    def _emergency_direct_stereo_scan(self, left_images, right_images):
        """Emergency direct stereo scanning bypassing pattern decoder complexity"""
        try:
            from .reconstruction import ScanningResult
            
            logger.info(f"Emergency stereo: Processing {len(left_images)} image pairs with basic stereo matching")
            
            # Process multiple pattern images for better reconstruction
            all_points = []
            
            # Use pattern images (skip reference white/black at indices 0,1)
            pattern_indices = range(2, min(len(left_images), 6))  # Use up to 4 pattern images
            
            for img_idx in pattern_indices:
                logger.info(f"Processing pattern image {img_idx}/{len(left_images)}")
                
                left_img = left_images[img_idx]
                right_img = right_images[img_idx]
                
                # Convert to grayscale
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img
                
                # Create stereo matcher with good settings
                stereo = cv2.StereoBM_create(numDisparities=48, blockSize=21)  # More conservative settings
                disparity = stereo.compute(left_gray, right_gray)
                
                # Process this image pair
                pattern_points = self._extract_points_from_disparity(disparity, left_img.shape[:2])
                if len(pattern_points) > 0:
                    all_points.extend(pattern_points)
                    logger.info(f"  Added {len(pattern_points)} points from pattern {img_idx}")
                else:
                    logger.warning(f"  No valid points from pattern {img_idx}")
            
            # Combine all points from multiple patterns
            if len(all_points) > 0:
                all_points = np.array(all_points)
                
                # Remove duplicates and subsample for realistic density
                if len(all_points) > 5000:
                    step = len(all_points) // 2000  # Max 2000 points total
                    final_points = all_points[::step]
                else:
                    final_points = all_points
                
                logger.info(f"Emergency stereo: Final result - {len(final_points)} points from {len(all_points)} total")
                
                return ScanningResult(
                    points_3d=final_points,
                    uncertainty_data=None,
                    correspondence_data=None,
                    processing_stats={'method': 'emergency_multi_pattern', 'num_points': len(final_points)},
                    success=True
                )
            
            logger.warning("Emergency stereo: No points from any pattern")
            return ScanningResult(
                points_3d=None,
                uncertainty_data=None,
                correspondence_data=None,
                processing_stats={'method': 'emergency_stereo', 'error': 'no_patterns_produced_points'},
                success=False,
                error_message="No valid points from any pattern"
            )
            
        except Exception as e:
            logger.error(f"Emergency stereo failed: {e}")
            return ScanningResult(
                points_3d=None,
                uncertainty_data=None,
                correspondence_data=None,
                processing_stats={'method': 'emergency_stereo', 'error': str(e)},
                success=False,
                error_message=f"Emergency stereo error: {e}"
            )
    
    def _advanced_stereo_reconstruction(self, left_images, right_images, session_path=None):
        """Advanced stereo reconstruction using fixed algorithms."""
        try:
            from .reconstruction import ScanningResult
            from .advanced_stereo_matcher import AdvancedStereoMatcher
            from .advanced_triangulator import AdvancedTriangulator
            
            logger.info("ADVANCED STEREO RECONSTRUCTION: Using improved algorithms")
            
            # Get image dimensions
            height, width = left_images[0].shape[:2]
            image_size = (width, height)
            baseline_mm = self.calibration_data.get('baseline_mm', 79.5)
            
            # Initialize advanced components
            stereo_matcher = AdvancedStereoMatcher(image_size, baseline_mm)
            triangulator = AdvancedTriangulator(self.calibration_data)
            
            # Log calibration info
            logger.info(f"Calibration info:")
            logger.info(f"  Image size: {image_size}")
            logger.info(f"  Baseline: {baseline_mm:.1f} mm")
            logger.info(f"  Focal length: {triangulator.focal_length:.1f} pixels")
            if hasattr(triangulator, 'calibration_scale_factor'):
                logger.info(f"  Calibration scale factor: {triangulator.calibration_scale_factor:.2f}")
            
            all_points = []
            
            # Process only a few key patterns for speed
            pattern_indices = range(2, min(len(left_images), 6))  # Use 4 patterns max
            
            for i, img_idx in enumerate(pattern_indices):
                logger.info(f"Processing pattern {img_idx}")
                
                left_img = left_images[img_idx]
                right_img = right_images[img_idx]
                
                # Convert to grayscale
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img
                
                # Use advanced stereo matcher
                disparity_map, disp_stats = stereo_matcher.compute_disparity(left_img, right_img)
                
                if disp_stats.get('total_valid_pixels', 0) == 0:
                    logger.warning(f"  No valid disparity for pattern {img_idx}")
                    continue
                
                # Save debug visualizations if enabled
                if self.config.save_debug_steps and session_path:
                    debug_dir = Path(session_path) / "debug_advanced_reconstruction"
                    debug_dir.mkdir(exist_ok=True)
                    
                    # Save disparity map
                    disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    cv2.imwrite(str(debug_dir / f"disparity_pattern_{img_idx}.png"), disparity_normalized)
                    
                    # Save disparity map with color
                    disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
                    cv2.imwrite(str(debug_dir / f"disparity_colored_pattern_{img_idx}.png"), disparity_colored)
                
                # Triangulate to 3D using advanced triangulator
                points_3d, tri_stats = triangulator.triangulate_from_disparity(disparity_map, left_img)
                
                if len(points_3d) > 0:
                    # The triangulator already applies calibration correction and filtering
                    # Just subsample if needed for reasonable density
                    if len(points_3d) > 2000:
                        step = len(points_3d) // 1000
                        points_3d = points_3d[::step]
                    
                    all_points.extend(points_3d)
                    logger.info(f"  [OK] Pattern {img_idx}: {len(points_3d)} points")
                    logger.info(f"     Depth range: {points_3d[:,2].min():.0f}-{points_3d[:,2].max():.0f} mm")
                else:
                    logger.warning(f"  [FAIL] Pattern {img_idx}: No valid points after triangulation")
            
            if len(all_points) > 0:
                final_points = np.array(all_points)
                
                # Apply final global filtering
                final_points = self._apply_global_point_filtering(final_points)
                
                logger.info(f"\nADVANCED RECONSTRUCTION SUCCESS:")
                logger.info(f"  Total points: {len(final_points)}")
                logger.info(f"  X range: {final_points[:,0].min():.0f} to {final_points[:,0].max():.0f} mm")
                logger.info(f"  Y range: {final_points[:,1].min():.0f} to {final_points[:,1].max():.0f} mm") 
                logger.info(f"  Z range: {final_points[:,2].min():.0f} to {final_points[:,2].max():.0f} mm")
                logger.info(f"  Mean depth: {final_points[:,2].mean():.0f} mm")
                
                # Check if depth is reasonable
                mean_depth = final_points[:,2].mean()
                if 200 < mean_depth < 800:
                    logger.info(f"  [OK] Mean depth is in expected range for desktop scanning")
                else:
                    logger.warning(f"  [WARNING] Mean depth {mean_depth:.0f}mm seems unusual for desktop scanning")
                
                return ScanningResult(
                    points_3d=final_points,
                    uncertainty_data=None,
                    correspondence_data=None,
                    processing_stats={
                        'method': 'advanced_stereo',
                        'num_points': len(final_points),
                        'mean_depth': float(mean_depth),
                        'calibration_corrected': hasattr(triangulator, 'calibration_scale_factor') and abs(triangulator.calibration_scale_factor - 1.0) > 0.01
                    },
                    success=True
                )
            else:
                logger.error("[FAILED] ADVANCED reconstruction: No points from any pattern")
                return ScanningResult(
                    points_3d=None,
                    uncertainty_data=None,
                    correspondence_data=None,
                    processing_stats={'method': 'advanced_stereo', 'error': 'no_points'},
                    success=False,
                    error_message="No valid points from any pattern"
                )
                
        except Exception as e:
            logger.error(f"ADVANCED reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return ScanningResult(
                points_3d=None,
                uncertainty_data=None,
                correspondence_data=None,
                processing_stats={'method': 'advanced_stereo', 'error': str(e)},
                success=False,
                error_message=f"Advanced reconstruction error: {e}"
            )
    
    def _apply_global_point_filtering(self, points: np.ndarray) -> np.ndarray:
        """Apply final global filtering to merged point cloud."""
        
        if len(points) == 0:
            return points
        
        logger.info(f"🔧 Applying global filtering to {len(points)} points")
        
        # Remove duplicate points within 2mm tolerance
        if len(points) > 1000:
            try:
                # Use spatial hashing for large point clouds (NumPy only)
                grid_size = 2.0  # 2mm grid
                quantized = np.round(points / grid_size).astype(np.int32)
                _, unique_indices = np.unique(quantized, axis=0, return_index=True)
                points = points[unique_indices]
                
                logger.info(f"  🔍 Removed duplicates: {len(points)} unique points remain")
                
            except Exception as e:
                logger.warning(f"Duplicate removal failed: {e}")
        
        # Remove statistical outliers
        if len(points) > 100:
            try:
                # Calculate point-to-centroid distances
                centroid = np.mean(points, axis=0)
                distances = np.linalg.norm(points - centroid, axis=1)
                
                # Remove points beyond 3 standard deviations
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                outlier_threshold = mean_dist + 3 * std_dist
                
                inlier_mask = distances < outlier_threshold
                points = points[inlier_mask]
                
                logger.info(f"  📊 Removed outliers: {len(points)} inliers remain")
                
            except Exception as e:
                logger.warning(f"Statistical filtering failed: {e}")
        
        # Final reasonable bounds check
        reasonable_bounds = {
            'x': [-1000, 1000],  # ±1m lateral
            'y': [-1000, 1000],  # ±1m vertical  
            'z': [100, 3000]     # 10cm to 3m depth
        }
        
        bounds_mask = (
            (points[:, 0] >= reasonable_bounds['x'][0]) & (points[:, 0] <= reasonable_bounds['x'][1]) &
            (points[:, 1] >= reasonable_bounds['y'][0]) & (points[:, 1] <= reasonable_bounds['y'][1]) &
            (points[:, 2] >= reasonable_bounds['z'][0]) & (points[:, 2] <= reasonable_bounds['z'][1])
        )
        
        final_points = points[bounds_mask]
        
        logger.info(f"  [OK] Final bounds check: {len(final_points)} points in reasonable range")
        
        return final_points
    
    def _extract_points_from_disparity(self, disparity, image_shape):
        """Extract 3D points from disparity map with proper scaling"""
        try:
            if self.calibration_data and 'Q' in self.calibration_data:
                Q = np.array(self.calibration_data['Q'])
                
                # Scale Q matrix for actual image size
                actual_h, actual_w = image_shape
                calib_size = self.calibration_data['image_size']
                scale_x = actual_w / calib_size[0]
                scale_y = actual_h / calib_size[1]
                
                # Scale Q matrix
                Q_scaled = Q.copy()
                Q_scaled[0, 3] *= scale_x  # cx
                Q_scaled[1, 3] *= scale_y  # cy
                Q_scaled[2, 3] *= scale_x  # focal length scaling
                
                # Reproject to 3D
                points_3d_img = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16.0, Q_scaled)
                
                # Extract valid points with reasonable filtering  
                valid_mask = (disparity > 0) & (disparity < 63*16)  # Use full disparity range with corrected calibration
                
                if np.any(valid_mask):
                    points_3d = points_3d_img[valid_mask]
                    
                    # Filter out infinite/nan points
                    finite_mask = np.all(np.isfinite(points_3d), axis=1)
                    if np.any(finite_mask):
                        points_3d_clean = points_3d[finite_mask]
                        
                        # Reasonable depth filtering with corrected calibration
                        depth_mask = (points_3d_clean[:, 2] > 500) & (points_3d_clean[:, 2] < 3000)  # 50cm-3m range
                        
                        if np.any(depth_mask):
                            points_filtered = points_3d_clean[depth_mask]
                            
                            # Subsample for reasonable density per pattern
                            if len(points_filtered) > 500:
                                step = len(points_filtered) // 300  # Max 300 points per pattern
                                return points_filtered[::step]
                            else:
                                return points_filtered
            
            return []
            
        except Exception as e:
            logger.error(f"Point extraction failed: {e}")
            return []
    
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