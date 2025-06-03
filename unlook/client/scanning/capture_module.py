"""
Capture Module for Structured Light 3D Scanning.

This module handles the connection to the scanner, pattern projection,
and synchronized image capture. It saves all captured data with metadata
for offline processing.
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

from unlook.core.constants import PreprocessingVersion
from unlook.client.scanner.scanner import UnlookClient, FocusAssessment
from unlook.client.projector.led_controller import LEDController
from .pattern_manager import PatternManager, PatternInfo

logger = logging.getLogger(__name__)


class CaptureModule:
    """
    Handles pattern projection and synchronized image capture.
    
    This module manages the connection to the UnLook scanner,
    projects patterns, captures synchronized stereo images,
    and saves everything with metadata for offline processing.
    """
    
    def __init__(self, 
                 output_dir: str = "captured_data",
                 save_format: str = "jpg",
                 jpg_quality: int = 95,
                 enable_led: bool = True):
        """
        Initialize the capture module.
        
        Args:
            output_dir: Base directory for saving captured data
            save_format: Image format ('jpg' or 'png')
            jpg_quality: JPEG quality (1-100) if using jpg format
            enable_led: Whether to use LED flood illuminator
        """
        self.output_dir = Path(output_dir)
        self.save_format = save_format.lower()
        self.jpg_quality = jpg_quality
        self.enable_led = enable_led
        
        # Scanner components
        self.client = None
        self.led_controller = None
        self.led_active = False
        
        # Pattern manager
        self.pattern_manager = PatternManager()
        
        # Focus assessment for debug
        self.focus_assessment = FocusAssessment()
        
        # Capture session info
        self.session_dir = None
        self.session_metadata = {}
        
    def connect_to_scanner(self, 
                          scanner_name: Optional[str] = None,
                          preprocessing_version: str = PreprocessingVersion.V2_ENHANCED) -> bool:
        """
        Connect to the UnLook scanner.
        
        Args:
            scanner_name: Optional specific scanner name to connect to
            preprocessing_version: Preprocessing pipeline version to use
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to UnLook scanner...")
            
            # Create client with V2 preprocessing for 3D scanning
            self.client = UnlookClient(
                client_name="CaptureModule",
                auto_discover=True,
                preprocessing_version=preprocessing_version
            )
            
            # Wait for discovery
            time.sleep(3)
            
            # Get available scanners
            scanners = self.client.get_discovered_scanners()
            if not scanners:
                logger.error("No scanners found!")
                return False
            
            # Select scanner
            if scanner_name:
                scanner_info = next((s for s in scanners if s.name == scanner_name), None)
                if not scanner_info:
                    logger.error(f"Scanner '{scanner_name}' not found")
                    return False
            else:
                scanner_info = scanners[0]
            
            logger.info(f"Connecting to: {scanner_info.name}")
            
            # Connect
            if not self.client.connect(scanner_info):
                logger.error("Failed to connect to scanner")
                return False
            
            # Store scanner info
            self.session_metadata['scanner_info'] = {
                'name': scanner_info.name,
                'host': scanner_info.host,
                'port': scanner_info.port,
                'preprocessing_version': preprocessing_version
            }
            
            # Check Protocol V2 status
            if self.client.is_protocol_v2_enabled():
                logger.info("Protocol V2 is enabled")
                self.session_metadata['protocol_v2'] = True
            else:
                logger.warning("Protocol V2 is NOT enabled")
                self.session_metadata['protocol_v2'] = False
            
            # Check preprocessing status
            preprocessing_info = self.client.get_preprocessing_info()
            if preprocessing_info:
                logger.info(f"GPU preprocessing: {'YES' if preprocessing_info.get('gpu_available') else 'NO'}")
                logger.info(f"Preprocessing level: {preprocessing_info.get('level', 'none')}")
                self.session_metadata['preprocessing_info'] = preprocessing_info
            
            # Enable synchronization
            if self.client.enable_sync(enable=True, fps=30.0):
                logger.info("Hardware synchronization enabled")
                self.session_metadata['sync_enabled'] = True
            else:
                logger.warning("Hardware synchronization not available")
                self.session_metadata['sync_enabled'] = False
            
            # Initialize LED if enabled
            if self.enable_led:
                self._setup_led_controller()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def _setup_led_controller(self):
        """Initialize and configure the LED flood illuminator."""
        try:
            logger.info("Initializing LED flood illuminator (AS1170)...")
            
            self.led_controller = LEDController(self.client)
            
            if self.led_controller.led_available:
                # Set LED2 to 100mA for optimal structured light scanning (same as enhanced pipeline)
                led2_intensity = 100  # mA - exactly like enhanced_3d_scanning_pipeline_v2.py
                
                # Turn on LED2 (index 0 controls LED2 in AS1170)
                if self.led_controller.set_intensity(0, led2_intensity):
                    self.led_active = True
                    logger.info(f"LED flood illuminator activated: LED2={led2_intensity}mA")
                    self.session_metadata['led_enabled'] = True
                    self.session_metadata['led_intensity'] = led2_intensity
                else:
                    logger.warning("Failed to activate LED flood illuminator")
                    self.led_controller = None
            else:
                logger.warning("LED flood illuminator not available")
                self.led_controller = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize LED controller: {e}")
            self.led_controller = None
    
    def create_session(self, session_name: Optional[str] = None) -> str:
        """
        Create a new capture session directory.
        
        Args:
            session_name: Optional session name, otherwise timestamp is used
            
        Returns:
            Path to session directory
        """
        # Create session directory
        if session_name:
            session_dirname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_name}"
        else:
            session_dirname = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.session_dir = self.output_dir / session_dirname
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session metadata
        self.session_metadata.update({
            'session_name': session_name or session_dirname,
            'timestamp': datetime.now().isoformat(),
            'capture_module_version': '1.0'
        })
        
        logger.info(f"Created session directory: {self.session_dir}")
        return str(self.session_dir)
    
    def capture_pattern_sequence(self, 
                               patterns: List[PatternInfo],
                               pattern_switch_delay: float = 0.1,
                               save_metadata: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Capture a sequence of patterns.
        
        Args:
            patterns: List of patterns to project and capture
            pattern_switch_delay: Delay after pattern projection (seconds)
            save_metadata: Whether to save metadata file
            
        Returns:
            Tuple of (success, capture_info)
        """
        if not self.client or not self.client.connected:
            logger.error("Not connected to scanner")
            return False, {"error": "Not connected"}
        
        if not self.session_dir:
            logger.error("No session created")
            return False, {"error": "No session"}
        
        logger.info(f"Starting capture of {len(patterns)} patterns")
        
        # Get available cameras
        cameras = self.client.camera.get_cameras()
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return False, {"error": "Insufficient cameras"}
        
        # Use first two cameras as left and right
        camera_ids = [cam['id'] for cam in cameras[:2]]
        logger.info(f"Using cameras: {camera_ids}")
        
        # Store camera info in metadata
        self.session_metadata['cameras'] = {
            'left': cameras[0],
            'right': cameras[1]
        }
        
        # Create capture info
        capture_info = {
            'num_patterns': len(patterns),
            'camera_ids': camera_ids,
            'captured_images': [],
            'pattern_info': self.pattern_manager.get_pattern_info_dict(patterns),
            'calibration_file': None
        }
        
        # Save calibration if available
        calibration_path = self._save_calibration()
        if calibration_path:
            capture_info['calibration_file'] = os.path.basename(calibration_path)
        
        # Ensure LED is active before starting capture sequence
        if self.led_controller and self.enable_led:
            if not self.led_active:
                logger.warning("LED not active, attempting to reactivate...")
                if self.led_controller.set_intensity(0, 100):
                    self.led_active = True
                    logger.info("LED2 reactivated at 100mA")
            else:
                logger.info("LED2 confirmed active at 100mA for capture sequence")
        
        # Capture each pattern
        start_time = time.time()
        
        for idx, pattern in enumerate(patterns):
            logger.info(f"Pattern {idx+1}/{len(patterns)}: {pattern.name}")
            
            # Project pattern
            success = self._project_pattern(pattern)
            if not success:
                logger.error(f"Failed to project pattern {idx}")
                continue
            
            # Wait for pattern to stabilize - extra time for sync
            time.sleep(pattern_switch_delay)
            
            # Additional stabilization for critical patterns
            if pattern.pattern_type == "vertical_lines":
                time.sleep(0.1)  # Extra time for structured light patterns
            
            # Flush camera buffers before capture for better sync
            if hasattr(self.client.camera, 'flush_buffers'):
                self.client.camera.flush_buffers(camera_ids)
            
            # Capture synchronized images
            try:
                # Try multiple times if sync fails
                images = None
                for attempt in range(3):
                    images = self.client.camera.capture_multi(camera_ids)
                    if images and len(images) == 2:
                        break
                    if attempt < 2:
                        logger.warning(f"Sync attempt {attempt+1} failed, retrying...")
                        time.sleep(0.05)
                
                if images and len(images) == 2:
                    # Extract left and right images
                    left_img = images[camera_ids[0]]
                    right_img = images[camera_ids[1]]
                    
                    # Focus assessment for each captured image
                    left_focus = self.focus_assessment.assess_camera_focus(left_img, 'left')
                    right_focus = self.focus_assessment.assess_camera_focus(right_img, 'right')
                    
                    # Assess projector focus if pattern is structural
                    projector_focus = None
                    if pattern.pattern_type == "vertical_lines":
                        projector_focus = self.focus_assessment.assess_projector_focus(left_img, 'lines')
                    elif pattern.pattern_type == "horizontal_lines":
                        projector_focus = self.focus_assessment.assess_projector_focus(left_img, 'lines')
                    
                    # Get overall focus status
                    overall_focus = self.focus_assessment.get_overall_focus_status()
                    
                    # Log focus quality
                    logger.info(f"Focus Quality - Left: {left_focus['quality']} ({left_focus['smoothed_score']:.1f}), Right: {right_focus['quality']} ({right_focus['smoothed_score']:.1f})")
                    if projector_focus:
                        logger.info(f"🔍 Projector Focus: {projector_focus['quality']} (contrast: {projector_focus.get('contrast', 0):.2f})")
                    logger.info(f"Overall Status: {overall_focus['message']}")
                    
                    # Save images
                    left_path = self._save_image(left_img, f"left_{idx:03d}_{pattern.name}")
                    right_path = self._save_image(right_img, f"right_{idx:03d}_{pattern.name}")
                    
                    # Record capture info with focus data
                    capture_record = {
                        'index': idx,
                        'pattern': pattern.to_dict(),
                        'left_image': os.path.basename(left_path),
                        'right_image': os.path.basename(right_path),
                        'timestamp': time.time(),
                        'focus_assessment': {
                            'left_camera': {
                                'quality': left_focus['quality'],
                                'score': left_focus['smoothed_score'],
                                'laplacian_var': left_focus.get('laplacian_var', 0),
                                'gradient_mean': left_focus.get('gradient_mean', 0),
                                'high_freq_ratio': left_focus.get('high_freq_ratio', 0)
                            },
                            'right_camera': {
                                'quality': right_focus['quality'],
                                'score': right_focus['smoothed_score'],
                                'laplacian_var': right_focus.get('laplacian_var', 0),
                                'gradient_mean': right_focus.get('gradient_mean', 0),
                                'high_freq_ratio': right_focus.get('high_freq_ratio', 0)
                            },
                            'overall_status': overall_focus['status'],
                            'overall_message': overall_focus['message']
                        }
                    }
                    
                    # Add projector focus if available
                    if projector_focus:
                        capture_record['focus_assessment']['projector'] = {
                            'quality': projector_focus['quality'],
                            'score': projector_focus.get('focus_score', 0),
                            'contrast': projector_focus.get('contrast', 0),
                            'pattern_regularity': projector_focus.get('pattern_regularity', 0)
                        }
                    
                    capture_info['captured_images'].append(capture_record)
                    
                    logger.debug(f"Captured: L={left_img.shape}, R={right_img.shape}")
                else:
                    logger.error(f"Failed to capture images for pattern {idx}")
                    
            except Exception as e:
                logger.error(f"Capture error for pattern {idx}: {e}")
        
        # Turn off projector and LED
        logger.info("Turning off projector and LED...")
        self.client.projector.show_solid_field("Black")
        self._cleanup_led()
        
        # Calculate capture statistics
        capture_time = time.time() - start_time
        capture_info['capture_time_seconds'] = capture_time
        capture_info['success_rate'] = len(capture_info['captured_images']) / len(patterns)
        
        # Calculate focus statistics
        focus_stats = self._calculate_focus_stats(capture_info['captured_images'])
        capture_info['focus_statistics'] = focus_stats
        
        # Save metadata
        if save_metadata:
            self._save_session_metadata(capture_info)
        
        logger.info(f"Capture completed in {capture_time:.1f}s")
        logger.info(f"Success rate: {capture_info['success_rate']*100:.1f}%")
        
        # Log focus summary
        logger.info("FOCUS SUMMARY:")
        if 'left_camera' in focus_stats:
            logger.info(f"  Left Camera - Avg: {focus_stats['left_camera']['avg_score']:.1f} | Quality: {focus_stats['left_camera']['quality_distribution']}")
            if focus_stats['left_camera']['avg_score'] < 50:
                logger.warning("LEFT CAMERA FOCUS IS POOR - Consider adjusting lens focus")
        else:
            logger.warning("  Left Camera - No data")
            
        if 'right_camera' in focus_stats:
            logger.info(f"  Right Camera - Avg: {focus_stats['right_camera']['avg_score']:.1f} | Quality: {focus_stats['right_camera']['quality_distribution']}")
            if focus_stats['right_camera']['avg_score'] < 50:
                logger.warning("RIGHT CAMERA FOCUS IS POOR - Consider adjusting lens focus")
        else:
            logger.warning("  Right Camera - No data")
            
        if focus_stats.get('projector'):
            logger.info(f"  Projector - Avg: {focus_stats['projector']['avg_score']:.1f} | Quality: {focus_stats['projector']['quality_distribution']}")
            if focus_stats.get('projector', {}).get('avg_score', 100) < 50:
                logger.warning("PROJECTOR FOCUS IS POOR - Consider adjusting projector focus")
                
        if 'overall_assessment' in focus_stats:
            logger.info(f"  Overall Status: {focus_stats['overall_assessment']}")
        
        return True, capture_info
    
    def _project_pattern(self, pattern: PatternInfo) -> bool:
        """Project a single pattern."""
        try:
            params = pattern.parameters
            
            if pattern.pattern_type == "solid_field":
                return self.client.projector.show_solid_field(params.get("color", "White"))
                
            elif pattern.pattern_type == "vertical_lines":
                return self.client.projector.show_vertical_lines(
                    foreground_color=params.get("foreground_color", "White"),
                    background_color=params.get("background_color", "Black"),
                    foreground_width=params.get("foreground_width", 4),
                    background_width=params.get("background_width", 4)
                )
                
            elif pattern.pattern_type == "horizontal_lines":
                return self.client.projector.show_horizontal_lines(
                    foreground_color=params.get("foreground_color", "White"),
                    background_color=params.get("background_color", "Black"),
                    foreground_width=params.get("foreground_width", 4),
                    background_width=params.get("background_width", 4)
                )
                
            elif pattern.pattern_type == "sinusoidal":
                # For phase shift patterns - would need projector support
                logger.warning(f"Sinusoidal patterns not yet implemented in projector")
                return False
                
            else:
                logger.warning(f"Unknown pattern type: {pattern.pattern_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error projecting pattern: {e}")
            return False
    
    def _save_image(self, image: np.ndarray, name: str) -> str:
        """Save an image to the session directory."""
        if self.save_format == 'jpg':
            filename = f"{name}.jpg"
            filepath = self.session_dir / filename
            cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality])
        else:
            filename = f"{name}.png"
            filepath = self.session_dir / filename
            cv2.imwrite(str(filepath), image)
        
        return str(filepath)
    
    def _save_calibration(self) -> Optional[str]:
        """Save calibration data to session directory."""
        try:
            # Try multiple calibration paths
            calibration_paths = [
                Path("unlook/calibration/custom/stereo_calibration_fixed.json"),
                Path("unlook/calibration/custom/stereo_calibration.json"), 
                Path("unlook/calibration/default/default_stereo.json")
            ]
            
            for calibration_path in calibration_paths:
                if calibration_path.exists():
                    # Copy to session directory
                    import shutil
                    dest_path = self.session_dir / "calibration.json"
                    shutil.copy(calibration_path, dest_path)
                    logger.info(f"Saved calibration from {calibration_path} to {dest_path}")
                    return str(dest_path)
            
            logger.warning("No calibration file found in any default location")
            return None
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return None
    
    def _calculate_focus_stats(self, captured_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate focus statistics from captured images."""
        if not captured_images:
            return {"error": "No captured images"}
        
        # Collect focus data
        left_scores = []
        right_scores = []
        projector_scores = []
        left_qualities = []
        right_qualities = []
        projector_qualities = []
        
        for img_info in captured_images:
            focus_data = img_info.get('focus_assessment', {})
            
            # Left camera data
            left_cam = focus_data.get('left_camera', {})
            if 'score' in left_cam:
                left_scores.append(left_cam['score'])
                left_qualities.append(left_cam.get('quality', 'unknown'))
            
            # Right camera data  
            right_cam = focus_data.get('right_camera', {})
            if 'score' in right_cam:
                right_scores.append(right_cam['score'])
                right_qualities.append(right_cam.get('quality', 'unknown'))
            
            # Projector data (if available)
            proj = focus_data.get('projector', {})
            if 'score' in proj:
                projector_scores.append(proj['score'])
                projector_qualities.append(proj.get('quality', 'unknown'))
        
        # Calculate statistics
        stats = {
            'total_images': len(captured_images),
            'images_with_focus_data': len([img for img in captured_images if 'focus_assessment' in img])
        }
        
        # Left camera stats
        if left_scores:
            stats['left_camera'] = {
                'avg_score': sum(left_scores) / len(left_scores),
                'min_score': min(left_scores),
                'max_score': max(left_scores),
                'quality_distribution': self._count_qualities(left_qualities),
                'total_measurements': len(left_scores)
            }
        
        # Right camera stats
        if right_scores:
            stats['right_camera'] = {
                'avg_score': sum(right_scores) / len(right_scores),
                'min_score': min(right_scores),
                'max_score': max(right_scores),
                'quality_distribution': self._count_qualities(right_qualities),
                'total_measurements': len(right_scores)
            }
        
        # Projector stats (if available)
        if projector_scores:
            stats['projector'] = {
                'avg_score': sum(projector_scores) / len(projector_scores),
                'min_score': min(projector_scores),
                'max_score': max(projector_scores),
                'quality_distribution': self._count_qualities(projector_qualities),
                'total_measurements': len(projector_scores)
            }
        
        # Overall assessment
        left_avg = stats.get('left_camera', {}).get('avg_score', 0)
        right_avg = stats.get('right_camera', {}).get('avg_score', 0)
        
        if left_avg >= 100 and right_avg >= 100:
            overall = "EXCELLENT - Both cameras well focused"
        elif left_avg >= 50 and right_avg >= 50:
            overall = "GOOD - Focus acceptable for scanning"
        elif left_avg >= 20 and right_avg >= 20:
            overall = "FAIR - May affect scan quality"
        else:
            overall = "POOR - Focus adjustment required"
        
        stats['overall_assessment'] = overall
        
        return stats
    
    def _count_qualities(self, qualities: List[str]) -> Dict[str, int]:
        """Count quality distribution."""
        from collections import Counter
        return dict(Counter(qualities))
    
    def _save_session_metadata(self, capture_info: Dict[str, Any]):
        """Save session metadata to JSON file."""
        metadata = {
            'session': self.session_metadata,
            'capture': capture_info,
            'timestamp_end': datetime.now().isoformat()
        }
        
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def _cleanup_led(self):
        """Turn off LED flood illuminator."""
        if self.led_controller and self.led_active:
            try:
                if self.led_controller.turn_off():
                    logger.info("LED flood illuminator deactivated")
                    self.led_active = False
            except Exception as e:
                logger.error(f"Error turning off LED: {e}")
    
    def disconnect(self):
        """Disconnect from scanner and cleanup."""
        self._cleanup_led()
        
        if self.client:
            self.client.disconnect()
            logger.info("Disconnected from scanner")
    
    def capture_gray_code_sequence(self, 
                                  num_bits: int = 8,
                                  use_blue: bool = True,
                                  session_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Convenience method to capture a Gray code sequence.
        
        Args:
            num_bits: Number of Gray code bits
            use_blue: Use blue patterns instead of white
            session_name: Optional session name
            
        Returns:
            Tuple of (success, session_directory)
        """
        # Create session
        session_dir = self.create_session(session_name or "gray_code")
        
        # Generate patterns
        patterns = self.pattern_manager.create_gray_code_patterns(
            num_bits=num_bits,
            use_blue=use_blue
        )
        
        # Capture
        success, info = self.capture_pattern_sequence(patterns)
        
        return success, session_dir