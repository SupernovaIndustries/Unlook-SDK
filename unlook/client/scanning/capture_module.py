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
from unlook.client.scanner.scanner import UnlookClient
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
                'ip': scanner_info.ip,
                'preprocessing_version': preprocessing_version
            }
            
            # Check Protocol V2 status
            if self.client.is_protocol_v2_enabled():
                logger.info("✅ Protocol V2 is enabled")
                self.session_metadata['protocol_v2'] = True
            else:
                logger.warning("⚠️ Protocol V2 is NOT enabled")
                self.session_metadata['protocol_v2'] = False
            
            # Check preprocessing status
            preprocessing_info = self.client.get_preprocessing_info()
            if preprocessing_info:
                logger.info(f"GPU preprocessing: {'✅' if preprocessing_info.get('gpu_available') else '❌'}")
                logger.info(f"Preprocessing level: {preprocessing_info.get('level', 'none')}")
                self.session_metadata['preprocessing_info'] = preprocessing_info
            
            # Enable synchronization
            if self.client.enable_sync(enable=True, fps=30.0):
                logger.info("✅ Hardware synchronization enabled")
                self.session_metadata['sync_enabled'] = True
            else:
                logger.warning("⚠️ Hardware synchronization not available")
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
            logger.info("🔆 Initializing LED flood illuminator (AS1170)...")
            
            self.led_controller = LEDController(self.client)
            
            if self.led_controller.led_available:
                # Set LED2 to 100mA for optimal structured light scanning
                led2_intensity = 100  # mA
                
                if self.led_controller.set_intensity(0, led2_intensity):
                    self.led_active = True
                    logger.info(f"✅ LED flood illuminator activated: LED2={led2_intensity}mA")
                    self.session_metadata['led_enabled'] = True
                    self.session_metadata['led_intensity'] = led2_intensity
                else:
                    logger.warning("⚠️ Failed to activate LED flood illuminator")
                    self.led_controller = None
            else:
                logger.warning("⚠️ LED flood illuminator not available")
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
        if not self.client or not self.client.is_connected():
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
        
        # Capture each pattern
        start_time = time.time()
        
        for idx, pattern in enumerate(patterns):
            logger.info(f"Pattern {idx+1}/{len(patterns)}: {pattern.name}")
            
            # Project pattern
            success = self._project_pattern(pattern)
            if not success:
                logger.error(f"Failed to project pattern {idx}")
                continue
            
            # Wait for pattern to stabilize
            time.sleep(pattern_switch_delay)
            
            # Capture synchronized images
            try:
                images = self.client.camera.capture_multi(camera_ids)
                
                if images and len(images) == 2:
                    # Extract left and right images
                    left_img = images[camera_ids[0]]
                    right_img = images[camera_ids[1]]
                    
                    # Save images
                    left_path = self._save_image(left_img, f"left_{idx:03d}_{pattern.name}")
                    right_path = self._save_image(right_img, f"right_{idx:03d}_{pattern.name}")
                    
                    # Record capture info
                    capture_info['captured_images'].append({
                        'index': idx,
                        'pattern': pattern.to_dict(),
                        'left_image': os.path.basename(left_path),
                        'right_image': os.path.basename(right_path),
                        'timestamp': time.time()
                    })
                    
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
        
        # Save metadata
        if save_metadata:
            self._save_session_metadata(capture_info)
        
        logger.info(f"Capture completed in {capture_time:.1f}s")
        logger.info(f"Success rate: {capture_info['success_rate']*100:.1f}%")
        
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
            # Try to get calibration from client or use default path
            calibration_path = Path("unlook/calibration/custom/stereo_calibration_fixed.json")
            if calibration_path.exists():
                # Copy to session directory
                import shutil
                dest_path = self.session_dir / "calibration.json"
                shutil.copy(calibration_path, dest_path)
                logger.info(f"Saved calibration to {dest_path}")
                return str(dest_path)
            else:
                logger.warning("No calibration file found")
                return None
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return None
    
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
                    logger.info("🔆 LED flood illuminator deactivated")
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