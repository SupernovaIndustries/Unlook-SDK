#!/usr/bin/env python3
"""
Camera Configuration Demo for UnLook SDK

This example demonstrates how to:
1. Connect to an UnLook scanner
2. Configure and apply different camera settings
3. Capture images with different configurations
4. Use scanner with different camera configurations

Usage:
    python camera_configuration_demo.py
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, '..')

from unlook import UnlookClient
from unlook.client.camera_config import CameraConfig, ColorMode, CompressionFormat, ImageQualityPreset
from unlook.client.scan_config import RealTimeScannerConfig, ScanningMode, PatternType, ScanningQuality


def save_image(image: np.ndarray, filename: str) -> None:
    """Save an image to disk, creating directories if needed."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, image)
    logger.info(f"Saved image to {filename} ({os.path.getsize(filename)} bytes)")


def capture_with_config(client: UnlookClient, camera_id: str, config: Dict[str, Any], filename: str) -> None:
    """Capture an image with specified configuration."""
    # Apply configuration
    client.camera.configure(camera_id, config)
    
    # Log configuration for clarity
    logger.info(f"Applied configuration: {config}")
    
    # Wait for configuration to take effect
    time.sleep(0.2)
    
    # Capture image
    image = client.camera.capture(camera_id)
    
    if image is not None:
        # Save image
        save_image(image, filename)
    else:
        logger.error(f"Failed to capture image with config: {config}")


def demo_camera_config(client: UnlookClient) -> None:
    """Demonstrate different camera configurations."""
    # Create output directory
    os.makedirs("camera_demo_output", exist_ok=True)
    
    # Get available cameras
    cameras = client.camera.get_cameras()
    if not cameras:
        logger.error("No cameras available")
        return
    
    # Use the first camera
    camera_id = cameras[0]["id"]
    logger.info(f"Using camera: {camera_id}")
    
    # Demo 1: Configure using CameraConfig object
    logger.info("=== Demo 1: Using CameraConfig object ===")
    
    # Create configuration
    config = CameraConfig()
    config.set_color_mode(ColorMode.COLOR)
    config.set_resolution(640, 480)  # Smaller resolution for demo
    config.set_exposure(10000, auto_exposure=False)
    config.set_gain(1.2, auto_gain=False)
    config.set_compression(CompressionFormat.JPEG, jpeg_quality=90)
    
    # Apply configuration and capture
    client.camera.apply_camera_config(camera_id, config)
    logger.info("Applied camera configuration")
    
    # Capture and save image
    image = client.camera.capture(camera_id)
    if image is not None:
        save_image(image, "camera_demo_output/demo1_config_object.jpg")
    
    # Demo 2: Test different image quality presets
    logger.info("=== Demo 2: Image Quality Presets ===")
    
    for preset in [ImageQualityPreset.LOW, ImageQualityPreset.MEDIUM, ImageQualityPreset.HIGH]:
        # Create config with quality preset
        config = CameraConfig()
        config.set_resolution(640, 480)
        config.set_quality_preset(preset)
        
        # Capture with this configuration
        client.camera.apply_camera_config(camera_id, config)
        image = client.camera.capture(camera_id)
        
        if image is not None:
            save_image(image, f"camera_demo_output/demo2_quality_{preset.value}.jpg")
    
    # Demo 3: Test different image formats
    logger.info("=== Demo 3: Image Formats ===")
    
    for format in [CompressionFormat.JPEG, CompressionFormat.PNG]:
        config = CameraConfig()
        config.set_resolution(640, 480)
        config.set_compression(format)
        
        # Capture with this configuration
        client.camera.apply_camera_config(camera_id, config)
        image = client.camera.capture(camera_id)
        
        if image is not None:
            extension = "jpg" if format == CompressionFormat.JPEG else "png"
            save_image(image, f"camera_demo_output/demo3_format_{format.value}.{extension}")
    
    # Demo 4: Test cropping
    logger.info("=== Demo 4: Image Cropping ===")
    
    # First, capture full image
    config = CameraConfig()
    config.set_resolution(640, 480)
    client.camera.apply_camera_config(camera_id, config)
    full_image = client.camera.capture(camera_id)
    
    if full_image is not None:
        save_image(full_image, "camera_demo_output/demo4_full_image.jpg")
        
        # Now capture with crop region
        crop_config = CameraConfig()
        crop_config.set_resolution(640, 480)
        crop_config.set_crop_region(160, 120, 320, 240)  # Center crop
        
        client.camera.apply_camera_config(camera_id, crop_config)
        cropped_image = client.camera.capture(camera_id)
        
        if cropped_image is not None:
            save_image(cropped_image, "camera_demo_output/demo4_cropped_image.jpg")
    
    # Demo 5: Color modes
    logger.info("=== Demo 5: Color Modes ===")
    
    for color_mode in [ColorMode.COLOR, ColorMode.GRAYSCALE]:
        config = CameraConfig()
        config.set_resolution(640, 480)
        config.set_color_mode(color_mode)
        
        client.camera.apply_camera_config(camera_id, config)
        image = client.camera.capture(camera_id)
        
        if image is not None:
            save_image(image, f"camera_demo_output/demo5_color_mode_{color_mode.value}.jpg")
    
    # Demo 6: Image processing
    logger.info("=== Demo 6: Image Processing Options ===")
    
    # Standard image
    standard_config = CameraConfig()
    standard_config.set_resolution(640, 480)
    client.camera.apply_camera_config(camera_id, standard_config)
    standard_image = client.camera.capture(camera_id)
    
    if standard_image is not None:
        save_image(standard_image, "camera_demo_output/demo6_standard.jpg")
    
    # Enhanced image
    enhanced_config = CameraConfig()
    enhanced_config.set_resolution(640, 480)
    enhanced_config.set_image_adjustments(contrast=1.2, brightness=0.1, sharpness=0.5)
    enhanced_config.set_image_processing(denoise=True)
    
    client.camera.apply_camera_config(camera_id, enhanced_config)
    enhanced_image = client.camera.capture(camera_id)
    
    if enhanced_image is not None:
        save_image(enhanced_image, "camera_demo_output/demo6_enhanced.jpg")


def demo_scanner_config(client: UnlookClient) -> None:
    """Demonstrate camera configuration through scanner configuration."""
    logger.info("=== Demo 7: Scanner Configuration with Camera Settings ===")
    
    # Create scanner instance
    scanner = client.real_time_scanner
    
    # Configure scanner for high quality scanning
    high_quality_config = RealTimeScannerConfig()
    high_quality_config.quality = ScanningQuality.HIGH
    high_quality_config.pattern_type = PatternType.PHASE_SHIFT
    high_quality_config.set_image_quality(ImageQualityPreset.HIGH)
    high_quality_config.set_color_mode(ColorMode.GRAYSCALE)
    high_quality_config.set_image_processing(contrast=1.2, denoise=True)
    high_quality_config.output_directory = "camera_demo_output/scanner_high_quality"
    
    # Apply configuration
    scanner.configure(high_quality_config)
    
    # Start scan (will run with these settings)
    logger.info("Starting scanner with HIGH quality settings")
    scanner.start()
    
    # Wait for scanner to run a bit
    time.sleep(5)
    
    # Stop scanner
    scanner.stop()
    
    # Now configure for fast scanning
    fast_config = RealTimeScannerConfig()
    fast_config.quality = ScanningQuality.LOW
    fast_config.pattern_type = PatternType.GRAY_CODE
    fast_config.set_image_quality(ImageQualityPreset.LOW)
    fast_config.set_resolution(640, 480)  # Lower resolution for speed
    fast_config.set_color_mode(ColorMode.GRAYSCALE)
    fast_config.output_directory = "camera_demo_output/scanner_fast"
    
    # Apply configuration
    scanner.configure(fast_config)
    
    # Start scan
    logger.info("Starting scanner with FAST (low quality) settings")
    scanner.start()
    
    # Wait for scanner to run a bit
    time.sleep(5)
    
    # Stop scanner
    scanner.stop()


def main():
    """Main function to run the example."""
    # Connect to the UnLook scanner
    client = UnlookClient()
    
    try:
        # Start discovery
        client.start_discovery()
        
        # Wait for scanners to be discovered
        logger.info("Waiting for scanners to be discovered...")
        time.sleep(3.0)
        
        # Find available scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanner found. Please make sure the UnLook scanner is running.")
            return
        
        # Connect to first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} at {scanner_info.host}:{scanner_info.port}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Run camera config demo
        demo_camera_config(client)
        
        # Run scanner config demo
        demo_scanner_config(client)
        
        logger.info("Demo completed!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Disconnect
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()