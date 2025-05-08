#!/usr/bin/env python3
"""
Example script demonstrating how to use advanced camera configuration options 
with the UnLook scanner for 3D scanning.

This example shows how to:
1. Configure camera settings for different scanning scenarios
2. Use different image quality presets
3. Configure image formats, resolutions, and crop regions
4. Apply image processing settings

Usage:
    python camera_config_example.py
"""

import time
import logging
import sys
import os
import numpy as np
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, '..')

from unlook import UnlookClient
from unlook.client.camera_config import CameraConfig, ColorMode, CompressionFormat, ImageQualityPreset
from unlook.client.scan_config import RealTimeScannerConfig, ScanningQuality, PatternType, ScanningMode


def save_scan_results(scanner, result, base_name):
    """Save scan results to disk with the given base name."""
    if result and result.has_data():
        timestamp = int(time.time())
        filepath = os.path.join(scanner.config.output_directory, f"{base_name}_{timestamp}")
        
        # Save point cloud
        ply_path = f"{filepath}.ply"
        scanner._save_point_cloud(result.point_cloud, result.texture_map, ply_path)
        logger.info(f"Saved point cloud to {ply_path}")
        
        return ply_path
    else:
        logger.warning("No valid scan result to save")
        return None


def on_scan_completed(result):
    """Callback for when a scan is completed."""
    logger.info(f"Scan completed with {len(result.point_cloud) if result.point_cloud is not None else 0} points")
    logger.info(f"Scan time: {result.scan_time:.2f} seconds")


def on_scan_progress(progress, metadata):
    """Callback for scan progress updates."""
    logger.info(f"Scan progress: {progress*100:.1f}%, frames: {metadata.get('frame_count', 0)}")


def scan_with_quality_preset(client: UnlookClient, quality_preset: ImageQualityPreset, output_name: str):
    """Perform a scan with a specific image quality preset."""
    logger.info(f"\n=== Scanning with {quality_preset.value} Quality Preset ===")
    
    # Create scanner
    scanner = client.real_time_scanner
    
    # Configure scanner
    config = RealTimeScannerConfig()
    config.mode = ScanningMode.SINGLE
    config.quality = ScanningQuality.MEDIUM
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_interval = 0.1
    config.output_directory = "scan_output"
    config.save_raw_images = True
    
    # Set image quality preset
    config.set_image_quality(quality_preset)
    
    # Register callbacks
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_progress(on_scan_progress)
    
    # Apply configuration
    scanner.configure(config)
    
    # Start scanning
    logger.info("Starting scan...")
    scanner.start()
    
    # Wait for scan to complete
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    while scanner.scanning and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        
    # Make sure scanning is stopped
    if scanner.scanning:
        logger.warning("Scan timed out, stopping scanner")
        scanner.stop()
    
    # Save results
    result = scanner.get_latest_result()
    save_scan_results(scanner, result, f"{output_name}_{quality_preset.value}")
    
    logger.info(f"Completed {quality_preset.value} quality preset scan")


def scan_with_image_format(client: UnlookClient, format: CompressionFormat, output_name: str):
    """Perform a scan with a specific image format."""
    logger.info(f"\n=== Scanning with {format.value} Image Format ===")
    
    # Create scanner
    scanner = client.real_time_scanner
    
    # Configure scanner
    config = RealTimeScannerConfig()
    config.mode = ScanningMode.SINGLE
    config.quality = ScanningQuality.MEDIUM
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_interval = 0.1
    config.output_directory = "scan_output"
    config.save_raw_images = True
    
    # Set image format
    config.set_image_format(format)
    
    # If using JPEG, set quality
    if format == CompressionFormat.JPEG:
        config.jpeg_quality = 90
    
    # Register callbacks
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_progress(on_scan_progress)
    
    # Apply configuration
    scanner.configure(config)
    
    # Start scanning
    logger.info("Starting scan...")
    scanner.start()
    
    # Wait for scan to complete
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    while scanner.scanning and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        
    # Make sure scanning is stopped
    if scanner.scanning:
        logger.warning("Scan timed out, stopping scanner")
        scanner.stop()
    
    # Save results
    result = scanner.get_latest_result()
    save_scan_results(scanner, result, f"{output_name}_{format.value}")
    
    logger.info(f"Completed {format.value} image format scan")


def scan_with_color_mode(client: UnlookClient, color_mode: ColorMode, output_name: str):
    """Perform a scan with a specific color mode."""
    logger.info(f"\n=== Scanning with {color_mode.value} Color Mode ===")
    
    # Create scanner
    scanner = client.real_time_scanner
    
    # Configure scanner
    config = RealTimeScannerConfig()
    config.mode = ScanningMode.SINGLE
    config.quality = ScanningQuality.MEDIUM
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_interval = 0.1
    config.output_directory = "scan_output"
    config.save_raw_images = True
    
    # Set color mode
    config.set_color_mode(color_mode)
    
    # Register callbacks
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_progress(on_scan_progress)
    
    # Apply configuration
    scanner.configure(config)
    
    # Start scanning
    logger.info("Starting scan...")
    scanner.start()
    
    # Wait for scan to complete
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    while scanner.scanning and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        
    # Make sure scanning is stopped
    if scanner.scanning:
        logger.warning("Scan timed out, stopping scanner")
        scanner.stop()
    
    # Save results
    result = scanner.get_latest_result()
    save_scan_results(scanner, result, f"{output_name}_{color_mode.value}")
    
    logger.info(f"Completed {color_mode.value} color mode scan")


def scan_with_crop_region(client: UnlookClient, output_name: str):
    """Perform a scan with a crop region."""
    logger.info("\n=== Scanning with Crop Region ===")
    
    # Get camera info for setting crop region
    cameras = client.camera.get_cameras()
    if not cameras:
        logger.error("No cameras found, cannot determine resolution for crop")
        return
    
    camera_info = cameras[0]
    resolution = camera_info.get("resolution", [1920, 1080])
    
    # Calculate crop region (center 50%)
    width, height = resolution
    crop_x = int(width * 0.25)
    crop_y = int(height * 0.25)
    crop_width = int(width * 0.5)
    crop_height = int(height * 0.5)
    
    logger.info(f"Setting crop region to center 50%: ({crop_x}, {crop_y}, {crop_width}, {crop_height})")
    
    # Create scanner
    scanner = client.real_time_scanner
    
    # Configure scanner
    config = RealTimeScannerConfig()
    config.mode = ScanningMode.SINGLE
    config.quality = ScanningQuality.MEDIUM
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_interval = 0.1
    config.output_directory = "scan_output"
    config.save_raw_images = True
    
    # Set crop region
    config.set_crop_region(crop_x, crop_y, crop_width, crop_height)
    
    # Register callbacks
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_progress(on_scan_progress)
    
    # Apply configuration
    scanner.configure(config)
    
    # Start scanning
    logger.info("Starting scan with crop region...")
    scanner.start()
    
    # Wait for scan to complete
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    while scanner.scanning and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        
    # Make sure scanning is stopped
    if scanner.scanning:
        logger.warning("Scan timed out, stopping scanner")
        scanner.stop()
    
    # Save results
    result = scanner.get_latest_result()
    save_scan_results(scanner, result, f"{output_name}_cropped")
    
    logger.info("Completed scan with crop region")


def scan_with_image_processing(client: UnlookClient, output_name: str):
    """Perform a scan with image processing options."""
    logger.info("\n=== Scanning with Image Processing Options ===")
    
    # Create scanner
    scanner = client.real_time_scanner
    
    # Configure scanner
    config = RealTimeScannerConfig()
    config.mode = ScanningMode.SINGLE
    config.quality = ScanningQuality.MEDIUM
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_interval = 0.1
    config.output_directory = "scan_output"
    config.save_raw_images = True
    
    # Set image processing options
    config.set_image_processing(
        denoise=True,          # Enable noise reduction
        contrast=1.2,          # Boost contrast slightly
        brightness=0.05,       # Slightly brighter
        sharpness=0.5          # Medium sharpness
    )
    
    # Register callbacks
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_progress(on_scan_progress)
    
    # Apply configuration
    scanner.configure(config)
    
    # Start scanning
    logger.info("Starting scan with image processing...")
    scanner.start()
    
    # Wait for scan to complete
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    while scanner.scanning and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        
    # Make sure scanning is stopped
    if scanner.scanning:
        logger.warning("Scan timed out, stopping scanner")
        scanner.stop()
    
    # Save results
    result = scanner.get_latest_result()
    save_scan_results(scanner, result, f"{output_name}_processed")
    
    logger.info("Completed scan with image processing")


def main():
    """Main function to run the examples."""
    # Create output directory
    os.makedirs("scan_output", exist_ok=True)
    
    # Connect to the UnLook scanner
    client = UnlookClient()
    
    try:
        # Make sure discovery is started
        client.start_discovery()
        
        # Wait a bit for scanners to be discovered
        logger.info("Waiting for scanners to be discovered...")
        time.sleep(3.0)
        
        # Try to connect to any available scanner
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanner found. Please make sure the UnLook scanner server is running.")
            return
        
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} at {scanner_info.host}:{scanner_info.port}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Run one example at a time (comment out the ones you don't want to run)
        
        # Example 1: Scan with different quality presets
        scan_with_quality_preset(client, ImageQualityPreset.LOW, "quality_scan")
        #scan_with_quality_preset(client, ImageQualityPreset.MEDIUM, "quality_scan")
        #scan_with_quality_preset(client, ImageQualityPreset.HIGH, "quality_scan")
        
        # Example 2: Scan with different image formats
        #scan_with_image_format(client, CompressionFormat.JPEG, "format_scan")
        #scan_with_image_format(client, CompressionFormat.PNG, "format_scan")
        
        # Example 3: Scan with different color modes
        #scan_with_color_mode(client, ColorMode.COLOR, "color_scan")
        #scan_with_color_mode(client, ColorMode.GRAYSCALE, "color_scan")
        
        # Example 4: Scan with crop region
        #scan_with_crop_region(client, "crop_scan")
        
        # Example 5: Scan with image processing options
        #scan_with_image_processing(client, "processing_scan")
        
        logger.info("All examples completed")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Always disconnect
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()