#!/usr/bin/env python3
"""
Simple Focus Adjustment Tool (OpenCV-based)

A simplified version of the focus adjustment tool using only OpenCV for display.
Perfect for quick testing without GUI dependencies.

Usage:
    python focus_adjustment_simple.py
    
Controls:
    Q - Quit
    SPACE - Toggle projector
    R - Reset focus history
"""

import logging
import time
import cv2
import numpy as np

from unlook.client.scanner.scanner import UnlookClient, FocusAssessment
from unlook.client.camera.camera_config import CameraConfig, CompressionFormat, ColorMode
from unlook.core.discovery import DiscoveryService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_focus_overlay(image, focus_result, camera_name):
    """Create overlay with focus information on image."""
    overlay = image.copy()
    
    if focus_result:
        score = focus_result.get('smoothed_score', 0)
        quality = focus_result.get('quality', 'unknown')
        status = focus_result.get('status', 'unknown')
        
        # Color based on quality
        if quality == 'excellent':
            color = (0, 255, 0)  # Green
        elif quality == 'good':
            color = (0, 255, 255)  # Yellow
        elif quality == 'fair':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Add text overlay
        cv2.putText(overlay, f"{camera_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(overlay, f"Focus: {score:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(overlay, f"Quality: {quality}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Focus indicator bar
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 10, 100
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Focus level bar
        focus_width = int((score / 200.0) * bar_width)  # Normalize to 0-200
        focus_width = min(focus_width, bar_width)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + focus_width, bar_y + bar_height), color, -1)
        
        # Threshold line
        threshold_x = int((focus_result.get('threshold', 100) / 200.0) * bar_width)
        cv2.line(overlay, (bar_x + threshold_x, bar_y), (bar_x + threshold_x, bar_y + bar_height), (255, 255, 255), 2)
    
    return overlay


def main():
    """Main function."""
    logger.info("Starting UnLook Simple Focus Adjustment Tool")
    
    # Initialize
    client = UnlookClient("SimpleFocusTool", auto_discover=True)
    focus_assessment = FocusAssessment()
    
    # Connect to scanner (using same pattern as enhanced_scanner)
    logger.info("Connecting to scanner...")
    time.sleep(3)  # Wait for discovery (same as enhanced_scanner)
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found. Make sure scanner server is running with:")
        logger.error("python unlook/server_bootstrap.py --enable-protocol-v2 --enable-pattern-preprocessing --enable-sync")
        return
    
    # Connect to first available scanner
    scanner_info = scanners[0]
    logger.info(f"Connecting to: {scanner_info.name}")
    
    if not client.connect(scanner_info):
        logger.error("Failed to connect to scanner")
        return
    
    logger.info("Connected to scanner successfully")
    
    # For this simple tool, we'll use capture_multi with direct parameters
    # instead of trying to configure cameras individually
    logger.info("Using direct capture parameters for better compatibility")
    
    # Pattern cycling
    projector_enabled = True
    pattern_cycle = [
        ('horizontal_lines', 20),
        ('vertical_lines', 20),
        ('horizontal_lines', 10),
        ('vertical_lines', 10),
    ]
    current_pattern = 0
    last_pattern_time = time.time()
    pattern_interval = 2.0  # seconds
    
    # Start projector with vertical lines (like in the working example)
    if projector_enabled:
        try:
            success = client.projector.show_vertical_lines(
                foreground_color="White",
                background_color="Black", 
                foreground_width=20,
                background_width=20
            )
            logger.info(f"Projector started: {success}")
        except Exception as e:
            logger.error(f"Error starting projector: {e}")
            projector_enabled = False
    
    logger.info("Focus adjustment tool started. Controls: Q=quit, SPACE=toggle projector, R=reset history")
    
    try:
        while True:
            # Capture images using simplified approach
            try:
                # Get camera list first (same as enhanced_scanner)
                cameras = client.camera.get_cameras()
                if len(cameras) < 2:
                    logger.error("Need at least 2 cameras")
                    continue
                    
                # Use same camera selection as enhanced_scanner
                camera_ids = [cam['id'] for cam in cameras[:2]]
                logger.debug(f"Using cameras: {camera_ids}")
                
                # Capture using same method as enhanced_scanner
                images = client.camera.capture_multi(camera_ids)
                
                if images and len(images) == 2:
                    # Extract left and right images (same as enhanced_scanner)
                    left_image = images[camera_ids[0]]
                    right_image = images[camera_ids[1]]
                    
                    # Check if images are valid
                    if left_image is None or right_image is None:
                        logger.warning("One or more images are None, skipping frame")
                        continue
                    
                    # Assess focus
                    left_focus = focus_assessment.assess_camera_focus(left_image, camera_ids[0])
                    right_focus = focus_assessment.assess_camera_focus(right_image, camera_ids[1])
                    
                    # Assess projector focus from left camera (if projector is on)
                    projector_focus = {}
                    if projector_enabled:
                        projector_focus = focus_assessment.assess_projector_focus(left_image, 'lines')
                    
                    # Create display images with overlays
                    left_display = create_focus_overlay(left_image, left_focus, f"Camera {camera_ids[0]}")
                    right_display = create_focus_overlay(right_image, right_focus, f"Camera {camera_ids[1]}")
                    
                    # Combine images side by side
                    combined = np.hstack([left_display, right_display])
                    
                    # Add overall status banner
                    overall_status = focus_assessment.get_overall_focus_status()
                    status_text = overall_status['message']
                    
                    # Status color
                    if overall_status['status'] == 'ready':
                        status_color = (0, 255, 0)  # Green
                    elif overall_status['status'] == 'poor':
                        status_color = (0, 0, 255)  # Red
                    else:
                        status_color = (0, 255, 255)  # Yellow
                    
                    # Add status banner
                    banner_height = 50
                    banner = np.zeros((banner_height, combined.shape[1], 3), dtype=np.uint8)
                    cv2.putText(banner, status_text, (10, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                    # Projector status
                    proj_text = f"Projector: {'ON' if projector_enabled else 'OFF'}"
                    cv2.putText(banner, proj_text, (combined.shape[1] - 200, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add projector focus info if available
                    if projector_focus and projector_enabled:
                        proj_score = projector_focus.get('smoothed_score', 0)
                        proj_quality = projector_focus.get('quality', 'unknown')
                        proj_info = f"Proj Focus: {proj_score:.1f} ({proj_quality})"
                        cv2.putText(banner, proj_info, (combined.shape[1] - 400, 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Combine banner and images
                    final_display = np.vstack([banner, combined])
                    
                    # Display
                    cv2.imshow('UnLook Focus Adjustment Tool', final_display)
                    
                    # Handle pattern cycling
                    if projector_enabled and time.time() - last_pattern_time > pattern_interval:
                        current_pattern = (current_pattern + 1) % len(pattern_cycle)
                        pattern_type, spacing = pattern_cycle[current_pattern]
                        
                        if pattern_type == 'horizontal_lines':
                            # Use solid field for horizontal (simpler pattern)
                            client.projector.show_solid_field("White")
                            time.sleep(0.5)
                            client.projector.show_solid_field("Black")
                        else:
                            client.projector.show_vertical_lines(
                                foreground_color="White",
                                background_color="Black",
                                foreground_width=spacing,
                                background_width=spacing
                            )
                        
                        last_pattern_time = time.time()
                        logger.debug(f"Switched to {pattern_type} pattern with spacing {spacing}")
                
                else:
                    logger.warning(f"Failed to capture images - got {len(images) if images else 0} images, expected 2")
                    if images:
                        logger.debug(f"Image keys: {list(images.keys())}")
                    
            except Exception as e:
                logger.error(f"Error during capture: {e}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit requested")
                break
            elif key == ord(' '):  # Space - toggle projector
                projector_enabled = not projector_enabled
                if projector_enabled:
                    logger.info("Projector enabled")
                    client.projector.show_vertical_lines(
                        foreground_color="White",
                        background_color="Black",
                        foreground_width=20,
                        background_width=20
                    )
                    current_pattern = 0
                    last_pattern_time = time.time()
                else:
                    logger.info("Projector disabled")
                    client.projector.show_solid_field("Black")  # Turn off by showing black
            elif key == ord('r') or key == ord('R'):
                logger.info("Resetting focus history")
                focus_assessment.focus_history = {'left': [], 'right': [], 'projector': []}
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        try:
            client.projector.show_solid_field("Black")  # Turn off projector
            client.disconnect()
        except:
            pass
        
        cv2.destroyAllWindows()
        logger.info("Focus adjustment tool stopped")


if __name__ == "__main__":
    main()