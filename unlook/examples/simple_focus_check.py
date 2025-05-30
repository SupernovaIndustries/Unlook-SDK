#!/usr/bin/env python3
"""
Simplified Focus Check Tool - Basic Version

A minimal focus assessment tool that works without projector patterns.
Uses only camera capture and basic focus metrics.

Usage:
    python simple_focus_check.py
    
Controls:
    Q - Quit
    R - Reset focus history
"""

import logging
import time
import cv2
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanner.scanner import UnlookClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleFocusAssessment:
    """Simplified focus assessment without complex features."""
    
    def __init__(self):
        self.focus_history = {}
    
    def assess_focus(self, image, camera_id):
        """Simple Laplacian variance focus assessment."""
        if image is None:
            return {'score': 0, 'quality': 'poor', 'status': 'no_image'}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Simple quality assessment
        if variance > 100:
            quality = 'excellent'
            status = 'ready'
        elif variance > 50:
            quality = 'good' 
            status = 'acceptable'
        elif variance > 20:
            quality = 'fair'
            status = 'needs_adjustment'
        else:
            quality = 'poor'
            status = 'out_of_focus'
        
        # Store in history
        if camera_id not in self.focus_history:
            self.focus_history[camera_id] = []
        
        self.focus_history[camera_id].append(variance)
        if len(self.focus_history[camera_id]) > 10:
            self.focus_history[camera_id].pop(0)
        
        # Smoothed score
        smoothed_score = np.mean(self.focus_history[camera_id])
        
        return {
            'score': variance,
            'smoothed_score': smoothed_score,
            'quality': quality,
            'status': status,
            'threshold': 50
        }


def create_focus_overlay(image, focus_result, camera_name):
    """Create overlay with focus information on image."""
    overlay = image.copy()
    
    if focus_result:
        score = focus_result.get('smoothed_score', 0)
        quality = focus_result.get('quality', 'unknown')
        
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
        
        # Focus level bar (normalize to 0-200 range)
        focus_width = int(min(score / 200.0, 1.0) * bar_width)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + focus_width, bar_y + bar_height), color, -1)
        
        # Threshold line
        threshold_x = int((focus_result.get('threshold', 50) / 200.0) * bar_width)
        cv2.line(overlay, (bar_x + threshold_x, bar_y), (bar_x + threshold_x, bar_y + bar_height), (255, 255, 255), 2)
    
    return overlay


def main():
    """Main function."""
    logger.info("Starting Simple Focus Check Tool")
    
    # Initialize
    client = UnlookClient("SimpleFocusCheck", auto_discover=True)
    focus_assessment = SimpleFocusAssessment()
    
    # Connect to scanner (same pattern as enhanced_scanner)
    logger.info("Connecting to scanner...")
    time.sleep(3)  # Wait for discovery
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found. Make sure scanner server is running.")
        return
    
    # Connect to first available scanner
    scanner_info = scanners[0]
    logger.info(f"Connecting to: {scanner_info.name}")
    
    if not client.connect(scanner_info):
        logger.error("Failed to connect to scanner")
        return
    
    logger.info("Connected to scanner successfully")
    logger.info("Simple focus check started. Controls: Q=quit, R=reset history")
    
    try:
        while True:
            # Capture images
            try:
                # Get camera list (same as enhanced_scanner)
                cameras = client.camera.get_cameras()
                logger.debug(f"Found {len(cameras)} cameras: {[cam.get('id', 'unknown') for cam in cameras]}")
                
                if len(cameras) < 1:
                    logger.warning("No cameras available")
                    time.sleep(1)
                    continue
                
                # Use available cameras
                camera_ids = [cam['id'] for cam in cameras]
                logger.debug(f"Using cameras: {camera_ids}")
                
                # Capture from available cameras
                images = client.camera.capture_multi(camera_ids)
                logger.debug(f"Captured {len(images) if images else 0} images")
                
                if images and len(images) > 0:
                    displays = []
                    
                    # Process each camera
                    for i, camera_id in enumerate(camera_ids):
                        if camera_id in images and images[camera_id] is not None:
                            image = images[camera_id]
                            logger.debug(f"Processing image from {camera_id}: {image.shape}")
                            
                            # Assess focus
                            focus_result = focus_assessment.assess_focus(image, camera_id)
                            
                            # Create display with overlay
                            display = create_focus_overlay(image, focus_result, f"Camera {camera_id}")
                            displays.append(display)
                        else:
                            logger.debug(f"No image from camera {camera_id}")
                    
                    if displays:
                        # Combine images
                        if len(displays) == 1:
                            combined = displays[0]
                        else:
                            # Stack horizontally
                            combined = np.hstack(displays)
                        
                        # Add overall status banner
                        overall_quality = 'good' if all(
                            focus_assessment.assess_focus(images[cid], cid)['quality'] in ['good', 'excellent'] 
                            for cid in camera_ids if cid in images and images[cid] is not None
                        ) else 'needs_adjustment'
                        
                        status_text = f"Overall: {overall_quality} - Cameras: {len(displays)}"
                        
                        # Status color
                        if overall_quality == 'good':
                            status_color = (0, 255, 0)  # Green
                        else:
                            status_color = (0, 255, 255)  # Yellow
                        
                        # Add status banner
                        banner_height = 50
                        banner = np.zeros((banner_height, combined.shape[1], 3), dtype=np.uint8)
                        cv2.putText(banner, status_text, (10, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                        
                        # Combine banner and images
                        final_display = np.vstack([banner, combined])
                        
                        # Display
                        cv2.imshow('Simple Focus Check Tool', final_display)
                    
                else:
                    logger.warning(f"Failed to capture images - got {len(images) if images else 0} images")
                    
            except Exception as e:
                logger.error(f"Error during capture: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Handle keyboard input
            key = cv2.waitKey(100) & 0xFF  # Longer wait for stability
            
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit requested")
                break
            elif key == ord('r') or key == ord('R'):
                logger.info("Resetting focus history")
                focus_assessment.focus_history = {}
            
            time.sleep(0.1)  # Slower refresh for stability
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        try:
            client.disconnect()
        except:
            pass
        
        cv2.destroyAllWindows()
        logger.info("Simple focus check tool stopped")


if __name__ == "__main__":
    main()