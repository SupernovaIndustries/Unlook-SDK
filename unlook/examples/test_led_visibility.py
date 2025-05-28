#!/usr/bin/env python3
"""
Test LED visibility in IR cameras to choose best color for scanning.
"""

import time
import cv2
import logging
import numpy as np
from unlook import UnlookClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = None
    
    try:
        # Connect
        client = UnlookClient(auto_discover=False)
        client.start_discovery()
        time.sleep(5)
        
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found!")
            return
            
        if not client.connect(scanners[0]):
            logger.error("Failed to connect!")
            return
            
        logger.info("Connected!")
        
        # Get camera
        cameras = client.camera.get_cameras()
        if not cameras:
            logger.error("No cameras found!")
            return
            
        camera_id = cameras[0]['id']
        client.projector.set_test_pattern_mode()
        
        # Test each color for IR camera visibility
        colors = [
            ("Black (Off)", "Black"),
            ("Red", "Red"), 
            ("Blue", "Blue"),
            ("White", "White")
        ]
        
        results = {}
        
        for color_name, color_value in colors:
            logger.info(f"\n=== Testing {color_name} ===")
            
            # Set the color
            client.projector.show_solid_field(color_value)
            time.sleep(1)  # Let it stabilize
            
            # Capture frame for analysis
            frame_captured = False
            brightness_values = []
            
            def analyze_frame(frame, metadata):
                nonlocal frame_captured, brightness_values
                
                if not frame_captured:
                    # Analyze brightness
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    max_brightness = np.max(gray)
                    
                    brightness_values = [mean_brightness, max_brightness]
                    results[color_name] = {
                        'mean': mean_brightness,
                        'max': max_brightness,
                        'color_value': color_value
                    }
                    
                    # Display info
                    cv2.putText(frame, f"Testing: {color_name}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Mean: {mean_brightness:.1f}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Max: {max_brightness:.1f}", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("LED Visibility Test", frame)
                    frame_captured = True
                
                return cv2.waitKey(30) != ord('q')
            
            # Stream briefly to capture frame
            client.stream.start(camera_id, analyze_frame)
            start_time = time.time()
            while not frame_captured and time.time() - start_time < 3:
                time.sleep(0.1)
            client.stream.stop()
        
        # Turn off
        client.projector.show_solid_field("Black")
        
        # Analyze results
        logger.info("\n=== RESULTS ===")
        best_contrast = None
        best_brightness = None
        best_contrast_value = 0
        
        black_mean = results.get("Black (Off)", {}).get('mean', 0)
        
        for color_name, data in results.items():
            if color_name != "Black (Off)":
                mean = data['mean']
                contrast = mean - black_mean
                logger.info(f"{color_name:15} - Mean: {mean:6.1f}, Contrast: {contrast:6.1f}")
                
                if contrast > best_contrast_value:
                    best_contrast_value = contrast
                    best_contrast = color_name
                    
                if not best_brightness or mean > results[best_brightness]['mean']:
                    best_brightness = color_name
        
        logger.info(f"\nBest contrast: {best_contrast} ({best_contrast_value:.1f})")
        logger.info(f"Brightest: {best_brightness}")
        
        # Show final recommendation
        recommended = best_contrast if best_contrast_value > 10 else best_brightness
        logger.info(f"\nRecommended for scanning: {recommended}")
        
        # Quick demo of recommended color
        if recommended:
            logger.info(f"\nDemonstrating {recommended} lines pattern...")
            client.projector.show_vertical_lines(
                foreground_color=results[recommended]['color_value'],
                background_color="Black",
                foreground_width=10,
                background_width=30
            )
            
            def show_demo(frame, metadata):
                cv2.putText(frame, f"Recommended: {recommended}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to exit", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Recommended Pattern", frame)
                return cv2.waitKey(30) != ord('q')
            
            client.stream.start(camera_id, show_demo)
            while True:
                time.sleep(0.1)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        if client:
            client.projector.show_solid_field("Black")
            cv2.destroyAllWindows()
            client.disconnect()


if __name__ == "__main__":
    main()