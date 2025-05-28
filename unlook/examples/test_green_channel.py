#!/usr/bin/env python3
"""
Test specifically the green channel with various intensities and patterns.
Useful for debugging VCSEL IR on green channel.
"""

import time
import cv2
import logging
from unlook import UnlookClient
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = None
    
    try:
        # Connect
        client = UnlookClient(auto_discover=False)
        client.start_discovery()
        logger.info("Discovering...")
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
        
        # Set test pattern mode
        client.projector.set_test_pattern_mode()
        
        # Test different approaches
        tests = [
            ("1. Solid Green - Default intensity", 
             lambda: client.projector.show_solid_field("Green")),
            
            ("2. Green at 100% current", 
             lambda: (client.projector.set_led_current(0, 1023, 0),
                     client.projector.show_solid_field("Green"))),
            
            ("3. White (all channels) to compare",
             lambda: (client.projector.set_led_current(1023, 1023, 1023),
                     client.projector.show_solid_field("White"))),
            
            ("4. Green lines pattern",
             lambda: client.projector.show_vertical_lines("Green", "Black", 50, 50)),
            
            ("5. Green checkerboard",
             lambda: client.projector.show_checkerboard("Green", "Black", 4, 4)),
            
            ("6. Flashing green (PWM simulation)",
             lambda: flash_green(client)),
        ]
        
        # Callback for display
        def show_frame(frame, metadata):
            # Enhance IR visibility
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            
            # Check for IR activity (bright spots)
            max_val = np.max(gray)
            mean_val = np.mean(gray)
            
            # Display info
            cv2.putText(frame, f"Max brightness: {max_val}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mean brightness: {mean_val:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show both original and enhanced
            combined = np.hstack([frame, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)])
            cv2.imshow("VCSEL Test - Original | Enhanced", combined)
            
            return cv2.waitKey(1) != ord('q')
        
        # Run tests
        for test_name, test_func in tests:
            logger.info(f"\n=== {test_name} ===")
            logger.info("Watch the camera feed for IR activity")
            logger.info("Press 'q' to skip to next test")
            
            # Execute test
            test_func()
            
            # Stream for a few seconds
            client.stream.start(camera_id, show_frame)
            time.sleep(5)
            client.stream.stop()
            
            # Brief pause
            client.projector.show_solid_field("Black")
            time.sleep(0.5)
        
        logger.info("\n=== Test complete ===")
        logger.info("\nIf no IR detected, check:")
        logger.info("1. VCSEL needs 1.2A - DLP may only provide 700mA")
        logger.info("2. Try external current source with DLP as trigger")
        logger.info("3. Check with phone camera for any IR emission")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if client:
            client.projector.show_solid_field("Black")
            cv2.destroyAllWindows()
            client.disconnect()


def flash_green(client, frequency=10):
    """Flash green channel at given frequency."""
    period = 1.0 / frequency
    for _ in range(frequency * 2):  # 2 seconds
        client.projector.show_solid_field("Green")
        time.sleep(period / 2)
        client.projector.show_solid_field("Black")
        time.sleep(period / 2)
    return True


if __name__ == "__main__":
    main()