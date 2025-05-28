#!/usr/bin/env python3
"""
Test LED control functionality step by step.
"""

import time
import logging
from unlook import UnlookClient

logging.basicConfig(level=logging.DEBUG)
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
        
        # Set test pattern mode
        client.projector.set_test_pattern_mode()
        
        # Test 1: Get current LED values
        logger.info("\n=== TEST 1: Get current LED values ===")
        current = client.projector.get_led_current()
        if current:
            r, g, b = current
            logger.info(f"Current LED values - R:{r}, G:{g}, B:{b}")
        else:
            logger.error("Failed to get LED current")
        
        # Test 2: Set all LEDs to 50%
        logger.info("\n=== TEST 2: Set all LEDs to 50% ===")
        success = client.projector.set_led_intensity_percent(50, 50, 50)
        logger.info(f"Set 50% intensity: {'Success' if success else 'Failed'}")
        
        # Show white pattern
        client.projector.show_solid_field("White")
        time.sleep(2)
        
        # Test 3: Test each channel individually
        channels = [
            ("Red", 100, 0, 0),
            ("Green", 0, 100, 0),
            ("Blue", 0, 0, 100)
        ]
        
        for name, r, g, b in channels:
            logger.info(f"\n=== TEST 3: {name} channel at 100% ===")
            success = client.projector.set_led_intensity_percent(r, g, b)
            logger.info(f"Set {name} 100%: {'Success' if success else 'Failed'}")
            
            client.projector.show_solid_field(name)
            time.sleep(2)
        
        # Test 4: Gradual increase on green channel
        logger.info("\n=== TEST 4: Gradual increase green channel ===")
        client.projector.show_solid_field("Green")
        
        for intensity in [10, 25, 50, 75, 100]:
            logger.info(f"Green at {intensity}%")
            client.projector.set_led_intensity_percent(0, intensity, 0)
            time.sleep(1)
        
        # Test 5: Enable/disable test
        logger.info("\n=== TEST 5: Enable/disable channels ===")
        
        # Disable red and blue, keep green
        success = client.projector.set_led_enable(False, True, False)
        logger.info(f"Disabled R&B, kept G: {'Success' if success else 'Failed'}")
        client.projector.show_solid_field("White")
        time.sleep(2)
        
        # Re-enable all
        client.projector.set_led_enable(True, True, True)
        
        # Turn off
        logger.info("\n=== Turning off ===")
        client.projector.show_solid_field("Black")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if client:
            client.projector.show_solid_field("Black")
            client.disconnect()


if __name__ == "__main__":
    main()