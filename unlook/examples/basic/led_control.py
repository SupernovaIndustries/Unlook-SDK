#!/usr/bin/env python3
"""
LED Control Example

This example demonstrates how to control the AS1170 LED flood illuminator 
on the UnLook scanner. The scanner contains a dual-channel LED with the
following configuration:

- LED1: Always set to 0 intensity (prevented from use to avoid overheating)
- LED2: Adjustable intensity from 0-450mA

This script shows various ways to control LED2 while ensuring LED1 remains off.
"""

import time
import logging
from unlook.client import UnlookClient
from unlook.client.projector import LEDController

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Connect to scanner
    client = UnlookClient()
    logger.info("Discovering scanners...")
    client.start_discovery()
    
    # Wait for discovery
    time.sleep(5)
    
    # Get discovered scanners
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found")
        return
    
    # Connect to the first scanner
    scanner_info = scanners[0]
    logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
    
    if not client.connect(scanner_info):
        logger.error("Failed to connect to scanner")
        return
    
    logger.info(f"Successfully connected to scanner: {scanner_info.name}")
    
    # Create LED controller
    led = LEDController(client)
    
    try:
        # Check if LED is available
        status = led.get_status()
        if not status['led_available']:
            logger.error("LED control not available on this scanner")
            return
        
        logger.info("LED control available, performing demo...")
        
        # Example 1: Turn on LED at full intensity
        logger.info("Setting LED2 to full intensity (LED1 remains at 0)...")
        led.set_intensity(0, 450)  # LED1 will automatically be set to 0
        
        # Example 2: Gradually increase LED2 intensity
        logger.info("Gradually increasing LED2 intensity...")
        for intensity in range(0, 451, 50):
            led.set_intensity(0, intensity)  # LED1 remains at 0
            logger.info(f"LED2 intensity: {intensity}mA")
            time.sleep(0.5)
        
        # Example 3: Pulse mode
        logger.info("Pulse mode demonstration...")
        for _ in range(3):
            # Flash at full intensity for 0.3 seconds
            led.pulse(duration=0.3, intensity=450)
            time.sleep(0.7)
        
        # Example 4: Toggle the LED
        logger.info("Toggling LED...")
        for _ in range(3):
            led.toggle()
            status = led.get_status()
            logger.info(f"LED active: {status['led_active']}")
            time.sleep(1)
        
        # Example 5: Turn off then on again
        logger.info("Turning LED off...")
        led.turn_off()
        time.sleep(1)
        
        logger.info("Turning LED on with medium intensity...")
        led.turn_on(intensity=300)  # LED1 remains at 0, LED2 at 300mA
        time.sleep(2)
        
        # Final cleanup
        logger.info("Turning LED off...")
        led.turn_off()
        
    finally:
        # Ensure LED is off and client is disconnected
        if led.get_status()['led_active']:
            led.turn_off()
        
        client.disconnect()
        client.stop_discovery()
        logger.info("Demo completed")

if __name__ == "__main__":
    main()