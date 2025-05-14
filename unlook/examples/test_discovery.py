#!/usr/bin/env python3
"""Test scanner discovery."""

import time
import logging
from unlook import UnlookClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to test discovery."""
    logger.info("Starting scanner discovery test...")
    
    # Create client
    client = UnlookClient()
    
    # Start discovery
    client.start_discovery()
    
    # Wait for scanners
    logger.info("Waiting 10 seconds for scanners to be discovered...")
    time.sleep(10)
    
    # Get discovered scanners
    scanners = client.get_discovered_scanners()
    
    if scanners:
        logger.info(f"Found {len(scanners)} scanner(s):")
        for scanner in scanners:
            logger.info(f"  - {scanner.name} ({scanner.uuid}) at {scanner.host}:{scanner.port}")
    else:
        logger.info("No scanners found.")
        logger.info("Please check that:")
        logger.info("  1. The scanner hardware is powered on")
        logger.info("  2. The scanner server is running on the device")
        logger.info("  3. Both devices are on the same network")
        logger.info("  4. mDNS/Zeroconf is not blocked by firewall")
    
    # Stop discovery
    client.stop_discovery()
    logger.info("Discovery stopped.")

if __name__ == "__main__":
    main()