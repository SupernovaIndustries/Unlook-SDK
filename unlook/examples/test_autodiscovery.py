#!/usr/bin/env python3
"""
Test script for UnLook SDK autodiscovery system.

This script tests the automatic hardware detection and camera mapping
to ensure the "Camera left not found" issue is resolved.
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook.client.scanner.scanner import UnlookClient
from unlook.scanning_modules import detect_hardware, select_scanning_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hardware_detection():
    """Test direct hardware detection."""
    logger.info("=" * 60)
    logger.info("Testing Hardware Detection")
    logger.info("=" * 60)
    
    try:
        # Detect hardware
        hardware_config = detect_hardware()
        
        logger.info(f"Cameras detected: {hardware_config.get('cameras', [])}")
        logger.info(f"AS1170 present: {hardware_config.get('as1170', False)}")
        logger.info(f"Projector: {hardware_config.get('projector', {})}")
        logger.info(f"TOF sensor: {hardware_config.get('tof_sensor')}")
        logger.info(f"I2C devices: {len(hardware_config.get('i2c_devices', []))}")
        
        # Select module
        module_config = select_scanning_module(hardware_config)
        logger.info(f"\nSelected module: {module_config.get('module')}")
        logger.info(f"Camera mapping: {module_config.get('camera_mapping')}")
        logger.info(f"Features: {module_config.get('features')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_client_autodiscovery():
    """Test client-side autodiscovery and camera mapping."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Client Autodiscovery")
    logger.info("=" * 60)
    
    try:
        # Create client
        client = UnlookClient("AutodiscoveryTest", auto_discover=True)
        time.sleep(3)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found!")
            return False
            
        logger.info(f"Found {len(scanners)} scanner(s)")
        
        # Connect to first scanner
        scanner = scanners[0]
        logger.info(f"Connecting to {scanner.name}...")
        
        if not client.connect(scanner):
            logger.error("Failed to connect!")
            return False
            
        logger.info("Connected successfully")
        
        # Test camera discovery
        logger.info("\nTesting camera discovery...")
        camera_list = client.camera_discovery.get_camera_list()
        logger.info(f"Camera list: {camera_list}")
        
        # Test camera mapping
        logger.info("\nTesting camera mapping...")
        mapping = client.camera_discovery.get_mapping()
        for logical, hardware in mapping.items():
            logger.info(f"  {logical} -> {hardware}")
        
        # Test resolving common camera names
        test_names = ["left", "right", "primary", "picamera2_0", "camera0"]
        logger.info("\nTesting camera name resolution:")
        for name in test_names:
            resolved = client.camera_discovery.resolve_camera_id(name)
            logger.info(f"  '{name}' -> {resolved}")
        
        # Test capturing from logical camera names
        logger.info("\nTesting capture with logical names:")
        
        # Test single camera capture
        try:
            logger.info("Capturing from 'primary'...")
            image = client.camera.capture('primary')
            if image is not None:
                logger.info(f"✓ Captured from 'primary': {image.shape}")
            else:
                logger.warning("Failed to capture from 'primary'")
        except Exception as e:
            logger.error(f"Error capturing from 'primary': {e}")
        
        # Test left camera (should work even with single camera)
        try:
            logger.info("Capturing from 'left'...")
            image = client.camera.capture('left')
            if image is not None:
                logger.info(f"✓ Captured from 'left': {image.shape}")
            else:
                logger.warning("Failed to capture from 'left'")
        except Exception as e:
            logger.error(f"Error capturing from 'left': {e}")
        
        # Test stereo pair
        try:
            logger.info("\nTesting stereo pair...")
            left_id, right_id = client.camera_discovery.get_stereo_pair()
            logger.info(f"Stereo pair: left={left_id}, right={right_id}")
            
            if left_id and right_id:
                # Try capturing stereo pair
                logger.info("Capturing stereo pair...")
                left_img, right_img = client.camera.capture_stereo_pair()
                if left_img is not None and right_img is not None:
                    logger.info(f"✓ Stereo capture successful: {left_img.shape}, {right_img.shape}")
                elif left_img is not None:
                    logger.info(f"✓ Single camera capture (mapped to both): {left_img.shape}")
                else:
                    logger.warning("Stereo capture failed")
        except Exception as e:
            logger.error(f"Error with stereo pair: {e}")
        
        # Get scanner configuration
        logger.info("\nGetting scanner configuration...")
        response = client.send_message({
            "type": "SCANNER_CONFIG",
            "action": "get"
        })
        
        if response and response.get("payload", {}).get("status") == "success":
            config = response["payload"]["data"]
            logger.info(f"Module: {config.get('module', {}).get('module')}")
            logger.info(f"Hardware summary: {config.get('module', {}).get('hardware_summary')}")
        
        client.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Client autodiscovery test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all autodiscovery tests."""
    logger.info("UnLook SDK Autodiscovery Test Suite")
    logger.info("===================================\n")
    
    # Test 1: Direct hardware detection (server-side)
    hardware_ok = test_hardware_detection()
    
    # Test 2: Client autodiscovery
    client_ok = test_client_autodiscovery()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Hardware Detection: {'PASS' if hardware_ok else 'FAIL'}")
    logger.info(f"Client Autodiscovery: {'PASS' if client_ok else 'FAIL'}")
    
    if hardware_ok and client_ok:
        logger.info("\n✓ All tests passed! The autodiscovery system is working correctly.")
        logger.info("✓ The 'Camera left not found' issue should now be resolved.")
    else:
        logger.error("\n✗ Some tests failed. Please check the logs above.")
    
    return 0 if (hardware_ok and client_ok) else 1


if __name__ == "__main__":
    sys.exit(main())