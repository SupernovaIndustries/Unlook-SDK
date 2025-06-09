#!/usr/bin/env python3
"""
Test LED control for calibration capture.

This script demonstrates:
1. V2 protocol client initialization
2. AS1170 LED controller usage
3. Basic camera capture with LED illumination

Usage:
    python test_led_calibration.py
"""

import sys
import time
import logging
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from unlook.client.scanner.scanner import UnlookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_led_calibration")

def test_led_and_capture():
    """Test LED control and camera capture."""
    
    # Create client with auto-discovery
    print("Creating UnLook client with v2 protocol...")
    client = UnlookClient("LEDCalibrationTest", auto_discover=True)
    
    # Wait for discovery
    print("Discovering scanners...")
    time.sleep(3)
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found on network")
        return False
    
    print(f"Found {len(scanners)} scanner(s)")
    scanner = scanners[0]
    print(f"Connecting to: {scanner.name} at {scanner.host}")
    
    # Connect
    if not client.connect(scanner):
        logger.error("Failed to connect to scanner")
        return False
    
    print("Connected successfully!")
    
    try:
        # Test LED control
        print("\n=== Testing LED Control ===")
        
        # Check LED status
        led_status = client.projector.led_controller.get_status()
        print(f"LED Status: {led_status}")
        
        # Test different LED intensities
        test_intensities = [0, 100, 200, 300, 400]
        
        for intensity in test_intensities:
            print(f"\nTesting LED intensity: {intensity}mA")
            
            # Set LED intensity (LED1=0 for flood only, LED2 for illumination)
            success = client.projector.led_set_intensity(led1_mA=0, led2_mA=intensity)
            if success:
                print(f"  ✓ LED set to {intensity}mA")
            else:
                print(f"  ✗ Failed to set LED to {intensity}mA")
            
            # Capture image with this LED setting
            print("  Capturing image...")
            cameras = client.camera.get_cameras()
            if cameras:
                camera_id = cameras[0]['id']
                try:
                    image = client.camera.capture(camera_id)
                    if image is not None:
                        # Analyze image brightness
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                        mean_brightness = np.mean(gray)
                        print(f"  ✓ Image captured - Mean brightness: {mean_brightness:.1f}")
                        
                        # Save sample image
                        filename = f"led_test_{intensity}mA.jpg"
                        cv2.imwrite(filename, image)
                        print(f"  ✓ Saved: {filename}")
                    else:
                        print("  ✗ Failed to capture image")
                except Exception as e:
                    print(f"  ✗ Capture error: {e}")
            
            time.sleep(1)  # Brief pause between tests
        
        # Test LED pulse
        print("\n=== Testing LED Pulse ===")
        success = client.projector.led_controller.pulse(duration=0.5, intensity=450)
        if success:
            print("✓ LED pulse successful")
        else:
            print("✗ LED pulse failed")
        
        # Turn off LED
        print("\n=== Turning off LED ===")
        success = client.projector.led_off()
        if success:
            print("✓ LED turned off")
        else:
            print("✗ Failed to turn off LED")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            client.projector.led_off()  # Ensure LED is off
            client.disconnect()
        except:
            pass

def main():
    """Main function."""
    print("LED Calibration Test")
    print("=" * 50)
    print("This script tests LED control for calibration")
    print("Make sure the scanner server is running with:")
    print("  python unlook/server_bootstrap.py")
    print("=" * 50)
    
    success = test_led_and_capture()
    
    if success:
        print("\n✅ LED test completed successfully!")
        return 0
    else:
        print("\n❌ LED test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())