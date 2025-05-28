#!/usr/bin/env python3
"""
VCSEL IR LED Debug Test Script.

This script helps debug why the VCSEL IR LED is not working.
It tests different configurations and provides diagnostic information.

Debug steps:
1. Test all color channels individually
2. Test with maximum intensity (White color)
3. Test with different patterns
4. Check camera IR sensitivity
5. Monitor power consumption if possible
"""

import time
import cv2
import logging
import numpy as np
from unlook import UnlookClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def enhance_ir_visibility(frame):
    """Enhance IR visibility in the image.
    
    Args:
        frame: Input frame
        
    Returns:
        Enhanced frame
    """
    # Convert to grayscale for better IR visibility
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    enhanced = cv2.equalizeHist(gray)
    
    # Convert back to BGR for display
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Create a side-by-side comparison
    combined = np.hstack([frame, enhanced_bgr])
    
    return combined


def test_pattern_sequence(client, camera_id):
    """Test different patterns to activate VCSEL.
    
    Args:
        client: UnlookClient instance
        camera_id: Camera ID to use
    """
    patterns = [
        {
            'name': 'Solid Green',
            'method': 'show_solid_field',
            'params': {'color': 'Green'}
        },
        {
            'name': 'Solid White (All channels)',
            'method': 'show_solid_field',
            'params': {'color': 'White'}
        },
        {
            'name': 'Green Lines',
            'method': 'show_vertical_lines',
            'params': {
                'foreground_color': 'Green',
                'background_color': 'Black',
                'foreground_width': 50,
                'background_width': 50
            }
        },
        {
            'name': 'White Lines',
            'method': 'show_vertical_lines',
            'params': {
                'foreground_color': 'White',
                'background_color': 'Black',
                'foreground_width': 50,
                'background_width': 50
            }
        },
        {
            'name': 'Green Checkerboard',
            'method': 'show_checkerboard',
            'params': {
                'color1': 'Green',
                'color2': 'Black',
                'width': 50,
                'height': 50
            }
        },
        {
            'name': 'Color Bars (Test all channels)',
            'method': 'show_colorbars',
            'params': {}
        }
    ]
    
    logger.info("Starting pattern sequence test...")
    
    for pattern in patterns:
        logger.info(f"\nTesting pattern: {pattern['name']}")
        logger.info(f"Method: {pattern['method']}")
        logger.info(f"Parameters: {pattern['params']}")
        
        # Set test pattern mode
        client.projector.set_test_pattern_mode()
        
        # Apply pattern
        method = getattr(client.projector, pattern['method'])
        success = method(**pattern['params'])
        
        if success:
            logger.info(f"✓ Pattern '{pattern['name']}' activated successfully")
        else:
            logger.error(f"✗ Failed to activate pattern '{pattern['name']}'")
        
        # Capture and analyze frames
        frame_count = 0
        pattern_active = True
        start_time = time.time()
        
        def analyze_frame(frame, metadata):
            nonlocal frame_count, pattern_active
            
            frame_count += 1
            
            # Enhance IR visibility
            enhanced = enhance_ir_visibility(frame)
            
            # Add text overlays
            cv2.putText(enhanced, f"Pattern: {pattern['name']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(enhanced, "Original | IR Enhanced", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Calculate mean brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            cv2.putText(enhanced, f"Mean Brightness: {mean_brightness:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Show frame
            cv2.imshow("VCSEL Debug Test", enhanced)
            
            # Run for 3 seconds per pattern
            if time.time() - start_time > 3.0:
                pattern_active = False
                return False
            
            # Check for 'q' to skip
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pattern_active = False
                return False
                
            return True
        
        # Start streaming for this pattern
        client.stream.start(camera_id, analyze_frame)
        
        while pattern_active:
            time.sleep(0.1)
        
        logger.info(f"Captured {frame_count} frames for pattern '{pattern['name']}'")
        
        # Small delay between patterns
        time.sleep(0.5)
    
    # Turn off projector
    client.projector.show_solid_field("Black")


def check_hardware_info(client):
    """Check and log hardware information.
    
    Args:
        client: UnlookClient instance
    """
    logger.info("\n=== Hardware Information ===")
    
    # Get projector info
    try:
        # This might not exist, but try anyway
        logger.info("Checking projector status...")
        client.projector.set_test_pattern_mode()
        logger.info("✓ Projector communication OK")
    except Exception as e:
        logger.error(f"✗ Projector error: {e}")
    
    # Get camera info
    cameras = client.camera.get_cameras()
    logger.info(f"Number of cameras: {len(cameras)}")
    for cam in cameras:
        logger.info(f"  - Camera: {cam}")


def main():
    """Main debug function."""
    client = None
    
    try:
        # Create client
        client = UnlookClient(auto_discover=False)
        
        # Start discovery
        client.start_discovery()
        logger.info("Discovering scanners...")
        time.sleep(5)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found!")
            return
        
        # Connect to scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to: {scanner_info.name}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect!")
            return
        
        logger.info("✓ Connected successfully")
        
        # Check hardware
        check_hardware_info(client)
        
        # Get camera
        cameras = client.camera.get_cameras()
        if not cameras:
            logger.error("No cameras found!")
            return
            
        camera_id = cameras[0]['id']
        
        # Instructions
        logger.info("\n=== VCSEL Debug Test ===")
        logger.info("This test will cycle through different patterns")
        logger.info("Watch for any IR illumination in the enhanced view")
        logger.info("Press 'q' to skip to next pattern")
        logger.info("\nStarting in 3 seconds...")
        time.sleep(3)
        
        # Run pattern tests
        test_pattern_sequence(client, camera_id)
        
        logger.info("\n=== Test Complete ===")
        logger.info("\nIf you didn't see any IR illumination, check:")
        logger.info("1. VCSEL power connection (3.2V)")
        logger.info("2. VCSEL ground connection")
        logger.info("3. VCSEL control signal from green channel")
        logger.info("4. VCSEL orientation (anode/cathode)")
        logger.info("5. Current limiting resistor (if needed)")
        logger.info("6. Try external IR camera/phone to verify")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        if client:
            try:
                # Turn off projector
                client.projector.show_solid_field("Black")
                
                # Stop streaming
                client.stream.stop()
                cv2.destroyAllWindows()
                
                # Disconnect
                client.disconnect()
                logger.info("Disconnected")
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()