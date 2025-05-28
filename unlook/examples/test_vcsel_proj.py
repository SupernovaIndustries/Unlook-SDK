#!/usr/bin/env python3
"""
Test VCSEL IR projector with green line patterns.

This script connects to the Unlook scanner, opens a camera stream,
and projects green lines for 10 seconds to test the VCSEL IR projector.
Since the VCSEL IR is mounted where the green LED was, projecting 
green patterns will activate the IR projector.
"""

import time
import cv2
import logging
from unlook import UnlookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test VCSEL projector with green line patterns."""
    client = None
    streaming = False
    
    try:
        # Create client with auto-discovery disabled initially
        client = UnlookClient(auto_discover=False)
        
        # Start discovery
        client.start_discovery()
        logger.info("Discovering scanners for 5 seconds...")
        time.sleep(5)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Check that your hardware is connected and powered on.")
            return
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner.")
            return
        
        logger.info(f"Successfully connected to scanner: {scanner_info.name}")
        
        # Get the first camera
        cameras = client.camera.get_cameras()
        if not cameras:
            logger.error("No cameras found!")
            return
            
        camera_id = cameras[0]['id']
        logger.info(f"Using camera: {camera_id}")
        
        # Set projector to test pattern mode
        logger.info("Setting projector to test pattern mode...")
        client.projector.set_test_pattern_mode()
        
        # Project vertical green lines
        # Since VCSEL IR is mounted at green LED position, this will activate IR
        logger.info("Projecting vertical green lines...")
        
        # Use the show_vertical_lines method directly
        if not client.projector.show_vertical_lines(
            foreground_color="Green",  # Green channel will activate VCSEL IR
            background_color="Black",
            foreground_width=10,       # 10 pixel wide lines
            background_width=30        # 30 pixel spacing
        ):
            logger.error("Failed to project pattern!")
            return
            
        # Frame counter and timing
        frame_count = 0
        start_time = time.time()
        
        # Define callback for streaming
        def show_frame(frame, metadata):
            nonlocal frame_count, start_time, streaming
            
            frame_count += 1
            elapsed = time.time() - start_time
            
            # Add text overlay
            cv2.putText(frame, f"VCSEL IR Test - Green Lines", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow("VCSEL IR Test", frame)
            
            # Stop after 10 seconds or if ESC is pressed
            if elapsed >= 10.0 or cv2.waitKey(1) == 27:
                logger.info(f"Stopping test after {elapsed:.1f} seconds")
                streaming = False
                return False
                
            return True
        
        # Start streaming
        logger.info("Starting camera stream...")
        streaming = True
        
        # Start the stream and keep it running
        try:
            client.stream.start(camera_id, show_frame)
            
            # Keep the main thread alive while streaming
            while streaming and (time.time() - start_time) < 10:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            
        logger.info(f"Test completed! Captured {frame_count} frames")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        
    finally:
        # Cleanup
        if client:
            try:
                # Turn off projector
                logger.info("Turning off projector...")
                client.projector.show_solid_field("Black")
                
                # Stop streaming if active
                if streaming:
                    client.stream.stop()
                    cv2.destroyAllWindows()
                
                # Disconnect
                client.disconnect()
                logger.info("Disconnected from scanner")
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()