#!/usr/bin/env python3
"""
Keyboard control for Unlook projector with RGB/VCSEL channels.

This example allows controlling the projector's color channels using keyboard
while streaming camera footage. The projector displays static vertical lines
with selectable color channel.

Keyboard Controls:
- '0' or 'b': Black (Off)
- '1' or 'r': Red
- '2' or 'g': Green (VCSEL IR)
- '3' or 'l': Blue
- '4' or 'w': White (All channels)
- '5' or 'c': Cyan (Green + Blue)
- '6' or 'm': Magenta (Red + Blue)
- '7' or 'y': Yellow (Red + Green)
- 'q': Quit

Note: The VCSEL IR is connected to the Green channel.
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


class ProjectorController:
    """Simple projector controller."""
    
    def __init__(self, client):
        """Initialize the controller.
        
        Args:
            client: UnlookClient instance
        """
        self.client = client
        self.current_color = "Green"  # Default to Green (VCSEL)
        
        # Color mapping
        self.color_map = {
            '0': 'Black',
            'b': 'Black',
            '1': 'Red',
            'r': 'Red',
            '2': 'Green',
            'g': 'Green',
            '3': 'Blue',
            'l': 'Blue',  # 'l' for bLue (b is taken)
            '4': 'White',
            'w': 'White',
            '5': 'Cyan',
            'c': 'Cyan',
            '6': 'Magenta',
            'm': 'Magenta',
            '7': 'Yellow',
            'y': 'Yellow'
        }
        
        # Initialize pattern
        self.update_pattern()
        
    def update_pattern(self):
        """Update the projector pattern with current color."""
        try:
            # Set projector to test pattern mode
            self.client.projector.set_test_pattern_mode()
            
            # Project vertical lines with selected color
            success = self.client.projector.show_vertical_lines(
                foreground_color=self.current_color,
                background_color="Black",
                foreground_width=10,  # 10 pixel wide lines
                background_width=30   # 30 pixel spacing
            )
            
            if success:
                logger.info(f"Pattern updated: {self.current_color} lines")
            else:
                logger.error("Failed to update pattern!")
                
        except Exception as e:
            logger.error(f"Failed to update pattern: {e}")
    
    def set_color(self, color_name):
        """Set the projector color.
        
        Args:
            color_name: Name of the color (e.g., 'Red', 'Green', 'Blue')
        """
        if color_name != self.current_color:
            self.current_color = color_name
            self.update_pattern()
    
    def handle_key(self, key):
        """Handle keyboard input.
        
        Args:
            key: The pressed key character
            
        Returns:
            True to continue, False to quit
        """
        if key == 'q':
            return False
            
        # Check if key maps to a color
        key_char = chr(key) if key < 256 else None
        if key_char and key_char in self.color_map:
            new_color = self.color_map[key_char]
            self.set_color(new_color)
            
        return True


def main():
    """Main function to run projector control."""
    client = None
    streaming = False
    
    try:
        # Create client
        client = UnlookClient(auto_discover=False)
        
        # Start discovery
        client.start_discovery()
        logger.info("Discovering scanners for 5 seconds...")
        time.sleep(5)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Check that your hardware is connected.")
            return
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner.")
            return
        
        logger.info("Successfully connected!")
        
        # Get the first camera
        cameras = client.camera.get_cameras()
        if not cameras:
            logger.error("No cameras found!")
            return
            
        camera_id = cameras[0]['id']
        logger.info(f"Using camera: {camera_id}")
        
        # Create projector controller
        controller = ProjectorController(client)
        
        # Frame counter
        frame_count = 0
        fps_timer = time.time()
        fps_counter = 0
        current_fps = 0
        
        # Define callback for streaming
        def show_frame(frame, metadata):
            nonlocal frame_count, streaming, fps_timer, fps_counter, current_fps
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Add overlay information
            cv2.putText(frame, "Projector Control - Keyboard Controls", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Current Color: {controller.current_color}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw control instructions
            y_pos = 130
            instructions = [
                "Controls:",
                "0/b: Black (Off)",
                "1/r: Red",
                "2/g: Green (VCSEL)",
                "3/l: Blue",
                "4/w: White",
                "5/c: Cyan",
                "6/m: Magenta",
                "7/y: Yellow",
                "q: Quit"
            ]
            
            for instruction in instructions:
                cv2.putText(frame, instruction, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
            
            # Highlight current color
            color_bgr = {
                'Black': (0, 0, 0),
                'Red': (0, 0, 255),
                'Green': (0, 255, 0),
                'Blue': (255, 0, 0),
                'White': (255, 255, 255),
                'Cyan': (255, 255, 0),
                'Magenta': (255, 0, 255),
                'Yellow': (0, 255, 255)
            }
            
            # Draw color indicator
            if controller.current_color in color_bgr:
                color = color_bgr[controller.current_color]
                if controller.current_color == 'Black':
                    # Draw white outline for black
                    cv2.rectangle(frame, (400, 50), (450, 100), (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (400, 50), (450, 100), color, -1)
                    cv2.rectangle(frame, (400, 50), (450, 100), (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Unlook Projector Control", frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # A key was pressed
                if not controller.handle_key(key):
                    logger.info("Quit requested by user")
                    streaming = False
                    return False
                    
            return True
        
        # Start streaming
        logger.info("Starting camera stream...")
        logger.info("\nPress number keys or letter keys to change colors")
        logger.info("Press 'q' to quit\n")
        streaming = True
        
        # Start the stream
        try:
            client.stream.start(camera_id, show_frame)
            
            # Keep the main thread alive while streaming
            while streaming:
                time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            
        logger.info(f"Stream ended. Total frames: {frame_count}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        # Cleanup
        if client:
            try:
                # Turn off projector
                logger.info("Turning off projector...")
                client.projector.show_solid_field("Black")
                
                # Stop streaming
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