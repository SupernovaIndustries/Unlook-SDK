#!/usr/bin/env python3
"""
Temporary version without LED current control to test basic functionality.
"""

import time
import cv2
import logging
import tkinter as tk
from tkinter import ttk
from unlook import UnlookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectorGUI:
    """GUI controller for Unlook projector."""
    
    def __init__(self, client):
        """Initialize the GUI controller."""
        self.client = client
        self.current_color = "Green"  # Default to Green (VCSEL)
        self.pattern_active = True
        self.root = None
        self.running = True
        
    def create_gui(self):
        """Create GUI window and widgets."""
        # Create GUI window
        self.root = tk.Tk()
        self.root.title("Unlook Projector Control (No LED)")
        self.root.geometry("400x300")
        
        # Prevent window from being resized
        self.root.resizable(False, False)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Create GUI elements
        self._create_widgets()
        
        # Update pattern with initial settings
        self._update_pattern()
        
    def _create_widgets(self):
        """Create GUI widgets."""
        # Title
        title_label = ttk.Label(
            self.root, 
            text="Projector Control (Basic)",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Info label
        info_label = ttk.Label(
            self.root,
            text="LED intensity control disabled - testing basic patterns only",
            font=("Arial", 10, "italic"),
            foreground="orange"
        )
        info_label.pack(pady=5)
        
        # Channel selection frame
        channel_frame = ttk.LabelFrame(self.root, text="Color Channel", padding=10)
        channel_frame.pack(pady=10, padx=20, fill="x")
        
        # Radio buttons for channel selection
        self.color_var = tk.StringVar(value=self.current_color)
        
        colors = [
            ("Black (Off)", "Black"),
            ("Red", "Red"),
            ("Green (VCSEL IR)", "Green"),
            ("Blue", "Blue"),
            ("White (All)", "White")
        ]
        
        for text, value in colors:
            radio = ttk.Radiobutton(
                channel_frame,
                text=text,
                value=value,
                variable=self.color_var,
                command=self._on_color_change
            )
            radio.pack(anchor="w", pady=2)
        
        # Status label
        self.status_label = ttk.Label(
            self.root,
            text="Pattern: Vertical Lines",
            font=("Arial", 10),
            foreground="green"
        )
        self.status_label.pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(
            self.root,
            text="Press 'q' in camera window to quit",
            font=("Arial", 9, "italic")
        )
        instructions.pack()
        
    def _on_color_change(self):
        """Handle color selection change."""
        self.current_color = self.color_var.get()
        logger.info(f"Color changed to: {self.current_color}")
        self._update_pattern()
        
    def _update_pattern(self):
        """Update the projector pattern with current settings."""
        try:
            # Set projector to test pattern mode
            self.client.projector.set_test_pattern_mode()
            
            # Project vertical lines with the selected color
            success = self.client.projector.show_vertical_lines(
                foreground_color=self.current_color,
                background_color="Black",
                foreground_width=10,
                background_width=30
            )
            
            if success:
                self.status_label.config(
                    text=f"Pattern Active: {self.current_color} Lines",
                    foreground="green"
                )
            else:
                self.status_label.config(
                    text="Pattern Update Failed!",
                    foreground="red"
                )
                
        except Exception as e:
            logger.error(f"Failed to update pattern: {e}")
            self.status_label.config(
                text=f"Error: {str(e)}",
                foreground="red"
            )
    
    def _on_closing(self):
        """Handle window closing."""
        self.running = False
        if self.root:
            self.root.quit()
    
    def run(self):
        """Run the GUI event loop."""
        if self.root:
            self.root.mainloop()
        
    def close(self):
        """Close the GUI window."""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()


def main():
    """Main function to run projector control with GUI."""
    client = None
    streaming = False
    gui = None
    
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
        
        # Create GUI 
        gui = ProjectorGUI(client)
        gui.create_gui()
        
        # Show GUI window
        gui.root.update()
        
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
            cv2.putText(frame, "Projector Control - Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show current projector settings
            if gui:
                color_text = f"Color: {gui.current_color}"
                cv2.putText(frame, color_text, (10, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Show the frame
            cv2.imshow("Unlook Camera Stream", frame)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested")
                streaming = False
                gui.running = False
                return False
                
            return True
        
        # Start streaming
        logger.info("Starting camera stream...")
        streaming = True
        
        # Start the stream
        try:
            client.stream.start(camera_id, show_frame)
            
            # Keep the main thread alive while streaming and update GUI
            while streaming and gui.running:
                gui.root.update()
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            
        logger.info(f"Stream ended. Total frames: {frame_count}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        # Cleanup
        if gui:
            try:
                gui.close()
            except:
                pass
                
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