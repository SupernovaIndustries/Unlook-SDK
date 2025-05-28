#!/usr/bin/env python3
"""
Interactive GUI control for Unlook projector with RGB/VCSEL channels.

This example creates a GUI interface to control the projector's color channels
while streaming camera footage. The projector displays static vertical lines
with selectable color channel.

Features:
- Continuous camera streaming (press 'q' to quit)
- GUI controls for channel selection (Red/Green-VCSEL/Blue/White)
- Real-time pattern updates
- Mixed colors (Cyan, Magenta, Yellow) for testing

Note: The projector DLP342X supports real LED current control (0-100%).
The VCSEL IR is connected to the Green channel.
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
        """Initialize the GUI controller.
        
        Args:
            client: UnlookClient instance
        """
        self.client = client
        self.current_color = "Green"  # Default to Green (VCSEL)
        self.pattern_active = True
        self.root = None
        self.running = True
        
        # Individual LED intensities (0-100)
        self.red_intensity = 100
        self.green_intensity = 100
        self.blue_intensity = 100
        
    def create_gui(self):
        """Create GUI window and widgets."""
        # Create GUI window
        self.root = tk.Tk()
        self.root.title("Unlook Projector Control")
        self.root.geometry("450x600")  # Made taller for all controls
        
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
            text="Projector RGB/VCSEL Control",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Channel selection frame
        channel_frame = ttk.LabelFrame(self.root, text="Color Channel", padding=10)
        channel_frame.pack(pady=10, padx=20, fill="x")
        
        # Radio buttons for channel selection
        self.color_var = tk.StringVar(value=self.current_color)
        
        # Primary colors
        primary_label = ttk.Label(channel_frame, text="Primary Colors:", font=("Arial", 10, "bold"))
        primary_label.pack(anchor="w")
        
        primary_colors = [
            ("Black (Off)", "Black"),
            ("Red", "Red"),
            ("Green (VCSEL IR)", "Green"),
            ("Blue", "Blue"),
            ("White (All)", "White")
        ]
        
        for text, value in primary_colors:
            radio = ttk.Radiobutton(
                channel_frame,
                text=text,
                value=value,
                variable=self.color_var,
                command=self._on_color_change
            )
            radio.pack(anchor="w", pady=2, padx=20)
        
        # Mixed colors
        ttk.Separator(channel_frame, orient='horizontal').pack(fill='x', pady=5)
        mixed_label = ttk.Label(channel_frame, text="Mixed Colors:", font=("Arial", 10, "bold"))
        mixed_label.pack(anchor="w")
        
        mixed_colors = [
            ("Cyan (G+B)", "Cyan"),
            ("Magenta (R+B)", "Magenta"),
            ("Yellow (R+G)", "Yellow")
        ]
        
        for text, value in mixed_colors:
            radio = ttk.Radiobutton(
                channel_frame,
                text=text,
                value=value,
                variable=self.color_var,
                command=self._on_color_change
            )
            radio.pack(anchor="w", pady=2, padx=20)
        
        # Status label
        self.status_label = ttk.Label(
            self.root,
            text="Pattern: Vertical Lines",
            font=("Arial", 10),
            foreground="green"
        )
        self.status_label.pack(pady=10)
        
        # Intensity control frame
        intensity_frame = ttk.LabelFrame(self.root, text="LED Intensity Control", padding=10)
        intensity_frame.pack(pady=10, padx=20, fill="x")
        
        # Red intensity
        red_frame = ttk.Frame(intensity_frame)
        red_frame.pack(fill="x", pady=5)
        ttk.Label(red_frame, text="Red:", width=8).pack(side="left")
        self.red_intensity_var = tk.IntVar(value=self.red_intensity)
        self.red_intensity_label = ttk.Label(red_frame, text=f"{self.red_intensity}%", width=5)
        self.red_intensity_label.pack(side="right")
        self.red_slider = ttk.Scale(
            red_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.red_intensity_var,
            command=lambda v: self._on_intensity_change('red', v)
        )
        self.red_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        # Green intensity
        green_frame = ttk.Frame(intensity_frame)
        green_frame.pack(fill="x", pady=5)
        ttk.Label(green_frame, text="Green:", width=8).pack(side="left")
        self.green_intensity_var = tk.IntVar(value=self.green_intensity)
        self.green_intensity_label = ttk.Label(green_frame, text=f"{self.green_intensity}%", width=5)
        self.green_intensity_label.pack(side="right")
        self.green_slider = ttk.Scale(
            green_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.green_intensity_var,
            command=lambda v: self._on_intensity_change('green', v)
        )
        self.green_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        # Blue intensity
        blue_frame = ttk.Frame(intensity_frame)
        blue_frame.pack(fill="x", pady=5)
        ttk.Label(blue_frame, text="Blue:", width=8).pack(side="left")
        self.blue_intensity_var = tk.IntVar(value=self.blue_intensity)
        self.blue_intensity_label = ttk.Label(blue_frame, text=f"{self.blue_intensity}%", width=5)
        self.blue_intensity_label.pack(side="right")
        self.blue_slider = ttk.Scale(
            blue_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.blue_intensity_var,
            command=lambda v: self._on_intensity_change('blue', v)
        )
        self.blue_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        # Instructions
        instructions = ttk.Label(
            self.root,
            text="Press 'q' in camera window to quit",
            font=("Arial", 9, "italic")
        )
        instructions.pack(pady=5)
        
    def _on_color_change(self):
        """Handle color selection change."""
        self.current_color = self.color_var.get()
        logger.info(f"Color changed to: {self.current_color}")
        self._update_pattern()
        
    def _on_intensity_change(self, channel, value):
        """Handle intensity slider change.
        
        Args:
            channel: 'red', 'green', or 'blue'
            value: Intensity value (0-100)
        """
        intensity = int(float(value))
        
        if channel == 'red':
            self.red_intensity = intensity
            self.red_intensity_label.config(text=f"{intensity}%")
        elif channel == 'green':
            self.green_intensity = intensity
            self.green_intensity_label.config(text=f"{intensity}%")
        elif channel == 'blue':
            self.blue_intensity = intensity
            self.blue_intensity_label.config(text=f"{intensity}%")
        
        logger.info(f"{channel.capitalize()} intensity: {intensity}%")
        self._update_pattern()
        
    def _update_pattern(self):
        """Update the projector pattern with current settings."""
        try:
            # Set projector to test pattern mode
            self.client.projector.set_test_pattern_mode()
            
            # First, set the LED current based on intensity sliders
            led_success = self.client.projector.set_led_intensity_percent(
                self.red_intensity,
                self.green_intensity,
                self.blue_intensity
            )
            
            if not led_success:
                logger.warning("Failed to set LED current, continuing with pattern")
            
            # Project vertical lines with the selected color
            success = self.client.projector.show_vertical_lines(
                foreground_color=self.current_color,
                background_color="Black",
                foreground_width=10,  # Fixed width
                background_width=30   # Fixed spacing
            )
            
            if success:
                intensity_text = f"R:{self.red_intensity}% G:{self.green_intensity}% B:{self.blue_intensity}%"
                self.status_label.config(
                    text=f"Pattern: {self.current_color} ({intensity_text})",
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