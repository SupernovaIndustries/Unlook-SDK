#!/usr/bin/env python3
"""
Real-time Focus Adjustment Tool for UnLook Scanner

This tool provides a real-time GUI for adjusting camera and projector focus.
It displays live streams from both cameras with projected line patterns
and provides real-time focus assessment feedback.

Usage:
    python focus_adjustment_tool.py [--scanner-ip IP] [--port PORT]
    
Controls:
    Q - Quit application
    SPACE - Toggle projector patterns
    R - Reset focus history
"""

import argparse
import logging
import sys
import time
import threading
from typing import Optional, Dict, Any
import numpy as np
import cv2

try:
    # Try Qt first (preferred)
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                QWidget, QLabel, QPushButton, QProgressBar, QFrame)
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QImage, QPixmap, QFont
    GUI_BACKEND = 'qt'
except ImportError:
    try:
        # Fallback to tkinter
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        GUI_BACKEND = 'tk'
    except ImportError:
        print("Error: No GUI backend available. Please install PyQt5 or ensure tkinter is available.")
        sys.exit(1)

# Import UnLook SDK
try:
    from unlook.client.scanner.scanner import UnlookClient, FocusAssessment
    from unlook.client.camera.camera_config import CameraConfig, CompressionFormat, ColorMode
    from unlook.core.discovery import DiscoveryService
except ImportError as e:
    print(f"Error importing UnLook SDK: {e}")
    print("Make sure you're running from the SDK root directory and have installed dependencies.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocusAdjustmentTool:
    """
    Real-time focus adjustment tool with GUI.
    """
    
    def __init__(self, scanner_ip: Optional[str] = None, scanner_port: int = 5555):
        """
        Initialize the focus adjustment tool.
        
        Args:
            scanner_ip: IP address of scanner (None for auto-discovery)
            scanner_port: Port of scanner
        """
        self.scanner_ip = scanner_ip
        self.scanner_port = scanner_port
        
        # UnLook components
        self.client = None
        self.focus_assessment = FocusAssessment()
        
        # State
        self.running = False
        self.projector_enabled = True
        self.pattern_cycle_time = 2.0  # seconds between pattern changes
        self.current_pattern = 0
        
        # Image data
        self.left_image = None
        self.right_image = None
        self.focus_results = {'left': {}, 'right': {}, 'projector': {}}
        
        # Threading
        self.capture_thread = None
        self.pattern_thread = None
        self.lock = threading.Lock()
        
        # Pattern definitions for cycling
        self.line_patterns = [
            {'type': 'horizontal_lines', 'spacing': 20},
            {'type': 'vertical_lines', 'spacing': 20},
            {'type': 'horizontal_lines', 'spacing': 10},
            {'type': 'vertical_lines', 'spacing': 10},
        ]
        
        # Initialize GUI
        if GUI_BACKEND == 'qt':
            self._init_qt_gui()
        else:
            self._init_tk_gui()
    
    def _init_qt_gui(self):
        """Initialize PyQt5 GUI."""
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("UnLook Scanner - Focus Adjustment Tool")
        self.window.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Status banner
        self.status_banner = QLabel("🔍 Initializing focus assessment...")
        self.status_banner.setStyleSheet("""
            QLabel {
                background-color: #ffeb3b;
                color: #333;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        main_layout.addWidget(self.status_banner)
        
        # Camera views layout
        cameras_layout = QHBoxLayout()
        
        # Left camera frame
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.Box)
        left_layout = QVBoxLayout(left_frame)
        
        self.left_label = QLabel("Left Camera")
        self.left_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(self.left_label)
        
        self.left_image_label = QLabel()
        self.left_image_label.setMinimumSize(400, 300)
        self.left_image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        left_layout.addWidget(self.left_image_label)
        
        self.left_focus_bar = QProgressBar()
        self.left_focus_bar.setMaximum(100)
        left_layout.addWidget(self.left_focus_bar)
        
        self.left_focus_label = QLabel("Focus: --")
        left_layout.addWidget(self.left_focus_label)
        
        cameras_layout.addWidget(left_frame)
        
        # Right camera frame
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.Box)
        right_layout = QVBoxLayout(right_frame)
        
        self.right_label = QLabel("Right Camera")
        self.right_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(self.right_label)
        
        self.right_image_label = QLabel()
        self.right_image_label.setMinimumSize(400, 300)
        self.right_image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        right_layout.addWidget(self.right_image_label)
        
        self.right_focus_bar = QProgressBar()
        self.right_focus_bar.setMaximum(100)
        right_layout.addWidget(self.right_focus_bar)
        
        self.right_focus_label = QLabel("Focus: --")
        right_layout.addWidget(self.right_focus_label)
        
        cameras_layout.addWidget(right_frame)
        
        main_layout.addLayout(cameras_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.pattern_button = QPushButton("Toggle Projector (SPACE)")
        self.pattern_button.clicked.connect(self.toggle_projector)
        controls_layout.addWidget(self.pattern_button)
        
        self.reset_button = QPushButton("Reset Focus History (R)")
        self.reset_button.clicked.connect(self.reset_focus_history)
        controls_layout.addWidget(self.reset_button)
        
        self.quit_button = QPushButton("Quit (Q)")
        self.quit_button.clicked.connect(self.quit_application)
        controls_layout.addWidget(self.quit_button)
        
        main_layout.addLayout(controls_layout)
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(33)  # ~30 FPS
        
        # Keyboard shortcuts
        self.window.keyPressEvent = self.handle_keypress
        
    def _init_tk_gui(self):
        """Initialize tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("UnLook Scanner - Focus Adjustment Tool")
        self.root.geometry("1200x800")
        
        # Status banner
        self.status_var = tk.StringVar(value="🔍 Initializing focus assessment...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, 
                                   bg="yellow", fg="black", font=("Arial", 14, "bold"),
                                   pady=10)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera frames
        cameras_frame = tk.Frame(self.root)
        cameras_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left camera
        left_frame = tk.LabelFrame(cameras_frame, text="Left Camera", font=("Arial", 12, "bold"))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.left_image_label = tk.Label(left_frame, bg="black", width=50, height=25)
        self.left_image_label.pack(padx=5, pady=5)
        
        self.left_focus_var = tk.StringVar(value="Focus: --")
        tk.Label(left_frame, textvariable=self.left_focus_var).pack()
        
        # Right camera
        right_frame = tk.LabelFrame(cameras_frame, text="Right Camera", font=("Arial", 12, "bold"))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.right_image_label = tk.Label(right_frame, bg="black", width=50, height=25)
        self.right_image_label.pack(padx=5, pady=5)
        
        self.right_focus_var = tk.StringVar(value="Focus: --")
        tk.Label(right_frame, textvariable=self.right_focus_var).pack()
        
        # Control buttons
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(controls_frame, text="Toggle Projector (SPACE)", 
                 command=self.toggle_projector).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Reset Focus History (R)", 
                 command=self.reset_focus_history).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Quit (Q)", 
                 command=self.quit_application).pack(side=tk.LEFT, padx=5)
        
        # Keyboard bindings
        self.root.bind('<KeyPress>', self.handle_keypress_tk)
        self.root.focus_set()
        
        # Timer for updates
        self.update_gui()
    
    def connect_to_scanner(self) -> bool:
        """Connect to UnLook scanner."""
        try:
            logger.info("Connecting to UnLook scanner...")
            self.client = UnlookClient("FocusAdjustmentTool", auto_discover=True)
            
            # Wait for discovery or use provided IP
            if self.scanner_ip:
                endpoint = f"tcp://{self.scanner_ip}:{self.scanner_port}"
                success = self.client.connect(endpoint, timeout=5000)
            else:
                # Auto-discovery
                time.sleep(2)  # Wait for discovery
                scanners = self.client.get_discovered_scanners()
                if not scanners:
                    logger.error("No scanners found. Make sure scanner is running.")
                    return False
                
                success = self.client.connect(scanners[0], timeout=5000)
            
            if not success:
                logger.error("Failed to connect to scanner")
                return False
            
            logger.info("Connected to scanner successfully")
            
            # Configure cameras for fast preview
            camera_config = CameraConfig.create_preset("streaming")
            camera_config.compression_format = CompressionFormat.JPEG
            camera_config.jpeg_quality = 70
            camera_config.color_mode = ColorMode.GRAYSCALE  # Grayscale for focus assessment
            camera_config.resolution = (640, 480)  # Medium size for fast processing
            
            self.client.camera.set_config(camera_config)
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to scanner: {e}")
            return False
    
    def start_capture_thread(self):
        """Start the image capture thread."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start pattern cycling thread
        self.pattern_thread = threading.Thread(target=self._pattern_cycle_loop, daemon=True)
        self.pattern_thread.start()
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.running:
            try:
                if not self.client or not self.client.connected:
                    time.sleep(0.1)
                    continue
                
                # Capture from both cameras
                images = self.client.camera.capture_multi(['left', 'right'])
                
                if images and 'left' in images and 'right' in images:
                    with self.lock:
                        self.left_image = images['left']
                        self.right_image = images['right']
                        
                        # Assess focus for both cameras
                        if self.left_image is not None:
                            self.focus_results['left'] = self.focus_assessment.assess_camera_focus(
                                self.left_image, 'left'
                            )
                            
                            # If projector is on, also assess projector focus from left camera
                            if self.projector_enabled:
                                self.focus_results['projector'] = self.focus_assessment.assess_projector_focus(
                                    self.left_image, 'lines'
                                )
                        
                        if self.right_image is not None:
                            self.focus_results['right'] = self.focus_assessment.assess_camera_focus(
                                self.right_image, 'right'
                            )
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(1)
    
    def _pattern_cycle_loop(self):
        """Cycle through different line patterns for projector focus assessment."""
        while self.running:
            try:
                if self.projector_enabled and self.client and self.client.connected:
                    pattern = self.line_patterns[self.current_pattern]
                    
                    if pattern['type'] == 'horizontal_lines':
                        self.client.projector.project_horizontal_lines(spacing=pattern['spacing'])
                    elif pattern['type'] == 'vertical_lines':
                        self.client.projector.project_vertical_lines(spacing=pattern['spacing'])
                    
                    # Next pattern
                    self.current_pattern = (self.current_pattern + 1) % len(self.line_patterns)
                
                time.sleep(self.pattern_cycle_time)
                
            except Exception as e:
                logger.error(f"Error in pattern cycle: {e}")
                time.sleep(1)
    
    def update_gui(self):
        """Update GUI with latest images and focus information."""
        try:
            with self.lock:
                # Update status banner
                overall_status = self.focus_assessment.get_overall_focus_status()
                status_message = overall_status['message']
                
                if GUI_BACKEND == 'qt':
                    # Update status banner color based on focus status
                    if overall_status['status'] == 'ready':
                        color = "#4caf50"  # Green
                    elif overall_status['status'] == 'poor':
                        color = "#f44336"  # Red
                    else:
                        color = "#ffeb3b"  # Yellow
                    
                    self.status_banner.setStyleSheet(f"""
                        QLabel {{
                            background-color: {color};
                            color: #333;
                            padding: 10px;
                            font-size: 16px;
                            font-weight: bold;
                            border-radius: 5px;
                            margin: 5px;
                        }}
                    """)
                    self.status_banner.setText(status_message)
                    
                    # Update left camera
                    if self.left_image is not None:
                        self._update_qt_image(self.left_image_label, self.left_image)
                        
                    if 'left' in self.focus_results:
                        result = self.focus_results['left']
                        score_normalized = min(100, int(result['smoothed_score'] / 2))  # Rough normalization
                        self.left_focus_bar.setValue(score_normalized)
                        self.left_focus_label.setText(f"Focus: {result['smoothed_score']:.1f} ({result['quality']})")
                    
                    # Update right camera
                    if self.right_image is not None:
                        self._update_qt_image(self.right_image_label, self.right_image)
                        
                    if 'right' in self.focus_results:
                        result = self.focus_results['right']
                        score_normalized = min(100, int(result['smoothed_score'] / 2))
                        self.right_focus_bar.setValue(score_normalized)
                        self.right_focus_label.setText(f"Focus: {result['smoothed_score']:.1f} ({result['quality']})")
                
                else:  # tkinter
                    self.status_var.set(status_message)
                    
                    # Update images and focus info for tkinter
                    if self.left_image is not None:
                        self._update_tk_image(self.left_image_label, self.left_image)
                    
                    if 'left' in self.focus_results:
                        result = self.focus_results['left']
                        self.left_focus_var.set(f"Focus: {result['smoothed_score']:.1f} ({result['quality']})")
                    
                    if self.right_image is not None:
                        self._update_tk_image(self.right_image_label, self.right_image)
                    
                    if 'right' in self.focus_results:
                        result = self.focus_results['right']
                        self.right_focus_var.set(f"Focus: {result['smoothed_score']:.1f} ({result['quality']})")
            
            # Schedule next update for tkinter
            if GUI_BACKEND == 'tk':
                self.root.after(33, self.update_gui)
                
        except Exception as e:
            logger.error(f"Error updating GUI: {e}")
    
    def _update_qt_image(self, label, image):
        """Update PyQt5 image label."""
        try:
            # Resize image to fit label
            height, width = image.shape[:2]
            label_size = label.size()
            
            # Calculate scaling to fit label while maintaining aspect ratio
            scale_w = label_size.width() / width
            scale_h = label_size.height() / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height))
            
            # Convert to Qt format
            if len(resized.shape) == 3:
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                h, w = resized.shape
                bytes_per_line = w
                qt_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap)
            
        except Exception as e:
            logger.error(f"Error updating Qt image: {e}")
    
    def _update_tk_image(self, label, image):
        """Update tkinter image label."""
        try:
            # Resize image for display
            display_image = cv2.resize(image, (320, 240))
            
            # Convert to PIL format
            if len(display_image.shape) == 3:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(display_image)
            else:
                pil_image = Image.fromarray(display_image)
            
            # Convert to tkinter format
            tk_image = ImageTk.PhotoImage(pil_image)
            label.configure(image=tk_image)
            label.image = tk_image  # Keep a reference
            
        except Exception as e:
            logger.error(f"Error updating Tk image: {e}")
    
    def toggle_projector(self):
        """Toggle projector on/off."""
        try:
            self.projector_enabled = not self.projector_enabled
            
            if self.client and self.client.connected:
                if self.projector_enabled:
                    logger.info("Enabling projector patterns")
                    # Start with first pattern
                    self.current_pattern = 0
                    pattern = self.line_patterns[self.current_pattern]
                    if pattern['type'] == 'horizontal_lines':
                        self.client.projector.project_horizontal_lines(spacing=pattern['spacing'])
                    else:
                        self.client.projector.project_vertical_lines(spacing=pattern['spacing'])
                else:
                    logger.info("Disabling projector")
                    self.client.projector.turn_off()
                    
        except Exception as e:
            logger.error(f"Error toggling projector: {e}")
    
    def reset_focus_history(self):
        """Reset focus assessment history."""
        self.focus_assessment.focus_history = {'left': [], 'right': [], 'projector': []}
        logger.info("Focus history reset")
    
    def handle_keypress(self, event):
        """Handle PyQt5 keypress events."""
        key = event.key()
        if key == ord('Q'):
            self.quit_application()
        elif key == ord(' '):  # Space
            self.toggle_projector()
        elif key == ord('R'):
            self.reset_focus_history()
    
    def handle_keypress_tk(self, event):
        """Handle tkinter keypress events."""
        if event.keysym.lower() == 'q':
            self.quit_application()
        elif event.keysym == 'space':
            self.toggle_projector()
        elif event.keysym.lower() == 'r':
            self.reset_focus_history()
    
    def quit_application(self):
        """Quit the application."""
        logger.info("Quitting focus adjustment tool")
        self.running = False
        
        try:
            if self.client:
                self.client.projector.turn_off()
                self.client.disconnect()
        except:
            pass
        
        if GUI_BACKEND == 'qt':
            self.app.quit()
        else:
            self.root.quit()
        
        sys.exit(0)
    
    def run(self):
        """Run the focus adjustment tool."""
        logger.info("Starting UnLook Focus Adjustment Tool")
        
        # Connect to scanner
        if not self.connect_to_scanner():
            if GUI_BACKEND == 'qt':
                self.status_banner.setText("❌ Failed to connect to scanner")
                self.status_banner.setStyleSheet("""
                    QLabel {
                        background-color: #f44336;
                        color: white;
                        padding: 10px;
                        font-size: 16px;
                        font-weight: bold;
                        border-radius: 5px;
                        margin: 5px;
                    }
                """)
            else:
                self.status_var.set("❌ Failed to connect to scanner")
            
            logger.error("Could not connect to scanner. Make sure scanner is running.")
        else:
            # Start capture
            self.start_capture_thread()
            
            # Start projector with initial pattern
            if self.projector_enabled:
                self.toggle_projector()
        
        # Start GUI event loop
        if GUI_BACKEND == 'qt':
            self.window.show()
            sys.exit(self.app.exec_())
        else:
            self.root.mainloop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="UnLook Scanner Focus Adjustment Tool")
    parser.add_argument("--scanner-ip", help="Scanner IP address (auto-discover if not specified)")
    parser.add_argument("--port", type=int, default=5555, help="Scanner port (default: 5555)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run tool
    tool = FocusAdjustmentTool(scanner_ip=args.scanner_ip, scanner_port=args.port)
    tool.run()


if __name__ == "__main__":
    main()