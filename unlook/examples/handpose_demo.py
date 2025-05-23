#!/usr/bin/env python3
"""Demo application for UnLook SDK hand tracking using Enhanced OpenCV GUI.

This application demonstrates the capabilities of the UnLook SDK handpose module
with real-time 3D hand tracking, gesture recognition, enhanced LED controls, and
advanced image processing. Uses OpenCV's GUI with enhanced display patterns.
"""

import os
import sys
import time
import argparse
import logging
import threading
import queue
import cv2
import numpy as np
import psutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unlook.handpose.demo")

# Import UnLook SDK modules
try:
    from unlook import UnlookClient
    from unlook.client.scanning.handpose import HandDetector, HandTracker, GestureRecognizer, GestureType
    from unlook.client.projector.led_controller import LEDController
except ImportError as e:
    logger.error(f"Failed to import UnLook SDK: {e}")
    logger.error("Make sure UnLook SDK is installed or in the correct path")
    sys.exit(1)


# Enhanced image preprocessing function from enhanced_gesture_demo
def preprocess_image(image):
    """
    Enhanced image preprocessing for better hand detection.
    Applies contrast enhancement and noise reduction.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image or None if input is None
    """
    # Skip preprocessing if image is None
    if image is None:
        return None
    # Make a copy to avoid modifying the original
    enhanced = image.copy()
    # Convert to LAB color space (L = lightness)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to lightness channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge channels back and convert to BGR
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    # Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
    return enhanced


def calculate_optimal_led_intensity(frame_left, frame_right):
    """
    Calculate optimal LED intensity based on image brightness analysis.
    
    Args:
        frame_left: Left camera frame
        frame_right: Right camera frame
        
    Returns:
        Optimal LED intensity (0-450 mA)
    """
    try:
        # Calculate average brightness of both frames
        if frame_left is not None:
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            brightness_left = np.mean(gray_left)
        else:
            brightness_left = 128
        
        if frame_right is not None:
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            brightness_right = np.mean(gray_right)
        else:
            brightness_right = 128
        
        # Average brightness across both cameras
        avg_brightness = (brightness_left + brightness_right) / 2
        
        # Calculate optimal LED intensity (inverse relationship with brightness)
        # Darker scenes need more LED illumination
        if avg_brightness < 80:  # Very dark
            return 300
        elif avg_brightness < 120:  # Dim
            return 200
        elif avg_brightness < 160:  # Normal
            return 100
        else:  # Bright
            return 50
            
    except Exception as e:
        logger.warning(f"LED intensity calculation failed: {e}")
        return 150  # Default fallback


def evaluate_image_quality(image):
    """
    Evaluate image quality based on contrast, brightness, and edge detail.
    
    Args:
        image: Input image
        
    Returns:
        Quality score (higher is better)
    """
    if image is None:
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    # Calculate brightness (mean)
    brightness = np.mean(gray)
    
    # Calculate edge detail (Sobel gradient magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_detail = np.mean(edge_magnitude)
    
    # Penalize too bright or too dark images (ideal around 127)
    brightness_penalty = 1.0 - abs(brightness - 127) / 127
    
    # Calculate score (weighted sum)
    score = (contrast * 0.5) + (edge_detail * 0.3) + (brightness_penalty * 100)
    
    return score


class GestureUI:
    """Enhanced UI handler for gesture recognition demo with advanced display and LED controls."""
    
    # Define LED settings
    MIN_LED_INTENSITY = 0   # Allow 0 for completely off
    MAX_LED_INTENSITY = 450
    LED_STEP = 5           # Fine-grained steps for LED control
    
    def __init__(self, presentation_mode=False, enable_point_projection=True):
        # Store if we're in presentation mode
        self.presentation_mode = presentation_mode
        self.enable_point_projection = enable_point_projection
        
        # LED control interface
        self.show_led_controls = False
        # LED1 for point projection (was previously disabled)
        self.led1_intensity = 50 if enable_point_projection else 0   # Enable LED1 for point projection
        self.led2_intensity = 50   # Default starting point for flood illumination
        self.active_control = None  # Which control is active
        self.led1_slider_bounds = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        self.led2_slider_bounds = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        # Define gesture colors
        self.gesture_colors = {
            GestureType.OPEN_PALM: (0, 255, 0),      # Green
            GestureType.CLOSED_FIST: (0, 0, 255),    # Red
            GestureType.POINTING: (255, 0, 0),       # Blue
            GestureType.PEACE: (255, 255, 0),        # Cyan
            GestureType.THUMBS_UP: (0, 255, 255),    # Yellow
            GestureType.THUMBS_DOWN: (255, 0, 255),  # Magenta
            GestureType.OK: (255, 165, 0),           # Orange
            GestureType.ROCK: (128, 0, 128),         # Purple
            GestureType.PINCH: (165, 42, 42),        # Brown
            GestureType.WAVE: (255, 255, 255),       # White
            GestureType.UNKNOWN: (128, 128, 128),     # Gray
        }
        
        # Gesture description to display
        self.gesture_descriptions = {
            GestureType.OPEN_PALM: "Open Palm - Reset",
            GestureType.CLOSED_FIST: "Closed Fist - Hold/Grab",
            GestureType.POINTING: "Pointing - Select",
            GestureType.PEACE: "Peace Sign - Split",
            GestureType.THUMBS_UP: "Thumbs Up - Approve/Like",
            GestureType.THUMBS_DOWN: "Thumbs Down - Reject",
            GestureType.OK: "OK Sign - Confirm",
            GestureType.ROCK: "Rock Sign - Cool!",
            GestureType.PINCH: "Pinch - Zoom/Scale",
            GestureType.WAVE: "Wave - Hello/Goodbye",
            GestureType.UNKNOWN: "Unknown Gesture",
        }
        
        # Initialize notification system
        self.notification = None
        self.notification_color = (255, 255, 255)
        self.notification_start_time = 0
        self.notification_duration = 2.0  # seconds
        
        # LED controller reference (will be set later)
        self.led_controller = None
        
        # Hand trajectory tracking
        self.hand_trajectory = []
        self.max_trajectory_points = 20
    
    def add_notification(self, text, color=(255, 255, 255), duration=2.0):
        """Add a temporary notification to the screen."""
        self.notification = text
        self.notification_color = color
        self.notification_start_time = time.time()
        self.notification_duration = duration
    
    def update_hand_trajectory(self, hand_position):
        """Update the hand trajectory for visualization.
        
        Args:
            hand_position: 2D or 3D position of hand center/wrist
        """
        if hand_position is not None:
            # Add to visualization trajectory
            self.hand_trajectory.append(hand_position.copy())
            if len(self.hand_trajectory) > self.max_trajectory_points:
                self.hand_trajectory.pop(0)
    
    def create_display(self, frame_left, frame_right, results, show_skeleton=True):
        """Create an enhanced full-screen display with hand tracking and gesture recognition results."""
        # Get original dimensions
        orig_h, orig_w = frame_left.shape[:2]
        
        # Get the screen resolution (use first monitor)
        try:
            screen_w, screen_h = 1920, 1080  # Default Full HD
        except:
            # If error, use defaults
            pass
        
        # Reserve space for the UI panel on the right
        panel_width = int(screen_w * 0.25)  # 25% of screen width for panel
        camera_space_width = screen_w - panel_width
        
        # Calculate dimensions for the camera views (stacked vertically)
        # Each camera takes half of the available height
        camera_height = (screen_h // 2) - 10  # Half height minus padding
        camera_width = min(camera_space_width, int(camera_height * orig_w / orig_h))  # Maintain aspect ratio
        
        # Create a display at screen resolution
        display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Fill with dark background
        display.fill(20)
        
        # Resize camera frames to fit the calculated dimensions
        frame_left_resized = cv2.resize(frame_left, (camera_width, camera_height))
        frame_right_resized = cv2.resize(frame_right, (camera_width, camera_height))
        
        # Place the left camera image at the top
        top_y_offset = 5  # Top padding
        display[top_y_offset:top_y_offset+camera_height, 0:camera_width] = frame_left_resized
        
        # Place the right camera image below the left one
        bottom_y_offset = camera_height + 10  # After left image + padding
        display[bottom_y_offset:bottom_y_offset+camera_height, 0:camera_width] = frame_right_resized
        
        # Add a UI panel on the right
        ui_panel = np.ones((screen_h, panel_width, 3), dtype=np.uint8) * 40  # Dark gray
        display[:, camera_width:camera_width+panel_width] = ui_panel
        
        # Store these values for other parts of the code to use
        self.display_width = screen_w
        self.display_height = screen_h
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.panel_width = panel_width
        self.top_y_offset = top_y_offset
        self.bottom_y_offset = bottom_y_offset
        
        # Add title to UI panel
        cv2.putText(display, "GESTURE RECOGNITION", (camera_width + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw hand skeletons if requested
        if show_skeleton:
            # Define connections for hand skeleton
            connections = [
                # Thumb
                [0, 1], [1, 2], [2, 3], [3, 4],
                # Index finger
                [0, 5], [5, 6], [6, 7], [7, 8],
                # Middle finger
                [0, 9], [9, 10], [10, 11], [11, 12],
                # Ring finger
                [0, 13], [13, 14], [14, 15], [15, 16],
                # Pinky
                [0, 17], [17, 18], [18, 19], [19, 20],
                # Palm connections
                [5, 9], [9, 13], [13, 17]
            ]
            
            # Draw left camera detections - now at the top
            for hand_idx, hand_kpts in enumerate(results.get('2d_left', [])):
                # Get pixel coordinates - scale to camera width/height
                pixel_coords = np.zeros((hand_kpts.shape[0], 2), dtype=np.int32)
                for i, kpt in enumerate(hand_kpts):
                    pixel_coords[i] = [int(kpt[0] * self.camera_width), 
                                       int(kpt[1] * self.camera_height) + self.top_y_offset]
                
                # Get hand type if available
                handedness = "Unknown"
                if 'handedness_left' in results and len(results['handedness_left']) > hand_idx:
                    handedness = results['handedness_left'][hand_idx]
                
                # Choose color based on hand type
                color = (0, 255, 0) if handedness == "Left" else (0, 0, 255)
                
                # Draw skeleton
                for connection in connections:
                    pt1 = tuple(pixel_coords[connection[0]])
                    pt2 = tuple(pixel_coords[connection[1]])
                    cv2.line(display, pt1, (pt2[0], pt2[1]), color, 2)
                
                # Draw keypoints
                for pt in pixel_coords:
                    cv2.circle(display, tuple(pt), 4, color, -1)
                
                # Add label
                cv2.putText(display, handedness, (pixel_coords[0][0], pixel_coords[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw right camera detections - below the left camera
            for hand_idx, hand_kpts in enumerate(results.get('2d_right', [])):
                # Get pixel coordinates - scale to camera width/height
                pixel_coords = np.zeros((hand_kpts.shape[0], 2), dtype=np.int32)
                for i, kpt in enumerate(hand_kpts):
                    pixel_coords[i] = [int(kpt[0] * self.camera_width), 
                                       int(kpt[1] * self.camera_height) + self.bottom_y_offset]
                
                # Get hand type if available
                handedness = "Unknown"
                if 'handedness_right' in results and len(results['handedness_right']) > hand_idx:
                    handedness = results['handedness_right'][hand_idx]
                
                # Choose color based on hand type
                color = (0, 255, 0) if handedness == "Left" else (0, 0, 255)
                
                # Draw skeleton
                for connection in connections:
                    pt1 = tuple(pixel_coords[connection[0]])
                    pt2 = tuple(pixel_coords[connection[1]])
                    cv2.line(display, pt1, (pt2[0], pt2[1]), color, 2)
                
                # Draw keypoints
                for pt in pixel_coords:
                    cv2.circle(display, tuple(pt), 4, color, -1)
                
                # Add label
                cv2.putText(display, handedness, (pixel_coords[0][0], pixel_coords[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display detected gestures
        if 'gestures' in results and results['gestures']:
            gesture_y = 70
            for gesture_idx, gesture_info in enumerate(results['gestures']):
                gesture_type = gesture_info['type']
                gesture_name = gesture_info['name']
                confidence = gesture_info['confidence']
                
                # Get color for this gesture
                color = self.gesture_colors.get(gesture_type, (255, 255, 255))
                
                # Get description
                description = self.gesture_descriptions.get(gesture_type, "Unknown Gesture")
                
                # Draw gesture info
                cv2.putText(display, f"Hand {gesture_idx}: {gesture_name}", 
                            (self.camera_width + 20, gesture_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, color, 2)
                
                # Draw confidence bar
                bar_length = int(confidence * (self.panel_width - 50))
                cv2.rectangle(display, 
                              (self.camera_width + 20, gesture_y + 10), 
                              (self.camera_width + 20 + bar_length, gesture_y + 20), 
                              color, -1)
                
                # Add description
                cv2.putText(display, description, 
                            (self.camera_width + 20, gesture_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (200, 200, 200), 1)
                
                gesture_y += 60
        else:
            # No gestures detected
            cv2.putText(display, "No gestures detected", 
                        (self.camera_width + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (150, 150, 150), 2)
        
        # Show instructions - place at the bottom of the UI panel
        bottom_section_y = self.display_height - 150
        cv2.putText(display, "CONTROLS:", (self.camera_width + 20, bottom_section_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "q - Quit", (self.camera_width + 30, bottom_section_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "s - Toggle skeleton", (self.camera_width + 30, bottom_section_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "l - Toggle LED controls", (self.camera_width + 30, bottom_section_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "r - Reset LED2", (self.camera_width + 30, bottom_section_y + 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Add labels for the camera views
        cv2.putText(display, "LEFT CAMERA", (10, self.top_y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "RIGHT CAMERA", (10, self.bottom_y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw LED control interface if enabled
        if self.show_led_controls:
            self._draw_led_controls(display)
        
        # Show notification if active
        if self.notification:
            current_time = time.time()
            elapsed = current_time - self.notification_start_time
            
            if elapsed < self.notification_duration:
                # Notification is still active, display it
                alpha = 1.0 - (elapsed / self.notification_duration)
                
                # Create a semi-transparent overlay - centered at bottom
                overlay = display.copy()
                cv2.rectangle(overlay, (50, self.display_height - 70), (self.display_width - 50, self.display_height - 20), 
                              (40, 40, 40), -1)
                
                # Add notification text
                cv2.putText(overlay, self.notification, (70, self.display_height - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            self.notification_color, 2)
                
                # Blend with alpha transparency
                cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            else:
                # Notification expired
                self.notification = None
        
        return display
    
    def _draw_led_controls(self, display):
        """Draw manual LED intensity control interface"""
        # Place at the bottom of the side panel, above the controls section
        panel_start_x = self.camera_width + 20
        panel_width = self.panel_width - 40
        controls_y = self.display_height - 310  # Move controls up to avoid overlap
        
        # Title for LED controls
        cv2.putText(display, "LED INTENSITY CONTROLS", (panel_start_x, controls_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # LED1 control (point projection)
        if self.enable_point_projection:
            led1_text = f"LED1: {self.led1_intensity} mA (point projection)"
            text_color = (255, 165, 0) if self.active_control == 'led1' else (255, 255, 255)
            cv2.putText(display, led1_text, (panel_start_x, controls_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
            # Draw LED1 slider bar
            slider_y = controls_y + 15
            slider_height = 8
            
            # Background bar
            cv2.rectangle(display, 
                          (panel_start_x, slider_y), 
                          (panel_start_x + panel_width, slider_y + slider_height),
                          (80, 80, 80), -1)
            
            # Calculate filled portion based on current value
            filled_width = int((self.led1_intensity / self.MAX_LED_INTENSITY) * panel_width)
            cv2.rectangle(display,
                          (panel_start_x, slider_y),
                          (panel_start_x + filled_width, slider_y + slider_height),
                          (255, 165, 0), -1)  # Orange for point projection
            
            # Draw slider thumb
            thumb_x = panel_start_x + filled_width
            thumb_size = 12
            cv2.rectangle(display,
                         (thumb_x - thumb_size//2, slider_y - thumb_size//2),
                         (thumb_x + thumb_size//2, slider_y + slider_height + thumb_size//2),
                         (255, 255, 255), -1)
            
            # Store control positions for mouse interaction
            self.led1_slider_bounds = {
                'x': panel_start_x,
                'y': slider_y,
                'width': panel_width,
                'height': slider_height
            }
            
            controls_y += 50
        else:
            cv2.putText(display, "LED1: 0 mA (point projection disabled)", (panel_start_x, controls_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            controls_y += 30
        
        # LED2 slider control
        led2_text = f"LED2: {self.led2_intensity} mA"
        text_color = (0, 255, 0) if self.active_control == 'led2' else (255, 255, 255)
        cv2.putText(display, led2_text, (panel_start_x, controls_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # Draw slider bar
        slider_y = controls_y + 15
        slider_height = 8
        
        # Background bar
        cv2.rectangle(display, 
                      (panel_start_x, slider_y), 
                      (panel_start_x + panel_width, slider_y + slider_height),
                      (80, 80, 80), -1)
        
        # Calculate filled portion based on current value
        filled_width = int((self.led2_intensity / self.MAX_LED_INTENSITY) * panel_width)
        cv2.rectangle(display,
                      (panel_start_x, slider_y),
                      (panel_start_x + filled_width, slider_y + slider_height),
                      (0, 200, 0), -1)
        
        # Draw slider thumb
        thumb_x = panel_start_x + filled_width
        thumb_size = 12
        cv2.rectangle(display,
                     (thumb_x - thumb_size//2, slider_y - thumb_size//2),
                     (thumb_x + thumb_size//2, slider_y + slider_height + thumb_size//2),
                     (255, 255, 255), -1)
        
        controls_y += 40
        
        # Add instructions for slider control
        cv2.putText(display, "Click slider to select", (panel_start_x, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(display, "Left/Right arrow keys to adjust", (panel_start_x, controls_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Store control positions for mouse interaction
        self.led2_slider_bounds = {
            'x': panel_start_x,
            'y': slider_y,
            'width': panel_width,
            'height': slider_height
        }
    
    def handle_mouse_event(self, event, x, y, flags, param):
        """Handle mouse interaction with LED controls"""
        if not self.show_led_controls:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within LED1 slider bounds (if point projection enabled)
            if self.enable_point_projection:
                bounds1 = self.led1_slider_bounds
                if (bounds1['x'] <= x <= bounds1['x'] + bounds1['width'] and
                    bounds1['y'] - 5 <= y <= bounds1['y'] + bounds1['height'] + 5):
                    # Activate LED1 control
                    self.active_control = 'led1'
                    
                    # Set LED1 intensity based on click position
                    relative_pos = (x - bounds1['x']) / bounds1['width']
                    new_intensity = round(relative_pos * self.MAX_LED_INTENSITY / self.LED_STEP) * self.LED_STEP
                    self.set_led1_intensity(new_intensity)
                    return
            
            # Check if click is within LED2 slider bounds
            bounds2 = self.led2_slider_bounds
            if (bounds2['x'] <= x <= bounds2['x'] + bounds2['width'] and
                bounds2['y'] - 5 <= y <= bounds2['y'] + bounds2['height'] + 5):
                # Activate LED2 control
                self.active_control = 'led2'
                
                # Set LED2 intensity based on click position
                relative_pos = (x - bounds2['x']) / bounds2['width']
                new_intensity = round(relative_pos * self.MAX_LED_INTENSITY / self.LED_STEP) * self.LED_STEP
                self.set_led2_intensity(new_intensity)
    
    def handle_key_event(self, key):
        """Handle keyboard controls for LED intensity"""
        if not self.show_led_controls or self.active_control is None:
            return False
        
        # Skip processing of empty key (255)
        if key == 255:
            return False
            
        if self.active_control == 'led1':
            if key == 27:  # ESC key
                # Deactivate control
                self.active_control = None
                return True
            # Left arrow - decrease LED1 intensity
            elif key in [81, 2424832, 2, 65361, 37, 63234, 105, 226]:  # Common left arrow codes
                new_intensity = max(self.MIN_LED_INTENSITY, self.led1_intensity - self.LED_STEP)
                self.set_led1_intensity(new_intensity)
                return True
            # Right arrow - increase LED1 intensity 
            elif key in [83, 2555904, 3, 65363, 39, 63235, 106, 227]:  # Common right arrow codes
                new_intensity = min(self.MAX_LED_INTENSITY, self.led1_intensity + self.LED_STEP)
                self.set_led1_intensity(new_intensity)
                return True
            # Alternative keys: A/D for left/right
            elif key == ord('a'):  # 'a' key as alternative to left arrow
                new_intensity = max(self.MIN_LED_INTENSITY, self.led1_intensity - self.LED_STEP)
                self.set_led1_intensity(new_intensity)
                return True
            elif key == ord('d'):  # 'd' key as alternative to right arrow
                new_intensity = min(self.MAX_LED_INTENSITY, self.led1_intensity + self.LED_STEP)
                self.set_led1_intensity(new_intensity)
                return True
        elif self.active_control == 'led2':
            if key == 27:  # ESC key
                # Deactivate control
                self.active_control = None
                return True
            # More comprehensive key code handling for different keyboard layouts/platforms
            # Left arrow - decrease LED2 intensity
            elif key in [81, 2424832, 2, 65361, 37, 63234, 105, 226]:  # Common left arrow codes
                # Decrease LED2 intensity
                new_intensity = max(self.MIN_LED_INTENSITY, self.led2_intensity - self.LED_STEP)
                self.set_led2_intensity(new_intensity)
                return True
            # Right arrow - increase LED2 intensity 
            elif key in [83, 2555904, 3, 65363, 39, 63235, 106, 227]:  # Common right arrow codes
                # Increase LED2 intensity
                new_intensity = min(self.MAX_LED_INTENSITY, self.led2_intensity + self.LED_STEP)
                self.set_led2_intensity(new_intensity)
                return True
            # Alternative keys: A/D for left/right
            elif key == ord('a'):  # 'a' key as alternative to left arrow
                new_intensity = max(self.MIN_LED_INTENSITY, self.led2_intensity - self.LED_STEP)
                self.set_led2_intensity(new_intensity)
                return True
            elif key == ord('d'):  # 'd' key as alternative to right arrow
                new_intensity = min(self.MAX_LED_INTENSITY, self.led2_intensity + self.LED_STEP)
                self.set_led2_intensity(new_intensity)
                return True
                
        return False
        
    def set_led1_intensity(self, intensity):
        "Set LED1 intensity for point projection"
        intensity = max(self.MIN_LED_INTENSITY, min(self.MAX_LED_INTENSITY, intensity))
        self.led1_intensity = int(intensity)
        
        # Update physical LED if controller is available
        if self.led_controller is not None and self.led_controller.led_available:
            self.led_controller.set_intensity(self.led1_intensity, self.led2_intensity)
        
        # Show notification
        self.add_notification(f"LED1 intensity: {self.led1_intensity}mA (point projection)", (255, 165, 0))
    
    def set_led2_intensity(self, intensity):
        """Set LED2 intensity and update controller if available"""
        intensity = max(self.MIN_LED_INTENSITY, min(self.MAX_LED_INTENSITY, intensity))
        self.led2_intensity = int(intensity)
        
        # Update physical LED if controller is available
        if self.led_controller is not None and self.led_controller.led_available:
            self.led_controller.set_intensity(self.led1_intensity, self.led2_intensity)
        
        # Show notification
        self.add_notification(f"LED2 intensity: {self.led2_intensity}mA", (0, 255, 0))


class OpenCVHandposeDemo:
    """Enhanced handpose demo using OpenCV GUI with advanced display patterns."""
    
    def __init__(self, calibration_file: Optional[str] = None, use_led: bool = True, 
                 led1_intensity: int = 50, led2_intensity: int = 50, 
                 auto_led_adjust: bool = False, auto_led_hand_control: bool = False, 
                 enable_point_projection: bool = True, timeout: int = 5):
        """
        Initialize enhanced demo application.
        
        Args:
            calibration_file: Optional path to calibration file
            use_led: Enable LED control
            led2_intensity: Initial LED2 intensity (0-450 mA)
            auto_led_adjust: Auto-adjust LED based on lighting conditions
            auto_led_hand_control: Auto-activate LED only when hands detected
            timeout: Discovery timeout in seconds
        """
        self.calibration_file = calibration_file
        self.use_led = use_led
        self.led1_intensity = led1_intensity
        self.led2_intensity = led2_intensity
        self.auto_led_adjust = auto_led_adjust
        self.auto_led_hand_control = auto_led_hand_control
        self.enable_point_projection = enable_point_projection
        self.timeout = timeout
        
        # Connection and components
        self.client = None
        self.hand_tracker = None
        self.gesture_recognizer = None
        self.led_controller = None
        self.connected = False
        self.running = False
        
        # Camera info
        self.cameras = []
        self.left_camera_id = None
        self.right_camera_id = None
        
        # Enhanced UI
        self.ui = None
        self.show_skeleton = True
        
        # Tracking state
        self.current_gesture = "No Gesture"
        self.gesture_confidence = 0.0
        self.hands_count = 0
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()
        
        # Images for display
        self.left_image = None
        self.right_image = None
        self.display_image = None
        
        # Frame streaming
        self.frame_buffer = {'left': None, 'right': None}
        self.frame_lock = {'left': False, 'right': False}
        
        # LED state
        self.led_active = False
        self.last_hand_detection_time = 0
        self.no_hands_frames = 0
        self.hand_detection_timeout = 1.5  # seconds
        
        # Connect to scanner
        self.connect()
    
    def create_windows(self):
        """Create enhanced OpenCV windows."""
        # Main display window - enhanced full-screen
        cv2.namedWindow('UnLook Enhanced Handpose Demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('UnLook Enhanced Handpose Demo', 1920, 1080)
        
        # Set mouse callback for LED controls
        if self.ui:
            cv2.setMouseCallback('UnLook Enhanced Handpose Demo', self.ui.handle_mouse_event)
    
    def get_memory_usage(self):
        """Get current memory usage of the process in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0
    
    def connect(self):
        """Connect to the UnLook server using enhanced auto-discovery."""
        logger.info("Initializing UnLook client...")
        
        try:
            # Create client with auto-discovery disabled initially
            self.client = UnlookClient(auto_discover=False)
            
            # Start discovery
            self.client.start_discovery()
            logger.info(f"Discovering scanners for {self.timeout} seconds...")
            time.sleep(self.timeout)
            
            # Get discovered scanners
            scanners = self.client.get_discovered_scanners()
            if not scanners:
                logger.error("No scanners found. Check that your hardware is connected and powered on.")
                return False
            
            # Connect to the first scanner
            scanner_info = scanners[0]
            logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
            
            if not self.client.connect(scanner_info):
                logger.error("Failed to connect to scanner.")
                return False
            
            logger.info(f"Successfully connected to scanner: {scanner_info.name}")
            self.connected = True
            
            # Get cameras
            self.cameras = self.client.camera.get_cameras()
            logger.info(f"Found {len(self.cameras)} cameras")
            
            # Identify stereo cameras (assuming first two are left/right)
            if len(self.cameras) >= 2:
                self.left_camera_id = self.cameras[0]['id']
                self.right_camera_id = self.cameras[1]['id']
                camera_names = [self.cameras[0]['name'], self.cameras[1]['name']]
                logger.info(f"Using cameras: {camera_names[0]} (left), {camera_names[1]} (right)")
            else:
                logger.error(f"Need at least 2 cameras, found {len(self.cameras)}")
                self.connected = False
                return False
            
            # Initialize components
            self.initialize_components()
            
            return True
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    def initialize_components(self):
        """Initialize enhanced tracking components."""
        try:
            # Initialize LED controller through server if requested
            if self.use_led:
                try:
                    # Create LED controller instance
                    self.led_controller = LEDController(self.client)
                    
                    # Check if LED is available
                    if self.led_controller.led_available:
                        # If auto hand detection control is disabled, turn on LED immediately
                        # Otherwise, it will be activated only when hands are detected
                        # Set both LEDs to their configured intensities and keep them on 
                        led1_val = self.led1_intensity if self.enable_point_projection else 0
                        if self.led_controller.set_intensity(led1_val, self.led2_intensity):
                            logger.info(f"LEDs activated on server (LED1={led1_val}mA, LED2={self.led2_intensity}mA)")
                            self.led_active = True
                            logger.info("Both LEDs are now active and will remain on unless changed via sliders")
                        else:
                            logger.warning("Failed to activate LEDs on server")
                            self.use_led = False
                    else:
                        logger.warning("LED control not available on this scanner")
                        self.use_led = False
                except Exception as e:
                    logger.error(f"Failed to initialize LED controller: {e}")
                    self.use_led = False
            
            # Initialize enhanced UI
            self.ui = GestureUI(presentation_mode=False, enable_point_projection=self.enable_point_projection)
            
            # Give UI access to the LED controller
            if self.use_led:
                self.ui.led_controller = self.led_controller
                self.ui.led1_intensity = self.led1_intensity
                self.ui.led2_intensity = self.led2_intensity
            
            # Get auto-loaded calibration if no calibration file specified
            if not self.calibration_file:
                self.calibration_file = self.client.camera.get_calibration_file_path()
                if self.calibration_file:
                    logger.info(f"Using auto-loaded calibration: {self.calibration_file}")
                else:
                    logger.warning("No calibration file available")
            
            # Initialize hand tracker with improved settings for reliable detection
            self.hand_tracker = HandTracker(
                calibration_file=self.calibration_file, 
                max_num_hands=2,  # Focus on primary hands (reduces false positives)
                detection_confidence=0.6,  # Higher threshold for more reliable detection
                tracking_confidence=0.6,   # Higher threshold for more stable tracking
                left_camera_mirror_mode=False,  # Left camera is world-facing in UnLook setup
                right_camera_mirror_mode=False  # Right camera is world-facing in UnLook setup
            )
            
            # Initialize gesture recognizer
            self.gesture_recognizer = GestureRecognizer(gesture_threshold=0.7)
            
            # Log initialization information
            logger.info(f"Hand tracker initialized with calibration file: {self.calibration_file}")
            logger.info("Enhanced components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced components: {e}")
    
    def frame_callback_left(self, frame, metadata):
        """Callback for left camera frames."""
        self.frame_buffer['left'] = frame
        self.frame_lock['left'] = True
        
    def frame_callback_right(self, frame, metadata):
        """Callback for right camera frames."""
        self.frame_buffer['right'] = frame
        self.frame_lock['right'] = True
    
    def start_streaming(self):
        """Start camera streaming with callbacks."""
        logger.info("Starting camera streams...")
        self.client.stream.start(self.left_camera_id, self.frame_callback_left, fps=30)
        self.client.stream.start(self.right_camera_id, self.frame_callback_right, fps=30)
        
        # Give streams time to start
        time.sleep(0.5)
        
        # Auto-adjust LED intensity if requested
        if self.use_led and self.auto_led_adjust:
            # Wait for initial frames 
            timeout_count = 0
            while not (self.frame_lock['left'] and self.frame_lock['right']):
                time.sleep(0.1)
                timeout_count += 1
                if timeout_count > 30:  # 3 seconds timeout
                    logger.warning("Timeout waiting for frames for LED calibration")
                    break
            
            # Get frames for calibration
            frame_left_cal = self.frame_buffer['left']
            frame_right_cal = self.frame_buffer['right']
            
            # Reset locks
            self.frame_lock['left'] = False 
            self.frame_lock['right'] = False
            
            if frame_left_cal is not None and frame_right_cal is not None:
                try:
                    # Calibrate LED based on image analysis
                    optimal_intensity = calculate_optimal_led_intensity(frame_left_cal, frame_right_cal)
                    
                    # Set LED to calculated optimal intensity using the controller
                    if self.led_controller.set_intensity(0, optimal_intensity):
                        # Store the optimal intensity for LED2 (LED1 is always 0)
                        self.led2_intensity = optimal_intensity
                        logger.info(f"Auto-adjusted LED2 intensity to {self.led2_intensity}mA (LED1 remains at 0mA)")
                        self.ui.add_notification(f"LED2 optimized: {self.led2_intensity}mA", color=(0, 255, 0))
                    else:
                        self.ui.add_notification("Couldn't set LED intensity", color=(255, 255, 0))
                except Exception as e:
                    logger.error(f"LED calibration error: {e}")
                    self.ui.add_notification(f"Using default LED: {self.led2_intensity}mA", color=(255, 255, 0))
    
    def process_frames(self):
        """Process current frame buffer for hand tracking."""
        try:
            # Get frames
            frame_left = self.frame_buffer['left'].copy() if self.frame_buffer['left'] is not None else None
            frame_right = self.frame_buffer['right'].copy() if self.frame_buffer['right'] is not None else None
            
            if frame_left is None or frame_right is None:
                return None
            
            # Store original frames for display
            self.left_image = frame_left.copy()
            self.right_image = frame_right.copy()
            
            # Preprocess frames for enhanced quality
            frame_left_enhanced = preprocess_image(frame_left)
            frame_right_enhanced = preprocess_image(frame_right)
            
            # Track hands in 3D using enhanced images
            results = self.hand_tracker.track_hands_3d(
                frame_left_enhanced, 
                frame_right_enhanced
            )
            
            # Update hand trajectory for visualization
            hands_detected = results['3d_keypoints'] and len(results['3d_keypoints']) > 0
            
            # Manage LED based on hand detection (only if auto LED hand control is enabled)
            if hands_detected:
                # Get the first hand's wrist position
                hand_position = results['3d_keypoints'][0][0, :2]  # X,Y only
                
                # Update trajectory
                self.ui.update_hand_trajectory(hand_position)
                
                # Update hand detection time and manage LED (if auto LED hand control is enabled)
                self.last_hand_detection_time = time.time()
                self.no_hands_frames = 0
                
                # Optional: Turn on LED if auto hand control is enabled
                if self.use_led and self.led_controller and not self.led_active and self.auto_led_hand_control:
                    led1_val = self.led1_intensity if self.enable_point_projection else 0
                    if self.led_controller.set_intensity(led1_val, self.led2_intensity):
                        self.led_active = True
                        logger.info(f"Hand detected: LEDs activated (LED1={led1_val}mA, LED2={self.led2_intensity}mA)")
                        self.ui.add_notification("Hand detected - LEDs activated", color=(0, 255, 0))
            elif self.auto_led_hand_control:
                # Count frames without hands (only if auto LED hand control is enabled)
                self.no_hands_frames += 1
            
            # Turn off LED if no hands for timeout period (only if auto LED hand control is enabled)
            if (self.use_led and self.led_controller and self.led_active and self.auto_led_hand_control and
                time.time() - self.last_hand_detection_time > self.hand_detection_timeout):
                if self.led_controller.set_intensity(0, 0):  # Turn off both LEDs
                    self.led_active = False
                    logger.info("No hands detected for timeout period: LEDs deactivated")
                    self.ui.add_notification("No hands - LEDs deactivated", color=(255, 255, 0))
            
            # Process gestures for enhanced display
            processed_gestures = []
            if '3d_keypoints' in results and results['3d_keypoints']:
                for i, keypoints_3d in enumerate(results['3d_keypoints']):
                    handedness = None
                    if results.get('handedness_left') and i < len(results['handedness_left']):
                        handedness = results['handedness_left'][i]
                    elif results.get('handedness_right') and i < len(results['handedness_right']):
                        handedness = results['handedness_right'][i]
                    
                    gesture_type, confidence = self.gesture_recognizer.recognize_gestures_3d(
                        keypoints_3d, handedness)
                    
                    if gesture_type != GestureType.UNKNOWN and confidence > 0.7:
                        gesture_name = gesture_type.value.replace('_', ' ').title()
                        processed_gestures.append({
                            'type': gesture_type,
                            'name': gesture_name,
                            'confidence': confidence
                        })
                        # Store for legacy access
                        self.current_gesture = gesture_name
                        self.gesture_confidence = confidence
                    else:
                        # Store for legacy access
                        self.current_gesture = "No Gesture"
                        self.gesture_confidence = confidence
            else:
                self.current_gesture = "No Gesture"
                self.gesture_confidence = 0.0
            
            # Add processed gestures to results
            results['gestures'] = processed_gestures
            
            # Count hands for legacy access
            left_count = len(results.get('2d_left', []))
            right_count = len(results.get('2d_right', []))
            self.hands_count = left_count + right_count
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frames: {e}")
            return None
    
    def create_enhanced_display(self, results):
        """Create enhanced display using the new UI system."""
        if self.left_image is not None and self.right_image is not None:
            # Create enhanced display with gesture recognition results
            display = self.ui.create_display(
                self.left_image, 
                self.right_image, 
                results,
                show_skeleton=self.show_skeleton
            )
            
            # Store display
            self.display_image = display
            
            # Show the enhanced display
            cv2.imshow('UnLook Enhanced Handpose Demo', display)
        else:
            # Show placeholder if no frames available
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "UnLook Enhanced Handpose Demo", (400, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Waiting for camera data...", (450, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('UnLook Enhanced Handpose Demo', placeholder)
    
    def disconnect(self):
        """Disconnect from server and cleanup."""
        logger.info("Disconnecting...")
        
        self.running = False
        self.connected = False
        
        # Cleanup components
        if self.hand_tracker:
            try:
                self.hand_tracker.close()
            except Exception as e:
                logger.error(f"Error closing hand tracker: {e}")
        
        if self.led_controller:
            try:
                # Turn off LED before disconnecting
                self.led_controller.set_intensity(0, 0)
                logger.info("LED turned off")
            except Exception as e:
                logger.error(f"Error stopping LED controller: {e}")
        
        if self.client:
            try:
                # Stop streams
                if hasattr(self.client, 'stream'):
                    self.client.stream.stop_all()
                self.client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting client: {e}")
        
        self.client = None
        self.hand_tracker = None
        self.gesture_recognizer = None
        self.led_controller = None
        self.ui = None
        
        logger.info("Disconnected")
    
    def run(self):
        """Run the enhanced main application loop."""
        if not self.connected:
            logger.error("Not connected to scanner. Cannot start demo.")
            return
        
        # Initialize display window
        logger.info("Creating UI window...")
        self.create_windows()
        
        # Start streaming
        self.start_streaming()
        
        print("\nStarting UnLook Enhanced Handpose Demo...")
        print("Press 'q' to quit")
        print("Press 's' to toggle hand skeleton")
        print("Press 'l' to toggle manual LED intensity controls")
        print("Press 'r' to reset LED2 intensity")
        
        # Initialize display settings
        frame_count = 0
        fps_timer = time.time()
        fps_count = 0
        current_fps = 0
        
        try:
            # Main UI loop
            while True:
                loop_start_time = time.time()
                
                # Wait for both frames
                if self.frame_lock['left'] and self.frame_lock['right']:
                    # Reset locks
                    self.frame_lock['left'] = False
                    self.frame_lock['right'] = False
                    
                    # Process frames
                    results = self.process_frames()
                    
                    if results is not None:
                        # Create enhanced display
                        self.create_enhanced_display(results)
                        
                        # Calculate FPS
                        frame_count += 1
                        if frame_count % 30 == 0:
                            current_time = time.time()
                            elapsed = current_time - fps_timer
                            self.fps = frame_count / elapsed
                            frame_count = 0
                            fps_timer = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                # First check UI-specific key events
                if self.ui and self.ui.handle_key_event(key):
                    continue  # Key was handled by UI
                
                # Handle main application keys
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.show_skeleton = not self.show_skeleton
                    logger.info(f"Hand skeleton {'enabled' if self.show_skeleton else 'disabled'}")
                elif key == ord('l'):
                    if self.ui:
                        self.ui.show_led_controls = not self.ui.show_led_controls
                        logger.info(f"LED controls {'shown' if self.ui.show_led_controls else 'hidden'}")
                elif key == ord('r'):
                    # Reset LED2 to default intensity
                    if self.ui and self.led_controller:
                        self.ui.set_led2_intensity(200)
                        logger.info("LED2 intensity reset to 200mA")
                
                # Show placeholder if no display available
                if self.display_image is None:
                    self.create_enhanced_display({})
                
                # Control frame rate - aim for ~30 FPS in UI thread
                time.sleep(0.033)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.disconnect()
            cv2.destroyAllWindows()


def main():
    """Main function for enhanced demo application."""
    parser = argparse.ArgumentParser(description='UnLook Enhanced Handpose Demo with OpenCV GUI')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to calibration file (optional)')
    parser.add_argument('--no-led', action='store_true',
                       help='Disable LED control')
    parser.add_argument('--led1-intensity', type=int, default=50,
                       help='Initial LED1 intensity for point projection (0-450 mA)')
    parser.add_argument('--led2-intensity', type=int, default=50,
                       help='Initial LED2 intensity for flood illumination (0-450 mA)')
    parser.add_argument('--no-point-projection', action='store_true',
                       help='Disable LED1 point projection')
    parser.add_argument('--auto-led-adjust', action='store_true',
                       help='Enable automatic LED adjustment based on lighting')
    parser.add_argument('--auto-led-hand-control', action='store_true',
                       help='Enable automatic LED activation based on hand presence')
    parser.add_argument('--timeout', type=int, default=5,
                       help='Scanner discovery timeout in seconds')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Create and run enhanced demo
    demo = OpenCVHandposeDemo(
        calibration_file=args.calibration,
        use_led=not args.no_led,
        led1_intensity=args.led1_intensity,
        led2_intensity=args.led2_intensity,
        auto_led_adjust=args.auto_led_adjust,
        auto_led_hand_control=args.auto_led_hand_control,
        enable_point_projection=not args.no_point_projection,
        timeout=args.timeout
    )
    demo.run()


if __name__ == "__main__":
    main()