#!/usr/bin/env python3
"""Enhanced Gesture Recognition Demo with Visual Feedback.

This demo shows real-time gesture recognition with clear visual feedback
and notification overlays. Supports common gestures including:
- Open palm
- Closed fist
- Pointing
- Peace sign
- Thumbs up/down
- Pinch
- Swipe gestures

Features:
- Automatic LED intensity adjustment
- Clear on-screen gesture visualization
- Direction detection for swipe gestures
- Works with UnLook stereo cameras
- Optional YOLOv10x integration for enhanced hand tracking and gesture recognition
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
from enum import Enum
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import UnLook SDK
from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker, GestureType
from unlook.client.scanning.handpose.dynamic_gesture_recognizer import DynamicGestureRecognizer, DynamicEvent
from unlook.client.projector import LEDController

# Try importing required packages for YOLOv10x
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv10x requires PyTorch and Ultralytics. Install with: pip install torch ultralytics")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_image(image):
    """
    Enhance image for better hand detection in poor lighting.
    
    This function:
    1. Increases brightness and contrast
    2. Applies adaptive histogram equalization for better feature visibility
    3. Applies light bilateral filtering to reduce noise while preserving edges
    
    Args:
        image: Input BGR image
        
    Returns:
        Enhanced image
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
    Calculate optimal LED2 intensity based on image brightness.
    
    This function analyzes the current images and calculates an optimal
    LED2 intensity without requiring multiple LED adjustments.
    LED1 will always be set to 0 regardless of this calculation
    (for hardware safety reasons).
    
    Args:
        frame_left: Left camera frame
        frame_right: Right camera frame
        
    Returns:
        Optimal LED2 intensity in mA (LED1 will be 0)
    """
    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    brightness_left = np.mean(gray_left)
    brightness_right = np.mean(gray_right)
    avg_brightness = (brightness_left + brightness_right) / 2
    
    # Desired brightness for hand tracking (empirically determined)
    target_brightness = 100  # Target grayscale value (0-255)
    
    # Calculate adjustment factor (inverse relation - brighter image means lower LED needed)
    # We want to increase intensity if image is too dark, decrease if too bright
    adjustment_factor = target_brightness / max(1, avg_brightness)
    
    # Range: valid LED intensity range (mA)
    min_intensity = 50
    max_intensity = 450
    
    # Calculate optimal intensity - start from a baseline of 150mA
    optimal_intensity = int(150 * adjustment_factor)
    
    # Clamp to valid range
    optimal_intensity = max(min_intensity, min(max_intensity, optimal_intensity))
    
    # Quantize to steps of 50mA for better control
    optimal_intensity = int(round(optimal_intensity / 50.0) * 50)
    
    logger.info(f"Image brightness: {avg_brightness:.1f}, Target: {target_brightness}")
    logger.info(f"Adjustment factor: {adjustment_factor:.2f}")
    logger.info(f"Calculated LED2 intensity: {optimal_intensity}mA (LED1 will be 0)")
    
    return optimal_intensity


def auto_adjust_led_intensity(client, frame_left, frame_right, current_intensity=200, steps=5):
    """
    Automatically adjust LED2 intensity for optimal hand detection.
    
    This function:
    1. Analyzes the current frames
    2. Calculates the optimal intensity for LED2
    3. Applies the optimal intensity while keeping LED1 at 0
    
    Args:
        client: UnlookClient instance
        frame_left: Left camera frame
        frame_right: Right camera frame
        current_intensity: Current LED2 intensity in mA
        steps: Number of intensity levels to test (unused in current implementation)
        
    Returns:
        Optimal LED2 intensity in mA (LED1 will be 0)
    """
    # Use the LEDController if available
    try:
        led_controller = LEDController(client)
        
        # Calculate the optimal value from current frames
        optimal_intensity = calculate_optimal_led_intensity(frame_left, frame_right)
        
        logger.info(f"Optimal LED2 intensity: {optimal_intensity}mA (LED1 will be 0)")
        
        # Set optimal intensity using the controller
        if led_controller.led_available:
            led_controller.set_intensity(0, optimal_intensity)
    except Exception as e:
        logger.error(f"Error adjusting LED intensity: {e}")
    
    return optimal_intensity


def evaluate_image_quality(image):
    """
    Evaluate image quality for hand detection.
    
    This function calculates a score based on:
    1. Image contrast (higher is better)
    2. Image brightness (medium is better)
    3. Edge detail (higher is better)
    
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
    """Handles visualization and UI for gesture recognition demo."""
    
    # Define LED settings
    MIN_LED_INTENSITY = 0   # Allow 0 for completely off
    MAX_LED_INTENSITY = 450
    LED_STEP = 5           # Fine-grained steps for LED control
    
    def __init__(self):
        # Load gesture icons
        self.icons = {}
        self.icon_size = (100, 100)  # Default icon size
        
        # LED control interface
        self.show_led_controls = False
        self.led1_intensity = 0     # LED1 is always 0 as per safety requirements
        self.led2_intensity = 200   # Default starting point
        self.active_control = None  # Which control is active
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
            
            # New dynamic gestures
            GestureType.SWIPE_RIGHT: (100, 255, 100),  # Light green
            GestureType.SWIPE_LEFT: (100, 100, 255),   # Light blue
            GestureType.SWIPE_UP: (255, 100, 100),     # Light red
            GestureType.SWIPE_DOWN: (255, 255, 100),   # Light yellow
            GestureType.CIRCLE: (100, 255, 255)        # Light cyan
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
            
            # New dynamic gestures
            GestureType.SWIPE_RIGHT: "Swipe Right - Next",
            GestureType.SWIPE_LEFT: "Swipe Left - Previous",
            GestureType.SWIPE_UP: "Swipe Up - Increase",
            GestureType.SWIPE_DOWN: "Swipe Down - Decrease",
            GestureType.CIRCLE: "Circle - Rotate/Loop"
        }
        
        # Initialize notification system
        self.notification = None
        self.notification_color = (255, 255, 255)
        self.notification_start_time = 0
        self.notification_duration = 2.0  # seconds
        
        # Initialize enhanced swipe detection
        self.prev_hand_position = None
        self.swipe_threshold = 100  # mm
        self.swipe_cooldown = 0
        self.swipe_min_velocity = 500  # mm/s
        self.last_swipe_time = 0
        self.consecutive_direction_frames = 0
        self.last_direction = None
        
        # Hand orientation tracking for rotation-invariant swipe detection
        self.hand_orientation = None  # Will track wrist-to-middle-finger orientation
        
        # LED controller reference (will be set later)
        self.led_controller = None
        
        # Hand trajectory tracking
        self.hand_trajectory = []
        self.max_trajectory_points = 20
        
        # Multi-point trajectory for better swipe detection
        self.trajectory_history = []
        self.trajectory_max_length = 10  # Store last 10 positions for robust swipe detection
        self.trajectory_timestamps = []
        self.trajectory_max_time = 0.5   # Only consider positions within last 0.5 seconds
    
    def add_notification(self, text, color=(255, 255, 255), duration=2.0):
        """Add a temporary notification to the screen."""
        self.notification = text
        self.notification_color = color
        self.notification_start_time = time.time()
        self.notification_duration = duration
    
    def update_hand_trajectory(self, hand_position, keypoints=None):
        """Update the hand trajectory for visualization and swipe detection.
        
        Args:
            hand_position: 2D or 3D position of hand center/wrist
            keypoints: Full hand keypoints (optional) for orientation detection
        """
        if hand_position is not None:
            # Add to visualization trajectory
            self.hand_trajectory.append(hand_position.copy())
            if len(self.hand_trajectory) > self.max_trajectory_points:
                self.hand_trajectory.pop(0)
            
            # Add to swipe detection trajectory with timestamp
            current_time = time.time()
            self.trajectory_history.append(hand_position.copy())
            self.trajectory_timestamps.append(current_time)
            
            # Remove old trajectory points beyond our time window
            while (len(self.trajectory_timestamps) > 1 and 
                   current_time - self.trajectory_timestamps[0] > self.trajectory_max_time):
                self.trajectory_history.pop(0)
                self.trajectory_timestamps.pop(0)
            
            # Update hand orientation if keypoints provided
            if keypoints is not None and len(keypoints) >= 21:
                # Calculate orientation vector from wrist to middle finger MCP
                wrist = keypoints[0]
                middle_mcp = keypoints[9]  # Middle finger base
                
                if wrist is not None and middle_mcp is not None:
                    # Orientation vector (normalized)
                    orientation = middle_mcp - wrist
                    norm = np.linalg.norm(orientation)
                    if norm > 0:
                        self.hand_orientation = orientation / norm
    
    def detect_swipe(self, hand_position):
        """Enhanced swipe detection that works across different hand positions and rotations.
        Uses trajectory history and velocity calculations for robustness.
        
        Args:
            hand_position: Current hand position (wrist or palm center)
            
        Returns:
            Detected swipe direction or None
        """
        current_time = time.time()
        
        # Basic validation
        if hand_position is None or len(self.trajectory_history) < 3:
            return None
        
        # Skip if we're in cooldown period
        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1
            return None
            
        # Skip if too soon after last swipe
        if current_time - self.last_swipe_time < 1.0:  # 1 second minimum between swipes
            return None
        
        # Get trajectory over last several frames
        positions = self.trajectory_history
        timestamps = self.trajectory_timestamps
        
        # Need at least 3 points for reliable velocity calculation
        if len(positions) < 3 or len(timestamps) < 3:
            return None
        
        # Method 1: Calculate average velocity vector over multiple frames
        # This makes detection more robust to small hand tremors
        start_pos = positions[0]
        end_pos = positions[-1]
        time_diff = timestamps[-1] - timestamps[0]  # seconds
        
        if time_diff <= 0.05:  # Avoid division by too small time differences
            return None
            
        # Calculate velocity vector in mm/s
        velocity_vector = (end_pos - start_pos) / time_diff
        speed = np.linalg.norm(velocity_vector)  # mm/s
        
        # Method 2: Check for consistent direction across multiple frames
        # This helps filter out erratic movements
        consistent_direction = self._check_consistent_trajectory(positions)
        
        # Method 3: Check acceleration - swipes have characteristic acceleration pattern
        # Higher acceleration at start, then consistent velocity
        has_swipe_accel = self._check_swipe_acceleration(positions, timestamps)
        
        # Method 4: Adjust for hand orientation if available
        # This makes swipe detection work regardless of hand rotation
        adjusted_velocity = velocity_vector
        if self.hand_orientation is not None:
            adjusted_velocity = self._adjust_for_hand_orientation(velocity_vector)
        
        # Primary detection logic: need both sufficient speed and consistent direction
        swipe_direction = None
        if speed > self.swipe_min_velocity and consistent_direction and has_swipe_accel:
            # Determine direction using the adjusted velocity
            if abs(adjusted_velocity[0]) > abs(adjusted_velocity[1]):
                # Horizontal swipe
                if adjusted_velocity[0] > 0:
                    swipe_direction = "right"
                else:
                    swipe_direction = "left"
            else:
                # Vertical swipe
                if adjusted_velocity[1] > 0:
                    swipe_direction = "down"
                else:
                    swipe_direction = "up"
            
            # Set cooldown to prevent multiple detections
            self.swipe_cooldown = 20  # frames - longer cooldown for robust detection
            self.last_swipe_time = current_time
            
            # Add notification with speed info
            self.add_notification(
                f"Swipe {swipe_direction.upper()} detected! ({speed:.0f} mm/s)", 
                color=(0, 255, 255)
            )
            
            return swipe_direction
        
        return None
    
    def _check_consistent_trajectory(self, positions):
        """Check if the trajectory has a consistent direction."""
        if len(positions) < 3:
            return False
            
        # Calculate direction vectors between consecutive points
        directions = []
        for i in range(1, len(positions)):
            vec = positions[i] - positions[i-1]
            norm = np.linalg.norm(vec)
            if norm > 0.001:  # Avoid dividing by zero or very small movements
                directions.append(vec / norm)  # Normalized direction vector
        
        if not directions:
            return False
            
        # Get primary direction (from oldest to newest point)
        primary_dir = positions[-1] - positions[0]
        primary_norm = np.linalg.norm(primary_dir)
        if primary_norm < 0.001:
            return False
            
        primary_dir = primary_dir / primary_norm
        
        # Check how many direction vectors align with primary direction
        alignment_count = 0
        for dir_vec in directions:
            # Dot product > 0.7 means vectors are pointing in roughly same direction (within ~45°)
            if np.dot(dir_vec, primary_dir) > 0.7:
                alignment_count += 1
                
        # If at least 70% of direction vectors align with primary direction
        return alignment_count >= 0.7 * len(directions)
    
    def _check_swipe_acceleration(self, positions, timestamps):
        """Check if the motion has a swipe-like acceleration pattern."""
        if len(positions) < 4 or len(timestamps) < 4:
            return False
            
        # Calculate speeds at different points in the trajectory
        speeds = []
        for i in range(1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[i-1])
            time_delta = timestamps[i] - timestamps[i-1]
            if time_delta > 0.001:  # Avoid division by very small time
                speeds.append(distance / time_delta)
        
        if len(speeds) < 3:
            return False
            
        # Get speeds at beginning, middle, and end
        start_speed = np.mean(speeds[:len(speeds)//3])
        mid_speed = np.mean(speeds[len(speeds)//3:2*len(speeds)//3])
        end_speed = np.mean(speeds[2*len(speeds)//3:])
        
        # Characteristic swipe has acceleration, then consistent or slight deceleration
        # mid_speed should be higher than start_speed, end_speed should be similar to or lower than mid_speed
        return (mid_speed > start_speed * 1.2 and  # Clear acceleration at start
                end_speed < mid_speed * 1.2)       # Plateaus or slightly decelerates
                
    def _adjust_for_hand_orientation(self, velocity_vector):
        """Adjust velocity vector based on hand orientation to make swipe detection rotation-invariant."""
        if self.hand_orientation is None:
            return velocity_vector
            
        # If we have hand orientation (wrist to middle MCP), we can make swipe detection invariant to rotation
        # For example, if hand is rotated 90 degrees, what was a "right" swipe is now a "down" swipe
        
        # Basic rotation matrix based on hand orientation
        # We define the "standard" orientation as wrist-to-middle pointing up (-y in image coordinates)
        standard_orientation = np.array([0, -1])  # Up direction
        
        # Get the angle between current orientation and standard orientation
        cos_angle = np.dot(self.hand_orientation[:2], standard_orientation)
        sin_angle = np.cross(self.hand_orientation[:2], standard_orientation)
        
        # Create rotation matrix to align hand orientation with standard
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # Apply rotation to get orientation-invariant velocity
        return np.dot(rotation_matrix, velocity_vector[:2])
    
    def create_display(self, frame_left, frame_right, results, show_skeleton=True):
        """Create a full-screen display with hand tracking and gesture recognition results."""
        # Get original dimensions
        orig_h, orig_w = frame_left.shape[:2]
        
        # Get the screen resolution (use first monitor)
        try:
            screen_w, screen_h = 1920, 1080  # Default Full HD
            # Try to get the actual screen resolution if possible
            screen = cv2.getWindowImageRect('UnLook Enhanced Gesture Recognition')
            if screen and len(screen) == 4:
                _, _, screen_w, screen_h = screen
        except:
            # If the window doesn't exist yet or other error, use defaults
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
        
        # Draw 3D hand trajectory
        if self.hand_trajectory and len(results.get('3d_keypoints', [])) > 0:
            # Project 3D trajectory onto left image (top camera now)
            trajectory_colors = [
                (int(255 * (1 - i/len(self.hand_trajectory))), 
                 int(255 * i/len(self.hand_trajectory)), 
                 255) 
                for i in range(len(self.hand_trajectory))
            ]
            
            for i in range(1, len(self.hand_trajectory)):
                # Project to 2D using first camera
                pt1 = results.get('projection', {}).get('left', [])
                if pt1:
                    # Simple projection using camera matrix with proper scaling to camera dimensions
                    x1 = int(self.hand_trajectory[i-1][0] * self.camera_width / orig_w)
                    y1 = int(self.hand_trajectory[i-1][1] * self.camera_height / orig_h) + self.top_y_offset
                    
                    x2 = int(self.hand_trajectory[i][0] * self.camera_width / orig_w)
                    y2 = int(self.hand_trajectory[i][1] * self.camera_height / orig_h) + self.top_y_offset
                    
                    cv2.line(display, (x1, y1), (x2, y2), trajectory_colors[i], 2)
        
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
        bottom_section_y = self.display_height - 150  # Adjusted to be below LED controls when they're visible
        cv2.putText(display, "CONTROLS:", (self.camera_width + 20, bottom_section_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "q - Quit", (self.camera_width + 30, bottom_section_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "s - Toggle skeleton", (self.camera_width + 30, bottom_section_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "r - Reset LED2", (self.camera_width + 30, bottom_section_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "l - Toggle LED controls", (self.camera_width + 30, bottom_section_y + 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(display, "Space - Capture gesture", (self.camera_width + 30, bottom_section_y + 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Add labels for the camera views - make them larger and more visible
        cv2.putText(display, "LEFT CAMERA", (10, self.top_y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "RIGHT CAMERA", (10, self.bottom_y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add YOLOv10x information if dynamic gestures are present
        if 'dynamic_gestures' in results and results['dynamic_gestures']:
            # Show YOLOv10x label in top-left corner
            cv2.putText(display, "YOLOv10x ACTIVE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw bounding boxes for each dynamic gesture
            for gesture in results['dynamic_gestures']:
                if 'bbox' in gesture and gesture['bbox'] is not None:
                    bbox = gesture['bbox']
                    if isinstance(bbox, np.ndarray) and bbox.size >= 4:
                        # Adjust bbox coordinates to display coordinates
                        h, w = frame_left.shape[:2]
                        x1 = int(bbox[0] * self.camera_width / w)
                        y1 = int(bbox[1] * self.camera_height / h) + self.top_y_offset
                        x2 = int(bbox[2] * self.camera_width / w)
                        y2 = int(bbox[3] * self.camera_height / h) + self.top_y_offset
                        
                        # Draw rectangle and label
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Add gesture name
                        label = gesture.get('name', 'Unknown')
                        conf = gesture.get('confidence', 0.0)
                        text = f"{label} ({conf:.2f})"
                        cv2.putText(display, text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
        
        # LED1 control (always 0, shown for reference only)
        cv2.putText(display, "LED1: 0 mA (fixed)", (panel_start_x, controls_y),
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
            # Check if click is within LED2 slider bounds
            bounds = self.led2_slider_bounds
            if (bounds['x'] <= x <= bounds['x'] + bounds['width'] and
                bounds['y'] - 5 <= y <= bounds['y'] + bounds['height'] + 5):
                # Activate LED2 control
                self.active_control = 'led2'
                
                # Set LED2 intensity based on click position
                relative_pos = (x - bounds['x']) / bounds['width']
                new_intensity = round(relative_pos * self.MAX_LED_INTENSITY / self.LED_STEP) * self.LED_STEP
                self.set_led2_intensity(new_intensity)
    
    def handle_key_event(self, key):
        """Handle keyboard controls for LED intensity"""
        if not self.show_led_controls or self.active_control is None:
            return False
        
        # Skip processing of empty key (255)
        if key == 255:
            return False
            
        if self.active_control == 'led2':
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
        
    def set_led2_intensity(self, intensity):
        """Set LED2 intensity and update controller if available"""
        intensity = max(self.MIN_LED_INTENSITY, min(self.MAX_LED_INTENSITY, intensity))
        self.led2_intensity = int(intensity)
        
        # Update physical LED if controller is available
        if self.led_controller is not None and self.led_controller.led_available:
            self.led_controller.set_intensity(0, self.led2_intensity)
        
        # Show notification
        self.add_notification(f"LED2 intensity: {self.led2_intensity}mA", (0, 255, 0))


def run_gesture_demo(calibration_file=None, timeout=10, verbose=False,
                     use_led=True, led1_intensity=0, led2_intensity=200, 
                     auto_led_adjust=True, yolo_model_path=None, yolo_hands_model_path=None,
                     use_yolo=True, performance_mode="balanced", downsample=1, fast_mode=False):
    # Note: led1_intensity parameter is kept for backward compatibility but is always set to 0
    """
    Run the enhanced gesture recognition demo with visual feedback.
    
    Args:
        calibration_file: Path to stereo calibration file
        timeout: Discovery timeout in seconds
        verbose: Enable verbose debug logging
        use_led: Enable LED illumination
        led1_intensity: LED1 intensity in mA (0-450)
        led2_intensity: LED2 intensity in mA (0-450)
        auto_led_adjust: Automatically adjust LED intensity
        yolo_model_path: Path to YOLOv10x_gestures.pt model
        yolo_hands_model_path: Path to YOLOv10x_hands.pt model
        use_yolo: Enable YOLOv10x models if available
        performance_mode: Performance mode (balanced, speed, accuracy)
    """
    # Initialize UnLook client
    logger.info("Initializing UnLook client...")
    client = UnlookClient(auto_discover=False)
    
    try:
        # Start discovery
        client.start_discovery()
        logger.info(f"Discovering scanners for {timeout} seconds...")
        time.sleep(timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Check that your hardware is connected and powered on.")
            return 1
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner.")
            return 1
        
        logger.info(f"Successfully connected to scanner: {scanner_info.name}")
        
        # Initialize LED control through server if requested
        led_controller = None
        if use_led:
            try:
                # Create LED controller instance
                led_controller = LEDController(client)
                
                # Check if LED is available
                if led_controller.led_available:
                    # Set initial intensity - LED1 will automatically be forced to 0 in the controller
                    if led_controller.set_intensity(0, led2_intensity):
                        logger.info(f"LED flood illuminator activated on server (LED1=0mA, LED2={led2_intensity}mA)")
                    else:
                        logger.warning("Failed to activate LED on server")
                        use_led = False
                else:
                    logger.warning("LED control not available on this scanner")
                    use_led = False
            except Exception as e:
                logger.error(f"Failed to initialize LED controller: {e}")
                use_led = False
        
        # Get auto-loaded calibration if no calibration file specified
        if not calibration_file:
            calibration_file = client.camera.get_calibration_file_path()
            if calibration_file:
                logger.info(f"Using auto-loaded calibration: {calibration_file}")
            else:
                logger.warning("No calibration file available")
        
        # Initialize dynamic gesture recognizer with YOLOv10x models if requested
        dynamic_recognizer = None
        if use_yolo and (yolo_model_path or yolo_hands_model_path):
            try:
                logger.info("Initializing dynamic gesture recognizer with YOLOv10x models")
                
                # Set model parameters based on performance mode
                imgsz = 320  # Default balanced mode size
                if performance_mode == "speed":
                    imgsz = 256  # Smaller size for faster processing
                elif performance_mode == "accuracy":
                    imgsz = 416  # Larger size for better accuracy
                    
                logger.info(f"Using {performance_mode} mode with image size {imgsz}px")
                
                # Create model config dictionary with optimized settings
                model_kwargs = {
                    "imgsz": imgsz,           # Image size for inference
                    "verbose": False,         # Disable verbose output
                    "conf": 0.65,             # Lower confidence threshold for better speed
                    "half": True,             # Use half-precision (FP16) for faster inference
                    "max_det": 4,             # Limit maximum detections (we only need a few hands)
                    "vid_stride": 2,          # Process every other frame for video (speeds up processing)
                    "iou": 0.5                # Higher IoU threshold for NMS (speeds up processing)
                }
                
                # Create dynamic recognizer with optimized settings - with error handling
                try:
                    # Try to create the recognizer with optimized settings first
                    dynamic_recognizer = DynamicGestureRecognizer(
                        yolo_model_path=yolo_model_path,
                        yolo_hands_model_path=yolo_hands_model_path,
                        model_kwargs=model_kwargs,
                        parallel_inference=True   # Enable parallel model loading
                    )
                except Exception as e:
                    logger.warning(f"Failed to create dynamic recognizer with parallel loading: {e}")
                    # Try again with more conservative settings
                    try:
                        safer_kwargs = {"imgsz": 320, "verbose": False, "half": False}
                        dynamic_recognizer = DynamicGestureRecognizer(
                            yolo_model_path=yolo_model_path,
                            yolo_hands_model_path=yolo_hands_model_path,
                            model_kwargs=safer_kwargs,
                            parallel_inference=False  # Disable parallel loading
                        )
                    except Exception as e:
                        logger.error(f"Failed to create dynamic recognizer even with safe settings: {e}")
                        dynamic_recognizer = None
                
                # Check which models were loaded
                models_loaded = []
                if dynamic_recognizer is not None:
                    if dynamic_recognizer.models_available and dynamic_recognizer.using_yolo:
                        models_loaded.append("gestures")
                        logger.info(f"YOLOv10x gestures model loaded successfully")
                    
                    if dynamic_recognizer.using_yolo_hands:
                        models_loaded.append("hands")
                        logger.info(f"YOLOv10x hands model loaded successfully")
                    
                    if not models_loaded:
                        logger.warning("Failed to load YOLOv10x models, falling back to standard recognition")
                        dynamic_recognizer = None
            except Exception as e:
                logger.error(f"Error initializing YOLOv10x models: {e}")
                dynamic_recognizer = None
        
        # Initialize hand tracker with improved settings for reliable detection
        # For the UnLook setup, both cameras are world-facing, so both should have mirror_mode=False
        # This ensures consistent handedness detection across cameras
        tracker = HandTracker(
            calibration_file=calibration_file, 
            max_num_hands=2,  # Focus on primary hands (reduces false positives)
            detection_confidence=0.6,  # Higher threshold for more reliable detection
            tracking_confidence=0.6,   # Higher threshold for more stable tracking
            left_camera_mirror_mode=False,  # Left camera is world-facing in UnLook setup
            right_camera_mirror_mode=False  # Right camera is world-facing in UnLook setup
        )
        
        # Set dynamic recognizer if YOLOv10x models were loaded
        if dynamic_recognizer is not None:
            tracker.dynamic_recognizer = dynamic_recognizer
            logger.info("Using YOLOv10x for enhanced hand tracking and gesture recognition")
        
        # Log initialization information
        logger.info(f"Hand tracker initialized with calibration file: {calibration_file}")
        
        # Configure cameras
        logger.info("Configuring cameras...")
        cameras = client.camera.get_cameras()
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return 1
        
        # Use first two cameras as left and right
        left_camera = cameras[0]['id']
        right_camera = cameras[1]['id']
        camera_names = [cameras[0]['name'], cameras[1]['name']]
        logger.info(f"Using cameras: {camera_names[0]} (left), {camera_names[1]} (right)")
        
        # Initialize UI handler
        ui = GestureUI()
        
        # Give UI access to the LED controller
        if use_led:
            ui.led_controller = led_controller
            ui.led2_intensity = led2_intensity
        
        print("\nStarting UnLook Enhanced Gesture Recognition Demo...")
        print("Press 'q' to quit")
        print("Press 's' to toggle hand skeleton")
        print("Press 'r' to recalibrate LED2 intensity based on current lighting (LED1 always 0)")
        print("Press 'l' to toggle manual LED intensity controls")
        print("Press 'space' to trigger a gesture capture")
        
        # Show optimization information
        if downsample > 1:
            print(f"\nRunning with PERFORMANCE OPTIMIZATIONS:")
            print(f"- Image downsampling: {downsample}x (1/{downsample} resolution)")
        if fast_mode:
            print("- Fast preprocessing mode enabled")
        
        print("\nGesture Recognition Ready!")
        print("Try: open palm, closed fist, pointing, peace sign, thumbs up/down, pinch")
        print("Dynamic gestures: swipe left/right/up/down, circle, wave gestures")
        
        # Update message based on whether YOLOv10x is being used
        if dynamic_recognizer is not None:
            model_types = []
            if dynamic_recognizer.using_yolo:
                model_types.append("gesture recognition")
            if dynamic_recognizer.using_yolo_hands:
                model_types.append("hand detection")
            
            if model_types:
                print(f"Enhanced with YOLOv10x for {' and '.join(model_types)}!")
            else:
                print("Enhanced gestures are powered by ML-based recognition for better accuracy!")
        else:
            print("Enhanced gestures are powered by ML-based recognition for better accuracy!")
        
        # Initialize display settings
        show_skeleton = True
        frame_count = 0
        fps_timer = time.time()
        fps_count = 0
        current_fps = 0
        
        # Use streaming with callback
        frame_buffer = {'left': None, 'right': None}
        frame_lock = {'left': False, 'right': False}
        
        def frame_callback_left(frame, metadata):
            frame_buffer['left'] = frame
            frame_lock['left'] = True
            
        def frame_callback_right(frame, metadata):
            frame_buffer['right'] = frame
            frame_lock['right'] = True
        
        # Start streaming
        logger.info("Starting camera streams...")
        client.stream.start(left_camera, frame_callback_left, fps=30)
        client.stream.start(right_camera, frame_callback_right, fps=30)
        
        # Give streams time to start
        time.sleep(0.5)
        
        # Auto-adjust LED intensity if requested
        if use_led and auto_led_adjust:
            # Wait for initial frames 
            timeout_count = 0
            while not (frame_lock['left'] and frame_lock['right']):
                time.sleep(0.1)
                timeout_count += 1
                if timeout_count > 30:  # 3 seconds timeout
                    logger.warning("Timeout waiting for frames for LED calibration")
                    break
            
            # Get frames for calibration
            frame_left_cal = frame_buffer['left']
            frame_right_cal = frame_buffer['right']
            
            # Reset locks
            frame_lock['left'] = False 
            frame_lock['right'] = False
            
            if frame_left_cal is not None and frame_right_cal is not None:
                try:
                    # Calibrate LED based on image analysis (without trying multiple LED values)
                    optimal_intensity = calculate_optimal_led_intensity(frame_left_cal, frame_right_cal)
                    
                    # Set LED to calculated optimal intensity using the controller
                    if led_controller.set_intensity(0, optimal_intensity):
                        # Store the optimal intensity for LED2 (LED1 is always 0)
                        led2_intensity = optimal_intensity
                        logger.info(f"Auto-adjusted LED2 intensity to {led2_intensity}mA (LED1 remains at 0mA)")
                        ui.add_notification(f"LED2 optimized: {led2_intensity}mA", color=(0, 255, 0))
                    else:
                        ui.add_notification("Couldn't set LED intensity", color=(255, 255, 0))
                except Exception as e:
                    logger.error(f"LED calibration error: {e}")
                    ui.add_notification(f"Using default LED: {led1_intensity}mA", color=(255, 255, 0))
        
        # Main loop
        while True:
            # Wait for both frames with timeout
            timeout_count = 0
            while not (frame_lock['left'] and frame_lock['right']):
                time.sleep(0.001)
                timeout_count += 1
                if timeout_count > 100:  # 100ms timeout
                    logger.debug("Frame sync timeout")
                    frame_lock['left'] = False
                    frame_lock['right'] = False
                    break
            
            # Get frames
            frame_left = frame_buffer['left']
            frame_right = frame_buffer['right']
            
            # Reset locks
            frame_lock['left'] = False
            frame_lock['right'] = False
            
            if frame_left is None or frame_right is None:
                logger.warning("Failed to get frames from cameras")
                continue
            
            # Preprocess images to improve hand detection in poor lighting conditions
            frame_left_enhanced = preprocess_image(frame_left)
            frame_right_enhanced = preprocess_image(frame_right)
            
            # Apply downsampling if requested - with error handling
            try:
                if downsample > 1:
                    if frame_left_enhanced is not None and frame_left_enhanced.size > 0:
                        h, w = frame_left_enhanced.shape[:2]
                        if h > 0 and w > 0:
                            frame_left_small = cv2.resize(frame_left_enhanced, (max(1, w//downsample), max(1, h//downsample)))
                        else:
                            frame_left_small = frame_left_enhanced
                    else:
                        frame_left_small = None
                        
                    if frame_right_enhanced is not None and frame_right_enhanced.size > 0:
                        h, w = frame_right_enhanced.shape[:2]
                        if h > 0 and w > 0:
                            frame_right_small = cv2.resize(frame_right_enhanced, (max(1, w//downsample), max(1, h//downsample)))
                        else:
                            frame_right_small = frame_right_enhanced
                    else:
                        frame_right_small = None
                else:
                    frame_left_small = frame_left_enhanced
                    frame_right_small = frame_right_enhanced
            except Exception as e:
                logger.warning(f"Downsampling failed, using original frames: {e}")
                frame_left_small = frame_left_enhanced
                frame_right_small = frame_right_enhanced
                
            # Track hands in 3D using enhanced images
            # Use prioritize_left_camera=True for better single-hand gesture detection
            # Also enable handedness stabilization to prevent rapid left/right switching
            results = tracker.track_hands_3d(
                frame_left_small, 
                frame_right_small, 
                prioritize_left_camera=True,
                stabilize_handedness=True  # Prevents rapid flipping between left/right hand detection
            )
            
            # If the tracker has dynamic_recognizer, pass our optimization parameters
            try:
                if hasattr(tracker, 'dynamic_recognizer') and tracker.dynamic_recognizer is not None:
                    # These parameters will be used in the next frame processing
                    tracker.dynamic_recognizer.fast_mode = fast_mode
                    tracker.dynamic_recognizer.downsample_factor = downsample
            except Exception as e:
                logger.warning(f"Failed to set optimization parameters: {e}")
            
            # Update hand trajectory for visualization and swipe detection
            if results['3d_keypoints'] and len(results['3d_keypoints']) > 0:
                # Get the first hand's wrist position and keypoints
                hand_position = results['3d_keypoints'][0][0, :2]  # X,Y only
                hand_keypoints_3d = results['3d_keypoints'][0]
                
                # Update trajectory with both position and orientation
                ui.update_hand_trajectory(hand_position, hand_keypoints_3d)
                
                # Detect swipe gestures with enhanced rotation-invariant algorithm
                swipe = ui.detect_swipe(hand_position)
                if swipe:
                    logger.info(f"Swipe {swipe} detected")
            
            # Process any dynamic gestures detected by YOLOv10x
            if 'dynamic_gestures' in results and results['dynamic_gestures']:
                for dynamic_gesture in results['dynamic_gestures']:
                    # Add a notification for the detected dynamic gesture
                    gesture_name = dynamic_gesture.get('name', 'Unknown')
                    ui.add_notification(f"Dynamic gesture: {gesture_name}", color=(0, 255, 255))
                    logger.info(f"YOLOv10x detected: {gesture_name}")
            
            # Create display with gesture recognition results
            display = ui.create_display(
                frame_left, 
                frame_right, 
                results,
                show_skeleton=show_skeleton
            )
            
            # Calculate FPS
            fps_count += 1
            if time.time() - fps_timer > 1.0:
                current_fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            
            # Add FPS display to top-right corner of display
            h, w = display.shape[:2]
            cv2.putText(display, f"FPS: {current_fps}", (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the display in fullscreen
            cv2.namedWindow('UnLook Enhanced Gesture Recognition', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('UnLook Enhanced Gesture Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('UnLook Enhanced Gesture Recognition', display)
            
            # Set mouse callback for LED controls
            cv2.setMouseCallback('UnLook Enhanced Gesture Recognition', lambda event, x, y, flags, param: ui.handle_mouse_event(event, x, y, flags, param))
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Debug key codes - useful for identifying different keyboard layouts
            if ui.show_led_controls and ui.active_control == 'led2' and key != 255:
                logger.debug(f"Key pressed: {key}")
            
            # First check if the UI is handling any LED-specific key events
            if ui.handle_key_event(key):
                pass  # Key was handled by the UI
            elif key == ord('q'):
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                msg = "Hand skeleton enabled" if show_skeleton else "Hand skeleton disabled"
                ui.add_notification(msg)
            elif key == ord('l'):
                # Toggle LED controls
                ui.show_led_controls = not ui.show_led_controls
                if ui.show_led_controls:
                    ui.add_notification("LED controls enabled", color=(0, 255, 0))
                else:
                    ui.add_notification("LED controls disabled", color=(200, 200, 200))
            elif key == ord('r') and use_led and led_controller:
                # Recalibrate LED intensity
                if frame_left is not None and frame_right is not None:
                    try:
                        ui.add_notification("Calibrating LED2 intensity... (LED1 always 0)", color=(0, 255, 255))
                        
                        # Calculate optimal intensity from current frames
                        optimal_intensity = calculate_optimal_led_intensity(frame_left, frame_right)
                        
                        # Apply the new intensity using the controller
                        if led_controller.set_intensity(0, optimal_intensity):
                            led2_intensity = optimal_intensity  # Update LED2 intensity; LED1 stays at 0
                            ui.led2_intensity = optimal_intensity  # Update UI's value too
                            ui.add_notification(f"LED2 calibrated: {optimal_intensity}mA (LED1 at 0mA)", color=(0, 255, 0))
                        else:
                            ui.add_notification("Failed to adjust LED", color=(255, 100, 100))
                    except Exception as e:
                        logger.error(f"LED calibration error: {e}")
                        ui.add_notification("LED calibration error", color=(255, 100, 100))
            elif key == ord(' '):
                # Capture current gesture (useful for debugging)
                if 'gestures' in results and results['gestures']:
                    gesture_names = [g['name'] for g in results['gestures']]
                    ui.add_notification(f"Gesture captured: {', '.join(gesture_names)}", 
                                     color=(255, 255, 0))
                else:
                    ui.add_notification("No gesture detected to capture", 
                                     color=(255, 100, 100))
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during operation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup LED if it was used
        if use_led and led_controller:
            try:
                if led_controller.turn_off():
                    logger.info("LED flood illuminator deactivated on server")
                else:
                    logger.warning("Failed to deactivate LED flood illuminator")
            except Exception as e:
                logger.error(f"Failed to turn off LED on server: {e}")
        
        # Cleanup
        logger.info("Cleaning up...")
        try:
            client.stream.stop_stream(left_camera)
            client.stream.stop_stream(right_camera)
        except:
            pass
        client.disconnect()
        client.stop_discovery()
        cv2.destroyAllWindows()
        tracker.close()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Gesture Recognition Demo')
    parser.add_argument('--calibration', type=str,
                       help='Path to stereo calibration file')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Discovery timeout in seconds (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    # LED control arguments
    parser.add_argument('--no-led', action='store_true',
                       help='Disable LED flood illuminator')
    parser.add_argument('--led1-intensity', type=int, default=0,
                       help='LED1 intensity in mA (always set to 0 for hardware safety, parameter kept for compatibility)')
    parser.add_argument('--led2-intensity', type=int, default=200,
                       help='LED2 intensity in mA (0-450, default: 200)')
    parser.add_argument('--no-auto-led', action='store_true',
                       help='Disable automatic LED intensity adjustment')
    # YOLOv10x model arguments
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Path to YOLOv10x_gestures.pt model for gesture recognition')
    parser.add_argument('--yolo-hands-model', type=str, default=None,
                       help='Path to YOLOv10x_hands.pt model for hand detection')
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLOv10x models even if available')
    parser.add_argument('--lightweight', action='store_true',
                       help='Run in lightweight mode for better performance (uses only one YOLO model)')
    parser.add_argument('--performance-mode', choices=['balanced', 'speed', 'accuracy'], default='balanced',
                      help='Performance mode: balanced, speed (faster), or accuracy (slower)')
    parser.add_argument('--downsample', type=int, choices=[1, 2, 4], default=1,
                      help='Downsample factor for processing (1=none, 2=half resolution, 4=quarter resolution)')
    parser.add_argument('--fast-mode', action='store_true',
                      help='Enable fast mode with simplified preprocessing for better performance')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger('unlook.client.scanning.handpose').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
    
    # Look for YOLOv10x models in common locations if not specified
    if not args.no_yolo and YOLO_AVAILABLE and not args.yolo_model and not args.yolo_hands_model:
        # Common model directories
        model_dirs = [
            os.path.join(Path(__file__).resolve().parent.parent, "unlook/models"),
            os.path.join(Path(__file__).resolve().parent.parent, "models"),
            os.path.join(Path.home(), ".unlook/models")
        ]
        
        # Try to find gesture model
        for model_dir in model_dirs:
            path = os.path.join(model_dir, "YOLOv10x_gestures.pt")
            if os.path.exists(path):
                args.yolo_model = path
                print(f"Found YOLOv10x gesture model at: {args.yolo_model}")
                break
        
        # Try to find hands model
        for model_dir in model_dirs:
            path = os.path.join(model_dir, "YOLOv10x_hands.pt")
            if os.path.exists(path):
                args.yolo_hands_model = path
                print(f"Found YOLOv10x hands model at: {args.yolo_hands_model}")
                break
    
    # If in lightweight mode, prioritize hands model over gestures model
    yolo_model = args.yolo_model
    yolo_hands_model = args.yolo_hands_model
    
    if args.lightweight and not args.no_yolo and YOLO_AVAILABLE:
        # In lightweight mode, only use one model (prioritize hands detection)
        if yolo_hands_model:
            print("Running in lightweight mode: Using only hand detection YOLO model for better performance")
            yolo_model = None
        elif yolo_model:
            print("Running in lightweight mode: Using only gesture recognition YOLO model for better performance")
            # Keep yolo_model, hands_model is already None
        else:
            print("Running in lightweight mode, but no YOLO models specified")
    
    run_gesture_demo(
        calibration_file=args.calibration,
        timeout=args.timeout,
        verbose=args.verbose,
        use_led=not args.no_led,
        led1_intensity=args.led1_intensity,
        led2_intensity=args.led2_intensity,
        auto_led_adjust=not args.no_auto_led,
        yolo_model_path=yolo_model,
        yolo_hands_model_path=yolo_hands_model,
        use_yolo=not args.no_yolo and YOLO_AVAILABLE,
        performance_mode=args.performance_mode,
        downsample=args.downsample,
        fast_mode=args.fast_mode
    )


if __name__ == '__main__':
    main()