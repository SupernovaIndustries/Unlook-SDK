#!/usr/bin/env python3
"""
YOLOv10x Gesture Recognition Demo - Highly optimized for stability and performance.

This demo uses YOLOv10x for hand detection and gesture recognition with UnLook stereo cameras.
Implemented with maximum stability as the priority.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
import cv2
import threading
import queue
import psutil
import gc
import multiprocessing
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import UnLook SDK
from unlook import UnlookClient
from unlook.client.scanning.handpose import HandTracker, GestureType
from unlook.client.projector import LEDController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
frame_buffer = {'left': None, 'right': None}
frame_lock = {'left': False, 'right': False}
result_queue = multiprocessing.Queue()
running = True


class SimpleUI:
    """Minimal UI handler for gesture recognition demo."""
    
    def __init__(self):
        self.window_name = "YOLOv10x Gesture Recognition Demo"
        self.notification = None
        self.notification_color = (255, 255, 255)
        self.notification_start_time = 0
        self.notification_duration = 2.0  # seconds
        
    def create_window(self):
        """Create the OpenCV window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def show_loading(self, message, submessage=None):
        """Show a loading screen with message."""
        display = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(display, message, (50, 360), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                  
        if submessage:
            cv2.putText(display, submessage, (50, 400), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                      
        cv2.putText(display, "Press 'q' to quit", (50, 680), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                  
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)
        
    def add_notification(self, text, color=(255, 255, 255), duration=2.0):
        """Add a temporary notification."""
        self.notification = text
        self.notification_color = color
        self.notification_start_time = time.time()
        self.notification_duration = duration
        
    def create_display(self, left_frame, right_frame, results=None, fps=0, mem_usage=0):
        """Create a simple side-by-side display with results."""
        # Handle None frames
        if left_frame is None or right_frame is None:
            display = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(display, "No camera frames available", (50, 360), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return display
            
        # Get dimensions
        h1, w1 = left_frame.shape[:2]
        h2, w2 = right_frame.shape[:2]
        
        # Resize to same height if needed
        if h1 != h2:
            new_w2 = int(w2 * (h1 / h2))
            right_frame = cv2.resize(right_frame, (new_w2, h1))
            h2, w2 = h1, new_w2
        
        # Create combined display
        display = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
        display[:, :w1] = left_frame
        display[:, w1:w1+w2] = right_frame
        
        # Add camera labels
        cv2.putText(display, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "RIGHT", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add stats
        cv2.putText(display, f"FPS: {fps:.1f}", (10, h1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Memory: {mem_usage:.0f}MB", (10, h1 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw YOLO results if available
        if results:
            # Display detected gestures
            if 'gestures' in results and results['gestures']:
                for i, gesture in enumerate(results['gestures']):
                    gesture_text = f"Gesture: {gesture['name']} ({gesture['confidence']:.2f})"
                    cv2.putText(display, gesture_text, (10, 70 + i*30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw bounding boxes on left image
            if 'bboxes' in results and len(results['bboxes']) > 0:
                for i, bbox in enumerate(results['bboxes']):
                    # Draw rectangle
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # Add label if available
                    if 'labels' in results and i < len(results['labels']):
                        label = results['labels'][i]
                        if isinstance(label, int):
                            label = f"Class {label}"
                        cv2.putText(display, str(label), (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw keypoints if available
            if 'keypoints' in results and len(results['keypoints']) > 0:
                for keypoints in results['keypoints']:
                    # Convert from normalized to pixel coordinates
                    pixel_points = []
                    for kp in keypoints:
                        x = int(kp[0] * w1)
                        y = int(kp[1] * h1)
                        pixel_points.append((x, y))
                    
                    # Draw circles at keypoints
                    for pt in pixel_points:
                        cv2.circle(display, pt, 3, (0, 0, 255), -1)
                        
                    # Connect keypoints to form hand skeleton (simplified)
                    if len(pixel_points) >= 21:  # Full hand landmarks
                        # Connect fingers
                        for finger in range(5):
                            start_idx = 4 * finger + 1  # Skip finger MCP joints
                            for i in range(3):  # 3 bones per finger
                                cv2.line(display, 
                                        pixel_points[start_idx + i],
                                        pixel_points[start_idx + i + 1], 
                                        (0, 255, 0), 2)
                        
                        # Connect palm
                        palm_indices = [0, 1, 5, 9, 13, 17]  # Wrist and MCP joints
                        for i in range(len(palm_indices) - 1):
                            cv2.line(display,
                                    pixel_points[palm_indices[i]],
                                    pixel_points[palm_indices[i + 1]],
                                    (0, 255, 0), 2)
                        # Close the palm loop
                        cv2.line(display,
                                pixel_points[palm_indices[-1]],
                                pixel_points[palm_indices[0]],
                                (0, 255, 0), 2)
        
        # Add notification if active
        if self.notification:
            current_time = time.time()
            elapsed = current_time - self.notification_start_time
            
            if elapsed < self.notification_duration:
                # Notification is still active, display it
                alpha = 1.0 - (elapsed / self.notification_duration)
                
                # Create a semi-transparent overlay for notification
                overlay = display.copy()
                cv2.rectangle(overlay, (50, h1 - 100), (display.shape[1] - 50, h1 - 60),
                           (40, 40, 40), -1)
                
                # Add notification text
                cv2.putText(overlay, self.notification, (70, h1 - 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.notification_color, 2)
                
                # Blend with transparency
                cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            else:
                # Notification expired
                self.notification = None
                
        return display


class YOLOModel:
    """Wrapper for YOLOv10x models with safety features."""
    
    def __init__(self, model_path, model_type="hands", performance_mode="balanced"):
        self.model_path = model_path
        self.model_type = model_type
        self.performance_mode = performance_mode
        self.model = None
        self.model_size = 128  # Default minimal size
        
        # Set model size based on performance mode
        if performance_mode == "speed":
            self.model_size = 96   # Tiny for speed
        elif performance_mode == "balanced":
            self.model_size = 128  # Small for balance
        else:  # accuracy
            self.model_size = 160  # Larger for accuracy
            
        # Always use CPU for initialization
        self.device = "cpu"
        
        # Initialize parameters with super conservative settings
        self.params = {
            "imgsz": self.model_size,
            "verbose": False,
            "conf": 0.7,
            "half": True,  # Use FP16
            "max_det": 1,  # Only detect one hand
            "batch": 1,
            "device": self.device
        }
        
    def load(self):
        """Load the model with error handling and fallbacks."""
        try:
            from ultralytics import YOLO
            
            # First check if model exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            # Try to determine the best device
            try:
                import torch
                
                if torch.cuda.is_available():
                    # Check GPU memory
                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_mem_mb = free_mem / (1024 * 1024)
                    
                    if free_mem_mb > 1000:  # At least 1GB free
                        self.device = 0  # Use CUDA
                        self.params["device"] = 0
                        print(f"Using CUDA with {free_mem_mb:.0f}MB free memory")
                    else:
                        print(f"Not enough GPU memory ({free_mem_mb:.0f}MB free). Using CPU.")
                elif hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'  # Use Apple Metal
                    self.params["device"] = 'mps'
                    print("Using Apple Metal (MPS) acceleration")
            except Exception as e:
                print(f"Error determining optimal device: {e}, falling back to CPU")
            
            # Load the model
            print(f"Loading {self.model_type} model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            # Force initialization with a small dummy inference
            dummy_img = np.zeros((self.model_size, self.model_size, 3), dtype=np.uint8)
            self.model(dummy_img, **self.params)
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def process(self, frame):
        """Process a frame with error handling."""
        if self.model is None or frame is None:
            return None
            
        try:
            # Make a copy to prevent race conditions
            frame_copy = frame.copy()
            
            # Run inference
            results = self.model(frame_copy, **self.params)
            return results
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None


def process_frames(model, frame_left, frame_right):
    """Process frames using YOLOv10x model."""
    # Skip if no model or frames
    if model is None or frame_left is None or frame_right is None:
        return None
        
    try:
        # Process left frame (typically better angle/quality)
        results_left = model.process(frame_left)
        
        # If no results from left, try right
        if not results_left or len(results_left) == 0:
            results_right = model.process(frame_right)
            results = results_right
        else:
            results = results_left
            
        # Parse results
        parsed = {
            'bboxes': [],
            'labels': [],
            'gestures': [],
            'keypoints': []
        }
        
        # Process results if any
        if results:
            for r in results:
                boxes = r.boxes
                if len(boxes) == 0:
                    continue
                
                # Check for keypoints
                keypoints = r.keypoints if hasattr(r, 'keypoints') else None
                
                # Process each box
                for i, box in enumerate(boxes):
                    # Get box coordinates and convert to numpy
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = np.array([x1, y1, x2, y2])
                    
                    # Get confidence
                    conf = float(box.conf[0].item())
                    
                    # Get class if available
                    cls = int(box.cls[0].item()) if len(box.cls) > 0 else 0
                    
                    # Map class to gesture name if possible
                    gesture_name = f"Class {cls}"
                    if hasattr(model.model, 'names') and cls in model.model.names:
                        gesture_name = model.model.names[cls]
                    
                    # Add to results
                    parsed['bboxes'].append(bbox)
                    parsed['labels'].append(cls)
                    parsed['gestures'].append({
                        'name': gesture_name,
                        'confidence': conf
                    })
                    
                    # Process keypoints if available
                    if keypoints is not None and i < len(keypoints):
                        try:
                            hand_kps = keypoints[i].data[0].cpu().numpy()
                            if len(hand_kps) == 21:  # Full hand keypoints
                                # Original coordinates
                                parsed['keypoints'].append(hand_kps)
                        except Exception as e:
                            print(f"Error processing keypoints: {e}")
        
        return parsed
    
    except Exception as e:
        print(f"Error in process_frames: {e}")
        return None


def model_worker_process(model_path, model_type, performance_mode, result_queue):
    """Worker process for running the model inference."""
    try:
        # Initialize model
        model = YOLOModel(model_path, model_type, performance_mode)
        success = model.load()
        
        if not success:
            result_queue.put(("error", "Failed to load model"))
            return
            
        # Send ready signal
        result_queue.put(("ready", f"{model_type} model ready"))
        
        # Process frames
        while True:
            try:
                # Get data from queue with timeout
                action, data = result_queue.get(timeout=1)
                
                # Check for exit signal
                if action == "exit":
                    break
                    
                # Process frame if requested
                if action == "process" and isinstance(data, tuple) and len(data) == 2:
                    frame_left, frame_right = data
                    results = process_frames(model, frame_left, frame_right)
                    result_queue.put(("results", results))
                    
            except queue.Empty:
                # Queue timeout, just continue
                continue
            except Exception as e:
                print(f"Error in worker loop: {e}")
                continue
                
    except Exception as e:
        print(f"Error initializing model worker: {e}")
        result_queue.put(("error", str(e)))
    finally:
        print("Model worker shutting down")


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_demo(calibration_file=None, timeout=5, yolo_model=None, yolo_hands_model=None,
           lightweight=False, performance_mode="balanced", downsample=4, use_led=False,
           led_intensity=200):
    """
    Run the YOLOv10x demo with extensive error handling and stability optimizations.
    
    Args:
        calibration_file: Path to calibration file
        timeout: Discovery timeout in seconds
        yolo_model: Path to YOLOv10x gesture model
        yolo_hands_model: Path to YOLOv10x hands model
        lightweight: Only use one model for better performance
        performance_mode: balanced, speed, or accuracy
        downsample: Downsampling factor for frames (smaller = faster)
        use_led: Enable LED illumination
        led_intensity: LED intensity (0-450)
    """
    global running, frame_buffer, frame_lock, result_queue
    
    # Initialize UI first
    ui = SimpleUI()
    ui.create_window()
    ui.show_loading("Starting YOLOv10x Gesture Recognition Demo", 
                  "Initializing system...")
    
    # Initialize model worker process
    worker_process = None
    
    try:
        # Initialize client
        ui.show_loading("Connecting to UnLook system...", 
                      "Searching for devices...")
        client = UnlookClient(auto_discover=False)
        
        # Start discovery
        client.start_discovery()
        print(f"Discovering scanners for {timeout} seconds...")
        time.sleep(timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            ui.show_loading("No scanners found", 
                         "Check that your hardware is connected and powered on")
            time.sleep(3)
            return
            
        # Connect to first scanner
        scanner_info = scanners[0]
        ui.show_loading(f"Connecting to {scanner_info.name}...", 
                      "Establishing connection...")
        
        if not client.connect(scanner_info):
            ui.show_loading("Failed to connect to scanner", 
                          "Check hardware connections")
            time.sleep(3)
            return
            
        print(f"Connected to scanner: {scanner_info.name}")
        
        # Initialize LED if requested
        led_controller = None
        if use_led:
            try:
                ui.show_loading("Initializing LED illumination...", 
                             "Setting up optimal lighting...")
                
                led_controller = LEDController(client)
                if led_controller.led_available:
                    if led_controller.set_intensity(0, led_intensity):
                        print(f"LED activated (LED1=0mA, LED2={led_intensity}mA)")
                        ui.add_notification(f"LED activated at {led_intensity}mA", (0, 255, 0))
                    else:
                        print("Failed to activate LED")
                else:
                    print("LED not available on this scanner")
            except Exception as e:
                print(f"Error initializing LED: {e}")
        
        # Get calibration
        if not calibration_file:
            calibration_file = client.camera.get_calibration_file_path()
            if calibration_file:
                print(f"Using auto-loaded calibration: {calibration_file}")
            else:
                print("No calibration file available")
                
        # Initialize hand tracker with conservative settings
        ui.show_loading("Setting up tracking system...", 
                      "Initializing hand tracker...")
        
        tracker = HandTracker(
            calibration_file=calibration_file,
            max_num_hands=1,  # Only one hand for performance
            detection_confidence=0.7,
            tracking_confidence=0.7,
            left_camera_mirror_mode=False,
            right_camera_mirror_mode=False
        )
        
        # Setup cameras
        ui.show_loading("Setting up cameras...", 
                      "Configuring camera streams...")
        
        cameras = client.camera.get_cameras()
        if len(cameras) < 2:
            ui.show_loading(f"Need at least 2 cameras, found {len(cameras)}", 
                         "Check camera connections")
            time.sleep(3)
            return
            
        # Use first two cameras
        left_camera = cameras[0]['id']
        right_camera = cameras[1]['id']
        print(f"Using cameras: {cameras[0]['name']} (left), {cameras[1]['name']} (right)")
        
        # Start model loading
        if (yolo_model or yolo_hands_model) and not lightweight:
            # Load both models if available and not in lightweight mode
            models_to_load = []
            if yolo_hands_model:
                models_to_load.append((yolo_hands_model, "hands"))
            if yolo_model:
                models_to_load.append((yolo_model, "gestures"))
        elif lightweight:
            # Only load one model in lightweight mode
            if yolo_hands_model:
                models_to_load = [(yolo_hands_model, "hands")]
            elif yolo_model:
                models_to_load = [(yolo_model, "gestures")]
            else:
                models_to_load = []
        else:
            models_to_load = []
            
        # Start loading the first model (prioritize hands model)
        if models_to_load:
            model_path, model_type = models_to_load[0]
            
            ui.show_loading(f"Loading {model_type} model...", 
                         "This may take a moment...")
            
            # Create worker process for model
            worker_process = multiprocessing.Process(
                target=model_worker_process,
                args=(model_path, model_type, performance_mode, result_queue)
            )
            worker_process.daemon = True
            worker_process.start()
            
            # Wait for model to load with animation
            start_time = time.time()
            dots = 0
            model_loaded = False
            
            while time.time() - start_time < 60 and not model_loaded:  # 60 second timeout
                # Try to get result with timeout
                try:
                    # Short timeout to keep UI responsive
                    if not result_queue.empty():
                        status, message = result_queue.get(timeout=0.1)
                        if status == "ready":
                            model_loaded = True
                            ui.add_notification(f"{model_type} model loaded successfully!", (0, 255, 0))
                        elif status == "error":
                            ui.show_loading(f"Error loading model", message)
                            time.sleep(3)
                            if worker_process.is_alive():
                                worker_process.terminate()
                            worker_process = None
                            break
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Error waiting for model: {e}")
                
                # Update loading animation
                dots = (dots + 1) % 10
                loading_text = "Loading" + "." * dots
                ui.show_loading(f"Loading {model_type} model", 
                              f"{loading_text} (elapsed: {time.time() - start_time:.1f}s)")
                time.sleep(0.1)
                
            # Check for timeout
            if not model_loaded and worker_process and worker_process.is_alive():
                ui.show_loading("Model loading timeout", 
                             "Model took too long to load, continuing without it")
                time.sleep(2)
                worker_process.terminate()
                worker_process = None
        
        # Start frame streaming
        def frame_callback_left(frame, metadata):
            if running:
                frame_buffer['left'] = frame.copy() if frame is not None else None
                frame_lock['left'] = True
            
        def frame_callback_right(frame, metadata):
            if running:
                frame_buffer['right'] = frame.copy() if frame is not None else None
                frame_lock['right'] = True
        
        ui.show_loading("Starting camera streams...", 
                      "Initializing video feeds...")
        
        client.stream.start(left_camera, frame_callback_left, fps=15)
        client.stream.start(right_camera, frame_callback_right, fps=15)
        
        # Give streams time to start
        time.sleep(0.5)
        
        # Ready to go
        ui.show_loading("Ready to start gesture recognition", 
                      "Press 'q' to quit at any time")
        time.sleep(1)
        
        # Main loop variables
        fps_timer = time.time()
        fps_count = 0
        current_fps = 0
        frame_count = 0
        last_memory_check = time.time()
        memory_usage = get_memory_usage()
        last_display = None
        last_results = None
        
        # Main loop
        while running:
            loop_start = time.time()
            
            # Check for frames
            if frame_lock['left'] and frame_lock['right']:
                # Get frames
                frame_left = frame_buffer['left']
                frame_right = frame_buffer['right']
                
                # Reset locks
                frame_lock['left'] = False
                frame_lock['right'] = False
                
                # Skip frames for performance
                frame_count += 1
                process_frame = frame_count % 3 == 0  # Process every 3rd frame
                
                # Process with basic tracker or model
                current_results = None
                if process_frame:
                    # Apply downsampling if requested
                    try:
                        if downsample > 1:
                            h, w = frame_left.shape[:2]
                            frame_left_small = cv2.resize(frame_left, (w//downsample, h//downsample))
                            
                            h, w = frame_right.shape[:2]
                            frame_right_small = cv2.resize(frame_right, (w//downsample, h//downsample))
                        else:
                            frame_left_small = frame_left
                            frame_right_small = frame_right
                    except Exception as e:
                        print(f"Error in downsampling: {e}")
                        frame_left_small = frame_left
                        frame_right_small = frame_right
                    
                    # If model worker is active, send frames to it
                    if worker_process and worker_process.is_alive():
                        try:
                            # Send frames for processing (non-blocking)
                            if result_queue.empty():
                                result_queue.put(("process", (frame_left_small, frame_right_small)), 
                                                block=False)
                        except queue.Full:
                            # Queue is full, skip this frame
                            pass
                    else:
                        # Do basic tracking with MediaPipe-based tracker
                        try:
                            # Track hands with conservative settings
                            mp_results = tracker.track_hands_3d(
                                frame_left_small, 
                                frame_right_small, 
                                prioritize_left_camera=True,
                                stabilize_handedness=True
                            )
                            current_results = mp_results
                        except Exception as e:
                            print(f"Error tracking hands: {e}")
                
                # Check for results from model worker
                if worker_process and worker_process.is_alive():
                    try:
                        # Non-blocking check for results
                        if not result_queue.empty():
                            action, data = result_queue.get_nowait()
                            if action == "results" and data is not None:
                                last_results = data
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"Error getting results: {e}")
                
                # Use latest results for display
                display_results = current_results if current_results else last_results
                
                # Create display
                if frame_count % 2 == 0:  # Update display every other frame for performance
                    # Calculate FPS
                    fps_count += 1
                    if time.time() - fps_timer > 1.0:
                        current_fps = fps_count
                        fps_count = 0
                        fps_timer = time.time()
                    
                    # Check memory periodically
                    if time.time() - last_memory_check > 5.0:
                        memory_usage = get_memory_usage()
                        last_memory_check = time.time()
                    
                    # Create and show display
                    try:
                        display = ui.create_display(
                            frame_left, frame_right, 
                            display_results,
                            current_fps, memory_usage
                        )
                        cv2.imshow(ui.window_name, display)
                        last_display = display.copy()
                    except Exception as e:
                        print(f"Error creating display: {e}")
                        # Try to show raw frame if display creation fails
                        if frame_left is not None:
                            cv2.imshow(ui.window_name, frame_left)
                elif last_display is not None:
                    # Show last display for skipped frames
                    cv2.imshow(ui.window_name, last_display)
            
            # Handle key events with short timeout to keep responsive
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
            
            # Throttle loop to prevent 100% CPU
            elapsed = time.time() - loop_start
            if elapsed < 0.01:  # Aim for ~100 Hz UI update
                time.sleep(0.01 - elapsed)
    
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error on UI
        ui.show_loading("Error occurred", str(e))
        time.sleep(3)
    
    finally:
        # Stop worker process
        running = False
        if worker_process:
            try:
                result_queue.put(("exit", None))
                worker_process.join(timeout=2)
                if worker_process.is_alive():
                    worker_process.terminate()
            except:
                pass
        
        # Clean up LED
        if use_led and led_controller:
            try:
                led_controller.turn_off()
                print("LED turned off")
            except:
                pass
        
        # Clean up streams
        print("Cleaning up resources...")
        try:
            if 'client' in locals():
                if hasattr(client, 'stream'):
                    if 'left_camera' in locals():
                        client.stream.stop_stream(left_camera)
                    if 'right_camera' in locals():
                        client.stream.stop_stream(right_camera)
                client.disconnect()
                client.stop_discovery()
        except:
            pass
            
        # Close window
        cv2.destroyAllWindows()
        
        # Force cleanup
        gc.collect()
        print("Cleanup complete")


def main():
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(description='YOLOv10x Gesture Recognition Demo')
    
    # Basic parameters
    parser.add_argument('--calibration', type=str, help='Path to stereo calibration file')
    parser.add_argument('--timeout', type=int, default=5, 
                       help='Discovery timeout in seconds (default: 5)')
    
    # YOLO models
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Path to YOLOv10x gesture model')
    parser.add_argument('--yolo-hands-model', type=str, default=None,
                       help='Path to YOLOv10x hands model')
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLO models completely')
    
    # Performance options
    parser.add_argument('--lightweight', action='store_true',
                       help='Run in lightweight mode (only one model)')
    parser.add_argument('--performance-mode', choices=['balanced', 'speed', 'accuracy'],
                      default='balanced', help='Performance mode')
    parser.add_argument('--downsample', type=int, choices=[1, 2, 4, 8], default=4,
                      help='Downsampling factor (default: 4)')
    
    # LED options
    parser.add_argument('--led', action='store_true', help='Enable LED illumination')
    parser.add_argument('--led-intensity', type=int, default=200,
                       help='LED intensity in mA (0-450, default: 200)')
    
    args = parser.parse_args()
    
    # Auto-detect models if not specified
    if not args.no_yolo and not args.yolo_model and not args.yolo_hands_model:
        # Common locations
        model_dirs = [
            os.path.join(Path(__file__).resolve().parent.parent, "unlook/models"),
            os.path.join(Path(__file__).resolve().parent.parent, "models"),
            os.path.join(Path.home(), ".unlook/models")
        ]
        
        # Look for gesture model
        for model_dir in model_dirs:
            path = os.path.join(model_dir, "YOLOv10x_gestures.pt")
            if os.path.exists(path):
                args.yolo_model = path
                print(f"Found gesture model at: {args.yolo_model}")
                break
                
        # Look for hands model
        for model_dir in model_dirs:
            path = os.path.join(model_dir, "YOLOv10x_hands.pt")
            if os.path.exists(path):
                args.yolo_hands_model = path
                print(f"Found hands model at: {args.yolo_hands_model}")
                break
    
    # If no models found or disabled
    if args.no_yolo or (not args.yolo_model and not args.yolo_hands_model):
        print("Running without YOLO models")
        args.yolo_model = None
        args.yolo_hands_model = None
    
    # Run demo
    run_demo(
        calibration_file=args.calibration,
        timeout=args.timeout,
        yolo_model=args.yolo_model,
        yolo_hands_model=args.yolo_hands_model,
        lightweight=args.lightweight,
        performance_mode=args.performance_mode,
        downsample=args.downsample,
        use_led=args.led,
        led_intensity=args.led_intensity
    )


if __name__ == '__main__':
    main()