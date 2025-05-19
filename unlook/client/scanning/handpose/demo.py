#!/usr/bin/env python3
"""Hand pose tracking demo module for UnLook SDK."""

import os
import sys
import time
import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any

from ....client import UnlookClient
from . import HandTracker

# LED control is done through server commands

logger = logging.getLogger(__name__)


class HandTrackingDemo:
    """Hand tracking demonstration with LED control support."""
    
    def __init__(self, 
                 calibration_file: Optional[str] = None,
                 use_led: bool = True,
                 led1_intensity: int = 450,
                 led2_intensity: int = 450,
                 verbose: bool = False):
        """
        Initialize hand tracking demo.
        
        Args:
            calibration_file: Path to stereo calibration file
            use_led: Enable LED control through server
            led1_intensity: LED1 intensity in mA (0-450)
            led2_intensity: LED2 intensity in mA (0-450)
            verbose: Enable verbose debug logging
        """
        self.calibration_file = calibration_file
        self.use_led = use_led
        self.led1_intensity = led1_intensity
        self.led2_intensity = led2_intensity
        self.verbose = verbose
        
        self.client = None
        self.tracker = None
        self.led_active = False
    
    def _init_led(self):
        """Initialize LED flood illuminator through server."""
        if not self.client:
            logger.warning("Cannot initialize LED - client not connected")
            return
            
        try:
            from ....core import MessageType
            logger.info(f"Initializing LED control (LED1={self.led1_intensity}mA, LED2={self.led2_intensity}mA)")
            response = self.client.send_message(MessageType.LED_SET_INTENSITY, {
                'led1': self.led1_intensity,
                'led2': self.led2_intensity
            })
            if response and response.payload.get('status') == 'success':
                self.led_active = True
                logger.info("LED flood illuminator activated on server")
            else:
                logger.warning("Failed to activate LED on server")
                self.use_led = False
        except Exception as e:
            logger.error(f"Failed to initialize LED: {e}")
            self.use_led = False
            self.led_active = False
    
    def _cleanup_led(self):
        """Turn off LED flood illuminator through server."""
        if self.led_active and self.client:
            try:
                from ....core import MessageType
                response = self.client.send_message(MessageType.LED_OFF, {})
                if response and response.payload.get('status') == 'success':
                    self.led_active = False
                    logger.info("LED flood illuminator deactivated on server")
            except Exception as e:
                logger.error(f"Failed to turn off LED: {e}")
    
    def connect(self, timeout: int = 10) -> bool:
        """Connect to UnLook scanner."""
        self.client = UnlookClient(auto_discover=False)
        
        try:
            # Start discovery
            self.client.start_discovery()
            logger.info(f"Discovering scanners for {timeout} seconds...")
            time.sleep(timeout)
            
            # Get discovered scanners
            scanners = self.client.get_discovered_scanners()
            if not scanners:
                logger.error("No scanners found")
                return False
            
            # Connect to the first scanner
            scanner_info = scanners[0]
            logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
            
            if not self.client.connect(scanner_info):
                logger.error("Failed to connect to scanner")
                return False
            
            logger.info(f"Successfully connected to scanner: {scanner_info.name}")
            
            # Initialize LED after connection
            if self.use_led:
                self._init_led()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def setup_tracker(self) -> bool:
        """Setup hand tracker with calibration."""
        if not self.client:
            logger.error("Client not connected")
            return False
        
        # Get calibration
        calibration_file = self.calibration_file
        if not calibration_file:
            calibration_file = self.client.camera.get_calibration_file_path()
            if calibration_file:
                logger.info(f"Using auto-loaded calibration: {calibration_file}")
            else:
                logger.warning("No calibration file available")
        
        # Initialize hand tracker
        self.tracker = HandTracker(
            calibration_file=calibration_file,
            max_num_hands=4,
            detection_confidence=0.3,
            tracking_confidence=0.3
        )
        
        return True
    
    def run(self, output_file: Optional[str] = None, visualize_3d: bool = False):
        """Run the hand tracking demo."""
        # Configure cameras
        logger.info("Configuring cameras...")
        cameras = self.client.camera.get_cameras()
        if len(cameras) < 2:
            logger.error(f"Need at least 2 cameras, found {len(cameras)}")
            return
        
        # Use first two cameras as left and right
        left_camera = cameras[0]['id']
        right_camera = cameras[1]['id']
        camera_names = [cameras[0]['name'], cameras[1]['name']]
        logger.info(f"Using cameras: {camera_names[0]} (left), {camera_names[1]} (right)")
        
        self._run_tracking_loop(left_camera, right_camera, output_file, visualize_3d)
    
    def _run_tracking_loop(self, left_camera: str, right_camera: str, 
                          output_file: Optional[str] = None, visualize_3d: bool = False):
        """Main tracking loop."""
        print("\nStarting UnLook 3D hand tracking demo...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame data")
        print("Press 'v' to toggle 3D visualization")
        print("Press 'd' to toggle debug mode (show raw images)")
        if self.use_led:
            print("Press 'l' to toggle LED intensity")
        
        show_3d = visualize_3d
        debug_mode = False
        frame_count = 0
        led_high = True  # Track LED intensity state
        
        # Frame buffers for streaming
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
        self.client.stream.start(left_camera, frame_callback_left, fps=30)
        self.client.stream.start(right_camera, frame_callback_right, fps=30)
        
        # Give streams time to start
        time.sleep(0.5)
        
        while True:
            # Wait for both frames
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
            
            # Track hands in 3D
            results = self.tracker.track_hands_3d(frame_left, frame_right)
            
            # Adaptive LED control based on detection
            if self.use_led and self.led_active and frame_count % 30 == 0:  # Check every second
                num_hands = len(results['3d_keypoints'])
                if num_hands == 0 and led_high:
                    # Increase LED intensity if no hands detected
                    try:
                        from ....core import MessageType
                        response = self.client.send_message(MessageType.LED_SET_INTENSITY, {
                            'led1': 450,
                            'led2': 450
                        })
                        led_high = True
                    except:
                        pass
                elif num_hands > 0 and not led_high:
                    # Decrease LED intensity if hands detected well
                    try:
                        from ....core import MessageType
                        response = self.client.send_message(MessageType.LED_SET_INTENSITY, {
                            'led1': 300,
                            'led2': 300
                        })
                        led_high = False
                    except:
                        pass
            
            # Display results
            self._display_results(frame_left, frame_right, results, 
                                 show_3d, debug_mode, frame_count)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_frame_data()
            elif key == ord('v'):
                show_3d = not show_3d
                logger.info(f"3D visualization: {'ON' if show_3d else 'OFF'}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                logger.info(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                if not debug_mode:
                    cv2.destroyWindow('Debug - Raw Cameras')
            elif key == ord('l') and self.use_led and self.led_active:
                # Toggle LED intensity
                led_high = not led_high
                intensity = 450 if led_high else 300
                try:
                    from ....core import MessageType
                    response = self.client.send_message(MessageType.LED_SET_INTENSITY, {
                        'led1': intensity,
                        'led2': intensity
                    })
                    if response and response.payload.get('status') == 'success':
                        logger.info(f"LED intensity: {intensity}mA")
                except:
                    pass
            
            frame_count += 1
        
        # Save tracking data if requested
        if output_file:
            self.tracker.save_tracking_data(output_file)
            logger.info(f"Saved tracking data to {output_file}")
    
    def _display_results(self, frame_left: np.ndarray, frame_right: np.ndarray,
                        results: Dict[str, Any], show_3d: bool, debug_mode: bool,
                        frame_count: int):
        """Display tracking results."""
        # Implementation would go here - simplified for brevity
        h, w = frame_left.shape[:2]
        
        # Create stereo display
        stereo_image = np.zeros((h, w*2, 3), dtype=np.uint8)
        stereo_image[:, :w] = frame_left
        stereo_image[:, w:] = frame_right
        
        cv2.imshow('UnLook Hand Tracking', stereo_image)
        
        if debug_mode:
            debug_display = np.zeros((h, w*2, 3), dtype=np.uint8)
            debug_display[:, :w] = frame_left
            debug_display[:, w:] = frame_right
            cv2.imshow('Debug - Raw Cameras', debug_display)
    
    def _save_frame_data(self):
        """Save current frame data."""
        timestamp = int(time.time())
        filename = f"unlook_hand_tracking_{timestamp}.json"
        self.tracker.save_tracking_data(filename)
        logger.info(f"Saved tracking data to {filename}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            self.client.stream.stop_all()
            self.client.disconnect()
            self.client.stop_discovery()
        
        if self.tracker:
            self.tracker.close()
        
        self._cleanup_led()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def run_demo(**kwargs):
    """Convenience function to run demo."""
    demo = HandTrackingDemo(**kwargs)
    try:
        if demo.connect():
            if demo.setup_tracker():
                demo.run()
    finally:
        demo.cleanup()