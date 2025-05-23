"""LED controller for UnLook Scanner with auto-activation.

This module provides enhanced LED control for the UnLook scanner with
automatic activation when hands are detected. It controls both the
projector points (LED1) and flood illuminator (LED2).
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class LEDController:
    """
    LED controller for UnLook Scanner with auto-activation.
    
    Controls the AS1170 LED illuminator (LED1: projector, LED2: flood illuminator)
    with automatic activation when hands are detected.
    
    Configuration:
    - I2C Bus: 4
    - Strobe Pin: GPIO 27
    - Base current: 50mA
    """
    
    def __init__(self, client, base_current: int = 50):
        """
        Initialize LED controller with auto-activation features.
        
        Args:
            client: UnlookClient instance
            base_current: Base current in mA (default 50mA)
        """
        self.client = client
        self.led_available = False
        self.led1_intensity = 0  # Projector points
        self.led2_intensity = 0  # Flood illuminator
        self.led_active = False
        self.base_current = max(0, min(450, base_current))  # Clamp to valid range
        
        # Auto-activation settings
        self.auto_activate_enabled = True
        self.last_hand_detection_time = 0
        self.hand_timeout = 1.0  # Turn off LEDs if no hands detected for this many seconds
        self.last_check_time = 0
        self.check_interval = 0.1  # Check hand presence every 100ms
        self.led_cooldown = 0.5  # Minimum time between LED state changes
        self.last_led_change = 0
        
        # Brightness levels for auto-activation (0-100%)
        self.brightness_levels = {
            "off": 0,
            "low": 30,  # 30% of max intensity (useful for dark environments)
            "medium": 60,  # 60% of max intensity (standard use)
            "high": 90,  # 90% of max intensity (bright environments)
            "max": 100  # 100% of max intensity
        }
        
        self.current_brightness = "medium"  # Default brightness level
        self.max_intensity = 450  # Maximum LED current in mA
        
        # Background thread for LED management
        self._stop_event = threading.Event()
        self._led_thread = None
        
        # Check LED availability
        self._check_availability()
        
        # If LEDs are available, start background thread
        if self.led_available:
            self._start_led_thread()
    
    def _check_availability(self) -> bool:
        """
        Check if LED control is available on the server.
        
        Returns:
            True if LEDs are available, False otherwise
        """
        try:
            # Try to import message type first
            try:
                from ...core.protocol import MessageType
                self.MessageType = MessageType
            except ImportError:
                # Define fallback class if core module not available
                class MessageType:
                    LED_SET_INTENSITY = "led_set_intensity"
                    LED_ON = "led_on"
                    LED_OFF = "led_off"
                    LED_STATUS = "led_status"
                
                self.MessageType = MessageType
            
            # Check LED status
            success, response, _ = self.client.send_message(self.MessageType.LED_STATUS, {})
            if success and response and response.payload.get('status') == 'success':
                self.led_available = response.payload.get('led_available', False)
                self.led_active = response.payload.get('led_active', False)
                self.led1_intensity = response.payload.get('led1_intensity', 0)
                self.led2_intensity = response.payload.get('led2_intensity', 0)
                
                logger.info(f"LED control available: {self.led_available}")
                if self.led_available:
                    logger.info(f"LED state: active={self.led_active}, LED1={self.led1_intensity}mA, LED2={self.led2_intensity}mA")
                
                # Initialize with base current if LEDs are available
                if self.led_available:
                    # Set LED1 to 0 (never used) and LED2 to base current
                    self.set_intensity(0, self.base_current)
                    # Turn off initially
                    self.turn_off()
                
                return self.led_available
            else:
                logger.warning("Failed to get LED status from server")
                return False
        except Exception as e:
            logger.error(f"Error checking LED availability: {e}")
            return False
    
    def _start_led_thread(self):
        """Start background thread for LED management."""
        if self._led_thread is None or not self._led_thread.is_alive():
            self._stop_event.clear()
            self._led_thread = threading.Thread(target=self._led_management_thread, daemon=True)
            self._led_thread.start()
            logger.info("LED management thread started")
    
    def _led_management_thread(self):
        """Background thread for LED management."""
        logger.info("LED management thread is running")
        
        while not self._stop_event.is_set():
            try:
                # Check if auto-activation is enabled
                if self.auto_activate_enabled:
                    current_time = time.time()
                    
                    # Check if it's time to check hand presence
                    if current_time - self.last_check_time >= self.check_interval:
                        self.last_check_time = current_time
                        
                        # Check if hands are still being detected
                        time_since_detection = current_time - self.last_hand_detection_time
                        
                        # If we haven't detected hands for a while and LEDs are on, turn them off
                        if time_since_detection > self.hand_timeout and self.led_active:
                            if current_time - self.last_led_change >= self.led_cooldown:
                                logger.info("Auto-deactivating LEDs: no hands detected")
                                self.turn_off()
                                self.last_led_change = current_time
                
                # Sleep a short amount to prevent high CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in LED management thread: {e}")
                # If an error occurs, sleep longer to prevent rapid error loops
                time.sleep(0.5)
    
    def stop(self):
        """Stop background thread and release resources."""
        self._stop_event.set()
        
        if self._led_thread and self._led_thread.is_alive():
            self._led_thread.join(timeout=1.0)
            
        # Make sure LEDs are off
        if self.led_available and self.led_active:
            self.turn_off()
    
    def set_intensity(self, 
                     led1: int = 0, 
                     led2: Optional[int] = None) -> bool:
        """
        Set the intensity of both LED channels.
        
        Args:
            led1: LED1 intensity in mA (0-450), defaults to 0
            led2: LED2 intensity in mA (0-450), defaults to current brightness level
            
        Returns:
            True if successful, False otherwise
        """
        if not self.led_available:
            logger.warning("LED control not available")
            return False
        
        # Always set LED1 to 0 (never used in our setup)
        led1_value = 0
        
        # If LED2 not specified, use current brightness level
        if led2 is None:
            brightness_percent = self.brightness_levels[self.current_brightness]
            led2_value = int(self.max_intensity * brightness_percent / 100)
        else:
            led2_value = max(0, min(self.max_intensity, led2))  # Clamp LED2 to valid range
        
        try:
            success, response, _ = self.client.send_message(self.MessageType.LED_SET_INTENSITY, {
                'led1': led1_value,
                'led2': led2_value
            })
            
            if success and response and response.payload.get('status') == 'success':
                self.led1_intensity = led1_value
                self.led2_intensity = led2_value
                self.led_active = True
                self.last_led_change = time.time()
                logger.info(f"LED intensity set: LED1={led1_value}mA (fixed to 0), LED2={led2_value}mA")
                return True
            else:
                error_msg = "Unknown error"
                if response and response.payload:
                    error_msg = response.payload.get('message', error_msg)
                logger.error(f"Failed to set LED intensity: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"Error setting LED intensity: {e}")
            return False
    
    def turn_on(self, intensity: Optional[int] = None) -> bool:
        """
        Turn on the LED flood illuminator.
        
        Args:
            intensity: Optional intensity for LED2 in mA (0-450)
                     If not specified, uses the current brightness level
                     
        Returns:
            True if successful, False otherwise
        """
        if not self.led_available:
            logger.warning("LED control not available")
            return False
        
        # Set intensity if specified
        if intensity is not None:
            return self.set_intensity(0, intensity)
        
        # Otherwise use current brightness level
        brightness_percent = self.brightness_levels[self.current_brightness]
        led2_value = int(self.max_intensity * brightness_percent / 100)
        
        try:
            # Try direct intensity setting first
            result = self.set_intensity(0, led2_value)
            if result:
                return True
                
            # Fall back to basic on command if intensity setting fails
            success, response, _ = self.client.send_message(self.MessageType.LED_ON, {})
            
            if success and response and response.payload.get('status') == 'success':
                self.led_active = True
                # Ensure LED1 is 0 after turning on
                self.set_intensity(0, led2_value)
                self.last_led_change = time.time()
                logger.info(f"LED turned on with LED1=0mA, LED2={led2_value}mA")
                return True
            else:
                error_msg = "Unknown error"
                if response and response.payload:
                    error_msg = response.payload.get('message', error_msg)
                logger.error(f"Failed to turn on LED: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"Error turning on LED: {e}")
            return False
    
    def turn_off(self) -> bool:
        """
        Turn off the LED flood illuminator.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.led_available:
            logger.warning("LED control not available")
            return False
        
        try:
            success, response, _ = self.client.send_message(self.MessageType.LED_OFF, {})
            
            if success and response and response.payload.get('status') == 'success':
                self.led_active = False
                self.last_led_change = time.time()
                logger.info("LED turned off")
                return True
            else:
                error_msg = "Unknown error"
                if response and response.payload:
                    error_msg = response.payload.get('message', error_msg)
                logger.error(f"Failed to turn off LED: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"Error turning off LED: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the LED flood illuminator.
        
        Returns:
            Dictionary containing status information:
            - led_available: Whether LED control is available
            - led_active: Whether LED is currently active
            - led1_intensity: Current LED1 intensity in mA
            - led2_intensity: Current LED2 intensity in mA
            - auto_activate: Whether auto-activation is enabled
            - brightness_level: Current brightness level name
        """
        try:
            success, response, _ = self.client.send_message(self.MessageType.LED_STATUS, {})
            
            if success and response and response.payload.get('status') == 'success':
                # Update local state
                self.led_available = response.payload.get('led_available', False)
                self.led_active = response.payload.get('led_active', False)
                self.led1_intensity = response.payload.get('led1_intensity', 0)
                self.led2_intensity = response.payload.get('led2_intensity', 0)
                
                return {
                    'led_available': self.led_available,
                    'led_active': self.led_active,
                    'led1_intensity': self.led1_intensity,
                    'led2_intensity': self.led2_intensity,
                    'auto_activate': self.auto_activate_enabled,
                    'brightness_level': self.current_brightness
                }
            else:
                logger.warning("Failed to get LED status")
                return {
                    'led_available': False,
                    'led_active': False,
                    'led1_intensity': 0,
                    'led2_intensity': 0,
                    'auto_activate': self.auto_activate_enabled,
                    'brightness_level': self.current_brightness,
                    'error': 'Failed to get LED status'
                }
        except Exception as e:
            logger.error(f"Error getting LED status: {e}")
            return {
                'led_available': False,
                'led_active': False,
                'led1_intensity': 0,
                'led2_intensity': 0,
                'auto_activate': self.auto_activate_enabled,
                'brightness_level': self.current_brightness,
                'error': str(e)
            }
    
    def toggle(self) -> bool:
        """
        Toggle the LED flood illuminator between on and off states.
        
        Returns:
            True if successful, False otherwise
        """
        if self.led_active:
            return self.turn_off()
        else:
            return self.turn_on()
    
    def set_brightness(self, level: str) -> bool:
        """
        Set brightness level by name.
        
        Args:
            level: Brightness level name ("off", "low", "medium", "high", "max")
            
        Returns:
            True if successful, False otherwise
        """
        if level not in self.brightness_levels:
            logger.error(f"Invalid brightness level: {level}")
            return False
        
        # Set new brightness level
        self.current_brightness = level
        
        # If LEDs are on, update intensity
        if self.led_active:
            brightness_percent = self.brightness_levels[level]
            led2_value = int(self.max_intensity * brightness_percent / 100)
            return self.set_intensity(0, led2_value)
        
        return True
    
    def set_auto_activate(self, enabled: bool) -> None:
        """
        Enable or disable auto-activation of LEDs.
        
        Args:
            enabled: Whether auto-activation should be enabled
        """
        self.auto_activate_enabled = enabled
        logger.info(f"LED auto-activation {'enabled' if enabled else 'disabled'}")
    
    def update_hand_presence(self, hand_detected: bool = False) -> None:
        """
        Update hand detection status for auto-activation.
        
        Args:
            hand_detected: Whether hands are currently detected
        """
        current_time = time.time()
        
        # Only update if auto-activation is enabled
        if not self.auto_activate_enabled:
            return
            
        # If hands are detected, update timestamp and activate LEDs if needed
        if hand_detected:
            self.last_hand_detection_time = current_time
            
            # Activate LEDs if they're off and cooldown period has passed
            if not self.led_active and current_time - self.last_led_change >= self.led_cooldown:
                logger.info("Auto-activating LEDs: hands detected")
                self.turn_on()
                self.last_led_change = current_time
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop()
        except Exception:
            pass