"""
Client for controlling the UnLook scanner LED flood illuminator.
"""

import logging
from typing import Dict, Any, Optional, Tuple

# Try importing from the SDK
try:
    from ...core.protocol import MessageType
except ImportError:
    # Define fallback classes for when the core modules are not available
    class MessageType:
        LED_SET_INTENSITY = "led_set_intensity"
        LED_ON = "led_on"
        LED_OFF = "led_off"
        LED_STATUS = "led_status"

logger = logging.getLogger(__name__)


class LEDController:
    """
    Client for controlling the UnLook scanner's LED flood illuminator.
    
    This class provides a simple interface to control the AS1170 flood illuminator
    attached to the UnLook scanner. 
    
    The LED illuminator provides programmable intensity in two channels 
    (LED1 and LED2) with a range of 0-450mA.
    
    Typical usage:
        ```python
        from unlook.client import UnlookClient
        from unlook.client.projector import LEDController
        
        # Connect to scanner
        client = UnlookClient()
        client.connect()
        
        # Create LED controller
        led = LEDController(client)
        
        # Use LED
        led.turn_on()
        led.set_intensity(400, 400)  # Set both LEDs to 400mA
        led.turn_off()
        ```
    """
    
    def __init__(self, client):
        """
        Initialize LED controller client.
        
        Args:
            client: Main UnlookClient
        """
        self.client = client
        self.led_available = False
        self.led1_intensity = 0
        self.led2_intensity = 0
        self.led_active = False
        
        # Check LED availability
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if LED control is available on the server."""
        try:
            success, response, _ = self.client.send_message(MessageType.LED_STATUS, {})
            if success and response and response.payload.get('status') == 'success':
                self.led_available = response.payload.get('led_available', False)
                self.led_active = response.payload.get('led_active', False)
                self.led1_intensity = response.payload.get('led1_intensity', 0)
                self.led2_intensity = response.payload.get('led2_intensity', 0)
                
                logger.info(f"LED control available: {self.led_available}")
                if self.led_available:
                    logger.info(f"LED state: active={self.led_active}, LED1={self.led1_intensity}mA, LED2={self.led2_intensity}mA")
                return self.led_available
            else:
                logger.warning("Failed to get LED status from server")
                return False
        except Exception as e:
            logger.error(f"Error checking LED availability: {e}")
            return False
    
    def set_intensity(self, led1: int = 0, led2: int = 450) -> bool:
        """
        Set the intensity of both LED channels.
        
        Args:
            led1: LED1 intensity in mA (0-450), will always be set to 0
            led2: LED2 intensity in mA (0-450)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.led_available:
            logger.warning("LED control not available")
            return False
        
        # Always set LED1 to 0, only adjust LED2
        led1 = 0  # LED1 is always set to 0
        led2 = max(0, min(450, led2))  # Clamp LED2 to valid range
        
        try:
            success, response, _ = self.client.send_message(MessageType.LED_SET_INTENSITY, {
                'led1': led1,
                'led2': led2
            })
            
            if success and response and response.payload.get('status') == 'success':
                self.led1_intensity = led1
                self.led2_intensity = led2
                self.led_active = True
                logger.info(f"LED intensity set: LED1={led1}mA (fixed to 0), LED2={led2}mA")
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
                      If not specified, uses the last set intensity or maximum
                      LED1 will always be set to 0
                      
        Returns:
            True if successful, False otherwise
        """
        if not self.led_available:
            logger.warning("LED control not available")
            return False
        
        # Set intensity if specified
        if intensity is not None:
            return self.set_intensity(0, intensity)
        
        # Otherwise just turn it on
        try:
            success, response, _ = self.client.send_message(MessageType.LED_ON, {})
            
            if success and response and response.payload.get('status') == 'success':
                self.led_active = True
                # Ensure LED1 is 0 after turning on
                self.set_intensity(0, self.led2_intensity or 450)
                logger.info("LED turned on with LED1=0mA")
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
            success, response, _ = self.client.send_message(MessageType.LED_OFF, {})
            
            if success and response and response.payload.get('status') == 'success':
                self.led_active = False
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
        """
        try:
            success, response, _ = self.client.send_message(MessageType.LED_STATUS, {})
            
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
                    'led2_intensity': self.led2_intensity
                }
            else:
                logger.warning("Failed to get LED status")
                return {
                    'led_available': False,
                    'led_active': False,
                    'led1_intensity': 0,
                    'led2_intensity': 0,
                    'error': 'Failed to get LED status'
                }
        except Exception as e:
            logger.error(f"Error getting LED status: {e}")
            return {
                'led_available': False,
                'led_active': False,
                'led1_intensity': 0,
                'led2_intensity': 0,
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
    
    def pulse(self, duration: float = 0.5, intensity: int = 450) -> bool:
        """
        Flash the LED2 flood illuminator for a short duration.
        LED1 will always remain at 0.
        
        Args:
            duration: Duration in seconds
            intensity: Flash intensity for LED2 in mA (0-450)
            
        Returns:
            True if successful, False otherwise
        """
        import time
        
        if not self.led_available:
            logger.warning("LED control not available")
            return False
        
        # Save current state
        was_active = self.led_active
        prev_led2 = self.led2_intensity
        
        # Flash the LED (LED1 always 0)
        success = self.set_intensity(0, intensity)
        if not success:
            return False
        
        # Wait for the specified duration
        time.sleep(duration)
        
        # Restore previous state
        if was_active:
            return self.set_intensity(0, prev_led2)
        else:
            return self.turn_off()