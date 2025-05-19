"""LED controller for AS1170 flood illuminator on Raspberry Pi server."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configuration for AS1170
I2C_BUS = 4  # I2C bus number
STROBE_PIN = 27  # GPIO pin for strobe

# LED control is only available on Raspberry Pi
try:
    from as1170 import led
    # Initialize with specific bus and strobe pin
    led.init(i2c_bus=I2C_BUS, strobe_pin=STROBE_PIN)
    LED_AVAILABLE = True
    logger.info(f"AS1170 LED controller initialized on I2C bus {I2C_BUS}, strobe pin {STROBE_PIN}")
except ImportError:
    LED_AVAILABLE = False
    logger.warning("AS1170 LED control not available. Install with: pip install AS1170-Python")
except Exception as e:
    LED_AVAILABLE = False
    logger.error(f"Failed to initialize AS1170: {e}")


class LEDController:
    """
    Controller for AS1170 LED flood illuminator.
    
    Hardware Configuration:
    - I2C Bus: 4
    - Strobe Pin: GPIO 27
    """
    
    def __init__(self):
        """Initialize LED controller."""
        self.led_active = False
        self.current_led1 = 0
        self.current_led2 = 0
        self.i2c_bus = I2C_BUS
        self.strobe_pin = STROBE_PIN
        
        if not LED_AVAILABLE:
            logger.warning("LED controller initialized but AS1170 library not available")
    
    def set_intensity(self, led1: int = 450, led2: int = 450) -> Dict[str, Any]:
        """
        Set LED intensity.
        
        Args:
            led1: LED1 intensity in mA (0-450)
            led2: LED2 intensity in mA (0-450)
            
        Returns:
            Response dictionary with status
        """
        if not LED_AVAILABLE:
            return {
                'status': 'error',
                'message': 'LED control not available'
            }
        
        try:
            # Clamp values
            led1 = max(0, min(450, led1))
            led2 = max(0, min(450, led2))
            
            logger.info(f"Setting LED intensity: LED1={led1}mA, LED2={led2}mA")
            led.set_intensity(led1=led1, led2=led2)
            
            self.current_led1 = led1
            self.current_led2 = led2
            
            # Turn on if not already on
            if not self.led_active:
                led.on()
                self.led_active = True
                
            return {
                'status': 'success',
                'led1': led1,
                'led2': led2
            }
            
        except Exception as e:
            logger.error(f"Failed to set LED intensity: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def turn_on(self) -> Dict[str, Any]:
        """Turn LEDs on."""
        if not LED_AVAILABLE:
            return {
                'status': 'error',
                'message': 'LED control not available'
            }
        
        try:
            led.on()
            self.led_active = True
            logger.info("LED turned on")
            return {'status': 'success'}
        except Exception as e:
            logger.error(f"Failed to turn on LED: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def turn_off(self) -> Dict[str, Any]:
        """Turn LEDs off."""
        if not LED_AVAILABLE:
            return {
                'status': 'error',
                'message': 'LED control not available'
            }
        
        try:
            led.off()
            self.led_active = False
            logger.info("LED turned off")
            return {'status': 'success'}
        except Exception as e:
            logger.error(f"Failed to turn off LED: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get LED status."""
        return {
            'status': 'success',
            'led_available': LED_AVAILABLE,
            'led_active': self.led_active,
            'led1_intensity': self.current_led1,
            'led2_intensity': self.current_led2,
            'i2c_bus': self.i2c_bus,
            'strobe_pin': self.strobe_pin
        }
    
    def cleanup(self):
        """Cleanup LED controller."""
        if self.led_active:
            self.turn_off()


# Global LED controller instance
led_controller = LEDController()