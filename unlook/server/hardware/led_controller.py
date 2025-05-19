"""LED controller for AS1170 flood illuminator on Raspberry Pi server."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configuration for AS1170
I2C_BUS = 4  # I2C bus number
STROBE_PIN = 27  # GPIO pin for strobe

# LED control is only available on Raspberry Pi
LED_AVAILABLE = False
led = None

try:
    logger.info("Attempting to import AS1170 library...")
    # Import the module itself
    import as1170
    logger.info(f"AS1170 module imported. Module contents: {dir(as1170)}")
    
    # The module might be used directly or have a different interface
    led = as1170  # Use the module directly
    LED_AVAILABLE = True
    logger.info(f"AS1170 library imported successfully. LED_AVAILABLE = {LED_AVAILABLE}")
except ImportError as e:
    logger.error(f"AS1170 LED control not available. Install with: pip install AS1170-Python. Error: {e}")
    logger.error(f"LED_AVAILABLE = {LED_AVAILABLE}")
except Exception as e:
    logger.error(f"Failed to import AS1170: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    logger.error(f"LED_AVAILABLE = {LED_AVAILABLE}")


class LEDController:
    """
    Controller for AS1170 LED flood illuminator.
    
    Hardware Configuration:
    - I2C Bus: 4
    - Strobe Pin: GPIO 27
    """
    
    def __init__(self):
        """Initialize LED controller."""
        logger.info(f"Initializing LED controller. LED_AVAILABLE = {LED_AVAILABLE}")
        self.led_active = False
        self.current_led1 = 0
        self.current_led2 = 0
        self.i2c_bus = I2C_BUS
        self.strobe_pin = STROBE_PIN
        self.initialized = False
        
        if LED_AVAILABLE:
            logger.info("LED is available, initializing hardware...")
            # Initialize hardware on controller creation
            self._init_hardware()
        else:
            logger.warning("LED controller initialized but AS1170 library not available")
    
    def _init_hardware(self):
        """Initialize the LED hardware."""
        if self.initialized:
            logger.info("Hardware already initialized, skipping.")
            return
            
        try:
            logger.info(f"Initializing AS1170 hardware with i2c_bus={self.i2c_bus}, strobe_pin={self.strobe_pin}")
            # Try different initialization methods
            init_methods = ['init', 'initialize', 'setup', 'Init', 'Initialize', 'Setup']
            initialized = False
            
            for method_name in init_methods:
                if hasattr(led, method_name):
                    method = getattr(led, method_name)
                    try:
                        method(i2c_bus=self.i2c_bus, strobe_pin=self.strobe_pin)
                        logger.info(f"Successfully called {method_name} method")
                        initialized = True
                        break
                    except TypeError:
                        # Method might not accept keyword arguments
                        try:
                            method(self.i2c_bus, self.strobe_pin)
                            logger.info(f"Successfully called {method_name} method with positional args")
                            initialized = True
                            break
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug(f"Failed to call {method_name}: {e}")
            
            if not initialized:
                logger.info("No initialization method found, assuming auto-initialization")
            
            self.initialized = True
            logger.info(f"AS1170 hardware initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AS1170 hardware: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.initialized = False
    
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
            
        # Ensure hardware is initialized
        if not self.initialized:
            self._init_hardware()
        
        try:
            # Clamp values
            led1 = max(0, min(450, led1))
            led2 = max(0, min(450, led2))
            
            logger.info(f"Setting LED intensity: LED1={led1}mA, LED2={led2}mA")
            
            # Try different method names for setting intensity
            intensity_methods = ['set_intensity', 'setIntensity', 'set_current', 'setCurrent', 'SetIntensity']
            success = False
            
            for method_name in intensity_methods:
                if hasattr(led, method_name):
                    method = getattr(led, method_name)
                    try:
                        method(led1=led1, led2=led2)
                        success = True
                        break
                    except TypeError:
                        # Try with positional arguments
                        try:
                            method(led1, led2)
                            success = True
                            break
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug(f"Failed {method_name}: {e}")
            
            if not success:
                raise Exception("No suitable method found for setting LED intensity")
            
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
            
        # Ensure hardware is initialized
        if not self.initialized:
            self._init_hardware()
        
        try:
            # Try different method names for turning on
            on_methods = ['on', 'On', 'turn_on', 'turnOn', 'enable', 'Enable']
            success = False
            
            for method_name in on_methods:
                if hasattr(led, method_name):
                    method = getattr(led, method_name)
                    try:
                        method()
                        success = True
                        break
                    except Exception as e:
                        logger.debug(f"Failed {method_name}: {e}")
            
            if not success:
                raise Exception("No suitable method found for turning LED on")
                
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
            
        # Ensure hardware is initialized
        if not self.initialized:
            self._init_hardware()
        
        try:
            # Try different method names for turning off
            off_methods = ['off', 'Off', 'turn_off', 'turnOff', 'disable', 'Disable']
            success = False
            
            for method_name in off_methods:
                if hasattr(led, method_name):
                    method = getattr(led, method_name)
                    try:
                        method()
                        success = True
                        break
                    except Exception as e:
                        logger.debug(f"Failed {method_name}: {e}")
            
            if not success:
                raise Exception("No suitable method found for turning LED off")
                
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
logger.info("Creating global LED controller instance...")
led_controller = LEDController()
logger.info(f"Global LED controller created: {led_controller}")