#!/usr/bin/env python3
"""Simple hand tracking demo with LED control.

This example demonstrates how to use the LED flood illuminator with the
UnLook scanner for hand tracking. The LED controller only controls LED2 intensity,
while LED1 is always kept at 0 intensity to prevent overheating and extend
the life of the LED module.

The AS1170 flood illuminator has two LED channels:
- LED1: Always set to 0 intensity
- LED2: Adjustable intensity from 0-450mA

Notes:
- Higher LED2 intensity provides better visibility in low-light conditions
- Reduce LED2 intensity when hands are detected clearly to save power
- LED intensity is automatically adjusted based on hand detection
"""

from unlook.client.scanning.handpose import HandTrackingDemo, run_demo

# Method 1: Using the convenience function
if __name__ == "__main__":
    # Run demo with default LED settings
    run_demo(
        calibration_file=None,  # Will use auto-loaded calibration
        use_led=True,          # Enable LED flood illuminator
        led1_intensity=0,      # LED1 is always 0 (regardless of what you pass here)
        led2_intensity=450,    # LED2 at full intensity
        verbose=False
    )
    
    # Or run with custom LED2 intensity
    # run_demo(
    #     use_led=True,
    #     led1_intensity=0,     # LED1 always stays at 0
    #     led2_intensity=300,   # LED2 at medium intensity
    # )
    
    # Or disable LED
    # run_demo(use_led=False)

"""
# Method 2: Using the standalone LED controller

If you want more control over the LED illuminator, you can use the LEDController class directly:

```python
from unlook.client import UnlookClient
from unlook.client.projector import LEDController

# Connect to scanner
client = UnlookClient()
client.connect()

# Create LED controller
led = LEDController(client)

# Use LED - LED1 is always set to 0 automatically
led.set_intensity(0, 450)  # Set LED2 to 450mA (LED1 is always 0)
led.turn_on()              # Turn on with current intensity settings
led.turn_off()             # Turn off LED

# Get LED status
status = led.get_status()
print(f"LED active: {status['led_active']}")
print(f"LED2 intensity: {status['led2_intensity']}mA")

# Pulse the LED for a short duration
led.pulse(duration=0.5, intensity=450)  # Flash LED2 at 450mA for 0.5 seconds
```
"""