#!/usr/bin/env python3
"""Simple hand tracking demo with LED control."""

from unlook.client.scanning.handpose import HandTrackingDemo, run_demo

# Method 1: Using the convenience function
if __name__ == "__main__":
    # Run demo with default LED settings
    run_demo(
        calibration_file=None,  # Will use auto-loaded calibration
        use_led=True,          # Enable LED flood illuminator
        led1_intensity=450,    # Full intensity
        led2_intensity=450,    # Full intensity
        verbose=False
    )
    
    # Or run with custom LED settings
    # run_demo(
    #     use_led=True,
    #     led1_intensity=300,   # Medium intensity
    #     led2_intensity=300,   # Medium intensity
    # )
    
    # Or disable LED
    # run_demo(use_led=False)