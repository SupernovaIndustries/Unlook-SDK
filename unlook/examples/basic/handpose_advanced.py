#!/usr/bin/env python3
"""Advanced hand tracking demo using the class interface."""

from unlook.client.scanning.handpose import HandTrackingDemo
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    # Create demo instance with custom settings
    demo = HandTrackingDemo(
        calibration_file=None,  # Use auto-loaded calibration
        use_led=True,          # Enable LED control
        led1_intensity=350,    # Custom intensity
        led2_intensity=350,
        verbose=True          # Enable debug output
    )
    
    # Connect to scanner
    if not demo.connect(timeout=10):
        print("Failed to connect to scanner")
        return
    
    # Setup tracker
    if not demo.setup_tracker():
        print("Failed to setup hand tracker")
        demo.cleanup()
        return
    
    # Run the demo
    try:
        demo.run(
            output_file="hand_tracking_data.json",  # Save tracking data
            visualize_3d=False                     # Disable 3D visualization
        )
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        # Cleanup resources
        demo.cleanup()


# Alternative: Using context manager
def main_with_context():
    with HandTrackingDemo(use_led=True) as demo:
        if demo.connect() and demo.setup_tracker():
            demo.run()


if __name__ == "__main__":
    main()
    # or use: main_with_context()