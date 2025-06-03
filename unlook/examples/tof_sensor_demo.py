#!/usr/bin/env python3
"""
MLX75027 ToF Sensor Demo
========================

This example demonstrates how to use the MLX75027 Time-of-Flight sensor
with the Unlook SDK.
"""

import sys
import os
import numpy as np
import cv2
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unlook.scanning_modules.ToF_module import MLX75027, ToFConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def depth_callback(magnitude, phase):
    """Example callback for processing depth data"""
    # Calculate statistics
    mag_mean = np.mean(magnitude)
    mag_std = np.std(magnitude)
    phase_mean = np.mean(phase)
    phase_std = np.std(phase)
    
    logger.info(f"Magnitude - Mean: {mag_mean:.2f}, Std: {mag_std:.2f}")
    logger.info(f"Phase - Mean: {phase_mean:.2f}°, Std: {phase_std:.2f}°")


def single_capture_demo(sensor):
    """Demonstrate single capture mode"""
    print("\n=== Single Capture Demo ===")
    
    # Capture phase frames
    print("Capturing phase frames...")
    frames = sensor.capture_phase_frames()
    
    # Compute depth
    print("Computing depth...")
    magnitude, phase = sensor.compute_depth()
    
    # Display statistics
    print(f"\nCapture Results:")
    print(f"  Magnitude shape: {magnitude.shape}")
    print(f"  Phase shape: {phase.shape}")
    print(f"  Magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print(f"  Phase range: [{phase.min():.2f}°, {phase.max():.2f}°]")
    
    # Visualize
    mag_rgb, phase_rgb, phase_cm = sensor.visualize_results(magnitude, phase)
    
    # Save results
    cv2.imwrite("tof_magnitude.png", mag_rgb)
    cv2.imwrite("tof_phase.png", phase_rgb)
    cv2.imwrite("tof_phase_colormap.png", phase_cm)
    print("\nSaved images: tof_magnitude.png, tof_phase.png, tof_phase_colormap.png")
    
    # Display
    cv2.imshow("ToF Magnitude", mag_rgb)
    cv2.imshow("ToF Phase Colormap", phase_cm)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def continuous_capture_demo(sensor, duration=None):
    """Demonstrate continuous capture mode"""
    print("\n=== Continuous Capture Demo ===")
    print("Press 'q' to quit")
    
    if duration:
        print(f"Running for {duration} seconds...")
        import time
        start_time = time.time()
        
        def timed_callback(magnitude, phase):
            if time.time() - start_time > duration:
                return False
            depth_callback(magnitude, phase)
            return True
            
        # Modified continuous capture with timing
        sensor.capture_continuous(display=True, callback=timed_callback)
    else:
        # Run until user quits
        sensor.capture_continuous(display=True, callback=depth_callback)


def main():
    parser = argparse.ArgumentParser(description="MLX75027 ToF Sensor Demo")
    parser.add_argument("--device", default="/dev/video0", 
                        help="Video device path (default: /dev/video0)")
    parser.add_argument("--subdevice", default="/dev/v4l-subdev0",
                        help="V4L2 subdevice path (default: /dev/v4l-subdev0)")
    parser.add_argument("--fps", type=int, default=12,
                        help="Frame rate (default: 12)")
    parser.add_argument("--mode", choices=["single", "continuous"], default="continuous",
                        help="Capture mode (default: continuous)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration for continuous mode in seconds")
    parser.add_argument("--integration-time", type=int, default=1000,
                        help="Integration time in microseconds (default: 1000)")
    parser.add_argument("--modulation-freq", type=int, default=10000000,
                        help="Modulation frequency in Hz (default: 10MHz)")
    
    args = parser.parse_args()
    
    # Configure sensor
    config = ToFConfig(
        fps=args.fps,
        phase_sequence=[0, 180, 90, 270],
        time_integration=[args.integration_time] * 4,
        modulation_frequency=args.modulation_freq
    )
    
    print("MLX75027 ToF Sensor Demo")
    print("========================")
    print(f"Device: {args.device}")
    print(f"Subdevice: {args.subdevice}")
    print(f"FPS: {config.fps}")
    print(f"Integration time: {config.time_integration[0]} µs")
    print(f"Modulation frequency: {config.modulation_frequency/1e6:.1f} MHz")
    
    try:
        # Use sensor
        with MLX75027(args.device, args.subdevice, config) as sensor:
            if args.mode == "single":
                single_capture_demo(sensor)
            else:
                continuous_capture_demo(sensor, args.duration)
                
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())