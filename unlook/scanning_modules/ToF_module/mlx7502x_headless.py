#!/usr/bin/env python3
"""
MLX7502x ToF Sensor - Headless Version (No GUI)
==============================================

Version for headless operation - saves data to files without display.
"""

import cv2
import numpy as np
import time
import os

def shift_pixel_values(frame, shift_value=4):
    """Shift pixel values left by shift_value bits (12-bit to 16-bit)"""
    return (frame.astype(np.int16) << shift_value)

def compute_magnitude_phase(frames):
    """Compute magnitude and phase from 4 phase-shifted frames"""
    # Convert to float64
    frame_0 = frames[0].astype(np.float64)
    frame_180 = frames[1].astype(np.float64) 
    frame_90 = frames[2].astype(np.float64)
    frame_270 = frames[3].astype(np.float64)
    
    # Calculate I and Q components
    I = frame_0 - frame_180
    Q = frame_90 - frame_270
    
    # Calculate magnitude
    magnitude = np.sqrt(I**2 + Q**2)
    
    # Calculate phase in degrees
    phase = np.arctan2(Q, I) * 180 / np.pi
    
    return magnitude, phase, I, Q

def save_visualization(magnitude, phase, frame_num):
    """Save magnitude and phase as images"""
    # Magnitude to 8-bit
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        mag_8bit = ((magnitude - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
    else:
        mag_8bit = np.zeros_like(magnitude, dtype=np.uint8)
    
    # Phase to 8-bit (0-360 degrees to 0-255)
    phase_8bit = (np.mod(phase, 360) / 360.0 * 255).astype(np.uint8)
    
    # Save magnitude
    cv2.imwrite(f"magnitude_{frame_num:03d}.png", mag_8bit)
    
    # Save phase (create a simple colormap manually)
    # Use HSV colormap-like visualization
    phase_colored = np.zeros((phase.shape[0], phase.shape[1], 3), dtype=np.uint8)
    phase_colored[:, :, 0] = phase_8bit  # Hue
    phase_colored[:, :, 1] = 255  # Saturation
    phase_colored[:, :, 2] = mag_8bit  # Value (use magnitude for brightness)
    
    # Convert HSV to BGR
    phase_bgr = cv2.cvtColor(phase_colored, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"phase_{frame_num:03d}.png", phase_bgr)
    
    return f"magnitude_{frame_num:03d}.png", f"phase_{frame_num:03d}.png"

def main():
    print("MLX7502x ToF Sensor - Headless Demo")
    print("==================================")
    print("Saving frames to disk without GUI...")
    
    # Open video capture
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("Error: Could not open /dev/video0")
        return 1
    
    print("✓ Video device opened successfully!")
    
    # Set parameters
    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    
    # Check actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Capture settings: {width}x{height} @ {fps} FPS")
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"tof_capture_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    frame_set = 0
    total_captures = 10  # Capture 10 sets of frames
    
    print(f"\nCapturing {total_captures} frame sets...")
    
    try:
        for capture_num in range(total_captures):
            print(f"\nCapture {capture_num + 1}/{total_captures}:")
            
            start_time = time.time()
            
            # Capture 4 frames for phase shifting
            frames = []
            
            # Wait for first frame with timeout
            capture_start = time.time()
            while True:
                ret, frame = cap.read()
                if ret or (time.time() - capture_start) > 0.05:  # 50ms timeout
                    break
            
            if ret:
                # Process and store first frame
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(shift_pixel_values(frame))
                print(f"  ✓ Frame 0: {frame.shape}")
                
                # Capture remaining 3 frames
                for i in range(3):
                    ret, frame = cap.read()
                    if ret:
                        if len(frame.shape) == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames.append(shift_pixel_values(frame))
                        print(f"  ✓ Frame {i+1}: {frame.shape}")
                    else:
                        print(f"  ✗ Failed to capture frame {i+1}")
                        frames.append(np.zeros((height, width), dtype=np.int16))
            
            # Process if we have 4 frames
            if len(frames) == 4:
                # Compute magnitude and phase
                magnitude, phase, I, Q = compute_magnitude_phase(frames)
                
                # Save raw data
                np.save(f"{output_dir}/magnitude_raw_{capture_num:03d}.npy", magnitude)
                np.save(f"{output_dir}/phase_raw_{capture_num:03d}.npy", phase)
                np.save(f"{output_dir}/I_component_{capture_num:03d}.npy", I)
                np.save(f"{output_dir}/Q_component_{capture_num:03d}.npy", Q)
                
                # Save all 4 raw frames
                for i, raw_frame in enumerate(frames):
                    np.save(f"{output_dir}/frame_{capture_num:03d}_phase_{i}.npy", raw_frame)
                
                # Save visualizations
                current_dir = os.getcwd()
                os.chdir(output_dir)
                mag_file, phase_file = save_visualization(magnitude, phase, capture_num)
                os.chdir(current_dir)
                
                # Statistics
                mag_mean = magnitude.mean()
                mag_std = magnitude.std()
                mag_min = magnitude.min()
                mag_max = magnitude.max()
                
                phase_mean = phase.mean()
                phase_std = phase.std()
                
                print(f"  ✓ Magnitude: mean={mag_mean:.1f}, std={mag_std:.1f}, range=[{mag_min:.1f}, {mag_max:.1f}]")
                print(f"  ✓ Phase: mean={phase_mean:.1f}°, std={phase_std:.1f}°")
                print(f"  ✓ Saved: {mag_file}, {phase_file}")
                print(f"  ✓ Processing time: {(time.time() - start_time)*1000:.1f}ms")
                
                # Save statistics to text file
                with open(f"{output_dir}/stats_{capture_num:03d}.txt", 'w') as f:
                    f.write(f"Capture {capture_num}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Magnitude: mean={mag_mean:.3f}, std={mag_std:.3f}, min={mag_min:.3f}, max={mag_max:.3f}\n")
                    f.write(f"Phase: mean={phase_mean:.3f}, std={phase_std:.3f}\n")
                    f.write(f"Processing time: {(time.time() - start_time)*1000:.1f}ms\n")
            
            # Wait between captures
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Cleanup
    cap.release()
    
    # Create summary
    print(f"\n{'='*50}")
    print("CAPTURE COMPLETE!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"Files per capture:")
    print(f"  - magnitude_raw_XXX.npy (raw magnitude data)")
    print(f"  - phase_raw_XXX.npy (raw phase data)")
    print(f"  - I_component_XXX.npy (I component)")
    print(f"  - Q_component_XXX.npy (Q component)")
    print(f"  - frame_XXX_phase_Y.npy (individual frames)")
    print(f"  - magnitude_XXX.png (magnitude image)")
    print(f"  - phase_XXX.png (phase image)")
    print(f"  - stats_XXX.txt (statistics)")
    
    # Count files
    files = os.listdir(output_dir)
    print(f"\nTotal files created: {len(files)}")
    
    # Create README
    with open(f"{output_dir}/README.txt", 'w') as f:
        f.write("MLX75027 ToF Sensor Capture Results\n")
        f.write("===================================\n\n")
        f.write(f"Capture timestamp: {timestamp}\n")
        f.write(f"Sensor resolution: {width}x{height}\n")
        f.write(f"Frame rate: {fps} FPS\n")
        f.write(f"Total captures: {total_captures}\n")
        f.write(f"Total files: {len(files)}\n\n")
        f.write("File descriptions:\n")
        f.write("- magnitude_raw_XXX.npy: Raw magnitude data (float64)\n")
        f.write("- phase_raw_XXX.npy: Raw phase data in degrees (float64)\n")
        f.write("- I_component_XXX.npy: I component (0° - 180°)\n")
        f.write("- Q_component_XXX.npy: Q component (90° - 270°)\n")
        f.write("- frame_XXX_phase_Y.npy: Individual frames (Y=0,1,2,3 for phases)\n")
        f.write("- magnitude_XXX.png: Magnitude visualization\n")
        f.write("- phase_XXX.png: Phase visualization with colormap\n")
        f.write("- stats_XXX.txt: Capture statistics\n")
    
    print(f"\n✓ Created README.txt with capture info")
    print(f"\nTo view results on your PC, copy the '{output_dir}' folder")
    print("The .npy files can be loaded with: numpy.load('filename.npy')")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())