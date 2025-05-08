#!/usr/bin/env python3
"""
Example script demonstrating real-time 3D scanning with the UnLook scanner.

This example shows how to:
1. Configure and use the RealTimeScanner
2. Perform different types of scans (single-shot, continuous)
3. Use callbacks to monitor the scanning process
4. Process and save the scan results

Usage:
    python real_time_scanning_example.py [--continuous] [--quality {low,medium,high,ultra}]
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
import threading
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, '..')

from unlook import UnlookClient
from unlook.client.real_time_scanner import RealTimeScanner, ScanResult, ScanFrameData
from unlook.client.scan_config import RealTimeScannerConfig, ScanningMode, ScanningQuality, PatternType


# Ensure output directory exists
OUTPUT_DIR = "scan_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
scanner = None
visualization_window = None
visualization_image = None
visualization_lock = threading.Lock()


def on_scan_progress(progress: float, metadata: Dict[str, Any]):
    """Callback for scan progress updates."""
    logger.info(f"Scan progress: {progress*100:.1f}% - Frames: {metadata.get('frame_count', 0)}")


def on_scan_completed(result: ScanResult):
    """Callback for scan completion."""
    logger.info(f"Scan completed with {len(result.point_cloud) if result.point_cloud is not None else 0} points")
    
    # Save the result
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"scan_{timestamp}.ply")
        
        # Display some stats about the point cloud
        if result.point_cloud is not None:
            points = result.point_cloud
            logger.info(f"Point cloud stats:")
            logger.info(f"  - Points: {len(points)}")
            logger.info(f"  - X range: {np.min(points[:,0]):.2f} to {np.max(points[:,0]):.2f}")
            logger.info(f"  - Y range: {np.min(points[:,1]):.2f} to {np.max(points[:,1]):.2f}")
            logger.info(f"  - Z range: {np.min(points[:,2]):.2f} to {np.max(points[:,2]):.2f}")
        
        # Save the result
        scanner.save_result(result, filepath)
        logger.info(f"Saved scan result to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving scan result: {e}")


def on_scan_error(error_message: str, exception: Exception):
    """Callback for scan errors."""
    logger.error(f"Scan error: {error_message}")
    if exception:
        logger.error(f"Exception: {str(exception)}")


def on_frame_captured(frame_data: ScanFrameData):
    """Callback for frame captures."""
    global visualization_window, visualization_image
    
    # Log frame info periodically
    if frame_data.index % 5 == 0:
        logger.debug(f"Captured frame {frame_data.index} from camera {frame_data.camera_id} "
                   f"with pattern: {frame_data.pattern_info.get('pattern_type', 'unknown')}")
    
    # Visualize the frame (optional)
    if visualization_window is not None:
        with visualization_lock:
            # Copy the image for display
            visualization_image = frame_data.image.copy()
            
            # Add information overlay
            cv2.putText(
                visualization_image,
                f"Camera: {frame_data.camera_id} | Pattern: {frame_data.index} | "
                f"Type: {frame_data.pattern_info.get('pattern_type', 'unknown')}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


def on_result_ready(result: ScanResult):
    """Callback for when a new result is ready (for continuous mode)."""
    logger.info(f"New scan result ready with {len(result.point_cloud) if result.point_cloud is not None else 0} points")
    
    # In continuous mode, we could save each result or just the final one
    # For this example, we'll save every 5th result
    if result.metadata.get("sequence_index", 0) % 5 == 0:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(OUTPUT_DIR, f"scan_continuous_{timestamp}.ply")
            
            # Save the result
            scanner.save_result(result, filepath)
            logger.info(f"Saved continuous scan result to {filepath}")
        except Exception as e:
            logger.error(f"Error saving continuous scan result: {e}")


def update_visualization():
    """Update the visualization window."""
    global visualization_window, visualization_image
    
    # Create window
    window_name = "UnLook Real-Time Scanning"
    visualization_window = window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Display loop
    while visualization_window is not None:
        try:
            with visualization_lock:
                if visualization_image is not None:
                    cv2.imshow(window_name, visualization_image)
            
            # Check for key press (ESC to quit)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
                
            # Small delay
            time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            break
    
    # Clean up
    cv2.destroyAllWindows()
    visualization_window = None


def single_scan_example(client: UnlookClient, quality: ScanningQuality):
    """Perform a single 3D scan."""
    global scanner, visualization_window
    
    logger.info(f"=== Single 3D Scan Example (Quality: {quality.value}) ===")
    
    # Create scanner
    scanner = RealTimeScanner(client)
    
    # Configure scanner
    config = RealTimeScannerConfig.create_preset(quality)
    config.set_mode(ScanningMode.SINGLE)
    config.enable_save_raw_images(True, output_dir=OUTPUT_DIR)
    
    # Configure scanner with more explicit parameters
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_steps = 8 if quality != ScanningQuality.LOW else 4
    
    # Apply the configuration
    scanner.configure(config)
    
    # Register callbacks
    scanner.on_scan_progress(on_scan_progress)
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_error(on_scan_error)
    scanner.on_frame_captured(on_frame_captured)
    
    # Start visualization in a separate thread
    vis_thread = threading.Thread(target=update_visualization, daemon=True)
    vis_thread.start()
    
    try:
        # Start scanner
        logger.info("Starting single 3D scan...")
        if not scanner.start():
            logger.error("Failed to start scanner")
            return
        
        # Wait for scanning to complete - in SINGLE mode it will stop automatically
        while scanner.scanning:
            time.sleep(0.1)
        
        # Get the final result
        result = scanner.get_latest_result()
        
        if result and result.has_data():
            logger.info(f"Scan completed successfully with {len(result.point_cloud)} points")
            
            # Save the result (if not already saved by callback)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(OUTPUT_DIR, f"final_scan_{timestamp}.ply")
            scanner.save_result(result, filepath)
            
            logger.info(f"Final result saved to {filepath}")
        else:
            logger.warning("Scan completed but no valid result available")
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        scanner.stop()
    
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        scanner.stop()
    
    finally:
        # Cleanup
        visualization_window = None
        vis_thread.join(timeout=1.0)
        
        logger.info("Single scan completed")


def continuous_scan_example(client: UnlookClient, quality: ScanningQuality):
    """Perform continuous 3D scanning."""
    global scanner, visualization_window
    
    logger.info(f"=== Continuous 3D Scanning Example (Quality: {quality.value}) ===")
    
    # Create scanner
    scanner = RealTimeScanner(client)
    
    # Configure scanner
    config = RealTimeScannerConfig.create_preset(quality)
    config.set_mode(ScanningMode.CONTINUOUS)
    config.real_time_processing = True  # Enable real-time processing
    
    # For continuous scanning, we might want faster scanning with lower quality
    if quality == ScanningQuality.HIGH:
        # Override to use fewer pattern steps for better performance
        config.pattern_steps = 6
    
    # Apply the configuration
    scanner.configure(config)
    
    # Register callbacks
    scanner.on_scan_progress(on_scan_progress)
    scanner.on_scan_error(on_scan_error)
    scanner.on_frame_captured(on_frame_captured)
    scanner.on_result_ready(on_result_ready)
    
    # Start visualization in a separate thread
    vis_thread = threading.Thread(target=update_visualization, daemon=True)
    vis_thread.start()
    
    try:
        # Start scanner
        logger.info("Starting continuous 3D scanning...")
        if not scanner.start():
            logger.error("Failed to start scanner")
            return
        
        # Run for a fixed duration (15 seconds)
        scan_duration = 15.0  # seconds
        logger.info(f"Scanning will run for {scan_duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < scan_duration:
            # Check if user pressed ESC key
            if visualization_window is None:
                logger.info("Visualization window closed, stopping scan")
                break
            
            # Display stats periodically
            if int(time.time() - start_time) % 5 == 0:
                # Get latest statistics
                num_results = len(scanner.get_all_results())
                logger.info(f"Scanning in progress: {time.time() - start_time:.1f}s elapsed, "
                           f"{num_results} results generated")
            
            time.sleep(0.1)
        
        # Stop the scanner
        logger.info("Stopping continuous scanning...")
        scanner.stop()
        
        # Get and save the final result
        results = scanner.get_all_results()
        
        if results:
            logger.info(f"Scan completed with {len(results)} results")
            
            # Save the final result
            final_result = results[-1]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(OUTPUT_DIR, f"final_continuous_scan_{timestamp}.ply")
            scanner.save_result(final_result, filepath)
            
            logger.info(f"Final continuous scan result saved to {filepath}")
        else:
            logger.warning("Continuous scan completed but no results available")
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        scanner.stop()
    
    except Exception as e:
        logger.error(f"Error during continuous scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        scanner.stop()
    
    finally:
        # Cleanup
        visualization_window = None
        vis_thread.join(timeout=1.0)
        
        logger.info("Continuous scanning completed")


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="UnLook Real-Time 3D Scanning Example")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--quality", choices=["low", "medium", "high", "ultra"], 
                       default="medium", help="Scanning quality")
    
    args = parser.parse_args()
    
    # Connect to the UnLook scanner
    client = UnlookClient(client_name="RealTimeScanningDemo")
    
    # Map string to enum
    quality_map = {
        "low": ScanningQuality.LOW,
        "medium": ScanningQuality.MEDIUM,
        "high": ScanningQuality.HIGH,
        "ultra": ScanningQuality.ULTRA
    }
    quality = quality_map[args.quality]
    
    try:
        # Try to connect to any available scanner
        client.start_discovery()
        
        # Wait for scanners to be discovered
        logger.info("Waiting for scanners to be discovered...")
        time.sleep(3.0)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        
        if not scanners:
            logger.error("No scanner found. Please make sure the UnLook scanner server is running.")
            return
        
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} at {scanner_info.host}:{scanner_info.port}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Run the appropriate example
        if args.continuous:
            continuous_scan_example(client, quality)
        else:
            single_scan_example(client, quality)
        
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Always disconnect
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()