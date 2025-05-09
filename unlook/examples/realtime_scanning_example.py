#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time 3D Scanning Example

This example demonstrates how to use the real-time 3D scanner for continuous scanning
at higher frame rates, suitable for handheld scanning.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("realtime_scanning")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import numpy as np
    import cv2
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    logger.error("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Visualization will be disabled.")
    logger.warning("Install open3d for better results: pip install open3d")
    OPEN3D_AVAILABLE = False

# Try to import optional GPU libraries
try:
    # Check CUDA availability
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA = torch.cuda.is_available()
    if TORCH_CUDA:
        logger.info(f"PyTorch CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("PyTorch found but CUDA not available")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA = False
    logger.warning("PyTorch not found, neural network enhancement disabled")

# Import scanner and client
from unlook import UnlookClient
from unlook.client.realtime_scanner import create_realtime_scanner, RealTimeScanner


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time 3D Scanning Example')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for scan results')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to stereo calibration file')
    parser.add_argument('--quality', type=str, default='medium',
                        choices=['fast', 'medium', 'high', 'ultra'],
                        help='Scan quality preset')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout in seconds for scanner discovery')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-neural-network', action='store_true',
                        help='Disable neural network enhancement')
    parser.add_argument('--continuous', action='store_true',
                        help='Enable continuous scanning mode')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable Open3D real-time visualization')
    parser.add_argument('--opencv-preview', action='store_true',
                        help='Show OpenCV preview (works without Open3D)')
    parser.add_argument('--record', action='store_true',
                        help='Record scan data to files')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


class ScanVisualizer:
    """Visualizer for real-time scanning."""

    def __init__(self):
        """Initialize visualizer."""
        self.window_created = False
        self.vis = None
        self.pcd = None
        self.fps_text = None
        self.scan_count = 0
        self.last_update_time = 0
        self.running = False
        self.viz_thread = None

        # Visualization parameters
        self.update_interval = 0.03  # seconds - higher refresh rate

        # Check Open3D availability
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for visualization")
            raise ImportError("Open3D is required for visualization")

    def initialize(self):
        """Initialize Open3D visualization."""
        import threading

        # Start visualization in a separate thread for real-time updates
        self.viz_thread = threading.Thread(target=self._run_visualization_loop, daemon=True)

        # Create visualization window first in main thread
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Real-time 3D Scanning", width=1280, height=720)

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        self.vis.add_geometry(coord_frame)

        # Create empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # Set view control
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.5)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])

        # Create render options
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        self.window_created = True
        self.running = True

        # Start visualization thread
        self.viz_thread.start()
        logger.info("Open3D real-time visualization started")

    def _run_visualization_loop(self):
        """Run the visualization loop in a separate thread."""
        while self.running:
            try:
                if not self.vis.poll_events():
                    # Window was closed
                    self.running = False
                    break

                self.vis.update_renderer()
                time.sleep(0.01)  # Small sleep to prevent CPU usage spike
            except Exception as e:
                logger.error(f"Error in visualization thread: {e}")
                self.running = False
                break

        # Cleanup if thread exits
        if self.window_created:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.window_created = False
            logger.info("Visualization window closed")

    def update(self, point_cloud, fps, scan_count):
        """Update visualization with new point cloud data."""
        if not self.window_created:
            self.initialize()
            return

        # Don't update if window is closed or not running
        if not self.running:
            return

        # Check if it's time to update the visualization
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        self.scan_count = scan_count

        # Update point cloud
        if point_cloud is not None:
            try:
                if isinstance(point_cloud, np.ndarray):
                    # Convert numpy array to Open3D point cloud
                    temp_pcd = o3d.geometry.PointCloud()
                    temp_pcd.points = o3d.utility.Vector3dVector(point_cloud)
                    self.pcd.points = temp_pcd.points
                else:
                    # Directly use Open3D point cloud
                    self.pcd.points = point_cloud.points

                # Update colors
                if len(self.pcd.points) > 0:
                    # Generate random colors for visualization
                    colors = np.zeros((len(self.pcd.points), 3))
                    # Assign colors based on Z value for better visualization
                    points = np.asarray(self.pcd.points)
                    z_values = points[:, 2]
                    z_min, z_max = np.min(z_values), np.max(z_values)
                    if z_max > z_min:
                        normalized_z = (z_values - z_min) / (z_max - z_min)
                        # Create color gradient (blue to red)
                        colors[:, 0] = normalized_z  # Red
                        colors[:, 2] = 1.0 - normalized_z  # Blue
                    else:
                        # Default to light blue if all points have same Z
                        colors[:, 2] = 1.0
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)

                # Update geometry - this is thread-safe in Open3D
                self.vis.update_geometry(self.pcd)

                # Update info text
                info_text = f"FPS: {fps:.1f} | Scan #: {scan_count} | Points: {len(self.pcd.points)}"
                print(info_text, end="\r", flush=True)

            except Exception as e:
                logger.error(f"Error updating visualization: {e}")

    def close(self):
        """Close visualization window."""
        self.running = False

        # Wait for visualization thread to terminate
        if self.viz_thread and self.viz_thread.is_alive():
            self.viz_thread.join(timeout=2.0)

        # Make sure window is destroyed
        if self.window_created:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.window_created = False


class ScanRecorder:
    """Recorder for 3D scan data."""
    
    def __init__(self, output_dir):
        """
        Initialize recorder.
        
        Args:
            output_dir: Base directory for scan recordings
        """
        self.output_dir = output_dir
        self.recording = False
        self.frame_count = 0
        self.start_time = 0
        self.current_session_dir = None
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self):
        """Start a new recording session."""
        if self.recording:
            logger.warning("Recording already in progress")
            return
        
        # Create new session directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = os.path.join(self.output_dir, f"scan_session_{timestamp}")
        os.makedirs(self.current_session_dir, exist_ok=True)
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info(f"Started recording to {self.current_session_dir}")
    
    def record_frame(self, point_cloud, scan_count, fps):
        """
        Record a frame to the current session.
        
        Args:
            point_cloud: Point cloud data
            scan_count: Current scan count
            fps: Current FPS
        """
        if not self.recording:
            return
        
        # Create frame directory
        frame_dir = os.path.join(self.current_session_dir, f"frame_{self.frame_count:04d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        # Save point cloud
        if point_cloud is not None:
            if OPEN3D_AVAILABLE and isinstance(point_cloud, o3d.geometry.PointCloud):
                cloud_path = os.path.join(frame_dir, "point_cloud.ply")
                o3d.io.write_point_cloud(cloud_path, point_cloud)
            elif isinstance(point_cloud, np.ndarray):
                cloud_path = os.path.join(frame_dir, "point_cloud.npy")
                np.save(cloud_path, point_cloud)
        
        # Save metadata
        metadata = {
            "frame_number": self.frame_count,
            "scan_count": scan_count,
            "fps": fps,
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
            "num_points": len(point_cloud.points) if hasattr(point_cloud, "points") else 
                         (len(point_cloud) if point_cloud is not None else 0)
        }
        
        # Write metadata
        import json
        with open(os.path.join(frame_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.frame_count += 1
    
    def stop_recording(self):
        """Stop the current recording session."""
        if not self.recording:
            return
        
        self.recording = False
        logger.info(f"Stopped recording. Recorded {self.frame_count} frames to {self.current_session_dir}")


def on_new_frame_callback(point_cloud, scan_count, fps):
    """
    Callback function for new frames.
    
    Args:
        point_cloud: New point cloud data
        scan_count: Current scan count
        fps: Current frames per second
    """
    # This is just a placeholder - update the global states in main
    pass


def main():
    """Run the real-time 3D scanning example."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create output directory if recording is enabled
    if args.record:
        if args.output:
            output_dir = args.output
        else:
            # Create default output directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"scans/realtime_{timestamp}_{args.quality}"
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Scan recordings will be saved to: {output_dir}")
        
        # Initialize recorder
        recorder = ScanRecorder(output_dir)
    else:
        recorder = None
    
    # Initialize visualizer if requested
    if args.visualize:
        if not OPEN3D_AVAILABLE:
            logger.error("Visualization requested but Open3D is not available.")
            logger.error("Please install Open3D: pip install open3d")
            return 1
        
        visualizer = ScanVisualizer()
    else:
        visualizer = None
    
    # Initialize client and connect to scanner
    logger.info("Initializing client and connecting to scanner...")
    try:
        # Create client with auto-discovery
        client = UnlookClient(auto_discover=True)
        
        # Start discovery
        client.start_discovery()
        logger.info(f"Discovering scanners for {args.timeout} seconds...")
        time.sleep(args.timeout)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            logger.error("No scanners found. Please ensure scanner hardware is connected and powered on.")
            return 1
            
        # Connect to the first scanner
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} ({scanner_info.uuid})")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return 1
            
        logger.info(f"Successfully connected to scanner: {scanner_info.name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize client and connect to scanner: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Create real-time scanner with callback
    logger.info(f"Creating real-time scanner with {args.quality} quality preset...")
    try:
        # Import RealTimeScanConfig for custom configuration
        from unlook.client.realtime_scanner import RealTimeScanConfig

        # Create configuration for real-time scanner
        config = RealTimeScanConfig()
        config.set_quality_preset(args.quality)

        # Configure GPU options
        if args.no_gpu:
            config.use_gpu = False
            logger.info("GPU acceleration disabled")
        else:
            logger.info(f"GPU acceleration {'enabled' if config.use_gpu else 'not available'}")

        # Configure neural network options
        if args.no_neural_network:
            config.use_neural_network = False
            logger.info("Neural network enhancement disabled")
        else:
            logger.info(f"Neural network enhancement {'enabled' if config.use_neural_network else 'not available'}")
            if config.use_neural_network:
                logger.info(f"  - Denoising strength: {config.nn_denoise_strength:.1f}")
                logger.info(f"  - Upsampling: {'enabled' if config.nn_upsample else 'disabled'}")
                if config.nn_upsample and config.nn_target_points:
                    logger.info(f"  - Target points: {config.nn_target_points}")

        # Configure scanning mode
        if args.continuous:
            config.continuous_scanning = True
            logger.info("Continuous scanning mode enabled")

        # Create real-time scanner with callback for new frames
        scanner = create_realtime_scanner(
            client=client,
            config=config,
            calibration_file=args.calibration,
            on_new_frame=on_new_frame_callback
        )
    except Exception as e:
        logger.error(f"Failed to create scanner: {e}")
        client.disconnect()
        return 1
    
    # Capture signals for clean shutdown
    import signal
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received signal to stop, shutting down...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start recording if enabled
    if recorder:
        recorder.start_recording()
    
    # Start scanner
    logger.info("Starting real-time scanning...")
    scanner.start()
    
    try:
        # Main application loop
        while running:
            # Get current point cloud and stats
            point_cloud = scanner.get_current_point_cloud()
            fps = scanner.get_fps()
            scan_count = scanner.get_scan_count()

            # Always print scan info at a regular interval
            if scan_count % 10 == 0:
                points_count = 0
                if point_cloud is not None:
                    if hasattr(point_cloud, 'points'):
                        points_count = len(point_cloud.points)
                    else:
                        points_count = len(point_cloud) if isinstance(point_cloud, np.ndarray) else 0

                    logger.info(f"Scan #{scan_count} | FPS: {fps:.1f} | Points: {points_count}")
                else:
                    logger.info(f"Scan #{scan_count} | FPS: {fps:.1f} | No points yet")

            # Update Open3D visualization if enabled
            if visualizer:
                # Create empty point cloud if none exists
                if point_cloud is None and OPEN3D_AVAILABLE:
                    empty_pcd = o3d.geometry.PointCloud()
                    visualizer.update(empty_pcd, fps, scan_count)
                elif point_cloud is not None:
                    visualizer.update(point_cloud, fps, scan_count)

                # Stop if visualization window is closed
                if hasattr(visualizer, 'running') and not visualizer.running:
                    logger.info("Visualization window closed, stopping...")
                    running = False

            # Show OpenCV preview if enabled (even with empty point cloud)
            if args.opencv_preview:
                if point_cloud is None:
                    # Create a blank image for preview
                    blank_img = np.zeros((600, 800, 3), dtype=np.uint8)
                    cv2.putText(blank_img, "Waiting for point cloud...", (50, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                    cv2.imshow("Real-time OpenCV Preview", blank_img)
                else:
                    scanner.show_preview(window_name="Real-time OpenCV Preview")

            # Record frame if enabled
            if recorder and recorder.recording and point_cloud is not None:
                recorder.record_frame(point_cloud, scan_count, fps)

            # Process keyboard inputs for OpenCV window (ESC to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                logger.info("ESC key pressed, stopping...")
                running = False

            # Brief pause to prevent CPU usage spike
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Stop scanning
        logger.info("Stopping real-time scanner...")
        scanner.stop()
        
        # Stop recording
        if recorder and recorder.recording:
            recorder.stop_recording()
        
        # Close visualization
        if visualizer:
            visualizer.close()
        
        # Disconnect from scanner
        client.disconnect()
        logger.info("Disconnected from scanner")
    
    logger.info("Real-time scanning completed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)