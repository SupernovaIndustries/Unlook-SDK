#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time 3D Scanning Example

This example demonstrates how to use the real-time 3D scanner for continuous scanning
at moderate frame rates, suitable for handheld 3D scanning with stereo cameras.

The scanner includes several robust features for reliable operation:
- CPU-optimized Gray code pattern decoding for stereo setups
- Memory-efficient point cloud processing
- Configurable synchronization between projector and cameras
- Configurable epipolar constraints for accurate stereo matching
- Statistical outlier filtering based on disparity consistency
- Enhanced triangulation with comprehensive validation
- Real-time point cloud visualization options (Open3D or OpenCV)

Usage examples:
  # Basic scanning with default settings (medium quality)
  python realtime_scanning_example.py

  # Fast quality scanning with visualization (better performance)
  python realtime_scanning_example.py --quality fast --opencv-preview

  # Debug mode with detailed diagnostics (helpful for troubleshooting)
  python realtime_scanning_example.py --debug --epipolar-tol 15.0

  # Skip image rectification and use enhanced sensitivity (recommended for initial tests)
  python realtime_scanning_example.py --debug --skip-rectification --mask-threshold 5 --gray-code-threshold 10 --epipolar-tol 20.0 --pattern-interval 0.3 --capture-delay 0.2 --sync-mode strict

  # Better synchronization between projector and cameras
  python realtime_scanning_example.py --sync-mode strict --capture-delay 0.1 --pattern-interval 0.3

  # For higher quality at the cost of performance
  python realtime_scanning_example.py --quality high --debug
"""

import os
import sys
import time
import argparse
import logging
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "unlook_scanner.log"))
    ]
)

# Create logger with custom error handler
logger = logging.getLogger("realtime_scanning")

# Make logger more robust to shutdown issues
class SafeHandler(logging.StreamHandler):
    """A handler that's safer during shutdown by catching errors."""
    def handleError(self, record):
        # Just silently ignore errors during shutdown
        pass

    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            # Ignoring all exceptions during logging
            pass

# Replace default handlers with safe ones
for handler in logger.handlers + logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.__class__ = SafeHandler

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

# Import scanner and client
from unlook import UnlookClient
from unlook.client.realtime_scanner import create_realtime_scanner, RealTimeScanConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time 3D Scanning Example')
    
    # File and output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for scan results')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to stereo calibration file')
    
    # Scanning quality and behavior
    parser.add_argument('--quality', type=str, default='medium',
                        choices=['fast', 'medium', 'high'],
                        help='Scan quality preset')
    parser.add_argument('--epipolar-tol', type=float, default=15.0,
                        help='Epipolar tolerance for stereo matching (pixels, default: 15.0)')
    parser.add_argument('--continuous', action='store_true',
                        help='Enable continuous scanning mode')
    parser.add_argument('--min-disparity', type=int, default=5,
                        help='Minimum disparity value (pixels, default: 5)')
    parser.add_argument('--max-disparity', type=int, default=100,
                        help='Maximum disparity value (pixels, default: 100)')
    parser.add_argument('--skip-rectification', action='store_true',
                        help='Skip image rectification (use for initial testing or uncalibrated cameras)')

    # Camera configuration
    parser.add_argument('--exposure', type=int, default=20,
                        help='Camera exposure time in milliseconds (default: 20)')
    parser.add_argument('--gain', type=float, default=1.5,
                        help='Camera gain value (default: 1.5)')

    # Synchronization parameters
    parser.add_argument('--pattern-interval', type=float, default=0.2,
                        help='Time interval between pattern projections (seconds, default: 0.2)')
    parser.add_argument('--capture-delay', type=float, default=0.0,
                        help='Delay before capturing images after projecting pattern (seconds, default: 0.0)')
    parser.add_argument('--sync-mode', type=str, default='normal',
                        choices=['normal', 'strict'],
                        help='Synchronization mode between projector and cameras (default: normal)')

    # Processing options
    parser.add_argument('--no-downsample', action='store_true',
                        help='Disable point cloud downsampling')
    parser.add_argument('--voxel-size', type=float, default=3.0,
                        help='Voxel size for downsampling (mm, default: 3.0)')
    parser.add_argument('--no-noise-filter', action='store_true',
                        help='Disable statistical noise filtering')
    parser.add_argument('--mask-threshold', type=int, default=15,
                        help='Threshold for projector illumination detection (default: 15)')
    parser.add_argument('--gray-code-threshold', type=int, default=20,
                        help='Threshold for Gray code bit decoding (default: 20)')
    parser.add_argument('--no-adaptive-threshold', action='store_true',
                        help='Disable adaptive thresholding for shadow masks')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='Save intermediate processing images for debugging')

    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                        help='Enable Open3D real-time visualization')
    parser.add_argument('--opencv-preview', action='store_true',
                        help='Show OpenCV preview (works without Open3D)')
    
    # Other options
    parser.add_argument('--record', action='store_true',
                        help='Record scan data to files')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging and output')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout in seconds for scanner discovery')
    parser.add_argument('--check-focus', action='store_true',
                        help='Run interactive focus check before scanning')
    parser.add_argument('--focus-roi', type=str, default=None,
                        help='Region of interest for focus check (x,y,width,height)')

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

        # Resource conflict management
        self.context_lock = threading.Lock()
        self.render_attempted = False
        self.max_retries = 5
        self.retry_delay = 0.5  # seconds

        # Visualization parameters
        self.update_interval = 0.03  # seconds - higher refresh rate

        # Check Open3D availability
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for visualization")
            raise ImportError("Open3D is required for visualization")

    def initialize(self):
        """Initialize Open3D visualization."""
        import threading

        # Set Open3D verbosity to error only to suppress warnings
        if hasattr(o3d, 'utility') and hasattr(o3d.utility, 'set_verbosity_level'):
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        # Try multiple times to create the visualization window
        max_tries = 3
        for attempt in range(max_tries):
            try:
                # Create visualization window in main thread
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(window_name="Real-time 3D Scanning", width=1280, height=720, visible=True)
                break
            except Exception as e:
                if attempt < max_tries - 1:
                    logger.warning(f"Failed to create visualization window (attempt {attempt+1}/{max_tries}): {e}")
                    time.sleep(1)  # Wait before retrying
                else:
                    logger.error(f"Failed to create visualization window: {e}")
                    raise

        # Start visualization in a separate thread for real-time updates
        self.viz_thread = threading.Thread(target=self._run_visualization_loop, daemon=True)

        try:
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
        except Exception as e:
            logger.error(f"Error setting up visualization: {e}")
            if self.vis:
                try:
                    self.vis.destroy_window()
                except:
                    pass

    def _run_visualization_loop(self):
        """Run the visualization loop in a separate thread."""
        retry_count = 0

        while self.running:
            try:
                with self.context_lock:
                    if not self.vis.poll_events():
                        # Window was closed
                        self.running = False
                        break

                    self.vis.update_renderer()
                    self.render_attempted = True
                    retry_count = 0  # Reset retry counter after successful render

                time.sleep(0.01)  # Small sleep to prevent CPU usage spike

            except Exception as e:
                retry_count += 1
                if "Failed to make context current" in str(e) and retry_count < self.max_retries:
                    # Resource conflict, wait and try again
                    time.sleep(self.retry_delay)
                    continue

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
            try:
                self.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize visualization: {e}")
                self.running = False
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
                with self.context_lock:
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
                        # Generate colors based on Z value for better visualization
                        points = np.asarray(self.pcd.points)
                        z_values = points[:, 2]
                        z_min, z_max = np.min(z_values), np.max(z_values)
                        if z_max > z_min:
                            normalized_z = (z_values - z_min) / (z_max - z_min)
                            # Create color gradient (blue to red)
                            colors = np.zeros((len(self.pcd.points), 3))
                            colors[:, 0] = normalized_z  # Red
                            colors[:, 2] = 1.0 - normalized_z  # Blue
                            self.pcd.colors = o3d.utility.Vector3dVector(colors)
                        else:
                            # Default to light blue if all points have same Z
                            colors = np.zeros((len(self.pcd.points), 3))
                            colors[:, 2] = 1.0  # Blue
                            self.pcd.colors = o3d.utility.Vector3dVector(colors)

                    try:
                        # Update geometry with proper error handling
                        self.vis.update_geometry(self.pcd)
                    except Exception as e:
                        if "Failed to make context current" in str(e):
                            # Skip this update if context issues
                            logger.warning("Skipping visualization update due to GL context conflict")
                            return
                        else:
                            raise

                    # Update info text
                    info_text = f"FPS: {fps:.1f} | Scan #: {scan_count} | Points: {len(self.pcd.points)}"
                    print(info_text, end="\r", flush=True)

            except Exception as e:
                logger.error(f"Error updating visualization: {e}")
                # Don't crash on visualization errors - allow scanning to continue
                if "Failed to make context current" in str(e):
                    # Resource conflict - common in Windows with OpenGL
                    logger.warning("OpenGL context conflict detected in visualization")
                else:
                    # More serious error
                    logger.error(f"Visualization error details: {type(e).__name__}: {str(e)}")

    def close(self):
        """Close visualization window."""
        self.running = False

        # Wait for visualization thread to terminate
        if self.viz_thread and self.viz_thread.is_alive():
            self.viz_thread.join(timeout=2.0)

        # Make sure window is destroyed
        if self.window_created:
            try:
                with self.context_lock:
                    if hasattr(self, 'vis') and self.vis is not None:
                        try:
                            self.vis.destroy_window()
                            logger.info("Visualization window successfully closed")
                        except Exception as e:
                            logger.warning(f"Non-critical error during window cleanup: {e}")
            except Exception:
                # Ignore errors during cleanup
                pass
            finally:
                self.window_created = False
                self.vis = None  # Clear reference to allow GC


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


def parse_roi(roi_str):
    """
    Parse region of interest string to tuple.

    Args:
        roi_str: Region of interest as string "x,y,width,height"

    Returns:
        Tuple (x, y, width, height) or None if invalid
    """
    if not roi_str:
        return None

    try:
        parts = roi_str.split(',')
        if len(parts) != 4:
            logger.error(f"Invalid ROI format: {roi_str}. Expected 'x,y,width,height'")
            return None

        roi = tuple(int(part) for part in parts)
        return roi
    except ValueError:
        logger.error(f"Invalid ROI values: {roi_str}. Expected integers")
        return None


def main():
    """Run the real-time 3D scanning example."""
    # Display banner
    print("\n" + "="*80)
    print(" REAL-TIME 3D SCANNER (CPU-OPTIMIZED)")
    print(" Reliable scanning for structured light systems")
    print("="*80 + "\n")

    args = parse_arguments()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Set higher logging level for more details on specific modules
        for module in ['unlook.client.realtime_scanner', 'unlook.client.camera', 'unlook.client.projector']:
            logging.getLogger(module).setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Detect Windows and warn about OpenGL issues
    import platform
    if platform.system() == "Windows":
        logger.info("Windows detected - OpenGL visualization may encounter context conflicts")
        logger.info("If visualization fails, try --opencv-preview option instead of --visualize")

    # Configure Open3D to handle failures better
    try:
        import open3d as o3d
        if hasattr(o3d, 'utility') and hasattr(o3d.utility, 'set_verbosity_level'):
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    except ImportError:
        pass
    
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

        # Run focus check if requested
        if args.check_focus:
            logger.info("Running camera focus check before scanning...")
            roi = parse_roi(args.focus_roi)
            if roi:
                logger.info(f"Using ROI for focus check: {roi}")

            try:
                # Get stereo camera pair for focus check
                focus_results, focus_images = client.camera.check_stereo_focus(
                    num_samples=3, roi=roi)

                # Display initial focus results
                for camera_id, (score, quality) in focus_results.items():
                    logger.info(f"Camera {camera_id} initial focus: {score:.2f} ({quality})")

                # Run interactive focus check
                logger.info("Starting interactive focus check. Adjust camera focus until both cameras show GOOD or EXCELLENT.")
                logger.info("Press Ctrl+C to continue when focus is good.")

                client.camera.interactive_stereo_focus_check(
                    interval=0.5,
                    roi=roi
                )

                logger.info("Focus check completed. Continuing with scanning.")
            except Exception as e:
                logger.error(f"Error during focus check: {e}")
                logger.warning("Continuing without focus check.")

    except Exception as e:
        logger.error(f"Failed to initialize client and connect to scanner: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Create real-time scanner with callback
    logger.info(f"Creating real-time scanner with {args.quality} quality preset...")
    try:
        # Create configuration for real-time scanner
        config = RealTimeScanConfig()
        config.set_quality_preset(args.quality)
        
        # Apply custom parameters from command line
        config.pattern_interval = args.pattern_interval
        config.capture_delay = args.capture_delay
        config.camera_exposure = args.exposure
        config.camera_gain = args.gain
        config.epipolar_tolerance = args.epipolar_tol
        config.min_disparity = args.min_disparity
        config.max_disparity = args.max_disparity
        config.continuous_scanning = args.continuous
        config.debug = args.debug
        
        # Configure downsampling
        if args.no_downsample:
            config.downsample = False
        else:
            config.downsample = True
            config.downsample_voxel_size = args.voxel_size
        
        # Configure noise filtering
        if args.no_noise_filter:
            config.noise_filter = False
        else:
            config.noise_filter = True

        # Configure advanced options
        config.mask_threshold = args.mask_threshold
        config.gray_code_threshold = args.gray_code_threshold
        config.use_adaptive_thresholding = not args.no_adaptive_threshold
        config.save_intermediate_images = args.save_intermediate
        config.verbose_logging = args.debug

        # Always enable continuous scanning mode regardless of command-line flag
        config.continuous_scanning = True
        logger.info("Enabling continuous scanning mode for persistent retries")

        # Skip rectification if requested (helpful for initial testing)
        if args.skip_rectification:
            logger.info("SKIPPING IMAGE RECTIFICATION - using original images directly")
            config.skip_rectification = True
        
        # Log configuration settings
        logger.info(f"Using quality preset: {config.quality}")
        logger.info(f"Pattern interval: {config.pattern_interval}s")
        logger.info(f"Capture delay: {config.capture_delay}s")
        logger.info(f"Camera settings: Exposure={config.camera_exposure}ms, Gain={config.camera_gain}")
        logger.info(f"Epipolar tolerance: {config.epipolar_tolerance} pixels")
        logger.info(f"Disparity range: {config.min_disparity}-{config.max_disparity} pixels")
        
        if config.downsample:
            logger.info(f"Downsampling enabled (voxel size: {config.downsample_voxel_size})")
        else:
            logger.info("Downsampling disabled")
        
        if config.noise_filter:
            logger.info("Noise filtering enabled")
        else:
            logger.info("Noise filtering disabled")
        
        # Create real-time scanner with callback for new frames
        scanner = create_realtime_scanner(
            client=client,
            config=config,
            calibration_file=args.calibration,
            on_new_frame=on_new_frame_callback
        )
        
        # Set synchronization mode
        scanner.pattern_sync_mode = args.sync_mode
        logger.info(f"Using {args.sync_mode} synchronization mode")
        
        # Store the capture delay for the scanner to use
        client.capture_delay = args.capture_delay
        
    except Exception as e:
        logger.error(f"Failed to create scanner: {e}")
        client.disconnect()
        return 1
    
    # Capture signals for clean shutdown
    import signal
    running = True

    def signal_handler(sig, frame):
        """Handle shutdown signals safely."""
        nonlocal running
        # Use print instead of logger to avoid reentrant logging issues
        # that can occur during shutdown
        print("\nShutdown signal received, cleaning up...")
        running = False

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start recording if enabled
    if recorder:
        recorder.start_recording()
    
    # Print helpful information and scanning tips
    print("\n" + "="*80)
    print("  SCANNING TIPS AND TROUBLESHOOTING")
    print("  - Keep the scanner stable during scanning")
    print("  - Ensure good lighting but avoid direct bright light on the object")
    print("  - Ideal scanning distance: 30-60cm from the cameras")
    print("  - Recommended command for initial testing:")
    print("    python realtime_scanning_example.py --debug --skip-rectification --mask-threshold 5 --gray-code-threshold 10 --epipolar-tol 20.0 --pattern-interval 0.3 --capture-delay 0.2 --sync-mode strict")
    print("\n  - If no points appear, try:")
    print("    1. Always use --debug to see diagnostic information and images")
    print("    2. Use --skip-rectification for initial testing with uncalibrated cameras")
    print("    3. Lower the mask threshold (--mask-threshold 5 or even 2)")
    print("    4. Lower the Gray code threshold (--gray-code-threshold 10 or even 5)")
    print("    5. Increase the pattern interval (--pattern-interval 0.3 or 0.4)")
    print("    6. Increase the capture delay (--capture-delay 0.2 or 0.3)")
    print("    7. Always use strict synchronization mode (--sync-mode strict)")
    print("    8. Increase the epipolar tolerance (--epipolar-tol 20.0 or 30.0)")
    print("    9. Adjust camera exposure settings (--exposure 50 --gain 2.0)")
    print("    10. Enable save-intermediate to debug (--save-intermediate)")
    print("\n  - Debug output is saved to: ./unlook_debug/scan_TIMESTAMP/")
    print("  - Check mask coverage and difference images in the debug output")
    print("  - For best results with calibrated cameras, remove --skip-rectification after initial testing")
    print("  - Press Ctrl+C to stop scanning")
    print("="*80 + "\n")

    # Start scanner
    logger.info("Starting real-time scanning...")
    scanner.start(debug_mode=args.debug)

    # Run diagnostics if in debug mode
    if args.debug:
        logger.info("Running scanner diagnostics...")
        diagnostics = scanner.run_diagnostics()

        # Print key diagnostics in a more user-friendly format
        logger.info("=== Scanner Diagnostics ===")

        # Camera info
        if "cameras" in diagnostics:
            logger.info(f"Cameras detected: {diagnostics.get('cameras', {}).get('count', 0)}")
            logger.info(f"Camera IDs: {diagnostics.get('cameras', {}).get('ids', [])}")

        # Focus information
        if "focus_status" in diagnostics:
            logger.info(f"Camera focus status: {diagnostics.get('focus_status', 'UNKNOWN')}")

        # Calibration information
        if "calibration" in diagnostics:
            calib_info = diagnostics.get("calibration", {})
            logger.info(f"Calibration loaded: {calib_info.get('loaded', False)}")
            if "baseline" in calib_info:
                logger.info(f"Stereo baseline: {calib_info.get('baseline', 0):.2f}mm")

        # Projector info
        if "projector" in diagnostics:
            proj_info = diagnostics.get("projector", {})
            logger.info(f"Projector test: {proj_info.get('test_result', 'UNKNOWN')}")

        logger.info("===========================")
    
    try:
        # Main application loop
        while running:
            # Get current point cloud and stats
            point_cloud = scanner.get_current_point_cloud()
            fps = scanner.get_fps()
            scan_count = scanner.get_scan_count()

            # Update loop status
            if scan_count % 10 == 0:
                points_count = 0
                if point_cloud is not None:
                    if hasattr(point_cloud, 'points'):
                        points_count = len(point_cloud.points)
                    else:
                        points_count = len(point_cloud) if isinstance(point_cloud, np.ndarray) else 0

                    # Color-coded status with performance indicators
                    if fps < 1.0:
                        fps_status = "SLOW"
                    elif fps < 3.0:
                        fps_status = "OK"
                    else:
                        fps_status = "GOOD"

                    points_status = "SPARSE" if points_count < 1000 else "GOOD"

                    logger.info(f"Scan #{scan_count} | FPS: {fps:.1f} ({fps_status}) | Points: {points_count} ({points_status})")
                else:
                    logger.info(f"Scan #{scan_count} | FPS: {fps:.1f} | No points yet (waiting for first scan)")

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

            # Show OpenCV preview if enabled
            if args.opencv_preview:
                try:
                    if point_cloud is None:
                        # Create a blank image for preview
                        blank_img = np.zeros((600, 800, 3), dtype=np.uint8)
                        cv2.putText(blank_img, "Waiting for point cloud...", (50, 300),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                        cv2.imshow("Real-time OpenCV Preview", blank_img)
                    else:
                        # Create a simple colored visualization of the point cloud
                        img_size = (800, 600)
                        preview = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

                        if hasattr(point_cloud, 'points') and len(point_cloud.points) > 0:
                            # Convert point cloud to 2D visualization
                            points = np.asarray(point_cloud.points)

                            # Simple projection - adjust these values based on your data scale
                            scale = 100
                            center_x, center_y = img_size[0]//2, img_size[1]//2

                            for p in points[:1000]:  # Limit to 1000 points for speed
                                x, y, z = p
                                screen_x = int(center_x + x * scale)
                                screen_y = int(center_y - y * scale)  # Y inverted for screen coords

                                # Only draw if within image bounds
                                if 0 <= screen_x < img_size[0] and 0 <= screen_y < img_size[1]:
                                    # Color based on depth (z)
                                    color = [
                                        int(255 * (z - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]))),
                                        100,
                                        255 - int(255 * (z - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2])))
                                    ]
                                    cv2.circle(preview, (screen_x, screen_y), 2, color, -1)

                        # Add status text
                        cv2.putText(preview, f"Points: {len(point_cloud.points) if hasattr(point_cloud, 'points') else 0}",
                                  (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        cv2.putText(preview, f"FPS: {fps:.1f}",
                                  (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                        cv2.imshow("Real-time OpenCV Preview", preview)
                except Exception as e:
                    # OpenCV GUI functionality not available
                    if args.opencv_preview:
                        logger.warning(f"OpenCV GUI functionality not available: {e}")
                        logger.warning("Disabling OpenCV preview - your OpenCV installation lacks GUI support")
                        args.opencv_preview = False

            # Record frame if enabled
            if recorder and recorder.recording and point_cloud is not None:
                recorder.record_frame(point_cloud, scan_count, fps)

            # Process keyboard inputs for OpenCV window (ESC to quit)
            if args.opencv_preview:
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        logger.info("ESC key pressed, stopping...")
                        running = False
                except Exception as e:
                    # Skip OpenCV GUI functions if not supported
                    logger.debug(f"OpenCV GUI functionality not available: {e}")
                    args.opencv_preview = False  # Disable for future iterations

            # Brief pause to prevent CPU usage spike
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Use try/except blocks for each cleanup operation to ensure
        # one failure doesn't prevent other cleanup steps

        # Stop scanning
        try:
            print("Stopping real-time scanner...")
            if scanner:
                scanner.stop()
        except Exception as e:
            print(f"Error stopping scanner: {e}")

        # Stop recording
        try:
            if recorder and hasattr(recorder, 'recording') and recorder.recording:
                recorder.stop_recording()
        except Exception as e:
            print(f"Error stopping recording: {e}")

        # Close visualization
        try:
            if visualizer:
                visualizer.close()
        except Exception as e:
            print(f"Error closing visualizer: {e}")

        # Disconnect from scanner
        try:
            if client:
                client.disconnect()
                print("Disconnected from scanner")
        except Exception as e:
            print(f"Error disconnecting from scanner: {e}")

        # Close any open CV windows
        if args.opencv_preview:
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    logger.info("Real-time scanning completed")
    return 0


def enable_headless_display():
    """Configure environment to run in headless mode if necessary."""
    try:
        import os

        # Check for display server (X11/Wayland on Linux, etc.)
        display = os.environ.get('DISPLAY')
        wayland_display = os.environ.get('WAYLAND_DISPLAY')

        if not display and not wayland_display:
            # No display server available
            logger.warning("No display server detected - trying EGL rendering backend")
            # Try to set EGL as the rendering backend for Open3D
            os.environ['OPEN3D_CPU_RENDERING'] = '1'
            os.environ['OPEN3D_HEADLESS_RENDERING'] = '1'
            return True
    except Exception as e:
        logger.warning(f"Error checking display configuration: {e}")

    return False


if __name__ == "__main__":
    try:
        # Setup headless mode if needed
        headless = enable_headless_display()
        if headless:
            logger.info("Configured for headless operation - visualization may be limited")

        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # In case of GL context errors, add helpful message
        if "Failed to make context current" in str(e):
            logger.error("\nGRAPHICS CONTEXT ERROR DETECTED: This is likely due to OpenGL conflicts.")
            logger.error("Possible solutions:")
            logger.error("1. Run with --opencv-preview instead of --visualize")
            logger.error("2. Close other applications using OpenGL/3D graphics")
            logger.error("3. Use a different visualization method or run without visualization")

        sys.exit(1)