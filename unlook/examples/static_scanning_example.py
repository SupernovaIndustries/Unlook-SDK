#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Static 3D Scanning Example

This example demonstrates how to use the static 3D scanner for high-quality
single scans. Unlike real-time scanning, this approach captures all patterns
at once before processing them, optimizing for quality rather than speed.

The scanner supports GPU acceleration and neural network processing for
improved speed and quality. GPU acceleration can greatly speed up triangulation
and correspondence matching, while neural networks can improve point cloud
quality through denoising and hole filling.

Usage examples:
  # Basic scanning with default settings (high quality)
  python static_scanning_example.py

  # Ultra-quality scanning with calibration (includes neural network processing)
  python static_scanning_example.py --quality ultra --calibration stereo_calibration.json

  # Specify output file
  python static_scanning_example.py --output my_scan.ply --debug

  # Control GPU and neural network options
  python static_scanning_example.py --use-gpu --use-neural-network --neural-model-path models/point_net.pth

  # Choose specific ML backend (PyTorch or TensorFlow)
  python static_scanning_example.py --quality ultra --ml-backend pytorch

  # Full control over scanning parameters
  python static_scanning_example.py --quality high --mask-threshold 5 --gray-code-threshold 10 --pattern-interval 0.4
"""

import os
import sys
import time
import argparse
import logging
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

# Create logger
logger = logging.getLogger("static_scanning")

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
    logger.warning("open3d not installed. Visualization will be limited.")
    logger.warning("Install open3d for better results: pip install open3d")
    OPEN3D_AVAILABLE = False

# Import scanner and client
from unlook import UnlookClient
from unlook.client.static_scanner import create_static_scanner, StaticScanConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Static 3D Scanning Example')

    # File and output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for point cloud (e.g., scan.ply)')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to stereo calibration file')

    # Scanning quality and behavior
    parser.add_argument('--quality', type=str, default='high',
                        choices=['medium', 'high', 'ultra'],
                        help='Scan quality preset')

    # Camera configuration
    parser.add_argument('--exposure', type=int, default=20,
                        help='Camera exposure time in milliseconds (default: 20)')
    parser.add_argument('--gain', type=float, default=1.5,
                        help='Camera gain value (default: 1.5)')

    # Pattern parameters
    parser.add_argument('--num-gray-codes', type=int, default=None,
                        help='Number of Gray code bits to use')
    parser.add_argument('--no-phase-shift', action='store_true',
                        help='Disable phase shift patterns')
    parser.add_argument('--pattern-interval', type=float, default=None,
                        help='Time interval between pattern projections (seconds)')
    parser.add_argument('--capture-delay', type=float, default=None,
                        help='Delay before capturing after projecting pattern (seconds)')

    # Processing options
    parser.add_argument('--no-downsample', action='store_true',
                        help='Disable point cloud downsampling')
    parser.add_argument('--voxel-size', type=float, default=None,
                        help='Voxel size for downsampling (mm)')
    parser.add_argument('--no-noise-filter', action='store_true',
                        help='Disable statistical noise filtering')
    parser.add_argument('--mask-threshold', type=int, default=None,
                        help='Threshold for projector illumination detection')
    parser.add_argument('--gray-code-threshold', type=int, default=None,
                        help='Threshold for Gray code bit decoding')
    parser.add_argument('--no-adaptive-threshold', action='store_true',
                        help='Disable adaptive thresholding for shadow masks')
    parser.add_argument('--skip-rectification', action='store_true',
                        help='Skip image rectification (use for initial testing or uncalibrated cameras)')

    # GPU and neural network options
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU acceleration (if available)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--use-neural-network', action='store_true',
                        help='Enable neural network processing (if available)')
    parser.add_argument('--no-neural-network', action='store_true',
                        help='Disable neural network processing')
    parser.add_argument('--neural-model-path', type=str, default=None,
                        help='Path to neural network model weights')
    parser.add_argument('--ml-backend', type=str, choices=['pytorch', 'tensorflow'], default=None,
                        help='Preferred ML backend to use (pytorch or tensorflow)')

    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the resulting point cloud with Open3D')

    # Other options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging and output')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout in seconds for scanner discovery')

    return parser.parse_args()


def visualize_point_cloud(point_cloud):
    """
    Visualize a point cloud using Open3D.
    
    Args:
        point_cloud: Open3D point cloud
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Cannot visualize: Open3D is not available")
        return
    
    try:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window("Static 3D Scan Result")
        
        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        vis.add_geometry(coord_frame)
        
        # Add point cloud
        vis.add_geometry(point_cloud)
        
        # Set view options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Set initial viewpoint
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        
        # Run visualization
        print("Visualizing point cloud. Close the window to continue.")
        vis.run()
        vis.destroy_window()
    
    except Exception as e:
        logger.error(f"Error visualizing point cloud: {e}")


def main():
    """Run the static 3D scanning example."""
    # Display banner
    print("\n" + "="*80)
    print(" STATIC 3D SCANNER (HIGH-QUALITY)")
    print(" Optimized for high-quality single scans")
    print("="*80 + "\n")

    args = parse_arguments()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

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
    
    # Create static scanner configuration
    logger.info(f"Creating static scanner with {args.quality} quality preset...")
    config = StaticScanConfig()
    config.set_quality_preset(args.quality)
    config.debug = args.debug
    
    # Apply custom parameters from command line
    if args.num_gray_codes is not None:
        config.num_gray_codes = args.num_gray_codes
    
    if args.no_phase_shift:
        config.use_phase_shift = False
    
    if args.pattern_interval is not None:
        config.pattern_interval = args.pattern_interval
    
    if args.capture_delay is not None:
        config.capture_delay = args.capture_delay
    
    if args.exposure is not None:
        config.camera_exposure = args.exposure
    
    if args.gain is not None:
        config.camera_gain = args.gain
    
    # Configure downsampling
    if args.no_downsample:
        config.downsample = False
    elif args.voxel_size is not None:
        config.downsample = True
        config.downsample_voxel_size = args.voxel_size
    
    # Configure noise filtering
    if args.no_noise_filter:
        config.noise_filter = False
    
    # Configure thresholding
    if args.mask_threshold is not None:
        config.mask_threshold = args.mask_threshold
    
    if args.gray_code_threshold is not None:
        config.gray_code_threshold = args.gray_code_threshold
    
    if args.no_adaptive_threshold:
        config.use_adaptive_thresholding = False
    
    # Configure rectification
    if args.skip_rectification:
        config.skip_rectification = True
    
    # Configure GPU acceleration
    if args.use_gpu:
        config.use_gpu = True
    elif args.no_gpu:
        config.use_gpu = False

    # Configure neural network processing
    if args.use_neural_network:
        config.use_neural_network = True
    elif args.no_neural_network:
        config.use_neural_network = False

    # Set neural network model path if provided
    if args.neural_model_path:
        # Check if the file exists
        if os.path.exists(args.neural_model_path):
            config.nn_model_path = args.neural_model_path
            # Enable neural network if model path is provided
            config.use_neural_network = True
        else:
            logger.warning(f"Neural network model file not found: {args.neural_model_path}")
            logger.warning("Neural network processing will use default filters")

    # Set ML backend preference if specified
    if args.ml_backend:
        config.ml_backend = args.ml_backend
        logger.info(f"Using preferred ML backend: {args.ml_backend}")

    # Log configuration settings
    logger.info(f"Using quality preset: {config.quality}")
    logger.info(f"Gray code bits: {config.num_gray_codes}")
    logger.info(f"Phase shift: {'Enabled' if config.use_phase_shift else 'Disabled'}")
    logger.info(f"Pattern interval: {config.pattern_interval}s")
    logger.info(f"Capture delay: {config.capture_delay}s")
    logger.info(f"Camera settings: Exposure={config.camera_exposure}ms, Gain={config.camera_gain}")

    if config.downsample:
        logger.info(f"Downsampling enabled (voxel size: {config.downsample_voxel_size}mm)")
    else:
        logger.info("Downsampling disabled")

    if config.noise_filter:
        logger.info("Noise filtering enabled")
    else:
        logger.info("Noise filtering disabled")

    # Log GPU and neural network settings
    logger.info(f"GPU acceleration: {'Enabled' if config.use_gpu else 'Disabled'}")
    logger.info(f"Neural network processing: {'Enabled' if config.use_neural_network else 'Disabled'}")
    if config.use_neural_network and config.nn_model_path:
        logger.info(f"Neural network model: {config.nn_model_path}")
    if config.use_neural_network and config.ml_backend:
        logger.info(f"Preferred ML backend: {config.ml_backend}")
    
    # Create scanner
    scanner = create_static_scanner(
        client=client,
        config=config,
        calibration_file=args.calibration
    )
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Create default output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"scan_{timestamp}_{args.quality}.ply"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    try:
        # Print scanning instructions
        print("\nPreparing to scan. Please ensure:")
        print("1. The object is properly positioned (30-60cm from cameras)")
        print("2. The environment has suitable lighting (avoid direct strong light)")
        print("3. The projector is aimed at the object")
        print("4. The cameras are properly focused")
        print("\nScanning will start in 3 seconds...\n")
        time.sleep(3)
        
        # Perform static scan
        logger.info("Starting static scan...")
        start_time = time.time()
        
        point_cloud = scanner.perform_scan()
        
        scan_time = time.time() - start_time
        
        if point_cloud is None:
            logger.error("Scan failed to produce a point cloud")
            return 1
        
        # Get scan statistics
        stats = scanner.get_processing_stats()
        
        # Print scan results
        print("\n" + "="*80)
        print(f" SCAN COMPLETE: {len(point_cloud.points)} points captured")
        print(f" Total scan time: {scan_time:.2f} seconds")
        print(f" Processing time: {stats['processing_time']:.2f} seconds")

        # Print GPU and neural network information if used
        if config.use_gpu:
            print(f" GPU acceleration: ENABLED")
        if config.use_neural_network:
            print(f" Neural network processing: ENABLED")
            if config.ml_backend:
                print(f" ML backend: {config.ml_backend.upper()}")

        print("="*80 + "\n")
        
        # Save point cloud
        if scanner.save_point_cloud(output_file):
            print(f"Point cloud saved to: {os.path.abspath(output_file)}")
        else:
            logger.error("Failed to save point cloud")
        
        # Visualize if requested
        if args.visualize and OPEN3D_AVAILABLE:
            visualize_point_cloud(point_cloud)
        
        logger.info("Static scanning completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Disconnect from scanner
        try:
            if client:
                client.disconnect()
                logger.info("Disconnected from scanner")
        except Exception as e:
            logger.warning(f"Error disconnecting from scanner: {e}")


if __name__ == "__main__":
    sys.exit(main())