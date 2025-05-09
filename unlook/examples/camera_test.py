#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Test and Focus Tool for Unlook SDK

This script provides tools to:
1. Test camera connections
2. Check camera focus quality
3. Interactively adjust camera focus with visual feedback
4. Test camera calibration
5. Test projector patterns
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("camera_test")

# Add parent directory to path to allow importing unlook module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import numpy as np
    import cv2
except ImportError as e:
    logger.error(f"Required dependency missing: {e}")
    logger.error("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Import unlook client
from unlook import UnlookClient


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Camera Testing and Focus Tool')
    
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout in seconds for scanner discovery')
    parser.add_argument('--output', type=str, default='camera_test_output',
                        help='Output directory for test results')
    parser.add_argument('--roi', type=str, default=None,
                        help='Region of interest for focus checks (x,y,width,height)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive focus adjustment')
    parser.add_argument('--stereo', action='store_true',
                        help='Test stereo cameras together')
    parser.add_argument('--test-projector', action='store_true',
                        help='Test projector patterns')
    parser.add_argument('--find-cuda', action='store_true',
                        help='Attempt to find CUDA installation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    return parser.parse_args()


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


def test_and_display_image(img, title="Camera Test"):
    """
    Display image and save to file.
    
    Args:
        img: Image to display
        title: Window title
        
    Returns:
        True if image is valid, False otherwise
    """
    if img is None:
        logger.error("No image received")
        return False
    
    # Display image
    cv2.imshow(title, img)
    cv2.waitKey(1)  # Update display
    
    return True


def find_cuda_paths():
    """
    Search for CUDA installation locations.
    
    Returns:
        List of potential CUDA paths
    """
    logger.info("Searching for CUDA installations...")
    
    potential_paths = []
    
    # Common CUDA installation paths on Windows
    windows_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
        r"C:\ProgramData\NVIDIA Corporation\CUDA"
    ]
    
    # Common CUDA installation paths on Linux
    linux_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda"
    ]
    
    # Check Windows paths
    for base_path in windows_paths:
        if os.path.exists(base_path):
            # Look for version directories
            try:
                for item in os.listdir(base_path):
                    version_path = os.path.join(base_path, item)
                    if os.path.isdir(version_path) and item.startswith("v"):
                        potential_paths.append(version_path)
            except Exception as e:
                logger.warning(f"Error accessing {base_path}: {e}")
    
    # Check Linux paths
    for base_path in linux_paths:
        if os.path.exists(base_path):
            potential_paths.append(base_path)
            # Check versioned directories
            try:
                for item in os.listdir(base_path):
                    version_path = os.path.join(base_path, item)
                    if os.path.isdir(version_path) and item.startswith("v"):
                        potential_paths.append(version_path)
            except Exception as e:
                logger.warning(f"Error accessing {base_path}: {e}")
    
    # Check environment variable
    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_path_env:
        if os.path.exists(cuda_path_env):
            potential_paths.append(cuda_path_env)
            logger.info(f"Found CUDA_PATH environment variable: {cuda_path_env}")
        else:
            logger.warning(f"CUDA_PATH environment variable set but path doesn't exist: {cuda_path_env}")
    
    # Verify paths with cuda libraries
    valid_paths = []
    for path in potential_paths:
        if os.path.exists(os.path.join(path, "bin")):
            cuda_libs = False
            for root, dirs, files in os.walk(os.path.join(path, "bin")):
                for file in files:
                    if file.startswith("cudart") or file.startswith("cublas"):
                        cuda_libs = True
                        break
                if cuda_libs:
                    break
            
            if cuda_libs:
                valid_paths.append(path)
                logger.info(f"Valid CUDA installation found at: {path}")
            else:
                logger.debug(f"Path exists but missing CUDA libraries: {path}")
    
    if not valid_paths:
        logger.warning("No valid CUDA installations found.")
        
        # Check for NVIDIA drivers only
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                result = subprocess.check_output("where nvidia-smi", shell=True, stderr=subprocess.STDOUT)
                logger.info("NVIDIA drivers found (nvidia-smi available)")
            else:  # Linux
                result = subprocess.check_output("which nvidia-smi", shell=True, stderr=subprocess.STDOUT)
                logger.info("NVIDIA drivers found (nvidia-smi available)")
        except:
            logger.warning("NVIDIA drivers not detected")
    
    # Instructions for setting CUDA_PATH
    if valid_paths:
        logger.info("\nTo set CUDA_PATH environment variable:")
        path = valid_paths[0]  # Use first valid path
        if os.name == 'nt':  # Windows
            logger.info(f"  1. Run in PowerShell: $env:CUDA_PATH=\"{path}\"")
            logger.info(f"  2. Set permanently in System Properties > Environment Variables")
        else:  # Linux
            logger.info(f"  1. Run: export CUDA_PATH=\"{path}\"")
            logger.info(f"  2. Add to ~/.bashrc: export CUDA_PATH=\"{path}\"")
    
    return valid_paths


def test_projector_patterns(client):
    """
    Test projector patterns.
    
    Args:
        client: UnlookClient instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Testing projector patterns...")
    
    # Check available cameras
    cameras = client.camera.get_cameras()
    if not cameras:
        logger.error("No cameras available to view patterns")
        return False
    
    # Select first camera
    camera_id = cameras[0]['id']
    logger.info(f"Using camera {camera_id} to view patterns")
    
    try:
        # Show different patterns
        patterns = [
            {"name": "White Field", "fn": lambda: client.projector.show_solid_field("White")},
            {"name": "Black Field", "fn": lambda: client.projector.show_solid_field("Black")},
            {"name": "Red Field", "fn": lambda: client.projector.show_solid_field("Red")},
            {"name": "Green Field", "fn": lambda: client.projector.show_solid_field("Green")},
            {"name": "Blue Field", "fn": lambda: client.projector.show_solid_field("Blue")},
            {"name": "Horizontal Lines (thin)", "fn": lambda: client.projector.show_horizontal_lines(foreground_width=2, background_width=8)},
            {"name": "Horizontal Lines (medium)", "fn": lambda: client.projector.show_horizontal_lines(foreground_width=4, background_width=16)},
            {"name": "Horizontal Lines (thick)", "fn": lambda: client.projector.show_horizontal_lines(foreground_width=8, background_width=32)},
            {"name": "Vertical Lines (thin)", "fn": lambda: client.projector.show_vertical_lines(foreground_width=2, background_width=8)},
            {"name": "Vertical Lines (medium)", "fn": lambda: client.projector.show_vertical_lines(foreground_width=4, background_width=16)},
            {"name": "Vertical Lines (thick)", "fn": lambda: client.projector.show_vertical_lines(foreground_width=8, background_width=32)},
        ]
        
        logger.info("Starting pattern test. Press ESC to quit, any other key to advance to next pattern.")
        logger.info("Pattern 1/11: White Field")
        
        # Show first pattern by default
        patterns[0]["fn"]()
        
        pattern_index = 0
        while True:
            # Capture image to show pattern
            img = client.camera.capture(camera_id)
            
            if img is not None:
                cv2.putText(img, f"{pattern_index+1}/11: {patterns[pattern_index]['name']}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Projector Pattern Test", img)
                
            # Wait for key press
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                break
                
            # Next pattern
            pattern_index = (pattern_index + 1) % len(patterns)
            logger.info(f"Pattern {pattern_index+1}/{len(patterns)}: {patterns[pattern_index]['name']}")
            patterns[pattern_index]["fn"]()
            
        # Reset to black at end
        client.projector.show_solid_field("Black")
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing projector patterns: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run the camera test and focus tool."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Test results will be saved to: {args.output}")
    
    # Parse ROI
    roi = parse_roi(args.roi)
    if roi:
        logger.info(f"Using ROI: {roi}")
    
    # Find CUDA if requested
    if args.find_cuda:
        cuda_paths = find_cuda_paths()
        if cuda_paths:
            logger.info(f"Found {len(cuda_paths)} potential CUDA installations.")
        return 0
    
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
        
        # Test projector patterns if requested
        if args.test_projector:
            test_projector_patterns(client)
            client.disconnect()
            return 0
        
        # Get available cameras
        cameras = client.camera.get_cameras()
        if not cameras:
            logger.error("No cameras available")
            client.disconnect()
            return 1
            
        logger.info(f"Found {len(cameras)} cameras:")
        for i, camera in enumerate(cameras):
            logger.info(f"  {i+1}. {camera['id']} - {camera.get('name', 'Unnamed')}")
        
        # Run interactive focus check
        if args.interactive:
            if args.stereo and len(cameras) >= 2:
                logger.info("Running interactive stereo focus check...")
                logger.info("Adjust camera focus until both cameras show GOOD or EXCELLENT focus.")
                logger.info("Press Ctrl+C to continue when focus is good.")
                
                client.camera.interactive_stereo_focus_check(
                    interval=0.5,
                    roi=roi
                )
            else:
                # If not stereo or only one camera available
                camera_id = cameras[0]["id"]
                logger.info(f"Running interactive focus check for camera {camera_id}...")
                logger.info("Adjust camera focus until it shows GOOD or EXCELLENT focus.")
                logger.info("Press Ctrl+C to continue when focus is good.")
                
                client.camera.interactive_focus_check(
                    camera_id=camera_id,
                    interval=0.5,
                    roi=roi
                )
        else:
            # Just display focus score without interactive adjustment
            if args.stereo and len(cameras) >= 2:
                logger.info("Checking stereo camera focus...")
                focus_results, focus_images = client.camera.check_stereo_focus(
                    num_samples=3, roi=roi)
                
                # Display focus results
                for camera_id, (score, quality) in focus_results.items():
                    logger.info(f"Camera {camera_id}: Focus score {score:.2f} - {quality}")
                    
                    # Save focus image
                    if args.output and camera_id in focus_images:
                        img = focus_images[camera_id]
                        if img is not None:
                            # Annotate image
                            cv2.putText(img, f"Focus: {score:.2f} - {quality}", 
                                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Save image
                            output_path = os.path.join(args.output, f"focus_{camera_id}.png")
                            cv2.imwrite(output_path, img)
                            logger.info(f"Saved focus image to {output_path}")
                
                # Display images
                for camera_id, img in focus_images.items():
                    if img is not None:
                        # Draw focus info
                        score, quality = focus_results[camera_id]
                        annotated = img.copy()
                        cv2.putText(annotated, f"Focus: {score:.2f} - {quality}", 
                                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Show image
                        cv2.imshow(f"Camera {camera_id} Focus", annotated)
                
                # Wait for key press
                logger.info("Press any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                # Single camera mode
                camera_id = cameras[0]["id"]
                logger.info(f"Checking focus for camera {camera_id}...")
                
                score, quality, img = client.camera.check_focus(
                    camera_id=camera_id,
                    num_samples=3,
                    roi=roi
                )
                
                logger.info(f"Focus score: {score:.2f} - {quality}")
                
                # Save and show focus image
                if img is not None:
                    # Annotate image
                    cv2.putText(img, f"Focus: {score:.2f} - {quality}", 
                               (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Save image
                    if args.output:
                        output_path = os.path.join(args.output, f"focus_{camera_id}.png")
                        cv2.imwrite(output_path, img)
                        logger.info(f"Saved focus image to {output_path}")
                    
                    # Show image
                    cv2.imshow(f"Camera {camera_id} Focus", img)
                    logger.info("Press any key to exit...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during camera test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Cleanup
        try:
            client.disconnect()
            logger.info("Disconnected from scanner")
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    logger.info("Camera test completed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)