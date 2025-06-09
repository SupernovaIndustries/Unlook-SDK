#!/usr/bin/env python3
"""
Projector-Camera Calibration Script

Calibrates a projector-camera system for structured light 3D scanning.
The projector is treated as an "inverse camera" to enable precise
projector-camera triangulation.

Usage:
    # Interactive calibration
    python calibrate_projector_camera.py --interactive
    
    # Process existing calibration images
    python calibrate_projector_camera.py --input calibration_images/ --output projector_calibration.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanning.calibration import ProjectorCalibrator
from unlook.client.scanner.scanner import UnlookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calibrate projector-camera system for structured light scanning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=Path,
                       help='Directory with existing calibration images')
    
    parser.add_argument('--output', type=Path, default=Path('projector_camera_calibration.json'),
                       help='Output calibration file')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive calibration mode with live capture')
    
    parser.add_argument('--checkerboard-size', type=str, default='9x6',
                       help='Checkerboard size as WIDTHxHEIGHT (internal corners)')
    
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Checkerboard square size in mm')
    
    parser.add_argument('--projector-width', type=int, default=1920,
                       help='Projector resolution width')
    
    parser.add_argument('--projector-height', type=int, default=1080,
                       help='Projector resolution height')
    
    parser.add_argument('--gray-bits', type=int, default=7,
                       help='Number of Gray code bits for calibration')
    
    parser.add_argument('--num-positions', type=int, default=10,
                       help='Number of checkerboard positions for interactive mode')
    
    parser.add_argument('--scanner', type=str,
                       help='Specific scanner name to connect to')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--save-images', action='store_true',
                       help='Save captured calibration images')
    
    return parser.parse_args()


def load_calibration_images(input_dir: Path):
    """Load existing calibration images from directory."""
    camera_images = []
    pattern_images = []
    
    # Look for camera images
    camera_dir = input_dir / "camera"
    if camera_dir.exists():
        for img_file in sorted(camera_dir.glob("*.png")):
            img = cv2.imread(str(img_file))
            if img is not None:
                camera_images.append(img)
                logger.info(f"Loaded camera image: {img_file.name}")
    
    # Look for pattern images (if available)
    pattern_dir = input_dir / "patterns"
    if pattern_dir.exists():
        for img_file in sorted(pattern_dir.glob("*.png")):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pattern_images.append(img)
                logger.info(f"Loaded pattern image: {img_file.name}")
    
    logger.info(f"Loaded {len(camera_images)} camera images, {len(pattern_images)} pattern images")
    return camera_images, pattern_images


def generate_gray_code_patterns(width: int, height: int, num_bits: int = 7):
    """Generate Gray code calibration patterns."""
    patterns = []
    
    # White and black reference patterns
    white_pattern = np.full((height, width), 255, dtype=np.uint8)
    black_pattern = np.zeros((height, width), dtype=np.uint8)
    patterns.extend([white_pattern, black_pattern])
    
    # Gray code patterns
    for bit in range(num_bits):
        # Normal pattern
        pattern = np.zeros((height, width), dtype=np.uint8)
        stripe_width = width // (2 ** (bit + 1))
        
        for x in range(width):
            if (x // stripe_width) % 2 == 0:
                pattern[:, x] = 255
        
        patterns.append(pattern)
        
        # Inverse pattern
        patterns.append(255 - pattern)
    
    logger.info(f"Generated {len(patterns)} Gray code patterns")
    return patterns


def interactive_calibration(args):
    """Perform interactive calibration with live scanner."""
    logger.info("Starting interactive projector-camera calibration")
    
    # Parse checkerboard size
    try:
        cb_width, cb_height = map(int, args.checkerboard_size.split('x'))
        checkerboard_size = (cb_width, cb_height)
    except ValueError:
        logger.error(f"Invalid checkerboard size: {args.checkerboard_size}")
        return 1
    
    # Initialize calibrator
    calibrator = ProjectorCalibrator(
        projector_width=args.projector_width,
        projector_height=args.projector_height,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size
    )
    
    # Connect to scanner
    logger.info("Connecting to scanner...")
    client = UnlookClient(auto_discover=True)
    time.sleep(2)  # Wait for discovery
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found")
        return 1
    
    # Select scanner
    if args.scanner:
        scanner = next((s for s in scanners if s.name == args.scanner), None)
        if not scanner:
            logger.error(f"Scanner '{args.scanner}' not found")
            return 1
    else:
        scanner = scanners[0]
    
    if not client.connect_to_scanner(scanner):
        logger.error("Failed to connect to scanner")
        return 1
    
    logger.info(f"Connected to scanner: {scanner.name}")
    
    # Generate Gray code patterns for projector calibration
    gray_patterns = generate_gray_code_patterns(
        args.projector_width, args.projector_height, args.gray_bits
    )
    
    # Collect calibration data
    camera_images = []
    captured_pattern_data = []
    
    try:
        for position in range(args.num_positions):
            print(f"\n{'='*60}")
            print(f"CALIBRATION POSITION {position + 1}/{args.num_positions}")
            print(f"{'='*60}")
            print("1. Place checkerboard in a new position")
            print("2. Ensure checkerboard is flat and well-lit")
            print("3. Press Enter when ready to capture...")
            input()
            
            position_images = []
            position_patterns = []
            
            # Capture sequence for this position
            for i, pattern in enumerate(gray_patterns):
                print(f"  Capturing pattern {i+1}/{len(gray_patterns)}...")
                
                # Project pattern
                # Note: This would need integration with the actual projector control
                # For now, we simulate by capturing with different patterns
                
                # Capture image from camera
                cameras = client.camera.get_cameras()
                if not cameras:
                    logger.error("No cameras available")
                    return 1
                
                # Capture from left camera
                left_cam = cameras[0]['id']
                captured_image = client.camera.capture_image(left_cam)
                
                if captured_image is not None:
                    position_images.append(captured_image)
                    position_patterns.append(pattern)
                    
                    # Save images if requested
                    if args.save_images:
                        save_dir = Path("calibration_capture")
                        save_dir.mkdir(exist_ok=True)
                        
                        img_file = save_dir / f"pos{position:02d}_pattern{i:02d}.png"
                        cv2.imwrite(str(img_file), captured_image)
                        
                        pattern_file = save_dir / f"pattern{i:02d}.png"
                        if not pattern_file.exists():
                            cv2.imwrite(str(pattern_file), pattern)
                
                time.sleep(0.5)  # Wait between captures
            
            # Check if checkerboard is visible in first image
            first_img = position_images[0] if position_images else None
            if first_img is not None:
                gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY) if len(first_img.shape) == 3 else first_img
                found, corners = cv2.findChessboardCorners(gray, checkerboard_size)
                
                if found:
                    camera_images.extend(position_images)
                    captured_pattern_data.extend(position_patterns)
                    logger.info(f"Position {position+1}: Checkerboard detected, {len(position_images)} images captured")
                else:
                    logger.warning(f"Position {position+1}: Checkerboard not detected, skipping")
            
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
    finally:
        client.disconnect()
    
    if len(camera_images) < 10:
        logger.error(f"Not enough calibration images: {len(camera_images)}")
        return 1
    
    # Perform calibration
    logger.info(f"Performing calibration with {len(camera_images)} images...")
    
    try:
        # For now, perform camera calibration only
        # Full projector-camera calibration requires proper Gray code decoding
        camera_intrinsics, camera_distortion, _ = calibrator.calibrate_camera(camera_images)
        
        # Save calibration results
        calibration_data = {
            'projector_resolution': [args.projector_width, args.projector_height],
            'checkerboard_size': list(checkerboard_size),
            'square_size_mm': args.square_size,
            'camera_intrinsics': camera_intrinsics.tolist(),
            'camera_distortion': camera_distortion.tolist(),
            'num_calibration_images': len(camera_images),
            'calibration_method': 'camera_only',  # Indicates partial calibration
            'note': 'Partial calibration - projector calibration requires full Gray code decoding implementation'
        }
        
        with open(args.output, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {args.output}")
        
        # Print calibration results
        print(f"\n{'='*60}")
        print("CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"Camera intrinsics:")
        print(f"  fx: {camera_intrinsics[0,0]:.2f}")
        print(f"  fy: {camera_intrinsics[1,1]:.2f}")
        print(f"  cx: {camera_intrinsics[0,2]:.2f}")
        print(f"  cy: {camera_intrinsics[1,2]:.2f}")
        print(f"Distortion coefficients: {camera_distortion.flatten()}")
        print(f"Calibration images: {len(camera_images)}")
        print(f"Output file: {args.output}")
        print(f"\nNote: This is a partial calibration (camera only).")
        print(f"For full projector-camera calibration, implement proper Gray code decoding.")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def process_existing_images(args):
    """Process existing calibration images."""
    logger.info("Processing existing calibration images")
    
    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return 1
    
    # Parse checkerboard size
    try:
        cb_width, cb_height = map(int, args.checkerboard_size.split('x'))
        checkerboard_size = (cb_width, cb_height)
    except ValueError:
        logger.error(f"Invalid checkerboard size: {args.checkerboard_size}")
        return 1
    
    # Initialize calibrator
    calibrator = ProjectorCalibrator(
        projector_width=args.projector_width,
        projector_height=args.projector_height,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size
    )
    
    # Load images
    camera_images, pattern_images = load_calibration_images(args.input)
    
    if not camera_images:
        logger.error("No camera images found")
        return 1
    
    try:
        # Perform camera calibration
        camera_intrinsics, camera_distortion, image_points = calibrator.calibrate_camera(camera_images)
        
        # Save results
        calibration_data = {
            'projector_resolution': [args.projector_width, args.projector_height],
            'checkerboard_size': list(checkerboard_size),
            'square_size_mm': args.square_size,
            'camera_intrinsics': camera_intrinsics.tolist(),
            'camera_distortion': camera_distortion.tolist(),
            'num_calibration_images': len(camera_images),
            'calibration_method': 'camera_only'
        }
        
        with open(args.output, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {args.output}")
        
        # Print results
        print(f"\n{'='*60}")
        print("CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"Method: Camera calibration")
        print(f"Images processed: {len(camera_images)}")
        print(f"Valid images: {len(image_points)}")
        print(f"Camera intrinsics:")
        print(f"  fx: {camera_intrinsics[0,0]:.2f}")
        print(f"  fy: {camera_intrinsics[1,1]:.2f}")
        print(f"  cx: {camera_intrinsics[0,2]:.2f}")
        print(f"  cy: {camera_intrinsics[1,2]:.2f}")
        print(f"Output: {args.output}")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger('unlook').setLevel(logging.DEBUG)
    
    # Check mode
    if args.interactive:
        return interactive_calibration(args)
    elif args.input:
        return process_existing_images(args)
    else:
        print("Error: Must specify either --interactive or --input")
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())