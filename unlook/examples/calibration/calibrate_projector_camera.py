#!/usr/bin/env python3
"""
Projector-Camera Calibration Script for UnLook SDK

Calibrates a projector-camera system for structured light 3D scanning.
Correctly implemented using the UnLook SDK architecture with proper pattern projection.

Features:
- Live preview with streaming like capture_checkerboard.py
- Working pattern projection using UnlookClient.projector
- Proper LED intensity control (0mA = OFF)
- Real-time checkerboard detection
- Interactive controls (c=capture, q=quit, ESC=force quit)
- Clean shutdown handling

Usage:
    # Interactive calibration with live preview (recommended)
    python calibrate_projector_camera.py --interactive --live-preview --led-intensity 0

    # With specific projector resolution
    python calibrate_projector_camera.py --interactive --live-preview --projector-width 1280 --projector-height 720
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import json
import signal
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanning.calibration.projector_calibration import ProjectorCalibrator
from unlook.client.scanner.scanner import UnlookClient
from unlook.client.scanning.pattern_manager import PatternManager
from unlook.client.scanner.pattern_decoder import PatternDecoder

# Global variables for clean shutdown
global_client = None
streaming_active = False
should_quit = False

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) to clean up before exiting."""
    global global_client, streaming_active, should_quit
    logger.info("Received interrupt signal, cleaning up...")
    
    streaming_active = False
    should_quit = True
    
    if global_client:
        try:
            # Stop any active streams first
            if hasattr(global_client, 'stream') and global_client.stream:
                try:
                    global_client.stream.stop()
                    logger.info("Stream stopped")
                except:
                    pass
            
            # Turn off LED before disconnecting
            global_client.projector.led_off()
            logger.info("LED turned off")
        except Exception as e:
            logger.warning(f"Could not turn off LED: {e}")
        
        try:
            global_client.disconnect()
            logger.info("Disconnected from scanner")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")
    
    logger.info("Cleanup complete, exiting...")
    cv2.destroyAllWindows()
    time.sleep(0.5)
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

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
    
    parser.add_argument('--live-preview', action='store_true',
                       help='Use live preview mode with streaming')
    
    parser.add_argument('--checkerboard-size', type=str, default='9x6',
                       help='Checkerboard size as WIDTHxHEIGHT (internal corners)')
    
    parser.add_argument('--square-size', type=float, default=23.13,
                       help='Checkerboard square size in mm')
    
    parser.add_argument('--projector-width', type=int, default=1280,
                       help='Projector resolution width')
    
    parser.add_argument('--projector-height', type=int, default=720,
                       help='Projector resolution height')
    
    parser.add_argument('--gray-bits', type=int, default=7,
                       help='Number of Gray code bits for calibration')
    
    parser.add_argument('--num-positions', type=int, default=8,
                       help='Number of checkerboard positions for interactive mode')
    
    parser.add_argument('--scanner', type=str,
                       help='Specific scanner name to connect to')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--save-images', action='store_true',
                       help='Save captured calibration images')
    
    parser.add_argument('--led-intensity', type=int, default=0,
                       help='LED intensity in mA (0-450, 0=OFF)')
    
    parser.add_argument('--camera-calibration', type=Path,
                       help='Path to existing camera calibration file to use fixed intrinsics')
    
    parser.add_argument('--process-captured', type=Path,
                       help='Process already captured calibration images from directory (e.g., calibration_capture)')
    
    return parser.parse_args()


def load_calibration_images(input_dir: Path):
    """Load existing calibration images from directory."""
    camera_images = []
    
    # Look for camera images
    camera_dir = input_dir / "camera"
    if camera_dir.exists():
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        for ext in image_extensions:
            for img_file in sorted(camera_dir.glob(ext)):
                img = cv2.imread(str(img_file))
                if img is not None:
                    camera_images.append(img)
                    logger.info(f"Loaded camera image: {img_file.name}")
    
    logger.info(f"Loaded {len(camera_images)} camera images")
    return camera_images


def generate_gray_code_patterns_for_calibration(projector_width: int, projector_height: int, num_bits: int) -> List[Dict[str, Any]]:
    """
    Generate Gray Code patterns for projector calibration.
    
    Args:
        projector_width: Projector resolution width
        projector_height: Projector resolution height  
        num_bits: Number of Gray code bits
    
    Returns:
        List of pattern configurations for projection
    """
    # Initialize pattern manager
    pattern_manager = PatternManager()
    
    # Generate Gray Code patterns for both orientations
    patterns = pattern_manager.create_gray_code_patterns(
        num_bits=num_bits,
        use_blue=True,
        include_inverse=True,
        orientation="both",
        projector_width=projector_width,
        projector_height=projector_height
    )
    
    logger.info(f"Generated {len(patterns)} Gray Code patterns for calibration")
    
    # Convert PatternInfo objects to a format suitable for projection
    pattern_configs = []
    for pattern in patterns:
        config = {
            "type": pattern.pattern_type,
            "name": pattern.name,
            "metadata": pattern.metadata,
            "parameters": pattern.parameters,
            "description": f"{pattern.name} - {pattern.pattern_type}"
        }
        pattern_configs.append(config)
    
    return pattern_configs


def project_pattern(client, pattern_config: Dict[str, Any], led_intensity: int = 0) -> bool:
    """
    Project a single pattern from the configuration.
    
    Args:
        client: UnlookClient instance
        pattern_config: Pattern configuration with type, parameters, etc.
        led_intensity: LED intensity in mA (0=OFF)
    
    Returns:
        True if projection successful, False otherwise
    """
    try:
        # Set LED intensity
        if led_intensity > 0:
            client.projector.led_set_intensity(led1_mA=0, led2_mA=led_intensity)
        else:
            client.projector.led_off()
            
        # Project pattern based on type
        pattern_type = pattern_config["type"]
        params = pattern_config["parameters"]
        
        if pattern_type == "solid_field":
            return client.projector.show_solid_field(params["color"])
        elif pattern_type == "vertical_lines":
            # Extract only valid parameters for show_vertical_lines
            return client.projector.show_vertical_lines(
                foreground_color=params.get("foreground_color", "White"),
                background_color=params.get("background_color", "Black"),
                foreground_width=params.get("foreground_width", 4),
                background_width=params.get("background_width", 20)
            )
        elif pattern_type == "horizontal_lines":
            # Extract only valid parameters for show_horizontal_lines
            return client.projector.show_horizontal_lines(
                foreground_color=params.get("foreground_color", "White"),
                background_color=params.get("background_color", "Black"),
                foreground_width=params.get("foreground_width", 4),
                background_width=params.get("background_width", 20)
            )
        else:
            logger.error(f"Unknown pattern type: {pattern_type}")
            return False
            
    except Exception as e:
        logger.error(f"Error projecting pattern: {e}")
        return False


def clear_projector_pattern(client):
    """Clear projector pattern (show black field)."""
    try:
        logger.info("🔵 DEBUG: Clearing projector...")
        result = client.projector.show_solid_field("Black")
        logger.info(f"🔵 DEBUG: Projector clear result: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ DEBUG: Error clearing projector: {e}")
        return False


def capture_gray_code_sequence(client, camera_id: str, patterns: List[Dict[str, Any]], 
                                checkerboard_size: Tuple[int, int],
                                led_intensity: int = 0, auto_capture: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Capture a sequence of Gray Code patterns with checkerboard visible.
    
    Args:
        client: UnlookClient instance
        camera_id: Camera ID to capture from
        patterns: List of Gray Code pattern configurations
        led_intensity: LED intensity in mA
        auto_capture: If True, capture all patterns automatically with delays
        
    Returns:
        Tuple of (captured_images, checkerboard_corners)
    """
    captured_images = []
    checkerboard_found = False
    checkerboard_corners = None
    
    logger.info(f"Capturing Gray Code sequence with {len(patterns)} patterns")
    
    if auto_capture:
        print("\n🚨 AUTOMATIC CAPTURE SEQUENCE STARTING!")
        print("="*60)
        print("⚠️  CRITICAL: Keep the checkerboard PERFECTLY STILL!")
        print("🤖 All patterns will be captured automatically")
        print(f"📊 Total patterns: {len(patterns)}")
        print("⏱️  Starting in 3 seconds...")
        print("="*60)
        time.sleep(3)
    
    for idx, pattern in enumerate(patterns):
        print(f"\n📸 [{idx + 1}/{len(patterns)}] Capturing: {pattern['name']}")
        
        # Project pattern
        if project_pattern(client, pattern, led_intensity):
            # Wait for pattern to stabilize
            time.sleep(0.3)
            
            # Auto-capture mode - show pattern info and capture automatically
            print(f"✅ Pattern projected successfully")
            print(f"📋 Pattern details: {pattern['name']} ({pattern.get('type', 'unknown')})") 
            params = pattern.get('parameters', {})
            print(f"📐 Parameters: {params}")
            
            # Show stripe width info if available
            if 'foreground_width' in params:
                print(f"📏 STRIPE WIDTH: {params['foreground_width']}px (ENORMOUS - 3-4x bigger for camera visibility!)")
            if 'stripe_width_info' in params:
                print(f"🔍 {params['stripe_width_info']}")
                
            print("\n🎯 ENORMOUS stripes projected - capturing automatically")
            print("📸 Capturing image...")
            
            # Capture image
            images = client.camera.capture_multi([camera_id])
            if images and camera_id in images and images[camera_id] is not None:
                captured_images.append(images[camera_id])
                
                print(f"✅ Image captured successfully for pattern {idx + 1}")
                
                # Try to detect checkerboard in reference white image
                if pattern['name'] == 'reference_white' and not checkerboard_found:
                    gray = cv2.cvtColor(images[camera_id], cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(
                        gray, 
                        checkerboard_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                    if found:
                        checkerboard_found = True
                        checkerboard_corners = corners
                        logger.info("Checkerboard detected in reference white image")
                        print("🎯 Checkerboard detected in reference white image!")
            else:
                logger.error(f"Failed to capture image for pattern {idx}")
                print(f"❌ Failed to capture image for pattern {idx + 1}")
                
        else:
            logger.error(f"Failed to project pattern {idx}")
            print(f"❌ Failed to project pattern {idx + 1}")
    
    return captured_images, checkerboard_corners


def load_captured_calibration_data(captured_dir: Path, checkerboard_size: Tuple[int, int], 
                                  projector_width: int, projector_height: int,
                                  debug: bool = True) -> List[Dict[str, Any]]:
    """
    Load and process captured calibration images from directory structure.
    
    Args:
        captured_dir: Directory containing position_XX subdirectories
        checkerboard_size: Checkerboard size (width, height)
        projector_width: Projector resolution width
        projector_height: Projector resolution height
        debug: Enable debug output
        
    Returns:
        List of calibration position data
    """
    logger.info(f"Loading captured calibration data from {captured_dir}")
    
    if not captured_dir.exists():
        raise ValueError(f"Captured directory not found: {captured_dir}")
    
    # Find all position directories
    position_dirs = sorted([d for d in captured_dir.iterdir() if d.is_dir() and d.name.startswith('position_')])
    logger.info(f"Found {len(position_dirs)} position directories: {[d.name for d in position_dirs]}")
    
    calibration_positions = []
    
    for pos_idx, pos_dir in enumerate(position_dirs):
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING POSITION {pos_idx + 1}: {pos_dir.name}")
        logger.info(f"{'='*60}")
        
        # Load all pattern images from this position
        pattern_files = sorted([f for f in pos_dir.glob("pattern_*.png")])
        logger.info(f"Found {len(pattern_files)} pattern images")
        
        if len(pattern_files) == 0:
            logger.warning(f"No pattern images found in {pos_dir}")
            continue
        
        # Load images
        captured_images = []
        for i, pattern_file in enumerate(pattern_files):
            img = cv2.imread(str(pattern_file))
            if img is not None:
                captured_images.append(img)
                if debug and i < 3:  # Debug first 3 images
                    logger.info(f"  Loaded pattern {i:03d}: {pattern_file.name} - {img.shape}")
            else:
                logger.warning(f"Failed to load: {pattern_file}")
        
        logger.info(f"Successfully loaded {len(captured_images)} images")
        
        if len(captured_images) < 10:  # Need at least white, black + some patterns
            logger.warning(f"Too few images ({len(captured_images)}) for position {pos_idx + 1}")
            continue
        
        # Find checkerboard in reference white image (first image)
        white_img = captured_images[0]
        gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        
        logger.info("Detecting checkerboard corners...")
        found, corners = cv2.findChessboardCorners(
            gray, 
            checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not found:
            logger.error(f"❌ Checkerboard not found in position {pos_idx + 1}")
            continue
        
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        logger.info(f"✅ Found {len(corners)} checkerboard corners")
        
        # Debug: Save checkerboard detection image
        if debug:
            debug_img = white_img.copy()
            cv2.drawChessboardCorners(debug_img, checkerboard_size, corners, found)
            debug_path = captured_dir / f"debug_checkerboard_pos_{pos_idx + 1:02d}.jpg"
            cv2.imwrite(str(debug_path), debug_img)
            logger.info(f"💾 Saved checkerboard debug: {debug_path}")
        
        # Decode projector coordinates with enhanced debugging
        logger.info(f"🔍 Decoding Gray Code patterns...")
        projector_points = decode_projector_coordinates_enhanced(
            captured_images, corners, projector_width, projector_height, 
            debug=debug, debug_dir=captured_dir / f"debug_pos_{pos_idx + 1:02d}"
        )
        
        if projector_points is not None and len(projector_points) > 0:
            position_data = {
                'camera_corners': corners,
                'projector_corners': projector_points,
                'captured_images': captured_images,
                'position_index': pos_idx + 1,
                'debug_info': {
                    'num_patterns': len(captured_images),
                    'num_corners': len(corners),
                    'num_projector_points': len(projector_points)
                }
            }
            calibration_positions.append(position_data)
            logger.info(f"✅ Position {pos_idx + 1} processed successfully")
        else:
            logger.error(f"❌ Failed to decode projector coordinates for position {pos_idx + 1}")
    
    logger.info(f"\n🏁 PROCESSING COMPLETE: {len(calibration_positions)}/{len(position_dirs)} positions successful")
    return calibration_positions


def decode_projector_coordinates_enhanced(captured_images: List[np.ndarray], 
                                        checkerboard_corners: np.ndarray,
                                        projector_width: int,
                                        projector_height: int,
                                        debug: bool = True,
                                        debug_dir: Optional[Path] = None) -> np.ndarray:
    """
    Enhanced version with extensive debugging for Gray Code decoding.
    """
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)
    
    logger.info(f"🔍 Enhanced Gray Code decoding: {len(captured_images)} images")
    logger.info(f"   Projector resolution: {projector_width}x{projector_height}")
    logger.info(f"   Checkerboard corners: {len(checkerboard_corners)}")
    
    # Analyze image quality first
    for i, img in enumerate(captured_images[:5]):  # Check first 5 images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        logger.info(f"   Image {i}: brightness={mean_brightness:.1f}±{std_brightness:.1f}")
        
        if debug_dir and i < 3:
            cv2.imwrite(str(debug_dir / f"input_pattern_{i:02d}.jpg"), img)
    
    try:
        # Use PatternDecoder to decode Gray Code with debug output
        x_coords, y_coords, mask = PatternDecoder.decode_gray_code(
            captured_images,
            projector_width,
            projector_height,
            threshold=5.0,
            debug_dir=str(debug_dir) if debug_dir else None
        )
        
        logger.info(f"   Decoded coordinates: {x_coords.shape}")
        logger.info(f"   Valid pixels: {np.sum(mask)}/{mask.size} = {np.sum(mask)/mask.size*100:.1f}%")
        
        if np.sum(mask) == 0:
            logger.error("❌ No valid pixels decoded!")
            return None
        
        # Analyze coordinate ranges
        valid_x = x_coords[mask]
        valid_y = y_coords[mask]
        logger.info(f"   X range: {np.min(valid_x):.1f} to {np.max(valid_x):.1f}")
        logger.info(f"   Y range: {np.min(valid_y):.1f} to {np.max(valid_y):.1f}")
        
        # Extract projector coordinates at checkerboard corners
        projector_points = []
        valid_count = 0
        invalid_count = 0
        interpolated_count = 0
        
        logger.info(f"🎯 Extracting coordinates for {len(checkerboard_corners)} corners:")
        
        for i, corner in enumerate(checkerboard_corners):
            x, y = corner[0]  # Corner coordinates in camera image
            x_int, y_int = int(round(x)), int(round(y))
            
            # Check bounds
            if not (0 <= x_int < x_coords.shape[1] and 0 <= y_int < x_coords.shape[0]):
                invalid_count += 1
                logger.warning(f"   Corner {i:2d}: OUT OF BOUNDS ({x:.1f}, {y:.1f})")
                # Fallback to simple scaling
                proj_x = x * projector_width / x_coords.shape[1]
                proj_y = y * projector_height / x_coords.shape[0]
                projector_points.append([proj_x, proj_y])
                continue
            
            # Check if valid decoding at this point
            if mask[y_int, x_int]:
                proj_x = x_coords[y_int, x_int]
                proj_y = y_coords[y_int, x_int]
                projector_points.append([proj_x, proj_y])
                valid_count += 1
                
                if debug and i % 10 == 0:  # Debug every 10th point
                    logger.info(f"   Corner {i:2d}: Camera({x:.1f}, {y:.1f}) -> Projector({proj_x:.1f}, {proj_y:.1f}) ✅")
            else:
                # Try interpolation from nearby valid points
                proj_x, proj_y = interpolate_projector_point(x_int, y_int, x_coords, y_coords, mask, search_radius=10)
                if proj_x is not None:
                    projector_points.append([proj_x, proj_y])
                    interpolated_count += 1
                    if debug and i % 10 == 0:
                        logger.info(f"   Corner {i:2d}: Camera({x:.1f}, {y:.1f}) -> Projector({proj_x:.1f}, {proj_y:.1f}) 🔧")
                else:
                    # Fallback to simple scaling
                    proj_x = x * projector_width / x_coords.shape[1]
                    proj_y = y * projector_height / x_coords.shape[0]
                    projector_points.append([proj_x, proj_y])
                    invalid_count += 1
                    if debug and i % 10 == 0:
                        logger.warning(f"   Corner {i:2d}: Camera({x:.1f}, {y:.1f}) -> Projector({proj_x:.1f}, {proj_y:.1f}) ❌")
        
        # Summary statistics
        total_corners = len(checkerboard_corners)
        valid_rate = valid_count / total_corners * 100
        interpolated_rate = interpolated_count / total_corners * 100
        invalid_rate = invalid_count / total_corners * 100
        
        logger.info(f"📊 DECODING SUMMARY:")
        logger.info(f"   Valid corners:        {valid_count:2d}/{total_corners} = {valid_rate:.1f}%")
        logger.info(f"   Interpolated corners: {interpolated_count:2d}/{total_corners} = {interpolated_rate:.1f}%")
        logger.info(f"   Invalid corners:      {invalid_count:2d}/{total_corners} = {invalid_rate:.1f}%")
        
        if valid_rate < 50:
            logger.error(f"❌ Low valid decoding rate: {valid_rate:.1f}% - calibration may be unreliable!")
        elif valid_rate < 80:
            logger.warning(f"⚠️ Moderate decoding rate: {valid_rate:.1f}% - check pattern quality")
        else:
            logger.info(f"✅ Good decoding rate: {valid_rate:.1f}%")
        
        # Save debug visualization
        if debug_dir:
            create_decoding_debug_visualization(
                captured_images[0], checkerboard_corners, projector_points,
                x_coords, y_coords, mask, debug_dir / "decoding_visualization.jpg"
            )
        
        return np.array(projector_points, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"❌ Error in Gray Code decoding: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_decoding_debug_visualization(reference_img: np.ndarray,
                                      camera_corners: np.ndarray,
                                      projector_points: np.ndarray,
                                      x_coords: np.ndarray,
                                      y_coords: np.ndarray,
                                      mask: np.ndarray,
                                      output_path: Path):
    """Create a comprehensive debug visualization."""
    
    # Create visualization image
    h, w = reference_img.shape[:2]
    debug_img = reference_img.copy()
    
    # Draw decoded coordinate field (as colored overlay)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize coordinates for visualization
    if np.sum(mask) > 0:
        x_norm = np.zeros_like(x_coords)
        y_norm = np.zeros_like(y_coords)
        
        valid_mask = mask & (x_coords >= 0) & (y_coords >= 0)
        if np.sum(valid_mask) > 0:
            x_valid = x_coords[valid_mask]
            y_valid = y_coords[valid_mask]
            
            x_min, x_max = np.min(x_valid), np.max(x_valid)
            y_min, y_max = np.min(y_valid), np.max(y_valid)
            
            if x_max > x_min:
                x_norm[valid_mask] = (x_coords[valid_mask] - x_min) / (x_max - x_min) * 255
            if y_max > y_min:
                y_norm[valid_mask] = (y_coords[valid_mask] - y_min) / (y_max - y_min) * 255
            
            overlay[:, :, 0] = x_norm.astype(np.uint8)  # Red = X coordinate
            overlay[:, :, 1] = y_norm.astype(np.uint8)  # Green = Y coordinate
            overlay[:, :, 2] = mask.astype(np.uint8) * 255  # Blue = Valid mask
    
    # Blend overlay with original image
    alpha = 0.3
    debug_img = cv2.addWeighted(debug_img, 1 - alpha, overlay, alpha, 0)
    
    # Draw checkerboard corners and projector correspondences
    for i, (corner, proj_point) in enumerate(zip(camera_corners, projector_points)):
        x, y = corner[0]
        proj_x, proj_y = proj_point
        
        # Draw camera corner
        cv2.circle(debug_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw text with projector coordinates
        if i % 5 == 0:  # Label every 5th point to avoid clutter
            text = f"P({proj_x:.0f},{proj_y:.0f})"
            cv2.putText(debug_img, text, (int(x) + 5, int(y) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Add legend
    cv2.putText(debug_img, "Red=ProjX, Green=ProjY, Blue=Valid", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Valid: {np.sum(mask)}/{mask.size} = {np.sum(mask)/mask.size*100:.1f}%", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), debug_img)
    logger.info(f"💾 Saved decoding visualization: {output_path}")


def decode_projector_coordinates(captured_images: List[np.ndarray], 
                                  checkerboard_corners: np.ndarray,
                                  projector_width: int,
                                  projector_height: int,
                                  debug: bool = True) -> np.ndarray:
    """
    Decode projector coordinates for each checkerboard corner.
    
    Args:
        captured_images: List of captured Gray Code images
        checkerboard_corners: Detected checkerboard corners in camera coordinates
        projector_width: Projector resolution width
        projector_height: Projector resolution height
        
    Returns:
        Array of projector coordinates corresponding to each checkerboard corner
    """
    # Use PatternDecoder to decode Gray Code
    x_coords, y_coords, mask = PatternDecoder.decode_gray_code(
        captured_images,
        projector_width,
        projector_height
    )
    
    # Extract projector coordinates at checkerboard corners
    projector_points = []
    valid_count = 0
    invalid_count = 0
    
    for i, corner in enumerate(checkerboard_corners):
        x, y = corner[0]  # Corner coordinates in camera image
        x_int, y_int = int(round(x)), int(round(y))
        
        # Check if valid decoding at this point
        if 0 <= x_int < x_coords.shape[1] and 0 <= y_int < x_coords.shape[0]:
            if mask[y_int, x_int]:
                proj_x = x_coords[y_int, x_int]
                proj_y = y_coords[y_int, x_int]
                projector_points.append([proj_x, proj_y])
                valid_count += 1
                
                if debug and i < 5:  # Debug first 5 points
                    logger.info(f"Corner {i}: Camera({x:.1f}, {y:.1f}) -> Projector({proj_x:.1f}, {proj_y:.1f})")
            else:
                invalid_count += 1
                logger.warning(f"Invalid decoding at corner ({x}, {y})")
                # Use interpolation from nearby valid points if possible
                proj_x, proj_y = interpolate_projector_point(x_int, y_int, x_coords, y_coords, mask)
                if proj_x is not None:
                    projector_points.append([proj_x, proj_y])
                else:
                    # Fallback to simple scaling
                    projector_points.append([x * projector_width / x_coords.shape[1], 
                                           y * projector_height / x_coords.shape[0]])
        else:
            invalid_count += 1
            logger.warning(f"Corner {i} out of bounds: ({x}, {y})")
            projector_points.append([x * projector_width / x_coords.shape[1], 
                                   y * projector_height / x_coords.shape[0]])
    
    if debug:
        logger.info(f"Decoded {valid_count} valid corners, {invalid_count} invalid/interpolated")
        if valid_count < len(checkerboard_corners) * 0.8:
            logger.warning(f"Low decoding rate: {valid_count}/{len(checkerboard_corners)} = {valid_count/len(checkerboard_corners)*100:.1f}%")
    
    return np.array(projector_points, dtype=np.float32)


def interpolate_projector_point(x: int, y: int, x_coords: np.ndarray, y_coords: np.ndarray, mask: np.ndarray, 
                               search_radius: int = 5) -> Tuple[Optional[float], Optional[float]]:
    """Interpolate projector coordinates from nearby valid points."""
    h, w = mask.shape
    
    # Search for valid neighbors
    valid_neighbors_x = []
    valid_neighbors_y = []
    weights = []
    
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    valid_neighbors_x.append(x_coords[ny, nx])
                    valid_neighbors_y.append(y_coords[ny, nx])
                    weights.append(1.0 / dist)
    
    if valid_neighbors_x:
        # Weighted average
        weights = np.array(weights)
        weights /= weights.sum()
        proj_x = np.sum(np.array(valid_neighbors_x) * weights)
        proj_y = np.sum(np.array(valid_neighbors_y) * weights)
        return proj_x, proj_y
    
    return None, None


def interactive_calibration(args):
    """Perform interactive calibration with live streaming preview."""
    global global_client, streaming_active, should_quit
    
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
    
    # Create UnlookClient with auto-discovery (same as capture_checkerboard.py)
    logger.info("Creating UnlookClient for pattern projection...")
    client = UnlookClient(client_name="ProjectorCalibration", auto_discover=True)
    global_client = client  # Set global reference for signal handler
    
    # Wait for discovery
    logger.info("Discovering scanners...")
    time.sleep(3)
    
    # Get discovered scanners
    scanners = client.get_discovered_scanners()
    if not scanners:
        logger.error("No scanners found. Check that your hardware is connected.")
        return 1
    
    # Select scanner
    if args.scanner:
        scanner = next((s for s in scanners if s.name == args.scanner), None)
        if not scanner:
            logger.error(f"Scanner '{args.scanner}' not found")
            return 1
    else:
        scanner = scanners[0]
    
    logger.info(f"Connecting to scanner: {scanner.name}")
    
    # Connect to scanner
    if not client.connect(scanner):
        logger.error("Failed to connect to scanner.")
        return 1
    
    logger.info("Successfully connected!")
    
    # Get available cameras
    cameras = client.camera.get_cameras()
    if not cameras:
        logger.error("No cameras found!")
        return 1
        
    # Use first camera for single-camera calibration
    camera_id = cameras[0]['id'] if isinstance(cameras[0], dict) else cameras[0]
    logger.info(f"Using camera: {camera_id}")
    
    # Generate Gray Code patterns for calibration
    gray_code_patterns = generate_gray_code_patterns_for_calibration(
        args.projector_width,
        args.projector_height,
        args.gray_bits
    )
    
    # Collect calibration data
    calibration_positions = []  # List of (camera_corners, projector_corners) for each position
    captured_count = 0
    
    if args.live_preview:
        print("\\n" + "="*60)
        print("LIVE PREVIEW MODE - PROJECTOR-CAMERA CALIBRATION")
        print("="*60)
        print(f"Checkerboard: {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
        print(f"Square size: {args.square_size}mm")
        print(f"Target positions: {args.num_positions}")
        print(f"LED intensity: {args.led_intensity}mA")
        print("\\nInstructions:")
        print("1. Position the checkerboard at various angles and distances")
        print("2. Ensure the ENTIRE checkerboard is visible in camera view")
        print("3. Press 'c' or SPACE to capture Gray Code sequence when ready")
        print("4. The system will capture all patterns automatically")
        print("5. Press 'q' or ESC to finish calibration")
        print("="*60 + "\\n")
        
        # Variables for capture control
        capture_requested = False
        checkerboard_found = False
        current_image = None
        
        # Project white field for checkerboard detection
        logger.info("🔵 DEBUG: Projecting white field for checkerboard detection...")
        pattern_success = client.projector.show_solid_field("White")
        
        if pattern_success:
            print("✅ White field projected - position your checkerboard")
        else:
            print("❌ Failed to project white field")
        
        # Use direct capture loop instead of streaming (faster and more reliable)
        logger.info("Starting direct capture mode (no streaming)...")
        
        try:
            while captured_count < args.num_positions and not should_quit:
                try:
                    # Direct capture without streaming
                    images = client.camera.capture_multi([camera_id])
                    if not images or camera_id not in images or images[camera_id] is None:
                        print("❌ Failed to capture image from camera")
                        time.sleep(0.1)
                        continue
                    
                    current_image = images[camera_id]
                    display_img = current_image.copy()
                    
                    # Detect checkerboard
                    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(
                        gray, 
                        checkerboard_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                    
                    checkerboard_found = found
                    
                    # Show status
                    cv2.putText(display_img, f"Position {captured_count + 1}/{args.num_positions}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show projector status
                    cv2.putText(display_img, f"Projector: White field for checkerboard detection", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if found:
                        # Draw checkerboard
                        cv2.drawChessboardCorners(display_img, checkerboard_size, corners, found)
                        
                        # Green border - ready
                        cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), 
                                     (0, 255, 0), 5)
                        cv2.putText(display_img, "READY - Press 'c' to capture patterns", 
                                   (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # Red border - not detected
                        cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), 
                                     (0, 0, 255), 5)
                        cv2.putText(display_img, "CHECKERBOARD NOT DETECTED", 
                                   (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Instructions
                    cv2.putText(display_img, "Controls: 'c'/SPACE=capture Gray Code sequence, 'q'/ESC=quit", 
                               (10, display_img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show image
                    cv2.imshow('Projector Calibration - Live Preview', display_img)
                    
                    # Handle keyboard input with optimized timeout for better performance
                    key = cv2.waitKey(10) & 0xFF  # Reduced timeout for faster response
                    
                    if key == ord('q') or key == 27:  # q or ESC
                        logger.info("User requested quit")
                        should_quit = True
                        break
                    elif key == 32 and found:  # SPACE (only) and checkerboard found
                        logger.info(f"Capture requested for position {captured_count + 1}")
                        
                        print(f"\\n🔵 Capturing Gray Code sequence for position {captured_count + 1}...")
                        print(f"Patterns: {len(gray_code_patterns)} total (with manual control)")
                        print(f"Each pattern has ENORMOUS stripes (400-900px) for camera visibility")
                        
                        # Store current corners before starting pattern sequence
                        current_corners = corners.copy()
                        
                        # CLOSE LIVE PREVIEW WINDOW BEFORE PATTERN CAPTURE
                        cv2.destroyAllWindows()
                        time.sleep(0.2)  # Give time for window to close
                        
                        # Capture Gray Code sequence with enhanced manual control
                        try:
                            print("\n🕰️ STARTING GRAY CODE PATTERN SEQUENCE")
                            print("="*60)
                            print("📋 Enhanced features active:")
                            print("  ✅ Manual pattern control (wait for your input)")
                            print("  ✅ ENORMOUS patterns (400-900px stripes, camera preview visible)")
                            print("  ✅ Live preview CLOSED to avoid conflicts")
                            print("  🔍 Each pattern will be CLEARLY visible on projector")
                            print("  📺 Check your projector screen directly")
                            print("  ⚠️ If stripes are not visible, check projector focus!")
                            print("="*60)
                            
                            captured_images, detected_corners = capture_gray_code_sequence(
                                client, camera_id, gray_code_patterns, checkerboard_size, args.led_intensity, auto_capture=True
                            )
                            
                            if len(captured_images) == len(gray_code_patterns):
                                # Decode projector coordinates
                                projector_points = decode_projector_coordinates(
                                    captured_images,
                                    current_corners,
                                    args.projector_width,
                                    args.projector_height
                                )
                                
                                # Store calibration data
                                calibration_positions.append({
                                    'camera_corners': current_corners,
                                    'projector_corners': projector_points,
                                    'captured_images': captured_images
                                })
                                
                                captured_count += 1
                                print(f"✅ Position {captured_count}: Gray Code sequence captured successfully!")
                                print(f"   Decoded {len(projector_points)} projector correspondences")
                                
                                # Save images if requested
                                if args.save_images:
                                    output_dir = Path("calibration_capture") / f"position_{captured_count:02d}"
                                    output_dir.mkdir(parents=True, exist_ok=True)
                                    for idx, img in enumerate(captured_images):
                                        img_path = output_dir / f"pattern_{idx:03d}.png"
                                        cv2.imwrite(str(img_path), img)
                                    logger.info(f"Saved {len(captured_images)} images to {output_dir}")
                                
                                # Return to white field for next position
                                client.projector.show_solid_field("White")
                                
                                # RESTART LIVE PREVIEW after pattern capture
                                print("\n🔄 Restarting live preview for next position...")
                                time.sleep(0.5)  # Give time before restarting preview
                            else:
                                print(f"❌ Position {captured_count+1}: Failed to capture all patterns")
                                
                        except Exception as e:
                            logger.error(f"❌ DEBUG: Error capturing Gray Code sequence: {e}")
                            print(f"❌ Position {captured_count+1}: Gray Code capture failed")
                            # Return to white field and restart live preview
                            client.projector.show_solid_field("White")
                            print("\n🔄 Restarting live preview...")
                            time.sleep(0.5)
                    
                    # Optimized delay to prevent excessive CPU usage but maintain responsiveness
                    time.sleep(0.02)  # Reduced delay for better performance
                    
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}")
                    time.sleep(0.1)
                        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            should_quit = True
            
        finally:
            # Clear projector pattern before closing
            logger.info("🔵 DEBUG: Clearing projector pattern on exit...")
            clear_projector_pattern(client)
            
            logger.info("Closing preview window...")
            cv2.destroyAllWindows()
            time.sleep(0.5)  # Give time for window to close
    
    else:
        # Traditional mode without live preview
        for position in range(args.num_positions):
            print(f"\\n{'='*60}")
            print(f"CALIBRATION POSITION {position + 1}/{args.num_positions}")
            print(f"{'='*60}")
            print("1. Place checkerboard in a new position")
            print("2. Ensure checkerboard is flat and well-lit")
            print("3. Press Enter when ready to capture Gray Code sequence...")
            input()
            
            # Show white field for checkerboard verification
            client.projector.show_solid_field("White")
            time.sleep(0.5)
            
            # Verify checkerboard is visible
            try:
                images = client.camera.capture_multi([camera_id])
                if images and camera_id in images and images[camera_id] is not None:
                    gray = cv2.cvtColor(images[camera_id], cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(gray, checkerboard_size)
                    
                    if found:
                        print(f"✅ Checkerboard detected at position {position+1}")
                        print(f"Capturing Gray Code sequence ({len(gray_code_patterns)} patterns)...")
                        
                        # Capture Gray Code sequence with enhanced manual control
                        print("\n🕰️ STARTING GRAY CODE PATTERN SEQUENCE")
                        print("="*60)
                        print("📋 Enhanced features active:")
                        print("  ✅ Manual pattern control (wait for your input)")
                        print("  ✅ Extra large patterns (800px minimum, 64x thicker)")
                        print("  ✅ Optimized capture performance")
                        print("  🔍 Each pattern will be clearly visible on projector")
                        print("="*60)
                        
                        captured_images, detected_corners = capture_gray_code_sequence(
                            client, camera_id, gray_code_patterns, checkerboard_size, args.led_intensity, auto_capture=True
                        )
                        
                        if len(captured_images) == len(gray_code_patterns):
                            # Decode projector coordinates
                            projector_points = decode_projector_coordinates(
                                captured_images,
                                corners,
                                args.projector_width,
                                args.projector_height
                            )
                            
                            # Store calibration data
                            calibration_positions.append({
                                'camera_corners': corners,
                                'projector_corners': projector_points,
                                'captured_images': captured_images
                            })
                            
                            print(f"✅ Position {position+1}: Gray Code sequence captured successfully!")
                            print(f"   Decoded {len(projector_points)} projector correspondences")
                        else:
                            print(f"❌ Position {position+1}: Failed to capture all patterns")
                    else:
                        print(f"⚠️ Position {position+1}: Checkerboard not detected")
                else:
                    print(f"❌ Position {position+1}: Failed to capture verification image")
                    
            except Exception as e:
                logger.error(f"Error capturing Gray Code sequence: {e}")
                print(f"❌ Position {position+1}: Capture error")
            
            # Clear projector
            clear_projector_pattern(client)
    
    # Disconnect and cleanup
    try:
        # Turn off LED
        client.projector.led_off()
        logger.info("LED turned off")
        
        # Disconnect
        client.disconnect()
        logger.info("Disconnected from scanner")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    
    if len(calibration_positions) < 3:
        logger.error(f"Not enough calibration positions: {len(calibration_positions)} (minimum 3 required)")
        return 1
    
    # Perform calibration
    logger.info(f"Performing projector-camera calibration with {len(calibration_positions)} positions...")
    
    try:
        # Extract data for calibration
        all_camera_corners = []
        all_projector_corners = []
        object_points = []
        
        # Generate object points (3D coordinates of checkerboard corners)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * args.square_size
        
        for position_data in calibration_positions:
            all_camera_corners.append(position_data['camera_corners'])
            all_projector_corners.append(position_data['projector_corners'])
            object_points.append(objp)
        
        # Get camera resolution from first captured image
        first_image = calibration_positions[0]['captured_images'][0]
        camera_resolution = (first_image.shape[1], first_image.shape[0])
        
        # Load camera calibration if provided, otherwise calibrate
        if args.camera_calibration and args.camera_calibration.exists():
            logger.info(f"Loading camera calibration from {args.camera_calibration}")
            with open(args.camera_calibration, 'r') as f:
                cam_calib = json.load(f)
            
            # Extract camera parameters
            camera_matrix = np.array(cam_calib['camera_matrix'])
            camera_dist = np.array(cam_calib['distortion_coefficients']).reshape(-1)
            
            # Verify resolution matches
            calib_res = tuple(cam_calib['image_size'])
            if calib_res != camera_resolution:
                logger.warning(f"Camera resolution mismatch! Calibration: {calib_res}, Current: {camera_resolution}")
            
            logger.info(f"Loaded camera intrinsics: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            logger.info(f"Camera principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            ret_cam = cam_calib.get('rms_reprojection_error', 0.0)
            
            # Calculate camera extrinsics for the current positions
            rvecs_cam = []
            tvecs_cam = []
            for corners in all_camera_corners:
                ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, camera_dist)
                if ret:
                    rvecs_cam.append(rvec)
                    tvecs_cam.append(tvec)
        else:
            # Calibrate camera from scratch
            logger.info("Calibrating camera...")
            ret_cam, camera_matrix, camera_dist, rvecs_cam, tvecs_cam = cv2.calibrateCamera(
                object_points, all_camera_corners, camera_resolution, None, None
            )
            logger.info(f"Camera calibration RMS error: {ret_cam:.3f}")
        
        # Calibrate projector as inverse camera
        logger.info("Calibrating projector as inverse camera...")
        projector_resolution = (args.projector_width, args.projector_height)
        
        # Initialize projector intrinsics with reasonable guess
        # Projector typically has different FOV than camera
        proj_focal_length = projector_resolution[0] * 0.8  # Initial guess
        projector_matrix_init = np.array([
            [proj_focal_length, 0, projector_resolution[0] / 2],
            [0, proj_focal_length, projector_resolution[1] / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # First try calibration with initial guess
        ret_proj, projector_matrix, projector_dist, rvecs_proj, tvecs_proj = cv2.calibrateCamera(
            object_points, all_projector_corners, projector_resolution, 
            projector_matrix_init, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        logger.info(f"Projector calibration RMS error: {ret_proj:.3f}")
        
        # If RMS is too high, try different strategies
        if ret_proj > 50.0:
            logger.warning(f"High projector RMS error ({ret_proj:.1f}), trying alternative calibration...")
            
            # Try with different initial focal lengths
            for focal_factor in [0.6, 0.7, 0.9, 1.0, 1.1]:
                proj_focal = projector_resolution[0] * focal_factor
                projector_matrix_init[0, 0] = proj_focal
                projector_matrix_init[1, 1] = proj_focal
                
                ret_proj_alt, proj_matrix_alt, proj_dist_alt, rvecs_alt, tvecs_alt = cv2.calibrateCamera(
                    object_points, all_projector_corners, projector_resolution,
                    projector_matrix_init, None,
                    flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
                )
                
                logger.info(f"  Focal factor {focal_factor}: RMS = {ret_proj_alt:.3f}")
                
                if ret_proj_alt < ret_proj:
                    ret_proj = ret_proj_alt
                    projector_matrix = proj_matrix_alt
                    projector_dist = proj_dist_alt
                    rvecs_proj = rvecs_alt
                    tvecs_proj = tvecs_alt
        
        logger.info(f"Final projector calibration RMS error: {ret_proj:.3f}")
        
        # Stereo calibration between camera and projector
        logger.info("Performing stereo calibration...")
        
        # Choose calibration flags based on whether we have fixed camera intrinsics
        if args.camera_calibration and args.camera_calibration.exists():
            # Fix camera intrinsics, only optimize projector and extrinsics
            stereo_flags = cv2.CALIB_FIX_INTRINSIC
            logger.info("Using FIXED camera intrinsics from loaded calibration")
        else:
            # Refine both camera and projector intrinsics
            stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
            logger.info("Refining both camera and projector intrinsics")
        
        ret_stereo, camera_matrix_stereo, camera_dist_stereo, projector_matrix_stereo, projector_dist_stereo, R, T, E, F = cv2.stereoCalibrate(
            object_points, all_camera_corners, all_projector_corners,
            camera_matrix, camera_dist,
            projector_matrix, projector_dist,
            camera_resolution,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=stereo_flags
        )
        logger.info(f"Stereo calibration RMS error: {ret_stereo:.3f}")
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix, camera_dist,
            projector_matrix, projector_dist,
            camera_resolution, R, T,
            alpha=0
        )
        
        # Save calibration results
        calibration_data = {
            'camera_resolution': list(camera_resolution),
            'projector_resolution': [args.projector_width, args.projector_height],
            'checkerboard_size': list(checkerboard_size),
            'square_size_mm': args.square_size,
            'camera_matrix': camera_matrix_stereo.tolist(),  # Use stereo-refined values
            'camera_distortion': camera_dist_stereo.tolist(),
            'projector_matrix': projector_matrix_stereo.tolist(),
            'projector_distortion': projector_dist_stereo.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'rectification_R1': R1.tolist(),
            'rectification_R2': R2.tolist(),
            'rectification_P1': P1.tolist(),
            'rectification_P2': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'num_calibration_positions': len(calibration_positions),
            'camera_rms_error': ret_cam,
            'projector_rms_error': ret_proj,
            'stereo_rms_error': ret_stereo,
            'calibration_method': 'gray_code_projector_camera',
            'camera_calibration_source': str(args.camera_calibration) if args.camera_calibration else 'computed',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {args.output}")
        
        # Print calibration results
        print(f"\\n{'='*60}")
        print("PROJECTOR-CAMERA CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"Camera intrinsics (stereo-refined):")
        print(f"  fx: {camera_matrix_stereo[0,0]:.2f}")
        print(f"  fy: {camera_matrix_stereo[1,1]:.2f}")
        print(f"  cx: {camera_matrix_stereo[0,2]:.2f}")
        print(f"  cy: {camera_matrix_stereo[1,2]:.2f}")
        print(f"  RMS error: {ret_cam:.3f}")
        if args.camera_calibration:
            print(f"  (Loaded from: {args.camera_calibration})")
        print(f"\\nProjector intrinsics (stereo-refined):")
        print(f"  fx: {projector_matrix_stereo[0,0]:.2f}")
        print(f"  fy: {projector_matrix_stereo[1,1]:.2f}")
        print(f"  cx: {projector_matrix_stereo[0,2]:.2f}")
        print(f"  cy: {projector_matrix_stereo[1,2]:.2f}")
        print(f"  RMS error: {ret_proj:.3f}")
        print(f"\\nStereo calibration:")
        print(f"  Baseline (mm): {np.linalg.norm(T):.2f}")
        print(f"  Stereo RMS error: {ret_stereo:.3f}")
        print(f"\\nCalibration positions: {len(calibration_positions)}")
        print(f"Output file: {args.output}")
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
    camera_images = load_calibration_images(args.input)
    
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
            'calibration_method': 'camera_only',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {args.output}")
        
        # Print results
        print(f"\\n{'='*60}")
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


def process_captured_calibration(args):
    """Process captured calibration images offline."""
    logger.info("🔧 PROCESSING CAPTURED CALIBRATION IMAGES")
    logger.info("="*60)
    
    # Parse checkerboard size
    try:
        cb_width, cb_height = map(int, args.checkerboard_size.split('x'))
        checkerboard_size = (cb_width, cb_height)
    except ValueError:
        logger.error(f"Invalid checkerboard size: {args.checkerboard_size}")
        return 1
    
    logger.info(f"Checkerboard size: {checkerboard_size}")
    logger.info(f"Square size: {args.square_size}mm")
    logger.info(f"Projector resolution: {args.projector_width}x{args.projector_height}")
    logger.info(f"Input directory: {args.process_captured}")
    
    # Load captured calibration data
    try:
        calibration_positions = load_captured_calibration_data(
            args.process_captured, 
            checkerboard_size,
            args.projector_width,
            args.projector_height,
            debug=True
        )
    except Exception as e:
        logger.error(f"Failed to load captured data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if len(calibration_positions) < 3:
        logger.error(f"Not enough valid calibration positions: {len(calibration_positions)} (minimum 3 required)")
        return 1
    
    logger.info(f"🎯 Successfully loaded {len(calibration_positions)} calibration positions")
    
    # Perform calibration using the same logic as interactive mode
    logger.info(f"Performing projector-camera calibration with {len(calibration_positions)} positions...")
    
    try:
        # Extract data for calibration
        all_camera_corners = []
        all_projector_corners = []
        object_points = []
        
        # Generate object points (3D coordinates of checkerboard corners)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * args.square_size
        
        for position_data in calibration_positions:
            all_camera_corners.append(position_data['camera_corners'])
            all_projector_corners.append(position_data['projector_corners'])
            object_points.append(objp)
            
            # Debug: Show some stats about this position
            pos_idx = position_data['position_index']
            num_corners = len(position_data['camera_corners'])
            logger.info(f"  Position {pos_idx}: {num_corners} corners")
        
        # Get camera resolution from first captured image
        first_image = calibration_positions[0]['captured_images'][0]
        camera_resolution = (first_image.shape[1], first_image.shape[0])
        logger.info(f"Camera resolution: {camera_resolution}")
        
        # Load camera calibration if provided, otherwise calibrate
        if args.camera_calibration and args.camera_calibration.exists():
            logger.info(f"Loading camera calibration from {args.camera_calibration}")
            with open(args.camera_calibration, 'r') as f:
                cam_calib = json.load(f)
            
            # Extract camera parameters
            camera_matrix = np.array(cam_calib['camera_matrix'])
            camera_dist = np.array(cam_calib['distortion_coefficients']).reshape(-1)
            
            # Verify resolution matches
            calib_res = tuple(cam_calib['image_size'])
            if calib_res != camera_resolution:
                logger.warning(f"Camera resolution mismatch! Calibration: {calib_res}, Current: {camera_resolution}")
            
            logger.info(f"Loaded camera intrinsics: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            logger.info(f"Camera principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            ret_cam = cam_calib.get('rms_reprojection_error', 0.0)
            
            # Calculate camera extrinsics for the current positions
            rvecs_cam = []
            tvecs_cam = []
            for corners in all_camera_corners:
                ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, camera_dist)
                if ret:
                    rvecs_cam.append(rvec)
                    tvecs_cam.append(tvec)
        else:
            # Calibrate camera from scratch
            logger.info("Calibrating camera...")
            ret_cam, camera_matrix, camera_dist, rvecs_cam, tvecs_cam = cv2.calibrateCamera(
                object_points, all_camera_corners, camera_resolution, None, None
            )
            logger.info(f"Camera calibration RMS error: {ret_cam:.3f}")
        
        # Debug projector coordinate ranges before calibration
        logger.info("🔍 ANALYZING PROJECTOR COORDINATES:")
        for i, proj_corners in enumerate(all_projector_corners):
            x_coords = proj_corners[:, 0]
            y_coords = proj_corners[:, 1]
            logger.info(f"  Position {i+1}: X=[{np.min(x_coords):.1f}, {np.max(x_coords):.1f}], Y=[{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
            
            # Check for suspicious coordinates
            if np.max(x_coords) > args.projector_width * 1.5 or np.max(y_coords) > args.projector_height * 1.5:
                logger.warning(f"  ⚠️ Position {i+1} has coordinates outside expected projector range!")
            if np.min(x_coords) < -args.projector_width * 0.5 or np.min(y_coords) < -args.projector_height * 0.5:
                logger.warning(f"  ⚠️ Position {i+1} has negative coordinates!")
        
        # Calibrate projector as inverse camera with enhanced logic
        logger.info("Calibrating projector as inverse camera...")
        projector_resolution = (args.projector_width, args.projector_height)
        
        # Initialize projector intrinsics with reasonable guess
        proj_focal_length = projector_resolution[0] * 0.8  # Initial guess
        projector_matrix_init = np.array([
            [proj_focal_length, 0, projector_resolution[0] / 2],
            [0, proj_focal_length, projector_resolution[1] / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # First try calibration with initial guess
        ret_proj, projector_matrix, projector_dist, rvecs_proj, tvecs_proj = cv2.calibrateCamera(
            object_points, all_projector_corners, projector_resolution, 
            projector_matrix_init, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        logger.info(f"Projector calibration RMS error: {ret_proj:.3f}")
        
        # If RMS is too high, try different strategies
        if ret_proj > 50.0:
            logger.warning(f"High projector RMS error ({ret_proj:.1f}), trying alternative calibration...")
            
            # Try with different initial focal lengths
            for focal_factor in [0.6, 0.7, 0.9, 1.0, 1.1]:
                proj_focal = projector_resolution[0] * focal_factor
                projector_matrix_init[0, 0] = proj_focal
                projector_matrix_init[1, 1] = proj_focal
                
                ret_proj_alt, proj_matrix_alt, proj_dist_alt, rvecs_alt, tvecs_alt = cv2.calibrateCamera(
                    object_points, all_projector_corners, projector_resolution,
                    projector_matrix_init, None,
                    flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
                )
                
                logger.info(f"  Focal factor {focal_factor}: RMS = {ret_proj_alt:.3f}")
                
                if ret_proj_alt < ret_proj:
                    ret_proj = ret_proj_alt
                    projector_matrix = proj_matrix_alt
                    projector_dist = proj_dist_alt
                    rvecs_proj = rvecs_alt
                    tvecs_proj = tvecs_alt
        
        logger.info(f"Final projector calibration RMS error: {ret_proj:.3f}")
        
        # Stereo calibration between camera and projector
        logger.info("Performing stereo calibration...")
        
        # Choose calibration flags based on whether we have fixed camera intrinsics
        if args.camera_calibration and args.camera_calibration.exists():
            # Fix camera intrinsics, only optimize projector and extrinsics
            stereo_flags = cv2.CALIB_FIX_INTRINSIC
            logger.info("Using FIXED camera intrinsics from loaded calibration")
        else:
            # Refine both camera and projector intrinsics
            stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
            logger.info("Refining both camera and projector intrinsics")
        
        ret_stereo, camera_matrix_stereo, camera_dist_stereo, projector_matrix_stereo, projector_dist_stereo, R, T, E, F = cv2.stereoCalibrate(
            object_points, all_camera_corners, all_projector_corners,
            camera_matrix, camera_dist,
            projector_matrix, projector_dist,
            camera_resolution,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=stereo_flags
        )
        logger.info(f"Stereo calibration RMS error: {ret_stereo:.3f}")
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_stereo, camera_dist_stereo,
            projector_matrix_stereo, projector_dist_stereo,
            camera_resolution, R, T,
            alpha=0
        )
        
        # Save calibration results
        calibration_data = {
            'camera_resolution': list(camera_resolution),
            'projector_resolution': [args.projector_width, args.projector_height],
            'checkerboard_size': list(checkerboard_size),
            'square_size_mm': args.square_size,
            'camera_matrix': camera_matrix_stereo.tolist(),  # Use stereo-refined values
            'camera_distortion': camera_dist_stereo.tolist(),
            'projector_matrix': projector_matrix_stereo.tolist(),
            'projector_distortion': projector_dist_stereo.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'rectification_R1': R1.tolist(),
            'rectification_R2': R2.tolist(),
            'rectification_P1': P1.tolist(),
            'rectification_P2': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'num_calibration_positions': len(calibration_positions),
            'camera_rms_error': ret_cam,
            'projector_rms_error': ret_proj,
            'stereo_rms_error': ret_stereo,
            'calibration_method': 'gray_code_projector_camera_offline',
            'camera_calibration_source': str(args.camera_calibration) if args.camera_calibration else 'computed',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {args.output}")
        
        # Print calibration results
        print(f"\\n{'='*60}")
        print("PROJECTOR-CAMERA CALIBRATION RESULTS (OFFLINE)")
        print(f"{'='*60}")
        print(f"Camera intrinsics (stereo-refined):")
        print(f"  fx: {camera_matrix_stereo[0,0]:.2f}")
        print(f"  fy: {camera_matrix_stereo[1,1]:.2f}")
        print(f"  cx: {camera_matrix_stereo[0,2]:.2f}")
        print(f"  cy: {camera_matrix_stereo[1,2]:.2f}")
        print(f"  RMS error: {ret_cam:.3f}")
        if args.camera_calibration:
            print(f"  (Loaded from: {args.camera_calibration})")
        print(f"\\nProjector intrinsics (stereo-refined):")
        print(f"  fx: {projector_matrix_stereo[0,0]:.2f}")
        print(f"  fy: {projector_matrix_stereo[1,1]:.2f}")
        print(f"  cx: {projector_matrix_stereo[0,2]:.2f}")
        print(f"  cy: {projector_matrix_stereo[1,2]:.2f}")
        print(f"  RMS error: {ret_proj:.3f}")
        print(f"\\nStereo calibration:")
        print(f"  Baseline (mm): {np.linalg.norm(T):.2f}")
        print(f"  Stereo RMS error: {ret_stereo:.3f}")
        print(f"\\nCalibration positions: {len(calibration_positions)}")
        print(f"Output file: {args.output}")
        print(f"Debug images saved to: {args.process_captured}/debug_*")
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
        logging.getLogger('__main__').setLevel(logging.DEBUG)
    
    # Check mode
    if args.process_captured:
        return process_captured_calibration(args)
    elif args.interactive:
        return interactive_calibration(args)
    elif args.input:
        return process_existing_images(args)
    else:
        print("Error: Must specify either --process-captured, --interactive, or --input")
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())