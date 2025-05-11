"""
Enhanced Gray Code Pattern Generation and Decoding.

This module provides improved Gray code pattern generation and decoding
optimized for robust structured light scanning in various environments.
It focuses on reliability rather than speed, with better thresholding and
more robust pattern decoding.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union

# Configure logger
logger = logging.getLogger(__name__)


def generate_enhanced_gray_code_patterns(
    width: int = 1024,
    height: int = 768,
    num_bits: int = 6,
    orientation: str = "horizontal"
) -> List[Dict[str, Any]]:
    """
    Generate a set of enhanced Gray code patterns with improved contrast.
    
    Args:
        width: Pattern width in pixels
        height: Pattern height in pixels
        num_bits: Number of Gray code bits
        orientation: "horizontal" or "vertical"
    
    Returns:
        List of pattern dictionaries
    """
    patterns = []
    
    # Add white and black reference patterns first
    patterns.append({
        "pattern_type": "solid_field",
        "color": "White",
        "name": "white_reference"
    })
    
    patterns.append({
        "pattern_type": "solid_field",
        "color": "Black",
        "name": "black_reference"
    })
    
    # Add Gray code patterns (regular and inverted)
    for bit in range(num_bits):
        # Regular pattern
        patterns.append({
            "pattern_type": "gray_code",
            "orientation": orientation,
            "bit": bit,
            "inverted": False,
            "name": f"gray_{orientation[0]}_bit{bit:02d}"
        })
        
        # Inverted pattern (for robust decoding)
        patterns.append({
            "pattern_type": "gray_code",
            "orientation": orientation,
            "bit": bit,
            "inverted": True,
            "name": f"gray_{orientation[0]}_bit{bit:02d}_inv"
        })
    
    logger.info(f"Generated {len(patterns)} enhanced Gray code patterns ({num_bits} bits, {orientation})")
    return patterns


def binary_to_gray(binary: int, num_bits: int) -> int:
    """
    Convert binary to Gray code.
    
    Args:
        binary: Binary value
        num_bits: Number of bits
    
    Returns:
        Gray code value
    """
    return binary ^ (binary >> 1)


def gray_to_binary(gray: int, num_bits: int) -> int:
    """
    Convert Gray code to binary.
    
    Args:
        gray: Gray code value
        num_bits: Number of bits
    
    Returns:
        Binary value
    """
    binary = 0
    for bit in range(num_bits - 1, -1, -1):
        if (gray >> bit) & 1:
            binary = binary ^ 1
        if bit > 0:
            binary = binary << 1
    return binary


def encode_pattern(
    bit: int, 
    orientation: str,
    inverted: bool,
    width: int,
    height: int
) -> np.ndarray:
    """
    Generate a single Gray code pattern image.
    
    Args:
        bit: Bit position for Gray code
        orientation: "horizontal" or "vertical"
        inverted: Whether pattern is inverted
        width: Pattern width
        height: Pattern height
    
    Returns:
        Pattern image as numpy array
    """
    # Create a blank image
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate stripe width based on bit position
    stripe_width = 2 ** bit
    
    # Fill with Gray code pattern
    if orientation == "horizontal":
        for x in range(width):
            # Use Gray code binary pattern based on position
            if ((x // stripe_width) % 2) == 0:
                img[:, x] = 255 if not inverted else 0
            else:
                img[:, x] = 0 if not inverted else 255
    else:  # vertical
        for y in range(height):
            if ((y // stripe_width) % 2) == 0:
                img[y, :] = 255 if not inverted else 0
            else:
                img[y, :] = 0 if not inverted else 255
    
    return img


def adaptive_threshold(
    white_img: np.ndarray,
    black_img: np.ndarray,
    threshold_multiplier: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an adaptive threshold for pattern decoding.
    
    Args:
        white_img: White reference image
        black_img: Black reference image
        threshold_multiplier: Multiplier for threshold value
    
    Returns:
        Tuple of (shadow_mask, threshold_image)
    """
    # Compute difference between white and black reference
    diff = cv2.absdiff(white_img, black_img)
    
    # Create shadow mask (areas where projector light is visible)
    # A pixel is considered illuminated if the difference between white and black
    # reference images is greater than a threshold
    min_threshold = 15  # Minimum threshold to consider a pixel illuminated
    shadow_mask = np.zeros_like(diff, dtype=np.uint8)
    shadow_mask[diff > min_threshold] = 255
    
    # Enhance the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    
    # Compute adaptive threshold for each pixel
    # This is the midpoint between white and black reference, plus a margin
    threshold = black_img.astype(np.float32) + (diff.astype(np.float32) * threshold_multiplier)
    threshold = threshold.astype(np.uint8)
    
    return shadow_mask, threshold


def decode_patterns(
    white_img: np.ndarray,
    black_img: np.ndarray,
    pattern_images: List[np.ndarray],
    num_bits: int,
    orientation: str = "horizontal",
    threshold_multiplier: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode Gray code patterns to get projector coordinates.
    
    Args:
        white_img: White reference image
        black_img: Black reference image
        pattern_images: List of pattern images (regular and inverted pairs)
        num_bits: Number of Gray code bits
        orientation: "horizontal" or "vertical"
        threshold_multiplier: Multiplier for threshold value
    
    Returns:
        Tuple of (coordinate_map, confidence_map, shadow_mask)
    """
    height, width = white_img.shape[:2]
    
    # Create adaptive threshold
    shadow_mask, threshold = adaptive_threshold(
        white_img, black_img, threshold_multiplier)
    
    # Prepare coordinate and confidence maps
    coordinate_map = np.zeros((height, width), dtype=np.uint16)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    # Pre-allocate bit arrays
    bit_values = np.zeros((height, width, num_bits), dtype=np.uint8)
    bit_confidence = np.zeros((height, width, num_bits), dtype=np.float32)
    
    # Process each bit (pair of normal and inverted patterns)
    for bit in range(num_bits):
        normal_idx = bit * 2
        inverted_idx = bit * 2 + 1
        
        # Get normal and inverted pattern images
        normal_img = pattern_images[normal_idx]
        inverted_img = pattern_images[inverted_idx]
        
        # Calculate differences (robust to global illumination changes)
        normal_diff = cv2.absdiff(normal_img, black_img)
        inverted_diff = cv2.absdiff(inverted_img, black_img)
        
        # Decode bit value using thresholds
        normal_mask = (normal_img > threshold) & (shadow_mask > 0)
        inverted_mask = (inverted_img > threshold) & (shadow_mask > 0)
        
        # Compute bit confidence as the difference between normal and inverted patterns
        bit_diff = cv2.absdiff(normal_img, inverted_img)
        
        # Set bit values and confidence
        bit_values[:, :, bit][normal_mask] = 1
        bit_values[:, :, bit][inverted_mask] = 0
        
        # Higher confidence for larger differences
        bit_confidence[:, :, bit] = bit_diff.astype(np.float32) / 255.0
    
    # Convert bit values to projector coordinates
    for y in range(height):
        for x in range(width):
            if shadow_mask[y, x] > 0:
                # Extract bit pattern for this pixel
                bits = bit_values[y, x]
                
                # Convert Gray code to binary (coordinate)
                gray_code = 0
                for b in range(num_bits):
                    gray_code |= (int(bits[b]) << b)
                
                # Convert Gray code to binary
                binary = gray_to_binary(gray_code, num_bits)
                
                # Store coordinate value
                coordinate_map[y, x] = binary
                
                # Calculate average confidence
                confidence_map[y, x] = np.mean(bit_confidence[y, x])
    
    return coordinate_map, confidence_map, shadow_mask


def create_refined_correspondence_maps(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    epipolar_tolerance: float = 5.0,
    min_disparity: int = 5,
    max_disparity: int = 100
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], np.ndarray]:
    """
    Create refined correspondence maps between left and right images.
    
    Args:
        left_coords: Projector coordinates for left image
        right_coords: Projector coordinates for right image
        left_mask: Mask of valid pixels in left image
        right_mask: Mask of valid pixels in right image
        epipolar_tolerance: Tolerance for epipolar constraint (pixels)
        min_disparity: Minimum disparity value (pixels)
        max_disparity: Maximum disparity value (pixels)
    
    Returns:
        Tuple of (correspondence map, disparity map)
    """
    height, width = left_mask.shape[:2]
    
    # Create empty correspondence and disparity maps
    correspondences = {}
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    # For each valid pixel in the left image
    for y in range(height):
        for x in range(width):
            if not left_mask[y, x]:
                continue
            
            # Get projector coordinate for this pixel
            proj_coord = left_coords[y, x]
            
            if proj_coord == 0:
                continue
            
            # Search along the epipolar line in the right image
            best_match_x = -1
            best_match_y = -1
            best_match_dist = float('inf')
            
            # For each pixel in the same row of the right image (epipolar line)
            for x_right in range(max(0, x - max_disparity), x - min_disparity + 1):
                if x_right < 0 or not right_mask[y, x_right]:
                    continue
                
                # Get projector coordinate in right image
                right_proj_coord = right_coords[y, x_right]
                
                if right_proj_coord == 0:
                    continue
                
                # Compute distance between projector coordinates
                # (they should be the same point in projector space)
                dist = abs(proj_coord - right_proj_coord)
                
                # Update best match if this is better
                if dist < best_match_dist and dist < epipolar_tolerance:
                    best_match_dist = dist
                    best_match_x = x_right
                    best_match_y = y
            
            # If a match was found, add to correspondences
            if best_match_x >= 0:
                correspondences[(x, y)] = (best_match_x, best_match_y)
                disparity = x - best_match_x
                disparity_map[y, x] = disparity
    
    logger.info(f"Found {len(correspondences)} refined correspondences")
    return correspondences, disparity_map


def filter_correspondences(
    correspondences: Dict[Tuple[int, int], Tuple[int, int]],
    disparity_map: np.ndarray,
    method: str = "statistical",
    std_multiplier: float = 2.0
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Filter correspondences to remove outliers.
    
    Args:
        correspondences: Dictionary of correspondences
        disparity_map: Disparity map
        method: Filtering method ("statistical" or "median")
        std_multiplier: Standard deviation multiplier for outlier detection
    
    Returns:
        Filtered correspondences
    """
    if not correspondences:
        return {}
    
    # Get all disparity values
    disparities = []
    for (x, y), _ in correspondences.items():
        disparities.append(disparity_map[y, x])
    
    disparities = np.array(disparities)
    
    # Filter based on method
    if method == "statistical":
        # Statistical outlier removal
        mean_disparity = np.mean(disparities)
        std_disparity = np.std(disparities)
        
        # Define thresholds
        min_threshold = mean_disparity - std_multiplier * std_disparity
        max_threshold = mean_disparity + std_multiplier * std_disparity
        
        # Filter correspondences
        filtered_correspondences = {}
        for (x, y), match in correspondences.items():
            disparity = disparity_map[y, x]
            if min_threshold <= disparity <= max_threshold:
                filtered_correspondences[(x, y)] = match
    
    elif method == "median":
        # Median filter
        median_disparity = np.median(disparities)
        mad = np.median(np.abs(disparities - median_disparity))  # Median Absolute Deviation
        
        # Define thresholds (more robust to outliers than mean/std)
        min_threshold = median_disparity - std_multiplier * mad
        max_threshold = median_disparity + std_multiplier * mad
        
        # Filter correspondences
        filtered_correspondences = {}
        for (x, y), match in correspondences.items():
            disparity = disparity_map[y, x]
            if min_threshold <= disparity <= max_threshold:
                filtered_correspondences[(x, y)] = match
    
    else:
        # No filtering
        filtered_correspondences = correspondences
    
    logger.info(f"Filtered correspondences from {len(correspondences)} to {len(filtered_correspondences)}")
    return filtered_correspondences


def correspondences_to_point_arrays(
    correspondences: Dict[Tuple[int, int], Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert correspondences to point arrays suitable for triangulation.
    
    Args:
        correspondences: Dictionary of correspondences
    
    Returns:
        Tuple of (left_points, right_points) as numpy arrays
    """
    # Convert to arrays for triangulation
    left_points = []
    right_points = []
    
    for (x_left, y_left), (x_right, y_right) in correspondences.items():
        left_points.append([x_left, y_left])
        right_points.append([x_right, y_right])
    
    # Reshape for OpenCV triangulation
    left_points = np.array(left_points).reshape(-1, 1, 2).astype(np.float32)
    right_points = np.array(right_points).reshape(-1, 1, 2).astype(np.float32)
    
    return left_points, right_points


def visualize_decoded_patterns(
    white_img: np.ndarray,
    coordinate_map: np.ndarray,
    confidence_map: np.ndarray,
    shadow_mask: np.ndarray,
    path: str = "debug_output/enhanced_gray_code",
    show_windows: bool = False
) -> bool:
    """
    Create visualization images for debugging.
    
    Args:
        white_img: White reference image
        coordinate_map: Decoded coordinates
        confidence_map: Decoding confidence
        shadow_mask: Shadow mask
        path: Output directory
        show_windows: Whether to show visualization windows
    
    Returns:
        True if successful, False otherwise
    """
    # Create output directory
    os.makedirs(path, exist_ok=True)
    
    # Normalize coordinate map for visualization
    if np.max(coordinate_map) > 0:
        coord_vis = (coordinate_map.astype(np.float32) * 255 / np.max(coordinate_map)).astype(np.uint8)
    else:
        coord_vis = np.zeros_like(coordinate_map, dtype=np.uint8)
    
    # Create confidence visualization (brighter = higher confidence)
    conf_vis = (confidence_map * 255).astype(np.uint8)
    
    # Create color visualization of coordinates
    # Red channel = coordinate value, Green channel = confidence, Blue channel = shadow mask
    color_vis = np.zeros((white_img.shape[0], white_img.shape[1], 3), dtype=np.uint8)
    color_vis[:, :, 0] = coord_vis  # Red = coordinate
    color_vis[:, :, 1] = conf_vis   # Green = confidence
    color_vis[:, :, 2] = shadow_mask  # Blue = shadow mask
    
    # Save visualization images
    try:
        cv2.imwrite(os.path.join(path, "coordinate_map.png"), coord_vis)
        cv2.imwrite(os.path.join(path, "confidence_map.png"), conf_vis)
        cv2.imwrite(os.path.join(path, "shadow_mask.png"), shadow_mask)
        cv2.imwrite(os.path.join(path, "color_visualization.png"), color_vis)
        
        # Display if requested
        if show_windows:
            cv2.imshow("Coordinate Map", coord_vis)
            cv2.imshow("Confidence Map", conf_vis)
            cv2.imshow("Shadow Mask", shadow_mask)
            cv2.imshow("Color Visualization", color_vis)
            cv2.waitKey(1)
        
        return True
    except Exception as e:
        logger.error(f"Error saving visualization images: {e}")
        return False