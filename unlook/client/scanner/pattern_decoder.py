"""
Pattern decoding utilities for structured light 3D scanning.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class PatternDecoder:
    """Handles decoding of structured light patterns."""
    
    @staticmethod
    def decode_gray_code(
        images: List[np.ndarray],
        pattern_width: int,
        pattern_height: int,
        threshold: float = 5.0,
        debug_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to get projector coordinates.
        
        Args:
            images: List of captured images with Gray code patterns
            pattern_width: Width of projected patterns
            pattern_height: Height of projected patterns
            threshold: Threshold for binary decoding
            debug_dir: Directory to save debug images
            
        Returns:
            Tuple of (x_coords, y_coords, mask) where mask indicates valid pixels
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images (white and black reference)")
        
        h, w = images[0].shape[:2]
        
        # Get white and black reference images
        white_img = images[0].astype(np.float32)
        black_img = images[1].astype(np.float32)
        
        # Calculate threshold image
        thresh_img = (white_img - black_img) * 0.5
        
        # Create mask for valid pixels
        mask = (white_img - black_img) > threshold
        
        # Initialize coordinate maps
        x_coords = np.zeros((h, w), dtype=np.float32)
        y_coords = np.zeros((h, w), dtype=np.float32)
        
        # Decode horizontal patterns (for X coordinates)
        num_x_bits = int(np.log2(pattern_width))
        x_bits = []
        
        for i in range(num_x_bits):
            if 2 + i * 2 + 1 >= len(images):
                logger.warning(f"Not enough images for {num_x_bits} horizontal bits")
                break
                
            # Get positive and inverted patterns
            pos_img = images[2 + i * 2].astype(np.float32)
            inv_img = images[2 + i * 2 + 1].astype(np.float32)
            
            # Decode bit
            bit_value = ((pos_img - black_img) > (inv_img - black_img)).astype(np.uint8)
            x_bits.append(bit_value)
        
        # Convert Gray code to binary
        if x_bits:
            x_binary = x_bits[0].copy()
            for i in range(1, len(x_bits)):
                x_binary = x_binary ^ x_bits[i]
                x_coords += x_binary * (pattern_width / (2 ** (i + 1)))
            x_coords += x_bits[-1] * (pattern_width / (2 ** len(x_bits)))
        
        # Decode vertical patterns (for Y coordinates)
        num_y_bits = int(np.log2(pattern_height))
        y_bits = []
        
        start_idx = 2 + num_x_bits * 2
        for i in range(num_y_bits):
            if start_idx + i * 2 + 1 >= len(images):
                logger.warning(f"Not enough images for {num_y_bits} vertical bits")
                break
                
            # Get positive and inverted patterns
            pos_img = images[start_idx + i * 2].astype(np.float32)
            inv_img = images[start_idx + i * 2 + 1].astype(np.float32)
            
            # Decode bit
            bit_value = ((pos_img - black_img) > (inv_img - black_img)).astype(np.uint8)
            y_bits.append(bit_value)
        
        # Convert Gray code to binary
        if y_bits:
            y_binary = y_bits[0].copy()
            for i in range(1, len(y_bits)):
                y_binary = y_binary ^ y_bits[i]
                y_coords += y_binary * (pattern_height / (2 ** (i + 1)))
            y_coords += y_bits[-1] * (pattern_height / (2 ** len(y_bits)))
        
        # Apply mask to coordinates
        x_coords[~mask] = -1
        y_coords[~mask] = -1
        
        # Save debug images if requested
        if debug_dir:
            PatternDecoder._save_debug_images(
                x_coords, y_coords, mask, thresh_img, debug_dir
            )
        
        return x_coords, y_coords, mask
    
    @staticmethod
    def decode_phase_shift(
        images: List[np.ndarray],
        num_shifts: int = 4,
        threshold: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode phase shift patterns.
        
        Args:
            images: List of phase-shifted sinusoidal patterns
            num_shifts: Number of phase shifts
            threshold: Minimum modulation threshold
            
        Returns:
            Tuple of (phase_map, modulation_map)
        """
        if len(images) < num_shifts:
            raise ValueError(f"Need at least {num_shifts} images for phase shift decoding")
        
        h, w = images[0].shape[:2]
        
        # Initialize arrays
        numerator = np.zeros((h, w), dtype=np.float32)
        denominator = np.zeros((h, w), dtype=np.float32)
        
        # Calculate phase using N-step phase shifting algorithm
        for i in range(num_shifts):
            phase = 2 * np.pi * i / num_shifts
            img = images[i].astype(np.float32)
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            numerator += img * np.sin(phase)
            denominator += img * np.cos(phase)
        
        # Calculate phase map
        phase_map = np.arctan2(numerator, denominator)
        
        # Calculate modulation (quality metric)
        modulation = np.sqrt(numerator**2 + denominator**2) * 2 / num_shifts
        
        # Create mask based on modulation threshold
        mask = modulation > threshold
        phase_map[~mask] = -np.pi
        
        return phase_map, modulation
    
    @staticmethod
    def unwrap_phase(
        phase_map: np.ndarray,
        gray_code_x: Optional[np.ndarray] = None,
        gray_code_y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Unwrap phase map using Gray code as guide.
        
        Args:
            phase_map: Wrapped phase map (-pi to pi)
            gray_code_x: X coordinates from Gray code (optional)
            gray_code_y: Y coordinates from Gray code (optional)
            
        Returns:
            Unwrapped phase map
        """
        h, w = phase_map.shape
        unwrapped = np.zeros_like(phase_map)
        
        # Simple spatial unwrapping if no Gray code available
        if gray_code_x is None and gray_code_y is None:
            # Use OpenCV phase unwrapping or simple row-by-row unwrapping
            for y in range(h):
                unwrapped[y, :] = np.unwrap(phase_map[y, :])
            return unwrapped
        
        # Use Gray code to guide unwrapping
        if gray_code_x is not None:
            # Determine period from Gray code
            period_x = w / np.max(gray_code_x[gray_code_x > 0])
            
            # Unwrap using Gray code as guide
            k = np.round(gray_code_x / period_x)
            unwrapped = phase_map + 2 * np.pi * k
        
        return unwrapped
    
    @staticmethod
    def _save_debug_images(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        mask: np.ndarray,
        threshold_img: np.ndarray,
        debug_dir: str
    ):
        """Save debug images for pattern decoding."""
        import os
        os.makedirs(debug_dir, exist_ok=True)
        
        # Normalize coordinates for visualization
        x_vis = np.zeros_like(x_coords, dtype=np.uint8)
        y_vis = np.zeros_like(y_coords, dtype=np.uint8)
        
        valid_mask = mask & (x_coords >= 0) & (y_coords >= 0)
        if np.any(valid_mask):
            x_vis[valid_mask] = (x_coords[valid_mask] / np.max(x_coords[valid_mask]) * 255).astype(np.uint8)
            y_vis[valid_mask] = (y_coords[valid_mask] / np.max(y_coords[valid_mask]) * 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(os.path.join(debug_dir, "x_coordinates.png"), x_vis)
        cv2.imwrite(os.path.join(debug_dir, "y_coordinates.png"), y_vis)
        cv2.imwrite(os.path.join(debug_dir, "valid_mask.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(debug_dir, "threshold.png"), 
                    (threshold_img / np.max(threshold_img) * 255).astype(np.uint8))
        
        # Create color-coded coordinate map
        coords_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        coords_color[:, :, 0] = x_vis  # Red channel for X
        coords_color[:, :, 1] = y_vis  # Green channel for Y
        coords_color[:, :, 2] = mask.astype(np.uint8) * 255  # Blue channel for mask
        
        cv2.imwrite(os.path.join(debug_dir, "coordinates_color.png"), coords_color)
        
        logger.info(f"Saved debug images to {debug_dir}")