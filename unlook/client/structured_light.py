"""
Advanced structured light scanning module for UnLook SDK.
Provides utilities for structured light pattern generation and decoding.
"""

import os
import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import time

logger = logging.getLogger(__name__)

class GrayCodeGenerator:
    """
    Gray code pattern generator for structured light scanning.
    Implements the methods from the Structured-light-stereo repository.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, white_threshold: int = 5):
        """
        Initialize the Gray code generator.
        
        Args:
            width: Projector width
            height: Projector height
            white_threshold: Threshold for white detection during decoding
        """
        self.proj_w = width
        self.proj_h = height
        self.white_threshold = white_threshold
        self.black_threshold = 40  # Threshold for shadow detection
        
        # Create OpenCV Gray code pattern generator
        self.graycode = cv2.structured_light.GrayCodePattern.create(width=self.proj_w, height=self.proj_h)
        self.graycode.setWhiteThreshold(self.white_threshold)
        
        # Store number of required images
        self.num_required_imgs = self.graycode.getNumberOfPatternImages()
        logger.info(f"Gray code requires {self.num_required_imgs} pattern images")
        
    def generate_pattern_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate a complete Gray code pattern sequence.
        
        Returns:
            List of pattern dictionaries for the projector
        """
        patterns = []
        
        # Get black and white images
        _, white_image = self.graycode.getWhiteImage()
        _, black_image = self.graycode.getBlackImage()
        
        # Convert OpenCV images to patterns
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._process_cv_image(white_image),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._process_cv_image(black_image),
            "name": "black"
        })
        
        # Get Gray code patterns
        pattern_cv_images = []
        self.graycode.getImagesForProjection(pattern_cv_images)
        
        # Add each pattern to the sequence
        for i, pattern_image in enumerate(pattern_cv_images):
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._process_cv_image(pattern_image),
                "name": f"gray_code_{i}"
            })
            
        logger.info(f"Generated {len(patterns)} Gray code patterns")
        return patterns
    
    def _process_cv_image(self, cv_image: np.ndarray) -> bytes:
        """
        Convert an OpenCV pattern image to a compatible format.
        
        Args:
            cv_image: OpenCV image
            
        Returns:
            JPEG encoded image
        """
        # Ensure image is 8-bit
        if cv_image.dtype != np.uint8:
            cv_image = cv_image.astype(np.uint8)
            
        # Encode to JPEG
        success, jpeg_data = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            return jpeg_data.tobytes()
        else:
            logger.error("Failed to encode pattern image")
            return b''
            
    def decode_pattern_images(self, pattern_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode a set of captured images to generate projector-camera correspondence.
        
        Args:
            pattern_images: List of captured images (including white, black and Gray code patterns)
            
        Returns:
            Tuple of (cam_proj, shadow_mask)
        """
        if len(pattern_images) < self.num_required_imgs + 2:
            logger.error(f"Not enough pattern images: {len(pattern_images)} (need {self.num_required_imgs + 2})")
            return None, None
            
        # Extract white, black and pattern images
        white_img = pattern_images[-2]
        black_img = pattern_images[-1]
        gray_patterns = pattern_images[:-2]
        
        # Compute shadow mask (areas occluded from projector)
        shadow_mask = self._compute_shadow_mask(black_img, white_img, self.black_threshold)
        
        # Decode Gray code patterns
        img_h, img_w = pattern_images[0].shape[:2]
        cam_proj = np.zeros((img_h, img_w, 2), dtype=np.float32)
        
        # Process each pixel
        for y in range(img_h):
            for x in range(img_w):
                # Skip shadowed pixels
                if shadow_mask[y, x] == 0:
                    continue
                    
                # Get projector pixel corresponding to this camera pixel
                error, proj_pixel = self.graycode.getProjPixel(gray_patterns, x, y)
                
                if not error:
                    # Store projection mapping (projector v, u)
                    cam_proj[y, x, 0] = proj_pixel[1]  # v (row)
                    cam_proj[y, x, 1] = proj_pixel[0]  # u (column)
        
        logger.info(f"Decoded {np.count_nonzero(cam_proj[:,:,0] > 0)} correspondence points")
        return cam_proj, shadow_mask
    
    @staticmethod
    def _compute_shadow_mask(black_img: np.ndarray, white_img: np.ndarray, threshold: int) -> np.ndarray:
        """
        Compute a shadow mask to identify areas that are not illuminated by the projector.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            threshold: Brightness difference threshold
            
        Returns:
            Binary mask where 0 indicates shadowed areas
        """
        shadow_mask = np.zeros_like(black_img)
        shadow_mask[white_img > black_img + threshold] = 1
        return shadow_mask
        
    @staticmethod
    def generate_point_cloud(cam_l_proj: np.ndarray, cam_r_proj: np.ndarray, 
                           P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Generate a point cloud from camera-projector correspondences.
        
        Args:
            cam_l_proj: Left camera-projector correspondence
            cam_r_proj: Right camera-projector correspondence
            P1: Left camera projection matrix
            P2: Right camera projection matrix
            
        Returns:
            3D point cloud
        """
        # Find pixel correspondence between cameras
        pts_l = []
        pts_r = []
        
        # Create projector-to-camera map for right camera
        img_h, img_w = cam_r_proj.shape[:2]
        proj_cam_r = np.zeros((1080, 1920, 2))  # Assuming projector is 1920x1080
        
        # Fill projector-to-camera map
        for y in range(img_h):
            for x in range(img_w):
                if cam_r_proj[y, x, 0] > 0 and cam_r_proj[y, x, 1] > 0:
                    proj_y = int(cam_r_proj[y, x, 0])
                    proj_x = int(cam_r_proj[y, x, 1])
                    
                    if 0 <= proj_y < 1080 and 0 <= proj_x < 1920:
                        proj_cam_r[proj_y, proj_x, 0] = y
                        proj_cam_r[proj_y, proj_x, 1] = x
        
        # Find correspondence points
        for y in range(img_h):
            for x in range(img_w):
                if cam_l_proj[y, x, 0] > 0 and cam_l_proj[y, x, 1] > 0:
                    proj_y = int(cam_l_proj[y, x, 0])
                    proj_x = int(cam_l_proj[y, x, 1])
                    
                    if 0 <= proj_y < 1080 and 0 <= proj_x < 1920:
                        cam_r_y = int(proj_cam_r[proj_y, proj_x, 0])
                        cam_r_x = int(proj_cam_r[proj_y, proj_x, 1])
                        
                        if cam_r_y > 0 and cam_r_x > 0:
                            pts_l.append([x, y])
                            pts_r.append([cam_r_x, cam_r_y])
        
        # Convert to numpy arrays for triangulation
        pts_l = np.array(pts_l)[:, np.newaxis, :]
        pts_r = np.array(pts_r)[:, np.newaxis, :]
        
        # Triangulate points
        pts4D = cv2.triangulatePoints(P1, P2, np.float32(pts_l), np.float32(pts_r)).T
        pts3D = pts4D[:, :3] / pts4D[:, -1:]
        
        return pts3D


class PhaseShiftPatternGenerator:
    """
    Phase shift pattern generator for structured light scanning.
    Implements sinusoidal phase shift patterns at multiple frequencies.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, phases: int = 4, frequencies: List[int] = None):
        """
        Initialize the phase shift pattern generator.
        
        Args:
            width: Projector width
            height: Projector height
            phases: Number of phase shifts per frequency
            frequencies: List of pattern frequencies (periods)
        """
        self.proj_w = width
        self.proj_h = height
        self.phases = phases
        self.frequencies = frequencies or [16, 32, 64, 128]  # Default frequencies
        
    def generate_pattern_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate a complete phase shift pattern sequence.
        
        Returns:
            List of pattern dictionaries for the projector
        """
        patterns = []
        
        # Add white and black reference patterns
        white_img = np.ones((self.proj_h, self.proj_w), dtype=np.uint8) * 255
        black_img = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(white_img),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(black_img),
            "name": "black"
        })
        
        # Generate horizontal and vertical patterns for each frequency
        directions = ['horizontal', 'vertical']
        for direction in directions:
            for freq in self.frequencies:
                # Generate patterns with different phase shifts
                for phase in range(self.phases):
                    phase_offset = (2 * np.pi * phase) / self.phases
                    
                    # Create the pattern
                    pattern_img = self._generate_phase_pattern(freq, phase_offset, direction)
                    
                    # Add to sequence
                    patterns.append({
                        "pattern_type": "raw_image",
                        "image": self._encode_image(pattern_img),
                        "name": f"{direction}_f{freq}_p{phase}"
                    })
        
        logger.info(f"Generated {len(patterns)} phase shift patterns")
        return patterns
    
    def _generate_phase_pattern(self, frequency: int, phase_offset: float, direction: str) -> np.ndarray:
        """
        Generate a single phase shift pattern.
        
        Args:
            frequency: Pattern frequency (period in pixels)
            phase_offset: Phase shift in radians
            direction: 'horizontal' or 'vertical'
            
        Returns:
            Pattern image
        """
        pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        
        # Create coordinate grid
        x = np.arange(self.proj_w)
        y = np.arange(self.proj_h)
        
        if direction == 'horizontal':
            # Horizontal sinusoidal pattern
            for i in range(self.proj_h):
                # Calculate sinusoidal intensity
                intensity = 0.5 + 0.5 * np.cos((2 * np.pi * x / frequency) + phase_offset)
                pattern[i, :] = (intensity * 255).astype(np.uint8)
        else:
            # Vertical sinusoidal pattern
            for i in range(self.proj_w):
                # Calculate sinusoidal intensity
                intensity = 0.5 + 0.5 * np.cos((2 * np.pi * y / frequency) + phase_offset)
                pattern[:, i] = (intensity * 255).astype(np.uint8)
        
        return pattern
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """
        Encode an image as JPEG.
        
        Args:
            image: Input image
            
        Returns:
            JPEG encoded image
        """
        success, jpeg_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            return jpeg_data.tobytes()
        else:
            logger.error("Failed to encode pattern image")
            return b''