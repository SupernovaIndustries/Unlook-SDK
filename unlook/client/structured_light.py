"""
Structured Light Module for UnLook SDK.

This module provides a comprehensive implementation of structured light scanning
techniques including Gray code and Phase shift, optimized for robust 3D scanning
in various lighting conditions.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. 3D mesh visualization and processing will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for open3d when not available
    class PlaceholderO3D:
        class geometry:
            class PointCloud:
                pass
            class TriangleMesh:
                pass
        class utility:
            class Vector3dVector:
                pass
        class visualization:
            pass
        class io:
            pass
    o3d = PlaceholderO3D()
    o3dg = PlaceholderO3D.geometry

# We now use Open3D exclusively for point cloud processing
# No need for python-pcl anymore
PCL_AVAILABLE = False  # Legacy flag kept for backward compatibility


class Pattern:
    """Base class for structured light patterns."""
    
    def __init__(self, name: str, pattern_type: str, width: int = 1024, height: int = 768):
        """
        Initialize base pattern.
        
        Args:
            name: Pattern name
            pattern_type: Type of pattern
            width: Pattern width
            height: Pattern height
        """
        self.name = name
        self.pattern_type = pattern_type
        self.width = width
        self.height = height
        self.data = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for projection."""
        raise NotImplementedError("Subclasses must implement to_dict()")


class SolidPattern(Pattern):
    """Solid color field pattern."""
    
    def __init__(self, color: str = "White", width: int = 1024, height: int = 768):
        """
        Initialize solid color pattern.
        
        Args:
            color: Color name (White, Black, Red, etc.)
            width: Pattern width
            height: Pattern height
        """
        super().__init__(name=f"solid_{color.lower()}", pattern_type="solid_field", width=width, height=height)
        self.color = color
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for projection."""
        return {
            "pattern_type": "solid_field",
            "color": self.color,
            "name": self.name
        }


class LinePattern(Pattern):
    """Horizontal or vertical line pattern."""
    
    def __init__(
        self, 
        orientation: str = "horizontal", 
        foreground_color: str = "White",
        background_color: str = "Black",
        foreground_width: int = 4,
        background_width: int = 20,
        width: int = 1024, 
        height: int = 768
    ):
        """
        Initialize line pattern.
        
        Args:
            orientation: "horizontal" or "vertical"
            foreground_color: Line color
            background_color: Background color
            foreground_width: Line width
            background_width: Space between lines
            width: Pattern width
            height: Pattern height
        """
        pattern_type = f"{orientation}_lines"
        name = f"{orientation}_lines_{foreground_color.lower()}"
        super().__init__(name=name, pattern_type=pattern_type, width=width, height=height)
        
        self.orientation = orientation
        self.foreground_color = foreground_color
        self.background_color = background_color
        self.foreground_width = foreground_width
        self.background_width = background_width
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for projection."""
        pattern_type = "horizontal_lines" if self.orientation == "horizontal" else "vertical_lines"
        return {
            "pattern_type": pattern_type,
            "foreground_color": self.foreground_color,
            "background_color": self.background_color,
            "foreground_width": self.foreground_width,
            "background_width": self.background_width,
            "name": self.name
        }


class GrayCodePattern(Pattern):
    """Gray code pattern for structured light scanning."""
    
    def __init__(
        self, 
        bit: int, 
        orientation: str = "horizontal",
        inverted: bool = False,
        width: int = 1024, 
        height: int = 768
    ):
        """
        Initialize Gray code pattern.
        
        Args:
            bit: Bit position for Gray code
            orientation: "horizontal" or "vertical"
            inverted: Whether pattern is inverted
            width: Pattern width
            height: Pattern height
        """
        inv_text = "_inv" if inverted else ""
        name = f"gray_code_{orientation[0]}_bit{bit:02d}{inv_text}"
        super().__init__(name=name, pattern_type="raw_image", width=width, height=height)
        
        self.bit = bit
        self.orientation = orientation
        self.inverted = inverted
        
        # Generate the pattern image
        self._generate_pattern()
    
    def _generate_pattern(self):
        """Generate Gray code pattern image."""
        # Create a blank image
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Calculate stripe width based on bit position
        stripe_width = 2 ** self.bit
        
        # Fill with Gray code pattern
        if self.orientation == "horizontal":
            for x in range(self.width):
                # Use Gray code binary pattern based on position
                if ((x // stripe_width) % 2) == 0:
                    img[:, x] = 255 if not self.inverted else 0
                else:
                    img[:, x] = 0 if not self.inverted else 255
        else:  # vertical
            for y in range(self.height):
                if ((y // stripe_width) % 2) == 0:
                    img[y, :] = 255 if not self.inverted else 0
                else:
                    img[y, :] = 0 if not self.inverted else 255
        
        # Store the raw image data
        success, encoded = cv2.imencode('.png', img)
        if success:
            self.data = encoded.tobytes()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for projection."""
        return {
            "pattern_type": "raw_image",
            "name": self.name,
            "orientation": self.orientation,
            "bit_position": self.bit,
            "is_inverse": self.inverted,
            "image": self.data
        }


class PhaseShiftPattern(Pattern):
    """Phase shift pattern for structured light scanning."""

    def __init__(
        self,
        frequency: int,
        step: int,
        steps_total: int,
        orientation: str = "horizontal",
        width: int = 1024,
        height: int = 768
    ):
        """
        Initialize phase shift pattern.

        Args:
            frequency: Spatial frequency
            step: Current phase step
            steps_total: Total number of phase steps
            orientation: "horizontal" or "vertical"
            width: Pattern width
            height: Pattern height
        """
        name = f"phase_{orientation[0]}_freq{frequency}_step{step}"
        super().__init__(name=name, pattern_type="raw_image", width=width, height=height)

        self.frequency = frequency
        self.step = step
        self.steps_total = steps_total
        self.orientation = orientation

        # Calculate phase shift
        self.phase_offset = 2 * np.pi * step / steps_total

        # Generate the pattern
        self._generate_pattern()

    def _generate_pattern(self):
        """Generate phase shift pattern image."""
        # Create a blank image
        img = np.zeros((self.height, self.width), dtype=np.uint8)

        # Generate sinusoidal pattern with phase shift
        if self.orientation == "horizontal":
            for x in range(self.width):
                # Calculate intensity using cosine wave with phase offset
                # Frequency is inversely proportional to the wave length in pixels
                # Higher frequency = more cycles across the image
                cycles_per_image = self.width / max(1, self.frequency)
                val = 127.5 + 127.5 * np.cos(2 * np.pi * x / cycles_per_image + self.phase_offset)
                img[:, x] = val
        else:  # vertical
            for y in range(self.height):
                cycles_per_image = self.height / max(1, self.frequency)
                val = 127.5 + 127.5 * np.cos(2 * np.pi * y / cycles_per_image + self.phase_offset)
                img[y, :] = val

        # Store the raw image data
        success, encoded = cv2.imencode('.png', img)
        if success:
            self.data = encoded.tobytes()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for projection."""
        return {
            "pattern_type": "raw_image",
            "name": self.name,
            "orientation": self.orientation,
            "frequency": self.frequency,
            "step": self.step,
            "steps_total": self.steps_total,  # Add total steps for better approximation
            "phase_offset": float(self.phase_offset),  # Convert from numpy to Python native type
            "image": self.data
        }


class PatternGenerator:
    """Generator for structured light patterns."""
    
    def __init__(
        self, 
        width: int = 1024, 
        height: int = 768,
        num_gray_codes: int = 10,
        num_phase_shifts: int = 8,
        phase_shift_frequencies: List[int] = [1, 8, 16]
    ):
        """
        Initialize pattern generator.
        
        Args:
            width: Pattern width
            height: Pattern height
            num_gray_codes: Number of Gray code patterns
            num_phase_shifts: Number of phase shifts per frequency
            phase_shift_frequencies: Frequencies for phase shifting
        """
        self.width = width
        self.height = height
        self.num_gray_codes = num_gray_codes
        self.num_phase_shifts = num_phase_shifts
        self.phase_shift_frequencies = phase_shift_frequencies
        
        logger.info(f"Initialized PatternGenerator with resolution {width}x{height}")
        logger.info(f"Gray code patterns: {num_gray_codes * 4} (including inversions)")
        logger.info(f"Phase shift patterns: {len(phase_shift_frequencies) * num_phase_shifts * 2} (both orientations)")
    
    def generate_white_black(self) -> List[Pattern]:
        """Generate white and black reference patterns."""
        return [
            SolidPattern(color="White", width=self.width, height=self.height),
            SolidPattern(color="Black", width=self.width, height=self.height)
        ]
    
    def generate_gray_code(self) -> List[Pattern]:
        """Generate Gray code patterns."""
        patterns = []
        
        # Generate for each bit (both horizontal and vertical)
        for bit in range(self.num_gray_codes):
            # Horizontal
            patterns.append(GrayCodePattern(bit=bit, orientation="horizontal", inverted=False, 
                                          width=self.width, height=self.height))
            patterns.append(GrayCodePattern(bit=bit, orientation="horizontal", inverted=True, 
                                          width=self.width, height=self.height))
            
            # Vertical
            patterns.append(GrayCodePattern(bit=bit, orientation="vertical", inverted=False, 
                                          width=self.width, height=self.height))
            patterns.append(GrayCodePattern(bit=bit, orientation="vertical", inverted=True, 
                                          width=self.width, height=self.height))
        
        return patterns
    
    def generate_phase_shift(self) -> List[Pattern]:
        """Generate phase shift patterns."""
        patterns = []
        
        # Generate for each frequency and step
        for freq in self.phase_shift_frequencies:
            for step in range(self.num_phase_shifts):
                # Horizontal
                patterns.append(PhaseShiftPattern(frequency=freq, step=step, steps_total=self.num_phase_shifts,
                                                orientation="horizontal", width=self.width, height=self.height))
                
                # Vertical
                patterns.append(PhaseShiftPattern(frequency=freq, step=step, steps_total=self.num_phase_shifts,
                                                orientation="vertical", width=self.width, height=self.height))
        
        return patterns
    
    def generate_all_patterns(self) -> List[Pattern]:
        """Generate a complete set of patterns for robust scanning."""
        # Start with white and black reference patterns
        patterns = self.generate_white_black()
        
        # Add Gray code patterns
        patterns.extend(self.generate_gray_code())
        
        # Add phase shift patterns
        patterns.extend(self.generate_phase_shift())
        
        logger.info(f"Generated {len(patterns)} structured light patterns")
        return patterns
    
    def patterns_to_dicts(self, patterns: List[Pattern]) -> List[Dict[str, Any]]:
        """Convert pattern objects to dictionaries for projection."""
        return [pattern.to_dict() for pattern in patterns]


class StructuredLightController:
    """Controller for structured light projection and capture."""
    
    def __init__(
        self,
        projector_client,
        camera_client,
        width: int = 1024,
        height: int = 768,
        pattern_interval: float = 0.5
    ):
        """
        Initialize structured light controller.
        
        Args:
            projector_client: Projector client for pattern projection
            camera_client: Camera client for image capture
            width: Pattern width
            height: Pattern height
            pattern_interval: Time interval between patterns in seconds
        """
        self.projector = projector_client
        self.camera = camera_client
        self.width = width
        self.height = height
        self.pattern_interval = pattern_interval
        
        # Initialize pattern generator
        self.pattern_generator = PatternGenerator(width=width, height=height)
        
        logger.info(f"Initialized StructuredLightController with resolution {width}x{height}")
    
    def project_and_capture(
        self,
        patterns: List[Dict[str, Any]],
        left_camera_id: str = None,
        right_camera_id: str = None,
        output_dir: Optional[str] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Project patterns and capture images.
        
        Args:
            patterns: List of pattern dictionaries
            left_camera_id: ID of left camera (if None, uses first camera)
            right_camera_id: ID of right camera (if None, uses second camera)
            output_dir: Directory to save captured images
            
        Returns:
            Tuple of (left_images, right_images)
        """
        # Prepare output directory if specified
        if output_dir:
            captures_dir = os.path.join(output_dir, "captures")
            os.makedirs(captures_dir, exist_ok=True)
        
        # Get camera IDs if not specified
        cameras = self.camera.get_cameras()
        if not cameras or len(cameras) < 2:
            raise ValueError("Need at least 2 cameras for stereo structured light")
        
        if left_camera_id is None:
            left_camera_id = cameras[0]["id"]
        if right_camera_id is None:
            right_camera_id = cameras[1]["id"]
        
        # Prepare for capture
        left_images = []
        right_images = []
        
        # Project patterns and capture images
        logger.info(f"Projecting {len(patterns)} patterns and capturing images...")
        
        for i, pattern in enumerate(patterns):
            pattern_name = pattern.get("name", f"pattern_{i}")
            logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern_name}")
            
            # Use the appropriate projector method based on pattern type
            pattern_type = pattern.get("pattern_type", "")
            
            if pattern_type == "solid_field":
                # Use direct solid field method
                success = self.projector.show_solid_field(pattern.get("color", "White"))
            elif pattern_type == "horizontal_lines":
                # Use direct horizontal lines method
                success = self.projector.show_horizontal_lines(
                    foreground_color=pattern.get("foreground_color", "White"),
                    background_color=pattern.get("background_color", "Black"),
                    foreground_width=pattern.get("foreground_width", 4),
                    background_width=pattern.get("background_width", 20)
                )
            elif pattern_type == "vertical_lines":
                # Use direct vertical lines method
                success = self.projector.show_vertical_lines(
                    foreground_color=pattern.get("foreground_color", "White"),
                    background_color=pattern.get("background_color", "Black"),
                    foreground_width=pattern.get("foreground_width", 4),
                    background_width=pattern.get("background_width", 20)
                )
            elif pattern_type == "raw_image":
                # For raw images with binary data, we need a different approach
                # Check if the client has a raw_image or send_pattern method
                if hasattr(self.projector, "show_raw_image") and callable(getattr(self.projector, "show_raw_image")):
                    # Use raw image method if available
                    if "image" in pattern:
                        success = self.projector.show_raw_image(pattern["image"])
                    else:
                        logger.warning(f"Raw image pattern missing image data: {pattern_name}")
                        success = False
                else:
                    # No direct raw image method, try alternate approaches
                    orientation = pattern.get("orientation", "horizontal")
                    
                    # For Gray code patterns (has bit_position)
                    if "bit_position" in pattern:
                        bit_pos = pattern.get("bit_position", 0)
                        is_inverse = pattern.get("is_inverse", False)
                        
                        # Calculate appropriate width based on bit position
                        stripe_width = max(1, 2 ** bit_pos // 4)  # Divide by 4 for more reasonable sizes
                        
                        # Choose pattern direction
                        if orientation == "horizontal":
                            success = self.projector.show_horizontal_lines(
                                foreground_color="Black" if is_inverse else "White",
                                background_color="White" if is_inverse else "Black",
                                foreground_width=stripe_width,
                                background_width=stripe_width
                            )
                        else:
                            success = self.projector.show_vertical_lines(
                                foreground_color="Black" if is_inverse else "White",
                                background_color="White" if is_inverse else "Black",
                                foreground_width=stripe_width,
                                background_width=stripe_width
                            )
                    
                    # For phase shift patterns (has frequency)
                    elif "frequency" in pattern:
                        # Get parameters for approximating phase shift patterns
                        frequency = pattern.get("frequency", 8)
                        step = pattern.get("step", 0)
                        steps_total = pattern.get("steps_total", 8)
                        orientation = pattern.get("orientation", "horizontal")

                        # Calculate phase offset (0 to 2π)
                        phase_offset = 2 * 3.14159 * step / steps_total

                        # Calculate line width based on frequency and phase
                        # Lower frequency = wider lines
                        base_width = max(1, 32 // frequency)

                        # Adjust width based on phase to approximate sinusoidal pattern
                        # Maps phase (0-2π) to width ratio (0.5-2.0)
                        width_ratio = 1.0 + 0.5 * np.cos(phase_offset)

                        fg_width = max(1, int(base_width * width_ratio))
                        bg_width = max(1, int(base_width * (2.0 - width_ratio)))

                        # Log what we're approximating
                        logger.debug(f"Approximating phase pattern: freq={frequency}, step={step}/{steps_total}, "
                                    f"phase={phase_offset:.2f}, using {fg_width}/{bg_width} width")

                        # Use horizontal or vertical lines based on orientation
                        if orientation.startswith("h"):
                            success = self.projector.show_horizontal_lines(
                                foreground_color="White",
                                background_color="Black",
                                foreground_width=fg_width,
                                background_width=bg_width
                            )
                        else:
                            success = self.projector.show_vertical_lines(
                                foreground_color="White",
                                background_color="Black",
                                foreground_width=fg_width,
                                background_width=bg_width
                            )
                    
                    else:
                        logger.warning(f"Unknown raw image pattern format: {pattern_name}")
                        success = False
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}")
                success = False
            
            if not success:
                logger.warning(f"Failed to project pattern {i+1}/{len(patterns)}: {pattern_name}")
            
            # Wait for projector to update
            time.sleep(self.pattern_interval)
            
            # Capture stereo pair
            try:
                left_img = self.camera.capture(left_camera_id)
                right_img = self.camera.capture(right_camera_id)
                
                # Only save debug images if explicitly enabled
                # Disabled by default to avoid performance issues
                if output_dir and os.environ.get("UNLOOK_SAVE_DEBUG_IMAGES", "0") == "1":
                    left_path = os.path.join(captures_dir, f"left_{i:03d}.png")
                    right_path = os.path.join(captures_dir, f"right_{i:03d}.png")

                    logger.debug(f"Saving debug images to {left_path} and {right_path}")
                    cv2.imwrite(left_path, left_img)
                    cv2.imwrite(right_path, right_img)
                
                # Append to image lists
                left_images.append(left_img)
                right_images.append(right_img)
                
            except Exception as e:
                logger.error(f"Error capturing images: {e}")
                continue
        
        # Reset projector to black field
        self.projector.show_solid_field("Black")
        
        return left_images, right_images


# Helper functions for structured light pattern generation
def generate_patterns(
    width: int = 1024,
    height: int = 768,
    num_gray_codes: int = 10,
    num_phase_shifts: int = 8,
    phase_shift_frequencies: List[int] = [1, 8, 16]
) -> List[Dict[str, Any]]:
    """Generate structured light patterns for scanning."""
    generator = PatternGenerator(
        width=width,
        height=height,
        num_gray_codes=num_gray_codes,
        num_phase_shifts=num_phase_shifts,
        phase_shift_frequencies=phase_shift_frequencies
    )
    
    patterns = generator.generate_all_patterns()
    return generator.patterns_to_dicts(patterns)


def project_patterns(
    projector_client,
    patterns: List[Dict[str, Any]],
    interval: float = 0.5
) -> bool:
    """Project structured light patterns."""
    # Check if client has sequence support
    if hasattr(projector_client, "start_pattern_sequence"):
        # Use pattern sequence for better timing
        return projector_client.start_pattern_sequence(patterns, interval=interval)
    
    # Otherwise project patterns individually
    for i, pattern in enumerate(patterns):
        pattern_name = pattern.get("name", f"pattern_{i}")
        logger.info(f"Projecting pattern {i+1}/{len(patterns)}: {pattern_name}")
        
        # Determine pattern type and use appropriate method
        pattern_type = pattern.get("pattern_type", "")
        
        if pattern_type == "solid_field":
            projector_client.show_solid_field(pattern.get("color", "White"))
        elif pattern_type == "horizontal_lines":
            projector_client.show_horizontal_lines(
                foreground_color=pattern.get("foreground_color", "White"),
                background_color=pattern.get("background_color", "Black"),
                foreground_width=pattern.get("foreground_width", 4),
                background_width=pattern.get("background_width", 20)
            )
        elif pattern_type == "vertical_lines":
            projector_client.show_vertical_lines(
                foreground_color=pattern.get("foreground_color", "White"),
                background_color=pattern.get("background_color", "Black"),
                foreground_width=pattern.get("foreground_width", 4),
                background_width=pattern.get("background_width", 20)
            )
        else:
            # For raw images, we don't have a direct method
            logger.warning(f"Cannot project raw image pattern directly: {pattern_name}")
            continue
        
        # Wait for projector to update
        time.sleep(interval)
    
    # Reset projector to black field
    projector_client.show_solid_field("Black")
    
    return True