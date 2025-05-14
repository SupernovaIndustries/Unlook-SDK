"""
Projector Adapter for Unlook SDK.

This module provides an adapter that normalizes the interface between
different projector implementations (Projector and ProjectorClient classes).
It abstracts away the differences between the two APIs, allowing the
real-time scanner to work with either implementation seamlessly.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ProjectorAdapter:
    """
    Adapter for projector implementations with a consistent interface.
    
    This adapter provides a unified interface for interacting with different
    projector implementations, abstracting away the differences between
    the Projector and ProjectorClient classes.
    """
    
    def __init__(self, projector_client):
        """
        Initialize the adapter with a projector client.
        
        Args:
            projector_client: The underlying projector client (either Projector or ProjectorClient)
        """
        self.projector = projector_client
        self._detect_client_type()
    
    def _detect_client_type(self):
        """Detect the type of projector client to determine appropriate adaption."""
        self.is_projector_class = hasattr(self.projector, 'project_pattern')
        self.is_projector_client = hasattr(self.projector, 'show_solid_field')
        
        if not (self.is_projector_class or self.is_projector_client):
            logger.warning("Unknown projector client type - adapter may not work correctly")
    
    def project_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Project a pattern using the underlying projector client.
        
        Args:
            pattern: Pattern definition dictionary with pattern_type and other attributes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pattern_type = pattern.get("pattern_type")
            
            # If we have the direct project_pattern method (Projector class)
            if self.is_projector_class:
                return self.projector.project_pattern(pattern)
            
            # If we have ProjectorClient class with different methods
            elif self.is_projector_client:
                # Dispatch to appropriate method based on pattern type
                if pattern_type == "solid_field":
                    return self.projector.show_solid_field(
                        pattern.get("color", "White")
                    )
                elif pattern_type == "horizontal_lines":
                    return self.projector.show_horizontal_lines(
                        pattern.get("foreground_color", "White"),
                        pattern.get("background_color", "Black"),
                        pattern.get("foreground_width", 4),
                        pattern.get("background_width", 20)
                    )
                elif pattern_type == "vertical_lines":
                    return self.projector.show_vertical_lines(
                        pattern.get("foreground_color", "White"),
                        pattern.get("background_color", "Black"),
                        pattern.get("foreground_width", 4),
                        pattern.get("background_width", 20)
                    )
                elif pattern_type == "grid":
                    return self.projector.show_grid(
                        pattern.get("foreground_color", "White"),
                        pattern.get("background_color", "Black"),
                        pattern.get("h_foreground_width", 4),
                        pattern.get("h_background_width", 20),
                        pattern.get("v_foreground_width", 4),
                        pattern.get("v_background_width", 20)
                    )
                elif pattern_type == "checkerboard":
                    return self.projector.show_checkerboard(
                        pattern.get("foreground_color", "White"),
                        pattern.get("background_color", "Black"),
                        pattern.get("horizontal_count", 8),
                        pattern.get("vertical_count", 6)
                    )
                elif pattern_type == "colorbars":
                    return self.projector.show_colorbars()
                # Handle all advanced pattern types similarly
                elif pattern_type in ["gray_code", "multi_scale", "variable_width", "multi_frequency", "phase_shift"]:
                    # For all structured light patterns, we need to synthesize a consistent pattern
                    # Convert the orientation, bit, and inverted status to horizontal/vertical lines
                    
                    orientation = pattern.get("orientation", "horizontal")
                    inverted = pattern.get("inverted", False)
                    bit = pattern.get("bit", pattern.get("width_bits", 0))
                    
                    # For better logging
                    pattern_name = pattern.get("name", f"{pattern_type}_unnamed")
                    logger.info(f"Converting enhanced pattern {pattern_name} to projector lines")
                    
                    # Choose colors based on inverted status
                    foreground = "Black" if inverted else "White"
                    background = "White" if inverted else "Black"
                    
                    # Scale line width based on bit value for variable patterns
                    try:
                        width_scale = max(1, 2 ** bit // 2)  # Ensure at least 1 pixel width
                    except (ValueError, TypeError):
                        width_scale = 4  # Default line width
                    
                    # Create appropriate line pattern based on orientation
                    if orientation == "horizontal":
                        logger.info(f"Showing horizontal lines for pattern {pattern_name}, width={width_scale}")
                        return self.projector.show_horizontal_lines(
                            foreground_color=foreground,
                            background_color=background,
                            foreground_width=width_scale,
                            background_width=width_scale
                        )
                    else:  # vertical
                        logger.info(f"Showing vertical lines for pattern {pattern_name}, width={width_scale}")
                        return self.projector.show_vertical_lines(
                            foreground_color=foreground,
                            background_color=background,
                            foreground_width=width_scale,
                            background_width=width_scale
                        )
                else:
                    logger.warning(f"Unsupported pattern type for projector adapter: {pattern_type}")
                    return False
            else:
                logger.error("No compatible projector methods found")
                return False
        
        except Exception as e:
            logger.error(f"Error projecting pattern with adapter: {e}")
            return False
    
    def project_sequence(self, patterns: List[Dict[str, Any]], interval: float = 0.5) -> bool:
        """
        Project a sequence of patterns.
        
        Args:
            patterns: List of pattern dictionaries
            interval: Time interval between patterns (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_projector_class and hasattr(self.projector, 'project_sequence'):
                return self.projector.project_sequence(patterns, interval)
            elif self.is_projector_client and hasattr(self.projector, 'start_pattern_sequence'):
                return self.projector.start_pattern_sequence(patterns, interval=interval)
            else:
                logger.warning("Pattern sequence not supported by underlying projector")
                return False
        except Exception as e:
            logger.error(f"Error projecting pattern sequence: {e}")
            return False
    
    def set_mode(self, mode: str) -> bool:
        """
        Set the projector mode.
        
        Args:
            mode: Projector mode
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_projector_client and hasattr(self.projector, 'set_mode'):
                return self.projector.set_mode(mode)
            else:
                # Projector class doesn't have mode control
                logger.warning("Projector mode control not supported")
                return False
        except Exception as e:
            logger.error(f"Error setting projector mode: {e}")
            return False
    
    def turn_off(self) -> bool:
        """
        Turn off the projector (project black field).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            pattern = {
                "pattern_type": "solid_field",
                "color": "Black",
                "name": "off"
            }
            return self.project_pattern(pattern)
        except Exception as e:
            logger.error(f"Error turning off projector: {e}")
            return False


def create_projector_adapter(projector_client) -> ProjectorAdapter:
    """
    Create a projector adapter from a projector client.
    
    Args:
        projector_client: Projector or ProjectorClient instance
        
    Returns:
        ProjectorAdapter instance
    """
    return ProjectorAdapter(projector_client)