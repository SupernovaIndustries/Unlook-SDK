"""
Client for controlling the UnLook scanner projector.
"""

import logging
import time
import os
import numpy as np
import cv2
from typing import Dict, Optional, Any, List, Union, Callable, Tuple

try:
    from ..core.protocol import MessageType
    from ..core.events import EventType, EventEmitter
except ImportError:
    # Define fallback classes for when the core modules are not available
    class MessageType:
        PROJECTOR_MODE = "projector_mode"
        PROJECTOR_PATTERN = "projector_pattern"
        PROJECTOR_PATTERN_SEQUENCE = "projector_pattern_sequence"
        PROJECTOR_PATTERN_SEQUENCE_STEP = "projector_pattern_sequence_step"
        PROJECTOR_PATTERN_SEQUENCE_STOP = "projector_pattern_sequence_stop"
        LED_SET_CURRENT = "led_set_current"
        LED_GET_CURRENT = "led_get_current"
        LED_SET_ENABLE = "led_set_enable"
    
    class EventType:
        PROJECTOR_PATTERN_CHANGED = "projector_pattern_changed"
        PROJECTOR_SEQUENCE_STARTED = "projector_sequence_started"
        PROJECTOR_SEQUENCE_STEPPED = "projector_sequence_stepped"
        PROJECTOR_SEQUENCE_COMPLETED = "projector_sequence_completed"
        PROJECTOR_SEQUENCE_STOPPED = "projector_sequence_stopped"
    
    class EventEmitter:
        def on(self, event_type, callback):
            pass

logger = logging.getLogger(__name__)


class Projector:
    """
    Class for projector operations in the UnLook scanner.
    """
    
    def __init__(self):
        """
        Initialize the projector.
        """
        self.client = None  # Will be set when connected to a real server

        # Always use real hardware mode by default
        self._is_simulation = False
        self.current_pattern = None
        self.pattern_sequence = []
        self.current_sequence_index = 0
        
        logger.info("Initialized projector")
    
    def project_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Project a pattern.

        Args:
            pattern: Pattern definition dictionary

        Returns:
            True if successful, False otherwise
        """
        # Add more detailed logging to debug pattern issues
        pattern_type = pattern.get("pattern_type", "unknown")
        pattern_name = pattern.get("name", "unnamed")
        logger.info(f"Projector.project_pattern called with: type={pattern_type}, name={pattern_name}")
        
        # If we have a real client, use it to project
        if self.client:
            # Convert the pattern to the right format based on pattern_type
            if isinstance(pattern, dict) and "pattern_type" in pattern:
                pattern_type = pattern["pattern_type"]

                try:
                    # Dispatch to appropriate method based on pattern type
                    if pattern_type == "solid_field":
                        result = self.client.show_solid_field(pattern.get("color", "White"))

                    elif pattern_type == "horizontal_lines":
                        result = self.client.show_horizontal_lines(
                            pattern.get("foreground_color", "White"),
                            pattern.get("background_color", "Black"),
                            pattern.get("foreground_width", 4),
                            pattern.get("background_width", 20)
                        )

                    elif pattern_type == "vertical_lines":
                        result = self.client.show_vertical_lines(
                            pattern.get("foreground_color", "White"),
                            pattern.get("background_color", "Black"),
                            pattern.get("foreground_width", 4),
                            pattern.get("background_width", 20)
                        )

                    elif pattern_type == "grid":
                        result = self.client.show_grid(
                            pattern.get("foreground_color", "White"),
                            pattern.get("background_color", "Black"),
                            pattern.get("h_foreground_width", 4),
                            pattern.get("h_background_width", 20),
                            pattern.get("v_foreground_width", 4),
                            pattern.get("v_background_width", 20)
                        )

                    elif pattern_type == "checkerboard":
                        result = self.client.show_checkerboard(
                            pattern.get("foreground_color", "White"),
                            pattern.get("background_color", "Black"),
                            pattern.get("horizontal_count", 8),
                            pattern.get("vertical_count", 6)
                        )

                    elif pattern_type == "colorbars":
                        result = self.client.show_colorbars()

                    elif pattern_type == "raw_image":
                        # For raw images, we need to directly pass the binary data
                        if "image" in pattern:
                            img = pattern["image"]
                            logger.info(f"Projecting raw image pattern: {pattern.get('name', 'unnamed')}")
                            
                            # Send the actual image to the projector
                            try:
                                if hasattr(self.client, 'send'):
                                    # Direct protocol send method
                                    message = {
                                        "command": "show_raw_image",
                                        "binary_data": img.tobytes(),
                                        "width": img.shape[1],
                                        "height": img.shape[0],
                                        "channels": 1 if len(img.shape) == 2 else img.shape[2]
                                    }
                                    reply = self.client.send(message)
                                    result = reply and reply.get("status") == "success"
                                elif hasattr(self.client, 'show_raw_image'):
                                    result = self.client.show_raw_image(img)
                                elif hasattr(self.client, 'show_pattern'):
                                    result = self.client.show_pattern(img)
                                else:
                                    logger.error("Client doesn't support raw image projection")
                                    result = False
                            except Exception as e:
                                logger.error(f"Error projecting raw image: {e}")
                                return False
                        else:
                            logger.warning(f"Raw image pattern missing image data: {pattern.get('name', 'unnamed')}")
                            return False
                            
                    # Support for enhanced pattern types
                    elif pattern_type in ["gray_code", "multi_scale", "variable_width", "multi_frequency"]:
                        # Convert Gray code patterns to line patterns
                        if pattern_type == "gray_code":
                            # Gray code patterns need to be displayed as horizontal or vertical lines
                            # The pattern depends on the bit value and orientation
                            orientation = pattern.get("orientation", "horizontal")
                            bit = pattern.get("bit", 0)
                            inverted = pattern.get("inverted", False)
                            name = pattern.get("name", "unnamed")
                            
                            logger.info(f"Converting Gray code pattern {name} (bit={bit}, inv={inverted}) to lines")
                            
                            # Calculate stripe width based on bit position
                            # Higher bits = wider stripes
                            stripe_width = 2 ** bit
                            
                            # Choose colors based on inverted flag
                            if inverted:
                                fg_color = "Black"
                                bg_color = "White"
                            else:
                                fg_color = "White"
                                bg_color = "Black"
                            
                            # Project appropriate line pattern
                            if orientation == "horizontal":
                                result = self.client.show_horizontal_lines(
                                    foreground_color=fg_color,
                                    background_color=bg_color,
                                    foreground_width=stripe_width,
                                    background_width=stripe_width
                                )
                            else:  # vertical
                                result = self.client.show_vertical_lines(
                                    foreground_color=fg_color,
                                    background_color=bg_color,
                                    foreground_width=stripe_width,
                                    background_width=stripe_width
                                )
                        else:
                            # For other pattern types, use checkerboard as fallback
                            logger.warning(f"Using checkerboard as placeholder for {pattern_type} pattern")
                            result = self.client.show_checkerboard(
                                foreground_color="White",
                                background_color="Black", 
                                horizontal_count=10,
                                vertical_count=10
                            )
                    
                    # Handle phase_shift patterns
                    elif pattern_type == "phase_shift":
                        # Phase shift patterns use sinusoidal intensity variations
                        # We'll simulate this with varying line widths
                        orientation = pattern.get("orientation", "horizontal")
                        step = pattern.get("step", 0)
                        total_steps = pattern.get("total_steps", 8)
                        name = pattern.get("name", "unnamed")
                        
                        logger.info(f"Converting phase shift pattern {name} (step={step}/{total_steps}) to lines")
                        
                        # Calculate phase offset
                        phase_offset = (2 * 3.14159 * step) / total_steps
                        
                        # For phase shift, we alternate between bright and dark regions
                        # based on the phase
                        if step < total_steps // 2:
                            fg_color = "White"
                            bg_color = "Black"
                        else:
                            fg_color = "Black"
                            bg_color = "White"
                        
                        # Use varying line widths to simulate sinusoidal pattern
                        width = 8 + int(4 * abs((step % 4) - 2))
                        
                        # Project appropriate line pattern
                        if orientation == "horizontal":
                            result = self.client.show_horizontal_lines(
                                foreground_color=fg_color,
                                background_color=bg_color,
                                foreground_width=width,
                                background_width=width
                            )
                        else:  # vertical
                            result = self.client.show_vertical_lines(
                                foreground_color=fg_color,
                                background_color=bg_color,
                                foreground_width=width,
                                background_width=width
                            )

                    else:
                        logger.warning(f"Unknown pattern type: {pattern_type}")
                        return False

                    # If we got here, the operation was successful
                    return result

                except Exception as e:
                    logger.error(f"Error projecting pattern: {e}")
                    return False
            else:
                logger.warning("Invalid pattern format")
                return False

        # If we're in development mode and don't have a client, fall back to logging the pattern
        # but treat this as an error case since we want to use real hardware
        logger.error(f"No projector client available. Real hardware required. Pattern: {pattern.get('name', 'unnamed')}")
        return False
    
    def project_sequence(self, patterns: List[Dict[str, Any]], interval: float = 0.5) -> bool:
        """
        Project a sequence of patterns with timing control.

        Args:
            patterns: List of pattern dictionaries
            interval: Time in seconds between patterns

        Returns:
            True if successful, False otherwise
        """
        # If we have a real client, use it to project sequence
        if self.client:
            try:
                # Start pattern sequence
                result = self.client.start_pattern_sequence(patterns, interval=interval)
                return result
            except Exception as e:
                logger.error(f"Error starting pattern sequence with {len(patterns)} patterns: {e}")
                return False

        # If we don't have a client, this is an error - we want to use real hardware only
        logger.error(f"No projector client available. Real hardware required. Pattern sequence with {len(patterns)} patterns.")
        return False
    
    def create_structured_light_patterns(self, 
                                      use_gray_code: bool = True, 
                                      use_phase_shift: bool = True,
                                      num_gray_codes: int = 10,
                                      num_phase_shifts: int = 8,
                                      phase_shift_frequencies: List[int] = [1, 8, 16]) -> List[Dict[str, Any]]:
        """
        Create a sequence of structured light patterns combining Gray code and phase shift.
        
        Args:
            use_gray_code: Whether to include Gray code patterns
            use_phase_shift: Whether to include phase shift patterns
            num_gray_codes: Number of Gray code patterns (bits)
            num_phase_shifts: Number of phase shifts per frequency
            phase_shift_frequencies: List of frequencies for phase shift patterns
            
        Returns:
            List of pattern dictionaries with actual image data
        """
        from .enhanced_gray_code import generate_enhanced_gray_code_patterns
        from .enhanced_phaseshift import generate_phase_shift_patterns
        
        # Get projector resolution
        width = getattr(self, 'width', 1024)
        height = getattr(self, 'height', 768)
        
        patterns = []
        
        # Always add white and black reference patterns
        patterns.append({
            "pattern_type": "solid_field",
            "color": "White",
            "name": "white_reference",
            "image": np.ones((height, width), dtype=np.uint8) * 255
        })
        patterns.append({
            "pattern_type": "solid_field",
            "color": "Black",
            "name": "black_reference",
            "image": np.zeros((height, width), dtype=np.uint8)
        })
        
        if use_gray_code:
            # Generate enhanced Gray code patterns with image data
            gray_patterns = generate_enhanced_gray_code_patterns(
                width=width,
                height=height,
                num_bits=num_gray_codes,
                orientation="both"
            )
            patterns.extend(gray_patterns[2:])  # Skip the reference patterns we already added
        
        if use_phase_shift:
            # Generate phase shift patterns with image data
            for freq in phase_shift_frequencies:
                # Generate horizontal phase shift patterns
                h_phase_patterns = generate_phase_shift_patterns(
                    width=width,
                    height=height,
                    frequency=freq,
                    num_frames=num_phase_shifts,
                    orientation="horizontal"
                )
                patterns.extend(h_phase_patterns)
                
                # Generate vertical phase shift patterns
                v_phase_patterns = generate_phase_shift_patterns(
                    width=width,
                    height=height,
                    frequency=freq,
                    num_frames=num_phase_shifts,
                    orientation="vertical"
                )
                patterns.extend(v_phase_patterns)
        
        logger.info(f"Created {len(patterns)} structured light patterns")
        return patterns
    
    def show_solid_field(self, color: str = "White") -> bool:
        """
        Show a solid field of a given color.
        
        Args:
            color: Color name (White, Black, Red, Green, Blue, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        pattern = {"pattern_type": "solid_field", "color": color}
        return self.project_pattern(pattern)
    
    def show_horizontal_lines(self, 
                            foreground_color: str = "White",
                            background_color: str = "Black",
                            foreground_width: int = 4,
                            background_width: int = 20) -> bool:
        """
        Show horizontal lines.
        
        Args:
            foreground_color: Line color
            background_color: Background color
            foreground_width: Line width
            background_width: Width of spaces between lines
            
        Returns:
            True if successful, False otherwise
        """
        pattern = {
            "pattern_type": "horizontal_lines",
            "foreground_color": foreground_color,
            "background_color": background_color,
            "foreground_width": foreground_width,
            "background_width": background_width
        }
        return self.project_pattern(pattern)
    
    def show_vertical_lines(self,
                          foreground_color: str = "White",
                          background_color: str = "Black",
                          foreground_width: int = 4,
                          background_width: int = 20) -> bool:
        """
        Show vertical lines.
        
        Args:
            foreground_color: Line color
            background_color: Background color
            foreground_width: Line width
            background_width: Width of spaces between lines
            
        Returns:
            True if successful, False otherwise
        """
        pattern = {
            "pattern_type": "vertical_lines",
            "foreground_color": foreground_color,
            "background_color": background_color,
            "foreground_width": foreground_width,
            "background_width": background_width
        }
        return self.project_pattern(pattern)


class ProjectorClient:
    """
    Client for controlling the UnLook scanner projector.
    """

    def __init__(self, parent_client):
        """
        Initialize the projector client.

        Args:
            parent_client: Main UnlookClient
        """
        self.client = parent_client
        
        # Pattern sequence state
        self.current_pattern_sequence = []
        self.current_sequence_id = None
        self.sequence_active = False
        
        # Initialize event callbacks dictionary
        self._event_callbacks = {
            EventType.PROJECTOR_PATTERN_CHANGED: [],
            EventType.PROJECTOR_SEQUENCE_STARTED: [],
            EventType.PROJECTOR_SEQUENCE_STEPPED: [],
            EventType.PROJECTOR_SEQUENCE_COMPLETED: [],
            EventType.PROJECTOR_SEQUENCE_STOPPED: []
        }
        
        # Register for projector events from parent client
        if hasattr(parent_client, 'events') and isinstance(parent_client.events, EventEmitter):
            self._register_event_handlers()
            
    def _register_event_handlers(self):
        """
        Register for projector-related events from the parent client.
        """
        # Register for projector events
        self.client.events.on(EventType.PROJECTOR_PATTERN_CHANGED, self._on_pattern_changed)
        self.client.events.on(EventType.PROJECTOR_SEQUENCE_STARTED, self._on_sequence_started)
        self.client.events.on(EventType.PROJECTOR_SEQUENCE_STEPPED, self._on_sequence_stepped)
        self.client.events.on(EventType.PROJECTOR_SEQUENCE_COMPLETED, self._on_sequence_completed)
        self.client.events.on(EventType.PROJECTOR_SEQUENCE_STOPPED, self._on_sequence_stopped)
    
    def on_pattern_changed(self, callback: Callable):
        """
        Register a callback for when a pattern changes.
        
        Args:
            callback: Function to call when pattern changes
        """
        self._event_callbacks[EventType.PROJECTOR_PATTERN_CHANGED].append(callback)
        
    def on_sequence_started(self, callback: Callable):
        """
        Register a callback for when a pattern sequence starts.
        
        Args:
            callback: Function to call when sequence starts
        """
        self._event_callbacks[EventType.PROJECTOR_SEQUENCE_STARTED].append(callback)
        
    def on_sequence_stepped(self, callback: Callable):
        """
        Register a callback for when a pattern sequence steps to the next pattern.
        
        Args:
            callback: Function to call when sequence steps
        """
        self._event_callbacks[EventType.PROJECTOR_SEQUENCE_STEPPED].append(callback)
        
    def on_sequence_completed(self, callback: Callable):
        """
        Register a callback for when a pattern sequence completes.
        
        Args:
            callback: Function to call when sequence completes
        """
        self._event_callbacks[EventType.PROJECTOR_SEQUENCE_COMPLETED].append(callback)
        
    def on_sequence_stopped(self, callback: Callable):
        """
        Register a callback for when a pattern sequence stops.
        
        Args:
            callback: Function to call when sequence stops
        """
        self._event_callbacks[EventType.PROJECTOR_SEQUENCE_STOPPED].append(callback)
    
    # Event handler methods (internal use only)
    def _on_pattern_changed(self, data):
        for callback in self._event_callbacks[EventType.PROJECTOR_PATTERN_CHANGED]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in pattern changed callback: {e}")
    
    def _on_sequence_started(self, data):
        self.sequence_active = True
        for callback in self._event_callbacks[EventType.PROJECTOR_SEQUENCE_STARTED]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in sequence started callback: {e}")
    
    def _on_sequence_stepped(self, data):
        for callback in self._event_callbacks[EventType.PROJECTOR_SEQUENCE_STEPPED]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in sequence stepped callback: {e}")
    
    def _on_sequence_completed(self, data):
        self.sequence_active = False
        for callback in self._event_callbacks[EventType.PROJECTOR_SEQUENCE_COMPLETED]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in sequence completed callback: {e}")
    
    def _on_sequence_stopped(self, data):
        self.sequence_active = False
        for callback in self._event_callbacks[EventType.PROJECTOR_SEQUENCE_STOPPED]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in sequence stopped callback: {e}")

    def set_mode(self, mode: str) -> bool:
        """
        Set the projector mode.

        Args:
            mode: Projector mode (see OperatingMode in server SDK)

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_MODE,
            {"mode": mode}
        )

        if success and response:
            logger.info(f"Projector mode set: {mode}")
            return True
        else:
            logger.error(f"Error setting projector mode: {mode}")
            return False

    def show_solid_field(self, color: str = "White") -> bool:
        """
        Show a solid field of a color.

        Args:
            color: Color (Black, Red, Green, Blue, Cyan, Magenta, Yellow, White)

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "solid_field",
                "color": color
            }
        )

        if success and response:
            logger.info(f"Solid field projected: {color}")
            return True
        else:
            logger.error(f"Error projecting solid field: {color}")
            return False

    def show_horizontal_lines(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            foreground_width: int = 4,
            background_width: int = 20
    ) -> bool:
        """
        Show horizontal lines.

        Args:
            foreground_color: Line color
            background_color: Background color
            foreground_width: Line width
            background_width: Width of spaces between lines

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "horizontal_lines",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "foreground_width": foreground_width,
                "background_width": background_width
            }
        )

        if success and response:
            logger.info("Horizontal lines projected")
            return True
        else:
            logger.error("Error projecting horizontal lines")
            return False

    def show_vertical_lines(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            foreground_width: int = 4,
            background_width: int = 20
    ) -> bool:
        """
        Show vertical lines.

        Args:
            foreground_color: Line color
            background_color: Background color
            foreground_width: Line width
            background_width: Width of spaces between lines

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "vertical_lines",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "foreground_width": foreground_width,
                "background_width": background_width
            }
        )

        if success and response:
            logger.info("Vertical lines projected")
            return True
        else:
            logger.error("Error projecting vertical lines")
            return False

    def show_grid(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            h_foreground_width: int = 4,
            h_background_width: int = 20,
            v_foreground_width: int = 4,
            v_background_width: int = 20
    ) -> bool:
        """
        Show a grid.

        Args:
            foreground_color: Line color
            background_color: Background color
            h_foreground_width: Horizontal line width
            h_background_width: Width of spaces between horizontal lines
            v_foreground_width: Vertical line width
            v_background_width: Width of spaces between vertical lines

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "grid",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "h_foreground_width": h_foreground_width,
                "h_background_width": h_background_width,
                "v_foreground_width": v_foreground_width,
                "v_background_width": v_background_width
            }
        )

        if success and response:
            logger.info("Grid projected")
            return True
        else:
            logger.error("Error projecting grid")
            return False

    def show_checkerboard(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            horizontal_count: int = 8,
            vertical_count: int = 6
    ) -> bool:
        """
        Show a checkerboard.

        Args:
            foreground_color: First square color
            background_color: Second square color
            horizontal_count: Number of horizontal squares
            vertical_count: Number of vertical squares

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "checkerboard",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "horizontal_count": horizontal_count,
                "vertical_count": vertical_count
            }
        )

        if success and response:
            logger.info("Checkerboard projected")
            return True
        else:
            logger.error("Error projecting checkerboard")
            return False

    def show_colorbars(self) -> bool:
        """
        Show color bars.

        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {"pattern_type": "colorbars"}
        )

        if success and response:
            logger.info("Color bars projected")
            return True
        else:
            logger.error("Error projecting color bars")
            return False

    def set_standby(self) -> bool:
        """
        Put the projector in a standby-equivalent state safely.
        Uses a black field instead of standby mode to avoid I2C timeout issues.

        Returns:
            True if successful, False otherwise
        """
        try:
            # First set test pattern mode
            mode_success = self.set_test_pattern_mode()
            if not mode_success:
                logger.warning("Unable to set test pattern mode, still trying to project black")

            # Project a black field (visual equivalent of standby)
            black_success = self.show_solid_field("Black")

            # Consider operation successful if we can at least project black
            return black_success

        except Exception as e:
            logger.error(f"Error setting 'visual standby': {e}")
            return False

    def set_test_pattern_mode(self) -> bool:
        """
        Set the projector in test pattern mode.

        Returns:
            True if successful, False otherwise
        """
        return self.set_mode("TestPatternGenerator")
        
    def start_pattern_sequence(self,
                              patterns: List[Dict[str, Any]],
                              interval: float = 0.5,
                              loop: bool = False,
                              sync_with_camera: bool = False,
                              start_immediately: bool = True,
                              max_retries: int = 3) -> Dict[str, Any]:
        """
        Define and start a pattern sequence on the projector.
        
        Args:
            patterns: List of pattern dictionaries defining the sequence
            interval: Time in seconds between pattern changes (default: 0.5s)
            loop: Whether to loop the sequence when completed (default: False)
            sync_with_camera: Whether to synchronize with camera captures (default: False)
            start_immediately: Whether to start the sequence immediately (default: True)
            
        Returns:
            Dictionary with sequence information or None if failed
        
        Example:
            ```python
            # Define a sequence of patterns
            patterns = [
                {"pattern_type": "solid_field", "color": "White"},
                {"pattern_type": "horizontal_lines", "foreground_color": "White", 
                  "background_color": "Black", "foreground_width": 4, "background_width": 20},
                {"pattern_type": "vertical_lines", "foreground_color": "White", 
                  "background_color": "Black", "foreground_width": 4, "background_width": 20},
                {"pattern_type": "grid", "foreground_color": "White", "background_color": "Black", 
                  "h_foreground_width": 4, "h_background_width": 20, 
                  "v_foreground_width": 4, "v_background_width": 20}
            ]
            
            # Start the sequence with 1s interval and looping enabled
            projector.start_pattern_sequence(patterns, interval=1.0, loop=True)
            ```
        """
        # First check if there is already an active sequence
        if self.sequence_active:
            # Stop the current sequence first
            self.stop_pattern_sequence()
            logger.info("Stopping current pattern sequence before starting a new one")
        
        # Update local state
        self.current_pattern_sequence = patterns
        
        # Send sequence to server
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN_SEQUENCE,
            {
                "sequence": patterns,
                "interval": interval,
                "loop": loop,
                "sync_with_camera": sync_with_camera,
                "start": start_immediately
            }
        )
        
        if success and response:
            # Update local state
            self.sequence_active = start_immediately
            self.current_sequence_id = f"seq_{int(time.time())}"
            
            logger.info(f"Pattern sequence started: {len(patterns)} patterns, "
                       f"interval: {interval}s, loop: {loop}")
            # Normalize the response - convert Message object to dictionary
            if hasattr(response, 'payload'):
                return response.payload
            return response
        else:
            error_msg = ""
            if response:
                if hasattr(response, 'payload'):
                    error_msg = response.payload.get("error_message", "Unknown error")
                else:
                    error_msg = response.get("error_message", "Unknown error")
            else:
                error_msg = "No response"
            
            logger.error(f"Error starting pattern sequence: {error_msg}")
            return None
    
    def step_pattern_sequence(self, steps: int = 1) -> Dict[str, Any]:
        """
        Step through the pattern sequence manually.
        
        Args:
            steps: Number of steps to advance (default: 1)
            
        Returns:
            Dictionary with step information or None if failed
        """
        if not self.current_pattern_sequence:
            logger.error("Cannot step sequence: no sequence defined")
            return None
            
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN_SEQUENCE_STEP,
            {"steps": steps}
        )
        
        if success and response:
            logger.info(f"Pattern sequence stepped by {steps}")
            # Normalize the response - convert Message object to dictionary
            if hasattr(response, 'payload'):
                return response.payload
            return response
        else:
            error_msg = ""
            if response:
                if hasattr(response, 'payload'):
                    error_msg = response.payload.get("error_message", "Unknown error")
                else:
                    error_msg = response.get("error_message", "Unknown error")
            else:
                error_msg = "No response"
            
            logger.error(f"Error stepping pattern sequence: {error_msg}")
            return None
    
    def stop_pattern_sequence(self, final_pattern: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Stop the currently running pattern sequence.
        
        Args:
            final_pattern: Optional pattern to display after stopping the sequence
            
        Returns:
            Dictionary with stop information or None if failed
        """
        # Only attempt to stop if we have an active sequence
        if not self.sequence_active:
            logger.info("No active pattern sequence to stop")
            return {"success": True, "was_running": False}
        
        payload = {}
        if final_pattern:
            payload["final_pattern"] = final_pattern
            
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN_SEQUENCE_STOP,
            payload
        )
        
        if success and response:
            # Update local state
            self.sequence_active = False
            logger.info("Pattern sequence stopped")
            # Normalize the response - convert Message object to dictionary
            if hasattr(response, 'payload'):
                return response.payload
            return response
        else:
            error_msg = ""
            if response:
                if hasattr(response, 'payload'):
                    error_msg = response.payload.get("error_message", "Unknown error")
                else:
                    error_msg = response.get("error_message", "Unknown error")
            else:
                error_msg = "No response"
            
            logger.error(f"Error stopping pattern sequence: {error_msg}")
            return None
    
    def set_led_current(self, red_current: int, green_current: int, blue_current: int) -> bool:
        """
        Set LED current for RGB channels in the DLP projector.
        
        Args:
            red_current: Red LED current (0-1023, where 1023 = 100%)
            green_current: Green LED current (0-1023)
            blue_current: Blue LED current (0-1023)
            
        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.LED_SET_CURRENT,
            {
                "red_current": red_current,
                "green_current": green_current,
                "blue_current": blue_current
            }
        )
        
        if success and response:
            logger.info(f"LED current set - R:{red_current}, G:{green_current}, B:{blue_current}")
            return True
        else:
            error_msg = self._get_error_message(response)
            logger.error(f"Failed to set LED current: {error_msg}")
            return False
    
    def get_led_current(self) -> Optional[Tuple[int, int, int]]:
        """
        Get current LED current values.
        
        Returns:
            Tuple of (red, green, blue) current values (0-1023), or None if error
        """
        success, response, _ = self.client.send_message(
            MessageType.LED_GET_CURRENT,
            {}
        )
        
        if success and response:
            payload = response.payload if hasattr(response, 'payload') else response
            return (
                payload.get("red_current", 0),
                payload.get("green_current", 0),
                payload.get("blue_current", 0)
            )
        else:
            error_msg = self._get_error_message(response)
            logger.error(f"Failed to get LED current: {error_msg}")
            return None
    
    def set_led_enable(self, red_enable: bool = True, green_enable: bool = True, blue_enable: bool = True) -> bool:
        """
        Enable or disable individual LED channels.
        
        Args:
            red_enable: Enable red LED
            green_enable: Enable green LED
            blue_enable: Enable blue LED
            
        Returns:
            True if successful, False otherwise
        """
        success, response, _ = self.client.send_message(
            MessageType.LED_SET_ENABLE,
            {
                "red_enable": red_enable,
                "green_enable": green_enable,
                "blue_enable": blue_enable
            }
        )
        
        if success and response:
            logger.info(f"LED enable set - R:{red_enable}, G:{green_enable}, B:{blue_enable}")
            return True
        else:
            error_msg = self._get_error_message(response)
            logger.error(f"Failed to set LED enable: {error_msg}")
            return False
    
    def set_led_intensity_percent(self, red_percent: float, green_percent: float, blue_percent: float) -> bool:
        """
        Set LED intensity as percentage (0-100%).
        
        Args:
            red_percent: Red LED intensity (0-100%)
            green_percent: Green LED intensity (0-100%)
            blue_percent: Blue LED intensity (0-100%)
            
        Returns:
            True if successful, False otherwise
        """
        # Convert percentage to 0-1023 range
        red_current = int((red_percent / 100.0) * 1023)
        green_current = int((green_percent / 100.0) * 1023)
        blue_current = int((blue_percent / 100.0) * 1023)
        
        return self.set_led_current(red_current, green_current, blue_current)
    
    def _get_error_message(self, response) -> str:
        """Extract error message from response."""
        if not response:
            return "No response"
        if hasattr(response, 'payload'):
            return response.payload.get("error_message", "Unknown error")
        elif isinstance(response, dict):
            return response.get("error_message", "Unknown error")
        return str(response)
            
    def create_pattern(self, pattern_type: str, **kwargs) -> Dict[str, Any]:
        """
        Create a pattern definition dictionary for use in pattern sequences.
        
        Args:
            pattern_type: Type of pattern ('solid_field', 'horizontal_lines', 'vertical_lines', 'grid', 'checkerboard')
            **kwargs: Pattern-specific parameters
            
        Returns:
            Pattern definition dictionary
            
        Example:
            ```python
            # Create a white solid field pattern
            white_pattern = projector.create_pattern('solid_field', color='White')
            
            # Create a horizontal lines pattern
            lines_pattern = projector.create_pattern('horizontal_lines', 
                                                    foreground_color='White', 
                                                    background_color='Black',
                                                    foreground_width=4,
                                                    background_width=20)
            ```
        """
        pattern = {"pattern_type": pattern_type}
        pattern.update(kwargs)
        return pattern
        
    def project_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Project a pattern on the projector.
        
        This method handles structured light patterns including gray codes and phase shifts.
        
        Args:
            pattern: Pattern dictionary containing:
                - pattern_type: Type of pattern (solid_field, gray_code, phase_shift, etc.)
                - image: Numpy array with the pattern image (for gray codes and phase shifts)
                - Other parameters specific to the pattern type
        
        Returns:
            True if successful, False otherwise
        """
        pattern_type = pattern.get("pattern_type", "unknown")
        pattern_name = pattern.get("name", "unnamed")
        
        logger.info(f"ProjectorClient.project_pattern: type={pattern_type}, name={pattern_name}")
        
        # Log all pattern parameters for debugging
        logger.debug(f"Pattern details: {[k for k in pattern.keys() if k != 'image']}")
        
        # Handle different pattern types
        try:
            if pattern_type == "solid_field":
                # Use the existing solid field method
                color = pattern.get("color", "White")
                return self.show_solid_field(color)
                
            elif pattern_type == "gray_code":
                # Convert Gray code patterns to line patterns
                orientation = pattern.get("orientation", "horizontal")
                bit = pattern.get("bit", 0)
                inverted = pattern.get("inverted", False)
                
                logger.info(f"Converting Gray code pattern (bit={bit}, inv={inverted}, orientation={orientation})")
                
                # Calculate stripe width based on bit position
                # Higher bits = wider stripes
                stripe_width = 2 ** bit
                
                # Choose colors based on inverted flag
                if inverted:
                    fg_color = "Black"
                    bg_color = "White"
                else:
                    fg_color = "White"
                    bg_color = "Black"
                
                # Project appropriate line pattern
                if orientation == "horizontal":
                    return self.show_horizontal_lines(
                        foreground_color=fg_color,
                        background_color=bg_color,
                        foreground_width=stripe_width,
                        background_width=stripe_width
                    )
                else:  # vertical
                    return self.show_vertical_lines(
                        foreground_color=fg_color,
                        background_color=bg_color,
                        foreground_width=stripe_width,
                        background_width=stripe_width
                    )
                    
            elif pattern_type == "phase_shift":
                # Phase shift patterns - simulate with varying line widths
                orientation = pattern.get("orientation", "horizontal")
                step = pattern.get("step", 0)
                total_steps = pattern.get("total_steps", 12)
                
                logger.info(f"Converting phase shift pattern (step={step}/{total_steps}, orientation={orientation})")
                
                # For phase shift, we alternate between bright and dark
                if step < total_steps // 2:
                    fg_color = "White"
                    bg_color = "Black"
                else:
                    fg_color = "Black"
                    bg_color = "White"
                
                # Use varying line widths to simulate sinusoidal pattern
                width = 8 + int(4 * abs((step % 4) - 2))
                
                # Project appropriate line pattern
                if orientation == "horizontal":
                    return self.show_horizontal_lines(
                        foreground_color=fg_color,
                        background_color=bg_color,
                        foreground_width=width,
                        background_width=width
                    )
                else:  # vertical
                    return self.show_vertical_lines(
                        foreground_color=fg_color,
                        background_color=bg_color,
                        foreground_width=width,
                        background_width=width
                    )
                    
            elif pattern_type == "enhanced_gray":
                # Enhanced gray patterns should already be processed as gray_code patterns
                logger.warning(f"Enhanced gray pattern should have been processed as gray_code. Using fallback.")
                # Enhanced gray patterns contain the actual pattern data
                if "image" in pattern:
                    # For now, project a simple checkerboard pattern as proxy
                    return self.show_checkerboard(
                        foreground_color="White",
                        background_color="Black",
                        horizontal_count=20,
                        vertical_count=20
                    )
                else:
                    return self.show_solid_field("White")
                    
            elif pattern_type == "custom":
                # Handle custom pattern types
                logger.info(f"Projecting custom pattern type: {pattern_type}")
                
                # For custom patterns, use checkerboard as fallback
                logger.warning(f"Custom pattern type {pattern_type} not directly supported, using checkerboard")
                return self.show_checkerboard(
                    foreground_color="White",
                    background_color="Black",
                    horizontal_count=10,
                    vertical_count=10
                )
                    
            else:
                logger.warning(f"Unsupported pattern type: {pattern_type}")
                # Try to use any existing pattern methods
                if pattern_type == "checkerboard":
                    return self.show_checkerboard(**pattern.get("params", {}))
                elif pattern_type == "horizontal_lines":
                    return self.show_horizontal_lines(**pattern.get("params", {}))
                elif pattern_type == "vertical_lines":
                    return self.show_vertical_lines(**pattern.get("params", {}))
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Error projecting pattern: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def create_structured_light_sequence(self, base_pattern_type: str = "horizontal_lines", 
                                       steps: int = 8, **kwargs) -> List[Dict[str, Any]]:
        """
        Create a structured light pattern sequence for 3D scanning.
        
        Args:
            base_pattern_type: Type of pattern to use ('horizontal_lines' or 'vertical_lines')
            steps: Number of pattern phase shifts to generate
            **kwargs: Additional pattern parameters
            
        Returns:
            List of pattern dictionaries suitable for pattern_sequence
            
        Example:
            ```python
            # Create an 8-step phase-shifted horizontal line pattern sequence
            patterns = projector.create_structured_light_sequence('horizontal_lines', steps=8)
            projector.start_pattern_sequence(patterns, interval=0.2)
            ```
        """
        if base_pattern_type not in ['horizontal_lines', 'vertical_lines']:
            logger.warning(f"Unsupported pattern type for structured light: {base_pattern_type}")
            base_pattern_type = 'horizontal_lines'
            
        # Default parameters
        params = {
            "foreground_color": "White",
            "background_color": "Black",
            "foreground_width": 4,
            "background_width": 4,
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        # Create the pattern sequence
        patterns = []
        
        # Create phase-shifted patterns
        base_width = params["foreground_width"] + params["background_width"]
        for i in range(steps):
            # Calculate phase shift
            shift = (i * base_width) // steps
            
            # Create pattern with shift
            pattern = {
                "pattern_type": base_pattern_type,
                "foreground_color": params["foreground_color"],
                "background_color": params["background_color"],
                "foreground_width": params["foreground_width"],
                "background_width": params["background_width"],
                "phase_shift": shift  # This will be handled by the server
            }
            patterns.append(pattern)
            
        # Add a white and black pattern at the beginning and end for reference
        patterns.insert(0, {"pattern_type": "solid_field", "color": "White"})
        patterns.append({"pattern_type": "solid_field", "color": "Black"})
        
        return patterns