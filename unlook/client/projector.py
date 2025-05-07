"""
Client for controlling the UnLook scanner projector.
"""

import logging
import time
from typing import Dict, Optional, Any, List, Union, Callable

from ..core.protocol import MessageType
from ..core.events import EventType, EventEmitter

logger = logging.getLogger(__name__)


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
        
        # Event callbacks dictionary
        self._event_callbacks = {
            EventType.PROJECTOR_PATTERN_CHANGED: [],
            EventType.PROJECTOR_SEQUENCE_STARTED: [],
            EventType.PROJECTOR_SEQUENCE_STEPPED: [],
            EventType.PROJECTOR_SEQUENCE_COMPLETED: [],
            EventType.PROJECTOR_SEQUENCE_STOPPED: []
        }
    
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
                              start_immediately: bool = True) -> Dict[str, Any]:
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
            return response
        else:
            error_msg = response.get("error_message", "Unknown error") if response else "No response"
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
            return response
        else:
            error_msg = response.get("error_message", "Unknown error") if response else "No response"
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
            return response
        else:
            error_msg = response.get("error_message", "Unknown error") if response else "No response"
            logger.error(f"Error stopping pattern sequence: {error_msg}")
            return None
            
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