"""
Client for controlling the UnLook scanner projector.
"""

import logging
import time
from typing import Dict, Optional, Any, List

from ..core.protocol import MessageType

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