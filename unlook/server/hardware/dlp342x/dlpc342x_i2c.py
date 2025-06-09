import struct
import time
import numpy as np
from enum import Enum
from smbus2 import SMBus
import logging
from . import packer

# Configure logging
logger = logging.getLogger(__name__)


# Enums for projector settings
class OperatingMode(Enum):
    ExternalVideoPort = 0
    TestPatternGenerator = 1
    SplashScreen = 2
    SensExternalPattern = 3
    SensInternalPattern = 4
    SensSplashPattern = 5
    Standby = 255


class Color(Enum):
    Black = 0
    Red = 1
    Green = 2
    Blue = 3
    Cyan = 4
    Magenta = 5
    Yellow = 6
    White = 7


class DiagonalLineSpacing(Enum):
    Dls3 = 3
    Dls7 = 7
    Dls15 = 15
    Dls31 = 31
    Dls63 = 63
    Dls127 = 127
    Dls255 = 255


class BorderEnable(Enum):
    Disable = 0
    Enable = 1


class TestPattern(Enum):
    SolidField = 0
    HorizontalRamp = 1
    VerticalRamp = 2
    HorizontalLines = 3
    DiagonalLines = 4
    VerticalLines = 5
    Grid = 6
    Checkerboard = 7
    Colorbars = 8


class GridLines:
    """Class for grid lines configuration, equivalent to the one in original dlpc342x.py"""

    def __init__(self):
        self.Border = BorderEnable.Enable
        self.BackgroundColor = Color.Black
        self.ForegroundColor = Color.White
        self.HorizontalForegroundLineWidth = 4
        self.HorizontalBackgroundLineWidth = 20
        self.VerticalForegroundLineWidth = 4
        self.VerticalBackgroundLineWidth = 20


class DLPC342XController:
    """Controller for DLPC342X projector via I2C."""

    def __init__(self, bus=3, address=0x1b):
        """Initialize the controller.

        Args:
            bus: I2C bus number (usually 3 for Unlook)
            address: I2C device address (default 0x1b)
        """
        self.bus = SMBus(bus)
        self.address = address
        self.summary = {"Command": "", "CommInterface": "I2C", "Successful": True}
        logger.info(f"DLPC342X I2C controller initialized (bus={bus}, address=0x{address:02X})")

    def close(self):
        """Close the I2C bus safely.
        
        This method should be called after all commands have been sent.
        """
        try:
            # Only close if bus still exists and is valid
            if hasattr(self, 'bus') and self.bus is not None:
                self.bus.close()
                self.bus = None
                logger.info("I2C bus closed successfully")
            else:
                logger.warning("I2C bus already closed or invalid")
        except Exception as e:
            logger.error(f"Error closing I2C bus: {e}")
            # Set to None to prevent further use
            self.bus = None

    def _write_command(self, command_bytes):
        """Write a command to the projector.

        Args:
            command_bytes: List of bytes to write

        Returns:
            True if successful, False otherwise
        """
        # Check if bus is valid
        if not hasattr(self, 'bus') or self.bus is None:
            logger.error("I2C bus is closed or invalid, cannot write command")
            return False
            
        try:
            # The first byte is the command, the rest are parameters
            command = command_bytes[0]
            data = command_bytes[1:] if len(command_bytes) > 1 else []

            # Write to I2C
            self.bus.write_i2c_block_data(self.address, command, data)
            logger.debug(f"I2C write: cmd=0x{command:02X}, data={[hex(b) for b in data]}")
            return True
        except Exception as e:
            logger.error(f"I2C write error: {e}")
            # If we get an I/O error, mark the bus as invalid
            if "I/O" in str(e) or "argument must be an int" in str(e):
                logger.warning("I2C bus appears to be in an invalid state, marking as closed")
                self.bus = None
            return False

    def _read_command(self, command_byte, length):
        """Read data from the projector.

        Args:
            command_byte: Command byte to specify what to read
            length: Number of bytes to read

        Returns:
            List of read bytes
        """
        # Check if bus is valid
        if not hasattr(self, 'bus') or self.bus is None:
            logger.error("I2C bus is closed or invalid, cannot read command")
            return [0] * length
            
        try:
            # First write the command to specify what to read
            self.bus.write_byte(self.address, command_byte)
            logger.debug(f"I2C read command: 0x{command_byte:02X}, length={length}")

            # Short delay to ensure device has processed the command
            time.sleep(0.01)

            # Read the response
            read_data = []
            for _ in range(length):
                read_data.append(self.bus.read_byte(self.address))

            logger.debug(f"I2C read response: {[hex(b) for b in read_data]}")
            return read_data
        except Exception as e:
            logger.error(f"I2C read error: {e}")
            # If we get an I/O error, mark the bus as invalid
            if "I/O" in str(e) or "argument must be an int" in str(e):
                logger.warning("I2C bus appears to be in an invalid state, marking as closed")
                self.bus = None
            return [0] * length

    def set_operating_mode(self, mode):
        """Set the operating mode of the projector.

        Args:
            mode: Operating mode (from OperatingMode enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Operating Mode Select"

        try:
            # Pack the command byte and mode value
            command_bytes = [5, mode.value]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Set operating mode to {mode.name}: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error setting operating mode: {e}")
            self.summary["Successful"] = False
            return False

    def get_operating_mode(self):
        """Get the current operating mode.

        Returns:
            OperatingMode enum value, or None if error
        """
        self.summary["Command"] = "Read Operating Mode Select"

        try:
            # Send read command and get response
            response = self._read_command(6, 1)

            if response and len(response) > 0:
                mode_value = response[0]
                mode = OperatingMode(mode_value)
                self.summary["Successful"] = True
                logger.info(f"Current operating mode: {mode.name}")
                return mode
            else:
                self.summary["Successful"] = False
                logger.warning("Failed to get operating mode")
                return None
        except Exception as e:
            logger.error(f"Error getting operating mode: {e}")
            self.summary["Successful"] = False
            return None

    def generate_solid_field(self, color, border=BorderEnable.Enable):
        """Generate a solid field of the specified color.

        Args:
            color: Field color (from Color enum)
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Solid Field"

        try:
            # Pack the data according to the WriteSolidField function
            packer.packerinit()
            value = packer.setbits(0, 4, 0)  # Pattern type 0 = solid field
            value = packer.setbits(border.value, 1, 7)  # Border setting

            packer.packerinit()
            color_value = packer.setbits(color.value, 3, 4)

            # Build the command
            command_bytes = [11, value, color_value]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate solid {color.name} field: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating solid field: {e}")
            self.summary["Successful"] = False
            return False

    def generate_horizontal_lines(self, background_color, foreground_color,
                                  foreground_line_width, background_line_width,
                                  border=BorderEnable.Enable):
        """Generate horizontal lines on the projector.

        Args:
            background_color: Background color (from Color enum)
            foreground_color: Line color (from Color enum)
            foreground_line_width: Width of foreground lines
            background_line_width: Width of background lines
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Horizontal Lines"

        try:
            # Convert to integers if they're strings or not integers
            if not isinstance(foreground_line_width, int):
                try:
                    foreground_line_width = int(foreground_line_width)
                except (ValueError, TypeError):
                    foreground_line_width = 4  # Default
                    logger.warning(f"Invalid foreground_line_width: {foreground_line_width}, using default (4)")
            
            if not isinstance(background_line_width, int):
                try:
                    background_line_width = int(background_line_width)
                except (ValueError, TypeError):
                    background_line_width = 4  # Default
                    logger.warning(f"Invalid background_line_width: {background_line_width}, using default (4)")
            
            # Validate width values to prevent crashes
            foreground_line_width = max(1, min(255, foreground_line_width))
            background_line_width = max(1, min(255, background_line_width))

            # Pack the data according to the WriteHorizontalLines function
            packer.packerinit()
            value = packer.setbits(3, 4, 0)  # Pattern type 3 = horizontal lines
            value = packer.setbits(border.value, 1, 7)  # Border setting

            packer.packerinit()
            color_value = packer.setbits(background_color.value, 3, 0)
            color_value = packer.setbits(foreground_color.value, 3, 4)

            # Build the command
            command_bytes = [11, value, color_value, foreground_line_width, background_line_width]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate horizontal lines: fg={foreground_color.name}({foreground_line_width}px), bg={background_color.name}({background_line_width}px) - {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating horizontal lines: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.summary["Successful"] = False
            return False

    def generate_vertical_lines(self, background_color, foreground_color,
                                foreground_line_width, background_line_width,
                                border=BorderEnable.Enable):
        """Generate vertical lines on the projector.

        Args:
            background_color: Background color (from Color enum)
            foreground_color: Line color (from Color enum)
            foreground_line_width: Width of foreground lines
            background_line_width: Width of background lines
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Vertical Lines"

        try:
            # Convert to integers if they're strings or not integers
            if not isinstance(foreground_line_width, int):
                try:
                    foreground_line_width = int(foreground_line_width)
                except (ValueError, TypeError):
                    foreground_line_width = 4  # Default
                    logger.warning(f"Invalid foreground_line_width: {foreground_line_width}, using default (4)")
            
            if not isinstance(background_line_width, int):
                try:
                    background_line_width = int(background_line_width)
                except (ValueError, TypeError):
                    background_line_width = 4  # Default
                    logger.warning(f"Invalid background_line_width: {background_line_width}, using default (4)")
            
            # Validate width values to prevent crashes
            foreground_line_width = max(1, min(255, foreground_line_width))
            background_line_width = max(1, min(255, background_line_width))

            # Pack the data according to the WriteVerticalLines function
            packer.packerinit()
            value = packer.setbits(5, 4, 0)  # Pattern type 5 = vertical lines
            value = packer.setbits(border.value, 1, 7)  # Border setting

            packer.packerinit()
            color_value = packer.setbits(background_color.value, 3, 0)
            color_value = packer.setbits(foreground_color.value, 3, 4)

            # Build the command
            command_bytes = [11, value, color_value, foreground_line_width, background_line_width]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate vertical lines: fg={foreground_color.name}({foreground_line_width}px), bg={background_color.name}({background_line_width}px) - {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating vertical lines: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.summary["Successful"] = False
            return False

    def generate_diagonal_lines(self, background_color, foreground_color,
                                horizontal_spacing, vertical_spacing,
                                border=BorderEnable.Enable):
        """Generate diagonal lines on the projector.

        Args:
            background_color: Background color (from Color enum)
            foreground_color: Line color (from Color enum)
            horizontal_spacing: Horizontal spacing (from DiagonalLineSpacing enum)
            vertical_spacing: Vertical spacing (from DiagonalLineSpacing enum)
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Diagonal Lines"

        try:
            # Pack the data according to the WriteDiagonalLines function
            packer.packerinit()
            value = packer.setbits(4, 4, 0)  # Pattern type 4 = diagonal lines
            value = packer.setbits(border.value, 1, 7)  # Border setting

            packer.packerinit()
            color_value = packer.setbits(background_color.value, 3, 0)
            color_value = packer.setbits(foreground_color.value, 3, 4)

            # Build the command
            command_bytes = [11, value, color_value, horizontal_spacing.value, vertical_spacing.value]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate diagonal lines: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating diagonal lines: {e}")
            self.summary["Successful"] = False
            return False

    def generate_grid(self, background_color, foreground_color,
                      horizontal_foreground_width, horizontal_background_width,
                      vertical_foreground_width, vertical_background_width,
                      border=BorderEnable.Enable):
        """Generate a grid pattern on the projector.

        Args:
            background_color: Background color (from Color enum)
            foreground_color: Line color (from Color enum)
            horizontal_foreground_width: Width of horizontal foreground lines
            horizontal_background_width: Width of spaces between horizontal lines
            vertical_foreground_width: Width of vertical foreground lines
            vertical_background_width: Width of spaces between vertical lines
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Grid Lines"

        try:
            # Convert to integers if they're strings or not integers
            params = [
                ('horizontal_foreground_width', horizontal_foreground_width, 4),
                ('horizontal_background_width', horizontal_background_width, 20),
                ('vertical_foreground_width', vertical_foreground_width, 4),
                ('vertical_background_width', vertical_background_width, 20)
            ]
            
            result_values = {}
            
            for name, value, default in params:
                if not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        value = default
                        logger.warning(f"Invalid {name}: {value}, using default ({default})")
                
                # Validate width values to prevent crashes
                value = max(1, min(255, value))
                result_values[name] = value
            
            # Extract converted values
            h_fg_width = result_values['horizontal_foreground_width']
            h_bg_width = result_values['horizontal_background_width']
            v_fg_width = result_values['vertical_foreground_width']
            v_bg_width = result_values['vertical_background_width']

            # Pack the data according to the WriteGridLines function
            packer.packerinit()
            value = packer.setbits(6, 4, 0)  # Pattern type 6 = grid
            value = packer.setbits(border.value, 1, 7)  # Border setting

            packer.packerinit()
            color_value = packer.setbits(background_color.value, 3, 0)
            color_value = packer.setbits(foreground_color.value, 3, 4)

            # Build the command
            command_bytes = [
                11, value, color_value,
                h_fg_width, h_bg_width,
                v_fg_width, v_bg_width
            ]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate grid: fg={foreground_color.name}, bg={background_color.name}, "
                       f"h={h_fg_width}/{h_bg_width}, v={v_fg_width}/{v_bg_width} - {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating grid: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.summary["Successful"] = False
            return False

    def generate_grid_from_object(self, grid_lines):
        """Generate a grid pattern using a GridLines object.

        Args:
            grid_lines: GridLines object with grid configuration

        Returns:
            True if successful, False otherwise
        """
        return self.generate_grid(
            grid_lines.BackgroundColor,
            grid_lines.ForegroundColor,
            grid_lines.HorizontalForegroundLineWidth,
            grid_lines.HorizontalBackgroundLineWidth,
            grid_lines.VerticalForegroundLineWidth,
            grid_lines.VerticalBackgroundLineWidth,
            grid_lines.Border
        )

    def generate_checkerboard(self, background_color, foreground_color,
                              horizontal_count, vertical_count,
                              border=BorderEnable.Enable):
        """Generate a checkerboard pattern on the projector.

        Args:
            background_color: Background color (from Color enum)
            foreground_color: Foreground color (from Color enum)
            horizontal_count: Number of horizontal checkers
            vertical_count: Number of vertical checkers
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Checkerboard"

        try:
            # Pack the data according to the WriteCheckerboard function
            packer.packerinit()
            value = packer.setbits(7, 4, 0)  # Pattern type 7 = checkerboard
            value = packer.setbits(border.value, 1, 7)  # Border setting

            packer.packerinit()
            color_value = packer.setbits(background_color.value, 3, 0)
            color_value = packer.setbits(foreground_color.value, 3, 4)

            # Pack horizontal and vertical counts as 16-bit values
            h_count_bytes = list(struct.pack('<H', horizontal_count))
            v_count_bytes = list(struct.pack('<H', vertical_count))

            # Build the command
            command_bytes = [11, value, color_value] + h_count_bytes + v_count_bytes

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate checkerboard: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating checkerboard: {e}")
            self.summary["Successful"] = False
            return False

    def generate_colorbars(self, border=BorderEnable.Enable):
        """Generate color bars pattern on the projector.

        Args:
            border: Whether to enable border (BorderEnable enum)

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Colorbars"

        try:
            # Pack the data according to the WriteColorbars function
            packer.packerinit()
            value = packer.setbits(8, 4, 0)  # Pattern type 8 = colorbars
            value = packer.setbits(border.value, 1, 7)  # Border setting

            # Build the command
            command_bytes = [11, value]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Generate colorbars: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating colorbars: {e}")
            self.summary["Successful"] = False
            return False

    def execute_flash_batch_file(self, batch_file_number):
        """Execute a batch file stored in flash.

        Args:
            batch_file_number: Batch file number to execute

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Execute Flash Batch File"

        try:
            # Build the command
            command_bytes = [45, batch_file_number]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Execute flash batch file {batch_file_number}: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error executing flash batch file: {e}")
            self.summary["Successful"] = False
            return False

    def select_splash_screen(self, splash_screen_index):
        """Select a splash screen.

        Args:
            splash_screen_index: Index of the splash screen

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Splash Screen Select"

        try:
            # Build the command
            command_bytes = [13, splash_screen_index]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Select splash screen {splash_screen_index}: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error selecting splash screen: {e}")
            self.summary["Successful"] = False
            return False

    def execute_splash_screen(self):
        """Execute the selected splash screen.

        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Splash Screen Execute"

        try:
            # Build the command
            command_bytes = [53]

            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success

            logger.info(f"Execute splash screen: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error executing splash screen: {e}")
            self.summary["Successful"] = False
            return False
    
    def set_led_current(self, red_current, green_current, blue_current):
        """Set LED current for RGB channels.
        
        According to DLPC342X documentation, command 0x4B controls RGB LED current.
        
        Args:
            red_current: Red LED current (0-1023, where 1023 = 100%)
            green_current: Green LED current (0-1023)
            blue_current: Blue LED current (0-1023)
            
        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write RGB LED Current"
        
        try:
            # Validate current values (0-1023 range)
            red_current = max(0, min(1023, int(red_current)))
            green_current = max(0, min(1023, int(green_current)))
            blue_current = max(0, min(1023, int(blue_current)))
            
            # Pack current values as 16-bit little-endian
            red_bytes = list(struct.pack('<H', red_current))
            green_bytes = list(struct.pack('<H', green_current))
            blue_bytes = list(struct.pack('<H', blue_current))
            
            # Command 0x4B for RGB LED current control
            command_bytes = [0x4B] + red_bytes + green_bytes + blue_bytes
            
            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success
            
            logger.info(f"Set LED current - R:{red_current}, G:{green_current}, B:{blue_current}: "
                       f"{'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error setting LED current: {e}")
            self.summary["Successful"] = False
            return False
    
    def get_led_current(self):
        """Get current LED current settings.
        
        Returns:
            Tuple of (red, green, blue) current values (0-1023), or None if error
        """
        self.summary["Command"] = "Read RGB LED Current"
        
        try:
            # Command 0x4C to read LED current
            response = self._read_command(0x4C, 6)  # 6 bytes for 3x 16-bit values
            
            if response and len(response) >= 6:
                # Unpack 16-bit values
                red_current = struct.unpack('<H', bytes(response[0:2]))[0]
                green_current = struct.unpack('<H', bytes(response[2:4]))[0]
                blue_current = struct.unpack('<H', bytes(response[4:6]))[0]
                
                self.summary["Successful"] = True
                logger.info(f"Current LED values - R:{red_current}, G:{green_current}, B:{blue_current}")
                return (red_current, green_current, blue_current)
            else:
                self.summary["Successful"] = False
                logger.warning("Failed to get LED current")
                return None
        except Exception as e:
            logger.error(f"Error getting LED current: {e}")
            self.summary["Successful"] = False
            return None
    
    def set_led_enable(self, red_enable, green_enable, blue_enable):
        """Enable or disable individual LED channels.
        
        Args:
            red_enable: Enable red LED (True/False)
            green_enable: Enable green LED (True/False)
            blue_enable: Enable blue LED (True/False)
            
        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write RGB LED Enable"
        
        try:
            # Pack enable bits
            enable_value = 0
            if red_enable:
                enable_value |= (1 << 0)
            if green_enable:
                enable_value |= (1 << 1)
            if blue_enable:
                enable_value |= (1 << 2)
            
            # Command 0x50 for LED enable control
            command_bytes = [0x50, enable_value]
            
            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success
            
            logger.info(f"Set LED enable - R:{red_enable}, G:{green_enable}, B:{blue_enable}: "
                       f"{'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error setting LED enable: {e}")
            self.summary["Successful"] = False
            return False

    def generate_sinusoidal_pattern(self, frequency, phase_shift=0.0, 
                                   orientation="vertical", amplitude=127.5, offset=127.5):
        """Generate sinusoidal pattern using approximation with vertical lines.
        
        Since the DLPC342X doesn't support true sinusoidal patterns, we approximate
        them using variable-width vertical lines that create a sinusoidal intensity profile.
        
        Args:
            frequency: Pattern frequency (cycles across projector width)
            phase_shift: Phase shift in radians (0 to 2π)
            orientation: "vertical" or "horizontal" 
            amplitude: Sinusoidal amplitude (0-127.5 for proper range)
            offset: DC offset (typically 127.5 for 0-255 range)
            
        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Generate Sinusoidal Pattern Approximation"
        
        try:
            logger.info(f"Generating sinusoidal pattern approximation: f={frequency}, "
                       f"phase={phase_shift:.2f}rad, orientation={orientation}")
            
            # For now, use a series of vertical lines with varying widths
            # to approximate sinusoidal pattern
            if orientation == "vertical":
                # Calculate line widths based on sinusoidal function
                # For frequency=1, we want one complete cycle across the projector
                # For frequency=8, we want 8 cycles, etc.
                
                # Use different line patterns for different frequencies/phases
                if frequency == 1:
                    # Low frequency - wide pattern variations
                    if 0 <= phase_shift < np.pi/2:
                        fg_width, bg_width = 8, 24
                    elif np.pi/2 <= phase_shift < np.pi:
                        fg_width, bg_width = 16, 16
                    elif np.pi <= phase_shift < 3*np.pi/2:
                        fg_width, bg_width = 24, 8
                    else:
                        fg_width, bg_width = 16, 16
                        
                elif frequency <= 8:
                    # Medium frequency - moderate variations
                    if 0 <= phase_shift < np.pi/2:
                        fg_width, bg_width = 4, 12
                    elif np.pi/2 <= phase_shift < np.pi:
                        fg_width, bg_width = 8, 8
                    elif np.pi <= phase_shift < 3*np.pi/2:
                        fg_width, bg_width = 12, 4
                    else:
                        fg_width, bg_width = 8, 8
                        
                else:
                    # High frequency - fine patterns
                    if 0 <= phase_shift < np.pi/2:
                        fg_width, bg_width = 2, 6
                    elif np.pi/2 <= phase_shift < np.pi:
                        fg_width, bg_width = 4, 4
                    elif np.pi <= phase_shift < 3*np.pi/2:
                        fg_width, bg_width = 6, 2
                    else:
                        fg_width, bg_width = 4, 4
                
                # Calculate colors based on amplitude and offset
                # For sinusoidal patterns, we vary between background and foreground
                intensity_factor = (np.sin(phase_shift) + 1) / 2  # 0 to 1
                
                # Map to grayscale levels
                bg_intensity = max(0, min(7, int((offset - amplitude) / 255 * 7)))
                fg_intensity = max(0, min(7, int((offset + amplitude) / 255 * 7)))
                
                # Use Color enum values
                bg_color = Color(bg_intensity) if bg_intensity < 8 else Color.White
                fg_color = Color(fg_intensity) if fg_intensity < 8 else Color.White
                
                # Generate the vertical line pattern
                success = self.generate_vertical_lines(
                    background_color=bg_color,
                    foreground_color=fg_color,
                    foreground_line_width=fg_width,
                    background_line_width=bg_width,
                    border=BorderEnable.Disable
                )
                
            else:  # horizontal orientation
                # Similar logic for horizontal patterns
                if frequency == 1:
                    if 0 <= phase_shift < np.pi/2:
                        fg_width, bg_width = 8, 24
                    elif np.pi/2 <= phase_shift < np.pi:
                        fg_width, bg_width = 16, 16
                    elif np.pi <= phase_shift < 3*np.pi/2:
                        fg_width, bg_width = 24, 8
                    else:
                        fg_width, bg_width = 16, 16
                else:
                    # Adjust for higher frequencies
                    base_fg = max(1, 8 // frequency)
                    base_bg = max(1, 16 // frequency)
                    
                    if 0 <= phase_shift < np.pi/2:
                        fg_width, bg_width = base_fg, base_bg * 2
                    elif np.pi/2 <= phase_shift < np.pi:
                        fg_width, bg_width = base_fg * 2, base_bg
                    elif np.pi <= phase_shift < 3*np.pi/2:
                        fg_width, bg_width = base_fg * 3, base_bg // 2
                    else:
                        fg_width, bg_width = base_fg * 2, base_bg
                
                # Similar intensity calculation as vertical
                intensity_factor = (np.sin(phase_shift) + 1) / 2
                bg_intensity = max(0, min(7, int((offset - amplitude) / 255 * 7)))
                fg_intensity = max(0, min(7, int((offset + amplitude) / 255 * 7)))
                
                bg_color = Color(bg_intensity) if bg_intensity < 8 else Color.White
                fg_color = Color(fg_intensity) if fg_intensity < 8 else Color.White
                
                success = self.generate_horizontal_lines(
                    background_color=bg_color,
                    foreground_color=fg_color,
                    foreground_line_width=fg_width,
                    background_line_width=bg_width,
                    border=BorderEnable.Disable
                )
            
            self.summary["Successful"] = success
            
            if success:
                logger.info(f"Sinusoidal pattern approximation generated successfully")
            else:
                logger.error(f"Failed to generate sinusoidal pattern approximation")
                
            return success
            
        except Exception as e:
            logger.error(f"Error generating sinusoidal pattern: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.summary["Successful"] = False
            return False

    def generate_horizontal_ramp(self, start_intensity=0, end_intensity=255):
        """Generate horizontal ramp pattern (can be used for sinusoidal approximation).
        
        Args:
            start_intensity: Starting intensity (0-255)
            end_intensity: Ending intensity (0-255)
            
        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Horizontal Ramp"
        
        try:
            # Pack the data for horizontal ramp (pattern type 1)
            packer.packerinit()
            value = packer.setbits(1, 4, 0)  # Pattern type 1 = horizontal ramp
            value = packer.setbits(0, 1, 7)  # No border
            
            # Build the command - ramp patterns may need intensity parameters
            command_bytes = [11, value, start_intensity, end_intensity]
            
            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success
            
            logger.info(f"Generate horizontal ramp ({start_intensity}->{end_intensity}): {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating horizontal ramp: {e}")
            self.summary["Successful"] = False
            return False

    def generate_vertical_ramp(self, start_intensity=0, end_intensity=255):
        """Generate vertical ramp pattern (can be used for sinusoidal approximation).
        
        Args:
            start_intensity: Starting intensity (0-255)
            end_intensity: Ending intensity (0-255)
            
        Returns:
            True if successful, False otherwise
        """
        self.summary["Command"] = "Write Vertical Ramp"
        
        try:
            # Pack the data for vertical ramp (pattern type 2)
            packer.packerinit()
            value = packer.setbits(2, 4, 0)  # Pattern type 2 = vertical ramp
            value = packer.setbits(0, 1, 7)  # No border
            
            # Build the command
            command_bytes = [11, value, start_intensity, end_intensity]
            
            # Send the command
            success = self._write_command(command_bytes)
            self.summary["Successful"] = success
            
            logger.info(f"Generate vertical ramp ({start_intensity}->{end_intensity}): {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logger.error(f"Error generating vertical ramp: {e}")
            self.summary["Successful"] = False
            return False