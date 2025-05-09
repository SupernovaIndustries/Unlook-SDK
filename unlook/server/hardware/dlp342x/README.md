# DLP342X Projector Driver

This directory contains driver code for Texas Instruments DLP342X-based projectors, used in structured light modules of the Unlook scanner.

## Components

- **__init__.py** - Module initialization and imports
- **dlpc342x_i2c.py** - I2C communication interface for the DLPC342X controller
- **packer.py** - Data packing/unpacking utilities for DLP command structures

## Key Features

The DLP342X driver provides low-level control of the projector:

- I2C-based communication with the DLPC342X controller
- Pattern generation and display
- Sequence programming and playback
- Hardware triggering and synchronization
- Status monitoring and error handling

## Hardware Connection

The driver is designed to work with Raspberry Pi or similar hardware:

- Connect the I2C pins (SDA, SCL) from the Raspberry Pi to the projector
- Ensure proper power supply to the projector module
- Use GPIO pins for hardware triggering if needed

## Usage

```python
from unlook.server.hardware.dlp342x.dlpc342x_i2c import DLPC342X

# Initialize the controller
dlp = DLPC342X(i2c_address=0x36, bus_number=1)

# Check connection
if dlp.is_connected():
    print("Connected to DLP projector")
    
# Display a solid white pattern
dlp.display_solid_field(color="white")

# Display a pattern sequence
dlp.program_pattern_sequence([
    {"pattern_type": "solid_field", "color": "white"},
    {"pattern_type": "horizontal_lines", "line_width": 4, "spacing": 4}
])
dlp.start_pattern_sequence(exposure_time_us=33000)
```

## Command Reference

The driver supports these main command categories:

1. **System Control**
   - Power management
   - Status queries
   - Firmware version

2. **Display Control**
   - Display mode selection
   - Resolution settings
   - Color settings

3. **Pattern Control**
   - Solid field patterns
   - Line patterns (horizontal/vertical)
   - Grid patterns
   - Checkerboard patterns
   - Custom bitmap patterns

4. **Sequence Control**
   - Program pattern sequences
   - Set sequence parameters
   - Start/stop sequences
   - Configure triggers

## Hardware Synchronization

The driver supports hardware synchronization features:

- GPIO-based triggering for pattern changes
- Camera synchronization signals
- External trigger inputs

## Error Handling

The driver includes comprehensive error detection:

- Communication errors
- Command validation
- Hardware status checks
- Proper error reporting

## References

This driver is based on the TI DLPC342X controller documentation:

- [DLPC342X Programming Guide](https://www.ti.com/lit/ug/dlpu018c/dlpu018c.pdf)
- [DLPC342X Datasheet](https://www.ti.com/lit/ds/symlink/dlpc3430.pdf)