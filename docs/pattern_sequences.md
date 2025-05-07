# Pattern Sequence API Documentation

The Pattern Sequence API allows fine-grained control over projector patterns with automatic timing and camera synchronization capabilities. This feature is particularly useful for structured light scanning, calibration, and advanced projection applications.

## Overview

Pattern sequences enable:

- Automatic, timed projection of multiple patterns
- Precise camera-projector synchronization
- Event-based notifications for pattern changes
- Controlled sequential scanning workflows

## Basic Usage

### Creating and Starting a Pattern Sequence

```python
from unlook import UnlookClient

# Connect to scanner
client = UnlookClient()
client.connect(scanner_address, scanner_port)

# Define a sequence of patterns
patterns = [
    {"pattern_type": "solid_field", "color": "White"},
    {"pattern_type": "horizontal_lines", "foreground_color": "White", 
     "background_color": "Black", "foreground_width": 4, "background_width": 20},
    {"pattern_type": "vertical_lines", "foreground_color": "White", 
     "background_color": "Black", "foreground_width": 4, "background_width": 20},
    {"pattern_type": "grid", "foreground_color": "White", "background_color": "Black"}
]

# Start the sequence with 1s interval, looping
result = client.projector.start_pattern_sequence(
    patterns=patterns,
    interval=1.0,      # 1 second between patterns
    loop=True,         # Loop continuously
    sync_with_camera=True  # Enable projector-camera synchronization
)
```

### Stopping a Pattern Sequence

```python
# Stop the sequence and show a black pattern when done
client.projector.stop_pattern_sequence(
    final_pattern={"pattern_type": "solid_field", "color": "Black"}
)
```

### Manual Stepping Through a Sequence

```python
# Define a sequence without starting it
client.projector.start_pattern_sequence(
    patterns=patterns,
    start_immediately=False
)

# Step through the sequence manually
client.projector.step_pattern_sequence()  # Step to first pattern
client.projector.step_pattern_sequence()  # Step to second pattern
```

## Event Callbacks

Register for pattern sequence events to get notifications:

```python
# Register pattern changed callback
def on_pattern_changed(data):
    print(f"Pattern changed: {data.get('pattern_type')}, index: {data.get('index')}")
    
client.projector.on_pattern_changed(on_pattern_changed)

# Register sequence completed callback
def on_sequence_completed(data):
    print(f"Sequence completed")
    
client.projector.on_sequence_completed(on_sequence_completed)
```

Available events:
- `on_pattern_changed` - Triggered when a pattern changes
- `on_sequence_started` - Triggered when a sequence starts
- `on_sequence_stepped` - Triggered when advancing to the next pattern
- `on_sequence_completed` - Triggered when a non-looping sequence completes
- `on_sequence_stopped` - Triggered when a sequence is stopped manually

## Helper Methods

### Creating Structured Light Sequences

The SDK provides helper methods to create structured light pattern sequences:

```python
# Create a structured light sequence with 8 phase shifts
patterns = client.projector.create_structured_light_sequence(
    base_pattern_type="horizontal_lines",  # or "vertical_lines"
    steps=8,                              # Number of phase shifts
    foreground_width=4,                   # Line width
    background_width=4                    # Space width
)

# Start the structured light sequence
client.projector.start_pattern_sequence(patterns, interval=0.2)
```

### Creating Pattern Definitions

Create individual pattern definitions using the helper method:

```python
# Create pattern definitions
white_pattern = client.projector.create_pattern('solid_field', color='White')

grid_pattern = client.projector.create_pattern(
    'grid',
    foreground_color='White',
    background_color='Black',
    h_foreground_width=4,
    h_background_width=20,
    v_foreground_width=4,
    v_background_width=20
)

# Create a sequence from these patterns
patterns = [white_pattern, grid_pattern]
```

## Available Pattern Types

The following pattern types are supported:

1. **Solid Field**
   ```python
   {"pattern_type": "solid_field", "color": "White"}
   ```
   Available colors: "White", "Black", "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow"

2. **Horizontal Lines**
   ```python
   {"pattern_type": "horizontal_lines", 
    "foreground_color": "White", 
    "background_color": "Black", 
    "foreground_width": 4, 
    "background_width": 20}
   ```

3. **Vertical Lines**
   ```python
   {"pattern_type": "vertical_lines", 
    "foreground_color": "White", 
    "background_color": "Black", 
    "foreground_width": 4, 
    "background_width": 20}
   ```

4. **Grid**
   ```python
   {"pattern_type": "grid", 
    "foreground_color": "White", 
    "background_color": "Black", 
    "h_foreground_width": 4, 
    "h_background_width": 20,
    "v_foreground_width": 4, 
    "v_background_width": 20}
   ```

5. **Checkerboard**
   ```python
   {"pattern_type": "checkerboard", 
    "foreground_color": "White", 
    "background_color": "Black", 
    "horizontal_count": 8, 
    "vertical_count": 6}
   ```

6. **Color Bars**
   ```python
   {"pattern_type": "colorbars"}
   ```

## Camera Synchronization

When `sync_with_camera=True` is enabled, the SDK will notify the camera subsystem when patterns change. This allows for:

- Capturing frames at precise moments after pattern changes
- Multi-capture structured light scanning
- Frame-pattern correlation for 3D reconstruction

To use camera synchronization:

```python
# Start a sequence with camera sync
client.projector.start_pattern_sequence(
    patterns=patterns,
    interval=0.5,
    sync_with_camera=True
)

# Configure the camera client to listen for pattern changes
client.stream.start_direct_stream(
    "camera_id",
    callback=frame_handler,
    fps=60,
    sync_with_projector=True  # Enable projector sync
)

# The frame_handler will receive pattern info in frame metadata
def frame_handler(frame_data):
    if frame_data.get("is_sync_frame"):
        pattern_info = frame_data.get("pattern_info", {})
        print(f"Frame synchronized with pattern: {pattern_info.get('pattern_type')}")
```

## Complete Example

See the `pattern_sequence_example.py` script in the `examples` folder for a complete, runnable example.