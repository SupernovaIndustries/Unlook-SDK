Pattern Decoder Module
=====================

The pattern decoder module handles decoding of structured light patterns for 3D reconstruction.

.. module:: unlook.client.scanner.pattern_decoder

Overview
--------

The :class:`PatternDecoder` provides utilities for decoding various structured light patterns:

- Gray code pattern decoding
- Phase shift pattern decoding
- Phase unwrapping with Gray code guidance
- Debug visualization generation

Classes
-------

.. autoclass:: PatternDecoder
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: decode_gray_code
   .. automethod:: decode_phase_shift
   .. automethod:: unwrap_phase

Pattern Decoding Methods
-----------------------

Gray Code Decoding
~~~~~~~~~~~~~~~~~~

Gray code patterns encode projector coordinates using binary patterns::

    from unlook.client.scanner import PatternDecoder
    
    # Decode Gray code patterns
    x_coords, y_coords, mask = PatternDecoder.decode_gray_code(
        images,           # List of captured images
        pattern_width,    # Projector width
        pattern_height,   # Projector height
        threshold=5.0,    # Decoding threshold
        debug_dir="debug" # Optional debug output
    )

The method returns:

- ``x_coords``: Projector X coordinates for each pixel
- ``y_coords``: Projector Y coordinates for each pixel  
- ``mask``: Valid pixel mask (where decoding succeeded)

Phase Shift Decoding
~~~~~~~~~~~~~~~~~~~

Phase shift patterns use sinusoidal patterns for sub-pixel accuracy::

    # Decode phase shift patterns
    phase_map, modulation = PatternDecoder.decode_phase_shift(
        images,        # Phase-shifted images
        num_shifts=4,  # Number of phase shifts
        threshold=5.0  # Minimum modulation
    )

Returns:

- ``phase_map``: Wrapped phase values (-π to π)
- ``modulation``: Modulation amplitude (quality metric)

Phase Unwrapping
~~~~~~~~~~~~~~~

Combine Gray code and phase shift for absolute coordinates::

    # Unwrap phase using Gray code
    absolute_coords = PatternDecoder.unwrap_phase(
        phase_map,     # Wrapped phase from phase shift
        gray_code_x,   # X coordinates from Gray code
        gray_code_y    # Y coordinates from Gray code
    )

Image Requirements
-----------------

Gray Code Images
~~~~~~~~~~~~~~~

Required image sequence:

1. White reference image (all projector pixels on)
2. Black reference image (all projector pixels off)
3. Horizontal Gray code patterns (normal and inverted pairs)
4. Vertical Gray code patterns (normal and inverted pairs)

Total images: 2 + 2×log₂(width) + 2×log₂(height)

Phase Shift Images
~~~~~~~~~~~~~~~~~

Required images:

- N phase-shifted sinusoidal patterns (typically 3-12)
- Same frequency, different phases (2π/N apart)

Debug Visualization
------------------

When ``debug_dir`` is specified, the decoder saves:

- ``x_coordinates.png``: Decoded X coordinates
- ``y_coordinates.png``: Decoded Y coordinates  
- ``valid_mask.png``: Valid pixel mask
- ``threshold.png``: Threshold visualization
- ``coordinates_color.png``: Color-coded coordinate map

Example Workflow
---------------

Complete decoding pipeline::

    # 1. Capture patterns
    white_ref = capture_image("white")
    black_ref = capture_image("black")
    gray_images = [capture_image(f"gray_{i}") for i in range(num_gray)]
    phase_images = [capture_image(f"phase_{i}") for i in range(num_phase)]
    
    # 2. Decode Gray code
    all_images = [white_ref, black_ref] + gray_images
    x_gray, y_gray, mask = PatternDecoder.decode_gray_code(
        all_images, proj_width, proj_height
    )
    
    # 3. Decode phase shift  
    phase, modulation = PatternDecoder.decode_phase_shift(
        phase_images, num_shifts=4
    )
    
    # 4. Unwrap phase for sub-pixel accuracy
    x_absolute = PatternDecoder.unwrap_phase(phase, x_gray)

Best Practices
-------------

1. **Threshold Selection**: Start with threshold=5.0, adjust based on ambient light
2. **Pattern Count**: More patterns = better accuracy but longer capture time
3. **Debug Output**: Always save debug images during development
4. **Image Quality**: Ensure good contrast between black/white references

See Also
--------

- :doc:`structured_light` - Pattern generation
- :doc:`triangulator` - 3D reconstruction from correspondences
- :doc:`../examples/realtime_scanning` - Complete scanning example