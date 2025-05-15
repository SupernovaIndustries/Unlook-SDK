# Claude Next Session Notes

## 1. Camera Auto-Optimization Implementation

### Implement intelligent camera settings optimization:
1. **Multi-shot Calibration Approach**
   - Take 3 reference images:
     - Shot 1: Ambient light only (projector off)
     - Shot 2: Solid white projection
     - Shot 3: Test pattern (coarse stripes)
   
2. **Auto-Adjustment Algorithm**
   - Calculate optimal exposure for pattern visibility
   - Determine best gain settings without saturation
   - Find ideal contrast that preserves object details
   - Optimize for pattern-to-ambient ratio
   - Consider HDR mode when appropriate
   
3. **Implementation Details**
   - Create `camera_auto_optimizer.py` module
   - Use picamera2 library controls
   - Store optimal settings per scene
   - Provide manual override options
   - Support future IR VCSEL integration

### PiCamera2 Integration:
- Study controls API: https://github.com/raspberrypi/picamera2
- Access exposure, gain, AWB, contrast settings
- Implement real-time adjustment during calibration
- Save settings profiles for different scenarios

## 2. Advanced Pattern Projection Systems

### Replace Gray codes with more robust patterns:

1. **Maze Pattern System**
   - Unique topology at every region
   - Natural correspondence points at intersections
   - More robust to partial occlusions
   - Position encoded in maze structure
   - Implementation: `patterns/maze_pattern.py`

2. **Voronoi Pattern System**
   - Organic-looking but mathematically unique
   - Each cell has unique neighbor configuration
   - Natural feature points at vertices
   - Robust to noise and partial visibility
   - Implementation: `patterns/voronoi_pattern.py`

3. **Hybrid ArUco + Dense Pattern**
   - Sparse ArUco markers for coarse alignment
   - Dense geometric patterns between markers
   - Best of both worlds approach
   - Fast correspondence with high accuracy
   - Implementation: `patterns/hybrid_aruco.py`

### Pattern Implementation Strategy:
- Create new pattern modules in `client/patterns/`
- Each pattern type gets its own generator and decoder
- Support for pattern preview and testing
- Compatibility with future IR VCSEL upgrade
- Maintain backward compatibility with Gray codes

## 3. Module Organization

### New file structure:
```
unlook/client/
├── camera/
│   ├── camera_auto_optimizer.py  (NEW)
│   └── ...
├── patterns/
│   ├── maze_pattern.py          (NEW)
│   ├── voronoi_pattern.py       (NEW)
│   ├── hybrid_aruco.py          (NEW)
│   └── ...
└── projector/
    ├── pattern_projector.py     (UPDATED)
    └── ...
```

## 4. IR VCSEL Preparation

### Design patterns with IR in mind:
- Higher density patterns (invisible spectrum)
- Temporal encoding possibilities
- Multi-pattern superposition
- Reduced ambient light interference
- Seamless transition from visible to IR

## 5. Implementation Priority

1. **Phase 1**: Camera auto-optimization
2. **Phase 2**: Maze pattern implementation
3. **Phase 3**: Voronoi pattern system
4. **Phase 4**: Hybrid ArUco approach
5. **Phase 5**: IR VCSEL integration prep

## Custom Pattern Projection for Easy Point Triangulation (Previous Notes)

### Explore implementing a custom pattern projection system that simplifies triangulation:

1. **Unique Marker Patterns**
   - Design patterns with easily identifiable unique markers
   - Could use ArUco markers or custom QR-like codes
   - Each marker has a unique ID that's easily decoded
   - Markers placed at known positions in the projector space
   - Would eliminate correspondence matching issues

2. **Color-Coded Patterns**
   - Use RGB channels to encode additional information
   - Red channel: horizontal position
   - Green channel: vertical position
   - Blue channel: error correction or unique ID
   - Would allow single-shot triangulation

3. **Hybrid Approach**
   - Combine traditional Gray codes with unique markers
   - Use markers for coarse positioning
   - Use Gray codes for fine positioning between markers
   - Best of both worlds: accuracy and reliability

4. **Time-Multiplexed Unique Points**
   - Project a sequence of unique dot patterns
   - Each dot has a temporal signature (blinking pattern)
   - Easy to track and match between cameras
   - No ambiguity in correspondence

5. **Structured Random Patterns**
   - Use pseudo-random but deterministic patterns
   - Each local neighborhood is unique
   - Can decode position from local pattern analysis
   - Similar to Microsoft Kinect approach

### Benefits:
- Eliminate correspondence matching ambiguity
- Faster processing (no complex matching algorithms)
- More robust in challenging lighting conditions
- Works better with limited patterns
- Could enable real-time processing more easily

### Implementation Ideas:
- Start with simple unique markers (ArUco-style)
- Test with small grid of markers first
- Gradually increase density
- Compare accuracy with current Gray code approach
- Optimize for specific use cases (handheld vs static scanning)

### Reference Materials:
- Look into Kinect v1 infrared pattern design
- Research coded structured light papers
- Study modern TOF/structured light hybrids
- Intel RealSense pattern designs

This could be a major improvement for the Unlook SDK, especially for:
- Handheld scanning (where fewer patterns = better)
- Real-time applications
- Low-contrast or reflective surfaces
- Beginner users (more forgiving of setup issues)