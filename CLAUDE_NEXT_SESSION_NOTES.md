# UnLook SDK - Next Session Notes (Updated 2025-05-17)

## Current Status & What We've Done

### Camera Configuration Issues (SOLVED)
1. **Fixed exposure control** - Camera settings now properly apply on server side
2. **Tested various exposure settings**:
   - 50,000μs (50ms): Severe overexposure, inverted references
   - 500μs: Best results, dynamic range 17.9
   - 100μs: Similar to 500μs
   - 50μs: Slightly worse, dynamic range 11.9  
3. **Identified ambient light interference** - Purple/blue cast affecting all scans

### Key Issues Still Present
1. **Low dynamic range** (always under 20) despite various settings
2. **Zero correspondences** - Pattern decoding failing
3. **Ambient light interference** - Can't scan in darkness during investor demos
4. **Missing debug outputs** - Only photos in 01_patterns folder, no triangulation data

## Hardware Upgrade Decision: VCSEL LEDs

### Why VCSEL is the Solution
- Works in fully lit rooms (invisible IR patterns)
- 10-20x better contrast than visible light
- Cameras already have IR filters removed (ready to go!)
- Industry standard for professional 3D scanners

### Implementation Plan
1. Replace LED projector with IR VCSEL array
2. Keep existing DLP controller  
3. Update wavelength parameters in software
4. Test with ambient room lighting

## Critical Code Issues to Fix

### 1. Triangulation Algorithm Review
Current issues:
- Debug folders empty (no decoded patterns, correspondences, triangulation)
- Zero correspondences despite pattern capture
- Need to integrate our debug tools as main processing modules

### 2. Reference Repositories to Study
Study these for proper triangulation implementation:
- https://github.com/3271130/3D-Reconstruction-Project-Using-Phase-Shifting-Method-and-Gray-Code-with-Stereo-Camera
- https://github.com/jakobwilm/slstudio  
- https://github.com/feiran-l/Structured-light-stereo

Key areas to examine:
- How they handle Gray code decoding
- Correspondence finding algorithms
- Triangulation mathematics
- Disparity to depth conversion

### 3. Code Cleanup Requirements
**IMPORTANT**: Keep it simple - "Arduino of Computer Vision"
- NO new files - use only existing modules
- Consolidate debug tools into main processing
- Remove experimental code paths
- Focus on single, working implementation

## Specific Tasks for Next Session

### 1. Fix Debug Output
- Ensure all debug folders are populated:
  - 02_rectified ✓ (working)
  - 03_decoded (missing)
  - 04_correspondence (missing) 
  - 05_triangulation (missing)
  - 06_point_cloud (missing)

### Extra - urgently needed for another investor demo:
- Analyze this repo https://github.com/TemugeB/handpose3d and integrate realtime handpose detection / gestures, as a module in client/scanning/handpose
- Analyze this repo https://github.com/TemugeB/bodypose3d and integrate realtime bodypose detection / gestures, as a module in client/scanning/bodypose

### 2. Rewrite Triangulation Pipeline
Based on reference repos, implement proper:
- Gray code decoding with ambient light tolerance
- Robust correspondence finding
- Correct triangulation using calibration matrices
- Disparity map generation
- Point cloud creation

### 3. Integrate Debug Tools
Convert our analysis scripts into main processing:
- `analyze_captured_images_v2.py` → main pattern decoder
- `depth_map_diagnostic.py` → correspondence finder
- `diagnose_pattern_issue_v2.py` → adaptive threshold calculator

### 4. Simplify Architecture
Current files to consolidate:
- `static_scanner.py` - Keep as main scanner
- `enhanced_gray_code.py` - Simplify pattern decoding
- `proper_correspondence_finder.py` - Merge into main pipeline
- Remove all experimental alternatives

## Testing Protocol for VCSEL Upgrade

1. **Baseline Test** (current visible light)
   - Document current performance metrics
   - Save sample scans for comparison

2. **VCSEL Integration**
   - Install IR VCSEL array
   - Update projector wavelength settings
   - Adjust camera exposure for IR

3. **Performance Validation**
   - Test in various lighting conditions
   - Measure dynamic range improvement
   - Verify correspondence detection
   - Compare point cloud quality

## Key Reminders

1. **Investor Demo Constraints**
   - Must work in normal room lighting
   - Should produce visible results quickly
   - Keep setup simple and reliable

2. **Code Philosophy**
   - "Arduino of Computer Vision" - simple, accessible, modular
   - One way to do things, not multiple options
   - Clear, documented algorithms
   - No magic numbers or complex heuristics

3. **Debug First, Optimize Later**
   - Get basic triangulation working
   - Ensure all debug outputs populated
   - Then optimize for speed/quality

## Questions to Answer

1. Why are correspondences failing with current patterns?
2. Is the triangulation math correct for our camera setup?
3. What threshold values work best for ambient light?
4. How do the reference repos handle similar issues?

## Next Session Priority

1. Study reference repository triangulation methods
2. Fix debug output generation
3. Rewrite correspondence finding based on proven methods
4. Test with VCSEL when hardware arrives
5. Prepare reliable demo for investors

## Settings Discovered This Session

### Best Camera Configuration (Ambient Light)
```python
# For demos in lit rooms:
exposure_time = 2000  # 2ms - moderate exposure
analog_gain = 2.0     # Higher gain for patterns
# Ultra-low threshold for pattern detection
min_threshold = 2     # For dynamic range < 30
```

### Debug Path Structure Expected
```
unlook_debug/scan_*/
├── 01_patterns/
│   └── raw/              ✓ Working
├── 02_rectified/         ✓ Working  
├── 03_decoded/           ✗ Missing
├── 04_correspondence/    ✗ Missing
├── 05_triangulation/     ✗ Missing
├── 06_point_cloud/       ✗ Missing
└── diagnostics/          ✓ Partial
```

Remember: Keep it simple, make it work reliably, then optimize!