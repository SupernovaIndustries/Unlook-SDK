# Claude Fixes Session - May 15, 2025

## Important Fixes and Learnings

### 1. Pattern Generation System

We implemented new pattern types for better 3D scanning:
- **Maze patterns** - Complex patterns for improved correspondence matching
- **Voronoi patterns** - Dense surface reconstruction patterns
- **Hybrid ArUco patterns** - Patterns with markers for registration

These are located in: `unlook/client/patterns/`

### 2. Projector Limitations

**IMPORTANT**: The DLP342X projector does NOT support arbitrary image projection via I2C. It only supports:
- Basic patterns (solid field, lines, grids, checkerboard)
- Pre-loaded splash screens (selected by index)
- External video port mode (requires video input)

We handle custom patterns by approximating them with checkerboard patterns based on image characteristics.

### 3. Camera Import Issues

**Problem**: Circular import issues between camera module and camera directory.

**Solution**: Use absolute import to avoid confusion between camera.py file and camera/ directory:
```python
# In scanner.py
import unlook.client.camera as camera_module
self._camera = camera_module.CameraClient(self)
```

The issue occurs because there's both:
- `unlook/client/camera.py` (file)
- `unlock/client/camera/` (directory)

Python was importing from the camera/ directory instead of camera.py file.

### 4. Calibration Path Structure

Calibration files are located at:
- Default: `unlook/calibration/default/default_stereo.json`
- Custom: `unlook/calibration/custom/stereo_calibration.json`

**Path fixes**:
- In `static_scanner.py`: Use `"..", ".."` (not `"..", "..", "..", ".."`)
- In `static_scanning_example_fixed.py`: Added one more parent to reach correct directory

### 5. Command Line Options

The scanning example supports:
```bash
python static_scanning_example_fixed.py --pattern maze --enhancement-level 3 --quality high --debug
```

Options:
- `--enhancement-level 0-3`: Software pattern enhancement (3 = maximum)
- `--auto-optimize`: Enable camera auto-optimization (enabled by default)
- `--no-auto-optimize`: Disable auto-optimization
- `--pattern`: Choose pattern type (maze, voronoi, hybrid_aruco, enhanced_gray)

### 6. Camera Auto-Optimization

Implemented automatic camera settings optimization:
1. Captures reference images (ambient, pattern, white, black, checkerboard)
2. Analyzes lighting conditions
3. Optimizes exposure and gain settings
4. Applies settings before scan

Lighting classifications:
- Dark: Mean < 30 → High exposure (20-100ms), High gain (2.0-8.0)
- Low light: Mean < 100 → Medium exposure (10-50ms), Medium gain (1.5-4.0)
- Normal: Mean < 180 → Standard exposure (5-20ms), Standard gain (1.0-2.0)
- Bright: Mean ≥ 180 → Low exposure (1-10ms), Low gain (0.5-1.5)

### 7. Code Structure

Key files and their purposes:
- `unlook/client/scanning/static_scanner.py` - Main scanning implementation
- `unlook/client/camera/camera_auto_optimizer.py` - Camera optimization
- `unlook/client/patterns/` - Pattern generators
- `unlook/client/projector/projector.py` - Projector control
- `unlook/examples/scanning/static_scanning_example_fixed.py` - Main example

### 8. Debug Information

When debugging doesn't work, check:
1. Camera hardware connection to server
2. Server has camera support enabled
3. Look for circular import errors
4. Check calibration file paths
5. Verify pattern type support

### 9. Key Development Guidelines

As per CLAUDE.local.md:
- DO NOT create new files when working on existing files
- Always modify existing files directly
- Keep code simple and avoid unnecessary complexity
- User must be able to run scanner with minimal code
- Create comprehensive documentation

### 10. Future Work Items

From CLAUDE.local.md:
- Implement uncertainty measurement for ISO/ASTM 52902 compliance
- Test GPU acceleration when hardware is available
- Create neural network model for point cloud enhancement
- Test with real hardware (cameras, IMU, etc.)
- Create smartphone app for scanner control

### 11. Current Status

As of May 15, 2025:
- ✅ Pattern generation working
- ✅ Projector pattern approximation working
- ✅ Calibration loading correctly
- ✅ Camera auto-optimization implemented
- ❌ Camera hardware not connected (server issue)
- ❌ No point clouds generated (due to missing camera)

The system is ready to scan once cameras are available on the hardware.