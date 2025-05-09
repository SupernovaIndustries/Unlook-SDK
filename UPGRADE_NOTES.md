 Unlook SDK Robust Scanning Implementation Upgrade

## Changes Made

The robust scanning implementation has been completely rewritten to properly use the UnlookClient architecture and communicate with real hardware directly instead of using simulation mode. This upgrade fixes the issue where projector commands weren't being sent to hardware.

### Key Improvements:

1. **No More Simulation Mode**: Removed all reliance on simulation mode in projector and camera clients
2. **Direct Hardware Communication**: All pattern projections now use real hardware projector commands
3. **Enhanced Error Handling**: Better error handling during pattern projection and capture
4. **Performance Optimization**: Debug images are no longer saved by default, improving performance
5. **Improved Phase Shift Patterns**: Fixed issues with "Gray" solid field patterns that caused errors
6. **Command-line Options**: Added `--save-debug-images` option to control debug image saving

### Files Created:

- `unlook/client/new_structured_light.py`: Comprehensive structured light pattern generation and projection
- `unlook/client/new_scanner3d.py`: Complete 3D scanner implementation with proper hardware integration
- `unlook/examples/new_robust_scanning_example.py`: Updated example that works with the new implementation

### Files to Remove:

These files contain the old implementation and should be removed to avoid confusion:

- `unlook/client/robust_scan.py`
- `unlook/client/robust_structured_light.py`
- `unlook/examples/robust_scanning_example.py`

## Usage

To use the new robust scanning implementation:

```bash
cd unlook/examples
python new_robust_scanning_example.py --quality high
```

### Options:

- `--output DIR`: Specify output directory for scan results
- `--calibration FILE`: Path to stereo calibration file
- `--quality {fast,medium,high,ultra}`: Scan quality preset (default: high)
- `--timeout SECONDS`: Timeout for scanner discovery (default: 5)
- `--mesh`: Generate mesh from point cloud
- `--visualize`: Visualize results after scanning
- `--debug`: Enable debug logging
- `--save-debug-images`: Save all captured images (may slow down scanning significantly)

## Debugging

If you encounter issues:
1. Run with `--debug` to get detailed logs
2. Use `--save-debug-images` to save captured images for troubleshooting (only when needed)

## Technical Details

The new implementation properly uses the ProjectorClient to send commands directly to the hardware. 
It combines Gray code and Phase shift techniques for robust 3D reconstruction.

Key components:
- Pattern generation and projection (new_structured_light.py)
- 3D triangulation and point cloud processing (new_scanner3d.py)
- Hardware communication via UnlookClient (new_robust_scanning_example.py)

This implementation no longer attempts to fallback to simulation mode, ensuring that it always communicates with real hardware.