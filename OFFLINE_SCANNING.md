# Offline Scanning from Pre-captured Images

The offline scanner allows you to process previously captured images without needing the hardware connected. This is extremely useful for:

- Debugging pattern decoding issues
- Testing different processing parameters
- Consistent testing with the same image set
- Working on the algorithm when hardware isn't available

## Usage

### Basic Usage

Process images from a previous scan:

```bash
python unlook/examples/scan_from_images.py path/to/scan_directory
```

For example, with your captured images:

```bash
python unlook/examples/scan_from_images.py "G:\Supernova\Prototipi\UnLook\Software\Unlook-SDK\unlook\examples\unlook_debug\scan_20250514_204726"
```

### With Enhanced Processor

Use the enhanced processor for better results with low-contrast images:

```bash
python unlook/examples/scan_from_images.py path/to/scan --enhanced-processor --enhancement-level 3
```

### With Calibration

Specify a calibration file for accurate 3D reconstruction:

```bash
python unlook/examples/scan_from_images.py path/to/scan --calibration path/to/calibration.json --enhanced-processor
```

### Full Example

Process your specific captured images with all options:

```bash
python unlook/examples/scan_from_images.py "G:\Supernova\Prototipi\UnLook\Software\Unlook-SDK\unlook\examples\unlook_debug\scan_20250514_204726" --output my_scan.ply --calibration unlook/calibration/custom/stereo_calibration.json --enhanced-processor --enhancement-level 3 --debug
```

## Expected Directory Structure

The offline scanner expects images in this structure:

```
scan_directory/
├── 01_patterns/
│   └── raw/
│       ├── pattern_00_black_camera_0.png
│       ├── pattern_00_black_camera_1.png
│       ├── pattern_01_white_camera_0.png
│       ├── pattern_01_white_camera_1.png
│       ├── pattern_02_gray_horizontal_00_camera_0.png
│       ├── pattern_02_gray_horizontal_00_camera_1.png
│       └── ... (more pattern images)
```

Or it can work with simpler naming:

```
scan_directory/
├── left_00.png
├── right_00.png
├── left_01.png
├── right_01.png
└── ...
```

## Image Order

The scanner expects images in this order:
1. Black reference (index 0)
2. White reference (index 1)
3. Horizontal Gray codes (indices 2-11)
4. Vertical Gray codes (indices 12-21)
5. Phase shift patterns (indices 22+, if present)

## Output

The scanner will create:
- `offline_scan.ply`: The reconstructed point cloud
- `correspondences.png`: Visualization of matched points (if --debug is used)

## Troubleshooting

### No images found
- Check the path is correct
- Ensure images are in PNG format
- Verify the directory structure matches expected format

### No correspondences found
- Try using `--enhanced-processor`
- Increase enhancement level to 3
- Check that calibration file is provided
- Verify images contain visible patterns

### Poor point cloud quality
- Ensure correct calibration file is used
- Check that all required images are present
- Try different enhancement levels
- Verify image quality is sufficient

## Benefits

1. **Consistent Testing**: Use the same image set to test algorithm improvements
2. **No Hardware Required**: Debug and develop without the scanner connected
3. **Faster Iteration**: Skip image capture phase during development
4. **Parameter Tuning**: Try different processing parameters on the same data
5. **Debugging**: Isolate algorithm issues from hardware issues