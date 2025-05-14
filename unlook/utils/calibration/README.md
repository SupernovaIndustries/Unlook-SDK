# Camera Calibration Utilities

This directory contains utilities for camera calibration in the Unlook SDK.

## Tools

### 1. Headless Calibration

The `headless_calibration.py` script provides a command-line interface for calibrating stereo cameras without requiring GUI support:

```bash
python -m unlook.utils.calibration.headless_calibration --checkerboard 8x5 --squares-size 19.4 --save-images
```

Key parameters:
- `--checkerboard`: Size of the checkerboard in inner corners (e.g., "8x5")
- `--squares-size`: Size of the checkerboard squares in mm
- `--save-images`: Whether to save calibration images
- `--output`: Path to save the calibration file

### 2. Load Calibration Images

The `load_calibration_images.py` script allows you to calibrate cameras using existing images:

```bash
python -m unlook.utils.calibration.load_calibration_images --images-dir path/to/images --checkerboard 8x5
```

Key parameters:
- `--images-dir`: Directory containing left and right calibration images
- `--checkerboard`: Size of the checkerboard in inner corners
- `--squares-size`: Size of the checkerboard squares in mm
- `--output`: Path to save the calibration file

### 3. Check Calibration

The `check_calibration.py` script verifies and displays parameters from a calibration file:

```bash
python -m unlook.utils.calibration.check_calibration --calibration path/to/calibration.json
```

Key parameters:
- `--calibration`: Path to the calibration file to check

## Using Calibration with Scanning

To use your calibration with the scanning tool:

```bash
python -m unlook.examples.static_scanning_example --calibration path/to/calibration.json
```

## Checkerboard Pattern

For a physical checkerboard with R rows and C columns of squares, the inner corner count is (R-1) × (C-1).

Example: A 9×6 checkerboard (9 columns, 6 rows of squares) has 8×5 inner corners, so use `--checkerboard 8x5`.

## Calibration Tips

1. Ensure the checkerboard is well-lit and visible in both cameras
2. Capture images with the checkerboard at different angles and distances
3. Keep the checkerboard still during each capture
4. Use accurate square size measurements for proper scaling
5. Position the checkerboard within the working range of your scanner (30-80cm)