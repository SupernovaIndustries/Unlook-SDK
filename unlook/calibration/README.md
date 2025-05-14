# Unlook SDK Calibration

This folder contains calibration files for the Unlook structured light scanner.

## File Structure

- `default/` - Default calibration files for common scanner configurations
- `custom/` - User-specific calibration files

## Calibration File Format

Unlook SDK supports two calibration file formats:

1. **JSON Format** (Preferred)
   ```json
   {
     "camera_matrix_left": [...],
     "dist_coeffs_left": [...],
     "camera_matrix_right": [...],
     "dist_coeffs_right": [...],
     "R": [...],
     "T": [...],
     "baseline_mm": 80.0
   }
   ```

2. **NumPy Format** (.npy)
   Contains the same parameters as above, saved in a NumPy array.

## Automatic Calibration Loading

The SDK will automatically search for calibration files in the following order:
1. Explicitly provided path
2. Custom calibration directory
3. Default calibration directory
4. Built-in default values

## Camera Setup Reference

Standard setups supported by default calibrations:
- Raspberry Pi Camera Modules (v2)
- FLIR Cameras (Blackfly S)
- Standard USB webcams