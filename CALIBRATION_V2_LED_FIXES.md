# Calibration Capture Script V2 Protocol and LED Control Fixes

## Overview
Fixed the calibration capture script to use the correct v2 protocol imports and added AS1170 LED controller support for better illumination during checkerboard capture.

## Key Changes Made

### 1. Import Updates
- Changed from `from unlook import UnlookClient` to `from unlook.client.scanner.scanner import UnlookClient`
- Changed from `from unlook.core.discovery import discover_scanners` to `from unlook.core.discovery import DiscoveryService`
- These imports align with the v2 protocol structure

### 2. Client Initialization
- Updated to use v2 protocol client initialization: `UnlookClient("Calibration2KCapture", auto_discover=True)`
- Added proper discovery wait time and scanner connection flow
- Uses `client.get_discovered_scanners()` to get available scanners
- Connects using `client.connect(scanner)` with ScannerInfo objects

### 3. LED Control Integration
- Added LED control in `configure_scanner_2k()` method
- Uses `client.projector.led_set_intensity(led1_mA=0, led2_mA=200)` for illumination
- LED1 is set to 0 (no point projection needed for calibration)
- LED2 is set to 200mA for flood illumination (adjustable via command line)
- Added `--led-intensity` command line argument (0-450mA range)

### 4. Camera Capture Updates
- Updated to use v2 protocol camera methods
- Single camera mode: Uses `client.camera.capture(camera_id)`
- Stereo mode: Uses `client.camera.capture_multi(camera_ids)`
- Properly gets camera list with `client.camera.get_cameras()`
- Handles camera IDs as strings from the camera dictionary

### 5. Error Handling
- Added proper exception handling for LED control
- Graceful fallback if LED control is not available
- Ensures LED is turned off after capture completion

## Usage Examples

### Basic Usage
```bash
python unlook/examples/calibration/capture_checkerboard_images.py
```

### With Custom LED Intensity
```bash
python unlook/examples/calibration/capture_checkerboard_images.py --led-intensity 300
```

### Single Camera Mode
```bash
python unlook/examples/calibration/capture_checkerboard_images.py --single-camera
```

### Full Example with All Options
```bash
python unlook/examples/calibration/capture_checkerboard_images.py \
    --output calibration_2k_images \
    --num-images 40 \
    --checkerboard-columns 9 \
    --checkerboard-rows 6 \
    --square-size 25.0 \
    --delay 2.0 \
    --led-intensity 250
```

## Test Script
Created `test_led_calibration.py` to verify LED control functionality:
- Tests different LED intensities (0-400mA)
- Captures images at each intensity level
- Analyzes mean brightness
- Saves sample images for comparison

## LED Controller Features
The AS1170 LED controller provides:
- Dual channel control (LED1 and LED2)
- 0-450mA current range per channel
- LED1: Typically used for point projection (set to 0 for calibration)
- LED2: Used for flood illumination (recommended 200-300mA for calibration)
- Pulse mode for temporary illumination
- Status monitoring

## Benefits
1. **Better Illumination**: LED provides consistent lighting for checkerboard detection
2. **Adjustable Intensity**: Can optimize for different environments
3. **V2 Protocol**: Uses latest protocol for better performance
4. **Proper Resource Management**: LED automatically turned off when done

## Troubleshooting
- If LED control fails, the script continues without illumination
- Check server logs for AS1170 hardware initialization
- Ensure I2C connection to LED controller is working
- Default intensity of 200mA works well for most scenarios