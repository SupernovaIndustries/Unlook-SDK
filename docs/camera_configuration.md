# UnLook SDK Camera Configuration Guide

This guide explains how to use the advanced camera configuration options in the UnLook SDK. These options allow you to fine-tune camera settings for different scanning scenarios, giving you control over:

- Image quality and compression
- Image resolution and cropping
- Color modes
- Exposure and gain settings
- Image processing options (sharpness, contrast, etc.)

## Table of Contents

1. [Basic Configuration](#basic-configuration)
2. [Setting Image Quality](#setting-image-quality)
3. [Using Different Image Formats](#using-different-image-formats)
4. [Controlling Image Size](#controlling-image-size)
5. [Using Color Modes](#using-color-modes)
6. [Exposure and Gain Control](#exposure-and-gain-control)
7. [Image Processing Options](#image-processing-options)
8. [Configuration Presets](#configuration-presets)
9. [Advanced Usage in 3D Scanning](#advanced-usage-in-3d-scanning)

## Basic Configuration

The `CameraConfig` class allows you to configure all camera settings in a single object. 

```python
from unlook.client.camera_config import CameraConfig, ColorMode, CompressionFormat, ImageQualityPreset

# Create a new camera configuration
config = CameraConfig()

# Configure basic settings
config.exposure_time = 20000  # in microseconds (20ms)
config.gain = 1.5
config.jpeg_quality = 90
config.color_mode = ColorMode.COLOR

# Apply configuration to a camera
client.camera.apply_camera_config("camera_1", config)
```

## Setting Image Quality

You can configure image quality using presets or by specifying explicit compression settings:

```python
# Using a preset (lowest, low, medium, high, highest, lossless)
config.set_quality_preset(ImageQualityPreset.HIGH)

# Or set compression format and quality manually
config.set_compression(CompressionFormat.JPEG, jpeg_quality=90)
```

You can also set these directly in the scanner configuration:

```python
# Configure scanner with quality preset
scanner_config = RealTimeScannerConfig()
scanner_config.set_image_quality(ImageQualityPreset.HIGHEST)

# Or set image format explicitly
scanner_config.set_image_format(CompressionFormat.PNG)
```

## Using Different Image Formats

The UnLook SDK supports three image formats:

- **JPEG**: Good balance of quality and file size, lossy compression
- **PNG**: Lossless compression, larger files but preserves all details
- **RAW**: Uncompressed raw image data, largest file size but no quality loss

```python
# Configure for PNG lossless format
config.set_compression(CompressionFormat.PNG)

# Configure for JPEG with 85% quality
config.set_compression(CompressionFormat.JPEG, jpeg_quality=85)

# Configure for raw uncompressed format
config.set_compression(CompressionFormat.RAW)
```

## Controlling Image Size

You can control image size through resolution settings and region of interest (ROI) cropping:

```python
# Set full camera resolution
config.set_resolution(1920, 1080)

# Set ROI crop to center region (x, y, width, height)
config.set_crop_region(480, 270, 960, 540)

# Reset to full frame (no cropping)
config.reset_crop_region()
```

In scanner configuration:

```python
# Set scan resolution
scanner_config.set_resolution(1280, 720)

# Set crop region
scanner_config.set_crop_region(320, 180, 640, 360)
```

## Using Color Modes

Choose between color and grayscale modes:

```python
# Set to full color mode
config.set_color_mode(ColorMode.COLOR)

# Set to grayscale for pattern detection or efficiency
config.set_color_mode(ColorMode.GRAYSCALE)
```

In scanner configuration:

```python
scanner_config.set_color_mode(ColorMode.GRAYSCALE)
```

## Exposure and Gain Control

Fine-tune camera exposure and gain settings:

```python
# Manual exposure settings
config.set_exposure(exposure_time=16000, auto_exposure=False)

# Auto exposure
config.set_exposure(exposure_time=10000, auto_exposure=True)

# Manual gain control
config.set_gain(gain=1.5, auto_gain=False)
```

## Image Processing Options

Enhance image quality with processing options:

```python
# Set image adjustments
config.set_image_adjustments(
    brightness=0.1,     # Slightly brighter (-1.0 to 1.0)
    contrast=1.2,       # Slightly higher contrast
    saturation=1.1,     # Slightly more saturated
    sharpness=0.5,      # Medium sharpness
    gamma=1.0           # Linear gamma
)

# Set image processing features
config.set_image_processing(
    denoise=True,       # Enable noise reduction
    hdr_mode=False,     # Disable HDR mode
    stabilization=False # Disable image stabilization
)
```

In scanner configuration:

```python
scanner_config.set_image_processing(
    denoise=True,
    contrast=1.2,
    brightness=0.05,
    sharpness=0.5
)
```

## Configuration Presets

Use predefined presets for common scanning scenarios:

```python
# Camera configuration presets
camera_config = CameraConfig.create_preset("scanning")
camera_config = CameraConfig.create_preset("streaming")
camera_config = CameraConfig.create_preset("high_quality")
camera_config = CameraConfig.create_preset("low_light")

# Apply a preset
client.camera.apply_camera_config("camera_1", camera_config)
```

## Advanced Usage in 3D Scanning

Configure camera settings for different scanning scenarios:

```python
from unlook import UnlookClient
from unlook.client.camera_config import ImageQualityPreset, CompressionFormat
from unlook.client.scan_config import RealTimeScannerConfig, PatternType, ScanningMode

# Create client and connect to scanner
client = UnlookClient()
client.connect_to_scanner("scanner_1")

# Create scanner with advanced configuration
scanner = client.real_time_scanner
config = RealTimeScannerConfig()

# Configure scanner for high quality
config.set_pattern_type(PatternType.ADVANCED_PHASE_SHIFT)
config.set_image_quality(ImageQualityPreset.HIGH)
config.set_image_format(CompressionFormat.PNG)
config.set_color_mode(ColorMode.GRAYSCALE)
config.set_image_processing(denoise=True, contrast=1.2)

# Apply configuration
scanner.configure(config)

# Start scanning
scanner.start()
```

## Example: Optimizing for Different Scenarios

### High Detail Scanning
```python
config = RealTimeScannerConfig()
config.set_pattern_type(PatternType.ADVANCED_PHASE_SHIFT)
config.set_image_quality(ImageQualityPreset.HIGHEST)
config.set_image_format(CompressionFormat.PNG)
config.set_color_mode(ColorMode.GRAYSCALE)
config.set_camera_settings(exposure_time=30000, gain=1.0)
config.set_image_processing(contrast=1.2, sharpness=0.7)
```

### Fast Scanning
```python
config = RealTimeScannerConfig()
config.set_pattern_type(PatternType.GRAY_CODE)
config.set_image_quality(ImageQualityPreset.LOW)
config.set_image_format(CompressionFormat.JPEG)
config.set_color_mode(ColorMode.GRAYSCALE)
config.set_camera_settings(exposure_time=5000, gain=1.5)
```

### Color Texture Capture
```python
config = RealTimeScannerConfig()
config.set_pattern_type(PatternType.ADVANCED_GRAY_CODE)
config.set_image_quality(ImageQualityPreset.HIGH)
config.set_image_format(CompressionFormat.JPEG)
config.set_color_mode(ColorMode.COLOR)
config.set_camera_settings(exposure_time=20000, gain=1.0)
config.set_image_processing(contrast=1.1, saturation=1.2, brightness=0.05, sharpness=0.5)
```

## Troubleshooting

If you encounter issues with camera configuration:

1. **Camera not responding to settings**: Make sure the camera is correctly initialized and supports the requested features.

2. **Poor image quality**: Try adjusting exposure time, gain, and image processing settings. For structured light scanning, grayscale mode with enhanced contrast often works best.

3. **Too dark/bright images**: Adjust exposure_time, gain, and brightness settings. Consider using the auto_exposure option for changing lighting conditions.

4. **Performance issues**: Lower resolution, use JPEG compression, and set a lower quality preset for faster performance.

5. **Memory usage concerns**: Use crop regions to capture only the needed part of the scene, reducing memory requirements.