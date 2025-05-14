# Scanning Examples

This directory contains all scanning-related examples for the Unlook SDK.

## Main Examples

### 1. Static Scanning
- `static_scanning_example_fixed.py` - The definitive static scanning example with enhanced processor enabled by default

### 2. Real-time Scanning  
- `realtime_scanning_example.py` - Real-time continuous scanning with stereo cameras

### 3. Comprehensive Debugging
- `comprehensive_scan_debug.py` - All-in-one tool for hardware scanning, image processing, and debugging

## Offline Processing

- `scan_from_images.py` - Process pre-captured images without hardware
- `debug_triangulation.py` - Debug triangulation and correspondence issues

## Diagnostic Tools

- `analyze_captured_images_v2.py` - Analyze pattern quality in captured images
- `diagnose_pattern_issue_v2.py` - Diagnose pattern decoding problems
- `test_enhanced_processor.py` - Test enhanced pattern processor

## Emergency/Demo Tools

- `investor_demo_scan.py` - Simple SIFT-based 3D reconstruction for demos
- `emergency_scan_fix.py` - Emergency fixes for problematic captures

## Usage

All examples can be run directly with Python:

```bash
python static_scanning_example_fixed.py
```

For enhanced processing (now default):
```bash
python static_scanning_example_fixed.py --enhancement-level 3
```

To process pre-captured images:
```bash
python scan_from_images.py "path/to/images" --enhanced-processor
```

## Key Features

- Enhanced pattern processor enabled by default
- Maximum enhancement level (3) for challenging conditions
- Automatic handling of inverted reference images
- Support for low-contrast patterns
- Debug output for troubleshooting