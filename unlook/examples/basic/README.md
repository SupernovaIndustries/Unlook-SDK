# UnLook Basic Examples

Welcome to UnLook - The Arduino of Computer Vision!

These examples show how simple it is to use UnLook for 3D vision. Each example is designed to be as simple as possible while demonstrating core functionality.

## Getting Started

The simplest way to use UnLook:

```python
from unlook.simple import quick_capture, quick_scan

# Capture an image in one line
quick_capture("photo.jpg")

# Do a 3D scan in one line  
quick_scan("object.ply")
```

## Basic Examples

### 1. Hello UnLook (`hello_unlook.py`)
- Test connection to your UnLook scanner
- Like the "blink LED" example for Arduino
- Shows available cameras

### 2. One Line Capture (`one_line_capture.py`)
- Capture and save an image with one line of code
- Simplest possible example

### 3. One Line Scan (`one_line_scan.py`)
- Perform a 3D scan with one line of code
- Automatic configuration

### 4. Capture Image (`capture_image.py`)
- Capture a single image
- Save it to disk
- ~10 lines of code

### 5. Simple Scan (`simple_scan.py`)
- Perform a basic 3D scan
- Save point cloud
- Everything automatic
- ~15 lines of code

### 6. Live Preview (`live_preview.py`)
- Stream live video from camera
- Press ESC to quit
- ~20 lines of code

### 7. Super Simple Demo (`super_simple_demo.py`)
- Complete workflow in under 20 lines
- Capture image + 3D scan + save results

## Key Concepts

1. **Auto-discovery**: UnLook automatically finds your scanner
2. **Smart defaults**: Everything works with minimal configuration
3. **Simple API**: Common tasks are one-liners
4. **Progressive complexity**: Advanced features available when needed

## Next Steps

Once you're comfortable with these basics:
- Check out `/advanced/` folder for more complex examples
- Read the full documentation for advanced features
- Join our community to share your projects

Remember: With UnLook, 3D vision is as easy as taking a photo!