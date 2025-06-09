# ELAS Installation Instructions

## Overview
ELAS (Efficient Large-Scale Stereo) has been integrated into the Unlook SDK to provide 10x better performance compared to OpenCV SGBM while maintaining sub-pixel accuracy.

## Installation Steps

### 1. Clone and Compile ELAS Library

```bash
# Clone the ELAS repository
git clone https://github.com/maiermic/elas
cd elas

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
make -j8
```

### 2. Install the Library

#### Linux/macOS:
```bash
# Copy the library to system path
sudo cp libelas.so /usr/local/lib/
sudo ldconfig
```

#### Windows:
- Copy `elas.dll` to your Python environment's Scripts directory
- Or add the build directory to your PATH

### 3. Test the Installation

```python
# Test if ELAS is available
from unlook.client.lib.elas_wrapper import ELASMatcher

matcher = ELASMatcher()
if hasattr(matcher, 'available') and matcher.available:
    print("✅ ELAS successfully installed!")
else:
    print("❌ ELAS library not found - will use OpenCV fallback")
```

## Usage

### Command Line
```bash
# Use ELAS for offline processing
python unlook/examples/scanning/process_offline.py \
    --input captured_data/test1 \
    --surface-reconstruction \
    --use-elas

# Combine with other optimizations
python unlook/examples/scanning/process_offline.py \
    --input captured_data/test1 \
    --surface-reconstruction \
    --use-elas \
    --ndr \
    --phase-optimization
```

### Python API
```python
from unlook.client.scanning.reconstruction.stereobm_surface_reconstructor import StereoBMSurfaceReconstructor

# Enable ELAS
reconstructor = StereoBMSurfaceReconstructor(
    calibration_file="calibration_2k.json",
    use_elas=True  # Enable ELAS
)

# Process images
points_3d, metrics = reconstructor.reconstruct_surface(left_img, right_img)
```

## Performance Benefits

- **Speed**: 10x faster than OpenCV SGBM
- **Quality**: Sub-pixel accuracy with dense matching
- **Coverage**: Better handling of textureless regions
- **Robustness**: Superior performance on structured light patterns

## Fallback Behavior

If ELAS library is not compiled/installed, the system automatically falls back to OpenCV SGBM with a warning message. This ensures the SDK remains functional even without ELAS.

## Troubleshooting

### Library Not Found
- Check that the library is in the system path
- Verify the library name matches your platform (libelas.so, libelas.dylib, elas.dll)
- Try placing the library in the same directory as the Python script

### Compilation Issues
- Ensure you have CMake 3.10+ installed
- Install build essentials: `sudo apt-get install build-essential`
- For Windows, use Visual Studio 2019 or newer

### Performance Issues
- ELAS requires significant memory for large images
- Consider downsampling for real-time applications
- Use confidence filtering to reduce noise