# OpenCV with CUDA Installation Guide

This document provides guidance on installing OpenCV with CUDA support for the Unlook SDK, which is essential for optimal performance in real-time 3D scanning applications.

## Automatic Installation

The Unlook SDK provides an automated script that handles the installation of OpenCV with CUDA support:

```bash
python setup_opencv_cuda.py
```

This script will:
1. Check if OpenCV with CUDA is already installed
2. Verify CUDA toolkit availability on your system
3. Attempt to install a pre-built package if available
4. Build from source with the correct configuration if needed

## Manual Installation Options

### Option 1: Pre-built Package (Faster but Less Reliable)

```bash
# First, uninstall any existing OpenCV installations
pip uninstall -y opencv-python opencv-contrib-python

# Install the pre-built CUDA package
pip install opencv-contrib-python-cuda
```

This approach is faster but may not work with all CUDA versions and system configurations.

### Option 2: Build from Source (Slower but More Reliable)

The SDK includes a script that automates the build process:

```bash
python scripts/build_opencv_cuda.py
```

You can customize the build with additional options:

```bash
# Specify CUDA architecture (e.g., 7.5 for RTX 2080, 8.6 for RTX 3090)
python scripts/build_opencv_cuda.py --cuda-arch 8.6

# Specify OpenCV version
python scripts/build_opencv_cuda.py --opencv-version 4.8.0

# Enable debug output
python scripts/build_opencv_cuda.py --debug
```

## Manual Build Process (Advanced)

If you need to build OpenCV with CUDA support manually, follow these steps:

1. **Install CUDA Toolkit**
   - Download and install the NVIDIA CUDA Toolkit matching your GPU
   - https://developer.nvidia.com/cuda-downloads

2. **Install Dependencies**
   - Ubuntu/Debian:
     ```bash
     sudo apt-get update
     sudo apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
         libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
         libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
         libatlas-base-dev gfortran python3-dev
     ```
   - Fedora/RHEL/CentOS:
     ```bash
     sudo dnf install -y gcc gcc-c++ cmake git pkgconfig gtk3-devel \
         ffmpeg-devel libv4l-devel libjpeg-devel libpng-devel libtiff-devel \
         atlas-devel gfortran python3-devel
     ```
   - Windows:
     - Install Visual Studio with C++ components
     - Install CMake

3. **Download OpenCV Source**
   ```bash
   wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
   wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.0.zip
   unzip opencv.zip
   unzip opencv_contrib.zip
   ```

4. **Configure and Build**
   ```bash
   cd opencv-4.8.0
   mkdir build
   cd build
   
   cmake -D CMAKE_BUILD_TYPE=RELEASE \
         -D CMAKE_INSTALL_PREFIX=/usr/local \
         -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.8.0/modules \
         -D WITH_CUDA=ON \
         -D WITH_CUDNN=ON \
         -D OPENCV_DNN_CUDA=ON \
         -D ENABLE_FAST_MATH=1 \
         -D CUDA_FAST_MATH=1 \
         -D CUDA_ARCH_BIN=7.5 \  # Update this for your GPU architecture
         -D WITH_CUBLAS=1 \
         -D BUILD_opencv_python3=ON \
         -D BUILD_opencv_python2=OFF \
         -D PYTHON3_EXECUTABLE=$(which python3) \
         -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
         -D BUILD_EXAMPLES=OFF \
         -D INSTALL_PYTHON_EXAMPLES=OFF \
         -D INSTALL_C_EXAMPLES=OFF \
         -D BUILD_TESTS=OFF \
         -D BUILD_PERF_TESTS=OFF \
         ..
   
   # Build with multiple cores
   make -j$(nproc)
   sudo make install
   ```

## Verifying Installation

Use this Python code to verify OpenCV CUDA support:

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# Check if CUDA is available
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA devices: {count}")
    
    if count > 0:
        # Test GPU operations
        gpu_mat = cv2.cuda.GpuMat()
        test_array = np.zeros((100, 100), dtype=np.float32)
        gpu_mat.upload(test_array)
        
        # Do a simple GPU operation
        result = cv2.cuda.GpuMat()
        cv2.cuda.threshold(gpu_mat, result, 0.5, 1.0, cv2.THRESH_BINARY)
        
        # Download the result
        cpu_result = result.download()
        
        # Release GPU memory
        gpu_mat.release()
        result.release()
        
        print("CUDA test passed!")
    else:
        print("No CUDA devices detected by OpenCV")
except Exception as e:
    print(f"CUDA not available: {e}")
```

## Troubleshooting

### Common Issues

1. **"OpenCV was compiled without CUDA support"**
   - Verify CUDA Toolkit is installed
   - Make sure you're using the OpenCV version with CUDA support
   - Check that your GPU is supported with `nvidia-smi`

2. **"No CUDA devices detected"**
   - Ensure your NVIDIA drivers are installed and working
   - Check that your GPU is detected with `nvidia-smi`
   - Verify that other CUDA applications work

3. **Build errors**
   - Make sure you have sufficient disk space
   - Ensure you have the correct CUDA version for your GPU
   - Check for compatibility between OpenCV version and CUDA version

### Getting Build Information

To get detailed OpenCV build information:

```python
import cv2
print(cv2.getBuildInformation())
```

Look for these sections to confirm CUDA support:

```
NVIDIA CUDA:                    YES (ver 11.8, CUFFT CUBLAS)
  NVIDIA GPU arch:              75
  NVIDIA PTX archs:
OpenCL:                         YES (no OpenGL support)
OpenVINO:                       NO
```

## Further Help

If you encounter issues with OpenCV CUDA installation:

1. Check the Unlook SDK documentation for updates
2. Refer to the OpenCV official documentation
3. Contact support at info@supernovaindustries.it