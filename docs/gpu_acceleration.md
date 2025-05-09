# GPU Acceleration for UnLook SDK

This document explains how to set up GPU acceleration for the UnLook SDK to achieve faster real-time 3D scanning.

## Overview

The UnLook SDK supports GPU acceleration for several compute-intensive tasks:

1. **Point Cloud Processing** - Using CUDA-accelerated OpenCV operations
2. **Neural Network Enhancement** - Using PyTorch for point cloud filtering and enhancement
3. **Array Operations** - Using CuPy for faster array processing

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.x or 12.x recommended)
- Appropriate GPU drivers installed
- Python packages: PyTorch, CuPy (matching your CUDA version)

## Installation

### 1. Install Base Requirements

First, install the base requirements:

```bash
pip install -r client-requirements.txt
```

### 2. Install CUDA Toolkit

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your system.

### 3. Install CuPy

Install CuPy matching your CUDA version:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# For CPU-only (fallback)
pip install cupy
```

### 4. Verify Installation

You can verify your GPU acceleration setup with:

```python
import torch
import cupy
import cv2

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

print(f"CuPy available: {hasattr(cupy, 'cuda')}")
print(f"OpenCV CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
```

## Configuring GPU Usage

The UnLook SDK automatically detects and uses GPU acceleration when available. You can control this behavior with options in the `ScanConfig` and `RealTimeScanConfig` classes:

```python
# Disable GPU acceleration if needed
scanner.config.use_gpu = False

# Disable neural network enhancement
scanner.config.use_neural_network = False
```

## Troubleshooting

1. **"No module named 'cupy'"** - Install CuPy matching your CUDA version
2. **"CUDA initialization failed"** - Check your CUDA installation and GPU drivers
3. **"No CUDA-capable device is detected"** - Ensure your GPU is properly recognized
4. **Poor performance with GPU acceleration** - Try adjusting voxel sizes for downsampling

## Related Documentation

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV CUDA Documentation](https://docs.opencv.org/master/d1/d1a/namespacecv_1_1cuda.html)