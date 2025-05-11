# Unlook SDK Installation Guide

This guide provides comprehensive installation instructions for the Unlook SDK, including all required and optional dependencies.

## Basic Installation

### Prerequisites

- Python 3.7+ (Python 3.9 or 3.10 recommended)
- Git
- pip (package installer for Python)

### Core Installation

```bash
# Clone the repository
git clone https://github.com/SupernovaIndustries/Unlook-SDK.git
cd Unlook-SDK

# Create and activate a virtual environment (recommended)
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
source .venv/bin/activate

# Install core requirements
pip install -r client-requirements.txt
```

## GPU Acceleration Setup

For optimal performance, installing GPU acceleration is recommended.

### OpenCV with CUDA Support (Recommended)

For real-time scanning functionality, OpenCV with CUDA support is highly recommended:

```bash
# Run the automated OpenCV CUDA setup script
python setup_opencv_cuda.py
```

This script will:
1. Check if OpenCV with CUDA is already installed
2. Install it from pre-built packages if possible
3. Build from source with proper CUDA configuration if needed

Alternatively, you can install manually:

```bash
# Option 1: Install pre-built package (easier but may not work with all CUDA versions)
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python-cuda

# Option 2: Build from source (more reliable but takes longer)
python scripts/build_opencv_cuda.py
```

### NVIDIA GPUs

```bash
# Install CUDA Toolkit from NVIDIA's website first:
# https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support (example for CUDA 11.7)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Install CuPy (match with your CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

### AMD GPUs

```bash
# Install ROCm first, following AMD's instructions:
# https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### CPU-only Fallback

If you don't have a compatible GPU, you can still run the SDK with CPU-only support:

```bash
# Install CPU-only versions
pip install torch torchvision
pip install cupy
pip install opencv-contrib-python  # Regular OpenCV without CUDA
```

## Open3D and Open3D-ML

For point cloud processing, Open3D and Open3D-ML are needed:

```bash
# Install latest Open3D (includes ML components)
pip install -U open3d

# If Open3D-ML is not automatically included, install from GitHub:
pip install git+https://github.com/isl-org/Open3D-ML.git
```

## Verifying Installation

After installation, verify that everything is set up correctly:

```bash
# Check GPU and component availability
python -m unlook.utils.check_gpu

# Run a simple test
python -m unlook.examples.test_client
```

## Server Installation (Raspberry Pi)

For the server component running on Raspberry Pi:

```bash
# SSH into your Raspberry Pi
ssh pi@your-pi-address

# Clone the repository
git clone https://github.com/SupernovaIndustries/Unlook-SDK.git
cd Unlook-SDK

# Install server requirements
pip install -r server-requirements.txt

# Install hardware-specific drivers
sudo ./scripts/install_drivers.sh
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure your virtual environment is activated before running scripts
2. **GPU not detected**: Verify CUDA/ROCm installation with `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
3. **Open3D installation fails**: Try installing individual components:
   ```bash
   pip install -U numpy
   pip install -U open3d
   ```

### Getting Help

If you encounter problems, try:

1. Check the troubleshooting section in our documentation
2. Run the GPU check tool: `python -m unlook.utils.check_gpu`
3. Contact support at [info@supernovaindustries.it](mailto:info@supernovaindustries.it)

## Updating

To update your installation:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r client-requirements.txt
```