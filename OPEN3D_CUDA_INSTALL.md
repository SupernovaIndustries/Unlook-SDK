# Installing Open3D with CUDA Support

This guide explains how to install Open3D with CUDA support for the UnLook SDK.

## Prerequisites

1. **CUDA Toolkit**: Install the latest CUDA Toolkit from NVIDIA
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Recommended version: CUDA 11.7 or 12.x

2. **Python environment**: A virtual environment is recommended
   - Python 3.8-3.10 is recommended for best compatibility

## Installation Options

### Option 1: PyPI Installation (Easiest)

Recent Open3D releases on PyPI include CUDA support:

```bash
# Make sure pip is updated
pip install --upgrade pip

# Install Open3D with CUDA support
pip install open3d==0.17.0

# Verify installation
python -c "import open3d as o3d; print(f'Open3D {o3d.__version__}'); o3d.core.initialize_cuda_device(); print('CUDA initialized')"
```

If successful, you'll see: `CUDA initialized`

### Option 2: Conda Installation

```bash
# Create conda environment
conda create -n unlook python=3.10
conda activate unlook

# Install Open3D with CUDA from conda-forge
conda install -c conda-forge open3d

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: Build from Source (Advanced)

For maximum performance, you can build Open3D from source:

1. Clone Open3D repository
```bash
git clone --recursive https://github.com/isl-org/Open3D
cd Open3D
```

2. Configure with CMake
```bash
mkdir build && cd build
cmake -DBUILD_CUDA_MODULE=ON -DBUILD_PYTORCH_OPS=ON -DBUILD_TENSORFLOW_OPS=ON ..
```

3. Build and install
```bash
make -j$(nproc)
make install-pip-package
```

## Troubleshooting CUDA Support

If CUDA support is not detected, check:

1. **CUDA_PATH environment variable**: Make sure it points to your CUDA installation
```bash
# Windows
echo %CUDA_PATH%

# Linux/macOS
echo $CUDA_PATH
```

2. **Run the CUDA setup helper**:
```bash
python setup_cuda_env.py
```

3. **Check Open3D CUDA status**:
```python
import open3d as o3d

# See if CUDA attributes are available
print(hasattr(o3d, 'core'))

# Try to initialize CUDA
try:
    o3d.core.initialize_cuda_device()
    print("CUDA initialized successfully")
except Exception as e:
    print(f"CUDA initialization failed: {e}")
```

## Installing Open3D ML

To enable all neural network features:

```bash
pip install open3d-ml

# With PyTorch backend
pip install torch torchvision

# With TensorFlow backend (optional)
pip install tensorflow
```

## Testing Installation

Run the CUDA test script to verify everything works:

```bash
python -c "
import open3d as o3d
import numpy as np

print(f'Open3D version: {o3d.__version__}')

# Check for CUDA support
has_cuda = False
try:
    o3d.core.initialize_cuda_device()
    has_cuda = True
    print('CUDA initialized successfully')
except:
    print('CUDA initialization failed')

# Create test data
points = np.random.rand(10000, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

if has_cuda and hasattr(o3d.core, 'Tensor'):
    # Try CUDA operation
    test_tensor = o3d.core.Tensor.ones((3, 3), o3d.core.Dtype.Float32, o3d.core.Device('CUDA:0'))
    print(f'Created CUDA tensor: {test_tensor}')
"
```