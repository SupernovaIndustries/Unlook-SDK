#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Acceleration Checker for UnLook SDK.

This utility script checks the availability of GPU acceleration
features in the current environment and reports their status.
"""

import sys
import os
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("unlook.utils.check_gpu")

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def check_cuda():
    """Check CUDA availability."""
    try:
        import cv2
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            print(f"✅ OpenCV CUDA Support: Available ({cuda_count} devices)")
            for i in range(cuda_count):
                device_name = cv2.cuda.getDevice()
                print(f"   - CUDA Device {i}: CUDA Device #{device_name}")
            return True
        else:
            print("❌ OpenCV CUDA Support: Not available")
            return False
    except Exception as e:
        print(f"❌ OpenCV CUDA Support: Error checking ({str(e)})")
        return False

def check_torch():
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        print(f"✅ PyTorch: Version {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: Available (Device: {torch.cuda.get_device_name(0)})")
            print(f"   - CUDA Version: {torch.version.cuda}")
            print(f"   - Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - Device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"     - Total Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"     - CUDA Capability: {props.major}.{props.minor}")
            return True
        else:
            print("❌ PyTorch CUDA: Not available")
            return False
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch: Error checking ({str(e)})")
        return False

def check_rocm():
    """Check for ROCm support with PyTorch."""
    try:
        import torch
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
            if torch.version.hip:
                print(f"✅ PyTorch ROCm: Available (Version: {torch.version.hip})")
                return True
        print("❌ PyTorch ROCm: Not available")
        return False
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch ROCm: Error checking ({str(e)})")
        return False

def check_cupy():
    """Check CuPy and its CUDA/ROCm support."""
    try:
        import cupy
        print(f"✅ CuPy: Version {cupy.__version__}")
        
        try:
            # Check for CUDA in CuPy
            if hasattr(cupy, 'cuda') and cupy.cuda.runtime.getDeviceCount() > 0:
                print(f"✅ CuPy CUDA: Available (Devices: {cupy.cuda.runtime.getDeviceCount()})")
                for i in range(cupy.cuda.runtime.getDeviceCount()):
                    dev_props = cupy.cuda.runtime.getDeviceProperties(i)
                    print(f"   - Device {i}: {dev_props['name'].decode('utf-8')}")
                    print(f"     - Total Memory: {dev_props['totalGlobalMem'] / 1024**3:.1f} GB")
                return True
            elif hasattr(cupy, 'hip') and hasattr(cupy.hip, 'runtime'):
                # Check for ROCm in CuPy
                print(f"✅ CuPy ROCm: Available")
                return True
            else:
                print("❌ CuPy GPU: Not available (CPU-only)")
                return False
        except Exception as e:
            print(f"❌ CuPy GPU support: Error checking ({str(e)})")
            return False
    except ImportError:
        print("❌ CuPy: Not installed")
        return False
    except Exception as e:
        print(f"❌ CuPy: Error checking ({str(e)})")
        return False

def check_open3d_cuda():
    """Check Open3D CUDA support."""
    try:
        import open3d as o3d
        print(f"✅ Open3D: Version {o3d.__version__}")
        
        try:
            # Check for CUDA support
            o3d.core.initialize_cuda_device()
            print("✅ Open3D CUDA: Available")
            return True
        except Exception as e:
            print("❌ Open3D CUDA: Not available")
            return False
    except ImportError:
        print("❌ Open3D: Not installed")
        return False
    except Exception as e:
        print(f"❌ Open3D: Error checking ({str(e)})")
        return False

def check_open3d_ml():
    """Check Open3D-ML availability."""
    try:
        import open3d.ml as ml3d
        print(f"✅ Open3D-ML: Available")
        ml_backends = []
        if hasattr(ml3d, "tf"):
            ml_backends.append("TensorFlow")
        if hasattr(ml3d, "torch"):
            ml_backends.append("PyTorch")
        if ml_backends:
            print(f"   - Machine Learning backends: {', '.join(ml_backends)}")
        return True
    except ImportError:
        print("❌ Open3D-ML: Not installed")
        print("   Install with: pip install open3d-ml")
        return False
    except Exception as e:
        print(f"❌ Open3D-ML: Error checking ({str(e)})")
        return False

def print_system_info():
    """Print basic system information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")
    
    if platform.system() == 'Windows':
        import wmi
        w = wmi.WMI()
        gpu_info = [gpu.Name for gpu in w.Win32_VideoController()]
        print("GPUs detected by system:")
        for gpu in gpu_info:
            print(f"  - {gpu}")
    elif platform.system() == 'Linux':
        try:
            import subprocess
            result = subprocess.run(['lspci', '-v'], stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            # Filter for VGA compatible controller for GPU info
            gpu_lines = [line for line in output.split('\n') if 'VGA compatible controller' in line]
            if gpu_lines:
                print("GPUs detected by system:")
                for line in gpu_lines:
                    print(f"  - {line.split(':', 1)[1].strip()}")
        except:
            pass
    print("=" * 60)

def main():
    """Main function to check GPU acceleration features."""
    print_system_info()

    print("UNLOOK SDK GPU ACCELERATION CHECK")
    print("=" * 60)

    # Check CUDA and ROCm support
    has_cuda = check_cuda()
    has_torch = check_torch()
    has_rocm = check_rocm()
    has_cupy = check_cupy()
    has_open3d_cuda = check_open3d_cuda()
    has_open3d_ml = check_open3d_ml()

    print("\nSUMMARY")
    print("=" * 60)

    any_gpu = has_cuda or has_torch or has_rocm or has_cupy or has_open3d_cuda

    if any_gpu:
        print("✅ GPU acceleration is available with some features.")
        if has_cuda:
            print("  ✓ CUDA is available for OpenCV operations")
        if has_torch:
            print("  ✓ PyTorch with CUDA is available for neural network enhancement")
        if has_rocm:
            print("  ✓ AMD ROCm is available for GPU acceleration")
        if has_cupy:
            print("  ✓ CuPy is available for faster array processing")
        if has_open3d_cuda:
            print("  ✓ Open3D with CUDA is available for point cloud processing")
    else:
        print("❌ No GPU acceleration is available.")
        print("  → The SDK will run in CPU-only mode, which is slower.")

    if not has_open3d_ml:
        print("⚠️ Open3D-ML is not available.")
        print("  → Advanced point cloud processing and neural network features will be limited.")

    print("\nRECOMMENDATIONS")
    print("=" * 60)

    if not any_gpu:
        print("→ For better performance, consider installing GPU acceleration:")
        if platform.system() == 'Windows':
            print("  1. Install CUDA Toolkit from the NVIDIA website")
            print("  2. Install CuPy: pip install cupy-cuda11x (or appropriate version)")
            print("  3. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117")
        elif platform.system() == 'Linux':
            print("  For NVIDIA GPUs:")
            print("  1. Install CUDA Toolkit: sudo apt install nvidia-cuda-toolkit")
            print("  2. Install CuPy: pip install cupy-cuda11x (or appropriate version)")
            print("  3. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117")
            print("\n  For AMD GPUs:")
            print("  1. Install ROCm from AMD website")
            print("  2. Install PyTorch with ROCm: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2")

    if not has_open3d_ml:
        print("→ To use advanced point cloud processing features:")
        print("  - Install Open3D-ML: pip install open3d-ml")
        print("  - If that fails, use: pip install -U open3d")
        print("  - Followed by: pip install git+https://github.com/isl-org/Open3D-ML.git")

    print("\nFor more information, see the GPU Acceleration Guide in the documentation.")
    print("=" * 60)
    
if __name__ == "__main__":
    main()