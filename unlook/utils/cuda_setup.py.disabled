"""
CUDA setup utility for UnLook SDK.

This module helps detect and set up CUDA paths for libraries like CuPy
that may have difficulty automatically finding the CUDA installation,
especially in virtual environments.
"""

import os
import sys
import logging
import subprocess
import re
from pathlib import Path

logger = logging.getLogger(__name__)

def find_cuda_path():
    """
    Find CUDA installation path through various methods.
    
    Returns:
        str: CUDA installation path or None if not found
    """
    # Check if CUDA_PATH is already set
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        logger.info(f"CUDA_PATH already set to: {cuda_path}")
        return cuda_path
        
    # Method 1: Look in standard locations (Windows)
    windows_cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files\NVIDIA\CUDA"
    ]
    
    for base_path in windows_cuda_paths:
        if os.path.exists(base_path):
            # Find the latest version
            versions = []
            for item in os.listdir(base_path):
                version_dir = os.path.join(base_path, item)
                if os.path.isdir(version_dir) and item.startswith('v'):
                    versions.append((item, version_dir))
            
            if versions:
                versions.sort(reverse=True)  # Sort by version descending
                latest_version_path = versions[0][1]
                logger.info(f"Found CUDA in standard Windows location: {latest_version_path}")
                return latest_version_path
    
    # Method 2: Look in standard locations (Linux)
    linux_cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda"
    ]
    
    for path in linux_cuda_paths:
        if os.path.exists(path):
            logger.info(f"Found CUDA in standard Linux location: {path}")
            return path
    
    # Method 3: Search in PATH
    path_var = os.environ.get("PATH", "")
    path_elements = path_var.split(os.pathsep)
    
    # Look for nvcc or other CUDA binaries in PATH
    for path_element in path_elements:
        if "CUDA" in path_element or "cuda" in path_element:
            # Look for nvcc
            nvcc_path = os.path.join(path_element, "nvcc") if os.name != "nt" else os.path.join(path_element, "nvcc.exe")
            if os.path.exists(nvcc_path):
                # Go up one level from bin directory
                cuda_path = os.path.dirname(path_element)
                logger.info(f"Found CUDA from PATH: {cuda_path}")
                return cuda_path
    
    # Method 4: Try running nvcc --version
    try:
        output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        # Parse output to find CUDA path
        match = re.search(r"release\s+(\d+\.\d+)", output)
        if match:
            version = match.group(1)
            # Try common locations with this version
            for base_path in windows_cuda_paths + [r"C:\Program Files"]:
                candidate = os.path.join(base_path, f"v{version}")
                if os.path.exists(candidate):
                    logger.info(f"Found CUDA {version} path: {candidate}")
                    return candidate
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
        
    logger.warning("Could not detect CUDA path automatically")
    return None

def setup_cuda_env():
    """
    Set up CUDA environment variables for libraries like CuPy.
    
    Returns:
        bool: True if CUDA was found and environment was set up, False otherwise
    """
    cuda_path = find_cuda_path()
    if not cuda_path:
        return False
        
    # Set environment variable for this process
    os.environ["CUDA_PATH"] = cuda_path
    
    # Set CuPy-specific environment variables
    os.environ["CUPY_CACHE_DIR"] = os.path.join(os.path.expanduser("~"), ".cupy", "kernel_cache")
    
    # Add CUDA bin directory to PATH if not already there
    cuda_bin = os.path.join(cuda_path, "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        
    # Add CUDA lib directory to LD_LIBRARY_PATH/PATH
    if sys.platform == "win32":
        cuda_lib = os.path.join(cuda_path, "lib", "x64")
        if cuda_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_lib + os.pathsep + os.environ.get("PATH", "")
    else:  # Linux/Mac
        cuda_lib = os.path.join(cuda_path, "lib64")
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda_lib not in ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = cuda_lib + os.pathsep + ld_library_path
    
    logger.info(f"CUDA environment set up with CUDA_PATH={cuda_path}")
    return True

def is_cuda_available():
    """
    Check if CUDA is available by trying to use a CUDA-enabled library.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    # Try PyTorch first (most reliable method)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass

    # Try CuPy
    try:
        setup_cuda_env()  # Ensure environment is set up
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            logger.info(f"CUDA available via CuPy: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            return True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"CuPy error when checking CUDA: {e}")

    # Try OpenCV CUDA with extended validation
    try:
        import cv2
        if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                # Verify OpenCV CUDA actually works with a small operation
                try:
                    # Create a small test array and transfer to GPU
                    import numpy as np
                    test_array = np.zeros((10, 10), dtype=np.float32)
                    test_gpu_mat = cv2.cuda.GpuMat()
                    test_gpu_mat.upload(test_array)

                    # Try a simple operation
                    result_gpu_mat = cv2.cuda.GpuMat()
                    cv2.cuda.threshold(test_gpu_mat, result_gpu_mat, 0, 1, cv2.THRESH_BINARY)

                    # Download result
                    result = result_gpu_mat.download()

                    # Release resources
                    test_gpu_mat.release()
                    result_gpu_mat.release()

                    logger.info("CUDA available via OpenCV with successful validation")
                    return True
                except Exception as e:
                    logger.warning(f"OpenCV CUDA available but validation failed: {e}")
                    # Don't return True here since validation failed
            else:
                logger.warning("OpenCV reports no CUDA devices available")
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"OpenCV CUDA error: {e}")

    # Check system CUDA directly via subprocess
    try:
        import subprocess
        try:
            output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, universal_newlines=True)
            if "NVIDIA-SMI" in output and "Driver Version" in output:
                logger.info("CUDA available via nvidia-smi, but no Python CUDA bindings working")
                # We don't return True here because we need Python bindings to work
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    except ImportError:
        pass

    logger.warning("CUDA not available through any supported library")
    return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup CUDA environment
    if setup_cuda_env():
        print(f"CUDA_PATH set to: {os.environ.get('CUDA_PATH')}")
    else:
        print("Failed to set CUDA_PATH")
    
    # Check if CUDA is available
    is_available = is_cuda_available()
    print(f"CUDA available: {is_available}")