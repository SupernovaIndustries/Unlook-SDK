!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unlook SDK - OpenCV CUDA Setup

This script checks if OpenCV with CUDA support is installed and offers to install
it automatically if not found. This ensures that the Unlook SDK has proper GPU
acceleration for real-time 3D scanning.

Usage:
    python setup_opencv_cuda.py

This will automatically:
1. Check if OpenCV with CUDA support is already installed
2. If not, offer to install it either from pre-built packages or by building from source
3. Verify the installation and test CUDA support
"""

import os
import sys
import subprocess
import importlib.util
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("unlook-opencv-setup")

def check_opencv_cuda():
    """Check if OpenCV with CUDA support is installed and working."""
    logger.info("Checking for OpenCV with CUDA support...")

    try:
        # Check if OpenCV is installed
        import cv2

        # Get version safely - handle different OpenCV versions
        version = "unknown"
        try:
            if hasattr(cv2, '__version__'):
                version = cv2.__version__
            elif hasattr(cv2, 'CV_VERSION'):
                version = cv2.CV_VERSION
            else:
                # Try to find version in module attributes
                for attr in dir(cv2):
                    if 'version' in attr.lower() and isinstance(getattr(cv2, attr), str):
                        version = getattr(cv2, attr)
                        break
        except Exception:
            pass

        logger.info(f"OpenCV version: {version}")

        # Check if CUDA is available
        try:
            # First check if the cuda module exists
            if not hasattr(cv2, 'cuda'):
                logger.warning("OpenCV was not built with CUDA support (cuda module not found)")
                return False

            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            logger.info(f"CUDA-enabled devices detected: {cuda_device_count}")

            if cuda_device_count > 0:
                # Verify CUDA operations work
                try:
                    import numpy as np
                    test_array = np.zeros((10, 10), dtype=np.float32)
                    gpu_mat = cv2.cuda.GpuMat()
                    gpu_mat.upload(test_array)
                    result_gpu = cv2.cuda.GpuMat()
                    cv2.cuda.threshold(gpu_mat, result_gpu, 0.5, 1.0, cv2.THRESH_BINARY)
                    result = result_gpu.download()
                    gpu_mat.release()
                    result_gpu.release()
                    logger.info("OpenCV with CUDA is properly installed and working!")
                    return True
                except Exception as e:
                    logger.warning(f"OpenCV CUDA operations failed: {e}")
                    return False
            else:
                logger.warning("No CUDA devices detected with OpenCV")
                return False
        except AttributeError as e:
            logger.warning(f"OpenCV was not built with CUDA support: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error checking OpenCV CUDA support: {e}")
            return False
    except ImportError:
        logger.warning("OpenCV is not installed")
        return False

def run_installer_script():
    """Run the OpenCV CUDA installer script."""
    # Find installer script in the scripts directory
    script_path = Path(__file__).parent / "scripts" / "install_opencv_cuda.py"
    
    if not script_path.exists():
        logger.error(f"Installer script not found: {script_path}")
        return False
    
    # Run the installer script
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError:
        logger.error("OpenCV CUDA installation failed")
        return False
    except Exception as e:
        logger.error(f"Error running installer script: {e}")
        return False

def check_cuda_toolkit():
    """Check if CUDA Toolkit is installed."""
    if sys.platform == 'win32':
        # Check Windows registry or common installation paths
        common_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files\NVIDIA\CUDA"
        ]
        for base_path in common_paths:
            if os.path.exists(base_path):
                logger.info(f"CUDA Toolkit found at: {base_path}")
                return True
    else:
        # Check for CUDA in common Linux/macOS paths
        common_paths = [
            "/usr/local/cuda",
            "/opt/cuda"
        ]
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"CUDA Toolkit found at: {path}")
                return True
    
    # Check environment variables
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_path and os.path.exists(cuda_path):
        logger.info(f"CUDA Toolkit found at: {cuda_path}")
        return True
    
    # Try running nvcc
    try:
        output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        logger.info(f"CUDA Toolkit found via nvcc: {output.strip()}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    logger.warning("CUDA Toolkit not found. OpenCV can still be built with CPU-only support.")
    return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up OpenCV with CUDA support for Unlook SDK")
    parser.add_argument('--force', action='store_true', help="Force reinstallation even if already installed")
    args = parser.parse_args()
    
    # Check for OpenCV with CUDA
    if not args.force and check_opencv_cuda():
        print("\nOpenCV with CUDA support is already installed and working properly!")
        print("No further action is needed for this component.")
        return 0
    
    # Check for CUDA Toolkit
    cuda_available = check_cuda_toolkit()
    
    if not cuda_available:
        print("\nWARNING: CUDA Toolkit not detected on this system.")
        print("You can still use the Unlook SDK, but without GPU acceleration.")
        print("To enable GPU acceleration, please install the NVIDIA CUDA Toolkit first.")
        
        try:
            choice = input("\nDo you want to continue with CPU-only OpenCV installation? [y/N]: ").lower()
            if choice not in ('y', 'yes'):
                print("Setup cancelled. Please install CUDA Toolkit and run this script again.")
                return 1
        except KeyboardInterrupt:
            print("\nSetup cancelled.")
            return 1
    
    print("\nSetting up OpenCV with CUDA support for Unlook SDK...")
    
    # Run the installer script
    success = run_installer_script()
    
    if success:
        print("\nOpenCV with CUDA setup completed successfully!")
        if check_opencv_cuda():
            print("Verification passed: OpenCV with CUDA is working properly!")
        else:
            print("Warning: OpenCV with CUDA was installed but failed verification.")
            print("You may need to restart your system or update your environment variables.")
    else:
        print("\nOpenCV with CUDA setup failed.")
        print("Please check the logs for more information.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)