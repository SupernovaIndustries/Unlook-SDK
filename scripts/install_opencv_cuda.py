#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenCV CUDA Installation Script for Unlook SDK

This script automates the installation of OpenCV with CUDA support for the Unlook SDK.
It provides a user-friendly interface for installing pre-built OpenCV packages
or building from source when needed.

Usage:
    python install_opencv_cuda.py [--build-from-source] [--cuda-arch ARCH] [--opencv-version VERSION]

Arguments:
    --build-from-source   Always build from source instead of trying pre-built packages
    --cuda-arch           CUDA architecture (default: auto-detect from installed GPU)
    --opencv-version      OpenCV version to install (default: 4.8.0)
    --no-interactive      Run in non-interactive mode (no user prompts)
    --no-progress         Don't show progress bars during downloads
    --debug               Enable debug logging
    --help                Show this help message and exit

Example:
    python install_opencv_cuda.py --build-from-source --cuda-arch 7.5
"""

import os
import sys
import platform
import subprocess
import logging
import argparse
import time
import importlib.util
from pathlib import Path
import re
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("opencv-cuda-install")

def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Install OpenCV with CUDA support')
    
    parser.add_argument('--build-from-source', action='store_true',
                      help='Always build from source instead of trying pre-built packages')
    parser.add_argument('--cuda-arch', type=str, default=None,
                      help='CUDA architecture (e.g., 7.5 for RTX 2080, 8.6 for RTX 3090)')
    parser.add_argument('--opencv-version', type=str, default='4.8.0',
                      help='OpenCV version to install (default: 4.8.0)')
    parser.add_argument('--no-interactive', action='store_true',
                      help='Run in non-interactive mode (no user prompts)')
    parser.add_argument('--no-progress', action='store_true',
                      help='Don\'t show progress bars during downloads')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    return parser.parse_args()

def run_command(cmd, cwd=None, check=True, shell=False, debug=False):
    """Run a command and return its output."""
    logger.info(f"Running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    if debug:
        # Run with normal output for debugging
        result = subprocess.run(cmd, cwd=cwd, check=check, shell=shell)
        return "", 0
    else:
        # Capture output
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.stdout, result.returncode

def check_existing_opencv():
    """Check if OpenCV is already installed and has CUDA support."""
    try:
        # Check if OpenCV is installed
        import cv2

        # Get version safely - support multiple OpenCV version formats
        version = "unknown"
        try:
            if hasattr(cv2, '__version__'):
                version = cv2.__version__
            elif hasattr(cv2, 'CV_VERSION'):
                version = cv2.CV_VERSION
            else:
                # Try to get version from module attributes
                version_attrs = [attr for attr in dir(cv2) if 'version' in attr.lower()]
                if version_attrs:
                    for attr in version_attrs:
                        if isinstance(getattr(cv2, attr), str):
                            version = getattr(cv2, attr)
                            break
        except Exception:
            pass

        logger.info(f"OpenCV is already installed (version {version})")

        # Check for CUDA support
        try:
            if not hasattr(cv2, 'cuda'):
                logger.info("OpenCV is installed, but CUDA module is not available")
                return False, version

            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            logger.info(f"CUDA-enabled devices detected: {cuda_device_count}")

            if cuda_device_count > 0:
                # Verify CUDA actually works
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

                    logger.info("OpenCV with working CUDA support is already installed!")
                    return True, version
                except Exception as e:
                    logger.info(f"OpenCV CUDA operations failed: {e}")
                    return False, version
            else:
                logger.info("OpenCV is installed, but CUDA support is not available (no CUDA devices detected)")
                return False, version
        except (AttributeError, Exception) as e:
            logger.info(f"OpenCV is installed, but CUDA support is not available: {e}")
            return False, version

    except ImportError:
        logger.info("OpenCV is not installed")
        return False, None

def get_cuda_info():
    """Get information about the installed CUDA toolkit."""
    cuda_info = {
        'available': False,
        'version': None,
        'path': None,
        'arch': None
    }
    
    # Check for CUDA toolkit
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if not cuda_path:
        # Try to find in common locations
        if platform.system() == 'Windows':
            base_dirs = [
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
                r'C:\Program Files\NVIDIA\CUDA'
            ]
            for base_dir in base_dirs:
                if os.path.exists(base_dir):
                    # Find all version directories
                    for item in os.listdir(base_dir):
                        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('v'):
                            cuda_path = os.path.join(base_dir, item)
                            break
                    if cuda_path:
                        break
        else:
            # Linux/macOS
            for path in ['/usr/local/cuda', '/opt/cuda']:
                if os.path.exists(path):
                    cuda_path = path
                    break
    
    if cuda_path and os.path.exists(cuda_path):
        cuda_info['available'] = True
        cuda_info['path'] = cuda_path
        
        # Try to detect CUDA version
        nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc' + ('.exe' if platform.system() == 'Windows' else ''))
        if os.path.exists(nvcc_path):
            try:
                output, _ = run_command([nvcc_path, '--version'])
                match = re.search(r'release (\d+\.\d+)', output)
                if match:
                    cuda_info['version'] = match.group(1)
            except Exception:
                pass
        
        # Try to detect GPU architecture
        try:
            output, _ = run_command(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], check=False)
            if output:
                lines = output.strip().split('\n')
                for line in lines:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        compute_cap = parts[1].strip()
                        cuda_info['arch'] = compute_cap.replace('.', '')
                        break
        except Exception:
            pass
    
    return cuda_info

def check_pip_install_opencv_cuda():
    """Try to install pre-built OpenCV with CUDA packages."""
    # Check system info
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    logger.info(f"Python version: {py_version}")
    
    # Check for pip
    try:
        import pip
        logger.info(f"pip version: {pip.__version__}")
    except ImportError:
        logger.warning("pip is not installed. Please install pip first.")
        return False
    
    # Check if we need to use pip3 instead of pip
    pip_cmd = [sys.executable, '-m', 'pip']
    
    # Try to install the opencv-contrib-python-cuda package
    logger.info("Trying to install pre-built OpenCV with CUDA package...")
    
    try:
        # First uninstall any existing OpenCV packages
        run_command(pip_cmd + ['uninstall', '-y', 'opencv-python', 'opencv-contrib-python'])
        
        # Try to install the CUDA package
        run_command(pip_cmd + ['install', 'opencv-contrib-python-cuda'])
        
        # Verify installation
        import cv2
        return check_existing_opencv()[0]
    except Exception as e:
        logger.warning(f"Failed to install pre-built OpenCV CUDA package: {e}")
        return False

def install_opencv_from_source(args):
    """Build and install OpenCV with CUDA support from source."""
    # Check if build script exists
    build_script = Path(__file__).parent / "build_opencv_cuda.py"
    
    if not build_script.exists():
        logger.error(f"Build script not found: {build_script}")
        return False
    
    # Construct command to run the build script
    cmd = [sys.executable, str(build_script)]
    
    if args.cuda_arch:
        cmd.extend(['--cuda-arch', args.cuda_arch])
    
    cmd.extend(['--opencv-version', args.opencv_version])
    
    if args.debug:
        cmd.append('--debug')
    
    # Run the build script
    logger.info("Starting OpenCV build process...")
    try:
        run_command(cmd, debug=args.debug)
        return True
    except Exception as e:
        logger.error(f"Error building OpenCV: {e}")
        return False

def ask_yes_no(prompt, default="yes"):
    """Ask a yes/no question and return the answer."""
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default is None:
        prompt += " [y/n] "
    elif default == "yes":
        prompt += " [Y/n] "
    elif default == "no":
        prompt += " [y/N] "
    else:
        raise ValueError(f"Invalid default answer: '{default}'")
    
    while True:
        try:
            choice = input(prompt).lower()
            if choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                print("Please respond with 'yes' or 'no' (or 'y' or 'n').")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(1)

def display_header():
    """Display script header."""
    print("\n" + "=" * 80)
    print("  OpenCV with CUDA Installer for Unlook SDK")
    print("=" * 80)
    print("This script will install OpenCV with CUDA support for the Unlook SDK.")
    print("It will check for existing installations and system requirements before proceeding.\n")

def display_summary(cuda_info):
    """Display system information."""
    print("\nSystem Information:")
    print("-" * 80)
    print(f"Operating System: {platform.system()} {platform.version()}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    if cuda_info['available']:
        print(f"CUDA Available: Yes (version {cuda_info['version'] or 'unknown'})")
        print(f"CUDA Path: {cuda_info['path']}")
        if cuda_info['arch']:
            print(f"GPU Architecture: {cuda_info['arch']}")
    else:
        print("CUDA Available: No")
    
    print("-" * 80 + "\n")

def main():
    """Main function."""
    args = setup_argparse()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Display header in interactive mode
    if not args.no_interactive:
        display_header()
    
    # Check if OpenCV with CUDA is already installed
    opencv_installed, opencv_version = check_existing_opencv()
    
    if opencv_installed:
        if not args.no_interactive:
            print("\nOpenCV with working CUDA support is already installed!")
            print(f"Version: {opencv_version}")
            
            if ask_yes_no("Would you like to reinstall OpenCV with CUDA support?", default="no"):
                pass  # Continue with installation
            else:
                print("Installation cancelled. Existing OpenCV with CUDA will be used.")
                return 0
        else:
            # In non-interactive mode, don't reinstall if already installed
            logger.info("OpenCV with CUDA is already properly installed. Skipping reinstallation.")
            return 0
    
    # Get CUDA information
    cuda_info = get_cuda_info()
    
    if not args.no_interactive:
        display_summary(cuda_info)
    
    if not cuda_info['available']:
        logger.warning("CUDA is not available on this system. OpenCV will be built without CUDA support.")
        if not args.no_interactive:
            if not ask_yes_no("Continue without CUDA support?", default="no"):
                print("Installation cancelled.")
                return 1
    
    # Try to install pre-built package first (unless build from source is requested)
    if not args.build_from_source:
        logger.info("Attempting to install pre-built OpenCV with CUDA package...")
        if check_pip_install_opencv_cuda():
            logger.info("Successfully installed OpenCV with CUDA support via pip!")
            return 0
        else:
            logger.info("Pre-built package installation failed. Falling back to building from source.")
    
    # Build from source
    logger.info("Building OpenCV with CUDA support from source...")
    if install_opencv_from_source(args):
        logger.info("Successfully built and installed OpenCV with CUDA support!")
        return 0
    else:
        logger.error("Failed to build and install OpenCV with CUDA support.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)