#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenCV CUDA Build Script for Unlook SDK

This script automatically builds OpenCV with CUDA support for the Unlook SDK.
It handles dependencies, downloads the source code, configures CMake with the
right options, and builds and installs OpenCV with CUDA integration.

Usage:
    python build_opencv_cuda.py [--cuda-arch ARCH] [--opencv-version VERSION] [--install-dir DIR]

Arguments:
    --cuda-arch       CUDA architecture (default: detect from installed GPU)
    --opencv-version  OpenCV version to build (default: 4.8.0)
    --install-dir     Installation directory (default: current Python environment)
    --jobs            Number of parallel build jobs (default: number of CPU cores)
    --no-contrib      Don't build OpenCV contrib modules (default: build with contrib)
    --debug           Enable debug output from build process
    --clean           Clean build directories before starting

Example:
    python build_opencv_cuda.py --cuda-arch 7.5 --opencv-version 4.8.0
"""

import os
import sys
import platform
import subprocess
import shutil
import argparse
import logging
import multiprocessing
import glob
import tempfile
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("opencv-cuda-build")

def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build OpenCV with CUDA support')
    
    parser.add_argument('--cuda-arch', type=str, default=None,
                        help='CUDA architecture (e.g., 7.5 for RTX 2080, 8.6 for RTX 3090)')
    parser.add_argument('--opencv-version', type=str, default='4.8.0',
                        help='OpenCV version to build (default: 4.8.0)')
    parser.add_argument('--install-dir', type=str, default=None,
                        help='Installation directory (default: current Python environment)')
    parser.add_argument('--jobs', type=int, default=multiprocessing.cpu_count(),
                        help=f'Number of parallel build jobs (default: {multiprocessing.cpu_count()})')
    parser.add_argument('--no-contrib', action='store_true',
                        help='Do not build OpenCV contrib modules')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output from build process')
    parser.add_argument('--clean', action='store_true',
                        help='Clean build directories before starting')
    
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

def get_python_info():
    """Get Python version and paths."""
    python_executable = sys.executable
    python_include_dir = os.path.join(os.path.dirname(os.path.dirname(python_executable)), 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')
    
    # Handle virtual environments
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Running in a virtual environment
        venv_path = sys.prefix
        python_library_path = None
        
        # Try to find the Python library
        if platform.system() == 'Windows':
            python_library = os.path.join(venv_path, 'libs', f'python{sys.version_info.major}{sys.version_info.minor}.lib')
        else:
            # Linux/macOS
            python_library_pattern = os.path.join(venv_path, 'lib', f'libpython{sys.version_info.major}.{sys.version_info.minor}*')
            python_libraries = glob.glob(python_library_pattern)
            python_library = python_libraries[0] if python_libraries else None
            
            # If not found in venv, try to find it in system
            if not python_library:
                system_lib_paths = [
                    '/usr/lib', 
                    '/usr/lib64',
                    '/usr/local/lib',
                    '/usr/local/lib64'
                ]
                for lib_path in system_lib_paths:
                    python_library_pattern = os.path.join(lib_path, f'libpython{sys.version_info.major}.{sys.version_info.minor}*')
                    python_libraries = glob.glob(python_library_pattern)
                    if python_libraries:
                        python_library = python_libraries[0]
                        break
    else:
        # System Python
        python_library = None  # Let CMake find it automatically
    
    # Ensure include directory exists
    if not os.path.exists(python_include_dir):
        # Try to find it
        if platform.system() == 'Windows':
            python_include_dir = os.path.join(os.path.dirname(python_executable), 'include')
        else:
            # Try to find in common locations
            for prefix in ['/usr', '/usr/local']:
                candidate = os.path.join(prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}*')
                matches = glob.glob(candidate)
                if matches:
                    python_include_dir = matches[0]
                    break
    
    numpy_include_dir = None
    try:
        import numpy
        numpy_include_dir = numpy.get_include()
    except ImportError:
        logger.warning("NumPy not found. Installing...")
        run_command([python_executable, '-m', 'pip', 'install', 'numpy'])
        import numpy
        numpy_include_dir = numpy.get_include()
    
    return {
        'executable': python_executable,
        'version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'include_dir': python_include_dir,
        'library': python_library,
        'numpy_include_dir': numpy_include_dir
    }

def detect_cuda():
    """Detect CUDA installation and version."""
    cuda_info = {'available': False, 'version': None, 'path': None, 'archs': []}
    
    # Check environment variable
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if not cuda_path:
        # Try to find in common locations
        possible_paths = []
        
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
                            possible_paths.append(os.path.join(base_dir, item))
        else:
            # Linux/macOS
            possible_paths = [
                '/usr/local/cuda',
                '/opt/cuda'
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cuda_path = path
                break
    
    if not cuda_path or not os.path.exists(cuda_path):
        logger.warning("CUDA installation not found")
        return cuda_info
    
    cuda_info['available'] = True
    cuda_info['path'] = cuda_path
    
    # Detect CUDA version
    nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc' + ('.exe' if platform.system() == 'Windows' else ''))
    
    if os.path.exists(nvcc_path):
        try:
            nvcc_output, _ = run_command([nvcc_path, '--version'])
            version_match = re.search(r'release (\d+\.\d+)', nvcc_output)
            if version_match:
                cuda_info['version'] = version_match.group(1)
                logger.info(f"CUDA version detected: {cuda_info['version']}")
        except Exception as e:
            logger.warning(f"Error detecting CUDA version: {e}")
    
    # Detect GPU architecture
    try:
        # Try using nvidia-smi to get GPU info
        nvidia_smi_output, _ = run_command(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], check=False)
        if nvidia_smi_output:
            for line in nvidia_smi_output.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    compute_cap = parts[1].strip().replace('.', '')
                    arch = f"{compute_cap[0]}.{compute_cap[1]}"
                    cuda_info['archs'].append(arch)
                    logger.info(f"Detected GPU: {gpu_name} (arch {arch})")
    except Exception as e:
        logger.warning(f"Error detecting GPU architecture: {e}")
    
    # Default architectures if detection failed
    if not cuda_info['archs']:
        # Choose architectures based on CUDA version
        if cuda_info['version']:
            cuda_version = float(cuda_info['version'])
            if cuda_version >= 11.0:
                # Support for Ampere (RTX 3xxx) and Turing (RTX 2xxx)
                cuda_info['archs'] = ['8.6', '7.5']
            elif cuda_version >= 10.0:
                # Support for Turing (RTX 2xxx) and Pascal (GTX 10xx)
                cuda_info['archs'] = ['7.5', '6.1']
            else:
                # Support for Pascal (GTX 10xx) and Maxwell (GTX 9xx)
                cuda_info['archs'] = ['6.1', '5.2']
        else:
            # Default to recent architectures
            cuda_info['archs'] = ['8.6', '7.5', '6.1']
        
        logger.info(f"Using default GPU architectures: {cuda_info['archs']}")
    
    return cuda_info

def install_dependencies():
    """Install required build dependencies."""
    if platform.system() == 'Linux':
        try:
            if os.path.exists('/etc/debian_version'):
                # Debian/Ubuntu
                deps = [
                    'build-essential', 'cmake', 'git', 'pkg-config', 'libgtk-3-dev',
                    'libavcodec-dev', 'libavformat-dev', 'libswscale-dev', 'libv4l-dev',
                    'libxvidcore-dev', 'libx264-dev', 'libjpeg-dev', 'libpng-dev',
                    'libtiff-dev', 'libatlas-base-dev', 'gfortran', 'python3-dev'
                ]
                cmd = ['apt-get', 'update']
                run_command(['sudo'] + cmd)
                cmd = ['apt-get', 'install', '-y'] + deps
                run_command(['sudo'] + cmd)
                
            elif os.path.exists('/etc/fedora-release') or os.path.exists('/etc/redhat-release'):
                # Fedora/RHEL/CentOS
                deps = [
                    'gcc', 'gcc-c++', 'make', 'cmake', 'git', 'pkgconfig',
                    'gtk3-devel', 'ffmpeg-devel', 'libv4l-devel',
                    'libjpeg-devel', 'libpng-devel', 'libtiff-devel',
                    'atlas-devel', 'gfortran', 'python3-devel'
                ]
                cmd = ['dnf', 'install', '-y'] + deps
                run_command(['sudo'] + cmd)
                
            else:
                logger.warning("Unsupported Linux distribution. Please install dependencies manually.")
                
        except Exception as e:
            logger.warning(f"Error installing dependencies: {e}")
            logger.warning("Please install build dependencies manually.")
    elif platform.system() == 'Darwin':
        # macOS
        try:
            # Check if Homebrew is installed
            brew_path = shutil.which('brew')
            if not brew_path:
                logger.warning("Homebrew not found. Please install it from https://brew.sh/")
                return
            
            deps = [
                'cmake', 'pkg-config', 'jpeg', 'libpng', 'libtiff',
                'openblas', 'ffmpeg'
            ]
            run_command(['brew', 'install'] + deps)
            
        except Exception as e:
            logger.warning(f"Error installing dependencies with Homebrew: {e}")
            logger.warning("Please install build dependencies manually.")
    elif platform.system() == 'Windows':
        # Windows - Most dependencies are bundled with the OpenCV source
        logger.info("Windows detected. Build dependencies should be bundled with OpenCV source.")
        logger.info("Ensure you have Visual Studio with C++ components installed.")
        
        # Check for CMake
        cmake_path = shutil.which('cmake')
        if not cmake_path:
            logger.warning("CMake not found. Please install it from https://cmake.org/download/")
            
        # Install pip dependencies
        python_info = get_python_info()
        pip_deps = ['numpy']
        run_command([python_info['executable'], '-m', 'pip', 'install'] + pip_deps)
    else:
        logger.warning(f"Unsupported platform: {platform.system()}")
        logger.warning("Please install build dependencies manually.")

def download_opencv(version, build_dir, with_contrib=True):
    """Download OpenCV and OpenCV contrib source code."""
    opencv_url = f"https://github.com/opencv/opencv/archive/{version}.zip"
    opencv_contrib_url = f"https://github.com/opencv/opencv_contrib/archive/{version}.zip"
    
    os.makedirs(build_dir, exist_ok=True)
    
    # Download and extract OpenCV
    opencv_zip = os.path.join(build_dir, f"opencv-{version}.zip")
    logger.info(f"Downloading OpenCV {version}...")
    
    if platform.system() == 'Windows':
        # Use PowerShell on Windows for better download experience
        download_cmd = [
            'powershell',
            '-Command',
            f"Invoke-WebRequest -Uri {opencv_url} -OutFile {opencv_zip}"
        ]
    else:
        download_cmd = ['curl', '-L', opencv_url, '-o', opencv_zip]
    
    try:
        run_command(download_cmd)
        
        logger.info("Extracting OpenCV...")
        if platform.system() == 'Windows':
            # Use PowerShell on Windows
            extract_cmd = [
                'powershell',
                '-Command',
                f"Expand-Archive -Path {opencv_zip} -DestinationPath {build_dir} -Force"
            ]
        else:
            extract_cmd = ['unzip', '-q', opencv_zip, '-d', build_dir]
        
        run_command(extract_cmd)
        
        # Clean up zip file
        os.remove(opencv_zip)
        
        # Get the extracted directory name
        opencv_dir = os.path.join(build_dir, f"opencv-{version}")
        
        # Ensure directory exists and has the expected structure
        if not os.path.exists(opencv_dir) or not os.path.exists(os.path.join(opencv_dir, 'CMakeLists.txt')):
            # Try to find the actual directory
            for item in os.listdir(build_dir):
                item_path = os.path.join(build_dir, item)
                if os.path.isdir(item_path) and 'opencv' in item.lower() and os.path.exists(os.path.join(item_path, 'CMakeLists.txt')):
                    opencv_dir = item_path
                    break
        
        if with_contrib:
            # Download and extract OpenCV contrib
            opencv_contrib_zip = os.path.join(build_dir, f"opencv_contrib-{version}.zip")
            logger.info(f"Downloading OpenCV contrib {version}...")
            
            if platform.system() == 'Windows':
                download_cmd = [
                    'powershell',
                    '-Command',
                    f"Invoke-WebRequest -Uri {opencv_contrib_url} -OutFile {opencv_contrib_zip}"
                ]
            else:
                download_cmd = ['curl', '-L', opencv_contrib_url, '-o', opencv_contrib_zip]
            
            run_command(download_cmd)
            
            logger.info("Extracting OpenCV contrib...")
            if platform.system() == 'Windows':
                extract_cmd = [
                    'powershell',
                    '-Command',
                    f"Expand-Archive -Path {opencv_contrib_zip} -DestinationPath {build_dir} -Force"
                ]
            else:
                extract_cmd = ['unzip', '-q', opencv_contrib_zip, '-d', build_dir]
            
            run_command(extract_cmd)
            
            # Clean up zip file
            os.remove(opencv_contrib_zip)
            
            # Get the extracted directory name
            opencv_contrib_dir = os.path.join(build_dir, f"opencv_contrib-{version}")
            
            # Ensure directory exists and has the expected structure
            if not os.path.exists(opencv_contrib_dir) or not os.path.exists(os.path.join(opencv_contrib_dir, 'modules')):
                # Try to find the actual directory
                for item in os.listdir(build_dir):
                    item_path = os.path.join(build_dir, item)
                    if os.path.isdir(item_path) and 'opencv_contrib' in item.lower() and os.path.exists(os.path.join(item_path, 'modules')):
                        opencv_contrib_dir = item_path
                        break
            
            return opencv_dir, opencv_contrib_dir
        
        return opencv_dir, None
        
    except Exception as e:
        logger.error(f"Error downloading and extracting OpenCV: {e}")
        sys.exit(1)

def configure_opencv_build(args, opencv_dir, opencv_contrib_dir, build_dir, cuda_info, python_info, install_dir):
    """Configure OpenCV build with CMake."""
    # Create build directory
    cmake_build_dir = os.path.join(build_dir, "build")
    os.makedirs(cmake_build_dir, exist_ok=True)
    
    # Basic CMake options
    cmake_options = [
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DWITH_GSTREAMER=OFF",  # Disable GStreamer to avoid runtime dependency
        "-DENABLE_NEON=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_opencv_java=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=ON",
        "-DWITH_OPENMP=ON",
        "-DWITH_IPP=ON",
        "-DWITH_TBB=ON",
        "-DWITH_EIGEN=ON",
        "-DWITH_V4L=ON",
        "-DWITH_FFMPEG=ON",
    ]
    
    # Python options
    cmake_options.extend([
        f"-DPYTHON3_EXECUTABLE={python_info['executable']}",
        f"-DPYTHON3_INCLUDE_DIR={python_info['include_dir']}",
        f"-DPYTHON3_NUMPY_INCLUDE_DIRS={python_info['numpy_include_dir']}",
    ])
    
    if python_info['library']:
        cmake_options.append(f"-DPYTHON3_LIBRARY={python_info['library']}")
    
    # OpenCV contrib options
    if opencv_contrib_dir:
        cmake_options.append(f"-DOPENCV_EXTRA_MODULES_PATH={os.path.join(opencv_contrib_dir, 'modules')}")
    
    # CUDA options
    if cuda_info['available']:
        # Use specified CUDA architecture or detected ones
        cuda_arch = args.cuda_arch or ','.join(cuda_info['archs'])
        
        cmake_options.extend([
            "-DWITH_CUDA=ON",
            "-DOPENCV_DNN_CUDA=ON",
            "-DWITH_CUBLAS=ON",
            "-DWITH_CUDNN=ON",
            "-DBUILD_opencv_cudacodec=ON",
            f"-DCUDA_ARCH_BIN={cuda_arch}",
            f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_info['path']}",
        ])
        
        # Check CUDA version for deprecation warnings
        if cuda_info['version'] and float(cuda_info['version']) >= 11.0:
            cmake_options.append("-DOPENCV_CUDA_FORCE_BUILTIN_CUFFT=ON")
    else:
        cmake_options.append("-DWITH_CUDA=OFF")
    
    # Platform-specific options
    if platform.system() == 'Windows':
        # Windows-specific options
        cmake_options.extend([
            "-DCMAKE_GENERATOR=Visual Studio 16 2019",  # Use VS 2019 - adjust if needed
            "-DCMAKE_GENERATOR_PLATFORM=x64",
            "-DCMAKE_CONFIGURATION_TYPES=Release",
        ])
    else:
        # Linux/macOS options
        cmake_options.extend([
            "-DCMAKE_BUILD_TYPE=Release",
        ])
    
    # Debug mode
    if args.debug:
        cmake_options.append("-DCMAKE_VERBOSE_MAKEFILE=ON")
    
    # Run CMake configuration
    cmake_cmd = ["cmake"] + cmake_options + [opencv_dir]
    try:
        logger.info("Configuring OpenCV build with CMake...")
        run_command(cmake_cmd, cwd=cmake_build_dir, debug=args.debug)
        
        return cmake_build_dir
    except Exception as e:
        logger.error(f"Error configuring OpenCV with CMake: {e}")
        sys.exit(1)

def build_and_install_opencv(build_dir, jobs, debug=False):
    """Build and install OpenCV."""
    try:
        # Build command depends on platform
        if platform.system() == 'Windows':
            # Windows Visual Studio build
            build_cmd = [
                "cmake", 
                "--build", ".", 
                "--config", "Release", 
                "--target", "INSTALL",
                "--", 
                f"/maxcpucount:{jobs}"
            ]
        else:
            # Linux/macOS make
            build_cmd = ["make", "-j", str(jobs), "install"]
        
        logger.info(f"Building OpenCV with {jobs} parallel jobs...")
        run_command(build_cmd, cwd=build_dir, debug=debug)
        
        logger.info("OpenCV with CUDA support built and installed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error building and installing OpenCV: {e}")
        sys.exit(1)

def test_opencv_cuda(python_executable):
    """Test OpenCV CUDA support."""
    test_script = """
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV build information:\\n{cv2.getBuildInformation()}")

# Check CUDA availability
try:
    cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA-enabled devices detected: {cuda_device_count}")
    
    if cuda_device_count > 0:
        # Basic test with GpuMat
        cpu_array = np.zeros((100, 100), dtype=np.float32)
        gpu_array = cv2.cuda.GpuMat()
        gpu_array.upload(cpu_array)
        
        # Try a simple CUDA operation
        result_gpu = cv2.cuda.GpuMat()
        cv2.cuda.threshold(gpu_array, result_gpu, 0.5, 1.0, cv2.THRESH_BINARY)
        
        # Download result
        result_cpu = result_gpu.download()
        
        # Release GPU memory
        gpu_array.release()
        result_gpu.release()
        
        print("CUDA test completed successfully!")
    else:
        print("No CUDA-enabled devices found.")
except Exception as e:
    print(f"CUDA test failed: {e}")

# Test importing the Python module
try:
    import cv2.cuda
    print("OpenCV CUDA Python module imported successfully!")
except ImportError as e:
    print(f"Failed to import cv2.cuda module: {e}")
"""
    
    test_file = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    test_file.write(test_script.encode('utf-8'))
    test_file.close()
    
    try:
        logger.info("Testing OpenCV CUDA support...")
        output, _ = run_command([python_executable, test_file.name])
        logger.info("OpenCV CUDA test output:")
        for line in output.split('\n'):
            logger.info(f"  {line}")
        
        # Clean up test file
        os.unlink(test_file.name)
        
        # Check if test was successful
        if "CUDA test completed successfully" in output:
            logger.info("OpenCV CUDA test passed!")
            return True
        else:
            logger.warning("OpenCV CUDA test did not complete successfully.")
            return False
    except Exception as e:
        logger.error(f"Error testing OpenCV CUDA support: {e}")
        return False

def main():
    """Main function."""
    args = setup_argparse()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Get Python info
    python_info = get_python_info()
    logger.info(f"Python executable: {python_info['executable']}")
    logger.info(f"Python version: {python_info['version']}")
    logger.info(f"Python include dir: {python_info['include_dir']}")
    if python_info['library']:
        logger.info(f"Python library: {python_info['library']}")
    logger.info(f"NumPy include dir: {python_info['numpy_include_dir']}")
    
    # Detect CUDA installation
    cuda_info = detect_cuda()
    if cuda_info['available']:
        logger.info(f"CUDA detected at: {cuda_info['path']}")
        if cuda_info['version']:
            logger.info(f"CUDA version: {cuda_info['version']}")
        logger.info(f"Using CUDA architectures: {cuda_info['archs']}")
    else:
        logger.warning("CUDA not detected. OpenCV will be built without CUDA support.")
    
    # Install build dependencies
    logger.info("Installing build dependencies...")
    install_dependencies()
    
    # Prepare build directory
    build_dir = os.path.join(tempfile.gettempdir(), f"opencv_build_{args.opencv_version}")
    logger.info(f"Using build directory: {build_dir}")
    
    if args.clean and os.path.exists(build_dir):
        logger.info(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)
    
    os.makedirs(build_dir, exist_ok=True)
    
    # Determine installation directory
    if args.install_dir:
        install_dir = args.install_dir
    else:
        # Install to Python site-packages
        python_path = Path(python_info['executable'])
        if platform.system() == 'Windows':
            # Windows path - use lib/site-packages
            install_dir = str(python_path.parent.parent / 'Lib' / 'site-packages' / 'opencv-cuda')
        else:
            # Linux/macOS path
            lib_dir = 'lib'
            for lib_name in [f'python{sys.version_info.major}.{sys.version_info.minor}', 'python3']:
                lib_path = python_path.parent.parent / lib_dir / lib_name / 'site-packages'
                if lib_path.exists():
                    install_dir = str(lib_path / 'opencv-cuda')
                    break
            else:
                # Fallback to users home directory
                install_dir = os.path.join(os.path.expanduser('~'), '.local', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'opencv-cuda')
    
    logger.info(f"Installing OpenCV to: {install_dir}")
    
    # Download OpenCV
    with_contrib = not args.no_contrib
    logger.info(f"Downloading OpenCV {args.opencv_version}...")
    opencv_dir, opencv_contrib_dir = download_opencv(args.opencv_version, build_dir, with_contrib)
    
    # Configure build
    cmake_build_dir = configure_opencv_build(args, opencv_dir, opencv_contrib_dir, build_dir, cuda_info, python_info, install_dir)
    
    # Build and install
    success = build_and_install_opencv(cmake_build_dir, args.jobs, args.debug)
    
    if success:
        # Update Python path to include the new installation
        sys.path.insert(0, install_dir)
        
        # Test installation
        test_result = test_opencv_cuda(python_info['executable'])
        
        if test_result:
            # Write installation instructions
            logger.info("\nOpenCV with CUDA support has been successfully built and installed!")
            logger.info("\nTo use this OpenCV build in your applications:")
            logger.info(f"1. Ensure {install_dir} is in your Python path")
            logger.info("2. Import OpenCV in your Python code: import cv2")
            logger.info("3. Verify CUDA support with: cv2.cuda.getCudaEnabledDeviceCount()")
            
            # Write a guide for installation
            guide_file = os.path.join(os.path.dirname(build_dir), "opencv_cuda_guide.txt")
            with open(guide_file, 'w') as f:
                f.write("OpenCV with CUDA Installation Guide\n")
                f.write("=================================\n\n")
                f.write(f"Installation directory: {install_dir}\n\n")
                f.write("To use this OpenCV build in your applications:\n")
                f.write(f"1. Add the following to your PYTHONPATH environment variable: {install_dir}\n")
                f.write("2. Import OpenCV in your Python code: import cv2\n")
                f.write("3. Check CUDA support with: cv2.cuda.getCudaEnabledDeviceCount()\n\n")
                f.write("Example code to test CUDA support:\n\n")
                f.write("```python\n")
                f.write("import cv2\n")
                f.write("import numpy as np\n\n")
                f.write("print(f\"OpenCV version: {cv2.__version__}\")\n")
                f.write("num_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()\n")
                f.write("print(f\"CUDA devices available: {num_cuda_devices}\")\n\n")
                f.write("if num_cuda_devices > 0:\n")
                f.write("    # Test GPU memory allocation\n")
                f.write("    gpu_mat = cv2.cuda.GpuMat()\n")
                f.write("    # Upload an array to GPU\n")
                f.write("    test_array = np.zeros((100, 100), dtype=np.float32)\n")
                f.write("    gpu_mat.upload(test_array)\n")
                f.write("    # Perform a simple operation\n")
                f.write("    result = cv2.cuda.GpuMat()\n")
                f.write("    cv2.cuda.threshold(gpu_mat, result, 0.5, 1.0, cv2.THRESH_BINARY)\n")
                f.write("    # Download the result\n")
                f.write("    cpu_result = result.download()\n")
                f.write("    print(\"CUDA test successful!\")\n")
                f.write("```\n")
            
            logger.info(f"\nA detailed guide has been saved to: {guide_file}")
        else:
            logger.warning("\nOpenCV was built but CUDA support test failed. Please check your CUDA installation.")
    else:
        logger.error("\nFailed to build OpenCV with CUDA support.")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nBuild cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)