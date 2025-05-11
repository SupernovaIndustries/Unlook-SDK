#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unlook SDK Client Installer

This script installs the Unlook SDK client components with all necessary dependencies.
It creates a virtual environment, installs dependencies, and configures the environment
for GPU acceleration if available.

Author: Unlook SDK Team
License: MIT
"""

import os
import sys
import platform
import subprocess
import argparse
import shutil
import site
import venv
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union, Callable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("unlook-installer")

# Color output for terminals
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'

def get_python_executable():
    """Get the Python executable path."""
    return sys.executable

def create_virtual_environment(venv_dir: str) -> str:
    """
    Create a Python virtual environment.
    
    Args:
        venv_dir: Directory to create the virtual environment in
        
    Returns:
        Path to the Python executable in the virtual environment
    """
    logger.info(f"Creating virtual environment in {venv_dir}")
    
    # Create the virtual environment
    venv.create(venv_dir, with_pip=True)
    
    # Get the Python executable path
    if platform.system() == "Windows":
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_dir, "bin", "python")
    
    logger.info(f"Virtual environment created successfully")
    return python_executable

def run_command(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None) -> Tuple[int, str, str]:
    """
    Run a command and return its output.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        env: Environment variables
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    logger.debug(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        universal_newlines=True
    )
    
    stdout, stderr = process.communicate()
    return_code = process.returncode
    
    logger.debug(f"Command returned {return_code}")
    
    return return_code, stdout, stderr

def install_dependencies(python_executable: str, requirements_file: str) -> bool:
    """
    Install Python dependencies from a requirements file.
    
    Args:
        python_executable: Path to the Python executable
        requirements_file: Path to the requirements file
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Installing dependencies from {requirements_file}")
    
    # Run pip install
    cmd = [python_executable, "-m", "pip", "install", "-r", requirements_file]
    return_code, stdout, stderr = run_command(cmd)
    
    if return_code != 0:
        logger.error(f"Failed to install dependencies: {stderr}")
        return False
    
    logger.info(f"Dependencies installed successfully")
    return True

def detect_gpu() -> Tuple[bool, str]:
    """
    Detect if a CUDA-compatible GPU is available.
    
    Returns:
        Tuple of (has_gpu, gpu_info)
    """
    logger.info("Detecting GPU...")
    
    # Try to import torch to detect CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            
            logger.info(f"CUDA-compatible GPU detected: {gpu_name} (CUDA {cuda_version})")
            return True, f"{gpu_name} (CUDA {cuda_version})"
        else:
            logger.info("No CUDA-compatible GPU detected")
            return False, ""
    except ImportError:
        logger.info("PyTorch not available, trying alternative detection method")
        
        # Try platform-specific GPU detection
        if platform.system() == "Windows":
            # Use Windows Management Instrumentation
            try:
                import wmi
                computer = wmi.WMI()
                gpu_info = computer.Win32_VideoController()[0].Name
                
                # Check if this is likely an NVIDIA GPU
                if "NVIDIA" in gpu_info or "GeForce" in gpu_info or "Quadro" in gpu_info:
                    logger.info(f"Potential CUDA-compatible GPU detected: {gpu_info}")
                    return True, gpu_info
                else:
                    logger.info(f"GPU detected but might not be CUDA-compatible: {gpu_info}")
                    return False, gpu_info
            except Exception as e:
                logger.warning(f"Failed to detect GPU using WMI: {e}")
        elif platform.system() == "Linux":
            # Use lspci
            try:
                cmd = ["lspci", "-v"]
                return_code, stdout, stderr = run_command(cmd)
                
                if return_code == 0 and ("NVIDIA" in stdout or "GeForce" in stdout or "Quadro" in stdout):
                    gpu_line = next((line for line in stdout.split('\n') if "NVIDIA" in line or "GeForce" in line or "Quadro" in line), "")
                    logger.info(f"Potential CUDA-compatible GPU detected: {gpu_line}")
                    return True, gpu_line
                else:
                    logger.info("No CUDA-compatible GPU detected")
                    return False, ""
            except Exception as e:
                logger.warning(f"Failed to detect GPU using lspci: {e}")
        
        # Default to no GPU
        logger.info("No GPU detection method available")
        return False, ""

def install_gpu_dependencies(python_executable: str, has_gpu: bool) -> bool:
    """
    Install GPU-specific dependencies if a GPU is available.
    
    Args:
        python_executable: Path to the Python executable
        has_gpu: Whether a GPU is available
        
    Returns:
        True if successful, False otherwise
    """
    if not has_gpu:
        logger.info("Skipping GPU dependencies installation (no GPU detected)")
        return True
    
    logger.info("Installing GPU dependencies")
    
    # Try to determine CUDA version
    cuda_version = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
    except ImportError:
        pass
    
    # Install appropriate CuPy version based on CUDA version
    if cuda_version:
        if cuda_version.startswith("11"):
            cupy_package = "cupy-cuda11x"
        elif cuda_version.startswith("12"):
            cupy_package = "cupy-cuda12x"
        else:
            cupy_package = "cupy-cuda11x"  # Default to 11.x
    else:
        cupy_package = "cupy-cuda11x"  # Default to 11.x
    
    logger.info(f"Installing {cupy_package} for GPU acceleration")
    cmd = [python_executable, "-m", "pip", "install", cupy_package]
    return_code, stdout, stderr = run_command(cmd)
    
    if return_code != 0:
        logger.warning(f"Failed to install {cupy_package}: {stderr}")
        logger.warning("GPU acceleration will not be available")
        return False
    
    logger.info(f"GPU dependencies installed successfully")
    return True

def setup_open3d_ml(python_executable: str, repo_dir: str, ml_backend: str = "") -> bool:
    """
    Set up Open3D-ML for neural network point cloud processing.
    
    Args:
        python_executable: Path to the Python executable
        repo_dir: Directory to clone the repository to
        ml_backend: ML backend to install (pytorch or tensorflow)
        
    Returns:
        True if successful, False otherwise
    """
    # Skip if the directory already exists
    if os.path.exists(repo_dir):
        logger.info(f"Open3D-ML repository already exists at {repo_dir}")
        return True
    
    logger.info(f"Setting up Open3D-ML in {repo_dir}")
    
    # Clone the repository
    cmd = ["git", "clone", "https://github.com/isl-org/Open3D-ML.git", repo_dir]
    return_code, stdout, stderr = run_command(cmd)
    
    if return_code != 0:
        logger.error(f"Failed to clone Open3D-ML repository: {stderr}")
        return False
    
    # Install dependencies based on the backend
    if ml_backend.lower() == "tensorflow":
        requirements_file = os.path.join(repo_dir, "requirements-tensorflow.txt")
    elif ml_backend.lower() == "pytorch":
        requirements_file = os.path.join(repo_dir, "requirements-torch.txt")
    else:
        # Install base requirements
        requirements_file = os.path.join(repo_dir, "requirements.txt")
    
    if os.path.exists(requirements_file):
        logger.info(f"Installing Open3D-ML dependencies from {requirements_file}")
        cmd = [python_executable, "-m", "pip", "install", "-r", requirements_file]
        return_code, stdout, stderr = run_command(cmd)
        
        if return_code != 0:
            logger.warning(f"Failed to install Open3D-ML dependencies: {stderr}")
    
    # Install the package in development mode
    logger.info(f"Installing Open3D-ML in development mode")
    cmd = [python_executable, "-m", "pip", "install", "-e", repo_dir]
    return_code, stdout, stderr = run_command(cmd)
    
    if return_code != 0:
        logger.warning(f"Failed to install Open3D-ML package: {stderr}")
        return False
    
    logger.info(f"Open3D-ML setup completed successfully")
    return True

def create_desktop_shortcut(venv_dir: str, script_path: str, shortcut_name: str) -> bool:
    """
    Create a desktop shortcut to run a script in the virtual environment.
    
    Args:
        venv_dir: Path to the virtual environment
        script_path: Path to the script to run
        shortcut_name: Name of the shortcut
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating desktop shortcut for {shortcut_name}")
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    if platform.system() == "Windows":
        # Create a batch file on Windows
        shortcut_path = os.path.join(desktop_path, f"{shortcut_name}.bat")
        
        with open(shortcut_path, "w") as f:
            f.write(f"@echo off\n")
            f.write(f'echo Starting {shortcut_name}...\n')
            f.write(f'"{os.path.join(venv_dir, "Scripts", "python.exe")}" "{script_path}" %*\n')
            f.write(f"pause\n")
    
    elif platform.system() == "Linux":
        # Create a desktop entry on Linux
        shortcut_path = os.path.join(desktop_path, f"{shortcut_name}.desktop")
        
        with open(shortcut_path, "w") as f:
            f.write(f"[Desktop Entry]\n")
            f.write(f"Version=1.0\n")
            f.write(f"Type=Application\n")
            f.write(f"Name={shortcut_name}\n")
            f.write(f"Comment=Unlook SDK Client\n")
            f.write(f"Exec={os.path.join(venv_dir, 'bin', 'python')} {script_path}\n")
            f.write(f"Terminal=true\n")
            f.write(f"Categories=Development;\n")
        
        # Make the shortcut executable
        os.chmod(shortcut_path, 0o755)
    
    elif platform.system() == "Darwin":  # macOS
        # Create an AppleScript on macOS
        shortcut_path = os.path.join(desktop_path, f"{shortcut_name}.command")
        
        with open(shortcut_path, "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f'echo "Starting {shortcut_name}..."\n')
            f.write(f'"{os.path.join(venv_dir, "bin", "python")}" "{script_path}" "$@"\n')
        
        # Make the shortcut executable
        os.chmod(shortcut_path, 0o755)
    
    else:
        logger.warning(f"Unsupported platform for creating desktop shortcuts: {platform.system()}")
        return False
    
    logger.info(f"Desktop shortcut created at {shortcut_path}")
    return True

def setup_unlock_console(venv_dir: str, sdk_dir: str) -> bool:
    """
    Set up the Unlook SDK console.
    
    Args:
        venv_dir: Path to the virtual environment
        sdk_dir: Path to the SDK directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Setting up Unlook SDK console")
    
    # Create console script
    console_dir = os.path.join(sdk_dir, "unlook_console")
    os.makedirs(console_dir, exist_ok=True)
    
    # Create console launcher script
    console_script_path = os.path.join(console_dir, "unlook_console.py")
    
    with open(console_script_path, "w") as f:
        f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
Unlook SDK Console

This is the main entry point for the Unlook SDK console.
It provides a convenient interface for working with the Unlook SDK.
\"\"\"

import os
import sys
import logging
import argparse
import platform
import time
from pathlib import Path

# Add the SDK to the Python path
sdk_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, sdk_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("unlook-console")

# Import Unlook SDK
try:
    from unlook import UnlookClient
    from unlook.client.static_scanner import StaticScanConfig, create_static_scanner
except ImportError as e:
    logger.error(f"Failed to import Unlook SDK: {{e}}")
    logger.error("Please make sure the Unlook SDK is installed correctly.")
    sys.exit(1)

def parse_arguments():
    \"\"\"Parse command line arguments.\"\"\"
    parser = argparse.ArgumentParser(description="Unlook SDK Console")
    parser.add_argument("--scan", action="store_true", help="Start a 3D scan")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--quality", type=str, default="high", choices=["medium", "high", "ultra"], help="Scan quality")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--neural", action="store_true", help="Enable neural network processing")
    
    return parser.parse_args()

def welcome_message():
    \"\"\"Display welcome message.\"\"\"
    print("\\n" + "="*80)
    print(" UNLOOK SDK CONSOLE")
    print(" 3D Scanning and Processing Toolkit")
    print("="*80 + "\\n")
    
    print(f"System information:")
    print(f"  - Platform: {{platform.system()}} {{platform.release()}}")
    print(f"  - Python: {{platform.python_version()}}")
    
    # Check for GPU
    gpu_info = "Not detected"
    try:
        if CUPY_AVAILABLE:
            import cupy as cp
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    print(f"  - GPU: {{gpu_info}}\\n")
    
    print("Available commands:")
    print("  --scan     Start a 3D scan")
    print("  --output   Specify output file")
    print("  --quality  Set scan quality (medium, high, ultra)")
    print("  --gpu      Enable GPU acceleration")
    print("  --neural   Enable neural network processing\\n")

def main():
    \"\"\"Run the Unlook SDK console.\"\"\"
    args = parse_arguments()
    
    welcome_message()
    
    if args.scan:
        print("Starting 3D scan...")
        
        # Create client
        client = UnlookClient(auto_discover=True)
        client.start_discovery()
        print("Discovering scanners (5 seconds)...")
        time.sleep(5)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        if not scanners:
            print("No scanners found. Please ensure scanner hardware is connected and powered on.")
            return 1
        
        # Connect to the first scanner
        scanner_info = scanners[0]
        print(f"Connecting to scanner: {{scanner_info.name}} ({{scanner_info.uuid}})")
        
        if not client.connect(scanner_info):
            print("Failed to connect to scanner")
            return 1
        
        print(f"Successfully connected to scanner: {{scanner_info.name}}")
        
        # Create configuration
        config = StaticScanConfig()
        config.set_quality_preset(args.quality)
        config.use_gpu = args.gpu
        config.use_neural_network = args.neural
        
        # Create scanner
        scanner = create_static_scanner(
            client=client,
            config=config
        )
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"scan_{{timestamp}}_{{args.quality}}.ply"
        
        # Perform scan
        print("\\nStarting scan...")
        point_cloud = scanner.perform_scan()
        
        if point_cloud is None:
            print("Scan failed to produce a point cloud")
            return 1
        
        # Save point cloud
        if scanner.save_point_cloud(output_file):
            print(f"Point cloud saved to: {{os.path.abspath(output_file)}}")
        else:
            print("Failed to save point cloud")
        
        # Disconnect
        client.disconnect()
        print("Disconnected from scanner")
    
    else:
        print("No action specified. Use --scan to start a 3D scan.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
    
    # Create desktop shortcut
    create_desktop_shortcut(venv_dir, console_script_path, "Unlook SDK Console")
    
    logger.info("Unlook SDK console setup completed successfully")
    return True

def main():
    """Run the installer."""
    print(f"{BOLD}{YELLOW}Unlook SDK Client Installer{END}\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Install Unlook SDK Client")
    parser.add_argument("--venv-dir", type=str, default=".venv", help="Virtual environment directory")
    parser.add_argument("--sdk-dir", type=str, default=".", help="SDK directory")
    parser.add_argument("--ml-backend", type=str, choices=["pytorch", "tensorflow", ""], default="", help="ML backend to install")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get SDK directory
    sdk_dir = os.path.abspath(args.sdk_dir)
    logger.info(f"SDK directory: {sdk_dir}")
    
    # Create virtual environment
    venv_dir = os.path.abspath(args.venv_dir)
    python_executable = create_virtual_environment(venv_dir)
    
    # Install base dependencies
    requirements_file = os.path.join(sdk_dir, "client-requirements.txt")
    if not install_dependencies(python_executable, requirements_file):
        logger.error("Failed to install dependencies")
        return 1
    
    # Detect GPU
    has_gpu, gpu_info = detect_gpu()
    if has_gpu and not args.no_gpu:
        install_gpu_dependencies(python_executable, has_gpu)
    
    # Set up Open3D-ML
    if args.ml_backend:
        open3d_ml_dir = os.path.join(sdk_dir, "unlook", "examples", "open3d")
        setup_open3d_ml(python_executable, open3d_ml_dir, args.ml_backend)
    
    # Set up Unlook console
    setup_unlock_console(venv_dir, sdk_dir)
    
    # Print summary
    print(f"\n{BOLD}{GREEN}Installation completed successfully!{END}\n")
    print(f"Unlook SDK client has been installed in a virtual environment at:\n{BLUE}{venv_dir}{END}\n")
    
    print(f"Details:")
    print(f"  - Python executable: {python_executable}")
    print(f"  - GPU support: {'Enabled' if has_gpu and not args.no_gpu else 'Disabled'}")
    if has_gpu and not args.no_gpu:
        print(f"  - GPU detected: {gpu_info}")
    print(f"  - ML backend: {args.ml_backend if args.ml_backend else 'None'}")
    
    print(f"\nTo use the Unlook SDK client:")
    print(f"  1. Use the desktop shortcut 'Unlook SDK Console'")
    print(f"  2. Or activate the virtual environment and run your scripts:")
    
    if platform.system() == "Windows":
        print(f"     {venv_dir}\\Scripts\\activate")
    else:
        print(f"     source {venv_dir}/bin/activate")
    
    print(f"\nThank you for installing Unlook SDK!\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())