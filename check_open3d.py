#!/usr/bin/env python3
"""
Open3D Verification Script for UnLook SDK

This script checks your Open3D installation and diagnoses common issues.
It will attempt to verify CUDA support and ML backends.

Usage:
    python check_open3d.py [--fix] [--verbose]
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("open3d_check")

def check_gpu_info():
    """Check GPU information."""
    try:
        # Try NVIDIA-SMI first
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout
        except:
            pass
            
        # Windows-specific WMI method
        if sys.platform == "win32":
            try:
                import wmi
                w = wmi.WMI()
                gpu_info = [gpu.Name for gpu in w.Win32_VideoController()]
                return "\n".join(gpu_info)
            except:
                pass
                
        return "Could not determine GPU information"
    except Exception as e:
        return f"Error getting GPU info: {e}"

def check_open3d_installation():
    """Check Open3D installation status."""
    print("\n=== Checking Open3D Installation ===\n")
    
    try:
        import open3d as o3d
        print(f"✓ Open3D is installed (version {o3d.__version__})")
        
        # Check for CUDA support
        try:
            print("\n=== Checking Open3D CUDA Support ===\n")

            # Setup CUDA environment first
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from unlook.utils.cuda_setup import setup_cuda_env
                setup_cuda_env()
                print(f"✓ CUDA_PATH set to: {os.environ.get('CUDA_PATH', 'Not set')}")
            except ImportError:
                print("! Could not import CUDA setup module")

            # Modern Open3D (0.13+)
            if hasattr(o3d, 'core') and hasattr(o3d.core, 'initialize_cuda_device'):
                try:
                    o3d.core.initialize_cuda_device()
                    print("✓ CUDA device initialized successfully")

                    # Verify with a tensor operation
                    if hasattr(o3d.core, 'Device'):
                        try:
                            tensor = o3d.core.Tensor.ones((3, 3), o3d.core.Dtype.Float32, o3d.core.Device("CUDA:0"))
                            print("✓ Successfully created CUDA tensor")
                        except Exception as e:
                            print(f"✗ Failed to create CUDA tensor: {e}")
                except Exception as e:
                    print(f"✗ Failed to initialize CUDA device: {e}")
            else:
                # Legacy Open3D
                if hasattr(o3d, 'cuda') or 'cuda' in dir(o3d):
                    print("✓ Legacy Open3D appears to have CUDA support")
                else:
                    print("✗ This version of Open3D does not appear to have CUDA support")
                    print("  Consider upgrading to the latest version (0.17+)")
        except Exception as e:
            print(f"✗ Error checking CUDA support: {e}")

        # Check for ML support
        print("\n=== Checking Open3D ML Support ===\n")
        
        if hasattr(o3d, 'ml'):
            print("✓ Open3D ML module found")
            
            # Check PyTorch backend
            try:
                import importlib
                torch_spec = importlib.util.find_spec("open3d.ml.torch")
                if torch_spec:
                    print("✓ PyTorch backend for Open3D ML is available")
                    
                    # Verify PyTorch CUDA
                    try:
                        import torch
                        if torch.cuda.is_available():
                            print(f"✓ PyTorch CUDA is available: {torch.cuda.get_device_name(0)}")
                        else:
                            print("✗ PyTorch installed but CUDA not available")
                    except ImportError:
                        print("! PyTorch not found, which is needed for the ML backend")
                else:
                    print("✗ PyTorch backend for Open3D ML not found")
            except:
                print("! Could not check PyTorch backend")
                
            # Check TensorFlow backend
            try:
                tf_spec = importlib.util.find_spec("open3d.ml.tf")
                if tf_spec:
                    print("✓ TensorFlow backend for Open3D ML is available")
                else:
                    print("✗ TensorFlow backend for Open3D ML not found")
            except Exception as e:
                print(f"! Could not check TensorFlow backend: {e}")
        else:
            print("✗ Open3D ML module not found")
            
    except ImportError as e:
        print(f"✗ Open3D is not installed: {e}")
        print("\nTo install Open3D with CUDA support, run:")
        print("  pip install open3d==0.17.0")
        print("\nSee OPEN3D_CUDA_INSTALL.md for more details.")
        return False
        
    return True

def try_fix_open3d(is_verbose=False):
    """Attempt to fix common Open3D issues."""
    print("\n=== Attempting to Fix Open3D Installation ===\n")
    
    # Try to set up CUDA environment first
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from unlook.utils.cuda_setup import setup_cuda_env
        if setup_cuda_env():
            print("✓ CUDA environment set up successfully")
        else:
            print("✗ Failed to set up CUDA environment")
    except ImportError:
        print("! Could not import CUDA setup module")
    
    # Check current installation
    try:
        import open3d as o3d
        version = o3d.__version__
        print(f"Current Open3D version: {version}")
        
        # Check if CUDA support is available
        has_cuda = False
        try:
            if hasattr(o3d, 'core') and hasattr(o3d.core, 'initialize_cuda_device'):
                o3d.core.initialize_cuda_device()
                has_cuda = True
        except:
            pass
            
        if has_cuda:
            print("✓ Current installation already has CUDA support")
            print("  No fix needed for CUDA support")
        else:
            print("✗ Current installation does not have CUDA support")
            print("  Attempting to reinstall with CUDA support...")
            
            # Uninstall existing version
            try:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "open3d"], 
                              capture_output=not is_verbose, check=False)
                print("✓ Uninstalled existing Open3D")
            except Exception as e:
                print(f"! Error uninstalling Open3D: {e}")
            
            # Install with CUDA support
            try:
                print("Installing Open3D 0.17.0 with CUDA support...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", "open3d==0.17.0"], 
                                      capture_output=not is_verbose, check=False)
                
                if result.returncode == 0:
                    print("✓ Successfully installed Open3D 0.17.0")
                else:
                    print("✗ Failed to install Open3D 0.17.0")
                    if is_verbose and result.stderr:
                        print(result.stderr)
            except Exception as e:
                print(f"! Error installing Open3D: {e}")
                
    except ImportError:
        print("Open3D not installed, attempting to install...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "open3d==0.17.0"], 
                                  capture_output=not is_verbose, check=False)
            
            if result.returncode == 0:
                print("✓ Successfully installed Open3D 0.17.0")
            else:
                print("✗ Failed to install Open3D")
                if is_verbose and result.stderr:
                    print(result.stderr)
        except Exception as e:
            print(f"! Error installing Open3D: {e}")
    
    # Check if ML support is needed
    try:
        import open3d as o3d
        if hasattr(o3d, 'ml'):
            print("✓ Open3D ML module already available")
        else:
            print("! Open3D ML module not available")
            print("  Attempting to install Open3D ML...")
            
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "open3d-ml"], 
                                      capture_output=not is_verbose, check=False)
                
                if result.returncode == 0:
                    print("✓ Successfully installed Open3D ML")
                else:
                    print("✗ Failed to install Open3D ML")
                    if is_verbose and result.stderr:
                        print(result.stderr)
            except Exception as e:
                print(f"! Error installing Open3D ML: {e}")
    except ImportError:
        print("! Open3D still not installed after fix attempt")
    
    print("\nDone with fix attempts. Please run check again to verify.")

def show_gpu_info():
    """Show information about available GPUs."""
    print("\n=== GPU Information ===\n")
    
    print(check_gpu_info())
    
    # Try PyTorch info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nPyTorch CUDA: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
        else:
            print("\nPyTorch cannot access CUDA")
    except ImportError:
        print("\nPyTorch not installed")
        
    # Show CUDA environment variables
    print("\nCUDA Environment:")
    for var in os.environ:
        if 'CUDA' in var:
            print(f"{var} = {os.environ[var]}")

def main():
    parser = argparse.ArgumentParser(description="Check and diagnose Open3D installation")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix Open3D installation issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--gpu-info", action="store_true", help="Show detailed GPU information")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    print("Open3D Verification Tool for UnLook SDK")
    print("======================================")
    
    if args.gpu_info:
        show_gpu_info()
        return 0
        
    if args.fix:
        try_fix_open3d(args.verbose)
    else:
        # Check installation
        if check_open3d_installation():
            print("\n✓ Open3D installation check completed.")
            print("  If you're still having issues, run with --fix to attempt repairs.")
        else:
            print("\n✗ Open3D installation check failed.")
            print("  Run with --fix to attempt repairs.")
            
    return 0

if __name__ == "__main__":
    sys.exit(main())