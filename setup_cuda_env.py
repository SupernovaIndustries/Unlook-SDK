#!/usr/bin/env python3
"""
CUDA Environment Setup Script for UnLook SDK

This script checks and sets up the CUDA environment for the UnLook SDK,
ensuring that all GPU-accelerated libraries can find the CUDA installation.

Usage:
    python setup_cuda_env.py
    
    # Then in the same terminal session, run your realtime scanning:
    python unlook/examples/realtime_scanning_example.py
"""

import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cuda_setup")

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def main():
    parser = argparse.ArgumentParser(description="Set up CUDA environment for UnLook SDK")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--check-only", action="store_true", help="Only check CUDA availability without setting up")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        from unlook.utils.cuda_setup import setup_cuda_env, is_cuda_available
        
        if args.check_only:
            # Just check if CUDA is available
            if is_cuda_available():
                print("\033[92m✓ CUDA is available\033[0m")
                return 0
            else:
                print("\033[91m✗ CUDA is not available\033[0m")
                return 1
        
        # Set up the CUDA environment
        success = setup_cuda_env()
        
        if success:
            print("\033[92m✓ CUDA environment set up successfully\033[0m")
            print(f"CUDA_PATH = {os.environ.get('CUDA_PATH', 'Not set')}")
            
            # Check if CUDA is actually available through libraries
            if is_cuda_available():
                print("\033[92m✓ CUDA functionality verified\033[0m")
                
                # Print instructions
                print("\n\033[1mTo use CUDA in your application:\033[0m")
                print("1. Run your Python script in this same terminal session")
                print("2. Or export these environment variables in your shell:")
                print(f"   export CUDA_PATH=\"{os.environ.get('CUDA_PATH', '')}\"")
                if os.name != 'nt':  # Linux/Mac
                    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                    if ld_path:
                        print(f"   export LD_LIBRARY_PATH=\"{ld_path}\"")
                
                return 0
            else:
                print("\033[91m✗ CUDA environment set up but CUDA is still not available\033[0m")
                print("This may indicate a problem with your CUDA installation or GPU drivers.")
                return 1
        else:
            print("\033[91m✗ Failed to set up CUDA environment\033[0m")
            print("Could not find CUDA path. Please make sure CUDA is installed correctly.")
            return 1
            
    except ImportError as e:
        print(f"\033[91m✗ Error: {e}\033[0m")
        print("Could not import the cuda_setup module. Make sure you're running from the root directory.")
        return 1
    except Exception as e:
        print(f"\033[91m✗ Error: {e}\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())