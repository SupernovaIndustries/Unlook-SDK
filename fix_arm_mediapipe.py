#!/usr/bin/env python3
"""
Fix MediaPipe installation for ARM processors (Surface Pro, etc.)
This script removes JAX dependencies that cause issues on ARM.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return its output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    print("Fixing MediaPipe for ARM processors...")
    print("=" * 50)
    
    # Step 1: Uninstall problematic packages
    print("\n1. Removing problematic packages...")
    packages_to_remove = [
        "jax",
        "jaxlib", 
        "tensorflow",
        "tensorflow-estimator",
        "tensorflow-io-gcs-filesystem"
    ]
    
    for package in packages_to_remove:
        run_command(f"{sys.executable} -m pip uninstall -y {package}")
    
    # Step 2: Install MediaPipe without dependencies
    print("\n2. Installing MediaPipe without dependencies...")
    if not run_command(f"{sys.executable} -m pip install mediapipe --no-deps"):
        print("Failed to install MediaPipe")
        return
    
    # Step 3: Install only necessary dependencies
    print("\n3. Installing necessary dependencies...")
    required_deps = [
        "numpy",
        "opencv-contrib-python",
        "protobuf>=3.11,<4",
        "attrs>=19.1.0",
        "matplotlib"
    ]
    
    for dep in required_deps:
        run_command(f"{sys.executable} -m pip install '{dep}'")
    
    # Step 4: Create a patched mediapipe import
    print("\n4. Creating import patch...")
    patch_content = '''"""
Patch for MediaPipe to work without TensorFlow/JAX on ARM
"""
import sys
import types

# Create dummy tensorflow module
tensorflow = types.ModuleType('tensorflow')
tensorflow.tools = types.ModuleType('tensorflow.tools')
tensorflow.tools.docs = types.ModuleType('tensorflow.tools.docs')
tensorflow.tools.docs.doc_controls = types.ModuleType('tensorflow.tools.docs.doc_controls')

# Add dummy decorators
tensorflow.tools.docs.doc_controls.do_not_generate_docs = lambda: lambda x: x
tensorflow.tools.docs.doc_controls.do_not_doc_inheritable = lambda: lambda x: x

# Inject into sys.modules
sys.modules['tensorflow'] = tensorflow
sys.modules['tensorflow.tools'] = tensorflow.tools
sys.modules['tensorflow.tools.docs'] = tensorflow.tools.docs
sys.modules['tensorflow.tools.docs.doc_controls'] = tensorflow.tools.docs.doc_controls

print("MediaPipe ARM patch applied successfully!")
'''
    
    patch_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(patch_dir, "mediapipe_arm_patch.py")
    
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    print(f"\nPatch file created: {patch_file}")
    
    # Step 5: Test the installation
    print("\n5. Testing MediaPipe installation...")
    test_code = f'''
import sys
sys.path.insert(0, r"{patch_dir}")
import mediapipe_arm_patch
import mediapipe as mp
print("MediaPipe version:", mp.__version__)
print("Hands solution available:", hasattr(mp.solutions, 'hands'))
'''
    
    test_file = os.path.join(patch_dir, "test_mediapipe.py")
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    if run_command(f"{sys.executable} {test_file}"):
        print("\n✅ MediaPipe installation successful!")
        print("\nTo use MediaPipe in your scripts, add this at the beginning:")
        print(f"import sys")
        print(f'sys.path.insert(0, r"{patch_dir}")')
        print(f"import mediapipe_arm_patch")
        print("import mediapipe as mp")
    else:
        print("\n❌ MediaPipe installation failed")
        print("You may need to install MediaPipe manually for ARM")
    
    # Cleanup
    os.remove(test_file)

if __name__ == "__main__":
    main()