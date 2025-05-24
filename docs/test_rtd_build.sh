#!/bin/bash
# Test script to simulate ReadTheDocs build environment

echo "=== Testing ReadTheDocs Build Locally ==="
echo

# Create a clean virtual environment
echo "Creating test environment..."
python -m venv .rtd-test-env
source .rtd-test-env/bin/activate || .rtd-test-env/Scripts/activate

# Simulate ReadTheDocs environment
export READTHEDOCS=True

# Install only documentation requirements
echo "Installing documentation requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Try to install the package with docs extra
echo "Installing UnLook SDK with docs extra..."
cd ..
pip install -e ".[docs]"
cd docs

# Build documentation
echo "Building documentation..."
make clean
make html

echo
echo "=== Build Complete ==="
echo "Check for errors above. If successful, documentation is in build/html/"
echo

# Deactivate and optionally remove test environment
deactivate
echo "Test environment created in .rtd-test-env/"
echo "To remove it: rm -rf .rtd-test-env/"