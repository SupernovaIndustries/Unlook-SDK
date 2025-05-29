#!/bin/bash
# Test script to simulate ReadTheDocs build environment
set -e  # Exit on any error

echo "=== Testing ReadTheDocs Build Locally ==="
echo "Testing on $(python3 --version) - $(uname -s)"
echo

# Check if we're in the right directory
if [ ! -f "conf.py" ] && [ ! -f "source/conf.py" ]; then
    echo "Error: Not in docs directory. Please run from docs/ folder."
    exit 1
fi

# Create a clean virtual environment
echo "Creating test environment..."
if [ -d ".rtd-test-env" ]; then
    echo "Removing existing test environment..."
    rm -rf .rtd-test-env
fi

python3 -m venv .rtd-test-env

# Activate virtual environment (cross-platform)
if [ -f ".rtd-test-env/bin/activate" ]; then
    source .rtd-test-env/bin/activate
elif [ -f ".rtd-test-env/Scripts/activate" ]; then
    source .rtd-test-env/Scripts/activate
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi

echo "✅ Virtual environment activated"

# Simulate ReadTheDocs environment
export READTHEDOCS=True
export RTD_ENVIRONMENT=True

# Install only documentation requirements
echo "Installing documentation requirements..."
pip install --upgrade pip setuptools wheel

# Install basic requirements
pip install -r requirements.txt

# Try to install the package with docs extra
echo "Installing UnLook SDK with docs extra..."
cd ..

# Check if we can install in development mode
if pip install -e ".[docs]" 2>/dev/null; then
    echo "✅ Installed with docs extras"
else
    echo "⚠️  Docs extras failed, installing base package..."
    pip install -e .
fi

cd docs

# Check if source directory exists
if [ -d "source" ]; then
    echo "✅ Found Sphinx source directory"
else
    echo "❌ No source directory found"
    exit 1
fi

# Build documentation
echo "Building documentation..."
if command -v make >/dev/null 2>&1; then
    make clean
    echo "Building HTML documentation..."
    make html
else
    echo "Make not available, using sphinx-build directly..."
    sphinx-build -b html source build/html
fi

echo
echo "=== Build Complete ==="
if [ -d "build/html" ]; then
    echo "✅ Documentation built successfully in build/html/"
    echo "📖 Main page: build/html/index.html"
    
    # Count generated files
    html_files=$(find build/html -name "*.html" | wc -l)
    echo "📄 Generated $html_files HTML files"
else
    echo "❌ Build failed - no output directory found"
    exit 1
fi

echo
echo "To view documentation:"
echo "  python -m http.server 8000 -d build/html"
echo "  Then open: http://localhost:8000"
echo

# Deactivate virtual environment
deactivate
echo "🧹 Test environment: .rtd-test-env/"
echo "   To remove: rm -rf docs/.rtd-test-env/"