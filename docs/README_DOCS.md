# UnLook SDK Documentation

This directory contains the documentation for the UnLook SDK.

## Building Documentation Locally

### Prerequisites

1. Install documentation dependencies:
```bash
pip install -e ".[docs]"
```

Or manually:
```bash
pip install sphinx sphinx_rtd_theme sphinx-copybutton myst-parser
```

### Building

From the `docs/` directory:

```bash
# Clean previous builds
make clean

# Build HTML documentation
make html
```

The built documentation will be in `docs/build/html/`.

### Viewing

Open `docs/build/html/index.html` in your browser.

## ReadTheDocs Integration

The documentation is automatically built and hosted on ReadTheDocs:
https://unlook-sdk.readthedocs.io/

### Configuration Files

- `.readthedocs.yaml`: Main ReadTheDocs configuration
- `docs/requirements.txt`: Minimal dependencies for RTD build
- `docs/source/conf.py`: Sphinx configuration with mocked imports

### Important Notes for ReadTheDocs

1. **Mocked Imports**: Heavy dependencies (OpenCV, Open3D, MediaPipe, etc.) are mocked in `conf.py` to avoid installation timeouts.

2. **Minimal Requirements**: Only essential documentation packages are installed on RTD.

3. **No PDF/EPUB**: Only HTML format is built to reduce memory usage.

4. **Python 3.9**: Using Python 3.9 for best compatibility.

## Troubleshooting ReadTheDocs Builds

### Common Issues

1. **Import Errors**: Add the module to `MOCK_MODULES` in `conf.py`

2. **Build Timeouts**: Remove heavy dependencies from `requirements.txt`

3. **Memory Errors**: Disable PDF/EPUB formats in `.readthedocs.yaml`

4. **Missing Files**: Ensure all referenced files are committed to git

### Testing Locally

To test the ReadTheDocs build locally:

```bash
# Create a clean environment
python -m venv rtd-test
source rtd-test/bin/activate  # or rtd-test\Scripts\activate on Windows

# Install only RTD requirements
pip install -r docs/requirements.txt

# Build
cd docs
make clean html
```

## Documentation Structure

```
docs/
├── source/
│   ├── _static/         # Static files (CSS, images)
│   ├── _templates/      # Custom templates
│   ├── api_reference/   # API documentation
│   ├── examples/        # Example documentation
│   ├── user_guide/      # User guides
│   ├── conf.py          # Sphinx configuration
│   └── index.rst        # Main documentation index
├── requirements.txt     # RTD requirements
├── Makefile            # Build commands
└── make.bat            # Windows build commands
```

## Adding New Documentation

1. Create `.rst` or `.md` files in appropriate directories
2. Add to relevant `index.rst` toctree
3. Test build locally
4. Commit and push - RTD will rebuild automatically

## API Documentation

API docs are generated automatically from docstrings using autodoc.
Ensure all public modules, classes, and functions have proper docstrings.

Example:
```python
def my_function(param1: str, param2: int) -> bool:
    """Short description.
    
    Longer description with more details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> my_function("hello", 42)
        True
    """
    pass
```