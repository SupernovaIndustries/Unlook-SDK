#!/usr/bin/env python
"""Setup script for Unlook SDK - backward compatibility wrapper."""

import warnings
from setuptools import setup

warnings.warn(
    "The setup.py file is deprecated. This project now uses pyproject.toml "
    "for configuration. Please use 'pip install .' or 'pip install -e .' "
    "instead of 'python setup.py install'.",
    DeprecationWarning,
    stacklevel=2
)

# Minimal setup() call for backward compatibility
# All actual configuration is now in pyproject.toml
if __name__ == "__main__":
    setup()