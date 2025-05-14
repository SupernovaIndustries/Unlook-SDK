#!/usr/bin/env python
"""Setup script for Unlook SDK."""

from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='unlook-sdk',
    version='0.1.0',
    description='SDK for Unlook 3D structured light scanning system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Unlook Team',
    author_email='team@unlook.com',
    url='https://github.com/unlook/unlook-sdk',
    packages=find_packages(),
    package_data={
        'unlook': [
            'calibration/**/*',
            'scanning_modules/*',
        ],
    },
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
        'pyzmq>=22.0.0',
        'pillow>=8.0',
        'matplotlib>=3.3.0',
        'scipy>=1.5.0',
        'zeroconf>=0.38.0',
        'netifaces>=0.11.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=20.8b1',
            'flake8>=3.8',
            'mypy>=0.782',
        ],
        '3d': [
            'open3d>=0.15.0',
            'trimesh>=3.9.0',
            'pymeshlab>=2021.7',
        ],
        'server': [
            'adafruit-circuitpython-busdevice>=5.0.0',
            'adafruit-circuitpython-register>=1.9.0',
            'picamera2>=0.3.0',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
    ],
    entry_points={
        'console_scripts': [
            'unlook-scanner=unlook.client.cli:main',
            'unlook-server=unlook.server.cli:main',
        ],
    },
)