from setuptools import setup, find_packages

setup(
    name="unlook-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pyzmq>=25.0.0",
        "zeroconf>=0.38.0",
        "smbus2>=0.4.1",
    ],
    extras_require={
        "server": ["picamera2"],
    },
)