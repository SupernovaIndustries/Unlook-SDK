Welcome to Unlook SDK's documentation!
=====================================

.. image:: _static/unlook_logo.png
   :width: 200px
   :align: center
   :alt: Unlook SDK Logo

**Unlook SDK** is a powerful and flexible Python framework designed to control Unlook - a modular open-source 3D scanning system that supports interchangeable optics and sensors, similar to how cameras support different lenses. This SDK provides a complete solution for managing various scanning modules, from structured light systems to depth sensors, point projectors, and more.

.. note::
   This documentation is continuously updated. Make sure you're viewing the most recent version.

Key Features
-----------

- **Modular Hardware Support**: Control multiple scanning technologies through a unified interface
- **Client-Server Architecture**: Clean separation between control clients and acquisition servers
- **Auto-Discovery**: Automatic detection of available Unlook scanners on local network
- **Comprehensive Control**:
   - DLPC342X projector interface for structured light modules
   - Multiple camera support for stereo and multi-view setups
   - Depth sensor integration
   - Point projector management
- **Advanced Video Streaming**:
   - Standard streams for visualization and monitoring
   - Low-latency direct streams for real-time applications
   - Automated pattern sequences for structured light scanning
   - Precise projector-camera synchronization
- **3D Scanning**:
   - Simplified 3D scanning API - create a 3D scan in just a few lines of code
   - Real-time scanning mode - for handheld applications with GPU acceleration
   - Stereo camera calibration and rectification
   - Point cloud generation and filtering
   - Mesh creation and export to multiple formats
- **GPU Acceleration**: Optimized processing using GPU when available
- **Neural Network Enhancement**: Point cloud filtering and enhancement using machine learning
- **Optimized Communication**: Efficient image transfer and control using ZeroMQ
- **Easy Expandability**: Modular architecture to add new hardware and algorithms

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   installation
   user_guide/index
   api_reference/index
   examples/index
   advanced/index
   troubleshooting
   roadmap
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`