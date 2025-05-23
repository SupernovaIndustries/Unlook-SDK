Overview
========

The Unlook SDK is a comprehensive software development kit designed for controlling and operating the Unlook 3D scanning system. This overview will help you understand the key components and architecture of the system.

System Architecture
-----------------

The Unlook SDK follows a client-server architecture:

- **Client**: Runs on your development machine or end-user device, providing control interface and processing capabilities
- **Server**: Runs on the hardware connected to cameras and projectors (typically a Raspberry Pi), handling low-level hardware control

This separation allows for flexible deployment scenarios, from single-machine operations to distributed setups where the scanning hardware is networked.

Key Components
------------

The SDK consists of several key components:

Core Components
^^^^^^^^^^^^^

- **Discovery Service**: Automatically finds Unlook scanners on the network
- **Event System**: Facilitates asynchronous communication between components
- **Protocol**: Defines the communication format between client and server

Client Components
^^^^^^^^^^^^^^

- **UnlookClient**: Main client class for connecting to and controlling scanners
- **Camera Module**: Manages camera operations, streaming, and configuration
- **Projector Module**: Controls projector patterns and sequences
- **Scanner Modules**: High-level modules for different scanning methods
- **Real-time Scanner**: Specialized module for continuous, high-speed scanning
- **Hand Pose Tracking**: Advanced hand detection and gesture recognition with 3D triangulation

Server Components
^^^^^^^^^^^^^^

- **UnlookServer**: Main server class that manages hardware and client connections
- **Hardware Interface**: Low-level control for cameras, projectors, and other devices
- **DLP Controller**: Specialized driver for DLP projectors

Scanning Technologies
------------------

The SDK supports several scanning technologies:

- **Structured Light**: Uses projected patterns and stereo vision for precise 3D reconstruction
- **Real-time Scanning**: Optimized for continuous, handheld scanning operations
- **Hand Pose Tracking**: Real-time hand detection and gesture recognition using stereo vision with LED point projection
- **Depth Sensor** (Coming Soon): Support for time-of-flight and structured light depth sensors
- **Point Projector** (Coming Soon): Support for laser/IR dot pattern projectors

Hardware Support
-------------

The SDK is designed to work with:

- **Cameras**: Stereo camera systems, with focus on Raspberry Pi cameras
- **Projectors**: DLP-based projectors, particularly those with DLP342X controllers
- **Processing Hardware**: Supports both CPU and GPU processing (NVIDIA/AMD)

Getting Started
------------

To start using the Unlook SDK, see the following sections:

- :doc:`../getting_started`: Quick start guide
- :doc:`../installation`: Detailed installation instructions
- :doc:`client_server_architecture`: Learn about how the client and server components work together
- :doc:`realtime_scanning`: Guide to using the real-time scanning features
- :doc:`handpose_setup`: Setting up and using hand pose tracking and gesture recognition