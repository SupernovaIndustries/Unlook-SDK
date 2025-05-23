Changelog
=========

This page documents the major changes and improvements made to the Unlook SDK.

Version 1.1.0 (2025-01-XX)
-------------------------

**New Features**

- **Enhanced Hand Pose Tracking Module**: Complete rewrite of the handpose tracking system
  
  - Added full-screen professional GUI with interactive LED controls
  - Implemented stereo vision 3D triangulation for accurate hand positioning
  - Added support for LED point projection (LED1) for enhanced depth sensing
  - Integrated flood illumination (LED2) for improved scene lighting
  - Added real-time gesture recognition with both rule-based and ML approaches
  - Implemented advanced image preprocessing with CLAHE and bilateral filtering
  - Added mouse and keyboard controls for LED intensity adjustment
  - Created streaming architecture with callback patterns for better performance

- **LED Controller Enhancements**:
  
  - Removed hard-coded LED1 limitation that prevented point projection
  - Added support for dual LED operation with independent intensity control  
  - Implemented default 50mA operation for both LEDs
  - Added interactive LED control sliders in GUI applications

- **Comprehensive Documentation**:
  
  - Added complete API reference for handpose module
  - Created extensive examples with working code samples
  - Added setup and configuration guides with troubleshooting
  - Integrated handpose documentation into main documentation structure

**Bug Fixes**

- Fixed gesture recognition method name from `recognize_gesture_3d()` to `recognize_gestures_3d()`
- Removed unsupported `prioritize_left_camera` parameter from `track_hands_3d()` method
- Fixed LED1 hard-coded limitation that prevented point projection functionality

**Documentation Improvements**

- Added handpose API reference with detailed class descriptions
- Created comprehensive handpose tracking examples
- Added handpose setup guide with system requirements and configuration
- Updated overview documentation to include handpose capabilities

Version 1.0.0 (2024-XX-XX)
-------------------------

**Initial Release**

- Core client-server architecture
- Basic camera and projector control
- Structured light scanning capabilities  
- Real-time scanning mode
- Auto-discovery of network scanners
- Basic documentation and examples