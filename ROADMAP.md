# Unlook SDK Roadmap

This document outlines the development roadmap for the Unlook SDK project, serving as a guide for current and future development priorities.

## Current Focus

- **Real-time Scanning Optimization** - Primary focus on making real-time scanning work as a handheld scanner
  - Enhance point cloud generation speed using Open3D
  - Optimize for smooth frame rates on consumer hardware
  - Improve visualization for real-time feedback

## Short-term Goals (Next 3 Months)

- Complete transition from python-pcl to Open3D
- Improve cross-platform compatibility (Windows, Linux, macOS)
- Enhance real-time point cloud visualization
- Clean up codebase by removing legacy robust scanning code
- Create comprehensive documentation

## Medium-term Goals (3-6 Months)

- **Neural Network Enhancement**
  - Integrate neural network-based point cloud enhancement
  - Add denoising capabilities
  - Implement point cloud upsampling
  - Optimize for both NVIDIA (CUDA) and AMD (ROCm) GPUs

- **Performance Optimization**
  - Profile and optimize critical processing paths
  - Implement multi-threaded processing for applicable components
  - Balance CPU/GPU workloads for optimal performance

## Long-term Vision (6+ Months)

- **Advanced Feature Development**
  - Mesh reconstruction from point clouds
  - Texture mapping capabilities
  - Surface normal estimation improvements
  - Integration with industry-standard 3D workflows

- **User Experience Improvements**
  - Intuitive calibration wizards
  - Visual feedback for scanning quality
  - Simplified scanner configuration

- **SDK Ecosystem Expansion**
  - Python package distribution through PyPI
  - Language bindings for C++, C#, etc.
  - Example integrations with popular 3D software

## Considerations and Constraints

- Focus on maintaining compatibility with consumer-grade hardware
- Prioritize real-time performance over high-resolution results
- Support progressive enhancement when higher-end hardware is available

---

This roadmap is a living document and will be updated as development progresses and priorities evolve. Future development decisions will be made based on user feedback, technical constraints, and emerging technologies in the 3D scanning field.