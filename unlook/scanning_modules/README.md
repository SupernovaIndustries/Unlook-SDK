# Unlook Scanning Modules

This directory contains specifications and metadata for Unlook 3D scanning modules. These specifications are used by both the client SDK and server components to understand the capabilities, limitations, and configurations of different scanning hardware.

## Purpose

The scanning module specifications serve several key purposes:

1. **Auto-Discovery**: Allows the server to automatically detect and identify connected hardware
2. **Capability Announcement**: Broadcasts available features to clients during discovery
3. **Parameter Boundaries**: Defines operational limits for configuration validation
4. **Performance Expectations**: Sets realistic expectations for scan quality and resolution
5. **Technical Documentation**: Provides detailed reference for developers and integrators

## Directory Structure

- `scanning_modules/` - Root directory for all scanning module specifications
  - `README.md` - This file
  - `structured_light_specs.json` - Specifications for the structured light module
  - `depth_camera_specs.json` - *(Future)* Specifications for depth camera modules
  - `photogrammetry_specs.json` - *(Future)* Specifications for photogrammetry modules
  - `configurations/` - *(Future)* Predefined configurations for specific use cases
  - `calibration/` - *(Future)* Calibration parameters and procedures

## Specification Format

Each scanning module's specifications are stored in a JSON file with a standardized schema that includes:

- **Module Identification**: Unique identifiers and versioning
- **Hardware Details**: Components, capabilities, and physical characteristics
- **Scanning Capabilities**: Resolution, accuracy, range, and limitations
- **Software Requirements**: Processing pipeline and compatibility
- **Performance Metrics**: Expected timeframes and quality levels
- **Limitations**: Known issues and constraints
- **Certification**: Compliance with standards

## Integration

These specifications are used by:

1. **Client SDK**: To validate scan parameters and provide appropriate user feedback
2. **Server**: To identify connected hardware and enforce operational limits
3. **Custom OS**: To enable hot-plugging and automatic configuration of scanning modules
4. **Applications**: To adapt their workflows to the available hardware capabilities

## Future Development

This directory will expand as new scanning modules are developed, with each module maintaining its own specification file. Module developers should follow the schema defined in the existing files to ensure compatibility with the Unlook platform.