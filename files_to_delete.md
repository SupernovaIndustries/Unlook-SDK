# Files to Delete

The codebase has been updated to focus exclusively on real-time scanning with Open3D-based point cloud processing. The following files are now obsolete and can be safely deleted to streamline the codebase:

## Core Modules to Delete

1. **Legacy Scanning Implementations**:
   - `/unlook/client/robust_scan.py` - Old robust scanning implementation
   - `/unlook/client/robust_structured_light.py` - Old structured light implementation
   - `/unlook/client/scanner3d.py` - Replaced by `/unlook/client/new_scanner3d.py`
   - `/unlook/client/structured_light.py` - Replaced by `/unlook/client/new_structured_light.py`
   - `/unlook/client/advanced_structured_light.py` - Outdated implementation

2. **Legacy Example Scripts**:
   - `/unlook/examples/robust_scanning_example.py` - Old example for robust scanning
   - `/unlook/examples/new_robust_scanning_example.py` - Will focus exclusively on real-time scanning

## Rename Operations

The following files should be renamed for clarity (they have "new_" prefixes):

1. `/unlook/client/new_scanner3d.py` → `/unlook/client/scanner3d.py`
2. `/unlook/client/new_structured_light.py` → `/unlook/client/structured_light.py`
3. `/unlook/examples/realtime_scanning_example.py` → `/unlook/examples/real_time_scanning_example.py`

## Renaming and Directory Structure

After cleaning up the codebase, the restructured scanner modules should look like:

```
/unlook/client/
  ├── camera.py                  # Camera interface
  ├── camera_calibration.py      # Stereo calibration
  ├── projector.py               # Projector interface
  ├── scanner.py                 # Base scanner class
  ├── structured_light.py        # Structured light patterns (renamed from new_structured_light.py)
  ├── scanner3d.py               # Base 3D scanner (renamed from new_scanner3d.py)
  ├── realtime_scanner.py        # Real-time 3D scanner implementation
  ├── point_cloud_nn.py          # Neural network point cloud processing
  └── visualization.py           # Visualization utilities
```

## Implementation Note

Moving forward, development should focus on:
1. Enhancing the real-time scanner with improved performance
2. Improving the neural network-based point cloud processing
3. Supporting both NVIDIA and AMD GPUs where possible
4. Implementing better user interfaces and visualization