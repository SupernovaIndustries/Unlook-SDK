# Files to Delete

This document keeps track of files that should be deleted or consolidated in the repository.

## Files to Delete

The following files are obsolete and can be safely deleted:

1. `requirements.txt` - Replaced by `client-requirements.txt` and server-specific requirements
2. `unlook/client/robust_scanner.py` - Replaced by improved implementation
3. `unlook/examples/robust_scanning_example.py` - Replaced by new example
4. `unlook/client/debug/left_mask.png` - Debug files that should not be in repository
5. `unlook/client/debug/right_mask.png` - Debug files that should not be in repository
6. `setup_cuda_env.py` - Replaced by improved GPU setup scripts
7. `unlook/client/robust_scan.py` - Old robust scanning implementation
8. `unlook/client/robust_structured_light.py` - Old structured light implementation
9. `unlook/client/advanced_structured_light.py` - Outdated implementation
10. `unlook/client/scanner3d.py` - Replaced by `/unlook/client/new_scanner3d.py`
11. `unlook/client/structured_light.py` - Replaced by `/unlook/client/new_structured_light.py`
12. Old debug files in the `unlook_debug` directory that are no longer needed

## Files to Consolidate

The following files should be merged or consolidated:

1. `INSTALLATION.md` and `INSTALL.md` - Merge into a single installation guide
2. `OPENCV_CUDA_INSTALL.md` and related scripts - Should be removed as they are obsolete
3. The following documentation files should be moved to the `docs` directory:
   - `DOCUMENTATION_TOOLS.md`
   - `MODULE_STRUCTURE.md`
   - `REALTIME_SCANNING.md`
   - `ROADMAP.md`
   - `UPGRADE_NOTES.md`

## Rename Operations

The following files should be renamed for clarity:

1. `/unlook/client/new_scanner3d.py` → `/unlook/client/scanner3d.py`
2. `/unlook/client/new_structured_light.py` → `/unlook/client/structured_light.py`
3. `/unlook/examples/realtime_scanning_example.py` → `/unlook/examples/real_time_scanning_example.py`

## Directory Structure After Cleanup

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
  ├── static_scanner.py          # Static 3D scanner implementation
  ├── gpu_utils.py               # GPU acceleration utilities
  ├── nn_processing.py           # Neural network processing
  ├── point_cloud_nn.py          # Neural network point cloud processing
  ├── enhanced_gray_code.py      # Enhanced gray code patterns
  ├── projector_adapter.py       # Projector adapter
  └── visualization.py           # Visualization utilities
```

## Implementation Note

Moving forward, development should focus on:
1. Enhancing both real-time and static scanners with improved performance
2. Improving the neural network-based point cloud processing
3. Supporting both NVIDIA and AMD GPUs where possible
4. Implementing better user interfaces and visualization
5. Creating a better installer and configuration system