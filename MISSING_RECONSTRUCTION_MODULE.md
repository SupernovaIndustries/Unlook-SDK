# Missing Reconstruction Module Issue

## Problem
The module `unlook.client.scanning.reconstruction` is missing but is required for 3D scanning functionality.

## Expected Classes
- `Integrated3DPipeline`
- `ScanningResult`

## Impact
Without this module, the 3D scanning pipeline cannot process captured images into point clouds.

## Temporary Workaround
The `enhanced_3d_scanning_pipeline_v2.py` example includes placeholder classes to allow testing of the capture functionality.

## TODO
1. Implement the reconstruction module with:
   - Gray code decoding
   - Stereo correspondence matching
   - Triangulation
   - Point cloud generation
   - ISO/ASTM 52902 compliance features

2. Or find the existing implementation if it exists elsewhere in the codebase.

## Priority
HIGH - This is essential for 3D scanning functionality.