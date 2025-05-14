# Unlook SDK - New Structure Overview

## Major Reorganization Completed

The Unlook SDK has been reorganized for better modularity and maintainability. Here's the new structure:

### Client Module Structure

```
unlook/client/
├── __init__.py
├── scanner.py (Main UnlookClient)
├── scan_config.py
├── scanner3d.py
├── camera.py
├── camera_config.py
├── streaming.py
├── visualization.py
├── scanning/              # All scanning functionality
│   ├── __init__.py
│   ├── static_scanner.py
│   ├── realtime_scanner.py
│   ├── patterns/         # Pattern generation and processing
│   │   ├── __init__.py
│   │   ├── enhanced_gray_code.py
│   │   ├── enhanced_phaseshift.py
│   │   ├── enhanced_patterns.py
│   │   ├── enhanced_pattern_processor.py
│   │   ├── pattern_generator.py
│   │   ├── pattern_processor.py
│   │   └── structured_light.py
│   ├── calibration/      # Calibration utilities
│   │   ├── __init__.py
│   │   ├── calibration_utils.py
│   │   └── stereo_calibrator.py
│   └── reconstruction/   # 3D reconstruction
│       ├── __init__.py
│       ├── direct_triangulator.py
│       └── proper_correspondence_finder.py
└── projector/           # Projector control
    ├── __init__.py
    ├── projector.py
    └── projector_adapter.py
```

### Examples Structure

```
unlook/examples/
├── scanning/            # All scanning examples
│   ├── __init__.py
│   ├── comprehensive_scan_debug.py
│   ├── scan_from_images.py
│   ├── static_scanning_example_fixed.py
│   ├── realtime_scanning_example.py
│   └── [other scanning examples]
├── camera_test.py
├── camera_config_example.py
├── pattern_sequence_example.py
└── [other examples]
```

## Import Changes

### Before:
```python
from unlook.client.static_scanner import StaticScanner
from unlook.client.realtime_scanner import RealTimeScanner
from unlook.client.projector import ProjectorClient
from unlook.client.enhanced_pattern_processor import EnhancedPatternProcessor
```

### After:
```python
from unlook.client.scanning import StaticScanner, RealTimeScanner
from unlook.client.projector import ProjectorClient
from unlook.client.scanning.patterns.enhanced_pattern_processor import EnhancedPatternProcessor
```

## Installation

To use the new structure, install the SDK in development mode:

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e .

# Or with 3D support
pip install -e ".[3d]"
```

## Benefits

1. **Better Organization**: Related functionality grouped together
2. **Clearer Imports**: More intuitive import paths
3. **Modularity**: Easier to add new features or patterns
4. **Maintainability**: Clear separation of concerns
5. **Scalability**: Room for future expansion

## Disabled Components

The following components have been temporarily disabled (renamed with .disabled):
- Neural network modules (nn_mvs2d, nn_processing, point_cloud_nn)
- GPU utilities (gpu_utils, cuda_setup, check_gpu)

These can be re-enabled when needed by removing the .disabled extension.

## Next Steps

1. Install the SDK using `pip install -e .`
2. Test your code with the new import structure
3. Run scanning examples to verify everything works
4. Update any custom scripts to use new import paths