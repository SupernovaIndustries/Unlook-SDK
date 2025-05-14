# Enhanced Pattern Processor - Now Default

The enhanced pattern processor is now enabled by default in all scanning operations to handle challenging lighting conditions and low-contrast patterns.

## What Changed

1. **StaticScanConfig** now has enhanced processor enabled by default:
   - `use_enhanced_processor=True` (default)
   - `enhancement_level=3` (maximum by default)

2. **All examples** updated to use enhanced processor automatically

3. **Simplified usage** - no need to specify `--enhanced-processor` flag anymore

## Quick Start

### Basic Hardware Scan
```bash
python unlook/examples/static_scanning_example_fixed.py
```

### Process Pre-captured Images
```bash
python unlook/examples/comprehensive_scan_debug.py --mode images --image-dir "path/to/images"
```

### Debug Pattern Issues
```bash
python unlook/examples/comprehensive_scan_debug.py --mode debug --image-dir "path/to/images"
```

### Test Triangulation
```bash
python unlook/examples/debug_triangulation.py "path/to/images" --calibration "path/to/calibration.json"
```

## Enhancement Levels

You can still adjust the enhancement level if needed:

- **Level 0**: No enhancement
- **Level 1**: CLAHE only
- **Level 2**: CLAHE + Gamma correction
- **Level 3**: CLAHE + Gamma + Bilateral filtering (default)

To change the level:
```bash
python unlook/examples/static_scanning_example_fixed.py --enhancement-level 2
```

## Benefits

With enhanced processor as default:
- Better handling of low-contrast patterns
- Improved performance with ambient lighting
- More robust correspondence finding
- Works with problematic reference images
- Handles purple/blue color casts

## Troubleshooting

If you still have issues:

1. **Check your images**:
   ```bash
   python unlook/examples/diagnose_pattern_issue_v2.py "path/to/images"
   ```

2. **Try SIFT-based scanning**:
   ```bash
   python unlook/examples/investor_demo_scan.py "path/to/images"
   ```

3. **Use comprehensive debugger**:
   ```bash
   python unlook/examples/comprehensive_scan_debug.py --mode debug --image-dir "path/to/images"
   ```

## For Developers

To use in your own code:
```python
from unlook.client.static_scanner import StaticScanConfig

# Enhanced processor is already enabled by default
config = StaticScanConfig(quality="high")  

# Or explicitly set parameters
config = StaticScanConfig(
    quality="high",
    use_enhanced_processor=True,  # Already default
    enhancement_level=3           # Already default
)
```

The enhanced processor automatically:
- Normalizes with reference images
- Applies adaptive histogram equalization
- Performs gamma correction
- Reduces noise with bilateral filtering
- Handles inverted reference images

No additional configuration needed!