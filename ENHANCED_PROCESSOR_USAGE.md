# Enhanced Pattern Processor Usage Guide

The enhanced pattern processor is designed to handle challenging lighting conditions and low-contrast structured light patterns.

## How to Use

### 1. With Static Scanner

The easiest way is to use it with the static scanner by enabling it in the configuration:

```python
from unlook.client.static_scanner import StaticScanner, StaticScanConfig

# Create config with enhanced processor enabled
config = StaticScanConfig(
    quality="high",
    use_enhanced_processor=True,      # Enable the enhanced processor
    enhancement_level=2,              # Enhancement level (0-3)
    debug=True
)

# Create scanner with this config
scanner = StaticScanner(client=client, config=config)
```

### 2. Command Line Usage

Use the updated static scanning example with enhanced processor flags:

```bash
# Basic enhanced scanning
python unlook/examples/static_scanning_example_fixed.py --enhanced-processor

# With maximum enhancement level
python unlook/examples/static_scanning_example_fixed.py --enhanced-processor --enhancement-level 3

# Combined with pattern type
python unlook/examples/static_scanning_example_fixed.py --pattern multi_scale --enhanced-processor
```

### 3. Enhancement Levels

- **Level 0**: No enhancement, just basic processing
- **Level 1**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Level 2**: CLAHE + Gamma correction (default, balanced)
- **Level 3**: CLAHE + Gamma + Bilateral filtering (maximum enhancement)

### 4. When to Use

Use the enhanced processor when:
- Patterns appear as solid colors instead of stripes
- Low contrast between black and white references
- Purple or colored ambient light affecting patterns
- Getting very few correspondences with standard processing

### 5. Test Script

Run the test script to verify the enhanced processor:

```bash
# Test standalone processor
python unlook/examples/test_enhanced_processor.py --mode standalone

# Test with scanner
python unlook/examples/test_enhanced_processor.py --mode scanner
```

## Troubleshooting

If patterns still aren't decoding properly:
1. Increase enhancement level to 3
2. Check that projector brightness is adequate
3. Reduce ambient lighting
4. Ensure proper focus of cameras and projector
5. Try different pattern types (multi_scale often works best)

## Example Output

With enhanced processor enabled, you should see:
- More detected correspondences
- Better pattern visibility in debug images
- Improved point cloud coverage
- More robust scanning in difficult conditions