# UnLook SDK Visualization Tools

This directory contains tools for visualizing and debugging 3D scan results from the UnLook SDK.

## Visualization Tools

### view_scan.py

A command-line tool for viewing and analyzing scan results.

**Usage:**
```bash
python view_scan.py [scan_directory] [options]
```

**Options:**
- `--raw` or `-r`: Include raw/temporary point clouds in analysis
- `--filter` or `-f`: Apply filtering to point clouds before visualization
- `--info-only` or `-i`: Show information only, without visualization
- `--list` or `-l`: List all scan directories and exit
- `--latest`: Automatically select the latest scan

**Example:**
```bash
# View the latest scan
python view_scan.py ./scans --latest

# List all available scans
python view_scan.py ./scans --list

# Show information about a specific scan without visualization
python view_scan.py ./scans/scan_20250509_123134_combined_ultra --info-only
```

### visualize_point_cloud.py

A simple utility for visualizing point clouds and meshes.

**Usage:**
```bash
python visualize_point_cloud.py <file_path> [options]
```

**Options:**
- `--debug` or `-d`: Enable debug mode with additional output
- `--output` or `-o`: Output directory for saving screenshots
- `--no-gui`: Run without GUI (screenshots only)

**Example:**
```bash
# Visualize a point cloud
python visualize_point_cloud.py ./scans/latest_scan/results/scan_point_cloud.ply

# Visualize a mesh and save screenshots
python visualize_point_cloud.py ./scans/latest_scan/results/scan_mesh.obj --output ./visualization_output
```

## Debugging Scan Issues

If you encounter issues with your 3D scans, the following tools can help diagnose the problem:

1. Use `view_scan.py` to examine the scan results and debug information:
   ```bash
   python view_scan.py ./scans/problematic_scan --info-only
   ```

2. Look at the debug images in the scan's debug directory:
   - Check shadow masks (`combined_mask.png`, `shadow_mask.png`)
   - Examine point correspondences (`correspondence_vis.png`)
   - Analyze pattern decoding in the left_debug and right_debug directories

3. Compare successful scans with problematic ones:
   ```bash
   python view_scan.py ./scans/successful_scan --info-only
   python view_scan.py ./scans/problematic_scan --info-only
   ```

4. Visualize point clouds to check for specific issues:
   ```bash
   python visualize_point_cloud.py ./scans/problematic_scan/results/scan_point_cloud.ply
   ```

## Common Issues and Solutions

### Empty or Sparse Point Clouds

- **Problem**: The scan produces very few points (< 100)
- **Diagnosis**: Use `view_scan.py` to check:
  - Number of valid pixels in masks
  - Number of correspondences found
- **Solutions**:
  - Ensure proper white/black calibration patterns
  - Try reducing the mask threshold (`--filter` option)
  - Ensure the object has enough texture and isn't too reflective
  - Check camera calibration

### Misaligned or Distorted Point Clouds

- **Problem**: The scan produces points, but they don't form a coherent object
- **Diagnosis**: Check correspondence visualization and rectified images
- **Solutions**:
  - Recalibrate stereo cameras
  - Ensure cameras haven't moved since calibration
  - Check for reflections or lighting issues

### White Squares in Captured Images

- **Problem**: Captured images show white squares instead of proper patterns
- **Diagnosis**: Check the captured images in the scan's captures directory
- **Solutions**:
  - Ensure proper implementation of `_display_pattern` method
  - Check if projector supports direct image display
  - Verify that pattern resolution matches projector resolution