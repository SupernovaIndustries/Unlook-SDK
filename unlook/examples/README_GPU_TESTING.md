# GPU Acceleration Testing for Unlook SDK

This document explains how to test GPU acceleration in the Unlook SDK using the provided test scripts.

## Overview

The Unlook SDK can use GPU acceleration to improve performance for compute-intensive operations like point triangulation and correspondence matching. This is especially important for real-time 3D scanning and large point clouds.

We provide several test scripts to verify that GPU acceleration is working correctly and to measure its performance benefits:

1. **test_gpu_acceleration.py**: Tests GPU acceleration using sample data from previous scans
2. **gpu_utils_benchmark.py**: Benchmarks GPU acceleration for matrix operations and triangulation with synthetic data
3. **static_scanning_example.py**: Test GPU acceleration during a real scan (when connected to hardware)

## Prerequisites

To use GPU acceleration, you need:

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (9.0 or later)
- CuPy installed (`pip install cupy-cuda11x`, adjust version to match your CUDA toolkit)
- OpenCV with CUDA support (optional, for additional acceleration)

## Testing GPU Acceleration

### 1. Verify CUDA Setup

First, verify that CUDA is properly set up by running:

```bash
python -c "from unlook.utils.cuda_setup import is_cuda_available; print(f'CUDA available: {is_cuda_available()}')"
```

This should output `CUDA available: True` if CUDA is correctly configured.

### 2. Run GPU Utilities Diagnostic

Run the GPU utilities diagnostic to get detailed information about your GPU:

```bash
python -c "from unlook.client.gpu_utils import diagnose_gpu; print(diagnose_gpu())"
```

This will show detailed information about GPU availability, CUDA version, memory, and device information.

### 3. Run GPU Utilities Benchmark

The benchmark script runs various operations on both CPU and GPU to measure the performance difference:

```bash
python unlook/examples/gpu_utils_benchmark.py
```

Options:
- `--matrix-size SIZE`: Change the size of test matrices (default: 2000)
- `--num-runs RUNS`: Change the number of benchmark runs (default: 5)
- `--cpu-only`: Force CPU-only processing (for comparison)
- `--verbose`: Enable verbose output

Example with custom parameters:
```bash
python unlook/examples/gpu_utils_benchmark.py --matrix-size 3000 --num-runs 3 --verbose
```

### 4. Test with Sample Data

Test GPU acceleration using sample data from previous scans:

```bash
python unlook/examples/test_gpu_acceleration.py
```

Options:
- `--scan-folder FOLDER`: Specify a scan folder to use (default: most recent scan in unlook_debug)
- `--cpu-only`: Force CPU-only processing (for comparison)
- `--verbose`: Enable verbose output

Example with custom scan folder:
```bash
python unlook/examples/test_gpu_acceleration.py --scan-folder unlook/examples/unlook_debug/static_scan_20250511_012924 --verbose
```

### 5. Test with Real Scan

If you have hardware connected, you can test GPU acceleration during a real scan:

```bash
python unlook/examples/static_scanning_example.py --use-gpu --debug
```

This will perform a scan using GPU acceleration if available.

## Troubleshooting

If GPU acceleration is not working properly, try the following:

1. **Check CUDA Environment Variables**:
   ```bash
   echo $CUDA_PATH
   echo $LD_LIBRARY_PATH  # Linux/Mac
   ```

2. **Ensure CuPy is Installed with the Correct CUDA Version**:
   ```bash
   pip show cupy
   ```

3. **Check for Error Messages in the Logs**:
   Enable verbose output with the `--verbose` flag to see detailed error messages.

4. **Try Running the CUDA Setup Script**:
   ```bash
   python setup_cuda_env.py --check-only
   ```

5. **View GPU Status with nvidia-smi**:
   ```bash
   nvidia-smi
   ```

## Common Issues

- **CuPy Installed but GPU Not Available**: This often happens when CuPy can't find the CUDA installation. The integrated setup_cuda_env.py should help resolve this.
  
- **NVIDIA Driver vs CUDA Version Mismatch**: Make sure your NVIDIA driver is compatible with your CUDA version.
  
- **Memory Errors During Large Operations**: Adjust batch sizes for processing or free memory between operations.

- **Slow Performance or No Speedup**: Sometimes memory transfers between CPU and GPU can offset the computational gains. This is normal for small operations, but larger computations should still see significant speedups.

## Performance Expectations

With a decent GPU (e.g., NVIDIA GTX 1660 or better), you should expect:

- Matrix operations: 5-20x speedup
- Triangulation: 2-10x speedup
- Correspondence matching: 3-15x speedup

Actual speedups will vary based on your specific hardware, the size of the data, and the operations being performed.

## Contact

If you're having trouble with GPU acceleration, please reach out to the Unlook SDK support team.