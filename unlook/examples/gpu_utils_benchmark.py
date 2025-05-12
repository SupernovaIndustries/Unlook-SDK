#!/usr/bin/env python3
"""
GPU Utilities Benchmark Script for Unlook SDK

This script benchmarks the GPU utilities in the Unlook SDK by running various
GPU operations and measuring their performance.

Usage:
    python gpu_utils_benchmark.py [--matrix-size SIZE] [--num-runs RUNS] [--cpu-only]
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import GPU utilities
from unlook.client.gpu_utils import get_gpu_accelerator, is_gpu_available, diagnose_gpu
from unlook.utils.cuda_setup import setup_cuda_env, is_cuda_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_utils_benchmark")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GPU Utilities Benchmark for Unlook SDK')
    
    parser.add_argument('--matrix-size', type=int, default=2000,
                      help='Size of test matrices (default: 2000)')
    parser.add_argument('--num-runs', type=int, default=5,
                      help='Number of benchmark runs (default: 5)')
    parser.add_argument('--cpu-only', action='store_true',
                      help='Force CPU-only processing (for comparison)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def benchmark_triangulation(gpu_accelerator, matrix_size, num_runs, use_gpu=True):
    """
    Benchmark triangulation performance.
    
    Args:
        gpu_accelerator: GPU accelerator instance
        matrix_size: Size of test matrices
        num_runs: Number of benchmark runs
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking triangulation with matrix size {matrix_size} ({num_runs} runs)")
    
    # Create test data
    num_points = matrix_size
    
    # Create projection matrices
    P1 = np.array([
        [800, 0, 640, 0],
        [0, 800, 360, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    P2 = np.array([
        [800, 0, 640, -80000],
        [0, 800, 360, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    # Create random points
    points_left = np.random.rand(2, num_points).astype(np.float32)
    points_right = np.random.rand(2, num_points).astype(np.float32)
    
    # Benchmark CPU implementation
    cpu_times = []
    for i in range(num_runs):
        start_time = time.time()
        points_3d_cpu = gpu_accelerator._triangulate_points_cpu(P1, P2, points_left, points_right)
        cpu_times.append(time.time() - start_time)
        
        logger.info(f"CPU triangulation run {i+1}/{num_runs}: {cpu_times[-1]:.4f} seconds")
    
    # Benchmark GPU implementation if available
    gpu_times = []
    if use_gpu and gpu_accelerator.gpu_available:
        for i in range(num_runs):
            start_time = time.time()
            points_3d_gpu = gpu_accelerator.triangulate_points_gpu(P1, P2, points_left, points_right)
            gpu_times.append(time.time() - start_time)
            
            logger.info(f"GPU triangulation run {i+1}/{num_runs}: {gpu_times[-1]:.4f} seconds")
    
    # Calculate average times
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else None
    
    # Calculate speedup
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time else None
    
    results = {
        "test": "triangulation",
        "matrix_size": matrix_size,
        "num_points": num_points,
        "avg_cpu_time": avg_cpu_time,
        "avg_gpu_time": avg_gpu_time,
        "speedup": speedup,
        "cpu_times": cpu_times,
        "gpu_times": gpu_times
    }
    
    logger.info(f"Triangulation benchmark results:")
    logger.info(f"  Matrix size: {matrix_size} x {matrix_size}")
    logger.info(f"  Number of points: {num_points}")
    logger.info(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
    
    if avg_gpu_time is not None:
        logger.info(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
    else:
        logger.info("  GPU triangulation not tested")
    
    return results

def benchmark_matrix_operations(gpu_accelerator, matrix_size, num_runs, use_gpu=True):
    """
    Benchmark matrix operations.
    
    Args:
        gpu_accelerator: GPU accelerator instance
        matrix_size: Size of test matrices
        num_runs: Number of benchmark runs
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking matrix operations with size {matrix_size} ({num_runs} runs)")
    
    # Create test matrices
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    
    # Benchmark CPU implementation
    cpu_times = []
    for i in range(num_runs):
        start_time = time.time()
        C_cpu = np.matmul(A, B)
        cpu_times.append(time.time() - start_time)
        
        logger.info(f"CPU matrix multiplication run {i+1}/{num_runs}: {cpu_times[-1]:.4f} seconds")
    
    # Benchmark GPU implementation if available
    gpu_times = []
    if use_gpu and gpu_accelerator.gpu_available:
        for i in range(num_runs):
            start_time = time.time()
            
            # Transfer to GPU
            A_gpu = gpu_accelerator.to_gpu(A)
            B_gpu = gpu_accelerator.to_gpu(B)
            
            # Matrix multiplication
            C_gpu = gpu_accelerator.to_cpu(A_gpu @ B_gpu)
            
            gpu_times.append(time.time() - start_time)
            
            logger.info(f"GPU matrix multiplication run {i+1}/{num_runs}: {gpu_times[-1]:.4f} seconds")
            
            # Free GPU memory
            gpu_accelerator.free_memory()
    
    # Calculate average times
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else None
    
    # Calculate speedup
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time else None
    
    results = {
        "test": "matrix_multiplication",
        "matrix_size": matrix_size,
        "avg_cpu_time": avg_cpu_time,
        "avg_gpu_time": avg_gpu_time,
        "speedup": speedup,
        "cpu_times": cpu_times,
        "gpu_times": gpu_times
    }
    
    logger.info(f"Matrix multiplication benchmark results:")
    logger.info(f"  Matrix size: {matrix_size} x {matrix_size}")
    logger.info(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
    
    if avg_gpu_time is not None:
        logger.info(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
    else:
        logger.info("  GPU matrix multiplication not tested")
    
    return results

def benchmark_memory_transfers(gpu_accelerator, matrix_size, num_runs, use_gpu=True):
    """
    Benchmark memory transfer operations.
    
    Args:
        gpu_accelerator: GPU accelerator instance
        matrix_size: Size of test matrices
        num_runs: Number of benchmark runs
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking memory transfers with size {matrix_size} ({num_runs} runs)")
    
    if not use_gpu or not gpu_accelerator.gpu_available:
        logger.info("Skipping memory transfer benchmark (GPU not available)")
        return None
    
    # Create test matrix
    data_size_mb = matrix_size * matrix_size * 4 / (1024 * 1024)  # Size in MB
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    
    # Benchmark host-to-device transfer
    h2d_times = []
    for i in range(num_runs):
        start_time = time.time()
        A_gpu = gpu_accelerator.to_gpu(A)
        h2d_times.append(time.time() - start_time)
        
        logger.info(f"Host-to-device transfer run {i+1}/{num_runs}: {h2d_times[-1]:.4f} seconds ({data_size_mb:.2f} MB)")
    
    # Benchmark device-to-host transfer
    d2h_times = []
    for i in range(num_runs):
        start_time = time.time()
        A_cpu = gpu_accelerator.to_cpu(A_gpu)
        d2h_times.append(time.time() - start_time)
        
        logger.info(f"Device-to-host transfer run {i+1}/{num_runs}: {d2h_times[-1]:.4f} seconds ({data_size_mb:.2f} MB)")
    
    # Free GPU memory
    gpu_accelerator.free_memory()
    
    # Calculate average times
    avg_h2d_time = sum(h2d_times) / len(h2d_times)
    avg_d2h_time = sum(d2h_times) / len(d2h_times)
    
    # Calculate bandwidth
    h2d_bandwidth = data_size_mb / avg_h2d_time
    d2h_bandwidth = data_size_mb / avg_d2h_time
    
    results = {
        "test": "memory_transfers",
        "matrix_size": matrix_size,
        "data_size_mb": data_size_mb,
        "avg_h2d_time": avg_h2d_time,
        "avg_d2h_time": avg_d2h_time,
        "h2d_bandwidth": h2d_bandwidth,
        "d2h_bandwidth": d2h_bandwidth,
        "h2d_times": h2d_times,
        "d2h_times": d2h_times
    }
    
    logger.info(f"Memory transfer benchmark results:")
    logger.info(f"  Data size: {data_size_mb:.2f} MB")
    logger.info(f"  Average host-to-device time: {avg_h2d_time:.4f} seconds ({h2d_bandwidth:.2f} MB/s)")
    logger.info(f"  Average device-to-host time: {avg_d2h_time:.4f} seconds ({d2h_bandwidth:.2f} MB/s)")
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CUDA environment
    logger.info("Initializing CUDA environment...")
    if setup_cuda_env():
        logger.info("CUDA environment set up successfully")
        
        # Check CUDA availability
        cuda_available = is_cuda_available()
        logger.info(f"CUDA available: {cuda_available}")
    else:
        logger.warning("Failed to set up CUDA environment")
    
    # Check GPU availability
    gpu_available = is_gpu_available()
    if gpu_available:
        logger.info("GPU acceleration is available")
        
        # Get detailed GPU information
        gpu_info = diagnose_gpu()
        logger.info(f"GPU information:")
        for k, v in gpu_info.items():
            logger.info(f"  {k}: {v}")
    else:
        logger.warning("GPU acceleration is not available")
    
    # Override GPU acceleration if requested
    use_gpu = gpu_available and not args.cpu_only
    
    # Initialize GPU accelerator
    gpu_accelerator = get_gpu_accelerator(enable_gpu=use_gpu)
    
    # Run benchmarks
    triangulation_results = benchmark_triangulation(
        gpu_accelerator, args.matrix_size, args.num_runs, use_gpu=use_gpu
    )
    
    matrix_results = benchmark_matrix_operations(
        gpu_accelerator, args.matrix_size, args.num_runs, use_gpu=use_gpu
    )
    
    memory_results = benchmark_memory_transfers(
        gpu_accelerator, args.matrix_size, args.num_runs, use_gpu=use_gpu
    )
    
    # Summary
    logger.info("\nBenchmark Summary:")
    logger.info(f"Matrix size: {args.matrix_size}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    logger.info(f"GPU available: {gpu_available}")
    
    # Triangulation summary
    if triangulation_results:
        logger.info("\nTriangulation:")
        logger.info(f"  Points: {triangulation_results['num_points']}")
        logger.info(f"  CPU time: {triangulation_results['avg_cpu_time']:.4f} seconds")
        if triangulation_results['avg_gpu_time'] is not None:
            logger.info(f"  GPU time: {triangulation_results['avg_gpu_time']:.4f} seconds")
            logger.info(f"  Speedup: {triangulation_results['speedup']:.2f}x")
        else:
            logger.info("  GPU triangulation not tested")
    
    # Matrix operations summary
    if matrix_results:
        logger.info("\nMatrix Multiplication:")
        logger.info(f"  Matrix size: {matrix_results['matrix_size']} x {matrix_results['matrix_size']}")
        logger.info(f"  CPU time: {matrix_results['avg_cpu_time']:.4f} seconds")
        if matrix_results['avg_gpu_time'] is not None:
            logger.info(f"  GPU time: {matrix_results['avg_gpu_time']:.4f} seconds")
            logger.info(f"  Speedup: {matrix_results['speedup']:.2f}x")
        else:
            logger.info("  GPU matrix multiplication not tested")
    
    # Memory transfers summary
    if memory_results:
        logger.info("\nMemory Transfers:")
        logger.info(f"  Data size: {memory_results['data_size_mb']:.2f} MB")
        logger.info(f"  Host-to-device time: {memory_results['avg_h2d_time']:.4f} seconds ({memory_results['h2d_bandwidth']:.2f} MB/s)")
        logger.info(f"  Device-to-host time: {memory_results['avg_d2h_time']:.4f} seconds ({memory_results['d2h_bandwidth']:.2f} MB/s)")
    
    # Overall recommendations
    logger.info("\nRecommendations:")
    
    if not gpu_available:
        logger.info("  - No GPU acceleration available. Consider installing CUDA and CuPy for better performance.")
    elif not use_gpu:
        logger.info("  - GPU acceleration is available but disabled. Enable it for better performance.")
    else:
        # Check if we have valid results to make recommendations
        if (triangulation_results and triangulation_results['speedup'] and 
            matrix_results and matrix_results['speedup']):
            
            # Check if GPU acceleration is providing a significant speedup
            if triangulation_results['speedup'] > 2 or matrix_results['speedup'] > 2:
                logger.info("  - GPU acceleration is working well and providing a significant speedup.")
            else:
                logger.info("  - GPU acceleration is working but not providing a significant speedup.")
                logger.info("    Consider checking for GPU memory bottlenecks or optimizing batch sizes.")

if __name__ == "__main__":
    main()