"""
GPU Acceleration Utilities for Unlook SDK

This module provides GPU-accelerated versions of key image processing
and computational functions used in the Unlook SDK 3D scanning pipeline.
"""

import logging
import numpy as np
import time
import cv2
import os
import sys
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import CUDA environment setup utilities
try:
    from ..utils.cuda_setup import setup_cuda_env, is_cuda_available, find_cuda_path
    CUDA_SETUP_AVAILABLE = True
    logger.info("CUDA environment setup utilities available")
except ImportError:
    CUDA_SETUP_AVAILABLE = False
    logger.warning("CUDA environment setup utilities not available. GPU acceleration may be limited.")

# Set up CUDA environment first before importing GPU libraries
if CUDA_SETUP_AVAILABLE:
    logger.info("Setting up CUDA environment before importing GPU libraries")
    try:
        # Find and set CUDA paths
        cuda_path = find_cuda_path()
        if cuda_path:
            logger.info(f"Found CUDA installation at: {cuda_path}")
            # Set environment variables
            os.environ["CUDA_PATH"] = cuda_path
            
            # Set up CUDA environment
            setup_cuda_env()
            logger.info(f"CUDA environment set up with CUDA_PATH={os.environ.get('CUDA_PATH')}")
        else:
            logger.warning("Could not find CUDA installation path")
    except Exception as e:
        logger.error(f"Error setting up CUDA environment: {e}")

# Try to import GPU libraries after setting up CUDA environment
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
    logger.info(f"CuPy is available for GPU acceleration (version: {cp.__version__})")
    
    # Check if CuPy can actually access CUDA devices
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        if num_gpus > 0:
            device = cp.cuda.Device(0)
            device_name = device.attributes["name"].decode("utf-8")
            memory_size = device.attributes["totalGlobalMem"] / (1024**3)  # Convert to GB
            logger.info(f"Found {num_gpus} CUDA device(s). Using: {device_name} with {memory_size:.2f} GB memory")
        else:
            logger.warning("CuPy is installed but no CUDA devices were detected")
            CUPY_AVAILABLE = False
    except Exception as e:
        logger.error(f"Error accessing CUDA devices with CuPy: {e}")
        CUPY_AVAILABLE = False
except ImportError as e:
    CUPY_AVAILABLE = False
    logger.warning(f"CuPy not found ({e}). GPU acceleration disabled. Install with: pip install cupy-cuda11x")

# Try to import OpenCV with CUDA support
OPENCV_CUDA_AVAILABLE = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        OPENCV_CUDA_AVAILABLE = True
        logger.info(f"OpenCV with CUDA support available. CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
    else:
        logger.warning("OpenCV with CUDA support not available. Some operations will run on CPU.")
except Exception as e:
    logger.warning(f"Error checking OpenCV CUDA support: {e}")

class GPUAccelerator:
    """
    Provides GPU-accelerated versions of key 3D scanning operations.
    """
    
    def __init__(self, enable_gpu=True):
        """
        Initialize the GPU accelerator.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration if available
        """
        self.enable_gpu = enable_gpu
        
        # Set up CUDA environment if needed
        if CUDA_SETUP_AVAILABLE and enable_gpu and not CUPY_AVAILABLE:
            logger.info("Attempting to set up CUDA environment again")
            setup_success = setup_cuda_env()
            logger.info(f"CUDA environment setup {'successful' if setup_success else 'failed'}")
        
        # Check GPU availability and set up resources
        self.gpu_available = CUPY_AVAILABLE and self.enable_gpu
        self.device = None
        self.mem_pool = None
        self.pinned_mem_pool = None
        
        if self.gpu_available:
            try:
                # Initialize CUDA device
                self.device = cp.cuda.Device(0)  # Use first GPU
                self.device.use()
                device_name = self.device.attributes['name'].decode("utf-8") if hasattr(self.device.attributes['name'], 'decode') else self.device.attributes['name']
                logger.info(f"Using CUDA device: {device_name}")
                logger.info(f"GPU memory: {self.device.mem_info[1]/1024**3:.2f} GB total")
                
                # Create memory pools for better management
                self.mem_pool = cp.get_default_memory_pool()
                self.pinned_mem_pool = cp.get_default_pinned_memory_pool()
                logger.info("GPU memory pools initialized")
                
                # Warm up the GPU
                self._warm_up_gpu()
            except Exception as e:
                logger.error(f"Failed to initialize GPU: {e}")
                self.gpu_available = False
                self.device = None
    
    def _warm_up_gpu(self):
        """Warm up the GPU with a simple computation to avoid initial delay."""
        try:
            a = cp.ones((1000, 1000), dtype=cp.float32)
            b = cp.ones((1000, 1000), dtype=cp.float32)
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            del a, b, c
            self.mem_pool.free_all_blocks()
            logger.info("GPU warm-up completed")
        except Exception as e:
            logger.warning(f"GPU warm-up failed: {e}")
    
    def to_gpu(self, data):
        """Transfer numpy array to GPU memory."""
        if not self.gpu_available:
            return data
        
        try:
            return cp.asarray(data)
        except Exception as e:
            logger.warning(f"Failed to transfer data to GPU: {e}")
            return data
    
    def to_cpu(self, data):
        """Transfer CuPy array to CPU memory."""
        if not self.gpu_available or not isinstance(data, cp.ndarray):
            return data
        
        try:
            return data.get()
        except Exception as e:
            logger.warning(f"Failed to transfer data to CPU: {e}")
            return data
    
    def free_memory(self):
        """Free unused GPU memory."""
        if self.gpu_available and self.mem_pool is not None:
            try:
                self.mem_pool.free_all_blocks()
                if self.pinned_mem_pool is not None:
                    self.pinned_mem_pool.free_all_blocks()
                logger.debug("Freed unused GPU memory")
            except Exception as e:
                logger.warning(f"Failed to free GPU memory: {e}")

    def triangulate_points_gpu(self, P1, P2, points_left, points_right):
        """
        GPU-accelerated version of triangulation to generate 3D points.
        
        Args:
            P1: 3x4 projection matrix for left camera
            P2: 3x4 projection matrix for right camera
            points_left: 2xN array of points in left image
            points_right: 2xN array of points in right image
            
        Returns:
            Nx3 array of 3D points
        """
        if not self.gpu_available:
            # Fall back to CPU version
            logger.info("Using CPU triangulation (GPU not available)")
            return self._triangulate_points_cpu(P1, P2, points_left, points_right)
        
        # Log detailed GPU status for debugging
        try:
            device_info = f"Device: {self.device.attributes['name'].decode() if hasattr(self.device.attributes['name'], 'decode') else self.device.attributes['name']}, "
            device_info += f"Memory: {self.device.mem_info[0]/1024**3:.2f}GB used / {self.device.mem_info[1]/1024**3:.2f}GB total"
            logger.info(f"GPU status: {device_info}")
        except Exception as e:
            logger.warning(f"Could not get GPU device info: {e}")
        
        try:
            start_time = time.time()
            
            # Ensure inputs are correctly shaped
            if len(points_left.shape) != 2 or points_left.shape[0] != 2:
                logger.warning(f"Reshaping points_left from {points_left.shape}")
                points_left = points_left.reshape(-1, 2).T
            
            if len(points_right.shape) != 2 or points_right.shape[0] != 2:
                logger.warning(f"Reshaping points_right from {points_right.shape}")
                points_right = points_right.reshape(-1, 2).T
            
            # Transfer to GPU
            P1_gpu = cp.asarray(P1, dtype=cp.float32)
            P2_gpu = cp.asarray(P2, dtype=cp.float32)
            points_left_gpu = cp.asarray(points_left, dtype=cp.float32)
            points_right_gpu = cp.asarray(points_right, dtype=cp.float32)
            
            num_points = points_left_gpu.shape[1]
            
            # Create the A matrix for the linear system for each point
            A = cp.zeros((num_points, 4, 4), dtype=cp.float32)
            
            # For each point, we create a system of equations using the projections
            # Extract rows from the projection matrices
            P1_row1 = P1_gpu[0]
            P1_row2 = P1_gpu[1]
            P2_row1 = P2_gpu[0]
            P2_row2 = P2_gpu[1]
            
            # Get x and y coordinates
            x1 = points_left_gpu[0]
            y1 = points_left_gpu[1]
            x2 = points_right_gpu[0]
            y2 = points_right_gpu[1]
            
            # Fill in the A matrix for each point
            # For left camera
            A[:, 0] = P1_row1 - x1.reshape(-1, 1) * P1_gpu[2]
            A[:, 1] = P1_row2 - y1.reshape(-1, 1) * P1_gpu[2]
            
            # For right camera
            A[:, 2] = P2_row1 - x2.reshape(-1, 1) * P2_gpu[2]
            A[:, 3] = P2_row2 - y2.reshape(-1, 1) * P2_gpu[2]
            
            # Solve the system using SVD for each point
            # We'll use batched SVD for better performance
            points_3d = cp.zeros((num_points, 4), dtype=cp.float32)
            
            # Process in batches to avoid GPU memory issues
            batch_size = min(10000, num_points)
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                batch = A[i:end_idx]
                
                # Compute SVD for this batch
                # For each 4x4 matrix, we want the right singular vector
                # corresponding to the smallest singular value
                u, s, vh = cp.linalg.svd(batch)
                
                # The solution is the last right singular vector
                points_3d[i:end_idx] = vh[:, -1]
            
            # Convert to homogeneous coordinates
            points_3d_cpu = self.to_cpu(points_3d)
            points_3d_cpu = points_3d_cpu[:, :3] / points_3d_cpu[:, 3:4]
            
            logger.info(f"GPU triangulation completed in {time.time() - start_time:.2f} seconds for {num_points} points")
            
            # Free GPU memory
            self.free_memory()
            
            return points_3d_cpu
            
        except Exception as e:
            logger.error(f"GPU triangulation failed: {e}")
            logger.info("Falling back to CPU triangulation")
            return self._triangulate_points_cpu(P1, P2, points_left, points_right)
    
    def _triangulate_points_cpu(self, P1, P2, points_left, points_right):
        """CPU implementation of triangulation."""
        start_time = time.time()
        
        # Execute OpenCV triangulation on CPU
        points_4d = cv2.triangulatePoints(P1, P2, points_left, points_right)
        
        # Convert to 3D coordinates
        points_3d = points_4d.T
        points_3d = points_3d[:, :3] / points_3d[:, 3:4]
        
        logger.info(f"CPU triangulation completed in {time.time() - start_time:.2f} seconds")
        
        return points_3d
    
    def find_correspondences_gpu(self, left_coords, right_coords, mask_left, mask_right, 
                                epipolar_tolerance=3.0, min_disparity=5, max_disparity=100, 
                                gray_code_threshold=15):
        """
        GPU-accelerated stereo correspondence search.
        
        Args:
            left_coords: Projector coordinates for left image
            right_coords: Projector coordinates for right image
            mask_left: Mask of valid pixels in left image
            mask_right: Mask of valid pixels in right image
            epipolar_tolerance: Pixel tolerance for epipolar line search
            min_disparity: Minimum valid disparity
            max_disparity: Maximum valid disparity
            gray_code_threshold: Threshold for projector coordinate matching
            
        Returns:
            Tuple of (points_left, points_right) arrays with corresponding points
        """
        if not self.gpu_available:
            logger.info("Using CPU correspondence matching (GPU not available)")
            return None, None  # Signal that GPU version isn't used
        
        try:
            start_time = time.time()
            
            # Transfer data to GPU
            left_coords_gpu = self.to_gpu(left_coords[:, :, 0])
            right_coords_gpu = self.to_gpu(right_coords[:, :, 0])
            mask_left_gpu = self.to_gpu(mask_left)
            mask_right_gpu = self.to_gpu(mask_right)
            
            h, w = mask_left.shape[:2]
            
            # Create indices for all pixels
            y_indices, x_indices = cp.mgrid[0:h, 0:w]
            
            # Filter left pixels using mask
            valid_left = cp.where(mask_left_gpu)
            left_y, left_x = valid_left
            left_proj_coords = left_coords_gpu[left_y, left_x]
            
            # Pre-allocate arrays for matches
            max_matches = len(left_y)
            matches_left_x = cp.zeros(max_matches, dtype=cp.int32)
            matches_left_y = cp.zeros(max_matches, dtype=cp.int32)
            matches_right_x = cp.zeros(max_matches, dtype=cp.int32)
            matches_right_y = cp.zeros(max_matches, dtype=cp.int32)
            
            # Counter for actual matches found
            match_count = 0
            
            # Process in smaller batches to avoid GPU memory pressure
            batch_size = 5000
            for batch_start in range(0, len(left_y), batch_size):
                batch_end = min(batch_start + batch_size, len(left_y))
                
                # Get batch of left points
                batch_left_y = left_y[batch_start:batch_end]
                batch_left_x = left_x[batch_start:batch_end]
                batch_proj_coords = left_proj_coords[batch_start:batch_end]
                
                # For each point, find corresponding point in right image
                for i, (y, x, proj_coord) in enumerate(zip(batch_left_y, batch_left_x, batch_proj_coords)):
                    # Define epipolar search region
                    min_y = max(0, y - int(epipolar_tolerance))
                    max_y = min(h - 1, y + int(epipolar_tolerance))
                    
                    # Search along epipolar line
                    for y_right in range(min_y, max_y + 1):
                        # Search from left edge up to current x position
                        search_range = slice(0, min(x, w - 1))
                        
                        # Get all potential matches based on mask
                        potential_matches = cp.where(mask_right_gpu[y_right, search_range])[0]
                        if len(potential_matches) == 0:
                            continue
                        
                        # Get x coordinates in the right image
                        x_rights = potential_matches
                        
                        # Get projector coordinates for all potential matches
                        right_proj_coords = right_coords_gpu[y_right, x_rights]
                        
                        # Calculate coordinate differences
                        diffs = cp.abs(proj_coord - right_proj_coords)
                        
                        # Find the best match
                        if len(diffs) > 0:
                            min_idx = cp.argmin(diffs)
                            min_diff = diffs[min_idx]
                            best_x = x_rights[min_idx]
                            
                            # Check if the match is good enough
                            if min_diff < gray_code_threshold:
                                # Check disparity
                                disparity = x - best_x
                                if disparity >= min_disparity and disparity <= max_disparity:
                                    # Store match
                                    idx = match_count
                                    matches_left_x[idx] = x
                                    matches_left_y[idx] = y
                                    matches_right_x[idx] = best_x
                                    matches_right_y[idx] = y_right
                                    match_count += 1
                                    
                                    # Avoid buffer overflow
                                    if match_count >= max_matches:
                                        break
                
                # Early exit if buffer is full
                if match_count >= max_matches:
                    break
            
            # Transfer results back to CPU
            matches_left_x_cpu = self.to_cpu(matches_left_x[:match_count])
            matches_left_y_cpu = self.to_cpu(matches_left_y[:match_count])
            matches_right_x_cpu = self.to_cpu(matches_right_x[:match_count])
            matches_right_y_cpu = self.to_cpu(matches_right_y[:match_count])
            
            # Create points arrays in the expected format
            points_left = np.column_stack((matches_left_x_cpu, matches_left_y_cpu))
            points_right = np.column_stack((matches_right_x_cpu, matches_right_y_cpu))
            
            # Reshape to the expected format (Nx1x2)
            points_left = points_left.reshape(-1, 1, 2).astype(np.float32)
            points_right = points_right.reshape(-1, 1, 2).astype(np.float32)
            
            logger.info(f"GPU correspondence matching found {match_count} matches in {time.time() - start_time:.2f} seconds")
            
            # Free GPU memory
            self.free_memory()
            
            return points_left, points_right
            
        except Exception as e:
            logger.error(f"GPU correspondence matching failed: {e}")
            logger.info("Falling back to CPU correspondence matching")
            return None, None  # Signal that GPU version failed
    
    def decode_gray_code_gpu(self, patterns, pattern_width, pattern_height, 
                          mask=None, threshold=15):
        """
        GPU-accelerated Gray code pattern decoding.
        
        Args:
            patterns: List of pattern images (normal/inverted pairs)
            pattern_width: Width of the projector pattern
            pattern_height: Height of the projector pattern
            mask: Optional mask of valid pixels
            threshold: Threshold for bit decoding
            
        Returns:
            Decoded projector coordinates
        """
        if not self.gpu_available:
            logger.info("Using CPU Gray code decoding (GPU not available)")
            return None
        
        try:
            start_time = time.time()
            
            # Get image dimensions
            h, w = patterns[0].shape[:2]
            
            # Transfer patterns to GPU
            patterns_gpu = [self.to_gpu(pat) for pat in patterns]
            
            # Create output array
            coords = cp.zeros((h, w), dtype=cp.uint16)
            
            # If mask is provided, transfer to GPU
            mask_gpu = None
            if mask is not None:
                mask_gpu = self.to_gpu(mask)
            
            # Number of bits (patterns / 2 since we have normal/inverted pairs)
            num_bits = len(patterns_gpu) // 2
            
            # Decode each bit plane
            for bit in range(num_bits):
                # Get normal and inverted patterns
                normal_idx = bit * 2
                inverted_idx = bit * 2 + 1
                
                normal = patterns_gpu[normal_idx]
                inverted = patterns_gpu[inverted_idx]
                
                # Compute difference
                diff = normal.astype(cp.int16) - inverted.astype(cp.int16)
                
                # Apply threshold
                bit_mask = diff > threshold
                
                # Apply mask if provided
                if mask_gpu is not None:
                    bit_mask = bit_mask & mask_gpu
                
                # Set bit in coordinates
                coords = coords | (bit_mask.astype(cp.uint16) << bit)
            
            # Transfer result back to CPU
            coords_cpu = self.to_cpu(coords)
            
            logger.info(f"GPU Gray code decoding completed in {time.time() - start_time:.2f} seconds")
            
            # Free GPU memory
            self.free_memory()
            
            return coords_cpu
            
        except Exception as e:
            logger.error(f"GPU Gray code decoding failed: {e}")
            return None

# Function to check if GPU acceleration is available
def is_gpu_available():
    """Check if GPU acceleration is available."""
    # First check CUDA availability through our utility
    if CUDA_SETUP_AVAILABLE:
        cuda_available = is_cuda_available()
        if cuda_available:
            logger.info("CUDA is available according to CUDA setup utility")
            return True
    
    # Then check CuPy and OpenCV
    if CUPY_AVAILABLE:
        logger.info("GPU acceleration available via CuPy")
        return True
    if OPENCV_CUDA_AVAILABLE:
        logger.info("GPU acceleration available via OpenCV CUDA")
        return True
    
    logger.warning("GPU acceleration not available")
    return False

# Create singleton instance
gpu_accelerator = None

def get_gpu_accelerator(enable_gpu=True):
    """Get or create the GPU accelerator singleton instance."""
    global gpu_accelerator
    if gpu_accelerator is None:
        # Set up CUDA environment before creating accelerator if it's not already set up
        if CUDA_SETUP_AVAILABLE and enable_gpu and not CUPY_AVAILABLE:
            logger.info("Setting up CUDA environment before creating GPU accelerator")
            setup_success = setup_cuda_env()
            logger.info(f"CUDA environment setup {'successful' if setup_success else 'failed'}")
        
        gpu_accelerator = GPUAccelerator(enable_gpu=enable_gpu)
    return gpu_accelerator

# Diagnostic function to check GPU configuration
def diagnose_gpu():
    """Run GPU diagnostics and return detailed status information."""
    diagnostics = {
        "cuda_setup_available": CUDA_SETUP_AVAILABLE,
        "cuda_path": os.environ.get("CUDA_PATH", "Not set"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "Not set"),
        "cupy_available": CUPY_AVAILABLE,
        "opencv_cuda_available": OPENCV_CUDA_AVAILABLE,
        "gpu_device_info": None,
        "cuda_version": None,
        "gpu_memory": None,
    }
    
    # Check if CuPy can access CUDA
    if CUPY_AVAILABLE:
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            diagnostics["cuda_device_count"] = num_gpus
            
            if num_gpus > 0:
                # Get device information
                device = cp.cuda.Device(0)
                device_name = device.attributes["name"].decode("utf-8") if hasattr(device.attributes["name"], "decode") else device.attributes["name"]
                diagnostics["gpu_device_info"] = device_name
                
                # Get memory information
                mem_free, mem_total = device.mem_info
                diagnostics["gpu_memory"] = {
                    "free_gb": mem_free / (1024**3),
                    "total_gb": mem_total / (1024**3),
                    "used_gb": (mem_total - mem_free) / (1024**3)
                }
                
                # Get CUDA version
                diagnostics["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
        except Exception as e:
            diagnostics["gpu_error"] = str(e)
    
    return diagnostics

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize GPU acceleration
    if setup_cuda_env():
        logger.info("CUDA environment set up successfully")
    else:
        logger.warning("Failed to set up CUDA environment")
    
    # Check GPU availability
    gpu_available = is_gpu_available()
    logger.info(f"GPU acceleration available: {gpu_available}")
    
    # Display diagnostics
    diag = diagnose_gpu()
    logger.info(f"GPU diagnostics: {diag}")
    
    # Test GPU accelerator
    if gpu_available:
        accelerator = get_gpu_accelerator()
        logger.info(f"GPU accelerator initialized: {accelerator.gpu_available}")
        
        # Run a simple test
        if accelerator.gpu_available:
            try:
                # Create test data
                test_size = 1000
                points_left = np.random.rand(2, test_size).astype(np.float32)
                points_right = np.random.rand(2, test_size).astype(np.float32)
                
                # Create test projection matrices
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
                
                # Test triangulation
                points_3d = accelerator.triangulate_points_gpu(P1, P2, points_left, points_right)
                logger.info(f"Test triangulation successful, generated {len(points_3d)} 3D points")
            except Exception as e:
                logger.error(f"Test failed: {e}")