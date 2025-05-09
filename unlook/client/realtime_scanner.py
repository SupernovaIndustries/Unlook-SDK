"""
Real-time 3D Scanner using GPU acceleration.

This module provides a faster implementation of 3D scanning designed for real-time
handheld operation. It uses GPU acceleration when available and optimized algorithms
for faster pattern generation, capture and processing.
"""

import os
import time
import logging
import numpy as np
import cv2
import json
import threading
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Set up CUDA environment first to help libraries find it
try:
    from ..utils.cuda_setup import setup_cuda_env, is_cuda_available
    # Try to set up CUDA environment
    setup_cuda_env()
    # Full CUDA check
    CUDA_AVAILABLE = is_cuda_available()
except ImportError:
    # Fallback to simple OpenCV check if our utility isn't available
    try:
        # Try to enable OpenCV GPU modules
        cv2.cuda.getCudaEnabledDeviceCount()
        CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    logger.info("CUDA support detected, GPU acceleration enabled")
else:
    logger.warning("CUDA not available, using CPU processing only")

# Try to import optional dependencies
try:
    # Ensure CUDA environment is set up first
    try:
        from ..utils.cuda_setup import setup_cuda_env
        setup_cuda_env()
    except ImportError:
        pass

    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True

    # Check for Open3D CUDA support
    OPEN3D_CUDA = False
    if CUDA_AVAILABLE:
        try:
            # Detect Open3D version
            o3d_version = o3d.__version__
            logger.info(f"Open3D version: {o3d_version}")

            # Different CUDA initialization methods for different Open3D versions
            if hasattr(o3d, 'core') and hasattr(o3d.core, 'initialize_cuda_device'):
                # Modern Open3D (0.13+)
                try:
                    # Try to initialize CUDA
                    o3d.core.initialize_cuda_device()
                    # Verify CUDA was enabled by running a small operation
                    if hasattr(o3d.core, 'Device'):
                        test_tensor = o3d.core.Tensor.ones((3, 3), o3d.core.Dtype.Float32,
                                                           o3d.core.Device("CUDA:0"))
                        OPEN3D_CUDA = True
                        logger.info("Open3D CUDA support enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize Open3D CUDA: {e}")
                    OPEN3D_CUDA = False
            elif hasattr(o3d, 'utility') and hasattr(o3d.utility, 'set_verbosity_level'):
                # Older Open3D versions (pre-0.13) - check if CUDA was compiled in
                try:
                    # No direct way to check CUDA, try to create a GPU device
                    if hasattr(o3d, 'cuda') or 'cuda' in dir(o3d):
                        OPEN3D_CUDA = True
                        logger.info("Open3D appears to have CUDA support (legacy version)")
                    else:
                        logger.warning("Open3D CUDA support not detected (legacy version)")
                except:
                    logger.warning("Failed to check Open3D CUDA support (legacy version)")

            if not OPEN3D_CUDA:
                logger.warning("Open3D CUDA support not available")
        except Exception as e:
            logger.warning(f"Error checking Open3D CUDA support: {e}")

    # Check for Open3D ML
    OPEN3D_ML_AVAILABLE = False
    OPEN3D_ML_TORCH = False
    OPEN3D_ML_TF = False
    try:
        # Check if ml module exists
        if hasattr(o3d, 'ml'):
            # Modern Open3D with ML module
            import importlib

            # Try to import torch backend
            torch_backend = importlib.util.find_spec("open3d.ml.torch")
            if torch_backend:
                OPEN3D_ML_TORCH = True
                logger.info("Open3D-ML with PyTorch backend available")

            # Try to import tensorflow backend
            tf_backend = importlib.util.find_spec("open3d.ml.tf")
            if tf_backend:
                OPEN3D_ML_TF = True
                logger.info("Open3D-ML with TensorFlow backend available")

            OPEN3D_ML_AVAILABLE = OPEN3D_ML_TORCH or OPEN3D_ML_TF

            if not OPEN3D_ML_AVAILABLE:
                logger.warning("Open3D-ML found but no ML backend detected")
    except Exception as e:
        logger.warning(f"Error checking Open3D-ML support: {e}")
except ImportError as e:
    logger.warning(f"open3d not installed ({e}). 3D mesh visualization and processing will be limited.")
    OPEN3D_AVAILABLE = False
    OPEN3D_CUDA = False
    OPEN3D_ML_AVAILABLE = False
    OPEN3D_ML_TORCH = False
    OPEN3D_ML_TF = False

    # Create placeholder for open3d when not available
    class PlaceholderO3D:
        class geometry:
            class PointCloud:
                pass
            class TriangleMesh:
                pass
        class utility:
            class Vector3dVector:
                pass
        class visualization:
            pass
        class io:
            pass
        class core:
            pass
        class ml:
            pass
    o3d = PlaceholderO3D()
    o3dg = PlaceholderO3D.geometry

# Try to import cupy for GPU array processing
try:
    # Ensure CUDA environment is set up properly first
    try:
        from ..utils.cuda_setup import setup_cuda_env
        setup_cuda_env()
    except ImportError:
        pass

    import cupy as cp
    # Verify CuPy actually works by performing a small calculation
    test_array = cp.array([1, 2, 3])
    result = cp.sum(test_array)

    CUPY_AVAILABLE = True
    logger.info("CuPy available, using GPU array processing")
except (ImportError, Exception) as e:
    CUPY_AVAILABLE = False
    logger.warning(f"CuPy not working properly ({str(e)}), using NumPy for array processing")
    cp = np

# Try to import torch first
try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

# Try to import our point cloud neural network module
try:
    from .point_cloud_nn import get_point_cloud_enhancer, TORCH_AVAILABLE, TORCH_CUDA
    POINT_CLOUD_NN_AVAILABLE = True
    if TORCH_AVAILABLE:
        if TORCH_CUDA and TORCH_INSTALLED:
            logger.info(f"PyTorch CUDA support enabled (device: {torch.cuda.get_device_name(0)})")
        else:
            logger.warning("PyTorch CUDA support not available, using CPU processing")
    else:
        logger.warning("PyTorch not installed, neural network features will be limited")
except ImportError:
    logger.warning("Point cloud neural network module not available, enhancement features will be disabled")
    POINT_CLOUD_NN_AVAILABLE = False
    TORCH_AVAILABLE = False
    TORCH_CUDA = False

# Import structured light modules
from .structured_light import Pattern, generate_patterns
from .scanner3d import ScanConfig


class RealTimeScanConfig(ScanConfig):
    """Configuration for real-time 3D scanning."""

    def __init__(self):
        """Initialize with real-time optimized settings."""
        super().__init__()

        # Real-time specific settings
        self.use_gpu = CUDA_AVAILABLE
        self.use_neural_network = POINT_CLOUD_NN_AVAILABLE and (TORCH_AVAILABLE or OPEN3D_AVAILABLE)
        self.streaming = True
        self.continuous_scanning = False
        self.max_fps = 15
        self.moving_average_frames = 5
        self.point_cloud_downsample = True
        self.downsample_voxel_size = 2.0  # Larger voxel size for faster processing
        self.optimized_patterns = True
        self.quality = "fast"  # Default to fast for real-time

        # Reduced pattern set for speed
        self.num_gray_codes = 6  # Fewer bits for speed
        self.num_phase_shifts = 4
        self.phase_shift_frequencies = [16]  # Just one high frequency
        self.pattern_interval = 0.1  # Very short interval for speed

        # Neural network enhancement parameters
        self.nn_denoise_strength = 0.5  # Default denoising strength (0.0 to 1.0)
        self.nn_upsample = False       # Whether to upsample points
        self.nn_target_points = None   # Target number of points after upsampling
    
    def set_quality_preset(self, quality: str):
        """Set parameters based on quality preset."""
        super().set_quality_preset(quality)

        # Adjust real-time parameters based on quality
        if quality == "fast":
            self.num_gray_codes = 5
            self.num_phase_shifts = 3
            self.phase_shift_frequencies = [16]
            self.pattern_interval = 0.05
            self.moving_average_frames = 3
            self.downsample_voxel_size = 3.0
            self.nn_denoise_strength = 0.3
            self.nn_upsample = False
        elif quality == "medium":
            self.num_gray_codes = 6
            self.num_phase_shifts = 4
            self.phase_shift_frequencies = [16]
            self.pattern_interval = 0.1
            self.moving_average_frames = 5
            self.downsample_voxel_size = 2.0
            self.nn_denoise_strength = 0.5
            self.nn_upsample = False
        elif quality == "high":
            self.num_gray_codes = 8
            self.num_phase_shifts = 4
            self.phase_shift_frequencies = [16, 8]
            self.pattern_interval = 0.15
            self.moving_average_frames = 8
            self.downsample_voxel_size = 1.0
            self.nn_denoise_strength = 0.7
            self.nn_upsample = True
            self.nn_target_points = 100000  # Target 100k points
        elif quality == "ultra":
            self.num_gray_codes = 10
            self.num_phase_shifts = 6
            self.phase_shift_frequencies = [16, 8, 4]
            self.pattern_interval = 0.2
            self.moving_average_frames = 10
            self.downsample_voxel_size = 0.5
            self.nn_denoise_strength = 0.8
            self.nn_upsample = True
            self.nn_target_points = 200000  # Target 200k points


class OptimizedPatternSet:
    """
    Generates optimized pattern sets for real-time 3D scanning.
    
    This class creates minimal sets of patterns that can be projected
    and processed quickly for real-time 3D reconstruction.
    """
    
    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        num_gray_codes: int = 6,
        use_gpu: bool = False
    ):
        """
        Initialize optimized pattern generator.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
            num_gray_codes: Number of Gray code bits
            use_gpu: Whether to use GPU acceleration
        """
        self.width = width
        self.height = height
        self.num_gray_codes = num_gray_codes
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        
        # Generate patterns
        self.patterns = self._generate_patterns()
        logger.info(f"Generated {len(self.patterns)} optimized patterns")
    
    def _generate_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate optimized pattern set.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Add white and black reference patterns (always needed)
        patterns.append({"pattern_type": "solid_field", "color": "White", "name": "white_reference"})
        patterns.append({"pattern_type": "solid_field", "color": "Black", "name": "black_reference"})
        
        # Add Gray code patterns (vertical only for real-time speed)
        # We only use vertical patterns since they encode horizontal disparity,
        # which is what we need for stereo reconstruction
        for bit in range(self.num_gray_codes):
            stripe_width = max(1, 2 ** bit)
            
            # Normal pattern
            patterns.append({
                "pattern_type": "vertical_lines",
                "foreground_color": "White",
                "background_color": "Black",
                "foreground_width": stripe_width,
                "background_width": stripe_width,
                "name": f"gray_code_v_bit{bit:02d}"
            })
            
            # Inverted pattern
            patterns.append({
                "pattern_type": "vertical_lines",
                "foreground_color": "Black",
                "background_color": "White",
                "foreground_width": stripe_width,
                "background_width": stripe_width,
                "name": f"gray_code_v_bit{bit:02d}_inv"
            })
        
        return patterns
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get the optimized pattern set."""
        return self.patterns


class RealTimeScanner:
    """
    Real-time 3D scanner with GPU acceleration.
    
    This scanner is optimized for speed and can operate continuously
    in a streaming mode, making it suitable for handheld scanning.
    """
    
    def __init__(
        self,
        client,
        config: Optional[RealTimeScanConfig] = None,
        calibration_file: Optional[str] = None,
        on_new_frame: Optional[Callable] = None
    ):
        """
        Initialize real-time scanner.

        Args:
            client: UnlookClient instance
            config: Scanner configuration
            calibration_file: Path to stereo calibration file
            on_new_frame: Callback when new scan data is available
        """
        self.client = client
        self.config = config or RealTimeScanConfig()
        self.calibration_file = calibration_file
        self.on_new_frame = on_new_frame

        # State variables
        self.running = False
        self.scanning_thread = None
        self.scan_count = 0
        self.last_scan_time = 0
        self.fps = 0
        self.current_point_cloud = None
        self.point_cloud_history = []
        self.debug_mode = True  # Enable detailed logging by default for troubleshooting

        # Create debug directory
        if self.debug_mode:
            import os
            self.debug_dir = os.path.join(os.getcwd(), "debug_output")
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug mode enabled, outputs will be saved to: {self.debug_dir}")

        # Initialize point cloud enhancer to None
        self.point_cloud_enhancer = None

        # Load calibration data
        self.calibration_data = self._load_calibration(calibration_file)

        # Initialize pattern set
        self.pattern_set = OptimizedPatternSet(
            width=self.config.pattern_resolution[0],
            height=self.config.pattern_resolution[1],
            num_gray_codes=self.config.num_gray_codes,
            use_gpu=self.config.use_gpu
        )

        # Initialize neural network if available
        if self.config.use_neural_network and TORCH_AVAILABLE and TORCH_CUDA:
            self._init_neural_network()

        logger.info(f"Initialized RealTimeScanner with {self.config.quality} quality preset")
    
    def _check_open3d_cuda(self):
        """
        Check if Open3D has proper CUDA support.
        Only run in debug mode to diagnose issues.
        """
        try:
            import open3d as o3d
            logger.info(f"Open3D version: {o3d.__version__}")

            # Check for CUDA support
            if hasattr(o3d, 'core') and hasattr(o3d.core, 'initialize_cuda_device'):
                try:
                    o3d.core.initialize_cuda_device()
                    logger.info("Open3D CUDA initialized successfully")
                    return True
                except Exception as e:
                    logger.warning(f"Open3D CUDA initialization failed: {e}")
            else:
                # Legacy Open3D
                if hasattr(o3d, 'cuda') or 'cuda' in dir(o3d):
                    logger.info("Open3D has legacy CUDA support")
                    return True
                else:
                    logger.warning("Open3D CUDA support not detected (legacy version)")

            return False
        except Exception as e:
            logger.warning(f"Error checking Open3D CUDA support: {e}")
            return False

    def _check_open3d_ml(self):
        """
        Check if Open3D-ML is available.
        Only run in debug mode to diagnose issues.
        """
        try:
            import open3d as o3d

            if hasattr(o3d, 'ml'):
                logger.info("Open3D-ML module found")

                # Check PyTorch backend
                try:
                    import importlib
                    torch_spec = importlib.util.find_spec("open3d.ml.torch")
                    if torch_spec:
                        logger.info("Open3D-ML with PyTorch backend available")
                        return True
                except Exception:
                    pass

                # Check TensorFlow backend
                try:
                    tf_spec = importlib.util.find_spec("open3d.ml.tf")
                    if tf_spec:
                        logger.info("Open3D-ML with TensorFlow backend available")
                        return True
                except Exception:
                    pass

                # Check if any ML backend is available
                if not ('torch_spec' in locals() and torch_spec) and not ('tf_spec' in locals() and tf_spec):
                    logger.warning("Open3D-ML found but no ML backend detected")
            else:
                logger.warning("Open3D-ML module not found")

        except Exception as e:
            logger.warning(f"Error checking Open3D-ML: {e}")

        return False

    def _init_neural_network(self):
        """Initialize neural network for enhanced 3D reconstruction."""
        if not POINT_CLOUD_NN_AVAILABLE:
            logger.warning("Point cloud neural network not available, enhancement disabled")
            self.config.use_neural_network = False
            return

        # Run diagnostic checks in debug mode
        if self.debug_mode:
            has_cuda = self._check_open3d_cuda()
            has_ml = self._check_open3d_ml()
            if not has_cuda:
                logger.warning("Open3D CUDA support not available")

        try:
            # Create a point cloud enhancer
            self.point_cloud_enhancer = get_point_cloud_enhancer(
                use_gpu=self.config.use_gpu and TORCH_CUDA,
                model_dir=os.path.join(os.path.expanduser("~"), ".unlook", "models")
            )

            logger.info("Neural network initialized for point cloud enhancement")
        except Exception as e:
            logger.error(f"Failed to initialize neural network: {e}")
            self.config.use_neural_network = False
    
    def _load_calibration(self, calibration_file: Optional[str]) -> Dict[str, Any]:
        """
        Load stereo calibration data from file.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            Calibration data dictionary
        """
        if not calibration_file:
            logger.info("No calibration file provided, using default calibration")
            return self._create_default_calibration()
        
        if not os.path.exists(calibration_file):
            logger.warning(f"Calibration file not found: {calibration_file}")
            logger.info("Using default calibration instead")
            return self._create_default_calibration()
        
        try:
            # Try JSON format first
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
                
                # Check required parameters
                required_keys = ["camera_matrix_left", "dist_coeffs_left",
                                "camera_matrix_right", "dist_coeffs_right",
                                "R", "T"]
                
                if all(key in calib_data for key in required_keys):
                    logger.info(f"Loaded calibration from {calibration_file}")
                    return calib_data
                else:
                    logger.warning("Calibration file is missing required parameters")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON calibration file: {calibration_file}")
        
        try:
            # Try numpy format
            calib_data = dict(np.load(calibration_file, allow_pickle=True))
            
            # Check required parameters (with alternate names)
            if all(key in calib_data for key in ["M1", "d1", "M2", "d2", "R", "T"]):
                # Rename keys to standardized format
                return {
                    "camera_matrix_left": calib_data["M1"],
                    "dist_coeffs_left": calib_data["d1"],
                    "camera_matrix_right": calib_data["M2"],
                    "dist_coeffs_right": calib_data["d2"],
                    "R": calib_data["R"],
                    "T": calib_data["T"]
                }
            else:
                logger.warning("NumPy calibration file is missing required parameters")
        except Exception as e:
            logger.warning(f"Failed to load NumPy calibration file: {e}")
        
        logger.warning("Using default calibration as fallback")
        return self._create_default_calibration()
    
    def _create_default_calibration(self) -> Dict[str, Any]:
        """Create default stereo calibration parameters."""
        # Use projector resolution for image size
        width, height = self.config.pattern_resolution
        
        # Standard intrinsic camera matrix
        fx = width * 0.8
        fy = height * 0.8
        cx = width / 2
        cy = height / 2
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # No distortion by default
        dist_coeffs = np.zeros(5)
        
        # Identity rotation between cameras
        R = np.eye(3)
        
        # 100mm baseline along X
        T = np.array([100.0, 0.0, 0.0]).reshape(3, 1)
        
        return {
            "camera_matrix_left": camera_matrix,
            "dist_coeffs_left": dist_coeffs,
            "camera_matrix_right": camera_matrix.copy(),
            "dist_coeffs_right": dist_coeffs.copy(),
            "R": R,
            "T": T,
            "image_size": (width, height)
        }
    
    def start(self, debug_mode: bool = False):
        """
        Start real-time scanning.

        Args:
            debug_mode: Whether to enable additional debugging information

        Returns:
            True if scanning was started, False otherwise
        """
        if self.running:
            logger.warning("Real-time scanner already running")
            return False

        # Set debug mode
        self.debug_mode = debug_mode
        if debug_mode:
            logger.info("Debug mode enabled - additional logging will be shown")
            # Create debug directory if it doesn't exist
            import os
            self.debug_dir = os.path.join(os.getcwd(), "debug_output")
            os.makedirs(self.debug_dir, exist_ok=True)

        self.running = True

        # Start scanning thread
        self.scanning_thread = threading.Thread(target=self._scanning_loop, daemon=True)
        self.scanning_thread.start()

        logger.info("Real-time scanning started")
        return True

    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostics to troubleshoot scanner issues.

        Returns:
            Dictionary with diagnostic results
        """
        results = {}

        # Check GPU and CUDA availability
        results["gpu"] = {
            "torch_cuda": TORCH_CUDA,
            "gpu_available": CUDA_AVAILABLE,
            "opencv_cuda_enabled": getattr(self.config, 'opencv_cuda_enabled', True),
        }

        # Check OpenCV CUDA support
        try:
            # Try to create a small test GpuMat to verify OpenCV CUDA bindings
            test_array = np.zeros((10, 10), dtype=np.float32)
            test_gpu_mat = cv2.cuda.GpuMat()
            test_gpu_mat.upload(test_array)
            test_gpu_mat.release()
            results["gpu"]["opencv_cuda"] = True
        except Exception as e:
            results["gpu"]["opencv_cuda"] = False
            results["gpu"]["opencv_cuda_error"] = str(e)

        # Check Open3D support if debug mode
        if self.debug_mode:
            has_open3d_cuda = self._check_open3d_cuda()
            has_open3d_ml = self._check_open3d_ml()
            results["gpu"]["open3d_cuda"] = has_open3d_cuda
            results["gpu"]["open3d_ml"] = has_open3d_ml

        # Check cameras and focus
        try:
            logger.info("Checking cameras...")
            cameras = self.client.camera.get_cameras()
            results["cameras"] = {
                "count": len(cameras),
                "ids": [cam["id"] for cam in cameras]
            }

            if len(cameras) < 2:
                results["camera_status"] = "FAILED - Need at least 2 cameras"
            else:
                # Check stereo focus
                logger.info("Checking camera focus...")
                focus_results, _ = self.check_stereo_focus()
                results["focus"] = focus_results

                # Determine overall focus status
                if all(quality == "excellent" for _, (_, quality) in focus_results.items()):
                    results["focus_status"] = "EXCELLENT"
                elif all(quality in ["good", "excellent"] for _, (_, quality) in focus_results.items()):
                    results["focus_status"] = "GOOD"
                elif any(quality == "poor" for _, (_, quality) in focus_results.items()):
                    results["focus_status"] = "POOR - Cameras need focus adjustment"
                else:
                    results["focus_status"] = "MODERATE - Focus could be improved"

        except Exception as e:
            logger.error(f"Error checking cameras: {e}")
            results["camera_status"] = f"ERROR - {str(e)}"

        # Check projector
        try:
            logger.info("Testing projector...")
            projector_test = self.client.projector.show_solid_field("White")
            time.sleep(0.5)
            self.client.projector.show_solid_field("Black")

            results["projector"] = {
                "test_result": "OK" if projector_test else "FAILED"
            }
        except Exception as e:
            logger.error(f"Error testing projector: {e}")
            results["projector"] = {
                "test_result": f"ERROR - {str(e)}"
            }

        # Check calibration
        results["calibration"] = {
            "file": self.calibration_file,
            "loaded": self.calibration_data is not None
        }

        if self.calibration_data:
            # Check basic calibration parameters
            try:
                camera_matrix_left = np.array(self.calibration_data["camera_matrix_left"])
                camera_matrix_right = np.array(self.calibration_data["camera_matrix_right"])
                R = np.array(self.calibration_data["R"])
                T = np.array(self.calibration_data["T"])

                # Basic sanity checks
                results["calibration"]["valid_camera_matrices"] = (
                    camera_matrix_left.shape == (3, 3) and
                    camera_matrix_right.shape == (3, 3))

                results["calibration"]["valid_rotation"] = (R.shape == (3, 3))
                results["calibration"]["valid_translation"] = (T.size == 3)

                # Check focal lengths
                results["calibration"]["focal_length_left"] = camera_matrix_left[0, 0]
                results["calibration"]["focal_length_right"] = camera_matrix_right[0, 0]

                # Check baseline (distance between cameras)
                baseline = np.linalg.norm(T)
                results["calibration"]["baseline"] = baseline

                if baseline < 10:
                    results["calibration"]["baseline_status"] = "WARNING - Baseline is very small"
                elif baseline > 500:
                    results["calibration"]["baseline_status"] = "WARNING - Baseline is very large"
                else:
                    results["calibration"]["baseline_status"] = "OK"

            except Exception as e:
                logger.error(f"Error checking calibration data: {e}")
                results["calibration"]["valid"] = False

        logger.info("Diagnostics completed")
        return results
    
    def stop(self):
        """Stop real-time scanning."""
        if not self.running:
            logger.warning("Real-time scanner not running")
            return False
        
        self.running = False
        
        # Wait for thread to terminate
        if self.scanning_thread and self.scanning_thread.is_alive():
            self.scanning_thread.join(timeout=2.0)
        
        self.scanning_thread = None
        
        logger.info("Real-time scanning stopped")
        return True
    
    def _scanning_loop(self):
        """Main scanning loop."""
        # Precompute rectification parameters
        rectification_params = self._compute_rectification()
        
        # Get pattern set
        patterns = self.pattern_set.get_patterns()
        
        # Performance tracking
        frame_times = []
        start_time = time.time()
        
        # Main loop
        while self.running:
            scan_start = time.time()
            
            try:
                # Project patterns and capture images
                left_images, right_images = self._project_and_capture(patterns)
                
                # Process images
                if left_images and right_images:
                    point_cloud = self._process_frame(left_images, right_images, rectification_params)
                    
                    # Update state
                    self.current_point_cloud = point_cloud
                    self.scan_count += 1
                    
                    # Calculate FPS
                    scan_duration = time.time() - scan_start
                    frame_times.append(scan_duration)
                    # Keep only recent times for FPS calculation
                    if len(frame_times) > 10:
                        frame_times.pop(0)
                    
                    # Average FPS over recent frames
                    if frame_times:
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        self.fps = 1.0 / max(avg_frame_time, 0.001)
                    
                    # Call callback if provided
                    if self.on_new_frame and callable(self.on_new_frame):
                        self.on_new_frame(point_cloud, self.scan_count, self.fps)
                    
                    # Add to history for moving average if enabled
                    if self.config.moving_average_frames > 1:
                        self.point_cloud_history.append(point_cloud)
                        # Keep only recent point clouds
                        while len(self.point_cloud_history) > self.config.moving_average_frames:
                            self.point_cloud_history.pop(0)
                
                # Adjust sleep time to maintain target FPS
                elapsed = time.time() - scan_start
                target_frame_time = 1.0 / self.config.max_fps
                sleep_time = max(0, target_frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Brief pause on error to avoid hammering
                time.sleep(0.5)
        
        # Cleanup
        logger.info("Scanning loop ended")
    
    def _compute_rectification(self) -> Dict[str, np.ndarray]:
        """Compute stereo rectification parameters."""
        # Extract calibration parameters
        camera_matrix_left = np.array(self.calibration_data["camera_matrix_left"])
        dist_coeffs_left = np.array(self.calibration_data["dist_coeffs_left"])
        camera_matrix_right = np.array(self.calibration_data["camera_matrix_right"])
        dist_coeffs_right = np.array(self.calibration_data["dist_coeffs_right"])
        R = np.array(self.calibration_data["R"])
        T = np.array(self.calibration_data["T"])
        image_size = self.calibration_data.get("image_size", (1280, 720))
        
        # Compute stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Compute undistortion maps
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_32FC1
        )
        
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_32FC1
        )
        
        # Transfer to GPU if available for faster remapping
        # Skip if opencv_cuda_enabled is explicitly set to False
        opencv_cuda_enabled = getattr(self.config, 'opencv_cuda_enabled', True)
        if self.config.use_gpu and CUDA_AVAILABLE and opencv_cuda_enabled:
            try:
                # Check if OpenCV was actually compiled with CUDA support
                try:
                    # Try to create a small test GpuMat to verify OpenCV CUDA bindings
                    test_array = np.zeros((10, 10), dtype=np.float32)
                    test_gpu_mat = cv2.cuda.GpuMat()
                    test_gpu_mat.upload(test_array)
                    test_gpu_mat.release()
                except Exception as e:
                    logger.warning(f"OpenCV CUDA not properly available: {e}")
                    raise ValueError("OpenCV CUDA support not available")

                # Convert to CUDA arrays using safer upload method
                map_left_x_gpu = cv2.cuda.GpuMat()
                map_left_y_gpu = cv2.cuda.GpuMat()
                map_right_x_gpu = cv2.cuda.GpuMat()
                map_right_y_gpu = cv2.cuda.GpuMat()

                # Ensure float32 type for GPU
                map_left_x_gpu.upload(map_left_x.astype(np.float32))
                map_left_y_gpu.upload(map_left_y.astype(np.float32))
                map_right_x_gpu.upload(map_right_x.astype(np.float32))
                map_right_y_gpu.upload(map_right_y.astype(np.float32))

                # Return both CPU and GPU maps
                return {
                    "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
                    "roi1": roi1, "roi2": roi2,
                    "map_left_x": map_left_x, "map_left_y": map_left_y,
                    "map_right_x": map_right_x, "map_right_y": map_right_y,
                    "map_left_x_gpu": map_left_x_gpu, "map_left_y_gpu": map_left_y_gpu,
                    "map_right_x_gpu": map_right_x_gpu, "map_right_y_gpu": map_right_y_gpu,
                    "use_gpu": True
                }
            except Exception as e:
                logger.error(f"Failed to create GPU rectification maps: {e}")
                if self.debug_mode:
                    import traceback
                    logger.debug(f"GPU rectification error traceback: {traceback.format_exc()}")
                # Set flag to avoid GPU usage throughout processing
                self.config.use_gpu = False
                logger.warning("Disabled GPU acceleration due to OpenCV CUDA issues")
        
        # Return CPU maps
        return {
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "roi1": roi1, "roi2": roi2,
            "map_left_x": map_left_x, "map_left_y": map_left_y,
            "map_right_x": map_right_x, "map_right_y": map_right_y,
            "use_gpu": False
        }
    
    def _project_and_capture(self, patterns: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Project patterns and capture images.

        Args:
            patterns: List of pattern dictionaries

        Returns:
            Tuple of (left_images, right_images)
        """
        # Get camera IDs
        cameras = self.client.camera.get_cameras()
        if not cameras or len(cameras) < 2:
            logger.error("Need at least 2 cameras for stereo scanning")
            return [], []

        left_camera_id = cameras[0]["id"]
        right_camera_id = cameras[1]["id"]

        # Log camera info if in debug mode
        if self.debug_mode:
            logger.info(f"Using cameras: Left={left_camera_id}, Right={right_camera_id}")
            logger.info(f"Camera info: {cameras}")

        # Prepare for capture
        left_images = []
        right_images = []

        # Project patterns and capture images
        for i, pattern in enumerate(patterns):
            pattern_name = pattern.get("name", f"pattern_{i}")

            # Use the appropriate projector method based on pattern type
            pattern_type = pattern.get("pattern_type", "")

            success = False
            if pattern_type == "solid_field":
                # Use direct solid field method
                success = self.client.projector.show_solid_field(pattern.get("color", "White"))
            elif pattern_type == "horizontal_lines":
                # Use direct horizontal lines method
                success = self.client.projector.show_horizontal_lines(
                    foreground_color=pattern.get("foreground_color", "White"),
                    background_color=pattern.get("background_color", "Black"),
                    foreground_width=pattern.get("foreground_width", 4),
                    background_width=pattern.get("background_width", 20)
                )
            elif pattern_type == "vertical_lines":
                # Use direct vertical lines method
                success = self.client.projector.show_vertical_lines(
                    foreground_color=pattern.get("foreground_color", "White"),
                    background_color=pattern.get("background_color", "Black"),
                    foreground_width=pattern.get("foreground_width", 4),
                    background_width=pattern.get("background_width", 20)
                )

            if not success:
                logger.warning(f"Failed to project pattern {pattern_name}")
                continue

            # Very short delay for projector to update
            time.sleep(self.config.pattern_interval)

            # Capture stereo pair
            try:
                left_img = self.client.camera.capture(left_camera_id)
                right_img = self.client.camera.capture(right_camera_id)

                # Debug output - save pattern images if enabled
                if self.debug_mode and hasattr(self, 'debug_dir') and i < 3:  # Save first few patterns only
                    try:
                        import os
                        pattern_dir = os.path.join(self.debug_dir, "patterns")
                        os.makedirs(pattern_dir, exist_ok=True)

                        # Save the captured images for debugging
                        if left_img is not None:
                            cv2.imwrite(os.path.join(pattern_dir, f"pattern_{i}_left.png"), left_img)
                        if right_img is not None:
                            cv2.imwrite(os.path.join(pattern_dir, f"pattern_{i}_right.png"), right_img)

                        logger.info(f"Saved pattern {i} images to {pattern_dir}")
                    except Exception as e:
                        logger.error(f"Error saving debug images: {e}")

                # Log image information in debug mode
                if self.debug_mode:
                    logger.info(f"Pattern {pattern_name}: Left shape={left_img.shape if left_img is not None else 'None'}, "
                               f"Right shape={right_img.shape if right_img is not None else 'None'}")

                # Append to image lists
                left_images.append(left_img)
                right_images.append(right_img)

            except Exception as e:
                logger.error(f"Error capturing images: {e}")
                continue

        # Reset projector to black field
        self.client.projector.show_solid_field("Black")

        return left_images, right_images
    
    def _process_frame(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        rectification_params: Dict[str, np.ndarray]
    ) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Process a frame to generate a point cloud.
        
        Args:
            left_images: Left camera images
            right_images: Right camera images
            rectification_params: Rectification parameters
            
        Returns:
            Point cloud
        """
        # Rectify images (using GPU if available)
        rect_left, rect_right = self._rectify_images(left_images, right_images, rectification_params)
        
        # Get white and black reference images
        white_left = rect_left[0]  # First image is white
        black_left = rect_left[1]  # Second image is black
        white_right = rect_right[0]
        black_right = rect_right[1]
        
        # Gray code images start from index 2
        gray_left = rect_left[2:]
        gray_right = rect_right[2:]
        
        # Decode Gray code patterns
        left_coords, left_mask = self._decode_gray_code_fast(
            gray_left, white_left, black_left,
            self.config.pattern_resolution[0], self.config.pattern_resolution[1]
        )
        
        right_coords, right_mask = self._decode_gray_code_fast(
            gray_right, white_right, black_right,
            self.config.pattern_resolution[0], self.config.pattern_resolution[1]
        )
        
        # Find correspondences
        left_points, right_points = self._find_stereo_correspondences(
            left_coords, right_coords, left_mask, right_mask
        )
        
        # Triangulate points
        points_3d = self._triangulate_points(left_points, right_points, rectification_params)
        
        # Filter point cloud
        point_cloud = self._filter_point_cloud(points_3d)
        
        # Apply neural network enhancement if enabled
        if self.config.use_neural_network and hasattr(self, 'point_cloud_enhancer') and TORCH_AVAILABLE and TORCH_CUDA:
            point_cloud = self._enhance_point_cloud(point_cloud)
        
        # Combine with previous frames if using moving average
        if self.config.moving_average_frames > 1 and self.point_cloud_history:
            point_cloud = self._combine_point_clouds([point_cloud] + self.point_cloud_history)
        
        return point_cloud
    
    def _rectify_images(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        rectification_params: Dict[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Rectify stereo image pairs with GPU acceleration if available.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            rectification_params: Rectification parameters
            
        Returns:
            Tuple of (rectified_left_images, rectified_right_images)
        """
        rect_left = []
        rect_right = []
        
        # Check if GPU maps are available
        use_gpu = rectification_params.get("use_gpu", False)
        
        if use_gpu:
            # GPU rectification
            for left_img, right_img in zip(left_images, right_images):
                # Convert images to GPU
                left_gpu = cv2.cuda_GpuMat(left_img)
                right_gpu = cv2.cuda_GpuMat(right_img)
                
                # Rectify using GPU
                left_rect_gpu = cv2.cuda.remap(
                    left_gpu,
                    rectification_params["map_left_x_gpu"],
                    rectification_params["map_left_y_gpu"],
                    cv2.INTER_LINEAR
                )
                
                right_rect_gpu = cv2.cuda.remap(
                    right_gpu,
                    rectification_params["map_right_x_gpu"],
                    rectification_params["map_right_y_gpu"],
                    cv2.INTER_LINEAR
                )
                
                # Download from GPU
                left_rect = left_rect_gpu.download()
                right_rect = right_rect_gpu.download()
                
                rect_left.append(left_rect)
                rect_right.append(right_rect)
        else:
            # CPU rectification
            map_left_x = rectification_params["map_left_x"]
            map_left_y = rectification_params["map_left_y"]
            map_right_x = rectification_params["map_right_x"]
            map_right_y = rectification_params["map_right_y"]
            
            for left_img, right_img in zip(left_images, right_images):
                # Rectify left image
                left_rect = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
                # Rectify right image
                right_rect = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
                
                rect_left.append(left_rect)
                rect_right.append(right_rect)
        
        return rect_left, rect_right
    
    def _decode_gray_code_fast(
        self,
        gray_images: List[np.ndarray],
        white_ref: np.ndarray,
        black_ref: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast Gray code decoding optimized for real-time stereo performance.

        Incorporates techniques from SLStudio but adapted for stereo camera configuration.

        Args:
            gray_images: Gray code images
            white_ref: White reference image
            black_ref: Black reference image
            width: Pattern width
            height: Pattern height

        Returns:
            Tuple of (decoded_coords, mask)
        """
        # Debug - check if input images exist
        if self.debug_mode:
            if white_ref is None or black_ref is None:
                logger.error("Reference images (white/black) are None!")
                return np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.uint8)

            logger.info(f"White ref shape: {white_ref.shape}, Black ref shape: {black_ref.shape}")
            logger.info(f"Gray images count: {len(gray_images)}")

            # Save reference images for debugging
            if hasattr(self, 'debug_dir'):
                import os
                ref_dir = os.path.join(self.debug_dir, "references")
                os.makedirs(ref_dir, exist_ok=True)

                # Save reference images
                cv2.imwrite(os.path.join(ref_dir, "white_ref.png"), white_ref)
                cv2.imwrite(os.path.join(ref_dir, "black_ref.png"), black_ref)

        # Compute shadow mask with better handling
        try:
            # Convert to grayscale first if color (SLStudio technique)
            if len(white_ref.shape) == 3:
                white_gray = cv2.cvtColor(white_ref, cv2.COLOR_BGR2GRAY)
            else:
                white_gray = white_ref

            if len(black_ref.shape) == 3:
                black_gray = cv2.cvtColor(black_ref, cv2.COLOR_BGR2GRAY)
            else:
                black_gray = black_ref

            # Improved difference calculation (from SLStudio)
            # This creates a higher contrast difference image
            diff = white_gray - black_gray

            # Use shading threshold from SLStudio (20)
            _, mask = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)

            # Remove noise using morphological operations (SLStudio approach)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Save mask for debugging
            if self.debug_mode and hasattr(self, 'debug_dir'):
                import os
                mask_dir = os.path.join(self.debug_dir, "masks")
                os.makedirs(mask_dir, exist_ok=True)

                # Save raw difference image
                cv2.imwrite(os.path.join(mask_dir, "diff_raw.png"), diff)

                # Save mask with enhanced visibility for viewing
                mask_vis = mask.copy() * 255
                cv2.imwrite(os.path.join(mask_dir, "mask.png"), mask_vis)

                # Log mask statistics
                non_zero = np.count_nonzero(mask)
                total = mask.shape[0] * mask.shape[1]
                logger.info(f"Mask: {non_zero}/{total} pixels ({non_zero/total*100:.2f}% coverage)")
        except Exception as e:
            logger.error(f"Error computing mask: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.debug_mode:
                logger.error(f"White ref: {white_ref.shape if white_ref is not None else 'None'}, "
                          f"Black ref: {black_ref.shape if black_ref is not None else 'None'}")
            # Return empty mask as fallback
            mask = np.zeros((1, 1), dtype=np.uint8)

        # Convert images to grayscale if needed
        gray_processed = []
        for i, img in enumerate(gray_images):
            if img is None:
                logger.warning(f"Gray code image {i} is None!")
                # Add placeholder empty image
                img = np.zeros_like(white_ref if white_ref is not None else np.zeros((100, 100)))

            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_processed.append(gray_img)
            else:
                gray_processed.append(img)

            # Save gray code images for debugging
            if self.debug_mode and hasattr(self, 'debug_dir') and i < 4: # Save first few only
                import os
                gc_dir = os.path.join(self.debug_dir, "gray_codes")
                os.makedirs(gc_dir, exist_ok=True)
                cv2.imwrite(os.path.join(gc_dir, f"gray_code_{i}.png"), img)

        # First, determine number of patterns (must be even with half for normal and half for inverted)
        num_patterns = len(gray_processed)
        if num_patterns % 2 != 0:
            logger.warning(f"Odd number of Gray code patterns: {num_patterns}, expected even number")
            # Truncate to even number if needed
            gray_processed = gray_processed[:num_patterns-1]
            num_patterns = len(gray_processed)

        # Number of bits is half the patterns (each bit has normal + inverted pattern)
        num_bits = num_patterns // 2

        # Create binary representation
        binary_codes = np.zeros((mask.shape[0], mask.shape[1], num_bits), dtype=np.uint8)

        # GPU acceleration if available
        # Skip if opencv_cuda_enabled is explicitly set to False
        opencv_cuda_enabled = getattr(self.config, 'opencv_cuda_enabled', True)
        if self.config.use_gpu and CUDA_AVAILABLE and opencv_cuda_enabled:
            try:
                # Verify the images have proper format and size first
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)

                # Check if any of the grayscale images are None or empty
                valid_images = True
                for i in range(num_bits * 2):
                    if i >= len(gray_processed) or gray_processed[i] is None or gray_processed[i].size == 0:
                        valid_images = False
                        break

                if not valid_images:
                    raise ValueError("One or more Gray code images are invalid")

                # Check if OpenCV was compiled with CUDA support
                try:
                    # Create GPU mat and upload mask - safer than direct constructor
                    mask_gpu = cv2.cuda.GpuMat()
                    mask_gpu.upload(mask)
                except Exception as e:
                    logger.warning(f"OpenCV CUDA not properly available: {e}")
                    raise ValueError("OpenCV CUDA support not available")

                # Process each bit with better error handling
                for i in range(num_bits):
                    normal_idx = i * 2
                    inverted_idx = i * 2 + 1

                    # Ensure grayscale format (single channel)
                    normal = gray_processed[normal_idx]
                    inverted = gray_processed[inverted_idx]

                    # Convert to GPU using upload method (safer)
                    try:
                        normal_gpu = cv2.cuda.GpuMat()
                        inverted_gpu = cv2.cuda.GpuMat()
                        normal_gpu.upload(normal)
                        inverted_gpu.upload(inverted)
                    except Exception as e:
                        logger.warning(f"Error uploading Gray code images to GPU for bit {i}: {e}")
                        raise ValueError(f"GPU upload failed for bit {i}")

                    try:
                        # SLStudio approach: measure absolute difference between normal/inverted
                        # for more robust threshold detection
                        diff_gpu = cv2.cuda.absdiff(normal_gpu, inverted_gpu)
                        thresh_gpu = cv2.cuda.threshold(diff_gpu, 20, 255, cv2.THRESH_BINARY)[1]
                    except Exception as e:
                        logger.warning(f"Error in GPU threshold operations for bit {i}: {e}")
                        raise ValueError(f"GPU threshold failed for bit {i}")

                    try:
                        # Download results
                        normal_cpu = normal_gpu.download()
                        inverted_cpu = inverted_gpu.download()
                        thresh_cpu = thresh_gpu.download()

                        # 1 where normal > inverted, 0 otherwise (Using SLStudio's approach)
                        bit_val = (normal_cpu > inverted_cpu).astype(np.uint8)

                        # Apply mask - only set bits where difference is significant
                        bit_val = bit_val & (thresh_cpu > 0)

                        # Store in binary_codes
                        binary_codes[:, :, i] = bit_val

                        # Release GPU memory
                        normal_gpu.release()
                        inverted_gpu.release()
                        diff_gpu.release()
                        thresh_gpu.release()
                    except Exception as e:
                        # Release memory even on error
                        try:
                            normal_gpu.release()
                            inverted_gpu.release()
                            diff_gpu.release()
                            thresh_gpu.release()
                        except:
                            pass
                        logger.warning(f"Error downloading GPU results for bit {i}: {e}")
                        raise ValueError(f"GPU download failed for bit {i}")

                # Release mask memory
                mask_gpu.release()
                logger.debug("GPU Gray code decoding completed successfully")

            except Exception as e:
                logger.error(f"GPU Gray code decoding failed: {type(e).__name__}: {e}, falling back to CPU")
                if self.debug_mode:
                    import traceback
                    logger.debug(f"GPU error traceback: {traceback.format_exc()}")

                # Fall back to CPU implementation (using SLStudio approach for bit determination)
                for i in range(num_bits):
                    normal_idx = i * 2
                    inverted_idx = i * 2 + 1

                    # Normal and inverted patterns
                    normal = gray_processed[normal_idx]
                    inverted = gray_processed[inverted_idx]

                    # SLStudio approach: use absolute difference for more robust bit determination
                    diff = cv2.absdiff(normal, inverted)
                    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                    # Determine bit value (SLStudio approach)
                    bit_val = (normal > inverted).astype(np.uint8)

                    # Apply mask - only set bits where difference is significant
                    bit_val = bit_val & (thresh > 0)

                    # Store in binary_codes
                    binary_codes[:, :, i] = bit_val
        else:
            # CPU implementation (based on SLStudio approach)
            for i in range(num_bits):
                normal_idx = i * 2
                inverted_idx = i * 2 + 1

                # Normal and inverted patterns
                normal = gray_processed[normal_idx]
                inverted = gray_processed[inverted_idx]

                # SLStudio approach: use absolute difference for more robust bit determination
                diff = cv2.absdiff(normal, inverted)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                # Determine bit value (SLStudio approach)
                bit_val = (normal > inverted).astype(np.uint8)

                # Apply mask - only set bits where difference is significant
                bit_val = bit_val & (thresh > 0)

                # Store in binary_codes
                binary_codes[:, :, i] = bit_val

        # Convert binary codes to pixel coordinates using SLStudio's approach
        # This is a more efficient implementation of Gray code to binary conversion
        decoded = np.zeros(mask.shape, dtype=np.int32) - 1  # -1 for invalid pixels

        # Use vectorized operations where possible
        img_height, img_width = mask.shape
        y_coords, x_coords = np.where(mask > 0)

        for idx in range(len(y_coords)):
            y, x = y_coords[idx], x_coords[idx]

            # Extract binary code for this pixel
            binary_code = binary_codes[y, x, :]

            # Convert binary to Gray code (SLStudio approach)
            gray_value = 0
            for j in range(num_bits):
                bit_val = binary_code[num_bits - j - 1]
                gray_value = (gray_value << 1) | bit_val

            # Convert Gray code to binary using SLStudio's algorithm
            # This is much more efficient than the previous approach
            binary_value = gray_value
            for shift in range(1, num_bits):
                binary_value ^= (binary_value >> shift)

            # Set decoded value only if within range
            if binary_value < width * height:
                decoded[y, x] = binary_value

        # Create coordinate map
        img_height, img_width = mask.shape
        coord_map = np.zeros((img_height, img_width, 2), dtype=np.float32)

        # For each valid pixel, convert projector coordinate to row, col
        valid_coords = np.where((decoded >= 0) & (mask > 0))
        y_coords, x_coords = valid_coords

        for idx in range(len(y_coords)):
            y, x = y_coords[idx], x_coords[idx]
            value = decoded[y, x]

            # Convert projector coordinate to row, col
            proj_y = value // width
            proj_x = value % width

            if 0 <= proj_x < width and 0 <= proj_y < height:
                coord_map[y, x, 0] = proj_y  # Row (V)
                coord_map[y, x, 1] = proj_x  # Column (U)

        return coord_map, mask
    
    def _find_stereo_correspondences(
        self,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        epipolar_tolerance: float = 5.0  # Balance between accuracy and robustness
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find corresponding points between stereo images for real-time scanning.

        For stereo camera configuration, we're finding correspondences between two camera views,
        not between camera and projector as in SLStudio. The projector patterns help us
        establish these correspondences indirectly.

        Args:
            left_coords: Left camera-projector correspondences
            right_coords: Right camera-projector correspondences
            left_mask: Left camera shadow mask
            right_mask: Right camera shadow mask
            epipolar_tolerance: Max distance from epipolar line

        Returns:
            Tuple of (left_points, right_points) as Nx2 arrays
        """
        height, width = left_mask.shape

        # Debug info about inputs
        if self.debug_mode:
            logger.info(f"Left mask non-zero pixels: {np.count_nonzero(left_mask)}")
            logger.info(f"Right mask non-zero pixels: {np.count_nonzero(right_mask)}")
            logger.info(f"Left coords shape: {left_coords.shape}")
            logger.info(f"Right coords shape: {right_coords.shape}")

        # Create mapping from projector to right camera (SLStudio approach)
        # This maps projector coordinates to camera coordinates for faster lookup
        proj_to_right = {}
        valid_right_count = 0

        # Build projection map using stricter validation (SLStudio approach)
        for y in range(height):
            for x in range(width):
                if right_mask[y, x] == 0:
                    continue  # Skip shadowed pixels

                # Get projector coordinates
                proj_v = int(right_coords[y, x, 0])
                proj_u = int(right_coords[y, x, 1])

                # SLStudio uses strict validation of projector coordinates
                if proj_v < 0 or proj_u < 0 or proj_v >= height or proj_u >= width:
                    continue  # Skip invalid coordinates

                valid_right_count += 1

                # Store coordinates - SLStudio groups by projector coordinate
                key = (proj_v, proj_u)
                if key not in proj_to_right:
                    proj_to_right[key] = []
                proj_to_right[key].append((x, y))

        if self.debug_mode:
            logger.info(f"Valid projector coordinates in right image: {valid_right_count}")
            logger.info(f"Unique projector coordinates: {len(proj_to_right)}")

        # Find correspondences using SLStudio approach
        left_points = []
        right_points = []
        valid_left_count = 0
        match_count = 0

        # Filter the most valid correspondences (better quality using SLStudio approach)
        for y in range(height):
            for x in range(width):
                if left_mask[y, x] == 0:
                    continue  # Skip shadowed pixels

                # Get projector coordinates
                proj_v = int(left_coords[y, x, 0])
                proj_u = int(left_coords[y, x, 1])

                # SLStudio uses strict validation of projector coordinates
                if proj_v < 0 or proj_u < 0 or proj_v >= height or proj_u >= width:
                    continue  # Skip invalid coordinates

                valid_left_count += 1

                # Find matches in right image using projector coordinates as key
                key = (proj_v, proj_u)
                if key in proj_to_right:
                    # SLStudio approach: find the match closest to the epipolar line
                    # This uses a much tighter epipolar constraint (2 pixels)
                    best_match = None
                    min_y_diff = float('inf')

                    for rx, ry in proj_to_right[key]:
                        y_diff = abs(y - ry)
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_match = (rx, ry)

                    # Only use if very close to epipolar line (SLStudio uses strict constraints)
                    if min_y_diff <= epipolar_tolerance:
                        match_count += 1
                        right_x, right_y = best_match
                        left_points.append([x, y])
                        right_points.append([right_x, right_y])

        # Apply statistical filtering optimized for stereo camera setup
        if len(left_points) > 20:
            # Convert to numpy arrays for vectorized operations
            left_arr = np.array(left_points)
            right_arr = np.array(right_points)

            # Calculate disparities (x-disparity is most important for stereo systems)
            disparities = left_arr[:, 0] - right_arr[:, 0]

            # In rectified stereo, disparities should be mostly positive
            # (right camera x-coordinates are smaller than left camera x-coordinates)
            # But we'll use robust statistics rather than enforcing this strictly

            # Find median disparity for central tendency
            median_disparity = np.median(disparities)

            # Use a robust threshold based on median absolute deviation (MAD)
            # MAD is less sensitive to outliers than standard deviation
            mad = np.median(np.abs(disparities - median_disparity))

            # Conservative threshold for real-time performance
            # Higher multiplier = keep more points at the cost of some outliers
            disparity_threshold = 3.0 * max(mad, 1.0)

            # Create a mask for valid disparities
            valid_mask = np.abs(disparities - median_disparity) <= disparity_threshold

            # Apply the filter
            filtered_left_points = []
            filtered_right_points = []

            for i in range(len(valid_mask)):
                if valid_mask[i]:
                    filtered_left_points.append(left_points[i])
                    filtered_right_points.append(right_points[i])

            # Update the point lists
            left_points = filtered_left_points
            right_points = filtered_right_points

            if self.debug_mode:
                logger.info(f"Outlier filtering removed {len(valid_mask) - sum(valid_mask)} points")
                logger.info(f"Median disparity: {median_disparity:.2f}, MAD: {mad:.2f}, threshold: {disparity_threshold:.2f}")

        # Always save debug images if in debug mode
        if self.debug_mode and hasattr(self, 'debug_dir'):
            import os
            os.makedirs(self.debug_dir, exist_ok=True)

            # Save masks for debugging
            left_mask_vis = left_mask.astype(np.uint8) * 255
            right_mask_vis = right_mask.astype(np.uint8) * 255

            # Create coordinate visualizations (SLStudio style)
            coord_viz_left = np.zeros((height, width, 3), dtype=np.uint8)
            coord_viz_right = np.zeros((height, width, 3), dtype=np.uint8)

            # Color pixels based on their projector coordinates (normalized)
            # This is similar to SLStudio's debug visualization
            for y in range(height):
                for x in range(width):
                    if left_mask[y, x] > 0:
                        proj_v = int(left_coords[y, x, 0])
                        proj_u = int(left_coords[y, x, 1])
                        if 0 <= proj_v < height and 0 <= proj_u < width:
                            # Normalize to 0-255 for visualization
                            r = min(255, (proj_v * 255) // height)
                            g = min(255, (proj_u * 255) // width)
                            coord_viz_left[y, x] = [r, g, 255]

                    if right_mask[y, x] > 0:
                        proj_v = int(right_coords[y, x, 0])
                        proj_u = int(right_coords[y, x, 1])
                        if 0 <= proj_v < height and 0 <= proj_u < width:
                            # Normalize to 0-255 for visualization
                            r = min(255, (proj_v * 255) // height)
                            g = min(255, (proj_u * 255) // width)
                            coord_viz_right[y, x] = [r, g, 255]

            # Save everything for debugging
            cv2.imwrite(os.path.join(self.debug_dir, "left_mask_debug.png"), left_mask_vis)
            cv2.imwrite(os.path.join(self.debug_dir, "right_mask_debug.png"), right_mask_vis)
            cv2.imwrite(os.path.join(self.debug_dir, "left_coords_viz.png"), coord_viz_left)
            cv2.imwrite(os.path.join(self.debug_dir, "right_coords_viz.png"), coord_viz_right)
            logger.info(f"Saved debug images to {self.debug_dir}")

            # If we found correspondences, visualize them too
            if len(left_points) > 0:
                # Create a combined visualization (SLStudio style)
                stereo_viz = np.zeros((height, width*2, 3), dtype=np.uint8)
                stereo_viz[:, :width] = coord_viz_left
                stereo_viz[:, width:] = coord_viz_right

                # Draw epipolar lines (SLStudio visualizes these)
                # Draw a grid of horizontal lines to show epipolar constraints
                for i in range(0, height, 20):
                    cv2.line(stereo_viz, (0, i), (width*2, i), (40, 40, 40), 1)

                # Draw lines between corresponding points
                for i in range(min(100, len(left_points))):  # Limit to 100 for clarity
                    pt1 = (int(left_points[i][0]), int(left_points[i][1]))
                    pt2 = (int(right_points[i][0]) + width, int(right_points[i][1]))
                    cv2.line(stereo_viz, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(stereo_viz, pt1, 3, (255, 0, 0), -1)
                    cv2.circle(stereo_viz, pt2, 3, (0, 0, 255), -1)

                # Add disparity colormap visualization (SLStudio approach)
                if len(left_points) > 20:
                    disparity_viz = np.zeros((height, width, 3), dtype=np.uint8)
                    for i in range(len(left_points)):
                        x1, y1 = int(left_points[i][0]), int(left_points[i][1])
                        x2 = int(right_points[i][0])
                        disparity = x1 - x2
                        # Map disparity to color (normalized heatmap)
                        normalized = max(0, min(255, int((disparity - median_disparity + 20) * 5)))
                        disparity_viz[y1, x1] = [normalized, 255 - normalized, 128]

                    cv2.imwrite(os.path.join(self.debug_dir, "disparity_viz.png"), disparity_viz)

                cv2.imwrite(os.path.join(self.debug_dir, "stereo_correspondences.png"), stereo_viz)

        if not left_points or not right_points:
            logger.warning("No stereo correspondences found!")
            logger.info(f"Valid projector coordinates in left image: {valid_left_count}")
            logger.info(f"Valid projector coordinates in right image: {valid_right_count}")
            logger.info(f"Matched points: {match_count}")

            # SLStudio diagnostic approach - log more specific information
            if valid_left_count == 0 or valid_right_count == 0:
                logger.warning("No valid projector coordinates found. Check pattern projection and detection.")
                # Log more specific diagnostic information
                logger.info("--- Diagnostic Information ---")
                logger.info(f"LEFT: Mask coverage: {np.count_nonzero(left_mask) / left_mask.size * 100:.2f}%")
                logger.info(f"RIGHT: Mask coverage: {np.count_nonzero(right_mask) / right_mask.size * 100:.2f}%")

                # Count valid pixels in coordinates
                left_valid = np.count_nonzero(left_coords[..., 0] >= 0) / left_coords.size * 100
                right_valid = np.count_nonzero(right_coords[..., 0] >= 0) / right_coords.size * 100
                logger.info(f"LEFT: Valid coordinates: {left_valid:.2f}%")
                logger.info(f"RIGHT: Valid coordinates: {right_valid:.2f}%")
            else:
                logger.warning(f"Found {valid_left_count} and {valid_right_count} valid projector coordinates, but no matches.")
                logger.warning(f"Try adjusting epipolar_tolerance (currently {epipolar_tolerance})")
        else:
            logger.info(f"Found {len(left_points)} stereo correspondences")

        return np.array(left_points), np.array(right_points)
    
    def _triangulate_points(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        rectification_params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences using SLStudio-inspired robustness.

        Args:
            left_points: Left image points (Nx2)
            right_points: Right image points (Nx2)
            rectification_params: Rectification parameters

        Returns:
            3D points (Nx3)
        """
        # Check if we have any points to triangulate
        if len(left_points) == 0 or len(right_points) == 0:
            logger.warning("No points to triangulate, returning empty point cloud")
            return np.array([])

        # Get projection matrices
        P1 = rectification_params["P1"]
        P2 = rectification_params["P2"]

        # SLStudio validation: ensure we have finite values only
        # Check both input arrays for non-finite values
        valid_mask_left = np.all(np.isfinite(left_points), axis=1)
        valid_mask_right = np.all(np.isfinite(right_points), axis=1)
        valid_mask = valid_mask_left & valid_mask_right

        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            logger.warning(f"Removed {n_invalid} points with non-finite coordinates before triangulation")
            left_points = left_points[valid_mask]
            right_points = right_points[valid_mask]

            # Return empty array if no valid points remain
            if len(left_points) == 0:
                logger.warning("No valid points remain after filtering, returning empty point cloud")
                return np.array([])

        # Reshape points for triangulation
        left_pts = left_points.reshape(-1, 1, 2).astype(np.float32)
        right_pts = right_points.reshape(-1, 1, 2).astype(np.float32)

        # OpenCV triangulatePoints expects points to be 2xN, not Nx2
        left_pts_2xn = np.transpose(left_pts, (2, 1, 0)).reshape(2, -1)
        right_pts_2xn = np.transpose(right_pts, (2, 1, 0)).reshape(2, -1)

        # Triangulate points
        try:
            # SLStudio uses a wrapper around triangulatePoints with error checking
            points_4d = cv2.triangulatePoints(P1, P2, left_pts_2xn, right_pts_2xn)
            points_4d = points_4d.T

            # SLStudio approach: validate homogeneous coordinates before conversion
            # Check for very small w values that can cause numerical instability
            w_values = points_4d[:, 3]
            valid_w = np.abs(w_values) > 1e-10

            if not np.all(valid_w):
                logger.warning(f"Removed {np.sum(~valid_w)} points with near-zero w coordinates")
                points_4d = points_4d[valid_w]

                # Return empty array if no valid points remain
                if len(points_4d) == 0:
                    logger.warning("No valid points after homogeneous coordinate validation")
                    return np.array([])

            # Convert to 3D points
            points_3d = points_4d[:, :3] / points_4d[:, 3:4]

            # SLStudio approach: final validation of 3D points
            # Check for invalid and extreme values
            valid_points = np.all(np.isfinite(points_3d), axis=1)

            # Check for extreme values (points very far from origin)
            # SLStudio uses a more sophistical approach, here we simplify
            dist_from_origin = np.linalg.norm(points_3d, axis=1)
            max_distance = 5000.0  # Filter very distant points (SLStudio uses scene-specific values)
            valid_distance = dist_from_origin < max_distance

            # Combine validations
            final_valid = valid_points & valid_distance

            if not np.all(final_valid):
                logger.warning(f"Removed {np.sum(~final_valid)} points with invalid or extreme 3D coordinates")
                points_3d = points_3d[final_valid]

            if self.debug_mode:
                logger.info(f"Triangulated {len(points_3d)} valid 3D points")
                logger.info(f"Point range: X={points_3d[:, 0].min():.2f} to {points_3d[:, 0].max():.2f}, "
                           f"Y={points_3d[:, 1].min():.2f} to {points_3d[:, 1].max():.2f}, "
                           f"Z={points_3d[:, 2].min():.2f} to {points_3d[:, 2].max():.2f}")

        except Exception as e:
            logger.error(f"Error during triangulation: {e}")
            logger.error(f"Left points shape: {left_points.shape}, Right points shape: {right_points.shape}")
            logger.error(f"P1 shape: {P1.shape}, P2 shape: {P2.shape}")
            # Return empty array on error
            return np.array([])

        return points_3d
    
    def _filter_point_cloud(
        self,
        points_3d: np.ndarray,
        max_distance: Optional[float] = None
    ) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Filter and clean 3D point cloud.

        Args:
            points_3d: 3D points (Nx3)
            max_distance: Maximum distance from origin

        Returns:
            Filtered point cloud
        """
        if max_distance is None:
            max_distance = self.config.max_distance

        # Check if points_3d is empty
        if points_3d.size == 0:
            logger.warning("Empty point cloud, nothing to filter")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()

        # Check dimensions before applying axis operations
        if points_3d.ndim > 1:
            # For multi-dimensional arrays (normal case)
            mask = ~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)
        else:
            # For 1D arrays
            mask = ~np.isnan(points_3d) & ~np.isinf(points_3d)

        clean_pts = points_3d[mask]
        
        if len(clean_pts) == 0:
            logger.warning("No valid points after removing NaN and infinite values")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Filter by distance
        if clean_pts.ndim > 1:
            dist = np.linalg.norm(clean_pts, axis=1)
        else:
            # For 1D array, just use its absolute value as distance
            dist = np.abs(clean_pts)

        mask = dist < max_distance
        clean_pts = clean_pts[mask]
        
        if len(clean_pts) == 0:
            logger.warning(f"No valid points after distance filtering (max_dist={max_distance})")
            return np.array([]) if not OPEN3D_AVAILABLE else o3d.geometry.PointCloud()
        
        # Use Open3D for advanced filtering if available
        if OPEN3D_AVAILABLE and len(clean_pts) > 20:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clean_pts)
            
            # Downsample for speed in real-time mode
            if self.config.point_cloud_downsample:
                pcd = pcd.voxel_down_sample(voxel_size=self.config.downsample_voxel_size)
                
            # Remove outliers (minimal processing for speed)
            if len(pcd.points) > 50:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=10,  # Reduced for speed
                    std_ratio=1.5     # Less strict for real-time
                )
            
            return pcd
        
        # If Open3D not available, just return filtered points
        return clean_pts
    
    def _enhance_point_cloud(self, point_cloud):
        """
        Enhance point cloud using neural network for noise reduction and feature enhancement.

        Args:
            point_cloud: Input point cloud

        Returns:
            Enhanced point cloud
        """
        if not POINT_CLOUD_NN_AVAILABLE or not hasattr(self, 'point_cloud_enhancer'):
            return point_cloud

        try:
            # Skip if empty
            if isinstance(point_cloud, o3dg.PointCloud) and len(point_cloud.points) == 0:
                return point_cloud
            elif isinstance(point_cloud, np.ndarray) and point_cloud.size == 0:
                return point_cloud

            # Check for 1D array and reshape if needed
            if isinstance(point_cloud, np.ndarray) and point_cloud.ndim == 1:
                # If it's a 1D array, it can't be properly processed
                logger.warning("Cannot enhance 1D point cloud, returning as is")
                return point_cloud

            # Check if we have self.nn_model defined (referenced in _process_frame)
            if not hasattr(self, 'nn_model') and not hasattr(self, 'point_cloud_enhancer'):
                return point_cloud

            # Use our advanced point cloud enhancer
            enhanced_point_cloud = self.point_cloud_enhancer.enhance(
                point_cloud,
                denoise_strength=self.config.nn_denoise_strength if hasattr(self.config, 'nn_denoise_strength') else 0.5,
                upsample=hasattr(self.config, 'nn_upsample') and self.config.nn_upsample,
                target_points=self.config.nn_target_points if hasattr(self.config, 'nn_target_points') else None
            )

            return enhanced_point_cloud

        except Exception as e:
            logger.error(f"Error enhancing point cloud: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return point_cloud
    
    def _combine_point_clouds(self, point_clouds):
        """
        Combine multiple point clouds for temporal averaging.

        Args:
            point_clouds: List of point clouds

        Returns:
            Combined point cloud
        """
        if not point_clouds:
            return None

        # Convert all to Open3D if they're not already
        o3d_clouds = []
        for pc in point_clouds:
            if pc is None:
                continue

            if isinstance(pc, o3dg.PointCloud):
                if len(pc.points) > 0:  # Only add non-empty point clouds
                    o3d_clouds.append(pc)
            elif isinstance(pc, np.ndarray):
                if pc.size > 0:  # Check if array is non-empty
                    # Handle 1D arrays
                    if pc.ndim == 1:
                        logger.warning("Skipping 1D point cloud in combine operation")
                        continue

                    # Convert to point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    o3d_clouds.append(pcd)

        if not o3d_clouds:
            logger.warning("No valid point clouds to combine")
            return None
        
        # Combine point clouds
        if len(o3d_clouds) == 1:
            return o3d_clouds[0]
        
        combined = o3d_clouds[0]
        for pc in o3d_clouds[1:]:
            combined += pc
        
        # Downsample the combined cloud for memory efficiency
        if len(combined.points) > 1000:
            combined = combined.voxel_down_sample(voxel_size=self.config.downsample_voxel_size)
        
        return combined
    
    def get_current_point_cloud(self):
        """Get the latest point cloud."""
        return self.current_point_cloud
    
    def get_fps(self):
        """Get the current frames per second rate."""
        return self.fps
    
    def get_scan_count(self):
        """Get the total number of scans performed."""
        return self.scan_count

    def show_preview(self, window_name: str = "Realtime 3D Preview", width: int = 800, height: int = 600):
        """
        Show a real-time OpenCV preview of the current scan.

        This method provides a simple 2D visualization of the point cloud using OpenCV,
        which is useful for systems that don't have Open3D available.

        Args:
            window_name: Name of the preview window
            width: Width of the preview window
            height: Height of the preview window

        Returns:
            True if the preview was shown, False otherwise
        """
        try:
            # Create a blank image regardless of point cloud status
            img = np.zeros((height, width, 3), dtype=np.uint8)
            half_width = width // 2

            # Add grid lines for reference
            grid_color = (30, 30, 30)
            grid_step = 50
            for i in range(0, width, grid_step):
                cv2.line(img, (i, 0), (i, height), grid_color, 1)
            for i in range(0, height, grid_step):
                cv2.line(img, (0, i), (width, i), grid_color, 1)

            # Add border between views
            cv2.line(img, (half_width, 0), (half_width, height), (70, 70, 70), 1)

            # Add view titles
            cv2.putText(img[:, :half_width], "Top View (X-Z)", (10, 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(img[:, half_width:], "Front View (X-Y)", (half_width + 10, 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Check if we have a point cloud to display
            if self.current_point_cloud is None or (
                hasattr(self.current_point_cloud, 'points') and len(self.current_point_cloud.points) == 0) or (
                isinstance(self.current_point_cloud, np.ndarray) and len(self.current_point_cloud) == 0):

                # Display status message when no points are available
                cv2.putText(img, "Waiting for point cloud data...", (width//4, height//2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

                # Add status info
                info_text = f"FPS: {self.fps:.1f} | Scan #: {self.scan_count} | Points: 0"
                cv2.putText(img, info_text, (10, height - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Show image
                cv2.imshow(window_name, img)
                cv2.waitKey(1)  # Refresh display
                return True

            # Convert point cloud to numpy array if needed
            if hasattr(self.current_point_cloud, 'points'):
                points = np.asarray(self.current_point_cloud.points)
            else:
                points = self.current_point_cloud

            if len(points) == 0:
                # Already handled above
                return True

            # Calculate point limits for normalization
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            range_vals = max_vals - min_vals

            # Avoid division by zero
            range_vals[range_vals == 0] = 1.0

            # Convert points to 2D for display (top view and front view)
            points_normalized = (points - min_vals) / range_vals

            # Top view (X-Z)
            top_view = img[:, :half_width].copy()
            for point in points_normalized:
                x, y, z = point
                px = int(x * (half_width - 20)) + 10
                py = int(z * (height - 20)) + 10
                if 0 <= px < half_width and 0 <= py < height:
                    # Color based on height (Y)
                    color = (
                        int(255 * (1 - y)),  # Blue (inverse of height)
                        int(255 * min(y, 0.5) * 2),  # Green (peak at middle height)
                        int(255 * y)   # Red (proportional to height)
                    )
                    cv2.circle(top_view, (px, py), 1, color, -1)

            # Front view (X-Y)
            front_view = img[:, half_width:].copy()
            for point in points_normalized:
                x, y, z = point
                px = int(x * (half_width - 20)) + 10 + half_width  # Add half_width to position in right side
                py = int((1-y) * (height - 20)) + 10  # Invert Y for display
                if half_width <= px < width and 0 <= py < height:
                    # Color based on depth (Z)
                    color = (
                        int(255 * z),  # Blue (proportional to depth)
                        int(255 * (1 - z)),  # Green (inverse of depth)
                        int(255 * min(z, 0.5) * 2)  # Red (peak at middle depth)
                    )
                    cv2.circle(front_view, (px - half_width, py), 1, color, -1)  # Adjust x coordinate for drawing

            # Add status info
            info_text = f"Points: {len(points)} | FPS: {self.fps:.1f} | Scan #: {self.scan_count}"
            cv2.putText(img, info_text, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Combine views
            img[:, :half_width] = top_view
            img[:, half_width:] = front_view

            # Show image
            cv2.imshow(window_name, img)
            cv2.waitKey(1)  # Refresh display

            return True
        except Exception as e:
            logger.error(f"Error showing OpenCV preview: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def save_preview_image(self, output_path: str, width: int = 800, height: int = 600):
        """
        Save a preview image of the current scan.

        Args:
            output_path: Path to save the preview image
            width: Width of the preview image
            height: Height of the preview image

        Returns:
            True if the image was saved, False otherwise
        """
        if self.current_point_cloud is None:
            logger.warning("No point cloud available to save")
            return False

        try:
            # Create a temporary window name
            temp_window = f"save_preview_{int(time.time())}"

            # Generate preview
            if not self.show_preview(temp_window, width, height):
                return False

            # Get the window content
            img = None
            # Allow a short delay for the window to render
            for _ in range(5):
                img = cv2.getWindowImage(temp_window)
                if img is not None:
                    break
                time.sleep(0.1)

            # Save image
            if img is not None:
                cv2.imwrite(output_path, img)
                logger.info(f"Preview image saved to {output_path}")
            else:
                logger.warning("Could not capture window content for saving")

            # Close temporary window
            cv2.destroyWindow(temp_window)

            return img is not None
        except Exception as e:
            logger.error(f"Error saving preview image: {e}")
            return False

    def check_camera_focus(self, camera_id: Optional[str] = None,
                           threshold_for_good: float = None,
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, str, np.ndarray]:
        """
        Check the focus quality of a camera.

        Args:
            camera_id: Camera ID (if None, uses the first camera)
            threshold_for_good: Optional custom threshold to determine good focus
            roi: Optional region of interest (x, y, width, height) to analyze

        Returns:
            Tuple of (focus_score, quality_level, image)
        """
        if not hasattr(self.client, 'camera'):
            logger.error("Client has no camera module")
            return 0.0, "unknown", None

        return self.client.camera.check_focus(camera_id,
                                           num_samples=3,
                                           roi=roi,
                                           threshold_for_good=threshold_for_good)

    def check_stereo_focus(self, left_camera_id: Optional[str] = None,
                           right_camera_id: Optional[str] = None,
                           threshold_for_good: float = None,
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Check the focus quality of both stereo cameras.

        Args:
            left_camera_id: ID of left camera (if None, auto-detected)
            right_camera_id: ID of right camera (if None, auto-detected)
            threshold_for_good: Optional custom threshold to determine good focus
            roi: Optional region of interest (x, y, width, height) to analyze

        Returns:
            Dictionary with focus results for both cameras
        """
        if not hasattr(self.client, 'camera'):
            logger.error("Client has no camera module")
            return {}, {}

        return self.client.camera.check_stereo_focus(left_camera_id,
                                                  right_camera_id,
                                                  num_samples=3,
                                                  roi=roi,
                                                  threshold_for_good=threshold_for_good)

    def interactive_focus_check(self, camera_id: Optional[str] = None, interval: float = 0.5,
                               roi: Optional[Tuple[int, int, int, int]] = None,
                               threshold_for_good: float = None) -> Tuple[float, str, np.ndarray]:
        """
        Run an interactive focus check with continuous feedback for a single camera.

        Args:
            camera_id: Camera ID (if None, uses the first camera)
            interval: Seconds between focus checks
            roi: Optional region of interest (x, y, width, height) to analyze
            threshold_for_good: Optional custom threshold to determine good focus

        Returns:
            Tuple of (focus_score, quality_level, image)
        """
        if not hasattr(self.client, 'camera'):
            logger.error("Client has no camera module")
            return 0.0, "unknown", None

        return self.client.camera.interactive_focus_check(camera_id,
                                                      interval=interval,
                                                      roi=roi,
                                                      threshold_for_good=threshold_for_good)

    def interactive_stereo_focus_check(self, left_camera_id: Optional[str] = None,
                                      right_camera_id: Optional[str] = None,
                                      interval: float = 0.5,
                                      roi: Optional[Tuple[int, int, int, int]] = None,
                                      threshold_for_good: float = None) -> Dict[str, Any]:
        """
        Run an interactive focus check with continuous feedback for stereo cameras.

        Args:
            left_camera_id: ID of left camera (if None, auto-detected)
            right_camera_id: ID of right camera (if None, auto-detected)
            interval: Seconds between focus checks
            roi: Optional region of interest (x, y, width, height) to analyze
            threshold_for_good: Optional custom threshold to determine good focus

        Returns:
            Dictionary with focus results for both cameras
        """
        if not hasattr(self.client, 'camera'):
            logger.error("Client has no camera module")
            return {}, {}

        return self.client.camera.interactive_stereo_focus_check(
            left_camera_id,
            right_camera_id,
            interval=interval,
            roi=roi,
            threshold_for_good=threshold_for_good
        )


def create_realtime_scanner(
    client,
    quality: str = "medium",
    config: Optional[RealTimeScanConfig] = None,
    calibration_file: Optional[str] = None,
    on_new_frame: Optional[Callable] = None
) -> RealTimeScanner:
    """
    Create a real-time scanner with the specified quality preset or custom config.

    Args:
        client: UnlookClient instance
        quality: Quality preset (fast, medium, high, ultra) - used only if config is None
        config: Custom scanner configuration - if provided, overrides quality parameter
        calibration_file: Path to stereo calibration file
        on_new_frame: Callback when new scan data is available

    Returns:
        Configured RealTimeScanner instance
    """
    # If config is provided, use it, otherwise create from quality preset
    if config is None:
        config = RealTimeScanConfig()
        config.set_quality_preset(quality)

    scanner = RealTimeScanner(
        client=client,
        config=config,
        calibration_file=calibration_file,
        on_new_frame=on_new_frame
    )

    return scanner