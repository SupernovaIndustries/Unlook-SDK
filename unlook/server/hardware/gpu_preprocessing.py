"""
GPU-accelerated preprocessing module for Raspberry Pi.
Utilizes VideoCore VI GPU for image processing operations.
"""

import logging
import time
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Try to import Pi-specific GPU libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import picamera2 for hardware acceleration
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder, Quality
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for GPU preprocessing."""
    lens_correction: bool = True
    pattern_preprocessing: bool = True  # Enabled for full preprocessing
    compression_level: str = "adaptive"  # adaptive, high, medium, low
    quality_assessment: bool = True
    delta_encoding: bool = True
    adaptive_quality: bool = True  # Enable adaptive quality levels
    edge_density_threshold: float = 0.1  # Threshold for edge density analysis
    max_downsampling_factor: float = 0.5  # Maximum downsampling (0.5 = half resolution)
    force_full_resolution: bool = False  # Force full resolution (for 2K mode)


class RaspberryProcessingV2:
    """
    GPU-accelerated preprocessing for Raspberry Pi CM4.
    Maximizes VideoCore VI GPU utilization for real-time processing.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize GPU preprocessing system.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.initialized = False
        
        # Check available acceleration
        self.gpu_available = self._check_gpu_availability()
        
        # Calibration data cache
        self.calibration_data = None
        
        # Delta encoding state
        self.previous_frames = {}
        
        # Performance metrics
        self.preprocessing_times = []
        
        if self.gpu_available:
            logger.info("GPU acceleration available for preprocessing")
            self._initialize_gpu_resources()
        else:
            logger.warning("GPU acceleration not available, using CPU fallback")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        if not CV2_AVAILABLE:
            return False
            
        # Check for VideoCore GPU support
        try:
            # Test GPU mat allocation
            test_mat = cv2.UMat(np.zeros((100, 100), dtype=np.uint8))
            return True
        except:
            return False
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources for preprocessing."""
        try:
            # Pre-allocate GPU buffers for common operations
            self.gpu_buffers = {
                'temp1': cv2.UMat(np.zeros((1088, 1456), dtype=np.uint8)),
                'temp2': cv2.UMat(np.zeros((1088, 1456), dtype=np.uint8)),
                'mask': cv2.UMat(np.zeros((1088, 1456), dtype=np.uint8))
            }
            self.initialized = True
            logger.info("GPU resources initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GPU resources: {e}")
            self.gpu_available = False
    
    def preprocess_frame(self, frame: np.ndarray, camera_id: str, 
                        metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform GPU-accelerated preprocessing on a frame.
        
        Args:
            frame: Input frame as numpy array
            camera_id: Camera identifier
            metadata: Optional frame metadata
            
        Returns:
            Dictionary with preprocessed frame and metadata
        """
        start_time = time.time()
        result = {
            'original_shape': frame.shape,
            'camera_id': camera_id,
            'preprocessing_applied': []
        }
        
        try:
            # Upload to GPU if available
            if self.gpu_available:
                gpu_frame = cv2.UMat(frame)
            else:
                gpu_frame = frame
            
            # 1. Lens correction
            if self.config.lens_correction and self.calibration_data:
                gpu_frame = self._apply_lens_correction_gpu(gpu_frame, camera_id)
                result['preprocessing_applied'].append('lens_correction')
            
            # 2. Adaptive quality level processing
            if self.config.adaptive_quality:
                gpu_frame, quality_info = self._apply_adaptive_quality_gpu(gpu_frame, metadata)
                result['adaptive_quality'] = quality_info
                if quality_info['downsampled']:
                    result['preprocessing_applied'].append('adaptive_downsampling')
            
            # 3. Edge-preserving filtering (bilateral filter)
            if metadata and metadata.get('pattern_type') in ['gray_code', 'phase_shift']:
                gpu_frame = self._apply_edge_preserving_filter_gpu(gpu_frame)
                result['preprocessing_applied'].append('edge_preserving_filter')
            
            # ROI detection removed - will be implemented later when scanner is fully functional
            
            # 4. Pattern preprocessing (enabled for full preprocessing)
            if self.config.pattern_preprocessing and metadata:
                pattern_type = metadata.get('pattern_type')
                if pattern_type == 'gray_code':
                    gpu_frame = self._preprocess_gray_code_gpu(gpu_frame)
                    result['preprocessing_applied'].append('gray_code_preprocessing')
                elif pattern_type == 'phase_shift':
                    gpu_frame = self._preprocess_phase_shift_gpu(gpu_frame)
                    result['preprocessing_applied'].append('phase_shift_preprocessing')
            
            # 5. Quality assessment
            if self.config.quality_assessment:
                quality_metrics = self._assess_quality_gpu(gpu_frame)
                result['quality_metrics'] = quality_metrics
            
            # 6. Compression
            compressed_frame, compression_info = self._compress_frame_gpu(
                gpu_frame, camera_id, self.config.compression_level
            )
            result['compressed_frame'] = compressed_frame
            result['compression_info'] = compression_info
            
            # Download from GPU if needed
            if self.gpu_available:
                result['frame'] = gpu_frame.get()
            else:
                result['frame'] = gpu_frame
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            result['frame'] = frame
            result['error'] = str(e)
        
        # Track performance
        preprocessing_time = time.time() - start_time
        self.preprocessing_times.append(preprocessing_time)
        result['preprocessing_time_ms'] = preprocessing_time * 1000
        
        return result
    
    def _apply_lens_correction_gpu(self, frame, camera_id: str):
        """Apply GPU-accelerated lens distortion correction."""
        if not self.calibration_data or camera_id not in self.calibration_data:
            return frame
            
        calib = self.calibration_data[camera_id]
        
        try:
            # Use GPU-accelerated undistortion
            if self.gpu_available:
                return cv2.undistort(frame, calib['camera_matrix'], 
                                   calib['dist_coeffs'], None, 
                                   calib['new_camera_matrix'])
            else:
                return cv2.undistort(frame, calib['camera_matrix'], 
                                   calib['dist_coeffs'])
        except Exception as e:
            logger.error(f"Lens correction failed: {e}")
            return frame
    
    def _apply_adaptive_quality_gpu(self, frame, metadata):
        """Apply adaptive quality processing based on scene complexity."""
        try:
            # Get frame for analysis
            frame_cpu = frame.get() if isinstance(frame, cv2.UMat) else frame
            
            # Convert to grayscale for edge analysis
            if len(frame_cpu.shape) == 3:
                gray = cv2.cvtColor(frame_cpu, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame_cpu
            
            # Calculate edge density using Sobel operator
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Calculate edge density
            edge_density = np.mean(edges > 50) / 255.0  # Normalized edge density
            
            # Determine if downsampling should be applied
            should_downsample = edge_density < self.config.edge_density_threshold
            
            quality_info = {
                'edge_density': float(edge_density),
                'threshold': self.config.edge_density_threshold,
                'downsampled': False,
                'original_size': frame_cpu.shape[:2],
                'final_size': frame_cpu.shape[:2],
                'downsampling_factor': 1.0
            }
            
            # Check if we should force full resolution (e.g., for 2K mode)
            if self.config.force_full_resolution:
                logger.debug("Force full resolution enabled - skipping downsampling")
                return frame, quality_info
            
            if should_downsample:
                # Simple scene - apply downsampling
                factor = self.config.max_downsampling_factor
                new_height = int(frame_cpu.shape[0] * factor)
                new_width = int(frame_cpu.shape[1] * factor)
                
                if self.gpu_available and isinstance(frame, cv2.UMat):
                    downsampled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    downsampled = cv2.resize(frame_cpu, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    if self.gpu_available:
                        downsampled = cv2.UMat(downsampled)
                
                quality_info.update({
                    'downsampled': True,
                    'final_size': (new_height, new_width),
                    'downsampling_factor': factor
                })
                
                logger.info(f"Applied adaptive downsampling: {frame_cpu.shape[:2]} -> {(new_height, new_width)} "
                           f"(edge density: {edge_density:.3f})")
                
                return downsampled, quality_info
            else:
                # Complex scene - keep full resolution
                logger.debug(f"Maintaining full resolution (edge density: {edge_density:.3f})")
                return frame, quality_info
                
        except Exception as e:
            logger.error(f"Adaptive quality processing failed: {e}")
            quality_info = {
                'edge_density': 0.0,
                'threshold': self.config.edge_density_threshold,
                'downsampled': False,
                'error': str(e)
            }
            return frame, quality_info
    
    def _apply_edge_preserving_filter_gpu(self, frame):
        """Apply edge-preserving bilateral filter to reduce noise while preserving edges."""
        try:
            # Bilateral filter parameters optimized for structured light patterns
            d = 9  # Neighborhood diameter
            sigma_color = 75  # Filter sigma in color space (higher = colors farther apart mix more)
            sigma_space = 75  # Filter sigma in coordinate space (higher = farther pixels influence more)
            
            if self.gpu_available and isinstance(frame, cv2.UMat):
                # GPU-accelerated bilateral filter
                filtered = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
            else:
                # CPU fallback
                frame_cpu = frame.get() if isinstance(frame, cv2.UMat) else frame
                filtered = cv2.bilateralFilter(frame_cpu, d, sigma_color, sigma_space)
                if self.gpu_available:
                    filtered = cv2.UMat(filtered)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Edge-preserving filtering failed: {e}")
            return frame
    
    # ROI detection methods removed - will be implemented later when scanner is fully functional
    
    def _preprocess_gray_code_gpu(self, frame):
        """Preprocess Gray code patterns on GPU."""
        try:
            # Enhance contrast for better Gray code detection
            if self.gpu_available:
                # GPU-accelerated histogram equalization
                if len(frame.size()) == 2:  # Grayscale
                    return cv2.equalizeHist(frame)
                else:  # Color
                    # Convert to YUV, equalize Y channel, convert back
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    yuv_planes = cv2.split(yuv)
                    yuv_planes[0] = cv2.equalizeHist(yuv_planes[0])
                    yuv = cv2.merge(yuv_planes)
                    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                # CPU fallback
                if len(frame.shape) == 2:
                    return cv2.equalizeHist(frame)
                else:
                    return frame
                    
        except Exception as e:
            logger.error(f"Gray code preprocessing failed: {e}")
            return frame
    
    def _preprocess_phase_shift_gpu(self, frame):
        """Preprocess phase shift patterns on GPU."""
        try:
            # Apply Gaussian filter to reduce noise in phase patterns
            if self.gpu_available:
                return cv2.GaussianBlur(frame, (5, 5), 1.0)
            else:
                return cv2.GaussianBlur(frame, (5, 5), 1.0)
                
        except Exception as e:
            logger.error(f"Phase shift preprocessing failed: {e}")
            return frame
    
    def _assess_quality_gpu(self, frame) -> Dict[str, float]:
        """Assess frame quality using GPU acceleration."""
        metrics = {}
        
        try:
            # Get frame data
            if isinstance(frame, cv2.UMat):
                frame_cpu = frame.get()
            else:
                frame_cpu = frame
            
            # 1. Sharpness (Laplacian variance)
            if self.gpu_available:
                laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                metrics['sharpness'] = float(np.var(laplacian.get() if isinstance(laplacian, cv2.UMat) else laplacian))
            else:
                laplacian = cv2.Laplacian(frame_cpu, cv2.CV_64F)
                metrics['sharpness'] = float(np.var(laplacian))
            
            # 2. Brightness
            metrics['brightness'] = float(np.mean(frame_cpu))
            
            # 3. Contrast
            metrics['contrast'] = float(np.std(frame_cpu))
            
            # 4. SNR estimate (simplified)
            if self.gpu_available:
                noise = cv2.GaussianBlur(frame, (5, 5), 0)
                noise = cv2.absdiff(frame, noise)
                metrics['snr_estimate'] = float(np.mean(frame_cpu) / (np.std(noise.get() if isinstance(noise, cv2.UMat) else noise) + 1e-6))
            else:
                metrics['snr_estimate'] = metrics['brightness'] / (metrics['contrast'] + 1e-6)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
        
        return metrics
    
    def _compress_frame_gpu(self, frame, camera_id: str, compression_level: str) -> Tuple[bytes, Dict]:
        """Compress frame with GPU acceleration."""
        info = {
            'method': 'jpeg',
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0
        }
        
        try:
            # Get frame data
            if isinstance(frame, cv2.UMat):
                frame_cpu = frame.get()
            else:
                frame_cpu = frame
            
            info['original_size'] = frame_cpu.nbytes
            
            # Adaptive compression based on level and quality metrics
            if compression_level == "adaptive":
                # Use quality metrics to determine compression
                quality = 85  # Default
                if hasattr(self, 'last_quality_metrics'):
                    sharpness = self.last_quality_metrics.get('sharpness', 100)
                    if sharpness < 50:
                        quality = 95  # Low sharpness, use higher quality
                    elif sharpness > 200:
                        quality = 75  # High sharpness, can compress more
            elif compression_level == "high":
                quality = 95
            elif compression_level == "medium":
                quality = 85
            else:  # low
                quality = 75
            
            # Delta encoding if enabled
            if self.config.delta_encoding and camera_id in self.previous_frames:
                # Compute difference from previous frame
                prev_frame = self.previous_frames[camera_id]
                if prev_frame.shape == frame_cpu.shape:
                    diff_frame = cv2.absdiff(frame_cpu, prev_frame)
                    
                    # Check if difference is significant
                    if np.mean(diff_frame) < 10:  # Small change
                        # Encode only the difference
                        _, compressed = cv2.imencode('.jpg', diff_frame, 
                                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
                        info['method'] = 'delta_jpeg'
                        info['delta_encoded'] = True
                    else:
                        # Too much change, encode full frame
                        _, compressed = cv2.imencode('.jpg', frame_cpu, 
                                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    _, compressed = cv2.imencode('.jpg', frame_cpu, 
                                               [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                # Standard compression
                _, compressed = cv2.imencode('.jpg', frame_cpu, 
                                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Store frame for delta encoding
            self.previous_frames[camera_id] = frame_cpu.copy()
            
            compressed_bytes = compressed.tobytes()
            info['compressed_size'] = len(compressed_bytes)
            info['compression_ratio'] = info['original_size'] / info['compressed_size']
            info['quality'] = quality
            
            return compressed_bytes, info
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return b'', info
    
    def set_calibration_data(self, calibration_data: Dict):
        """Set calibration data for lens correction."""
        self.calibration_data = calibration_data
        logger.info(f"Calibration data set for {len(calibration_data)} cameras")
    
    def store_reference_pattern(self, image: np.ndarray, camera_id: str, reference_type: str):
        """
        Store reference pattern for pattern preprocessing.
        
        Args:
            image: Reference pattern image
            camera_id: Camera identifier
            reference_type: Type of reference (e.g., 'white', 'black')
        """
        if not hasattr(self, 'reference_patterns'):
            self.reference_patterns = {}
        
        if camera_id not in self.reference_patterns:
            self.reference_patterns[camera_id] = {}
        
        self.reference_patterns[camera_id][reference_type] = image.copy()
        logger.debug(f"Stored {reference_type} reference pattern for camera {camera_id}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get preprocessing performance statistics."""
        if not self.preprocessing_times:
            return {}
            
        return {
            'avg_preprocessing_ms': np.mean(self.preprocessing_times) * 1000,
            'max_preprocessing_ms': np.max(self.preprocessing_times) * 1000,
            'min_preprocessing_ms': np.min(self.preprocessing_times) * 1000,
            'gpu_enabled': self.gpu_available
        }