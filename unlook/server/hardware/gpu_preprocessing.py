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
    roi_detection: bool = True
    pattern_preprocessing: bool = False  # Disabled by default
    compression_level: str = "adaptive"  # adaptive, high, medium, low
    quality_assessment: bool = True
    delta_encoding: bool = True


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
        self.roi_cache = {}
        
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
            
            # 2. ROI detection and cropping
            if self.config.roi_detection:
                roi = self._detect_roi_gpu(gpu_frame, camera_id)
                if roi:
                    gpu_frame = self._crop_roi_gpu(gpu_frame, roi)
                    result['roi'] = roi
                    result['preprocessing_applied'].append('roi_detection')
            
            # 3. Pattern preprocessing (if enabled)
            if self.config.pattern_preprocessing and metadata:
                pattern_type = metadata.get('pattern_type')
                if pattern_type == 'gray_code':
                    gpu_frame = self._preprocess_gray_code_gpu(gpu_frame)
                    result['preprocessing_applied'].append('gray_code_preprocessing')
                elif pattern_type == 'phase_shift':
                    gpu_frame = self._preprocess_phase_shift_gpu(gpu_frame)
                    result['preprocessing_applied'].append('phase_shift_preprocessing')
            
            # 4. Quality assessment
            if self.config.quality_assessment:
                quality_metrics = self._assess_quality_gpu(gpu_frame)
                result['quality_metrics'] = quality_metrics
            
            # 5. Compression
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
    
    def _detect_roi_gpu(self, frame, camera_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Detect region of interest using GPU acceleration."""
        # Check cache first
        if camera_id in self.roi_cache:
            return self.roi_cache[camera_id]
        
        try:
            # Simple edge-based ROI detection
            if self.gpu_available:
                # GPU-accelerated edge detection
                edges = cv2.Canny(frame, 50, 150)
                
                # Find contours
                contours, _ = cv2.findContours(edges.get() if isinstance(edges, cv2.UMat) else edges,
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Add margin
                    margin = 50
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.get().shape[1] if isinstance(frame, cv2.UMat) else frame.shape[1] - x, w + 2 * margin)
                    h = min(frame.get().shape[0] if isinstance(frame, cv2.UMat) else frame.shape[0] - y, h + 2 * margin)
                    
                    roi = (x, y, w, h)
                    self.roi_cache[camera_id] = roi
                    return roi
            
        except Exception as e:
            logger.error(f"ROI detection failed: {e}")
        
        return None
    
    def _crop_roi_gpu(self, frame, roi: Tuple[int, int, int, int]):
        """Crop frame to ROI using GPU."""
        x, y, w, h = roi
        try:
            if isinstance(frame, cv2.UMat):
                # GPU cropping using cv2.UMat compatible method
                # Create ROI rectangle
                roi_rect = (x, y, w, h)
                # Use cv2.UMat's ROI functionality
                cropped = cv2.UMat(frame, roi_rect)
                return cropped
            else:
                # CPU cropping
                return frame[y:y+h, x:x+w]
        except Exception as e:
            logger.error(f"ROI cropping failed: {e}")
            # If GPU cropping fails, try CPU fallback
            try:
                if isinstance(frame, cv2.UMat):
                    frame_cpu = frame.get()
                    cropped_cpu = frame_cpu[y:y+h, x:x+w]
                    return cv2.UMat(cropped_cpu)
                else:
                    return frame[y:y+h, x:x+w]
            except Exception as fallback_e:
                logger.error(f"ROI cropping fallback also failed: {fallback_e}")
                return frame
    
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