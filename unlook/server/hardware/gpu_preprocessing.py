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
    roi_detection_method: str = "reference_based"  # reference_based, edge_based
    roi_margin: int = 50  # Pixel margin around detected ROI
    adaptive_quality: bool = True  # Enable adaptive quality levels
    edge_density_threshold: float = 0.1  # Threshold for edge density analysis
    max_downsampling_factor: float = 0.5  # Maximum downsampling (0.5 = half resolution)


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
        
        # Reference patterns for ROI detection
        self.reference_patterns = {
            'white': {},  # Store white reference per camera
            'black': {}   # Store black reference per camera
        }
        
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
            
            # 4. ROI detection and cropping
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
    
    def store_reference_pattern(self, frame, camera_id: str, pattern_type: str):
        """Store reference pattern (white/black) for improved ROI detection."""
        if pattern_type in ['white', 'black']:
            self.reference_patterns[pattern_type][camera_id] = frame
            logger.info(f"Stored {pattern_type} reference pattern for camera {camera_id}")
    
    def _detect_roi_gpu(self, frame, camera_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Detect region of interest using GPU acceleration."""
        # Check cache first
        if camera_id in self.roi_cache:
            return self.roi_cache[camera_id]
        
        try:
            if self.config.roi_detection_method == "reference_based":
                # Use reference patterns if available
                if camera_id in self.reference_patterns['white'] and camera_id in self.reference_patterns['black']:
                    roi = self._detect_roi_reference_based(
                        self.reference_patterns['white'][camera_id],
                        self.reference_patterns['black'][camera_id],
                        camera_id
                    )
                    if roi:
                        return roi
            
            # Fallback to edge-based detection
            return self._detect_roi_edge_based(frame, camera_id)
            
        except Exception as e:
            logger.error(f"ROI detection failed: {e}")
        
        return None
    
    def _detect_roi_reference_based(self, white_ref, black_ref, camera_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Detect ROI using white and black reference patterns."""
        try:
            # Convert to grayscale using maximum channel response (color-agnostic)
            # This works for blue->red shift in IR cameras and future VCSEL patterns
            if isinstance(white_ref, cv2.UMat):
                # For GPU UMat, convert to CPU first
                white_cpu = white_ref.get()
                black_cpu = black_ref.get()
                if len(white_cpu.shape) == 3:
                    white_gray = np.max(white_cpu, axis=2).astype(np.uint8)
                    black_gray = np.max(black_cpu, axis=2).astype(np.uint8)
                else:
                    white_gray = white_cpu
                    black_gray = black_cpu
                # Convert back to GPU if available
                if self.gpu_available:
                    white_gray = cv2.UMat(white_gray)
                    black_gray = cv2.UMat(black_gray)
            else:
                # CPU path
                if len(white_ref.shape) == 3:
                    white_gray = np.max(white_ref, axis=2).astype(np.uint8)
                    black_gray = np.max(black_ref, axis=2).astype(np.uint8)
                else:
                    white_gray = white_ref
                    black_gray = black_ref
            
            # Calculate difference image
            if self.gpu_available and isinstance(white_gray, cv2.UMat):
                diff = cv2.subtract(white_gray, black_gray)
            else:
                diff = cv2.subtract(white_gray.get() if isinstance(white_gray, cv2.UMat) else white_gray,
                                   black_gray.get() if isinstance(black_gray, cv2.UMat) else black_gray)
            
            # Apply adaptive threshold to identify bright regions (object area)
            # Use adaptive threshold based on image statistics for robustness
            if self.gpu_available and isinstance(diff, cv2.UMat):
                diff_cpu = diff.get()
            else:
                diff_cpu = diff
            
            # Calculate threshold dynamically based on image statistics
            mean_val = np.mean(diff_cpu)
            std_val = np.std(diff_cpu)
            threshold_val = max(20, min(100, mean_val + 0.5 * std_val))
            
            if self.gpu_available:
                _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
            else:
                _, mask = cv2.threshold(diff_cpu, threshold_val, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            if self.gpu_available and isinstance(mask, cv2.UMat):
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            else:
                mask_cpu = mask.get() if isinstance(mask, cv2.UMat) else mask
                mask = cv2.morphologyEx(mask_cpu, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            mask_cpu = mask.get() if isinstance(mask, cv2.UMat) else mask
            contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (main object)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add margin
                margin = self.config.roi_margin
                shape = white_ref.get().shape if isinstance(white_ref, cv2.UMat) else white_ref.shape
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(shape[1] - x, w + 2 * margin)
                h = min(shape[0] - y, h + 2 * margin)
                
                roi = (x, y, w, h)
                self.roi_cache[camera_id] = roi
                logger.info(f"Reference-based ROI detected for {camera_id}: {roi}")
                return roi
                
        except Exception as e:
            logger.error(f"Reference-based ROI detection failed: {e}")
        
        return None
    
    def _detect_roi_edge_based(self, frame, camera_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Fallback edge-based ROI detection."""
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
                    margin = self.config.roi_margin
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.get().shape[1] if isinstance(frame, cv2.UMat) else frame.shape[1] - x, w + 2 * margin)
                    h = min(frame.get().shape[0] if isinstance(frame, cv2.UMat) else frame.shape[0] - y, h + 2 * margin)
                    
                    roi = (x, y, w, h)
                    self.roi_cache[camera_id] = roi
                    return roi
            
        except Exception as e:
            logger.error(f"Edge-based ROI detection failed: {e}")
        
        return None
    
    def _crop_roi_gpu(self, frame, roi: Tuple[int, int, int, int]):
        """Mask frame to ROI using GPU, keeping original dimensions."""
        x, y, w, h = roi
        try:
            # Expand ROI by a margin (e.g., 20 pixels) for smoother transitions
            margin = 20
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            
            if isinstance(frame, cv2.UMat):
                # Get frame dimensions
                frame_cpu = frame.get()
                height, width = frame_cpu.shape[:2]
                x_end = min(width, x + w + margin)
                y_end = min(height, y + h + margin)
                
                # Create mask on GPU
                mask_cpu = np.zeros((height, width), dtype=np.uint8)
                mask = cv2.UMat(mask_cpu)
                # Fill ROI area with white (255)
                cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
                
                # Apply Gaussian blur to mask for smooth edges
                mask = cv2.GaussianBlur(mask, (margin*2+1, margin*2+1), margin/2)
                
                # Convert mask to 3-channel if needed
                if len(frame_cpu.shape) == 3:
                    mask_3ch = cv2.merge([mask, mask, mask])
                else:
                    mask_3ch = mask
                
                # Apply mask to frame
                masked_frame = cv2.multiply(frame, mask_3ch, scale=1.0/255.0)
                return masked_frame
            else:
                # CPU masking
                height, width = frame.shape[:2]
                x_end = min(width, x + w + margin)
                y_end = min(height, y + h + margin)
                
                # Create mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y_start:y_end, x_start:x_end] = 255
                
                # Apply Gaussian blur for smooth edges
                mask = cv2.GaussianBlur(mask, (margin*2+1, margin*2+1), margin/2)
                
                # Apply mask
                if len(frame.shape) == 3:
                    mask_3ch = np.stack([mask, mask, mask], axis=2)
                    masked_frame = (frame * (mask_3ch / 255.0)).astype(frame.dtype)
                else:
                    masked_frame = (frame * (mask / 255.0)).astype(frame.dtype)
                
                return masked_frame
        except Exception as e:
            logger.error(f"ROI masking failed: {e}")
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