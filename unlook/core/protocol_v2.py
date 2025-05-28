"""
Optimized protocol v2 with delta encoding and adaptive compression.
"""

import json
import time
import zlib
import struct
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics for compression performance."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    method: str


class ProtocolOptimizer:
    """
    Optimizes ZMQ protocol with delta encoding and adaptive compression.
    """
    
    def __init__(self, max_frame_history: int = 10):
        """
        Initialize protocol optimizer.
        
        Args:
            max_frame_history: Maximum frames to keep for delta encoding
        """
        self.frame_history = {}  # camera_id -> deque of frames
        self.max_frame_history = max_frame_history
        self.compression_stats = deque(maxlen=100)
        
        # Adaptive compression thresholds
        self.quality_thresholds = {
            'high_motion': 0.3,     # High motion threshold
            'low_motion': 0.1,      # Low motion threshold
            'static': 0.05          # Nearly static threshold
        }
        
        # Compression level mapping
        self.compression_levels = {
            'static': 9,            # Maximum compression for static
            'low_motion': 6,        # Good compression
            'normal': 3,            # Fast compression
            'high_motion': 1        # Minimal compression
        }
    
    def optimize_message(self, msg_type: str, metadata: Dict[str, Any], 
                        binary_data: Optional[bytes] = None, 
                        camera_id: Optional[str] = None) -> Tuple[bytes, CompressionStats]:
        """
        Optimize a message for transmission.
        
        Args:
            msg_type: Message type
            metadata: Message metadata
            binary_data: Optional binary payload (e.g., image data)
            camera_id: Camera ID for delta encoding
            
        Returns:
            Tuple of (optimized message bytes, compression stats)
        """
        start_time = time.time()
        original_size = len(binary_data) if binary_data else 0
        
        # Determine if we should use delta encoding
        if binary_data and camera_id and msg_type in ['camera_frame', 'direct_frame']:
            optimized_data, method = self._apply_delta_encoding(
                binary_data, camera_id, metadata
            )
        else:
            optimized_data = binary_data
            method = 'none'
        
        # Apply adaptive compression
        if optimized_data and len(optimized_data) > 1024:  # Only compress if > 1KB
            compressed_data, compression_level = self._apply_adaptive_compression(
                optimized_data, metadata
            )
        else:
            compressed_data = optimized_data
            compression_level = 0
        
        # Create optimized message
        optimized_metadata = metadata.copy()
        optimized_metadata['optimization'] = {
            'delta_encoded': method != 'none',
            'delta_method': method,
            'compression_level': compression_level,
            'original_size': original_size
        }
        
        # Serialize with optimized format
        message = self._serialize_optimized(msg_type, optimized_metadata, compressed_data)
        
        # Calculate stats
        compression_time = (time.time() - start_time) * 1000
        compressed_size = len(message)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time,
            method=f"{method}+compression_level_{compression_level}"
        )
        
        self.compression_stats.append(stats)
        
        return message, stats
    
    def optimize_multi_camera_message(self, camera_data: Dict[str, bytes], metadata: Dict[str, Any]) -> Tuple[bytes, CompressionStats]:
        """
        Optimize multi-camera message for transmission.
        
        Args:
            camera_data: Dictionary of camera_id -> image_data
            metadata: Message metadata
            
        Returns:
            Tuple of (optimized message bytes, compression stats)
        """
        start_time = time.time()
        total_original_size = sum(len(data) for data in camera_data.values())
        
        # Create multi-camera optimized metadata
        optimized_metadata = metadata.copy()
        optimized_metadata['cameras'] = {}
        optimized_metadata['optimization'] = {
            'multi_camera': True,
            'num_cameras': len(camera_data),
            'compression_enabled': True,
            'original_total_size': total_original_size
        }
        
        # Optimize each camera's data
        optimized_data = bytearray()
        camera_metadata = {}
        
        for camera_id, image_data in camera_data.items():
            # Apply compression to image data
            if len(image_data) > 1024:  # Only compress if > 1KB
                compressed_data, compression_level = self._apply_adaptive_compression(image_data, metadata)
            else:
                compressed_data = image_data
                compression_level = 0
            
            # Store camera metadata
            camera_metadata[camera_id] = {
                'offset': len(optimized_data),
                'size': len(compressed_data),
                'original_size': len(image_data),
                'compression_level': compression_level
            }
            
            # Append compressed data
            optimized_data.extend(compressed_data)
        
        optimized_metadata['cameras'] = camera_metadata
        
        # Serialize with V2 format
        message = self._serialize_optimized('multi_camera_response', optimized_metadata, bytes(optimized_data))
        
        # Calculate stats
        compression_time = (time.time() - start_time) * 1000
        compressed_size = len(message)
        compression_ratio = total_original_size / compressed_size if compressed_size > 0 else 0
        
        stats = CompressionStats(
            original_size=total_original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time,
            method=f"multi_camera_v2+compression"
        )
        
        self.compression_stats.append(stats)
        
        return message, stats
    
    def _apply_delta_encoding(self, data: bytes, camera_id: str, 
                             metadata: Dict) -> Tuple[bytes, str]:
        """Apply delta encoding if beneficial."""
        # Initialize history for this camera if needed
        if camera_id not in self.frame_history:
            self.frame_history[camera_id] = deque(maxlen=self.max_frame_history)
        
        history = self.frame_history[camera_id]
        
        # Can't do delta without history
        if not history:
            history.append(data)
            return data, 'none'
        
        # Try different delta strategies
        best_delta = None
        best_size = len(data)
        best_method = 'none'
        
        # 1. Simple frame difference
        prev_frame = history[-1]
        if len(prev_frame) == len(data):
            delta = self._compute_byte_delta(data, prev_frame)
            if len(delta) < best_size * 0.8:  # Only use if 20% smaller
                best_delta = delta
                best_size = len(delta)
                best_method = 'frame_diff'
        
        # 2. Key frame detection (every N frames)
        frame_num = metadata.get('frame_number', 0)
        if frame_num % 30 == 0:  # Key frame every 30 frames
            # Force full frame
            history.append(data)
            return data, 'keyframe'
        
        # Store current frame
        history.append(data)
        
        if best_delta is not None:
            return best_delta, best_method
        else:
            return data, 'none'
    
    def _compute_byte_delta(self, current: bytes, previous: bytes) -> bytes:
        """Compute byte-level delta between frames."""
        # Convert to numpy for efficient operations
        curr_array = np.frombuffer(current, dtype=np.uint8)
        prev_array = np.frombuffer(previous, dtype=np.uint8)
        
        # Compute difference
        diff = curr_array.astype(np.int16) - prev_array.astype(np.int16)
        
        # Pack as int8 with overflow handling
        diff_packed = np.clip(diff, -128, 127).astype(np.int8)
        
        # Run-length encode zeros
        rle_data = self._run_length_encode(diff_packed)
        
        return rle_data.tobytes()
    
    def _run_length_encode(self, data: np.ndarray) -> np.ndarray:
        """Simple run-length encoding for zeros."""
        # This is a simplified version - in production, use a proper RLE
        # For now, just return the data
        return data
    
    def _apply_adaptive_compression(self, data: bytes, 
                                   metadata: Dict) -> Tuple[bytes, int]:
        """Apply adaptive compression based on content."""
        # Determine motion level from metadata or analyze data
        motion_level = self._estimate_motion_level(data, metadata)
        
        # Select compression level
        if motion_level < self.quality_thresholds['static']:
            level = self.compression_levels['static']
        elif motion_level < self.quality_thresholds['low_motion']:
            level = self.compression_levels['low_motion']
        elif motion_level < self.quality_thresholds['high_motion']:
            level = self.compression_levels['normal']
        else:
            level = self.compression_levels['high_motion']
        
        # Apply zlib compression
        compressed = zlib.compress(data, level=level)
        
        # Only use compressed if smaller
        if len(compressed) < len(data) * 0.95:  # 5% threshold
            return compressed, level
        else:
            return data, 0
    
    def _estimate_motion_level(self, data: bytes, metadata: Dict) -> float:
        """Estimate motion level from data or metadata."""
        # Check if preprocessing provided quality metrics
        preprocessing_info = metadata.get('preprocessing_info', {})
        if 'quality_metrics' in preprocessing_info:
            # Use contrast as proxy for motion
            contrast = preprocessing_info['quality_metrics'].get('contrast', 50)
            return contrast / 255.0
        
        # Simple heuristic: check data entropy
        # High entropy = high motion/detail
        data_sample = data[:1000]  # Sample first 1KB
        unique_bytes = len(set(data_sample))
        entropy = unique_bytes / 256.0
        
        return entropy
    
    def _serialize_optimized(self, msg_type: str, metadata: Dict, 
                           data: Optional[bytes]) -> bytes:
        """Serialize with optimized format."""
        # Create header with optimization info
        header = {
            'type': msg_type,
            'metadata': metadata,
            'timestamp': time.time(),
            'version': 2  # Protocol v2
        }
        
        # Serialize header
        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
        header_size = len(header_json)
        
        # Build message
        # Format: [header_size:4][header:N][data_size:4][data:M]
        message = bytearray()
        message.extend(struct.pack('<I', header_size))
        message.extend(header_json)
        
        if data:
            message.extend(struct.pack('<I', len(data)))
            message.extend(data)
        else:
            message.extend(struct.pack('<I', 0))
        
        return bytes(message)
    
    def deserialize_optimized(self, message: bytes) -> Tuple[str, Dict, Optional[bytes]]:
        """Deserialize optimized message."""
        try:
            # Read header size
            header_size = struct.unpack('<I', message[:4])[0]
            
            # Read header
            header_json = message[4:4+header_size]
            header = json.loads(header_json.decode('utf-8'))
            
            # Read data if present
            data_start = 4 + header_size
            if len(message) > data_start + 4:
                data_size = struct.unpack('<I', message[data_start:data_start+4])[0]
                if data_size > 0:
                    data = message[data_start+4:data_start+4+data_size]
                    
                    # Apply reverse transformations if needed
                    metadata = header.get('metadata', {})
                    optimization = metadata.get('optimization', {})
                    
                    # Handle multi-camera format
                    if optimization.get('multi_camera', False):
                        data = self._deserialize_multi_camera_data(data, metadata)
                    elif optimization.get('compression_level', 0) > 0:
                        data = zlib.decompress(data)
                    
                    # Delta decoding would happen on client side
                else:
                    data = None
            else:
                data = None
            
            # Return type, metadata (or entire header as fallback), data
            msg_type = header.get('type', 'unknown_message_type')
            payload = header.get('metadata', header.get('payload', {}))
            
            return msg_type, payload, data
            
        except Exception as e:
            logger.error(f"Error deserializing optimized message: {e}")
            return 'error', {'error': str(e)}, None
    
    def _deserialize_multi_camera_data(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Deserialize multi-camera data from V2 format.
        
        Args:
            data: Compressed multi-camera data
            metadata: Message metadata with camera information
            
        Returns:
            Dictionary of camera_id -> decompressed_image_data
        """
        try:
            cameras_info = metadata.get('cameras', {})
            camera_images = {}
            
            for camera_id, camera_meta in cameras_info.items():
                offset = camera_meta.get('offset', 0)
                size = camera_meta.get('size', 0)
                compression_level = camera_meta.get('compression_level', 0)
                
                if offset + size <= len(data):
                    # Extract camera data
                    camera_data = data[offset:offset + size]
                    
                    # Decompress if needed
                    if compression_level > 0:
                        try:
                            camera_data = zlib.decompress(camera_data)
                        except Exception as e:
                            logger.warning(f"Failed to decompress camera {camera_id} data: {e}")
                    
                    camera_images[camera_id] = camera_data
                else:
                    logger.error(f"Invalid offset/size for camera {camera_id}: offset={offset}, size={size}, data_len={len(data)}")
            
            return camera_images
            
        except Exception as e:
            logger.error(f"Error deserializing multi-camera data: {e}")
            return {}
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression performance statistics."""
        if not self.compression_stats:
            return {}
            
        stats_list = list(self.compression_stats)
        
        return {
            'avg_compression_ratio': np.mean([s.compression_ratio for s in stats_list]),
            'avg_compression_time_ms': np.mean([s.compression_time_ms for s in stats_list]),
            'total_original_mb': sum(s.original_size for s in stats_list) / 1024 / 1024,
            'total_compressed_mb': sum(s.compressed_size for s in stats_list) / 1024 / 1024,
            'bandwidth_savings_percent': (1 - sum(s.compressed_size for s in stats_list) / 
                                         sum(s.original_size for s in stats_list)) * 100
        }