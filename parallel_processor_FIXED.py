#!/usr/bin/env python3
"""
Parallel Stereo Processor - FIXED VERSION
Risolve il problema del freeze su Windows con OpenCV e multiprocessing

MODIFICHE PRINCIPALI:
1. Worker carica immagini internamente (no pickle di numpy arrays)
2. Spawn method esplicito per Windows
3. Limiti conservativi su workers e memoria
4. Timeout e fallback automatici
5. Better error handling
"""

import os
import sys
import time
import logging
import platform
import subprocess
import re
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count, Manager, set_start_method
from threading import Lock
import gc
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import psutil

# Set spawn method explicitly for all platforms
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

logger = logging.getLogger(__name__)

class CPUProfiler:
    """
    CPU detection with CONSERVATIVE settings to avoid memory issues
    """
    
    def __init__(self):
        """Initialize CPU profiler with system detection."""
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        self.platform_info = self._detect_platform()
        self.performance_profile = self._create_performance_profile()
        
        logger.info(f"🖥️ CPU DETECTION COMPLETED:")
        logger.info(f"  Processor: {self.cpu_info['brand']} ({self.cpu_info['architecture']})")
        logger.info(f"  Cores: {self.cpu_info['cores']} physical, {self.cpu_info['threads']} logical")
        logger.info(f"  Memory: {self.memory_info['total_gb']:.1f}GB total, {self.memory_info['available_gb']:.1f}GB available")
        logger.info(f"  Performance tier: {self.performance_profile['tier']}")
        logger.info(f"  Optimized workers: {self.performance_profile['recommended_workers']}")
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information and capabilities."""
        cpu_info = {
            'cores': cpu_count() or 4,
            'threads': cpu_count() or 4,
            'brand': platform.processor() or 'Unknown',
            'architecture': platform.machine(),
            'frequency_mhz': 0,
            'features': [],
            'tier': 'unknown'
        }
        
        try:
            if platform.system() == "Windows":
                cpu_info.update(self._detect_windows_cpu())
            elif platform.system() == "Linux":
                cpu_info.update(self._detect_linux_cpu())
            elif platform.system() == "Darwin":
                cpu_info.update(self._detect_macos_cpu())
        except Exception as e:
            logger.warning(f"Detailed CPU detection failed: {e}")
        
        cpu_info['tier'] = self._classify_cpu_tier(cpu_info)
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory information."""
        memory_info = {
            'total_gb': 8.0,
            'available_gb': 4.0
        }
        
        try:
            vm = psutil.virtual_memory()
            memory_info['total_gb'] = vm.total / (1024**3)
            memory_info['available_gb'] = vm.available / (1024**3)
        except Exception as e:
            logger.warning(f"Memory detection failed: {e}")
        
        return memory_info
    
    def _detect_windows_cpu(self) -> Dict[str, Any]:
        """Detect CPU info on Windows."""
        info = {}
        try:
            result = subprocess.run([
                'wmic', 'cpu', 'get', 'Name,NumberOfCores,NumberOfLogicalProcessors', '/format:csv'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:
                    if line.strip() and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            info['brand'] = parts[1].strip()
                            info['cores'] = int(parts[2]) if parts[2].isdigit() else cpu_count()
                            info['threads'] = int(parts[3]) if parts[3].isdigit() else cpu_count()
                            break
        except Exception as e:
            logger.debug(f"Windows CPU detection failed: {e}")
        
        return info
    
    def _detect_linux_cpu(self) -> Dict[str, Any]:
        """Detect CPU info on Linux."""
        info = {}
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            for line in cpuinfo.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'model name' in key and not info.get('brand'):
                        info['brand'] = value
                    elif 'cpu cores' in key:
                        try:
                            info['cores'] = int(value)
                        except:
                            pass
        except Exception as e:
            logger.debug(f"Linux CPU detection failed: {e}")
        
        return info
    
    def _detect_macos_cpu(self) -> Dict[str, Any]:
        """Detect CPU info on macOS."""
        info = {}
        try:
            commands = [
                ('hw.ncpu', 'threads'),
                ('hw.physicalcpu', 'cores'),
                ('machdep.cpu.brand_string', 'brand')
            ]
            
            for cmd, key in commands:
                try:
                    result = subprocess.run(['sysctl', '-n', cmd], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        value = result.stdout.strip()
                        if key in ['threads', 'cores']:
                            info[key] = int(value)
                        else:
                            info[key] = value
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"macOS CPU detection failed: {e}")
        
        return info
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform information."""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'python_version': platform.python_version()
        }
    
    def _classify_cpu_tier(self, cpu_info: Dict[str, Any]) -> str:
        """Classify CPU into performance tiers."""
        brand = cpu_info['brand'].lower()
        cores = cpu_info['cores']
        arch = cpu_info['architecture'].lower()
        
        if any(x in brand for x in ['i9', 'i7-12', 'i7-13', 'i7-14', 'ryzen 9', 'ryzen 7']):
            if cores >= 8:
                return 'high_end_desktop'
        
        if any(x in arch for x in ['arm', 'aarch64']) or any(x in brand for x in ['apple m', 'snapdragon']):
            return 'arm_mobile'
        
        if cores >= 6:
            return 'medium_desktop'
        
        if cores <= 4:
            return 'low_end_mobile'
        
        return 'medium_desktop'
    
    def _create_performance_profile(self) -> Dict[str, Any]:
        """Create CONSERVATIVE performance profile to avoid memory issues."""
        tier = self.cpu_info['tier']
        cores = self.cpu_info['cores']
        memory_gb = self.memory_info['available_gb']
        
        # CONSERVATIVE profiles to avoid freeze
        profiles = {
            'high_end_desktop': {
                'tier': 'High-End Desktop (i9/Ryzen 9)',
                'recommended_workers': min(cores // 2, 4),  # MAX 4 workers
                'batch_size': 2,  # Small batches
                'memory_per_worker_gb': 2.0,  # Conservative estimate
                'enable_hyperthreading': False,  # Disable for stability
                'opencv_threads': 2,  # Limit OpenCV threads
                'io_threads': 4,
                'aggressive_optimization': False  # Conservative mode
            },
            'medium_desktop': {
                'tier': 'Medium Desktop',
                'recommended_workers': min(cores // 2, 3),
                'batch_size': 2,
                'memory_per_worker_gb': 1.5,
                'enable_hyperthreading': False,
                'opencv_threads': 2,
                'io_threads': 3,
                'aggressive_optimization': False
            },
            'arm_mobile': {
                'tier': 'ARM Mobile/Laptop',
                'recommended_workers': min(cores // 2, 2),
                'batch_size': 1,
                'memory_per_worker_gb': 1.0,
                'enable_hyperthreading': False,
                'opencv_threads': 1,
                'io_threads': 2,
                'aggressive_optimization': False
            },
            'low_end_mobile': {
                'tier': 'Low-End Mobile',
                'recommended_workers': 1,
                'batch_size': 1,
                'memory_per_worker_gb': 0.8,
                'enable_hyperthreading': False,
                'opencv_threads': 1,
                'io_threads': 1,
                'aggressive_optimization': False
            }
        }
        
        profile = profiles.get(tier, profiles['medium_desktop']).copy()
        
        # Further limit based on available memory
        max_workers_by_memory = max(1, int(memory_gb / profile['memory_per_worker_gb']))
        profile['recommended_workers'] = min(profile['recommended_workers'], max_workers_by_memory)
        
        # Extra safety: never use more than 50% of available memory
        if memory_gb < 8.0:
            profile['recommended_workers'] = min(profile['recommended_workers'], 2)
            profile['batch_size'] = 1
        
        return profile

@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None
    batch_size: int = 2
    memory_limit_gb: float = 8.0
    enable_threading: bool = True
    enable_multiprocessing: bool = True
    progress_bar: bool = True
    cleanup_memory: bool = True
    timeout_seconds: int = 300  # 5 minutes timeout per batch

class ParallelStereoProcessor:
    """
    FIXED parallel processor that avoids OpenCV/multiprocessing conflicts
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None, auto_optimize: bool = True):
        self.cpu_profiler = CPUProfiler() if auto_optimize else None
        
        if config is None and auto_optimize:
            self.config = self._get_safe_config()
        else:
            self.config = config or ProcessingConfig()
        
        if self.config.max_workers is None:
            self.config.max_workers = 2  # Safe default
        
        # Apply CPU optimizations
        if auto_optimize and self.cpu_profiler:
            self._apply_cpu_optimizations()
        
        self.stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'average_frame_time': 0,
            'errors': 0
        }
        
        logger.info(f"🛡️ SAFE PARALLEL PROCESSOR INITIALIZED")
        logger.info(f"  Max workers: {self.config.max_workers}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Timeout: {self.config.timeout_seconds}s per batch")
    
    def _get_safe_config(self) -> ProcessingConfig:
        """Get safe configuration based on system resources."""
        profile = self.cpu_profiler.performance_profile
        
        return ProcessingConfig(
            max_workers=profile['recommended_workers'],
            batch_size=profile['batch_size'],
            memory_limit_gb=self.cpu_profiler.memory_info['available_gb'] * 0.5,
            enable_threading=True,
            enable_multiprocessing=True,
            progress_bar=True,
            cleanup_memory=True,
            timeout_seconds=300
        )
    
    def _apply_cpu_optimizations(self):
        """Apply conservative CPU optimizations."""
        profile = self.cpu_profiler.performance_profile
        
        # Limit OpenCV threads to avoid conflicts
        cv2.setNumThreads(profile['opencv_threads'])
        
        # Set environment variables
        os.environ['OMP_NUM_THREADS'] = str(profile['opencv_threads'])
        os.environ['OPENBLAS_NUM_THREADS'] = str(profile['opencv_threads'])
        os.environ['MKL_NUM_THREADS'] = str(profile['opencv_threads'])
        
        logger.info(f"Applied conservative optimizations:")
        logger.info(f"  OpenCV threads: {profile['opencv_threads']}")
    
    def process_stereo_frames_parallel(self, stereo_pairs: List[Tuple[str, str]], 
                                     reconstructor, progress_callback=None) -> List[Tuple[np.ndarray, Dict]]:
        """
        FIXED: Process frames by passing paths instead of numpy arrays
        """
        if not self.config.enable_multiprocessing or len(stereo_pairs) <= 1:
            logger.info("Using sequential processing")
            return self._process_sequential(stereo_pairs, reconstructor, progress_callback)
        
        logger.info(f"🚀 STARTING SAFE PARALLEL PROCESSING: {len(stereo_pairs)} frames")
        logger.info(f"⚙️ Configuration: {self.config.max_workers} workers, batch size {self.config.batch_size}")
        
        # Check memory before starting
        if not self._check_memory_available():
            logger.warning("Insufficient memory - falling back to sequential")
            return self._process_sequential(stereo_pairs, reconstructor, progress_callback)
        
        start_time = time.time()
        all_results = []
        
        try:
            # Process in small batches
            for batch_idx in range(0, len(stereo_pairs), self.config.batch_size):
                batch_pairs = stereo_pairs[batch_idx:batch_idx + self.config.batch_size]
                
                logger.info(f"📦 Processing batch {batch_idx//self.config.batch_size + 1}")
                
                # Process batch with timeout
                try:
                    batch_results = self._process_batch_safe(
                        batch_pairs, reconstructor, batch_idx, progress_callback
                    )
                    all_results.extend(batch_results)
                except TimeoutError:
                    logger.error("Batch processing timeout - falling back to sequential")
                    # Process remaining frames sequentially
                    remaining_pairs = stereo_pairs[batch_idx:]
                    seq_results = self._process_sequential(remaining_pairs, reconstructor, progress_callback)
                    all_results.extend(seq_results)
                    break
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Try sequential for this batch
                    seq_results = self._process_sequential(batch_pairs, reconstructor, progress_callback)
                    all_results.extend(seq_results)
                
                # Memory cleanup
                if self.config.cleanup_memory:
                    gc.collect()
                
                # Check if we should continue
                if self._memory_pressure_high():
                    logger.warning("Memory pressure high - switching to sequential")
                    remaining_pairs = stereo_pairs[batch_idx + self.config.batch_size:]
                    if remaining_pairs:
                        seq_results = self._process_sequential(remaining_pairs, reconstructor, progress_callback)
                        all_results.extend(seq_results)
                    break
        
        except Exception as e:
            logger.error(f"Parallel processing failed completely: {e}")
            # Full fallback to sequential
            return self._process_sequential(stereo_pairs, reconstructor, progress_callback)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Processing completed in {total_time:.2f}s")
        logger.info(f"📊 Success rate: {len(all_results)}/{len(stereo_pairs)}")
        
        return all_results
    
    def _process_batch_safe(self, batch_pairs: List[Tuple[str, str]], 
                           reconstructor, batch_offset: int,
                           progress_callback) -> List[Tuple[np.ndarray, Dict]]:
        """
        SAFE batch processing - passes paths, not numpy arrays
        """
        # Prepare work items (paths only)
        work_items = []
        for idx, (left_path, right_path) in enumerate(batch_pairs):
            work_items.append({
                'left_path': left_path,
                'right_path': right_path,
                'calibration': reconstructor.calibration if hasattr(reconstructor, 'calibration') else None,
                'config': {
                    'use_advanced_disparity': getattr(reconstructor, 'use_advanced_disparity', False),
                    'use_ndr': getattr(reconstructor, 'use_ndr', False),
                    'use_phase_optimization': getattr(reconstructor, 'use_phase_optimization', False),
                    'use_cgal': getattr(reconstructor, 'use_cgal', False)
                }
            })
        
        results = []
        
        # Use ProcessPoolExecutor with explicit max_tasks_per_child
        with ProcessPoolExecutor(max_workers=self.config.max_workers, 
                               max_tasks_per_child=1) as executor:  # Restart worker after each task
            
            # Submit all tasks
            futures = {}
            for idx, work_item in enumerate(work_items):
                future = executor.submit(
                    process_single_frame_worker_safe,
                    work_item['left_path'],
                    work_item['right_path'],
                    work_item['calibration'],
                    work_item['config']
                )
                futures[future] = idx
            
            # Collect results with timeout
            completed = 0
            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                idx = futures[future]
                
                try:
                    points_3d, quality = future.result(timeout=60)  # 1 minute timeout per frame
                    
                    if len(points_3d) > 100:
                        results.append((points_3d, quality))
                        status = f"✅ Frame {idx+1}: {len(points_3d):,} points"
                    else:
                        status = f"❌ Frame {idx+1}: Insufficient points"
                        
                except Exception as e:
                    logger.warning(f"Frame {idx+1} failed: {e}")
                    status = f"❌ Frame {idx+1}: Error"
                    self.stats['errors'] += 1
                
                completed += 1
                if progress_callback:
                    progress_callback(batch_offset + completed, status)
        
        return results
    
    def _process_sequential(self, stereo_pairs: List[Tuple[str, str]], 
                          reconstructor, progress_callback) -> List[Tuple[np.ndarray, Dict]]:
        """Safe sequential fallback."""
        results = []
        
        logger.info(f"📝 Sequential processing of {len(stereo_pairs)} frames")
        
        for i, (left_path, right_path) in enumerate(stereo_pairs):
            try:
                # Load images
                left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
                right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
                
                if left_img is not None and right_img is not None:
                    # Process
                    points_3d, quality = reconstructor.reconstruct_surface(left_img, right_img)
                    
                    if len(points_3d) > 100:
                        results.append((points_3d, quality))
                        status = f"✅ {len(points_3d):,} points"
                    else:
                        status = "❌ Insufficient points"
                else:
                    status = "❌ Failed to load images"
                    
                if progress_callback:
                    progress_callback(i + 1, f"Frame {i+1}/{len(stereo_pairs)}: {status}")
                    
            except Exception as e:
                logger.warning(f"Sequential processing error on frame {i+1}: {e}")
                if progress_callback:
                    progress_callback(i + 1, f"Frame {i+1}: ❌ Error")
        
        return results
    
    def _check_memory_available(self) -> bool:
        """Check if enough memory is available."""
        try:
            vm = psutil.virtual_memory()
            available_gb = vm.available / (1024**3)
            
            # Need at least 2GB per worker
            required_gb = self.config.max_workers * 2.0
            
            if available_gb < required_gb:
                logger.warning(f"Insufficient memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required")
                return False
            
            return True
        except:
            return True  # Assume OK if can't check
    
    def _memory_pressure_high(self) -> bool:
        """Check if memory pressure is high."""
        try:
            vm = psutil.virtual_memory()
            return vm.percent > 85  # High memory usage
        except:
            return False

# SAFE Worker function - loads images internally
def process_single_frame_worker_safe(left_path: str, right_path: str,
                                   calibration_dict: Optional[Dict],
                                   config: Dict) -> Tuple[np.ndarray, Dict]:
    """
    SAFE worker that loads images INSIDE the process
    Avoids pickle serialization of numpy arrays
    """
    try:
        # Set OpenCV threads for this worker
        cv2.setNumThreads(1)  # Minimal threads in worker
        
        # Load images in worker process
        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            return np.array([]), {'points': 0, 'quality_score': 0.0, 'description': 'Failed to load images'}
        
        # Import and create reconstructor in worker
        sys.path.insert(0, str(Path(__file__).parent))
        from stereobm_surface_reconstructor import StereoBMSurfaceReconstructor
        
        # Create reconstructor with config
        reconstructor = StereoBMSurfaceReconstructor(
            calibration_file=None,
            use_cgal=config.get('use_cgal', False),
            use_advanced_disparity=config.get('use_advanced_disparity', False),
            use_ndr=config.get('use_ndr', False),
            use_phase_optimization=config.get('use_phase_optimization', False)
        )
        
        # Set calibration if provided
        if calibration_dict:
            reconstructor.calibration = calibration_dict
            reconstructor._load_calibration_from_dict(calibration_dict)
        
        # Process frame
        points_3d, quality = reconstructor.reconstruct_surface(left_img, right_img)
        
        # Convert points to list for serialization (avoid large numpy arrays)
        if len(points_3d) > 0:
            # Downsample if too many points to avoid serialization issues
            if len(points_3d) > 50000:
                indices = np.random.choice(len(points_3d), 50000, replace=False)
                points_3d = points_3d[indices]
        
        return points_3d, quality
        
    except Exception as e:
        import traceback
        error_msg = f"Worker error: {str(e)}\n{traceback.format_exc()}"
        return np.array([]), {'points': 0, 'quality_score': 0.0, 'description': error_msg}

# Initialize on module load
if __name__ != '__main__':
    # Ensure spawn method is set when imported
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass