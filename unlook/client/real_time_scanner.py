"""
Real-time 3D scanner implementation for the UnLook system.
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import cv2
import numpy as np

from .scan_config import RealTimeScannerConfig, ScanningMode, PatternType, ScanningQuality
from ..core.events import EventType
from ..core.constants import DEFAULT_JPEG_QUALITY

logger = logging.getLogger(__name__)


class ScanFrameData:
    """Data container for a single frame in the scanning process."""
    
    def __init__(self, 
                image: np.ndarray, 
                camera_id: str,
                pattern_info: Dict[str, Any],
                timestamp: float,
                index: int,
                metadata: Dict[str, Any] = None):
        """
        Initialize frame data.
        
        Args:
            image: Captured image as numpy array
            camera_id: ID of the camera that captured the image
            pattern_info: Information about the pattern projected when capturing
            timestamp: Timestamp of the capture
            index: Index in the sequence
            metadata: Additional metadata
        """
        self.image = image
        self.camera_id = camera_id
        self.pattern_info = pattern_info
        self.timestamp = timestamp
        self.index = index
        self.metadata = metadata or {}


class ScanResult:
    """Container for 3D scanning results."""
    
    def __init__(self):
        """Initialize an empty scan result."""
        self.point_cloud = None  # Will be a numpy array of points
        self.confidence_map = None  # Confidence value for each point
        self.texture_map = None  # Color/texture information
        self.scan_time = 0.0  # Time taken for the scan
        self.frame_count = 0  # Number of frames processed
        self.metadata = {}  # Additional metadata
        
    def has_data(self) -> bool:
        """Check if the result contains valid point cloud data."""
        return self.point_cloud is not None and len(self.point_cloud) > 0


class RealTimeScanner:
    """
    Real-time 3D scanner for the UnLook system.
    Provides high-level API for performing real-time 3D scans.
    """
    
    def __init__(self, client):
        """
        Initialize the real-time scanner.
        
        Args:
            client: Main UnlookClient instance
        """
        self.client = client
        self.config = RealTimeScannerConfig()
        
        # Scanning state
        self.scanning = False
        self.scan_thread = None
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=100)  # Queue for captured frames
        self.result_queue = queue.Queue()  # Queue for processed results
        
        # Results
        self.latest_result = None
        self.all_results = []
        
        # Statistics
        self.frame_count = 0
        self.start_time = 0
        self.scan_duration = 0
        
        # Callbacks
        self.on_scan_progress_callback = None
        self.on_scan_completed_callback = None
        self.on_scan_error_callback = None
        self.on_frame_captured_callback = None
        self.on_result_ready_callback = None
        
        # Ensure output directory exists
        if not os.path.exists(self.config.output_directory):
            os.makedirs(self.config.output_directory, exist_ok=True)
    
    def configure(self, config: RealTimeScannerConfig) -> 'RealTimeScanner':
        """
        Configure the scanner with the provided settings.
        
        Args:
            config: Scanner configuration
            
        Returns:
            Self for method chaining
        """
        self.config = config
        
        # Ensure output directory exists
        if not os.path.exists(self.config.output_directory):
            os.makedirs(self.config.output_directory, exist_ok=True)
            
        return self
    
    def on_scan_progress(self, callback: Callable[[float, Dict[str, Any]], None]) -> 'RealTimeScanner':
        """
        Set callback for scan progress updates.
        
        Args:
            callback: Function called with progress info (progress percentage, metadata)
            
        Returns:
            Self for method chaining
        """
        self.on_scan_progress_callback = callback
        return self
    
    def on_scan_completed(self, callback: Callable[[ScanResult], None]) -> 'RealTimeScanner':
        """
        Set callback for scan completion.
        
        Args:
            callback: Function called with scan result
            
        Returns:
            Self for method chaining
        """
        self.on_scan_completed_callback = callback
        return self
    
    def on_scan_error(self, callback: Callable[[str, Exception], None]) -> 'RealTimeScanner':
        """
        Set callback for scan errors.
        
        Args:
            callback: Function called with error message and exception
            
        Returns:
            Self for method chaining
        """
        self.on_scan_error_callback = callback
        return self
    
    def on_frame_captured(self, callback: Callable[[ScanFrameData], None]) -> 'RealTimeScanner':
        """
        Set callback for frame captures.
        
        Args:
            callback: Function called with frame data
            
        Returns:
            Self for method chaining
        """
        self.on_frame_captured_callback = callback
        return self
    
    def on_result_ready(self, callback: Callable[[ScanResult], None]) -> 'RealTimeScanner':
        """
        Set callback for when a new result is ready (for continuous mode).
        
        Args:
            callback: Function called with scan result
            
        Returns:
            Self for method chaining
        """
        self.on_result_ready_callback = callback
        return self
    
    def start(self) -> bool:
        """
        Start the scanning process.
        
        Returns:
            True if scanning started successfully, False otherwise
        """
        if self.scanning:
            logger.warning("Scanner is already running")
            return False
        
        # Reset state
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
        # Set scanning flag
        self.scanning = True
        
        # Start processing thread
        if self.config.real_time_processing:
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
        
        # Start scanning thread
        self.scan_thread = threading.Thread(
            target=self._scan_loop,
            daemon=True
        )
        self.scan_thread.start()
        
        logger.info(f"Started 3D scanning in {self.config.mode.value} mode")
        return True
    
    def stop(self) -> bool:
        """
        Stop the scanning process.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.scanning:
            logger.warning("Scanner is not running")
            return False
        
        # Set flag to stop threads
        self.scanning = False
        
        # Wait for threads to terminate
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Update scan duration
        self.scan_duration = time.time() - self.start_time
        
        logger.info(f"Stopped 3D scanning after {self.scan_duration:.2f} seconds")
        
        # Process remaining frames if not already processed
        if not self.config.real_time_processing:
            self._process_captured_frames()
        
        return True
    
    def get_latest_result(self) -> Optional[ScanResult]:
        """
        Get the most recent scan result.
        
        Returns:
            Latest scan result or None if no scans completed
        """
        return self.latest_result
    
    def get_all_results(self) -> List[ScanResult]:
        """
        Get all scan results (for continuous mode).
        
        Returns:
            List of all scan results
        """
        return self.all_results
    
    def save_result(self, result: ScanResult = None, filepath: str = None) -> str:
        """
        Save scan result to disk.
        
        Args:
            result: Result to save (uses latest result if None)
            filepath: Path to save to (generates one if None)
            
        Returns:
            Path where the result was saved
        """
        result = result or self.latest_result
        
        if not result or not result.has_data():
            logger.warning("No valid result to save")
            return None
        
        # Generate filepath if not provided
        if not filepath:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config.output_directory, f"scan_{timestamp}.ply")
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the point cloud
        try:
            self._save_point_cloud(result.point_cloud, result.texture_map, filepath)
            logger.info(f"Saved scan result to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            if self.on_scan_error_callback:
                self.on_scan_error_callback(f"Error saving result: {e}", e)
            return None
    
    def _scan_loop(self):
        """Main scanning thread function."""
        logger.info("Scan thread started")
        
        try:
            # Configure cameras
            self._configure_cameras()
            
            # Create pattern sequence
            patterns = self._create_pattern_sequence()
            
            # Get camera IDs to use
            camera_ids = self._get_camera_ids()
            
            if not camera_ids:
                raise RuntimeError("No cameras available for scanning")
                
            logger.info(f"Using cameras: {camera_ids}")
            logger.info(f"Prepared pattern sequence with {len(patterns)} patterns")
            
            # Start pattern sequence
            self._start_pattern_sequence(patterns)
            
            # Main scanning loop
            start_time = time.time()
            sequence_count = 0
            
            while self.scanning:
                # Check scanning mode
                if self.config.mode == ScanningMode.SINGLE and sequence_count > 0:
                    logger.info("Single scan completed, stopping")
                    self.scanning = False
                    break
                
                logger.info(f"Starting scan sequence {sequence_count+1}")
                
                # Capture images synchronized with pattern sequence
                frames = self._capture_synchronized_frames(camera_ids, patterns)
                
                # Make sure we got some frames
                if not frames:
                    logger.warning("No frames captured in this sequence")
                    if self.config.mode == ScanningMode.SINGLE:
                        # For single mode, retry once
                        logger.info("Retrying capture for single mode...")
                        frames = self._capture_synchronized_frames(camera_ids, patterns)
                        if not frames:
                            logger.error("Failed to capture frames after retry, stopping")
                            break
                    else:
                        # For continuous mode, just continue to the next iteration
                        continue
                
                # Process captured frames
                self._handle_captured_frames(frames, sequence_count)
                
                # Update counters and progress
                sequence_count += 1
                progress = min(1.0, sequence_count / (2 if self.config.mode == ScanningMode.SINGLE else 10))
                
                # Report progress
                if self.on_scan_progress_callback:
                    self.on_scan_progress_callback(progress, {
                        "sequence_count": sequence_count,
                        "elapsed_time": time.time() - start_time,
                        "frame_count": self.frame_count
                    })
                
                # Add delay for continuous mode
                if self.config.mode == ScanningMode.CONTINUOUS:
                    time.sleep(0.1)  # Small delay between sequences
                    
                # For single mode, log that we're done
                if self.config.mode == ScanningMode.SINGLE:
                    logger.info("Single scan completed, stopping")
            
            # Finalize the scan
            self._finalize_scan()
            
            # Stop pattern sequence
            self._stop_pattern_sequence()
            
        except Exception as e:
            logger.error(f"Error in scan thread: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if self.on_scan_error_callback:
                self.on_scan_error_callback(f"Scan error: {e}", e)
        
        finally:
            # Ensure pattern sequence is stopped
            try:
                self._stop_pattern_sequence()
            except:
                pass
            
            # Mark scanning as stopped
            self.scanning = False
            
            logger.info("Scan thread terminated")
    
    def _processing_loop(self):
        """Thread function for processing captured frames."""
        logger.info("Processing thread started")
        
        try:
            frame_sets = {}  # Dictionary to collect complete sets of frames
            
            while self.scanning or not self.frame_queue.empty():
                try:
                    # Get frame from queue with timeout
                    frame_data = self.frame_queue.get(timeout=0.5)
                    
                    # Get set key (combines sequence and pattern index)
                    set_key = f"{frame_data.metadata.get('sequence_index', 0)}_{frame_data.index}"
                    
                    # Initialize set if needed
                    if set_key not in frame_sets:
                        frame_sets[set_key] = {}
                    
                    # Add frame to set
                    frame_sets[set_key][frame_data.camera_id] = frame_data
                    
                    # Check if we have a complete set
                    expected_count = len(self._get_camera_ids())
                    
                    if len(frame_sets[set_key]) == expected_count:
                        # Process complete set
                        result = self._process_frame_set(frame_sets[set_key])
                        
                        if result:
                            # Update latest result
                            self.latest_result = result
                            self.all_results.append(result)
                            
                            # Put in result queue
                            self.result_queue.put(result)
                            
                            # Call callback
                            if self.on_result_ready_callback:
                                self.on_result_ready_callback(result)
                        
                        # Remove processed set
                        del frame_sets[set_key]
                    
                    # Mark task as done
                    self.frame_queue.task_done()
                    
                except queue.Empty:
                    # No frames in queue, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if self.on_scan_error_callback:
                self.on_scan_error_callback(f"Processing error: {e}", e)
        
        finally:
            logger.info("Processing thread terminated")
    
    def _process_captured_frames(self):
        """Process all captured frames when not using real-time processing."""
        logger.info("Processing all captured frames")
        
        # Process frames from queue
        frames = []
        
        while not self.frame_queue.empty():
            try:
                frames.append(self.frame_queue.get(block=False))
                self.frame_queue.task_done()
            except queue.Empty:
                break
        
        logger.info(f"Retrieved {len(frames)} frames from queue for processing")
        
        # Group frames by set
        frame_sets = {}
        
        for frame in frames:
            set_key = f"{frame.metadata.get('sequence_index', 0)}_{frame.index}"
            
            if set_key not in frame_sets:
                frame_sets[set_key] = {}
            
            frame_sets[set_key][frame.camera_id] = frame
        
        logger.info(f"Grouped frames into {len(frame_sets)} sets")
        
        # Process each complete set
        results = []
        
        for set_key, frame_set in frame_sets.items():
            expected_count = len(self._get_camera_ids())
            
            if len(frame_set) == expected_count:
                logger.info(f"Processing frame set {set_key} with {expected_count} cameras")
                result = self._process_frame_set(frame_set)
                
                if result and result.has_data():
                    logger.info(f"Generated result with {len(result.point_cloud)} points")
                    results.append(result)
            else:
                logger.warning(f"Incomplete frame set {set_key}: has {len(frame_set)}/{expected_count} cameras")
        
        # Update results
        if results:
            self.latest_result = results[-1]
            self.all_results.extend(results)
            
            logger.info(f"Processed {len(results)} results with point clouds")
            
            # Call callback for final result
            if self.on_scan_completed_callback:
                self.on_scan_completed_callback(self.latest_result)
        else:
            logger.warning("No valid results generated during processing")
    
    def _configure_cameras(self):
        """Configure cameras for scanning."""
        # Get camera IDs
        camera_ids = self._get_camera_ids()
        
        # Configure each camera
        for camera_id in camera_ids:
            try:
                # Set exposure and gain
                self.client.camera.set_exposure(
                    camera_id,
                    exposure_time=self.config.exposure_time,
                    gain=self.config.gain
                )
                
                logger.info(f"Configured camera {camera_id} for scanning")
            except Exception as e:
                logger.warning(f"Error configuring camera {camera_id}: {e}")
    
    def _create_pattern_sequence(self) -> List[Dict[str, Any]]:
        """Create pattern sequence based on configuration."""
        # Use custom sequence if provided
        if self.config.custom_pattern_sequence:
            return self.config.custom_pattern_sequence
        
        # Create appropriate pattern sequence based on type
        patterns = []
        
        # Add white and black reference patterns at the start for calibration
        patterns.append({"pattern_type": "solid_field", "color": "White"})
        patterns.append({"pattern_type": "solid_field", "color": "Black"})
        
        if self.config.pattern_type == PatternType.PHASE_SHIFT:
            # Determine line widths based on quality setting
            line_widths = []
            if self.config.quality.value == "low":
                line_widths = [32, 16, 8, 4]
            elif self.config.quality.value == "medium":
                line_widths = [64, 32, 16, 8, 4]
            elif self.config.quality.value == "high":
                line_widths = [128, 64, 32, 16, 8, 4, 2]
            else:  # ultra
                line_widths = [128, 64, 32, 16, 8, 4, 2]
            
            logger.info(f"Using line widths: {line_widths} for quality {self.config.quality.value}")
            
            # Add horizontal binary patterns (each width is half the previous)
            for width in line_widths:
                patterns.append({
                    "pattern_type": "horizontal_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": width,
                    "background_width": width
                })
            
            # Add vertical binary patterns (each width is half the previous)
            for width in line_widths:
                patterns.append({
                    "pattern_type": "vertical_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": width,
                    "background_width": width
                })
            
            # Add phase-shifted patterns for the finest detail level
            finest_width = line_widths[-1]
            num_phases = 4
            
            # Add phase-shifted horizontal patterns
            for i in range(num_phases):
                shift = (i * finest_width * 2) // num_phases
                patterns.append({
                    "pattern_type": "horizontal_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": finest_width,
                    "background_width": finest_width,
                    "phase_shift": shift
                })
            
            # Add phase-shifted vertical patterns
            for i in range(num_phases):
                shift = (i * finest_width * 2) // num_phases
                patterns.append({
                    "pattern_type": "vertical_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": finest_width,
                    "background_width": finest_width,
                    "phase_shift": shift
                })
            
            # Add complementary patterns (inverse of the finest patterns)
            # This helps with handling reflective surfaces
            patterns.append({
                "pattern_type": "horizontal_lines",
                "foreground_color": "Black",
                "background_color": "White",
                "foreground_width": finest_width,
                "background_width": finest_width
            })
            
            patterns.append({
                "pattern_type": "vertical_lines",
                "foreground_color": "Black",
                "background_color": "White",
                "foreground_width": finest_width,
                "background_width": finest_width
            })
            
            # Add a grid pattern at the end for calibration
            patterns.append({
                "pattern_type": "grid",
                "foreground_color": "White",
                "background_color": "Black",
                "h_foreground_width": 4,
                "h_background_width": 16,
                "v_foreground_width": 4,
                "v_background_width": 16
            })
            
        elif self.config.pattern_type == PatternType.HORIZONTAL_LINES:
            # Create only horizontal line patterns with different widths
            # Use a wider range of widths for better 3D reconstruction
            line_widths = [128, 64, 32, 16, 8, 4, 2][:self.config.pattern_steps]
            
            for width in line_widths:
                patterns.append({
                    "pattern_type": "horizontal_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": width,
                    "background_width": width
                })
                
                # Add complementary pattern for each width
                if width <= 8:  # Only add complementary for finer patterns
                    patterns.append({
                        "pattern_type": "horizontal_lines",
                        "foreground_color": "Black",
                        "background_color": "White",
                        "foreground_width": width,
                        "background_width": width
                    })
        
        elif self.config.pattern_type == PatternType.VERTICAL_LINES:
            # Create only vertical line patterns with different widths
            line_widths = [128, 64, 32, 16, 8, 4, 2][:self.config.pattern_steps]
            
            for width in line_widths:
                patterns.append({
                    "pattern_type": "vertical_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": width,
                    "background_width": width
                })
                
                # Add complementary pattern for each width
                if width <= 8:  # Only add complementary for finer patterns
                    patterns.append({
                        "pattern_type": "vertical_lines",
                        "foreground_color": "Black",
                        "background_color": "White",
                        "foreground_width": width,
                        "background_width": width
                    })
        
        elif self.config.pattern_type == PatternType.GRAY_CODE:
            # Create Gray code pattern sequence
            # Binary subdivision with Gray code encoding (only one bit changes at a time)
            
            # Horizontal Gray code patterns (powers of 2 width)
            for i in range(8):  # 8 patterns for horizontal resolution
                stripe_width = max(2, 256 // (2**i))  # Don't go smaller than 2 pixels
                patterns.append({
                    "pattern_type": "horizontal_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": stripe_width,
                    "background_width": stripe_width
                })
            
            # Vertical Gray code patterns (powers of 2 width)
            for i in range(8):  # 8 patterns for vertical resolution
                stripe_width = max(2, 256 // (2**i))  # Don't go smaller than 2 pixels
                patterns.append({
                    "pattern_type": "vertical_lines",
                    "foreground_color": "White", 
                    "background_color": "Black",
                    "foreground_width": stripe_width,
                    "background_width": stripe_width
                })
                
            # Add phase-shifted patterns for the finest detail
            finest_width = 2
            for i in range(4):  # 4 phase shifts
                shift = i * finest_width // 2
                patterns.append({
                    "pattern_type": "horizontal_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": finest_width,
                    "background_width": finest_width,
                    "phase_shift": shift
                })
                patterns.append({
                    "pattern_type": "vertical_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": finest_width,
                    "background_width": finest_width,
                    "phase_shift": shift
                })
        
        else:
            # Fall back to phase shift if unknown type
            logger.warning(f"Unknown pattern type {self.config.pattern_type}, falling back to default patterns")
            
            # Default structured light pattern sequence
            widths = [16, 8, 4, 2]
            
            for width in widths:
                patterns.append({
                    "pattern_type": "horizontal_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": width,
                    "background_width": width
                })
                patterns.append({
                    "pattern_type": "vertical_lines",
                    "foreground_color": "White",
                    "background_color": "Black",
                    "foreground_width": width,
                    "background_width": width
                })
        
        # Add a black reference pattern at the end
        patterns.append({"pattern_type": "solid_field", "color": "Black"})
        
        # Log generated patterns for debugging
        logger.info(f"Created pattern sequence with {len(patterns)} patterns")
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get("pattern_type", "unknown")
            if pattern_type == "solid_field":
                logger.info(f"Pattern {i}: {pattern_type} - color={pattern.get('color')}")
            elif pattern_type in ["horizontal_lines", "vertical_lines"]:
                direction = "horizontal" if pattern_type == "horizontal_lines" else "vertical"
                fg_width = pattern.get("foreground_width", "N/A")
                bg_width = pattern.get("background_width", "N/A")
                phase = pattern.get("phase_shift", 0)
                phase_info = f", phase={phase}" if phase > 0 else ""
                logger.info(f"Pattern {i}: {direction} lines - width={fg_width}/{bg_width}{phase_info}")
            elif pattern_type == "grid":
                h_fg = pattern.get("h_foreground_width", "N/A")
                h_bg = pattern.get("h_background_width", "N/A")
                v_fg = pattern.get("v_foreground_width", "N/A")
                v_bg = pattern.get("v_background_width", "N/A")
                logger.info(f"Pattern {i}: grid - h={h_fg}/{h_bg}, v={v_fg}/{v_bg}")
            else:
                logger.info(f"Pattern {i}: {pattern_type}")
        
        return patterns
    
    def _get_camera_ids(self) -> List[str]:
        """Get camera IDs to use for scanning."""
        # Use custom camera IDs if provided
        if self.config.custom_camera_ids:
            return self.config.custom_camera_ids
        
        # Use stereo pair if enabled
        if self.config.use_stereo:
            left, right = self.client.camera.get_stereo_pair()
            if left and right:
                return [left, right]
        
        # Otherwise get all available cameras
        cameras = self.client.camera.get_cameras()
        return [cam["id"] for cam in cameras]
    
    def _start_pattern_sequence(self, patterns: List[Dict[str, Any]]) -> bool:
        """Start projecting the pattern sequence."""
        # Start the sequence
        result = self.client.projector.start_pattern_sequence(
            patterns=patterns,
            interval=self.config.pattern_interval,
            loop=self.config.mode == ScanningMode.CONTINUOUS,
            sync_with_camera=True,
            start_immediately=False  # We'll step manually
        )
        
        if not result:
            logger.error("Error starting pattern sequence")
            raise RuntimeError("Failed to start pattern sequence")
        
        logger.info(f"Pattern sequence started with {len(patterns)} patterns")
        return True
    
    def _stop_pattern_sequence(self) -> bool:
        """Stop the pattern sequence."""
        try:
            # Stop with a black pattern
            result = self.client.projector.stop_pattern_sequence(
                final_pattern={"pattern_type": "solid_field", "color": "Black"}
            )
            
            logger.info("Pattern sequence stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping pattern sequence: {e}")
            return False
    
    def _capture_synchronized_frames(self, camera_ids: List[str], patterns: List[Dict[str, Any]]) -> List[ScanFrameData]:
        """Capture frames synchronized with pattern sequence."""
        frames = []
        
        # For each pattern in the sequence
        for i, pattern in enumerate(patterns):
            # Step to the next pattern
            step_result = self.client.projector.step_pattern_sequence()
            
            if not step_result:
                logger.warning(f"Error stepping to pattern {i}")
                continue
            
            # Small delay to allow pattern to display fully
            time.sleep(self.config.pattern_interval / 2)
            
            # Capture from all cameras
            captures = self.client.camera.capture_multi(
                camera_ids,
                jpeg_quality=self.config.jpeg_quality
            )
            
            # Get pattern info - handle different response types
            if hasattr(step_result, 'payload'):
                # It's a Message object
                pattern_info = step_result.payload.get("pattern", pattern)
                timestamp = step_result.payload.get("timestamp", time.time())
            else:
                # It's a dictionary
                pattern_info = step_result.get("pattern", pattern)
                timestamp = step_result.get("timestamp", time.time())
            
            # Process each capture
            for camera_id, image in captures.items():
                if image is None:
                    logger.warning(f"No image captured from camera {camera_id} for pattern {i}")
                    continue
                
                # Create frame data
                frame_data = ScanFrameData(
                    image=image,
                    camera_id=camera_id,
                    pattern_info=pattern_info,
                    timestamp=timestamp,
                    index=i,
                    metadata={
                        "sequence_index": 0,  # Will be updated for multiple sequences
                        "pattern_type": pattern.get("pattern_type"),
                        "total_patterns": len(patterns)
                    }
                )
                
                # Save raw image if enabled
                if self.config.save_raw_images:
                    self._save_raw_image(frame_data)
                
                # Call frame callback if registered
                if self.on_frame_captured_callback:
                    self.on_frame_captured_callback(frame_data)
                
                # Add to frames list
                frames.append(frame_data)
                
                # Increment frame counter
                self.frame_count += 1
        
        logger.info(f"Captured {len(frames)} frames for {len(patterns)} patterns")
        return frames
    
    def _handle_captured_frames(self, frames: List[ScanFrameData], sequence_index: int):
        """Handle captured frames - queue for processing or process immediately."""
        # Update sequence index in metadata
        for frame in frames:
            frame.metadata["sequence_index"] = sequence_index
        
        # If real-time processing is enabled, add to queue
        if self.config.real_time_processing:
            for frame in frames:
                try:
                    # Add to queue, don't block if full (drop frames)
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    logger.warning("Processing queue full, dropping frame")
        else:
            # If not real-time, just add to queue for later processing
            for frame in frames:
                self.frame_queue.put(frame)
    
    def _process_frame_set(self, frame_set: Dict[str, ScanFrameData]) -> Optional[ScanResult]:
        """
        Process a complete set of frames to generate a 3D scan result.
        This is a placeholder implementation - in a real system, this would perform
        the actual 3D reconstruction using structured light principles.
        """
        # Create a new result
        result = ScanResult()
        
        try:
            # Get list of frames sorted by pattern index
            frames_by_camera = {}
            for camera_id, frame in frame_set.items():
                if camera_id not in frames_by_camera:
                    frames_by_camera[camera_id] = []
                frames_by_camera[camera_id].append(frame)
            
            # Sort frames for each camera by index
            for camera_id in frames_by_camera:
                frames_by_camera[camera_id].sort(key=lambda f: f.index)
            
            # Create dummy point cloud for demonstration
            # In a real implementation, this would perform triangulation and 3D reconstruction
            # using the structured light patterns
            
            # Simulate point cloud creation using the first camera's first frame
            # This is just a placeholder - real implementation would use all the pattern frames
            first_camera_id = list(frames_by_camera.keys())[0]
            reference_frame = frames_by_camera[first_camera_id][0]
            
            # Extract a subsample of points from the image
            # In real implementation, would extract 3D coordinates from pattern analysis
            h, w = reference_frame.image.shape[:2]
            
            # Create a simple simulated point cloud
            # In a real implementation this would come from actual 3D reconstruction
            try:
                # Generate more points for a better visualization
                num_points = 5000
                
                # Create a grid of points
                grid_size = int(np.sqrt(num_points))
                xs = np.linspace(-1, 1, grid_size)
                ys = np.linspace(-1, 1, grid_size)
                x, y = np.meshgrid(xs, ys)
                
                # Flatten the grid
                x = x.flatten()
                y = y.flatten()
                
                # Create a wavy surface for z
                z = 2 + 0.3 * np.sin(5 * x) * np.cos(5 * y)
                
                # Add some noise
                z += np.random.normal(0, 0.02, z.shape)
                
                # Combine into point cloud
                point_cloud = np.column_stack((x, y, z))
                
                # Create color array
                colors = np.zeros((len(point_cloud), 3), dtype=np.uint8)
                
                # Sample colors from the reference image for more realism
                # Make sure the frame has valid image data
                if reference_frame.image is not None and reference_frame.image.size > 0:
                    h, w = reference_frame.image.shape[:2]
                    
                    # Map x,y coordinates to image coordinates
                    img_x = np.clip(((x + 1) / 2 * w).astype(int), 0, w-1)
                    img_y = np.clip(((y + 1) / 2 * h).astype(int), 0, h-1)
                    
                    # Sample colors from the image
                    for i in range(len(point_cloud)):
                        colors[i] = reference_frame.image[img_y[i], img_x[i]]
                else:
                    # If image is invalid, use random colors
                    colors = np.random.randint(0, 255, (len(point_cloud), 3), dtype=np.uint8)
                
                logger.info(f"Generated simulated point cloud with {len(point_cloud)} points")
            
            except Exception as e:
                # If there's an error in the fancy point cloud generation, fall back to a simpler approach
                logger.warning(f"Error generating fancy point cloud: {e}, falling back to simple method")
                
                # Simple fallback
                num_points = 1000
                x = np.random.uniform(-1, 1, num_points)
                y = np.random.uniform(-1, 1, num_points)
                z = 2 + 0.2 * (x**2 + y**2)  # Simple paraboloid
                
                # Combine into point cloud
                point_cloud = np.column_stack((x, y, z))
                
                # Random colors
                colors = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
            
            # Update result with generated data
            result.point_cloud = point_cloud
            result.texture_map = colors
            result.confidence_map = np.random.uniform(0.5, 1.0, num_points)  # Random confidence
            result.scan_time = time.time() - self.start_time
            result.frame_count = len(frame_set)
            result.metadata = {
                "camera_ids": list(frame_set.keys()),
                "pattern_count": max(frame.index for frame in frame_set.values()) + 1,
                "timestamp": time.time()
            }
            
            logger.info(f"Generated point cloud with {len(point_cloud)} points")
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame set: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if self.on_scan_error_callback:
                self.on_scan_error_callback(f"Processing error: {e}", e)
            
            return None
    
    def _finalize_scan(self):
        """Finalize the scan process and generate final result."""
        # If we have real-time processing, we already have results
        if self.config.real_time_processing and self.latest_result:
            if self.on_scan_completed_callback:
                self.on_scan_completed_callback(self.latest_result)
            return
        
        # Otherwise process all queued frames
        self._process_captured_frames()
    
    def _save_raw_image(self, frame_data: ScanFrameData):
        """Save a raw captured image to disk."""
        if not self.config.save_raw_images:
            return
        
        try:
            # Create directory structure
            pattern_dir = os.path.join(
                self.config.output_directory,
                f"raw_{int(self.start_time)}",
                f"pattern_{frame_data.index:03d}"
            )
            
            os.makedirs(pattern_dir, exist_ok=True)
            
            # Save image
            filename = os.path.join(pattern_dir, f"camera_{frame_data.camera_id}.png")
            cv2.imwrite(filename, frame_data.image)
            
            logger.debug(f"Saved raw image to {filename}")
        except Exception as e:
            logger.error(f"Error saving raw image: {e}")
    
    def _save_point_cloud(self, points: np.ndarray, colors: np.ndarray, filepath: str):
        """
        Save point cloud to a PLY file.
        
        Args:
            points: Nx3 array of points
            colors: Nx3 array of RGB colors (0-255)
            filepath: Path to save to
        """
        if points is None or len(points) == 0:
            raise ValueError("No point cloud data to save")
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create PLY header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header"
        ]
        
        # Write file
        with open(filepath, 'w') as f:
            # Write header
            for line in header:
                f.write(line + "\n")
            
            # Write vertices with colors
            for i in range(len(points)):
                # Get point and color
                x, y, z = points[i]
                
                # Handle color
                if colors is not None and i < len(colors):
                    r, g, b = colors[i]
                else:
                    r, g, b = 128, 128, 128  # Default gray
                
                # Write line
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        
        logger.info(f"Saved point cloud with {len(points)} points to {filepath}")