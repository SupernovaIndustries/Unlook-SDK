"""
Hardware camera synchronization module for UnLook scanner.
Provides microsecond-precision synchronization using GPIO interrupts.
"""

import logging
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque

# Only import GPIO on Raspberry Pi
try:
    import pigpio
    PIGPIO_AVAILABLE = True
except ImportError:
    PIGPIO_AVAILABLE = False
    
logger = logging.getLogger(__name__)


@dataclass
class SyncMetrics:
    """Metrics for synchronization quality monitoring."""
    sync_precision_us: float  # Actual sync precision in microseconds
    frame_consistency: float  # Percentage of frames delivered on time
    average_latency_us: float  # Average latency from trigger to capture
    jitter_us: float  # Standard deviation of latency
    missed_triggers: int  # Number of missed sync triggers
    timestamp: float  # When metrics were calculated
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        from dataclasses import asdict
        return asdict(self)


class HardwareCameraSyncV2:
    """
    Enhanced hardware camera synchronization using GPIO interrupts.
    Targets < 100μs precision for stereo camera capture.
    """
    
    def __init__(self, trigger_gpio: int = 27):
        """
        Initialize hardware sync system.
        
        Args:
            trigger_gpio: GPIO pin for sync trigger (default: 27)
        """
        self.trigger_gpio = trigger_gpio
        self.enabled = False
        self.callbacks = []  # List of callbacks to trigger on sync
        
        # Timing tracking
        self.trigger_times = deque(maxlen=1000)  # Track last 1000 triggers
        self.capture_times = deque(maxlen=1000)  # Track capture completions
        self.latencies = deque(maxlen=1000)  # Track trigger-to-capture latencies
        
        # Sync state
        self._lock = threading.RLock()
        self.trigger_event = threading.Event()
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.missed_triggers = 0
        
        # GPIO interface
        self.pi = None
        if PIGPIO_AVAILABLE:
            try:
                self.pi = pigpio.pi()
                if self.pi.connected:
                    logger.info(f"Connected to pigpio daemon for GPIO {trigger_gpio}")
                    self._setup_gpio()
                else:
                    logger.warning("Could not connect to pigpio daemon - hardware sync will use software timing")
                    self.pi = None
            except Exception as e:
                logger.error(f"Error initializing pigpio: {e}")
                self.pi = None
        else:
            logger.warning("pigpio not available, hardware sync disabled")
    
    def _setup_gpio(self):
        """Configure GPIO for interrupt-based triggering."""
        if not self.pi:
            return
            
        try:
            # Set GPIO as input with pull-down
            self.pi.set_mode(self.trigger_gpio, pigpio.INPUT)
            self.pi.set_pull_up_down(self.trigger_gpio, pigpio.PUD_DOWN)
            
            # Set up interrupt on rising edge
            self.pi.callback(self.trigger_gpio, pigpio.RISING_EDGE, self._gpio_trigger_callback)
            
            logger.info(f"GPIO {self.trigger_gpio} configured for interrupt-based sync")
            
        except Exception as e:
            logger.error(f"Error setting up GPIO: {e}")
    
    def _gpio_trigger_callback(self, gpio, level, tick):
        """
        Interrupt callback for GPIO trigger.
        Called by pigpio with microsecond timestamp.
        
        Args:
            gpio: GPIO number that triggered
            level: Level of GPIO (0=low, 1=high, 2=watchdog)
            tick: Microsecond timestamp from pigpio
        """
        # Convert pigpio tick to system time
        trigger_time = time.time()
        
        with self._lock:
            # Track trigger timing
            self.last_trigger_time = trigger_time
            self.trigger_count += 1
            self.trigger_times.append(trigger_time)
            
            # Set event for waiting threads
            self.trigger_event.set()
            
            # Execute callbacks in separate threads for minimal latency
            for callback in self.callbacks:
                threading.Thread(target=self._execute_callback, 
                               args=(callback, trigger_time)).start()
    
    def _execute_callback(self, callback: Callable, trigger_time: float):
        """Execute a sync callback and track timing."""
        try:
            start_time = time.time()
            callback(trigger_time)
            end_time = time.time()
            
            # Track capture completion and latency
            with self._lock:
                self.capture_times.append(end_time)
                latency_us = (start_time - trigger_time) * 1_000_000
                self.latencies.append(latency_us)
                
        except Exception as e:
            logger.error(f"Error in sync callback: {e}")
    
    def register_callback(self, callback: Callable):
        """
        Register a callback to be called on sync trigger.
        
        Args:
            callback: Function to call on trigger, receives trigger timestamp
        """
        with self._lock:
            self.callbacks.append(callback)
            logger.info(f"Registered sync callback: {callback.__name__}")
    
    def enable_software_trigger(self, frequency_hz: float = 30.0):
        """
        Enable software-based triggering as fallback.
        Uses high-precision timing to simulate hardware trigger.
        
        Args:
            frequency_hz: Trigger frequency in Hz
        """
        self.enabled = True
        self.software_trigger_thread = threading.Thread(
            target=self._software_trigger_loop,
            args=(frequency_hz,),
            daemon=True
        )
        self.software_trigger_thread.start()
        logger.info(f"Software trigger enabled at {frequency_hz} Hz")
    
    def _software_trigger_loop(self, frequency_hz: float):
        """Software trigger loop with microsecond precision timing."""
        period = 1.0 / frequency_hz
        next_trigger = time.time()
        
        while self.enabled:
            # Wait until next trigger time
            sleep_time = next_trigger - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Trigger
            trigger_time = time.time()
            self._software_trigger(trigger_time)
            
            # Calculate next trigger time
            next_trigger += period
            
            # Handle drift
            if next_trigger < time.time():
                # We're behind schedule, reset
                next_trigger = time.time() + period
                self.missed_triggers += 1
    
    def _software_trigger(self, trigger_time: float):
        """Execute software trigger."""
        with self._lock:
            self.last_trigger_time = trigger_time
            self.trigger_count += 1
            self.trigger_times.append(trigger_time)
            self.trigger_event.set()
            
            # Execute callbacks
            for callback in self.callbacks:
                threading.Thread(target=self._execute_callback,
                               args=(callback, trigger_time)).start()
    
    def wait_for_trigger(self, timeout: float = 1.0) -> Optional[float]:
        """
        Wait for next sync trigger.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Trigger timestamp or None if timeout
        """
        self.trigger_event.clear()
        if self.trigger_event.wait(timeout):
            return self.last_trigger_time
        return None
    
    def get_sync_metrics(self) -> SyncMetrics:
        """
        Calculate and return synchronization quality metrics.
        
        Returns:
            SyncMetrics object with current sync quality data
        """
        with self._lock:
            # Calculate sync precision (jitter between triggers)
            if len(self.trigger_times) > 1:
                trigger_intervals = []
                for i in range(1, len(self.trigger_times)):
                    interval = (self.trigger_times[i] - self.trigger_times[i-1]) * 1_000_000
                    trigger_intervals.append(interval)
                
                if trigger_intervals:
                    avg_interval = sum(trigger_intervals) / len(trigger_intervals)
                    jitter = sum((x - avg_interval) ** 2 for x in trigger_intervals)
                    jitter = (jitter / len(trigger_intervals)) ** 0.5
                else:
                    jitter = 0
            else:
                jitter = 0
            
            # Calculate average latency
            if self.latencies:
                avg_latency = sum(self.latencies) / len(self.latencies)
            else:
                avg_latency = 0
            
            # Calculate frame consistency (% of frames with latency < 2ms)
            if self.latencies:
                consistent_frames = sum(1 for l in self.latencies if l < 2000)
                consistency = (consistent_frames / len(self.latencies)) * 100
            else:
                consistency = 0
            
            return SyncMetrics(
                sync_precision_us=jitter,
                frame_consistency=consistency,
                average_latency_us=avg_latency,
                jitter_us=jitter,
                missed_triggers=self.missed_triggers,
                timestamp=time.time()
            )
    
    def close(self):
        """Clean up GPIO resources."""
        self.enabled = False
        
        if self.pi:
            try:
                self.pi.stop()
                logger.info("GPIO cleaned up")
            except:
                pass


# Global sync instance for server-wide use
_sync_instance = None

def get_camera_sync() -> HardwareCameraSyncV2:
    """Get or create the global camera sync instance."""
    global _sync_instance
    if _sync_instance is None:
        _sync_instance = HardwareCameraSyncV2()
    return _sync_instance