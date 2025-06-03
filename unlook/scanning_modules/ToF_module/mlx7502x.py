#!/usr/bin/env python3
"""
MLX7502x Time-of-Flight Sensor Python Wrapper
============================================

This module provides a Python interface for MLX7502x ToF sensors (MLX75026/MLX75027)
using V4L2 (Video4Linux2) API.

Based on the original C++ implementation for the MLX75027 sensor.
"""

import os
import numpy as np
import cv2
import fcntl
import struct
import time
import logging
from typing import List, Tuple, Optional, Dict
from enum import IntEnum
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# V4L2 Constants from linux/videodev2.h
class V4L2Constants:
    # Buffer types
    V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
    
    # Field types
    V4L2_FIELD_NONE = 1
    
    # Pixel formats
    V4L2_PIX_FMT_Y12P = ord('Y') | (ord('1') << 8) | (ord('2') << 16) | (ord('P') << 24)
    
    # Control IDs
    V4L2_CID_PRIVATE_BASE = 0x08000000
    V4L2_CID_TOF_PHASE_SEQ = V4L2_CID_PRIVATE_BASE + 0x60
    V4L2_CID_TOF_TIME_INTEGRATION = V4L2_CID_PRIVATE_BASE + 0x61
    V4L2_CID_TOF_FREQ_MOD = V4L2_CID_PRIVATE_BASE + 0x62
    V4L2_CID_MLX7502X_OUTPUT_MODE = V4L2_CID_PRIVATE_BASE + 0x63
    
    # IOCTL commands
    VIDIOC_S_FMT = 0xc0cc5605
    VIDIOC_S_EXT_CTRLS = 0xc0185648
    VIDIOC_G_PARM = 0xc0cc5615
    VIDIOC_S_PARM = 0xc0cc5616
    VIDIOC_SUBDEV_S_FRAME_INTERVAL = 0xc0305616
    VIDIOC_DBG_S_REGISTER = 0x4038564f
    
    # Chip match types
    V4L2_CHIP_MATCH_SUBDEV = 4
    
    # Subdev format
    V4L2_SUBDEV_FORMAT_ACTIVE = 1


@dataclass
class ToFConfig:
    """Configuration for MLX7502x ToF sensors"""
    width: int = 640
    height: int = 480
    fps: int = 12
    phase_sequence: List[int] = None
    time_integration: List[int] = None
    modulation_frequency: int = 10000000  # 10 MHz
    output_mode: int = 0
    mipi_lanes: int = 4  # Number of MIPI CSI-2 data lanes (2 or 4)
    mipi_speed: int = 960000000  # MIPI speed in Hz (default 960MHz)
    
    def __post_init__(self):
        if self.phase_sequence is None:
            self.phase_sequence = [0, 180, 90, 270]
        if self.time_integration is None:
            self.time_integration = [1000, 1000, 1000, 1000]
        
        # Validate MIPI lanes
        if self.mipi_lanes not in [2, 4]:
            raise ValueError("MIPI lanes must be 2 or 4")
            
        # Adjust MIPI speed based on lanes
        # 2-lane mode typically needs higher speed per lane
        if self.mipi_lanes == 2:
            # Double the speed per lane for 2-lane mode to maintain bandwidth
            self.mipi_speed = min(self.mipi_speed * 2, 1200000000)  # Cap at 1.2GHz


class MLX7502x:
    """Python wrapper for MLX7502x Time-of-Flight sensors"""
    
    def __init__(self, video_device: str = "/dev/video0", 
                 subdevice: str = "/dev/v4l-subdev0",
                 config: Optional[ToFConfig] = None):
        """
        Initialize MLX75027 ToF sensor
        
        Args:
            video_device: Path to video device (default: /dev/video0)
            subdevice: Path to V4L2 subdevice (default: /dev/v4l-subdev0)
            config: ToF configuration (uses defaults if None)
        """
        self.video_device = video_device
        self.subdevice = subdevice
        self.config = config or ToFConfig()
        
        self.fd = None
        self.subfd = None
        self.cap = None
        
        # Frame buffers
        self.frames = {
            0: None,    # 0 degree phase
            90: None,   # 90 degree phase
            180: None,  # 180 degree phase
            270: None   # 270 degree phase
        }
        
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def open(self):
        """Open and configure the ToF sensor"""
        try:
            # Open video device
            self.fd = os.open(self.video_device, os.O_RDWR)
            logger.info(f"Opened {self.video_device} device")
            
            # Open subdevice
            self.subfd = os.open(self.subdevice, os.O_RDWR)
            logger.info(f"Opened {self.subdevice} subdevice")
            
            # Configure sensor
            self._configure_sensor()
            
            # Open OpenCV capture
            self.cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open V4L2 stream")
                
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to open ToF sensor: {e}")
            
    def close(self):
        """Close the ToF sensor"""
        try:
            # Stop streaming before closing
            if self.fd is not None:
                self._stop_streaming()
                self._enter_standby()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            
        if self.subfd is not None:
            os.close(self.subfd)
            self.subfd = None
            
        logger.info("Closed ToF sensor")
        
    def _configure_sensor(self):
        """Configure the sensor with initial parameters"""
        # Set phase sequence
        self._set_phase_sequence(self.config.phase_sequence)
        
        # Set image format
        self._set_image_format()
        
        # Set time integration
        self._set_time_integration(self.config.time_integration)
        
        # Set output mode
        self._set_output_mode(self.config.output_mode)
        
        # Set modulation frequency
        self._set_modulation_frequency(self.config.modulation_frequency)
        
        # Set frame rate
        self._set_frame_rate(self.config.fps)
        
        # Enter standby mode for configuration
        self._enter_standby()
        
        # Configure MIPI lanes if different from default
        if self.config.mipi_lanes != 4:
            self._configure_mipi_lanes()
        
        # Set registers (equivalent to C++ code)
        self._set_tof_register(0x21a0, 0x22, 0x01)
        self._set_tof_register(0x21a1, 0x22, 0x01)
        
        # Exit standby and start streaming
        self._exit_standby()
        self._start_streaming()
        
    def _set_phase_sequence(self, phase_seq: List[int]):
        """Set the phase sequence for ToF measurement"""
        logger.info(f"Setting phase sequence: {phase_seq}")
        self._set_array_control(V4L2Constants.V4L2_CID_TOF_PHASE_SEQ, phase_seq)
        
    def _set_time_integration(self, time_int: List[int]):
        """Set the integration time for each phase"""
        logger.info(f"Setting time integration: {time_int}")
        self._set_array_control(V4L2Constants.V4L2_CID_TOF_TIME_INTEGRATION, time_int)
        
    def _set_modulation_frequency(self, freq: int):
        """Set the modulation frequency"""
        logger.info(f"Setting modulation frequency: {freq} Hz")
        self._set_array_control(V4L2Constants.V4L2_CID_TOF_FREQ_MOD, [freq])
        
    def _set_output_mode(self, mode: int):
        """Set the output mode"""
        logger.info(f"Setting output mode: {mode}")
        
        # Prepare control structure
        ctrl_data = struct.pack("=IIIIq", 
                               V4L2Constants.V4L2_CID_MLX7502X_OUTPUT_MODE,  # id
                               0,  # size
                               0,  # reserved
                               0,  # reserved2
                               mode)  # value
        
        # Prepare ext_controls structure
        ext_ctrl_data = struct.pack("=IIIP",
                                   0,  # ctrl_class
                                   1,  # count
                                   0,  # error_idx
                                   id(ctrl_data))  # controls pointer
        
        try:
            fcntl.ioctl(self.subfd, V4L2Constants.VIDIOC_S_EXT_CTRLS, ext_ctrl_data)
        except IOError as e:
            logger.error(f"Failed to set output mode: {e}")
            raise
            
    def _set_array_control(self, control_id: int, values: List[int]):
        """Set an array control value"""
        # Convert to uint16 array
        array_data = struct.pack(f"={len(values)}H", *values)
        
        # Prepare control structure
        ctrl_data = struct.pack("=IIIIP",
                               control_id,  # id
                               len(array_data),  # size
                               0,  # reserved
                               0,  # reserved2  
                               id(array_data))  # ptr
        
        # Prepare ext_controls structure
        ext_ctrl_data = struct.pack("=IIIP",
                                   0,  # ctrl_class
                                   1,  # count
                                   0,  # error_idx
                                   id(ctrl_data))  # controls pointer
        
        try:
            fcntl.ioctl(self.subfd, V4L2Constants.VIDIOC_S_EXT_CTRLS, ext_ctrl_data)
        except IOError as e:
            logger.error(f"Failed to set array control {control_id}: {e}")
            raise
            
    def _set_image_format(self):
        """Set the image format"""
        # v4l2_format structure
        fmt_data = struct.pack("=I"  # type
                              "IIII"  # pix.width, height, pixelformat, field
                              "I"     # pix.bytesperline
                              "I"     # pix.sizeimage
                              "I"     # pix.colorspace
                              "I"     # pix.priv
                              "48x",  # padding
                              V4L2Constants.V4L2_BUF_TYPE_VIDEO_CAPTURE,
                              self.config.width,
                              self.config.height,
                              V4L2Constants.V4L2_PIX_FMT_Y12P,
                              V4L2Constants.V4L2_FIELD_NONE,
                              0, 0, 0, 0)
        
        try:
            fcntl.ioctl(self.fd, V4L2Constants.VIDIOC_S_FMT, fmt_data)
        except IOError as e:
            logger.error(f"Failed to set image format: {e}")
            raise
            
    def _set_frame_rate(self, fps: int):
        """Set the frame rate"""
        # Get current parameters
        parm_data = bytearray(204)  # v4l2_streamparm size
        struct.pack_into("=I", parm_data, 0, V4L2Constants.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        
        try:
            fcntl.ioctl(self.fd, V4L2Constants.VIDIOC_G_PARM, parm_data)
            
            # Set new frame rate
            struct.pack_into("=II", parm_data, 180, 1, fps)  # timeperframe offset
            
            fcntl.ioctl(self.fd, V4L2Constants.VIDIOC_S_PARM, parm_data)
        except IOError as e:
            logger.error(f"Failed to set frame rate: {e}")
            raise
            
    def _configure_mipi_lanes(self):
        """Configure MIPI CSI-2 lane count according to MLX75027 datasheet"""
        logger.info(f"Configuring MIPI lanes: {self.config.mipi_lanes}")
        
        # MLX75027 specific registers from datasheet
        SOFTWARE_STANDBY_REG = 0x1000
        STREAMING_REG = 0x1001
        DATA_LANE_CONFIG_REG = 0x1010
        
        # Enter software standby before configuration
        logger.info("Entering software standby")
        self._set_tof_register(SOFTWARE_STANDBY_REG, 0x01, 0x01)
        
        # Configure data lanes
        # DATA_LANE_CONFIG = 0 for 2 lanes, 1 for 4 lanes
        lane_value = 0x00 if self.config.mipi_lanes == 2 else 0x01
        self._set_tof_register(DATA_LANE_CONFIG_REG, lane_value, 0x01)
        logger.info(f"Set DATA_LANE_CONFIG to {lane_value} ({self.config.mipi_lanes} lanes)")
        
        # Configure MIPI speed registers (0x100C to 0x1071 range)
        # Note: Specific speed configuration depends on desired data rate
        # This is a simplified implementation - refer to datasheet for exact values
        if self.config.mipi_lanes == 2:
            # 2-lane mode may need different timing
            logger.info("Configuring timing for 2-lane mode")
            # Example: Set some key timing registers
            self._set_tof_register(0x100C, 0x04, 0x01)  # Example value
            self._set_tof_register(0x100D, 0x00, 0x01)  # Example value
        
        logger.info(f"MIPI configuration complete: {self.config.mipi_lanes} lanes")
    
    def _enter_standby(self):
        """Enter software standby mode (required before configuration)"""
        SOFTWARE_STANDBY_REG = 0x1000
        logger.info("Entering software standby mode")
        self._set_tof_register(SOFTWARE_STANDBY_REG, 0x01, 0x01)
        time.sleep(0.01)  # Small delay to ensure standby is active
        
    def _exit_standby(self):
        """Exit software standby mode"""
        SOFTWARE_STANDBY_REG = 0x1000
        logger.info("Exiting software standby mode")
        self._set_tof_register(SOFTWARE_STANDBY_REG, 0x00, 0x01)
        time.sleep(0.01)  # Small delay to ensure standby is exited
        
    def _start_streaming(self):
        """Start sensor streaming"""
        STREAMING_REG = 0x1001
        logger.info("Starting sensor streaming")
        self._set_tof_register(STREAMING_REG, 0x01, 0x01)
        
    def _stop_streaming(self):
        """Stop sensor streaming"""
        STREAMING_REG = 0x1001
        logger.info("Stopping sensor streaming")
        self._set_tof_register(STREAMING_REG, 0x00, 0x01)
    
    def _set_tof_register(self, addr: int, value: int, size: int):
        """Set a ToF register value"""
        # v4l2_dbg_register structure
        reg_data = struct.pack("=IIQQI",
                              V4L2Constants.V4L2_CHIP_MATCH_SUBDEV,  # match.type
                              0,  # match.addr
                              addr,  # reg
                              value,  # val
                              size)  # size
        
        try:
            fcntl.ioctl(self.fd, V4L2Constants.VIDIOC_DBG_S_REGISTER, reg_data)
        except IOError as e:
            logger.warning(f"Failed to set ToF register 0x{addr:04x}: {e}")
            
    def capture_phase_frames(self) -> Dict[int, np.ndarray]:
        """
        Capture frames for all phase angles
        
        Returns:
            Dictionary with phase angles as keys and frames as values
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Sensor not opened")
            
        # Capture frames with timing to ensure we get the right phases
        start_time = time.time()
        
        # Wait for first frame (0 degree phase)
        while True:
            ret, frame = self.cap.read()
            if time.time() - start_time > 0.025:  # 25ms
                break
                
        self.frames[0] = self._process_raw_frame(frame)
        
        # Capture remaining phases
        ret, frame = self.cap.read()
        self.frames[180] = self._process_raw_frame(frame)
        
        ret, frame = self.cap.read()
        self.frames[90] = self._process_raw_frame(frame)
        
        ret, frame = self.cap.read()
        self.frames[270] = self._process_raw_frame(frame)
        
        return self.frames
        
    def _process_raw_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Process raw Y12P frame to 16-bit values
        
        Args:
            raw_frame: Raw frame from sensor
            
        Returns:
            Processed 16-bit frame
        """
        if raw_frame is None:
            return None
            
        # Convert to 16-bit and shift left by 4 bits (12-bit to 16-bit)
        frame_16 = raw_frame.astype(np.int16) << 4
        return frame_16
        
    def compute_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute depth from captured phase frames
        
        Returns:
            Tuple of (magnitude_image, phase_image)
        """
        if any(self.frames[phase] is None for phase in [0, 90, 180, 270]):
            raise ValueError("Not all phase frames captured")
            
        # Convert to float64 for calculations
        frame_0 = self.frames[0].astype(np.float64)
        frame_90 = self.frames[90].astype(np.float64)
        frame_180 = self.frames[180].astype(np.float64)
        frame_270 = self.frames[270].astype(np.float64)
        
        # Calculate I and Q components
        I = frame_0 - frame_180
        Q = frame_90 - frame_270
        
        # Calculate magnitude
        magnitude = np.sqrt(I**2 + Q**2)
        
        # Calculate phase (in degrees)
        phase = np.arctan2(Q, I) * 180 / np.pi
        phase = np.mod(phase, 360)  # Wrap to 0-360 degrees
        
        return magnitude, phase
        
    def visualize_results(self, magnitude: np.ndarray, phase: np.ndarray):
        """
        Visualize magnitude and phase images
        
        Args:
            magnitude: Magnitude image
            phase: Phase image (in degrees)
        """
        # Convert magnitude to 8-bit RGB
        mag_min, mag_max = magnitude.min(), magnitude.max()
        magnitude_8bit = ((magnitude - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
        magnitude_rgb = cv2.cvtColor(magnitude_8bit, cv2.COLOR_GRAY2BGR)
        
        # Convert phase to 8-bit RGB (0-360 degrees to 0-255)
        phase_8bit = (phase / 360.0 * 255).astype(np.uint8)
        phase_rgb = cv2.cvtColor(phase_8bit, cv2.COLOR_GRAY2BGR)
        
        # Apply colormap to phase for better visualization
        phase_colormap = cv2.applyColorMap(phase_8bit, cv2.COLORMAP_HSV)
        
        return magnitude_rgb, phase_rgb, phase_colormap
        
    def capture_continuous(self, display: bool = True, 
                          callback: Optional[callable] = None) -> None:
        """
        Continuously capture and process ToF data
        
        Args:
            display: Whether to display the results
            callback: Optional callback function for each frame
        """
        if display:
            cv2.namedWindow("ToF Magnitude", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("ToF Phase", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("ToF Phase Colormap", cv2.WINDOW_AUTOSIZE)
            
        try:
            while True:
                # Capture phase frames
                self.capture_phase_frames()
                
                # Compute depth
                magnitude, phase = self.compute_depth()
                
                # Visualize
                mag_rgb, phase_rgb, phase_cm = self.visualize_results(magnitude, phase)
                
                # Call callback if provided
                if callback:
                    callback(magnitude, phase)
                    
                # Display if requested
                if display:
                    cv2.imshow("ToF Magnitude", mag_rgb)
                    cv2.imshow("ToF Phase", phase_rgb)
                    cv2.imshow("ToF Phase Colormap", phase_cm)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        finally:
            if display:
                cv2.destroyAllWindows()


def main():
    """Example usage of MLX75027 wrapper"""
    # Configure sensor
    config = ToFConfig(
        fps=12,
        phase_sequence=[0, 180, 90, 270],
        time_integration=[1000, 1000, 1000, 1000]
    )
    
    # Use sensor
    with MLX75027(config=config) as sensor:
        # Continuous capture with display
        sensor.capture_continuous(display=True)
        
        # Or single capture
        # sensor.capture_phase_frames()
        # magnitude, phase = sensor.compute_depth()
        # print(f"Magnitude shape: {magnitude.shape}")
        # print(f"Phase shape: {phase.shape}")


if __name__ == "__main__":
    main()