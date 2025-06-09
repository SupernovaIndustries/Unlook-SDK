"""
Hardware detection module for UnLook SDK.

This module automatically detects available hardware components:
- PiCamera2 cameras and their IDs
- AS1170 LED controller on I2C bus
- DLP342x projector on I2C bus
- Future: TOF sensors and other peripherals

The detection results are used to automatically configure the appropriate
scanning module (Phase Shift, Stereo Vision, etc.) without manual configuration.
"""

import logging
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# I2C addresses
AS1170_I2C_ADDRESS = 0x30  # AS1170 LED controller (updated address)
DLP342X_I2C_ADDRESS = 0x1B  # DLP342x projector
AS1170_I2C_BUS = 4  # AS1170 is on bus 4
DLP342X_I2C_BUS = 3  # DLP342x is on bus 3

# TOF sensor addresses (for future use)
VL53L0X_I2C_ADDRESS = 0x29
VL53L1X_I2C_ADDRESS = 0x29
TMF8801_I2C_ADDRESS = 0x41


class HardwareDetector:
    """
    Detects and identifies hardware components connected to the UnLook system.
    """
    
    def __init__(self):
        """Initialize the hardware detector."""
        self.hardware_config = {
            "cameras": [],
            "as1170": False,
            "projector": None,
            "tof_sensor": None,
            "i2c_devices": []
        }
        
    def detect_all(self) -> Dict[str, Any]:
        """
        Perform complete hardware detection.
        
        Returns:
            Dict containing detected hardware configuration:
            {
                "cameras": ["picamera2_0", "picamera2_1"],
                "as1170": True/False,
                "projector": {"type": "DLP342x", "address": "0x1b", "bus": 3},
                "tof_sensor": {"type": "VL53L0X", "address": "0x29", "bus": 1},
                "i2c_devices": [{"bus": 3, "address": "0x1b", "name": "DLP342x"}, ...]
            }
        """
        logger.info("Starting hardware detection...")
        
        # Detect cameras
        self.hardware_config["cameras"] = self._detect_cameras()
        
        # Detect I2C devices
        self.hardware_config["i2c_devices"] = self._scan_i2c_buses()
        
        # Check for specific devices
        self.hardware_config["as1170"] = self._check_as1170()
        self.hardware_config["projector"] = self._check_projector()
        self.hardware_config["tof_sensor"] = self._check_tof_sensor()
        
        # Log summary
        self._log_detection_summary()
        
        return self.hardware_config
    
    def _detect_cameras(self) -> List[str]:
        """
        Detect available PiCamera2 cameras.
        
        Returns:
            List of camera IDs (e.g., ["picamera2_0", "picamera2_1"])
        """
        cameras = []
        
        try:
            # Try to import PiCamera2
            from picamera2 import Picamera2
            
            # Get camera info
            camera_info = Picamera2.global_camera_info()
            num_cameras = len(camera_info)
            
            logger.info(f"Detected {num_cameras} PiCamera2 camera(s)")
            
            # Generate camera IDs
            for i in range(num_cameras):
                camera_id = f"picamera2_{i}"
                cameras.append(camera_id)
                logger.debug(f"Camera {i}: {camera_id}")
                
        except ImportError:
            logger.warning("PiCamera2 not available - running in simulation mode")
            # In simulation mode, return no cameras
            
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
            
        return cameras
    
    def _scan_i2c_buses(self) -> List[Dict[str, Any]]:
        """
        Scan I2C buses for connected devices.
        
        Returns:
            List of detected I2C devices with bus and address information
        """
        i2c_devices = []
        
        # Check multiple I2C buses (0, 1, 3, 4 are common on Raspberry Pi)
        for bus in [0, 1, 3, 4]:
            try:
                # Use i2cdetect to scan the bus
                result = subprocess.run(
                    ["i2cdetect", "-y", str(bus)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Parse i2cdetect output
                    lines = result.stdout.strip().split('\n')
                    
                    for line in lines[1:]:  # Skip header
                        parts = line.split()
                        if len(parts) > 1:
                            # First part is row address (00:, 10:, etc.)
                            base_addr = int(parts[0].rstrip(':'), 16)
                            
                            # Check each column
                            for i, val in enumerate(parts[1:]):
                                if val != '--' and val != 'UU':
                                    addr = base_addr + i
                                    device_info = {
                                        "bus": bus,
                                        "address": f"0x{addr:02x}",
                                        "address_int": addr,
                                        "name": self._identify_i2c_device(addr)
                                    }
                                    i2c_devices.append(device_info)
                                    logger.debug(f"I2C device found: Bus {bus}, Address {device_info['address']} ({device_info['name']})")
                                    
            except FileNotFoundError:
                logger.debug(f"i2cdetect not found - skipping I2C bus {bus} scan")
            except subprocess.TimeoutExpired:
                logger.warning(f"i2cdetect timeout on bus {bus}")
            except Exception as e:
                logger.debug(f"Error scanning I2C bus {bus}: {e}")
                
        return i2c_devices
    
    def _identify_i2c_device(self, address: int) -> str:
        """
        Identify known I2C devices by their address.
        
        Args:
            address: I2C address as integer
            
        Returns:
            Device name or "Unknown"
        """
        known_devices = {
            AS1170_I2C_ADDRESS: "AS1170 LED Controller",
            DLP342X_I2C_ADDRESS: "DLP342x Projector",
            VL53L0X_I2C_ADDRESS: "VL53L0X/VL53L1X TOF Sensor",
            TMF8801_I2C_ADDRESS: "TMF8801 TOF Sensor",
            0x40: "PCA9685 PWM Controller",
            0x48: "ADS1115 ADC",
            0x68: "MPU6050 IMU",
            0x76: "BMP280 Pressure Sensor",
            0x77: "BME280 Environmental Sensor"
        }
        
        return known_devices.get(address, "Unknown")
    
    def _check_as1170(self) -> bool:
        """
        Check if AS1170 LED controller is present.
        
        Returns:
            True if AS1170 is detected, False otherwise
        """
        for device in self.hardware_config["i2c_devices"]:
            if device["address_int"] == AS1170_I2C_ADDRESS and device["bus"] == AS1170_I2C_BUS:
                logger.info(f"AS1170 LED controller detected on bus {AS1170_I2C_BUS}, address {device['address']}")
                return True
                
        logger.debug(f"AS1170 LED controller not detected (looking for address 0x{AS1170_I2C_ADDRESS:02x} on bus {AS1170_I2C_BUS})")
        return False
    
    def _check_projector(self) -> Optional[Dict[str, Any]]:
        """
        Check if DLP342x projector is present.
        
        Returns:
            Projector info dict if detected, None otherwise
        """
        for device in self.hardware_config["i2c_devices"]:
            if device["address_int"] == DLP342X_I2C_ADDRESS and device["bus"] == DLP342X_I2C_BUS:
                projector_info = {
                    "type": "DLP342x",
                    "address": device["address"],
                    "bus": device["bus"],
                    "resolution": [1280, 720],  # Native DLP342x resolution
                    "capabilities": ["structured_light", "pattern_sequence", "high_speed"]
                }
                logger.info(f"DLP342x projector detected on bus {device['bus']}, address {device['address']}")
                return projector_info
                
        logger.debug(f"DLP342x projector not detected (looking for address 0x{DLP342X_I2C_ADDRESS:02x} on bus {DLP342X_I2C_BUS})")
        return None
    
    def _check_tof_sensor(self) -> Optional[Dict[str, Any]]:
        """
        Check for TOF (Time of Flight) sensors.
        
        Returns:
            TOF sensor info dict if detected, None otherwise
        """
        # Check for various TOF sensors
        tof_addresses = [
            (VL53L0X_I2C_ADDRESS, "VL53L0X", 1),  # Usually on bus 1
            (VL53L1X_I2C_ADDRESS, "VL53L1X", 1),
            (TMF8801_I2C_ADDRESS, "TMF8801", 1)
        ]
        
        for addr, sensor_type, expected_bus in tof_addresses:
            for device in self.hardware_config["i2c_devices"]:
                if device["address_int"] == addr and device["bus"] == expected_bus:
                    tof_info = {
                        "type": sensor_type,
                        "address": device["address"],
                        "bus": device["bus"],
                        "capabilities": ["distance_measurement", "auto_focus", "presence_detection"]
                    }
                    logger.info(f"{sensor_type} TOF sensor detected on bus {device['bus']}, address {device['address']}")
                    return tof_info
                    
        logger.debug("No TOF sensor detected")
        return None
    
    def _log_detection_summary(self):
        """Log a summary of detected hardware."""
        logger.info("=" * 60)
        logger.info("Hardware Detection Summary:")
        logger.info(f"  Cameras: {len(self.hardware_config['cameras'])} detected")
        for cam in self.hardware_config["cameras"]:
            logger.info(f"    - {cam}")
            
        logger.info(f"  AS1170 LED Controller: {'Yes' if self.hardware_config['as1170'] else 'No'}")
        
        if self.hardware_config["projector"]:
            logger.info(f"  Projector: {self.hardware_config['projector']['type']} at {self.hardware_config['projector']['address']}")
        else:
            logger.info("  Projector: Not detected")
            
        if self.hardware_config["tof_sensor"]:
            logger.info(f"  TOF Sensor: {self.hardware_config['tof_sensor']['type']} at {self.hardware_config['tof_sensor']['address']}")
        else:
            logger.info("  TOF Sensor: Not detected")
            
        logger.info(f"  Total I2C devices: {len(self.hardware_config['i2c_devices'])}")
        logger.info("=" * 60)


def detect_hardware() -> Dict[str, Any]:
    """
    Convenience function to detect all hardware.
    
    Returns:
        Hardware configuration dictionary
    """
    detector = HardwareDetector()
    return detector.detect_all()


if __name__ == "__main__":
    # Test hardware detection
    import json
    
    logging.basicConfig(level=logging.DEBUG)
    
    print("Detecting UnLook hardware...")
    config = detect_hardware()
    
    print("\nDetected configuration:")
    print(json.dumps(config, indent=2))