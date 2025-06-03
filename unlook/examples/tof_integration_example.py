#!/usr/bin/env python3
"""
ToF Integration with Unlook SDK
===============================

This example shows how to integrate the MLX75027 ToF sensor 
with the Unlook SDK scanning system.
"""

import sys
import os
import numpy as np
import cv2
import argparse
import logging
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unlook.scanning_modules.ToF_module import MLX75027, ToFConfig
from unlook.core import UnlookScanner, discovery
from unlook.client import Camera, Scanner3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToFScanner:
    """Combined ToF and structured light scanner"""
    
    def __init__(self, tof_device: str = "/dev/video0", 
                 tof_subdevice: str = "/dev/v4l-subdev0",
                 scanner_ip: Optional[str] = None):
        """
        Initialize combined scanner
        
        Args:
            tof_device: ToF video device path
            tof_subdevice: ToF V4L2 subdevice path
            scanner_ip: IP address of Unlook scanner (auto-discover if None)
        """
        # Initialize ToF sensor
        self.tof_config = ToFConfig(
            fps=12,
            phase_sequence=[0, 180, 90, 270],
            time_integration=[1000, 1000, 1000, 1000]
        )
        self.tof = MLX75027(tof_device, tof_subdevice, self.tof_config)
        
        # Initialize Unlook scanner
        if scanner_ip is None:
            scanners = discovery.discover_scanners(timeout=5)
            if not scanners:
                raise RuntimeError("No Unlook scanners found")
            self.scanner = scanners[0]
        else:
            self.scanner = UnlookScanner(scanner_ip)
            
        self.scanner3d = None
        
    def __enter__(self):
        """Context manager entry"""
        self.tof.open()
        self.scanner3d = Scanner3D(self.scanner)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.tof.close()
        if self.scanner3d:
            self.scanner3d.close()
            
    def capture_tof_depth(self):
        """Capture depth data from ToF sensor"""
        logger.info("Capturing ToF depth data...")
        
        # Capture phase frames
        self.tof.capture_phase_frames()
        
        # Compute depth
        magnitude, phase = self.tof.compute_depth()
        
        # Convert phase to distance (simplified - needs calibration)
        # Distance = (c * phase) / (4 * pi * f_mod)
        # where c = speed of light, f_mod = modulation frequency
        c = 299792458  # m/s
        f_mod = self.tof_config.modulation_frequency
        
        # Convert phase from degrees to radians
        phase_rad = phase * np.pi / 180.0
        
        # Calculate distance in meters
        distance = (c * phase_rad) / (4 * np.pi * f_mod)
        
        return distance, magnitude
        
    def capture_structured_light(self):
        """Capture structured light scan"""
        logger.info("Capturing structured light scan...")
        
        # Perform 3D scan
        result = self.scanner3d.scan_3d()
        
        return result.point_cloud
        
    def fuse_depth_data(self, tof_distance, tof_magnitude, sl_point_cloud):
        """
        Fuse ToF and structured light depth data
        
        This is a simplified fusion - real implementation would need:
        - Proper calibration between ToF and cameras
        - Spatial alignment of point clouds
        - Weighted fusion based on confidence
        """
        logger.info("Fusing depth data...")
        
        # For demonstration, we'll create a confidence map
        # High magnitude ToF readings are more reliable
        tof_confidence = tof_magnitude / np.max(tof_magnitude)
        
        # Create ToF point cloud (simplified - needs proper calibration)
        height, width = tof_distance.shape
        fx = fy = 600  # Placeholder focal length
        cx = width / 2
        cy = height / 2
        
        # Generate mesh grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate 3D points from ToF
        z = tof_distance * 1000  # Convert to mm
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        
        # Stack into point cloud
        tof_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Filter by confidence
        confidence_threshold = 0.3
        mask = tof_confidence.flatten() > confidence_threshold
        tof_points_filtered = tof_points[mask]
        
        # In a real implementation, you would:
        # 1. Transform ToF points to structured light coordinate system
        # 2. Find correspondences between point clouds
        # 3. Fuse based on local confidence and accuracy
        
        logger.info(f"ToF points: {len(tof_points_filtered)}")
        logger.info(f"Structured light points: {len(sl_point_cloud.points) if sl_point_cloud else 0}")
        
        return tof_points_filtered, tof_confidence
        
    def visualize_fusion(self, tof_distance, tof_magnitude, tof_confidence):
        """Visualize the fused results"""
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ToF distance
        im1 = axes[0, 0].imshow(tof_distance, cmap='jet')
        axes[0, 0].set_title('ToF Distance (m)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # ToF magnitude
        im2 = axes[0, 1].imshow(tof_magnitude, cmap='gray')
        axes[0, 1].set_title('ToF Magnitude')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Confidence map
        im3 = axes[1, 0].imshow(tof_confidence, cmap='hot')
        axes[1, 0].set_title('ToF Confidence')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 3D visualization placeholder
        axes[1, 1].text(0.5, 0.5, 'Point Cloud\n(Use Open3D for\n3D visualization)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Fused Point Cloud')
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="ToF and Structured Light Fusion Demo")
    parser.add_argument("--tof-device", default="/dev/video0",
                        help="ToF video device path")
    parser.add_argument("--tof-subdevice", default="/dev/v4l-subdev0",
                        help="ToF V4L2 subdevice path")
    parser.add_argument("--scanner-ip", default=None,
                        help="Unlook scanner IP (auto-discover if not specified)")
    parser.add_argument("--mode", choices=["tof-only", "fusion"], default="tof-only",
                        help="Operation mode")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "tof-only":
            # ToF only mode
            config = ToFConfig()
            with MLX75027(args.tof_device, args.tof_subdevice, config) as tof:
                # Single capture
                tof.capture_phase_frames()
                magnitude, phase = tof.compute_depth()
                
                # Visualize
                mag_rgb, phase_rgb, phase_cm = tof.visualize_results(magnitude, phase)
                
                cv2.imshow("ToF Magnitude", mag_rgb)
                cv2.imshow("ToF Phase", phase_cm)
                print("Press any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            # Fusion mode
            with ToFScanner(args.tof_device, args.tof_subdevice, args.scanner_ip) as scanner:
                # Capture ToF depth
                tof_distance, tof_magnitude = scanner.capture_tof_depth()
                
                # Capture structured light (if scanner connected)
                try:
                    sl_point_cloud = scanner.capture_structured_light()
                except Exception as e:
                    logger.warning(f"Structured light capture failed: {e}")
                    sl_point_cloud = None
                    
                # Fuse data
                tof_points, tof_confidence = scanner.fuse_depth_data(
                    tof_distance, tof_magnitude, sl_point_cloud
                )
                
                # Visualize
                try:
                    import matplotlib.pyplot as plt
                    scanner.visualize_fusion(tof_distance, tof_magnitude, tof_confidence)
                except ImportError:
                    logger.warning("Matplotlib not available for visualization")
                    
                    # Simple OpenCV visualization
                    distance_vis = cv2.normalize(tof_distance, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    distance_color = cv2.applyColorMap(distance_vis, cv2.COLORMAP_JET)
                    
                    cv2.imshow("ToF Distance", distance_color)
                    cv2.imshow("ToF Magnitude", cv2.normalize(tof_magnitude, None, 0, 255, 
                                                              cv2.NORM_MINMAX, cv2.CV_8U))
                    print("Press any key to exit...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())