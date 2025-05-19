"""
UnLook Simple API - Arduino-style simplicity for 3D vision.

This module provides the simplest possible interface to UnLook,
hiding all complexity while keeping advanced features accessible.
"""

import time
from pathlib import Path
from typing import Optional, Any, Callable

from .client import UnlookClient
from .client.scanner.scanner3d import create_scanner


class UnlookSimple:
    """
    Simple UnLook interface - like Arduino for computer vision.
    
    Basic usage:
        unlook = UnlookSimple()
        unlook.connect()
        image = unlook.capture()
        point_cloud = unlook.scan_3d()
    """
    
    def __init__(self, debug: bool = False):
        """Initialize UnLook simple interface."""
        self.client = None
        self.scanner_info = None
        self.scanner3d = None
        self.debug = debug
        
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Auto-connect to first available UnLook scanner.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Create client
            self.client = UnlookClient(auto_discover=True)
            
            # Wait for discovery
            time.sleep(timeout)
            
            # Get scanners
            scanners = self.client.get_discovered_scanners()
            if not scanners:
                if self.debug:
                    print("No UnLook scanner found")
                return False
            
            # Connect to first scanner
            self.scanner_info = scanners[0]
            if self.client.connect(self.scanner_info):
                if self.debug:
                    print(f"Connected to {self.scanner_info.name}")
                return True
            
            return False
            
        except Exception as e:
            if self.debug:
                print(f"Connection error: {e}")
            return False
    
    def capture(self, camera_index: int = 0) -> Optional[Any]:
        """
        Capture a single image from camera.
        
        Args:
            camera_index: Which camera to use (0 = first, 1 = second)
            
        Returns:
            Image as numpy array or None
        """
        if not self.client or not self.client.connected:
            return None
            
        try:
            cameras = self.client.camera.get_cameras()
            if camera_index >= len(cameras):
                return None
                
            return self.client.camera.capture(cameras[camera_index]['id'])
            
        except Exception as e:
            if self.debug:
                print(f"Capture error: {e}")
            return None
    
    def scan_3d(self, quality: str = "fast") -> Optional[Any]:
        """
        Perform a 3D scan with automatic settings.
        
        Args:
            quality: "fast", "balanced", or "high"
            
        Returns:
            Point cloud or None
        """
        if not self.client or not self.client.connected:
            return None
            
        try:
            # Create scanner if needed
            if not self.scanner3d:
                self.scanner3d = create_scanner(self.client, quality=quality)
            
            # Perform scan
            result = self.scanner3d.scan()
            return result.point_cloud if result else None
            
        except Exception as e:
            if self.debug:
                print(f"Scan error: {e}")
            return None
    
    def save_scan(self, filename: str = "scan.ply") -> bool:
        """
        Save last scan to file.
        
        Args:
            filename: Output filename (supports .ply, .pcd, .xyz)
            
        Returns:
            True if saved successfully
        """
        if not self.scanner3d:
            return False
            
        try:
            self.scanner3d.save_point_cloud(filename)
            return True
        except:
            return False
    
    def stream(self, callback: Callable, camera_index: int = 0) -> bool:
        """
        Stream live video from camera.
        
        Args:
            callback: Function to call for each frame
            camera_index: Which camera to use
            
        Returns:
            True if streaming started
        """
        if not self.client or not self.client.connected:
            return False
            
        try:
            cameras = self.client.camera.get_cameras()
            if camera_index >= len(cameras):
                return False
                
            return self.client.stream.start(
                cameras[camera_index]['id'], 
                callback
            )
        except:
            return False
    
    def disconnect(self):
        """Disconnect from scanner."""
        if self.client:
            self.client.disconnect()
            self.client.stop_discovery()
    
    def __enter__(self):
        """Context manager support."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.disconnect()
        
    # Convenience properties
    @property
    def connected(self) -> bool:
        """Check if connected to scanner."""
        return self.client and self.client.connected
    
    @property
    def cameras(self) -> list:
        """Get list of camera names."""
        if not self.connected:
            return []
        try:
            return [cam['name'] for cam in self.client.camera.get_cameras()]
        except:
            return []


# Convenience functions for one-line operations
def quick_capture(filename: str = "capture.jpg") -> bool:
    """Capture and save an image in one line."""
    with UnlookSimple() as unlook:
        import cv2
        image = unlook.capture()
        if image is not None:
            cv2.imwrite(filename, image)
            return True
    return False


def quick_scan(filename: str = "scan.ply", quality: str = "fast") -> bool:
    """Perform and save a 3D scan in one line."""
    with UnlookSimple() as unlook:
        point_cloud = unlook.scan_3d(quality)
        if point_cloud:
            return unlook.save_scan(filename)
    return False


# Example usage:
if __name__ == "__main__":
    # One-line capture
    quick_capture("test.jpg")
    
    # One-line scan
    quick_scan("test.ply")
    
    # Simple usage
    unlook = UnlookSimple(debug=True)
    if unlook.connect():
        print("Connected!")
        
        # Capture image
        img = unlook.capture()
        print(f"Captured: {img.shape if img is not None else 'Failed'}")
        
        # 3D scan
        pc = unlook.scan_3d()
        print(f"Scanned: {len(pc.points) if pc else 0} points")
        
        unlook.disconnect()