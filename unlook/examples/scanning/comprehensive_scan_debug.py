#!/usr/bin/env python3
"""
Comprehensive scanning and debugging tool with enhanced processor enabled by default.
This combines all the fixes and debugging tools into one comprehensive solution.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import all necessary modules
from unlook import UnlookClient
from unlook.client.scanning import StaticScanner, StaticScanConfig
from unlook.client.scanning.patterns.enhanced_pattern_processor import EnhancedPatternProcessor
from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns
from unlook.client.scanning.reconstruction.proper_correspondence_finder import find_stereo_correspondences_from_projector_coords

# Check for Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not installed. Some features will be limited.")


class ComprehensiveScanner:
    """Comprehensive scanner with all fixes and enhanced processing enabled."""
    
    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file
        self.calibration_data = {}
        self.enhanced_processor = EnhancedPatternProcessor(enhancement_level=3)
        self.load_calibration()
        
    def load_calibration(self):
        """Load calibration data with validation."""
        if self.calibration_file and os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                
                # Extract key matrices
                self.P1 = np.array(self.calibration_data.get('P1', []))
                self.P2 = np.array(self.calibration_data.get('P2', []))
                self.Q = np.array(self.calibration_data.get('Q', []))
                
                # Validate calibration
                self.validate_calibration()
                print(f"Loaded calibration from {self.calibration_file}")
                
            except Exception as e:
                print(f"Failed to load calibration: {e}")
                self.use_default_calibration()
        else:
            print("No calibration file provided, using defaults")
            self.use_default_calibration()
    
    def validate_calibration(self):
        """Validate calibration data and fix common issues."""
        # Check if white/black references need swapping
        if 'reference_swap_needed' in self.calibration_data:
            print("Calibration indicates reference swap is needed")
        
        # Extract baseline
        if self.P2.shape == (3, 4):
            fx = self.P2[0, 0]
            baseline_m = -self.P2[0, 3] / fx
            print(f"Baseline: {baseline_m * 1000:.1f} mm")
            
            # Check if baseline is reasonable
            if baseline_m < 0.04 or baseline_m > 0.2:
                print(f"WARNING: Baseline {baseline_m*1000:.1f}mm may be incorrect")
    
    def use_default_calibration(self):
        """Use reasonable default calibration."""
        fx = 800.0
        fy = 800.0
        cx = 640.0
        cy = 360.0
        baseline = 0.08  # 80mm
        
        self.P1 = np.array([[fx, 0, cx, 0],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0]])
        
        self.P2 = np.array([[fx, 0, cx, -fx*baseline],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0]])
        
        self.Q = np.array([[1, 0, 0, -cx],
                          [0, 1, 0, -cy],
                          [0, 0, 0, fx],
                          [0, 0, -1/baseline, 0]])
    
    def scan_from_hardware(self, timeout=10):
        """Perform a scan using connected hardware."""
        print("\nSCANNING FROM HARDWARE")
        print("="*50)
        
        # Create client
        client = UnlookClient(auto_discover=True)
        
        try:
            # Discover scanners
            client.start_discovery()
            print(f"Discovering scanners for {timeout} seconds...")
            time.sleep(timeout)
            
            scanners = client.get_discovered_scanners()
            if not scanners:
                print("No scanners found")
                return None
            
            # Connect to first scanner
            scanner_info = scanners[0]
            print(f"Connecting to {scanner_info.name}...")
            
            if not client.connect(scanner_info):
                print("Failed to connect")
                return None
            
            # Create scanner with enhanced processor enabled by default
            config = StaticScanConfig(
                quality="high",
                use_enhanced_processor=True,  # Already default
                enhancement_level=3,          # Maximum enhancement
                debug=True,
                save_intermediate_images=True,
                save_raw_images=True
            )
            
            scanner = StaticScanner(client=client, config=config)
            
            # Perform scan
            print("Starting enhanced scan...")
            point_cloud = scanner.perform_scan()
            
            if point_cloud and hasattr(point_cloud, 'points') and len(point_cloud.points) > 0:
                filename = f"enhanced_scan_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                scanner.save_point_cloud(filename)
                print(f"Saved scan with {len(point_cloud.points)} points to {filename}")
                return point_cloud
            else:
                print("Scan failed to produce points")
                return None
                
        except Exception as e:
            print(f"Error during scanning: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            client.disconnect()
    
    def scan_from_images(self, image_dir):
        """Process pre-captured images with maximum enhancement."""
        print("\nPROCESSING PRE-CAPTURED IMAGES")
        print("="*50)
        
        pattern_dir = Path(image_dir) / "01_patterns" / "raw"
        if not pattern_dir.exists():
            pattern_dir = Path(image_dir)
        
        # Load images with proper naming
        images_left = []
        images_right = []
        
        # Load reference images
        black_left = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_GRAYSCALE)
        white_left = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_GRAYSCALE)
        black_right = cv2.imread(str(pattern_dir / "black_reference_right.png"), cv2.IMREAD_GRAYSCALE)
        white_right = cv2.imread(str(pattern_dir / "white_reference_right.png"), cv2.IMREAD_GRAYSCALE)
        
        if black_left is None or white_left is None:
            print("Failed to load reference images")
            return None
        
        # Check if references need swapping (white darker than black)
        if np.mean(white_left) < np.mean(black_left):
            print("Swapping inverted references")
            white_left, black_left = black_left, white_left
            white_right, black_right = black_right, white_right
        
        # Add references to image lists
        images_left.extend([black_left, white_left])
        images_right.extend([black_right, white_right])
        
        # Load Gray code patterns
        print("Loading Gray code patterns...")
        
        # Horizontal patterns (bits 0-4)
        for i in range(5):
            normal_left = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
            inv_left = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_inv_left.png"), cv2.IMREAD_GRAYSCALE)
            normal_right = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_right.png"), cv2.IMREAD_GRAYSCALE)
            inv_right = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_inv_right.png"), cv2.IMREAD_GRAYSCALE)
            
            if normal_left is not None:
                images_left.extend([normal_left, inv_left])
                images_right.extend([normal_right, inv_right])
        
        # Vertical patterns (bits 5-9, but files start at 05)
        for i in range(5, 10):
            normal_left = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
            inv_left = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_inv_left.png"), cv2.IMREAD_GRAYSCALE)
            normal_right = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_right.png"), cv2.IMREAD_GRAYSCALE)
            inv_right = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_inv_right.png"), cv2.IMREAD_GRAYSCALE)
            
            if normal_left is not None:
                images_left.extend([normal_left, inv_left])
                images_right.extend([normal_right, inv_right])
        
        print(f"Loaded {len(images_left)} images per camera")
        
        # Process with enhanced processor
        print("Applying enhanced processing...")
        processed_left = self.enhanced_processor.preprocess_images(images_left[2:], black_left, white_left)
        processed_right = self.enhanced_processor.preprocess_images(images_right[2:], black_right, white_right)
        
        # Decode patterns
        print("Decoding patterns...")
        
        # Combine processed and reference images
        all_left = [black_left, white_left] + processed_left
        all_right = [black_right, white_right] + processed_right
        
        # Decode horizontal and vertical separately
        h_patterns_left = all_left[2:12]  # 10 horizontal patterns
        h_patterns_right = all_right[2:12]
        v_patterns_left = all_left[12:22]  # 10 vertical patterns  
        v_patterns_right = all_right[12:22]
        
        # Decode
        x_coord_left, x_conf_left, x_mask_left = decode_patterns(
            white_left, black_left, h_patterns_left,
            num_bits=5, orientation="horizontal"
        )
        
        x_coord_right, x_conf_right, x_mask_right = decode_patterns(
            white_right, black_right, h_patterns_right,
            num_bits=5, orientation="horizontal"
        )
        
        y_coord_left, y_conf_left, y_mask_left = decode_patterns(
            white_left, black_left, v_patterns_left,
            num_bits=5, orientation="vertical"
        )
        
        y_coord_right, y_conf_right, y_mask_right = decode_patterns(
            white_right, black_right, v_patterns_right,
            num_bits=5, orientation="vertical"
        )
        
        # Combine masks
        mask_left = x_mask_left & y_mask_left
        mask_right = x_mask_right & y_mask_right
        
        print(f"Valid pixels - Left: {np.sum(mask_left)}, Right: {np.sum(mask_right)}")
        
        # Find correspondences
        print("Finding correspondences...")
        points_left, points_right = find_stereo_correspondences_from_projector_coords(
            x_coord_left, y_coord_left, mask_left,
            x_coord_right, y_coord_right, mask_right,
            epipolar_threshold=10.0  # Relaxed for difficult images
        )
        
        print(f"Found {len(points_left)} correspondences")
        
        if len(points_left) > 0:
            # Triangulate
            points_3d = self.triangulate_points(points_left, points_right)
            
            if points_3d is not None and len(points_3d) > 0:
                # Create point cloud
                point_cloud = self.create_point_cloud(points_3d)
                
                # Save
                filename = f"processed_scan_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                self.save_point_cloud(point_cloud, filename)
                print(f"Saved {len(points_3d)} points to {filename}")
                
                return point_cloud
        
        print("No valid point cloud generated")
        return None
    
    def triangulate_points(self, points_left, points_right):
        """Triangulate 3D points with validation."""
        # Convert to homogeneous
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                         points_left.T, points_right.T)
        
        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T
        
        # Filter by reasonable depth
        depths = points_3d[:, 2]
        valid = (depths > 0.1) & (depths < 5.0)  # 10cm to 5m
        
        print(f"Triangulated {np.sum(valid)}/{len(depths)} valid points")
        
        return points_3d[valid]
    
    def create_point_cloud(self, points_3d):
        """Create Open3D point cloud."""
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # Estimate normals
            pcd.estimate_normals()
            
            # Remove outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            return pcd
        else:
            return points_3d
    
    def save_point_cloud(self, point_cloud, filename):
        """Save point cloud to file."""
        if OPEN3D_AVAILABLE and hasattr(point_cloud, 'points'):
            o3d.io.write_point_cloud(filename, point_cloud)
        else:
            # Save as simple PLY
            points = point_cloud if isinstance(point_cloud, np.ndarray) else np.asarray(point_cloud.points)
            
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                
                for p in points:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")
    
    def debug_patterns(self, image_dir):
        """Debug pattern quality and processing."""
        print("\nDEBUGGING PATTERNS")
        print("="*50)
        
        pattern_dir = Path(image_dir) / "01_patterns" / "raw"
        
        # Load a sample pattern
        pattern = cv2.imread(str(pattern_dir / "gray_h_bit00_left.png"), cv2.IMREAD_GRAYSCALE)
        pattern_inv = cv2.imread(str(pattern_dir / "gray_h_bit00_inv_left.png"), cv2.IMREAD_GRAYSCALE)
        black_ref = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_GRAYSCALE)
        white_ref = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_GRAYSCALE)
        
        if pattern is None:
            print("Failed to load pattern images")
            return
        
        print(f"Pattern stats - Min: {np.min(pattern)}, Max: {np.max(pattern)}, Mean: {np.mean(pattern):.1f}")
        print(f"Black ref mean: {np.mean(black_ref):.1f}")
        print(f"White ref mean: {np.mean(white_ref):.1f}")
        
        # Apply enhancement
        enhanced = self.enhanced_processor.preprocess_images([pattern], black_ref, white_ref)[0]
        
        print(f"Enhanced stats - Min: {np.min(enhanced)}, Max: {np.max(enhanced)}, Mean: {np.mean(enhanced):.1f}")
        
        # Save comparison
        comparison = np.hstack([pattern, enhanced])
        cv2.imwrite("pattern_enhancement_comparison.png", comparison)
        print("Saved pattern_enhancement_comparison.png")
        
        # Test decoding
        patterns = [pattern, pattern_inv]
        enhanced_patterns = self.enhanced_processor.preprocess_images(patterns, black_ref, white_ref)
        
        # Decode single bit
        decoded, confidence, mask = self.enhanced_processor.decode_enhanced_gray_code(
            enhanced_patterns, num_bits=1, orientation="horizontal"
        )
        
        print(f"Decoded {np.sum(mask)} valid pixels ({100*np.sum(mask)/mask.size:.1f}%)")
        
        # Save decoded visualization
        decoded_vis = (decoded * 255 / np.max(decoded)).astype(np.uint8) if np.max(decoded) > 0 else decoded
        cv2.imwrite("decoded_pattern.png", decoded_vis)
        print("Saved decoded_pattern.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive scanning and debugging tool")
    
    parser.add_argument("--mode", choices=["hardware", "images", "debug"], default="hardware",
                       help="Operation mode")
    parser.add_argument("--image-dir", help="Directory with captured images (for image mode)")
    parser.add_argument("--calibration", help="Calibration file")
    parser.add_argument("--timeout", type=int, default=10,
                       help="Scanner discovery timeout")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" COMPREHENSIVE SCANNER WITH ENHANCED PROCESSING")
    print(" Enhanced processor is enabled by default at maximum level")
    print("="*70 + "\n")
    
    scanner = ComprehensiveScanner(calibration_file=args.calibration)
    
    if args.mode == "hardware":
        # Scan from connected hardware
        result = scanner.scan_from_hardware(timeout=args.timeout)
        
    elif args.mode == "images":
        # Process pre-captured images
        if not args.image_dir:
            print("ERROR: --image-dir required for image mode")
            return
        
        result = scanner.scan_from_images(args.image_dir)
        
    elif args.mode == "debug":
        # Debug patterns
        if not args.image_dir:
            print("ERROR: --image-dir required for debug mode")
            return
        
        scanner.debug_patterns(args.image_dir)
        result = None
    
    if result is not None:
        print(f"\nSUCCESS! Generated point cloud with {len(result.points) if hasattr(result, 'points') else len(result)} points")
    else:
        print("\nFailed to generate point cloud")


if __name__ == "__main__":
    main()