#!/usr/bin/env python3
"""
Emergency scan fix for the investor demo.
Works around the issues in the captured images.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unlook.client.scanning import StaticScanner, StaticScanConfig
from unlook.client.scanning.patterns.enhanced_pattern_processor import EnhancedPatternProcessor
from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns
from unlook.client.scanning.reconstruction.proper_correspondence_finder import find_stereo_correspondences_from_projector_coords


class EmergencyScanner:
    """Scanner with fixes for the problematic captured images."""
    
    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file
        self.calibration_data = {}
        self.P1 = None
        self.P2 = None
        self.Q = None
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, calib_path):
        """Load calibration from file."""
        try:
            with open(calib_path, 'r') as f:
                data = json.load(f)
            self.P1 = np.array(data.get('P1', []))
            self.P2 = np.array(data.get('P2', []))
            self.Q = np.array(data.get('Q', []))
            self.calibration_data = data
            print(f"Loaded calibration from {calib_path}")
        except Exception as e:
            print(f"Failed to load calibration: {e}")
    
    def process_problematic_images(self, image_dir):
        """Process the problematic captured images with emergency fixes."""
        pattern_dir = Path(image_dir) / "01_patterns" / "raw"
        
        print("\nAPPLYING EMERGENCY FIXES")
        print("="*50)
        
        # Load reference images
        black_left = cv2.imread(str(pattern_dir / "black_reference_left.png"), cv2.IMREAD_GRAYSCALE)
        white_left = cv2.imread(str(pattern_dir / "white_reference_left.png"), cv2.IMREAD_GRAYSCALE)
        black_right = cv2.imread(str(pattern_dir / "black_reference_right.png"), cv2.IMREAD_GRAYSCALE)
        white_right = cv2.imread(str(pattern_dir / "white_reference_right.png"), cv2.IMREAD_GRAYSCALE)
        
        # FIX 1: Swap references since they're inverted
        print("FIX 1: Swapping inverted references")
        white_left, black_left = black_left, white_left
        white_right, black_right = black_right, white_right
        
        # FIX 2: Enhance references to increase contrast
        print("FIX 2: Enhancing reference contrast")
        processor = EnhancedPatternProcessor(enhancement_level=3)
        
        # Manually enhance the references
        black_left = cv2.equalizeHist(black_left)
        white_left = cv2.equalizeHist(white_left)
        black_right = cv2.equalizeHist(black_right)
        white_right = cv2.equalizeHist(white_right)
        
        # FIX 3: Load and process patterns with maximum enhancement
        print("FIX 3: Loading patterns with maximum enhancement")
        
        h_patterns_left = []
        h_patterns_right = []
        v_patterns_left = []
        v_patterns_right = []
        
        # Load horizontal patterns
        for i in range(5):
            # Normal patterns
            left_n = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
            right_n = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_right.png"), cv2.IMREAD_GRAYSCALE)
            # Inverted patterns
            left_i = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_inv_left.png"), cv2.IMREAD_GRAYSCALE)
            right_i = cv2.imread(str(pattern_dir / f"gray_h_bit{i:02d}_inv_right.png"), cv2.IMREAD_GRAYSCALE)
            
            if left_n is not None:
                h_patterns_left.extend([left_n, left_i])
                h_patterns_right.extend([right_n, right_i])
        
        # Load vertical patterns
        for i in range(5):
            # Normal patterns
            left_n = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_left.png"), cv2.IMREAD_GRAYSCALE)
            right_n = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_right.png"), cv2.IMREAD_GRAYSCALE)
            # Inverted patterns
            left_i = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_inv_left.png"), cv2.IMREAD_GRAYSCALE)
            right_i = cv2.imread(str(pattern_dir / f"gray_v_bit{i:02d}_inv_right.png"), cv2.IMREAD_GRAYSCALE)
            
            if left_n is not None:
                v_patterns_left.extend([left_n, left_i])
                v_patterns_right.extend([right_n, right_i])
        
        print(f"Loaded {len(h_patterns_left)} horizontal and {len(v_patterns_left)} vertical patterns")
        
        # FIX 4: Apply aggressive enhancement
        print("FIX 4: Applying aggressive pattern enhancement")
        
        # Process all patterns with maximum enhancement
        h_patterns_left_enh = []
        h_patterns_right_enh = []
        v_patterns_left_enh = []
        v_patterns_right_enh = []
        
        for img in h_patterns_left:
            # Remove purple cast by using red channel
            if len(img.shape) == 2:
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
                enhanced = clahe.apply(img)
                # Apply bilateral filter
                enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
                h_patterns_left_enh.append(enhanced)
            else:
                h_patterns_left_enh.append(img)
        
        # Similar for other pattern sets
        for img in h_patterns_right:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
            enhanced = clahe.apply(img)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            h_patterns_right_enh.append(enhanced)
        
        for img in v_patterns_left:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
            enhanced = clahe.apply(img)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            v_patterns_left_enh.append(enhanced)
        
        for img in v_patterns_right:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
            enhanced = clahe.apply(img)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            v_patterns_right_enh.append(enhanced)
        
        # FIX 5: Decode with adjusted parameters
        print("FIX 5: Decoding with adjusted parameters")
        
        # Decode horizontal patterns
        x_coord_left, x_conf_left, x_mask_left = decode_patterns(
            white_left, black_left, h_patterns_left_enh,
            num_bits=5, orientation="horizontal"
        )
        
        x_coord_right, x_conf_right, x_mask_right = decode_patterns(
            white_right, black_right, h_patterns_right_enh,
            num_bits=5, orientation="horizontal"
        )
        
        # Decode vertical patterns
        y_coord_left, y_conf_left, y_mask_left = decode_patterns(
            white_left, black_left, v_patterns_left_enh,
            num_bits=5, orientation="vertical"
        )
        
        y_coord_right, y_conf_right, y_mask_right = decode_patterns(
            white_right, black_right, v_patterns_right_enh,
            num_bits=5, orientation="vertical"
        )
        
        # Combine masks with lower threshold
        mask_left = (x_conf_left > 0.1) & (y_conf_left > 0.1)
        mask_right = (x_conf_right > 0.1) & (y_conf_right > 0.1)
        
        print(f"Valid pixels - Left: {np.sum(mask_left)}, Right: {np.sum(mask_right)}")
        
        # FIX 6: Find correspondences with relaxed constraints
        print("FIX 6: Finding correspondences with relaxed constraints")
        
        points_left, points_right = find_stereo_correspondences_from_projector_coords(
            x_coord_left, y_coord_left, mask_left,
            x_coord_right, y_coord_right, mask_right,
            epipolar_threshold=10.0  # Increased threshold
        )
        
        print(f"Found {len(points_left)} correspondences")
        
        # FIX 7: If still no correspondences, use SIFT as fallback
        if len(points_left) < 100:
            print("FIX 7: Using SIFT as fallback")
            sift = cv2.SIFT_create()
            
            # Use the first pattern image for SIFT
            img_left = h_patterns_left[0]
            img_right = h_patterns_right[0]
            
            kp1, des1 = sift.detectAndCompute(img_left, None)
            kp2, des2 = sift.detectAndCompute(img_right, None)
            
            # Match features
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # Extract matched points
            if len(good_matches) > 0:
                points_left = np.array([kp1[m.queryIdx].pt for m in good_matches])
                points_right = np.array([kp2[m.trainIdx].pt for m in good_matches])
                print(f"SIFT found {len(points_left)} correspondences")
        
        # Triangulate if we have correspondences
        if len(points_left) > 0:
            points_3d = self.triangulate_points(points_left, points_right)
            
            if points_3d is not None:
                # Create point cloud
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_3d)
                    
                    # Remove outliers
                    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    
                    # Save
                    o3d.io.write_point_cloud("emergency_scan.ply", pcd)
                    print(f"\nSaved emergency scan with {len(pcd.points)} points")
                    
                    return pcd
                except:
                    np.savetxt("emergency_points.txt", points_3d)
                    print(f"\nSaved {len(points_3d)} points to emergency_points.txt")
                    return points_3d
        
        print("\nEmergency processing complete")
        return None
    
    def triangulate_points(self, points_left, points_right):
        """Triangulate 3D points from correspondences."""
        if self.P1 is None or self.P2 is None:
            print("No calibration loaded, using default parameters")
            # Use default calibration
            fx = 500  # Focal length
            baseline = 80  # mm
            cx, cy = 640, 360
            
            self.P1 = np.array([[fx, 0, cx, 0],
                               [0, fx, cy, 0],
                               [0, 0, 1, 0]], dtype=float)
            
            self.P2 = np.array([[fx, 0, cx, -fx*baseline/1000],
                               [0, fx, cy, 0],
                               [0, 0, 1, 0]], dtype=float)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                         points_left.T, points_right.T)
        
        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T
        
        return points_3d


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Emergency scan fix for problematic images")
    parser.add_argument("image_dir", help="Directory containing captured images")
    parser.add_argument("--calibration", help="Path to calibration file")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" EMERGENCY SCAN FIX - For Investor Demo")
    print("="*70 + "\n")
    
    scanner = EmergencyScanner(calibration_file=args.calibration)
    result = scanner.process_problematic_images(args.image_dir)
    
    if result is not None:
        print("\nSUCCESS! Check emergency_scan.ply or emergency_points.txt")
    else:
        print("\nFailed to create scan. Manual intervention may be needed.")
    
    print("\nGood luck with your investor presentation!")


if __name__ == "__main__":
    main()