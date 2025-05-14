#!/usr/bin/env python3
"""
Debug triangulation and correspondence issues.
This tool helps identify why the point cloud is just random points in space.
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory  
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TriangulationDebugger:
    """Debug triangulation and correspondence issues."""
    
    def __init__(self, calibration_file=None):
        self.calibration_data = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.R1 = None
        self.R2 = None
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.image_size = None
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, calib_path):
        """Load and validate calibration data."""
        print(f"\nLoading calibration from: {calib_path}")
        
        try:
            with open(calib_path, 'r') as f:
                self.calibration_data = json.load(f)
            
            # Load essential matrices
            self.P1 = np.array(self.calibration_data.get('P1', []))
            self.P2 = np.array(self.calibration_data.get('P2', []))
            self.Q = np.array(self.calibration_data.get('Q', []))
            
            # Load rectification matrices if available
            self.R1 = np.array(self.calibration_data.get('R1', [])) if 'R1' in self.calibration_data else None
            self.R2 = np.array(self.calibration_data.get('R2', [])) if 'R2' in self.calibration_data else None
            
            # Load camera matrices and distortion
            self.K1 = np.array(self.calibration_data.get('camera_matrix_left', []))
            self.K2 = np.array(self.calibration_data.get('camera_matrix_right', []))
            self.D1 = np.array(self.calibration_data.get('dist_coeffs_left', []))
            self.D2 = np.array(self.calibration_data.get('dist_coeffs_right', []))
            
            # Get image size
            if 'image_size' in self.calibration_data:
                self.image_size = tuple(self.calibration_data['image_size'])
            else:
                self.image_size = (1280, 720)  # Default
            
            print("Calibration loaded successfully")
            self.validate_calibration()
            
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            self.use_default_calibration()
    
    def validate_calibration(self):
        """Validate calibration data."""
        print("\nValidating calibration...")
        
        # Check P1 and P2
        if self.P1.shape != (3, 4) or self.P2.shape != (3, 4):
            print(f"ERROR: Invalid P matrix shapes: P1={self.P1.shape}, P2={self.P2.shape}")
            return
        
        # Extract focal lengths and baseline
        fx1 = self.P1[0, 0]
        fx2 = self.P2[0, 0]
        baseline = -self.P2[0, 3] / fx2  # in meters
        
        print(f"Focal length (left): {fx1:.1f} pixels")
        print(f"Focal length (right): {fx2:.1f} pixels")
        print(f"Baseline: {baseline*1000:.1f} mm")
        
        # Check if baseline is reasonable (between 40mm and 200mm)
        if baseline < 0.04 or baseline > 0.2:
            print(f"WARNING: Baseline {baseline*1000:.1f}mm seems unrealistic!")
        
        # Check Q matrix
        if self.Q is not None and self.Q.shape == (4, 4):
            print(f"Q matrix available for disparity-to-depth mapping")
            # Q[2,3] should be -f (negative focal length)
            # Q[3,3] should be 0
            print(f"Q[2,3] = {self.Q[2,3]} (should be -f)")
            print(f"Q[3,3] = {self.Q[3,3]} (should be 0)")
    
    def use_default_calibration(self):
        """Use reasonable default calibration."""
        print("\nUsing default calibration...")
        
        # Default parameters
        fx = 800.0
        fy = 800.0
        cx = 640.0
        cy = 360.0
        baseline = 0.08  # 80mm in meters
        
        self.K1 = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.K2 = self.K1.copy()
        
        self.D1 = np.zeros(5)
        self.D2 = np.zeros(5)
        
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
        
        self.image_size = (1280, 720)
    
    def test_correspondence_quality(self, pts_left, pts_right):
        """Test the quality of correspondences."""
        print(f"\nTesting {len(pts_left)} correspondences...")
        
        if len(pts_left) < 8:
            print("ERROR: Not enough correspondences for analysis")
            return None
        
        # Compute fundamental matrix
        F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 3.0)
        
        if F is None:
            print("ERROR: Failed to compute fundamental matrix")
            return None
        
        # Count inliers
        inliers = mask.ravel() == 1
        num_inliers = np.sum(inliers)
        print(f"RANSAC inliers: {num_inliers}/{len(pts_left)} ({100*num_inliers/len(pts_left):.1f}%)")
        
        # Compute epipolar error for inliers
        pts_left_in = pts_left[inliers]
        pts_right_in = pts_right[inliers]
        
        errors = []
        for i in range(len(pts_left_in)):
            pt1 = np.array([pts_left_in[i][0], pts_left_in[i][1], 1])
            pt2 = np.array([pts_right_in[i][0], pts_right_in[i][1], 1])
            
            # Epipolar constraint: pt2^T * F * pt1 = 0
            error = abs(np.dot(pt2, np.dot(F, pt1)))
            errors.append(error)
        
        errors = np.array(errors)
        print(f"Epipolar error - Mean: {np.mean(errors):.3f}, Max: {np.max(errors):.3f}")
        
        # Check disparity distribution
        disparities = pts_left_in[:, 0] - pts_right_in[:, 0]
        print(f"Disparity range: {np.min(disparities):.1f} to {np.max(disparities):.1f} pixels")
        print(f"Mean disparity: {np.mean(disparities):.1f} pixels")
        
        # Vertical alignment check
        y_differences = pts_left_in[:, 1] - pts_right_in[:, 1]
        print(f"Vertical alignment - Mean: {np.mean(np.abs(y_differences)):.1f}, Max: {np.max(np.abs(y_differences)):.1f} pixels")
        
        return pts_left[inliers], pts_right[inliers]
    
    def triangulate_and_validate(self, pts_left, pts_right):
        """Triangulate points and validate the results."""
        print("\nTriangulating points...")
        
        # Convert to homogeneous coordinates
        pts_left_h = cv2.convertPointsToHomogeneous(pts_left)[:, 0, :]
        pts_right_h = cv2.convertPointsToHomogeneous(pts_right)[:, 0, :]
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                         pts_left_h[:, :2].T, 
                                         pts_right_h[:, :2].T)
        
        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T
        
        # Validate triangulated points
        print(f"\nValidating {len(points_3d)} triangulated points...")
        
        # Check depth values
        depths = points_3d[:, 2]
        print(f"Depth range: {np.min(depths):.3f} to {np.max(depths):.3f} meters")
        print(f"Mean depth: {np.mean(depths):.3f} meters")
        
        # Filter out points with unrealistic depths
        valid_depth = (depths > 0.1) & (depths < 10.0)  # Between 10cm and 10m
        print(f"Points with valid depth: {np.sum(valid_depth)}/{len(depths)}")
        
        # Check reprojection error
        errors = self.compute_reprojection_error(points_3d, pts_left, pts_right)
        print(f"Reprojection error - Mean: {np.mean(errors):.1f}, Max: {np.max(errors):.1f} pixels")
        
        # Filter based on reprojection error
        valid_reproj = errors < 5.0  # Less than 5 pixels error
        print(f"Points with valid reprojection: {np.sum(valid_reproj)}/{len(errors)}")
        
        # Combined filter
        valid = valid_depth & valid_reproj
        points_3d_valid = points_3d[valid]
        
        print(f"\nFinal valid points: {len(points_3d_valid)}/{len(points_3d)}")
        
        # Check 3D distribution
        if len(points_3d_valid) > 0:
            x_range = np.max(points_3d_valid[:, 0]) - np.min(points_3d_valid[:, 0])
            y_range = np.max(points_3d_valid[:, 1]) - np.min(points_3d_valid[:, 1])
            z_range = np.max(points_3d_valid[:, 2]) - np.min(points_3d_valid[:, 2])
            
            print(f"3D bounding box: {x_range:.3f} x {y_range:.3f} x {z_range:.3f} meters")
            
            # Convert to mm for better readability
            print(f"3D bounding box: {x_range*1000:.1f} x {y_range*1000:.1f} x {z_range*1000:.1f} mm")
        
        return points_3d_valid
    
    def compute_reprojection_error(self, points_3d, pts_left, pts_right):
        """Compute reprojection error for triangulated points."""
        # Project 3D points back to image planes
        pts_proj_left, _ = cv2.projectPoints(points_3d, 
                                            np.zeros(3), np.zeros(3), 
                                            self.K1, self.D1)
        pts_proj_right, _ = cv2.projectPoints(points_3d, 
                                             np.zeros(3), np.array([0, 0, 0]), 
                                             self.K2, self.D2)
        
        # Compute errors
        pts_proj_left = pts_proj_left.reshape(-1, 2)
        pts_proj_right = pts_proj_right.reshape(-1, 2)
        
        error_left = np.linalg.norm(pts_left - pts_proj_left, axis=1)
        error_right = np.linalg.norm(pts_right - pts_proj_right, axis=1)
        
        return (error_left + error_right) / 2
    
    def test_simple_stereo_matching(self, img_left, img_right):
        """Test with simple block matching to verify calibration."""
        print("\nTesting with stereo block matching...")
        
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
        
        # Compute disparity
        disparity = stereo.compute(gray_left, gray_right)
        
        # Filter valid disparities
        valid_disp = disparity > 0
        
        if np.sum(valid_disp) > 0:
            valid_disparities = disparity[valid_disp]
            print(f"Disparity range: {np.min(valid_disparities)/16:.1f} to {np.max(valid_disparities)/16:.1f} pixels")
            
            # Convert to depth using Q matrix
            if self.Q is not None:
                points_3d = cv2.reprojectImageTo3D(disparity/16.0, self.Q)
                valid_points = points_3d[valid_disp]
                
                # Filter by depth
                depths = valid_points[:, 2]
                valid_depth = (depths > 100) & (depths < 5000)  # 10cm to 5m in mm
                
                print(f"Depth range: {np.min(depths[valid_depth]):.1f} to {np.max(depths[valid_depth]):.1f} mm")
                
                # Save disparity map
                disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                cv2.imwrite("debug_disparity.png", disp_norm)
                print("Saved disparity map to debug_disparity.png")
    
    def debug_full_pipeline(self, image_dir):
        """Debug the full scanning pipeline."""
        pattern_dir = Path(image_dir) / "01_patterns" / "raw"
        
        print("\nDEBUGGING FULL PIPELINE")
        print("="*50)
        
        # Load test images
        img_left = cv2.imread(str(pattern_dir / "ambient_left.png"))
        img_right = cv2.imread(str(pattern_dir / "ambient_right.png"))
        
        if img_left is None or img_right is None:
            print("Failed to load ambient images, trying patterns...")
            img_left = cv2.imread(str(pattern_dir / "gray_h_bit00_left.png"))
            img_right = cv2.imread(str(pattern_dir / "gray_h_bit00_right.png"))
        
        if img_left is None or img_right is None:
            print("ERROR: Failed to load any images")
            return
        
        print(f"Image size: {img_left.shape}")
        
        # Test 1: Block matching
        self.test_simple_stereo_matching(img_left, img_right)
        
        # Test 2: Feature matching
        print("\nTesting feature matching...")
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Try SIFT
        sift = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = sift.detectAndCompute(gray_left, None)
        kp2, des2 = sift.detectAndCompute(gray_right, None)
        
        print(f"Found {len(kp1)} features in left, {len(kp2)} in right")
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
        
        if len(good_matches) > 10:
            # Extract points
            pts_left = np.array([kp1[m.queryIdx].pt for m in good_matches])
            pts_right = np.array([kp2[m.trainIdx].pt for m in good_matches])
            
            # Test correspondence quality
            pts_left_good, pts_right_good = self.test_correspondence_quality(pts_left, pts_right)
            
            if pts_left_good is not None and len(pts_left_good) > 10:
                # Triangulate
                points_3d = self.triangulate_and_validate(pts_left_good, pts_right_good)
                
                # Save results
                if len(points_3d) > 0:
                    self.save_debug_output(img_left, img_right, pts_left_good, pts_right_good, points_3d)
                    print(f"\nSuccessfully created {len(points_3d)} 3D points")
                else:
                    print("\nERROR: No valid 3D points after filtering")
            else:
                print("\nERROR: Not enough good correspondences after filtering")
        else:
            print("\nERROR: Not enough initial matches")
    
    def save_debug_output(self, img_left, img_right, pts_left, pts_right, points_3d):
        """Save debug visualization."""
        # Create correspondence visualization
        h, w = img_left.shape[:2]
        combined = np.hstack([img_left, img_right])
        
        # Draw matches
        for i in range(0, len(pts_left), 5):
            pt1 = tuple(map(int, pts_left[i]))
            pt2 = tuple(map(int, pts_right[i] + [w, 0]))
            
            color = (0, 255, 0)
            cv2.circle(combined, pt1, 5, color, -1)
            cv2.circle(combined, pt2, 5, color, -1)
            cv2.line(combined, pt1, pt2, color, 2)
        
        cv2.imwrite("debug_correspondences.jpg", combined)
        print("Saved debug_correspondences.jpg")
        
        # Save 3D points
        with open("debug_points.ply", 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for p in points_3d:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        
        print("Saved debug_points.ply")
        
        # Plot 3D distribution
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(131)
        ax1.scatter(points_3d[:, 0], points_3d[:, 2], s=1)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Side View')
        ax1.grid(True)
        
        ax2 = fig.add_subplot(132)
        ax2.scatter(points_3d[:, 0], points_3d[:, 1], s=1)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View')
        ax2.grid(True)
        
        ax3 = fig.add_subplot(133)
        ax3.scatter(points_3d[:, 1], points_3d[:, 2], s=1)
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Front View')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('debug_3d_distribution.png', dpi=150)
        plt.close()
        print("Saved debug_3d_distribution.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug triangulation issues")
    parser.add_argument("image_dir", help="Directory with captured images")
    parser.add_argument("--calibration", help="Calibration file")
    
    args = parser.parse_args()
    
    debugger = TriangulationDebugger(calibration_file=args.calibration)
    debugger.debug_full_pipeline(args.image_dir)


if __name__ == "__main__":
    main()