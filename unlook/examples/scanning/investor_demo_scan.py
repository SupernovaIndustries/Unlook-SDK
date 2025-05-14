#!/usr/bin/env python3
"""
Simple SIFT-based 3D reconstruction for investor demo.
This bypasses the problematic pattern decoding and uses feature matching instead.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory  
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def create_demo_scan(image_dir, calibration_file=None):
    """Create a basic 3D scan using SIFT feature matching."""
    pattern_dir = Path(image_dir) / "01_patterns" / "raw"
    
    print("\nCREATING INVESTOR DEMO SCAN")
    print("="*50)
    
    # Use the ambient images which should have the most features
    img_left = cv2.imread(str(pattern_dir / "ambient_left.png"))
    img_right = cv2.imread(str(pattern_dir / "ambient_right.png"))
    
    if img_left is None or img_right is None:
        # Fallback to first gray pattern
        img_left = cv2.imread(str(pattern_dir / "gray_h_bit00_left.png"))
        img_right = cv2.imread(str(pattern_dir / "gray_h_bit00_right.png"))
    
    if img_left is None or img_right is None:
        print("Failed to load images")
        return
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    print("Finding features with SIFT...")
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=2000)
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)
    
    print(f"Found {len(kp1)} features in left, {len(kp2)} in right")
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches")
    
    if len(good_matches) < 10:
        print("Not enough matches found")
        return
    
    # Extract matched points
    pts_left = np.array([kp1[m.queryIdx].pt for m in good_matches])
    pts_right = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    # Estimate fundamental matrix for additional filtering
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    
    # Keep only inliers
    pts_left = pts_left[mask.ravel() == 1]
    pts_right = pts_right[mask.ravel() == 1]
    
    print(f"After RANSAC: {len(pts_left)} matches")
    
    # Load calibration or use defaults
    if calibration_file:
        try:
            import json
            with open(calibration_file, 'r') as f:
                calib = json.load(f)
            P1 = np.array(calib['P1'])
            P2 = np.array(calib['P2'])
            print("Loaded calibration")
        except:
            P1, P2 = get_default_calibration()
    else:
        P1, P2 = get_default_calibration()
    
    # Triangulate points
    print("Triangulating 3D points...")
    points_4d = cv2.triangulatePoints(P1, P2, pts_left.T, pts_right.T)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    # Filter outliers
    distances = np.linalg.norm(points_3d, axis=1)
    mask = (distances > 100) & (distances < 2000)  # Keep points between 10cm and 2m
    points_3d = points_3d[mask]
    
    print(f"Generated {len(points_3d)} 3D points")
    
    # Save as simple PLY
    save_ply(points_3d, "investor_demo.ply")
    
    # Also save as point cloud visualization
    save_visualization(img_left, img_right, pts_left, pts_right)
    
    print("\nDEMO SCAN COMPLETE!")
    print("Files created:")
    print("- investor_demo.ply (3D point cloud)")
    print("- investor_demo_matches.jpg (correspondence visualization)")
    print("\nGood luck with your presentation!")


def get_default_calibration():
    """Get default calibration matrices."""
    # Reasonable defaults for stereo camera
    fx = 800  # Focal length in pixels
    cx, cy = 640, 360  # Principal point
    baseline = 80  # mm
    
    P1 = np.array([[fx, 0, cx, 0],
                   [0, fx, cy, 0],
                   [0, 0, 1, 0]], dtype=float)
    
    P2 = np.array([[fx, 0, cx, -fx*baseline/1000],
                   [0, fx, cy, 0],
                   [0, 0, 1, 0]], dtype=float)
    
    return P1, P2


def save_ply(points, filename):
    """Save points as PLY file."""
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
    
    print(f"Saved {filename}")


def save_visualization(img_left, img_right, pts_left, pts_right):
    """Save correspondence visualization."""
    # Create side-by-side image
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    h = max(h1, h2)
    
    combined = np.zeros((h, w1 + w2, 3), dtype=img_left.dtype)
    combined[:h1, :w1] = img_left
    combined[:h2, w1:w1+w2] = img_right
    
    # Draw matches
    for i in range(0, len(pts_left), 5):  # Draw every 5th match to avoid clutter
        pt1 = tuple(map(int, pts_left[i]))
        pt2 = tuple(map(int, pts_right[i] + [w1, 0]))
        
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.circle(combined, pt1, 5, color, -1)
        cv2.circle(combined, pt2, 5, color, -1)
        cv2.line(combined, pt1, pt2, color, 2)
    
    cv2.imwrite("investor_demo_matches.jpg", combined)
    print("Saved investor_demo_matches.jpg")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create demo scan for investor presentation")
    parser.add_argument("image_dir", help="Directory with captured images")
    parser.add_argument("--calibration", help="Calibration file (optional)")
    
    args = parser.parse_args()
    
    create_demo_scan(args.image_dir, args.calibration)


if __name__ == "__main__":
    main()