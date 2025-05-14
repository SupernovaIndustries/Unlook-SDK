#!/usr/bin/env python3
"""
Scan from pre-captured images for debugging and testing.
This allows you to re-process captured images without needing the hardware.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Unlook modules
from unlook.client.scanning import StaticScanner, StaticScanConfig
from unlook.client.scanning.patterns.enhanced_gray_code import decode_patterns
from unlook.client.scanning.reconstruction.proper_correspondence_finder import find_stereo_correspondences_from_projector_coords
from unlook.client.scanning.patterns.enhanced_pattern_processor import EnhancedPatternProcessor


class OfflineScanner:
    """Scanner that processes pre-captured images without hardware."""
    
    def __init__(self, config=None, calibration_file=None):
        self.config = config or StaticScanConfig()
        self.calibration_file = calibration_file
        self.calibration_data = {}
        self.P1 = None
        self.P2 = None
        self.Q = None
        
        # Load calibration if provided
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, calib_path):
        """Load calibration from file."""
        try:
            with open(calib_path, 'r') as f:
                data = json.load(f)
            
            # Load essential matrices
            self.P1 = np.array(data.get('P1', []))
            self.P2 = np.array(data.get('P2', []))
            self.Q = np.array(data.get('Q', []))
            self.calibration_data = data
            
            logger.info(f"Loaded calibration from {calib_path}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
    
    def load_images_from_directory(self, directory):
        """Load captured images from a debug directory."""
        images = {'left': [], 'right': []}
        pattern_types = []
        
        # Look for pattern subdirectory
        pattern_dir = Path(directory) / "01_patterns"
        raw_dir = pattern_dir / "raw"
        
        if not raw_dir.exists():
            raw_dir = pattern_dir  # Try pattern dir directly
        
        if not raw_dir.exists():
            raw_dir = Path(directory)  # Try base directory
        
        logger.info(f"Loading images from {raw_dir}")
        
        # Load images in order
        image_files = sorted([f for f in raw_dir.glob("*.png") if f.is_file()])
        
        for img_file in image_files:
            # Parse filename to determine camera and pattern
            filename = img_file.stem
            
            # Determine if it's left or right camera
            if "left" in filename or "camera_0" in filename:
                camera = "left"
            elif "right" in filename or "camera_1" in filename:
                camera = "right"
            else:
                # Skip if can't determine camera
                continue
            
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Failed to load {img_file}")
                continue
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            images[camera].append(img)
            
            # Try to determine pattern type from filename
            if "white" in filename:
                pattern_types.append("white")
            elif "black" in filename:
                pattern_types.append("black")
            elif "gray" in filename or "horizontal" in filename:
                pattern_types.append("gray_horizontal")
            elif "vertical" in filename:
                pattern_types.append("gray_vertical")
            elif "phase" in filename:
                pattern_types.append("phase_shift")
            else:
                pattern_types.append("unknown")
        
        logger.info(f"Loaded {len(images['left'])} left and {len(images['right'])} right images")
        
        return images, pattern_types
    
    def decode_patterns(self, left_images, right_images):
        """Decode patterns to find correspondences."""
        if len(left_images) < 22 or len(right_images) < 22:
            logger.error(f"Not enough images: {len(left_images)} left, {len(right_images)} right")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Extract reference images
        white_left = left_images[0]
        white_right = right_images[0]
        black_left = left_images[1]
        black_right = right_images[1]
        
        # Extract Gray code patterns
        h_patterns_left = left_images[2:12]
        h_patterns_right = right_images[2:12]
        v_patterns_left = left_images[12:22]
        v_patterns_right = right_images[12:22]
        
        # Apply enhanced processing if enabled
        if self.config.use_enhanced_processor:
            logger.info("Using enhanced pattern processor")
            processor = EnhancedPatternProcessor(enhancement_level=self.config.enhancement_level)
            
            # Process patterns
            h_patterns_left = processor.preprocess_images(h_patterns_left, black_left, white_left)
            v_patterns_left = processor.preprocess_images(v_patterns_left, black_left, white_left)
            h_patterns_right = processor.preprocess_images(h_patterns_right, black_right, white_right)
            v_patterns_right = processor.preprocess_images(v_patterns_right, black_right, white_right)
        
        # Decode horizontal patterns
        x_coord_left, x_conf_left, x_mask_left = decode_patterns(
            white_left, black_left, h_patterns_left,
            num_bits=5, orientation="horizontal"
        )
        
        x_coord_right, x_conf_right, x_mask_right = decode_patterns(
            white_right, black_right, h_patterns_right,
            num_bits=5, orientation="horizontal"
        )
        
        # Decode vertical patterns
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
        
        # Find correspondences
        points_left, points_right = find_stereo_correspondences_from_projector_coords(
            x_coord_left, y_coord_left, mask_left,
            x_coord_right, y_coord_right, mask_right,
            epipolar_threshold=5.0
        )
        
        logger.info(f"Found {len(points_left)} correspondences")
        
        return points_left, points_right, mask_left, mask_right
    
    def triangulate_points(self, points_left, points_right):
        """Triangulate 3D points from correspondences."""
        if self.P1 is None or self.P2 is None:
            logger.error("No calibration loaded")
            return None
        
        if len(points_left) == 0:
            logger.error("No points to triangulate")
            return None
        
        # Convert to homogeneous coordinates
        points_left_h = cv2.convertPointsToHomogeneous(points_left)[:, 0, :]
        points_right_h = cv2.convertPointsToHomogeneous(points_right)[:, 0, :]
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                         points_left_h[:, :2].T, 
                                         points_right_h[:, :2].T)
        
        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T
        
        # Apply Q matrix if available
        if self.Q is not None:
            # Using Q matrix for disparity-based reconstruction
            height, width = 720, 1280  # Default camera resolution
            
            # Create disparity from correspondences
            disparity = np.zeros((height, width), dtype=np.float32)
            for i, (left_pt, right_pt) in enumerate(zip(points_left, points_right)):
                x, y = int(left_pt[0]), int(left_pt[1])
                if 0 <= x < width and 0 <= y < height:
                    disparity[y, x] = left_pt[0] - right_pt[0]
            
            # Reproject using Q
            points_3d_q = cv2.reprojectImageTo3D(disparity, self.Q)
            
            # Extract valid points
            valid_points = []
            for i, (left_pt, _) in enumerate(zip(points_left, points_right)):
                x, y = int(left_pt[0]), int(left_pt[1])
                if 0 <= x < width and 0 <= y < height:
                    pt_3d = points_3d_q[y, x]
                    if not np.isinf(pt_3d).any():
                        valid_points.append(pt_3d)
            
            if valid_points:
                points_3d = np.array(valid_points)
        
        logger.info(f"Triangulated {len(points_3d)} points")
        return points_3d
    
    def create_point_cloud(self, points_3d):
        """Create Open3D point cloud from 3D points."""
        try:
            import open3d as o3d
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # Estimate normals
            pcd.estimate_normals()
            
            # Remove outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            return pcd
        except ImportError:
            logger.warning("Open3D not available, returning raw points")
            return points_3d
    
    def process_images(self, image_dir, output_file=None):
        """Process pre-captured images to create point cloud."""
        # Load images
        images, pattern_types = self.load_images_from_directory(image_dir)
        
        if not images['left'] or not images['right']:
            logger.error("No images found")
            return None
        
        # Decode patterns
        points_left, points_right, mask_left, mask_right = self.decode_patterns(
            images['left'], images['right']
        )
        
        # Save correspondence visualization
        if self.config.debug and len(points_left) > 0:
            self.save_correspondence_visualization(
                images['left'][0], images['right'][0],
                points_left, points_right,
                output_dir=os.path.dirname(output_file) if output_file else "."
            )
        
        # Triangulate points
        points_3d = self.triangulate_points(points_left, points_right)
        
        if points_3d is None:
            return None
        
        # Create point cloud
        point_cloud = self.create_point_cloud(points_3d)
        
        # Save if output file specified
        if output_file:
            self.save_point_cloud(point_cloud, output_file)
        
        return point_cloud
    
    def save_correspondence_visualization(self, img_left, img_right, 
                                        points_left, points_right, output_dir):
        """Save visualization of correspondences."""
        # Create side-by-side image
        height, width = img_left.shape[:2]
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Convert to color if needed
        if len(img_left.shape) == 2:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        if len(img_right.shape) == 2:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
        
        combined[:, :width] = img_left
        combined[:, width:] = img_right
        
        # Draw correspondences
        for left_pt, right_pt in zip(points_left, points_right):
            # Draw points
            cv2.circle(combined, (int(left_pt[0]), int(left_pt[1])), 3, (0, 255, 0), -1)
            cv2.circle(combined, (int(right_pt[0] + width), int(right_pt[1])), 3, (0, 255, 0), -1)
            
            # Draw line
            cv2.line(combined, 
                    (int(left_pt[0]), int(left_pt[1])),
                    (int(right_pt[0] + width), int(right_pt[1])),
                    (0, 100, 0), 1)
        
        output_path = os.path.join(output_dir, "correspondences.png")
        cv2.imwrite(output_path, combined)
        logger.info(f"Saved correspondence visualization to {output_path}")
    
    def save_point_cloud(self, point_cloud, output_file):
        """Save point cloud to file."""
        try:
            import open3d as o3d
            
            if isinstance(point_cloud, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(output_file, point_cloud)
            else:
                # Save as simple PLY
                with open(output_file, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(point_cloud)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    
                    for point in point_cloud:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            logger.info(f"Saved point cloud to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save point cloud: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process pre-captured images to create point cloud")
    
    parser.add_argument("image_dir", help="Directory containing captured images")
    parser.add_argument("--output", default="offline_scan.ply", help="Output point cloud file")
    parser.add_argument("--calibration", help="Path to calibration file")
    parser.add_argument("--enhanced-processor", action="store_true", 
                       help="Use enhanced pattern processor")
    parser.add_argument("--enhancement-level", type=int, default=2, choices=[0,1,2,3],
                       help="Enhancement level (0-3)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "="*70)
    print(" OFFLINE SCANNER - Process Pre-captured Images")
    print("="*70 + "\n")
    
    # Create config
    config = StaticScanConfig(
        use_enhanced_processor=args.enhanced_processor,
        enhancement_level=args.enhancement_level,
        debug=args.debug
    )
    
    # Create scanner
    scanner = OfflineScanner(config=config, calibration_file=args.calibration)
    
    # Process images
    print(f"Processing images from: {args.image_dir}")
    point_cloud = scanner.process_images(args.image_dir, args.output)
    
    if point_cloud is not None:
        if hasattr(point_cloud, 'points'):
            print(f"\nCreated point cloud with {len(point_cloud.points)} points")
        else:
            print(f"\nCreated point cloud with {len(point_cloud)} points")
        print(f"Saved to: {args.output}")
    else:
        print("\nFailed to create point cloud")
    
    print("\nDone!")


if __name__ == "__main__":
    main()