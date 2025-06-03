#!/usr/bin/env python3
"""
Advanced Disparity Analyzer - Diagnose and fix flat point cloud problem.

This script analyzes why the StereoBM produces flat point clouds (all same Z)
and implements improved disparity computation with proper depth variation.

Problem identified: All points have same Z coordinate (77.604mm)
Root cause: Poor disparity estimation - need better stereo matching algorithms
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedDisparityAnalyzer:
    """
    Advanced disparity analysis to fix flat point cloud problem.
    """
    
    def __init__(self, capture_dir="captured_data/20250531_005620"):
        self.capture_dir = Path(capture_dir)
        self.calib_file = "unlook/calibration/custom/stereo_calibration_fixed.json"
        self.output_dir = self.capture_dir / "advanced_disparity_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load calibration
        self.load_calibration()
        
    def load_calibration(self):
        """Load stereo calibration."""
        with open(self.calib_file, 'r') as f:
            calib = json.load(f)
        
        self.K1 = np.array(calib['camera_matrix_left'])
        self.K2 = np.array(calib['camera_matrix_right'])
        self.D1 = np.array(calib['dist_coeffs_left']).flatten()
        self.D2 = np.array(calib['dist_coeffs_right']).flatten()
        self.R = np.array(calib['R'])
        self.T = np.array(calib['T']).flatten()
        
        self.baseline_mm = np.linalg.norm(self.T) * 1000.0
        self.focal_length = self.K1[0, 0]
        
        logger.info(f"Calibration: baseline={self.baseline_mm:.1f}mm, focal={self.focal_length:.1f}px")
        
    def load_best_stereo_pair(self):
        """Load the best stereo image pair."""
        # Find best phase shift images
        left_files = list(self.capture_dir.glob("left_*phase_shift_f8_s0*"))
        if not left_files:
            left_files = list(self.capture_dir.glob("left_*phase_shift_f1_s0*"))
        
        if not left_files:
            raise FileNotFoundError("No suitable stereo images found")
        
        left_file = left_files[0]
        right_file = Path(str(left_file).replace("left_", "right_"))
        
        if not right_file.exists():
            raise FileNotFoundError(f"Right image not found: {right_file}")
        
        left_img = cv2.imread(str(left_file), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_file), cv2.IMREAD_GRAYSCALE)
        
        logger.info(f"Loaded: {left_file.name} + {right_file.name}")
        logger.info(f"Resolution: {left_img.shape[1]}x{left_img.shape[0]}")
        
        return left_img, right_img
    
    def rectify_images(self, left_img, right_img):
        """Rectify stereo images."""
        h, w = left_img.shape
        
        R1, R2, P1_new, P2_new, Q_new, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T,
            alpha=0.0
        )
        
        map1x, map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1_new, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2_new, (w, h), cv2.CV_32FC1)
        
        left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect, Q_new
    
    def compute_multiple_disparity_methods(self, left_rect, right_rect):
        """Compute disparity using multiple advanced methods."""
        methods = {}
        
        # Method 1: Original StereoBM (problematic)
        logger.info("Testing StereoBM (original)...")
        stereo_bm = cv2.StereoBM_create(numDisparities=96, blockSize=15)
        stereo_bm.setPreFilterCap(31)
        stereo_bm.setTextureThreshold(10)
        stereo_bm.setUniquenessRatio(15)
        stereo_bm.setSpeckleWindowSize(50)
        stereo_bm.setSpeckleRange(2)
        
        disp_bm = stereo_bm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        methods['StereoBM_Original'] = disp_bm
        
        # Method 2: StereoBM with wider disparity range
        logger.info("Testing StereoBM (wide range)...")
        stereo_bm_wide = cv2.StereoBM_create(numDisparities=144, blockSize=21)  # Wider range, larger block
        stereo_bm_wide.setPreFilterCap(63)
        stereo_bm_wide.setTextureThreshold(5)  # More sensitive
        stereo_bm_wide.setUniquenessRatio(10)
        stereo_bm_wide.setSpeckleWindowSize(100)
        stereo_bm_wide.setSpeckleRange(4)
        
        disp_bm_wide = stereo_bm_wide.compute(left_rect, right_rect).astype(np.float32) / 16.0
        methods['StereoBM_Wide'] = disp_bm_wide
        
        # Method 3: StereoSGBM (more accurate but slower)
        logger.info("Testing StereoSGBM (high quality)...")
        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=144,  # Wider range
            blockSize=7,         # Smaller block for detail
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
            disp12MaxDiff=2,
            uniquenessRatio=5,   # More selective
            speckleWindowSize=150,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        disp_sgbm = stereo_sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        methods['StereoSGBM_HQ'] = disp_sgbm
        
        # Method 4: Multi-scale approach
        logger.info("Testing Multi-scale approach...")
        disp_multiscale = self.compute_multiscale_disparity(left_rect, right_rect)
        methods['MultiScale'] = disp_multiscale
        
        return methods
    
    def compute_multiscale_disparity(self, left_rect, right_rect):
        """Compute disparity using multi-scale approach for better depth variation."""
        scales = [1.0, 0.75, 0.5, 0.25]
        disparities = []
        weights = []
        
        for scale in scales:
            if scale != 1.0:
                new_h, new_w = int(left_rect.shape[0] * scale), int(left_rect.shape[1] * scale)
                left_scaled = cv2.resize(left_rect, (new_w, new_h))
                right_scaled = cv2.resize(right_rect, (new_w, new_h))
            else:
                left_scaled = left_rect
                right_scaled = right_rect
            
            # Adjusted parameters for each scale
            num_disp = int(96 * scale)
            if num_disp % 16 != 0:
                num_disp = ((num_disp // 16) + 1) * 16
            num_disp = max(16, num_disp)
            
            block_size = max(5, int(15 * scale))
            if block_size % 2 == 0:
                block_size += 1
            
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=8 * 3 * block_size**2,
                P2=32 * 3 * block_size**2,
                uniquenessRatio=int(10 / scale),  # More selective at higher scales
                speckleWindowSize=int(50 * scale),
                speckleRange=2
            )
            
            disp_scaled = stereo.compute(left_scaled, right_scaled).astype(np.float32) / 16.0
            
            # Resize back to original size
            if scale != 1.0:
                disp_scaled = cv2.resize(disp_scaled, (left_rect.shape[1], left_rect.shape[0]))
                disp_scaled = disp_scaled / scale  # Adjust disparity for scale
            
            disparities.append(disp_scaled)
            weights.append(scale)  # Higher weight for higher resolution
        
        # Weighted combination
        combined_disparity = np.zeros_like(disparities[0])
        total_weight = np.zeros_like(disparities[0])
        
        for disp, weight in zip(disparities, weights):
            valid_mask = disp > 0
            combined_disparity[valid_mask] += disp[valid_mask] * weight
            total_weight[valid_mask] += weight
        
        # Avoid division by zero
        valid_mask = total_weight > 0
        combined_disparity[valid_mask] /= total_weight[valid_mask]
        
        return combined_disparity
    
    def analyze_disparity_quality(self, disparity, method_name):
        """Analyze disparity map quality."""
        valid_mask = disparity > 0
        valid_disp = disparity[valid_mask]
        
        if len(valid_disp) == 0:
            return {
                'method': method_name,
                'valid_pixels': 0,
                'coverage': 0.0,
                'mean_disparity': 0,
                'std_disparity': 0,
                'min_disparity': 0,
                'max_disparity': 0,
                'depth_range_mm': 0
            }
        
        # Calculate depth range
        min_depth = (self.focal_length * self.baseline_mm) / np.max(valid_disp)
        max_depth = (self.focal_length * self.baseline_mm) / np.min(valid_disp)
        
        stats = {
            'method': method_name,
            'valid_pixels': len(valid_disp),
            'coverage': len(valid_disp) / disparity.size * 100,
            'mean_disparity': float(np.mean(valid_disp)),
            'std_disparity': float(np.std(valid_disp)),
            'min_disparity': float(np.min(valid_disp)),
            'max_disparity': float(np.max(valid_disp)),
            'depth_range_mm': float(max_depth - min_depth),
            'min_depth_mm': float(min_depth),
            'max_depth_mm': float(max_depth)
        }
        
        return stats
    
    def triangulate_with_proper_depth(self, disparity, method_name):
        """Triangulate points with proper depth calculation."""
        valid_mask = disparity > 0.5
        
        if np.sum(valid_mask) == 0:
            return np.array([]), {}
        
        h, w = disparity.shape
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        disp_valid = disparity[valid_mask]
        
        # Proper triangulation formula
        cx = self.K1[0, 2]
        cy = self.K1[1, 2]
        
        # Calculate depth (Z) from disparity
        Z = (self.focal_length * self.baseline_mm) / disp_valid
        
        # Calculate X and Y from image coordinates and depth
        X = (x_valid - cx) * Z / self.focal_length
        Y = (y_valid - cy) * Z / self.focal_length
        
        points_3d = np.column_stack([X, Y, Z])
        
        # Filter by reasonable depth range
        depth_mask = (Z > 50) & (Z < 2000)  # 5cm to 2m
        filtered_points = points_3d[depth_mask]
        
        # Calculate statistics
        if len(filtered_points) > 0:
            stats = {
                'method': method_name,
                'total_points': len(filtered_points),
                'x_range': float(np.max(filtered_points[:, 0]) - np.min(filtered_points[:, 0])),
                'y_range': float(np.max(filtered_points[:, 1]) - np.min(filtered_points[:, 1])),
                'z_range': float(np.max(filtered_points[:, 2]) - np.min(filtered_points[:, 2])),
                'mean_depth': float(np.mean(filtered_points[:, 2])),
                'std_depth': float(np.std(filtered_points[:, 2])),
                'is_flat': float(np.std(filtered_points[:, 2])) < 5.0  # Less than 5mm variation
            }
        else:
            stats = {'method': method_name, 'total_points': 0, 'is_flat': True}
        
        return filtered_points, stats
    
    def save_disparity_visualization(self, disparity, method_name):
        """Save disparity visualization."""
        if np.max(disparity) == 0:
            logger.warning(f"No disparity data for {method_name}")
            return
        
        # Normalize for visualization
        disp_vis = disparity.copy()
        disp_vis[disp_vis < 0] = 0
        disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply color map
        disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        # Save files
        cv2.imwrite(str(self.output_dir / f"disparity_{method_name}.png"), disp_norm)
        cv2.imwrite(str(self.output_dir / f"disparity_{method_name}_colored.png"), disp_colored)
        
        logger.info(f"Saved disparity visualization: {method_name}")
    
    def save_point_cloud(self, points_3d, method_name):
        """Save point cloud to PLY."""
        if len(points_3d) == 0:
            logger.warning(f"No points to save for {method_name}")
            return
        
        filename = self.output_dir / f"pointcloud_{method_name}.ply"
        
        with open(filename, 'w') as f:
            f.write("ply\\n")
            f.write("format ascii 1.0\\n")
            f.write(f"element vertex {len(points_3d)}\\n")
            f.write("property float x\\n")
            f.write("property float y\\n")
            f.write("property float z\\n")
            f.write("end_header\\n")
            
            for point in points_3d:
                f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\\n")
        
        logger.info(f"Saved point cloud: {filename} ({len(points_3d)} points)")
    
    def run_complete_analysis(self):
        """Run complete disparity analysis."""
        logger.info("🔬 ADVANCED DISPARITY ANALYSIS")
        logger.info("="*60)
        
        # Load images
        left_img, right_img = self.load_best_stereo_pair()
        
        # Rectify
        left_rect, right_rect, Q = self.rectify_images(left_img, right_img)
        
        # Compute disparities with multiple methods
        logger.info("\\nComputing disparities with multiple methods...")
        disparity_methods = self.compute_multiple_disparity_methods(left_rect, right_rect)
        
        # Analyze each method
        results = []
        
        for method_name, disparity in disparity_methods.items():
            logger.info(f"\\n--- Analyzing {method_name} ---")
            
            # Analyze disparity quality
            disp_stats = self.analyze_disparity_quality(disparity, method_name)
            
            # Triangulate points
            points_3d, point_stats = self.triangulate_with_proper_depth(disparity, method_name)
            
            # Save visualizations
            self.save_disparity_visualization(disparity, method_name)
            self.save_point_cloud(points_3d, method_name)
            
            # Combine statistics
            combined_stats = {**disp_stats, **point_stats}
            results.append(combined_stats)
            
            # Log results
            logger.info(f"Disparity coverage: {disp_stats['coverage']:.1f}%")
            logger.info(f"Depth range: {disp_stats['depth_range_mm']:.1f}mm")
            if point_stats['total_points'] > 0:
                logger.info(f"Points: {point_stats['total_points']}")
                logger.info(f"Z variation: {point_stats['z_range']:.1f}mm")
                logger.info(f"Is flat: {'YES' if point_stats['is_flat'] else 'NO'}")
            
        # Save complete analysis
        self.save_analysis_report(results)
        
        # Find best method
        best_method = self.find_best_method(results)
        logger.info(f"\\n🏆 BEST METHOD: {best_method['method']}")
        logger.info(f"Z variation: {best_method.get('z_range', 0):.1f}mm (target: >50mm)")
        logger.info(f"Coverage: {best_method['coverage']:.1f}%")
        
        return results
    
    def save_analysis_report(self, results):
        """Save detailed analysis report."""
        report_file = self.output_dir / "disparity_analysis_report.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'analysis_timestamp': '2025-01-06',
                'problem': 'Flat point clouds - all points have same Z coordinate',
                'baseline_mm': self.baseline_mm,
                'focal_length_px': self.focal_length,
                'methods_tested': len(results),
                'results': results
            }, f, indent=2)
        
        logger.info(f"Analysis report saved: {report_file}")
    
    def find_best_method(self, results):
        """Find the method with best depth variation."""
        valid_results = [r for r in results if r.get('total_points', 0) > 100]
        
        if not valid_results:
            return results[0] if results else {}
        
        # Sort by Z range (depth variation) - we want maximum variation
        best = max(valid_results, key=lambda x: x.get('z_range', 0))
        return best


def main():
    """Main function."""
    analyzer = AdvancedDisparityAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\\n" + "="*60)
    print("ANALYSIS COMPLETED!")
    print("="*60)
    print("Check the 'advanced_disparity_analysis' folder for:")
    print("  - Disparity visualizations")
    print("  - Point cloud PLY files")
    print("  - Complete analysis report")
    print("\\nNext: Use the best method to fix the flat point cloud problem!")


if __name__ == "__main__":
    main()