#!/usr/bin/env python3
"""
Balanced Stereo Reconstructor - Best balance between quality and detail.

This provides the optimal balance between surface quality and point density,
using lessons learned from all previous experiments.

- SGBM for detail + surface coherence
- Minimal WLS filtering (preserves detail)  
- Wide disparity range for depth variation
- Proper PLY format for MeshLab compatibility
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BalancedStereoReconstructor:
    """
    Balanced reconstruction with optimal quality/detail trade-off.
    """
    
    def __init__(self, capture_dir="captured_data/20250531_005620"):
        self.capture_dir = Path(capture_dir)
        self.calib_file = "unlook/calibration/custom/stereo_calibration_fixed.json"
        self.output_dir = self.capture_dir / "balanced_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load calibration
        self.load_calibration()
        
        logger.info("Balanced Stereo Reconstructor initialized")
        logger.info("Target: Best balance of quality and detail")
        
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
        left_files = list(self.capture_dir.glob("left_*phase_shift_f8_s0*"))
        if not left_files:
            left_files = list(self.capture_dir.glob("left_*phase_shift_f1_s0*"))
        
        if not left_files:
            raise FileNotFoundError("No suitable stereo images found")
        
        left_file = left_files[0]
        right_file = Path(str(left_file).replace("left_", "right_"))
        
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
    
    def compute_optimal_disparity(self, left_rect, right_rect):
        """Compute disparity with optimal parameters for quality/detail balance."""
        
        # Method 1: High-Quality SGBM (lots of detail)
        logger.info("Computing high-detail SGBM disparity...")
        
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=144,        # Wide range for depth variation
            blockSize=5,               # Small block for maximum detail
            P1=8 * 3 * 5**2,          # Smoothness for small disparity changes
            P2=32 * 3 * 5**2,         # Smoothness for large disparity changes
            disp12MaxDiff=2,           # Left-right consistency
            uniquenessRatio=5,         # Strict uniqueness (quality)
            speckleWindowSize=150,     # Remove noise clusters
            speckleRange=2,            # Speckle filtering
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Best quality mode
        )
        
        disp_sgbm = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        
        # Method 2: StereoBM (reliable surfaces)
        logger.info("Computing reliable StereoBM disparity...")
        
        stereobm = cv2.StereoBM_create(numDisparities=144, blockSize=21)
        stereobm.setPreFilterCap(63)
        stereobm.setTextureThreshold(5)
        stereobm.setUniquenessRatio(10)
        stereobm.setSpeckleWindowSize(100)
        stereobm.setSpeckleRange(4)
        
        disp_bm = stereobm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        
        # Combine both approaches: SGBM for detail, BM for reliability
        logger.info("Creating hybrid disparity map...")
        
        # Use SGBM where it's confident, BM for backup
        sgbm_valid = disp_sgbm > 0.5
        bm_valid = disp_bm > 0.5
        
        # Hybrid approach
        combined_disparity = np.zeros_like(disp_sgbm)
        
        # Prefer SGBM (more detail)
        combined_disparity[sgbm_valid] = disp_sgbm[sgbm_valid]
        
        # Fill gaps with BM (more reliable)
        gaps = ~sgbm_valid & bm_valid
        combined_disparity[gaps] = disp_bm[gaps]
        
        # Light filtering to clean up
        filtered_disparity = cv2.medianBlur(combined_disparity.astype(np.uint8), 3).astype(np.float32)
        
        # Preserve original values where they're good
        good_mask = combined_disparity > 0.5
        filtered_disparity[good_mask] = combined_disparity[good_mask]
        
        logger.info(f"Hybrid disparity: {np.sum(filtered_disparity > 0)} valid pixels")
        
        return {
            'sgbm_disparity': disp_sgbm,
            'bm_disparity': disp_bm,
            'hybrid_disparity': filtered_disparity,
            'best_disparity': filtered_disparity
        }
    
    def triangulate_points(self, disparity, method_name=""):
        """Triangulate 3D points from disparity."""
        valid_mask = disparity > 0.5
        
        if np.sum(valid_mask) == 0:
            return np.array([]), {}
        
        h, w = disparity.shape
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        disp_valid = disparity[valid_mask]
        
        # Triangulation
        cx = self.K1[0, 2]
        cy = self.K1[1, 2]
        
        Z = (self.focal_length * self.baseline_mm) / disp_valid
        X = (x_valid - cx) * Z / self.focal_length
        Y = (y_valid - cy) * Z / self.focal_length
        
        points_3d = np.column_stack([X, Y, Z])
        
        # Depth filtering
        depth_mask = (Z > 50) & (Z < 2000)
        filtered_points = points_3d[depth_mask]
        
        if len(filtered_points) == 0:
            return np.array([]), {}
        
        # Center points preserving Z variation
        centroid = np.mean(filtered_points, axis=0)
        centered_points = filtered_points.copy()
        
        # Center X and Y only
        centered_points[:, 0] -= centroid[0]
        centered_points[:, 1] -= centroid[1]
        
        # Shift Z to target but preserve variation
        target_z = 300.0
        z_shift = target_z - centroid[2]
        centered_points[:, 2] += z_shift
        
        # Light outlier removal
        center = np.mean(centered_points, axis=0)
        distances = np.sqrt(np.sum((centered_points - center)**2, axis=1))
        distance_threshold = np.mean(distances) + 3 * np.std(distances)  # More generous
        inlier_mask = distances < distance_threshold
        
        final_points = centered_points[inlier_mask]
        
        # Calculate stats
        stats = {
            'method': method_name,
            'total_points': len(final_points),
            'x_range': float(np.max(final_points[:, 0]) - np.min(final_points[:, 0])) if len(final_points) > 0 else 0,
            'y_range': float(np.max(final_points[:, 1]) - np.min(final_points[:, 1])) if len(final_points) > 0 else 0,
            'z_range': float(np.max(final_points[:, 2]) - np.min(final_points[:, 2])) if len(final_points) > 0 else 0,
            'mean_depth': float(np.mean(final_points[:, 2])) if len(final_points) > 0 else 0
        }
        
        logger.info(f"{method_name}: {stats['total_points']} points, Z range: {stats['z_range']:.1f}mm")
        
        return final_points, stats
    
    def save_disparity_visualizations(self, disparity_results):
        """Save disparity visualizations."""
        for name, disparity in disparity_results.items():
            if 'disparity' not in name:
                continue
                
            if np.max(disparity) > 0:
                disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                
                cv2.imwrite(str(self.output_dir / f"{name}.png"), disp_norm)
                cv2.imwrite(str(self.output_dir / f"{name}_colored.png"), disp_colored)
        
        logger.info("Saved disparity visualizations")
    
    def save_point_cloud(self, points_3d, filename, stats=None):
        """Save point cloud in proper PLY format."""
        if len(points_3d) == 0:
            logger.warning(f"No points to save for {filename}")
            return
        
        filepath = self.output_dir / f"{filename}.ply"
        
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Generated by Balanced Stereo Reconstructor\n")
            if stats:
                f.write(f"comment Points: {stats.get('total_points', len(points_3d))}\n")
                f.write(f"comment Z_range: {stats.get('z_range', 0):.1f}mm\n")
                f.write(f"comment Method: {stats.get('method', 'Unknown')}\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points_3d:
                f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")
        
        logger.info(f"Saved: {filepath} ({len(points_3d)} points)")
    
    def run_balanced_reconstruction(self):
        """Run balanced reconstruction."""
        logger.info("🎯 BALANCED STEREO RECONSTRUCTION")
        logger.info("="*60)
        
        # Load images
        left_img, right_img = self.load_best_stereo_pair()
        
        # Rectify
        left_rect, right_rect, Q = self.rectify_images(left_img, right_img)
        
        # Compute optimal disparity
        disparity_results = self.compute_optimal_disparity(left_rect, right_rect)
        
        # Save visualizations
        self.save_disparity_visualizations(disparity_results)
        
        # Test different methods
        results = []
        
        methods_to_test = [
            ('SGBM_Detail', disparity_results['sgbm_disparity']),
            ('StereoBM_Reliable', disparity_results['bm_disparity']),
            ('Hybrid_Best', disparity_results['hybrid_disparity'])
        ]
        
        for method_name, disparity in methods_to_test:
            points_3d, stats = self.triangulate_points(disparity, method_name)
            
            if len(points_3d) > 100:
                self.save_point_cloud(points_3d, method_name, stats)
                results.append((method_name, points_3d, stats))
        
        # Find best balance: good detail + good depth variation
        best_result = None
        best_score = 0
        
        for method_name, points_3d, stats in results:
            # Quality score: balance points and depth variation
            point_score = min(stats['total_points'] / 1000, 100)  # Normalize to 100 max
            depth_score = min(stats['z_range'] / 10, 100)        # Normalize to 100 max
            
            # Balanced score: 60% depth variation + 40% point count
            balance_score = depth_score * 0.6 + point_score * 0.4
            
            logger.info(f"{method_name}: Balance score = {balance_score:.1f}")
            
            if balance_score > best_score:
                best_score = balance_score
                best_result = (method_name, points_3d, stats)
        
        # Save best result
        if best_result:
            method_name, points_3d, stats = best_result
            
            logger.info(f"\n🏆 BEST BALANCED RESULT: {method_name}")
            logger.info(f"Points: {stats['total_points']}")
            logger.info(f"Z variation: {stats['z_range']:.1f}mm")
            logger.info(f"Balance score: {best_score:.1f}")
            
            # Save as FINAL result
            self.save_point_cloud(points_3d, "BALANCED_FINAL", stats)
            
            # Save analysis
            report = {
                'best_method': method_name,
                'balance_score': best_score,
                'stats': stats,
                'all_results': [{'method': r[0], 'stats': r[2]} for r in results]
            }
            
            with open(self.output_dir / "balanced_analysis.json", 'w') as f:
                json.dump(report, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("BALANCED RECONSTRUCTION COMPLETED!")
        logger.info(f"Check '{self.output_dir}' for results")
        logger.info("Files should now open correctly in MeshLab!")
        logger.info("="*60)
        
        return best_result


def main():
    """Main function."""
    reconstructor = BalancedStereoReconstructor()
    result = reconstructor.run_balanced_reconstruction()
    
    if result:
        method_name, points_3d, stats = result
        print(f"\n🎉 SUCCESS! Best balanced result:")
        print(f"  Method: {method_name}")
        print(f"  Points: {stats['total_points']}")
        print(f"  Z variation: {stats['z_range']:.1f}mm")
        print(f"  File: balanced_results/BALANCED_FINAL.ply")
        print("\nTry opening BALANCED_FINAL.ply in MeshLab!")
    else:
        print("\n❌ No successful reconstruction")


if __name__ == "__main__":
    main()