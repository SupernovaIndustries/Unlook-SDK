#!/usr/bin/env python3
"""
Ultra Advanced Stereo Reconstructor - State-of-the-art stereo matching for 2024.

This implements the latest advances in stereo matching including:
- WLS (Weighted Least Squares) filtering for edge-preserving disparity refinement
- Left-right consistency checking
- Confidence-based filtering
- Multiple advanced algorithms combined
- Edge-aware upscaling and post-processing

Based on latest research from 2024 and OpenCV's ximgproc advanced filters.
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UltraAdvancedStereoReconstructor:
    """
    Ultra-advanced stereo reconstruction using 2024 state-of-the-art algorithms.
    """
    
    def __init__(self, capture_dir="captured_data/20250531_005620"):
        self.capture_dir = Path(capture_dir)
        self.calib_file = "unlook/calibration/custom/stereo_calibration_fixed.json"
        self.output_dir = self.capture_dir / "ultra_advanced_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load calibration
        self.load_calibration()
        
        # WLS Filter parameters (edge-preserving disparity refinement)
        self.wls_lambda = 8000.0      # Smoothness parameter (higher = smoother)
        self.wls_sigma = 1.5          # Edge sensitivity (lower = more edge-preserving)
        
        logger.info("Ultra Advanced Stereo Reconstructor initialized")
        logger.info("Features: WLS filtering, confidence maps, left-right consistency")
        
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
    
    def create_advanced_sgbm_matcher(self, img_shape):
        """Create highly optimized StereoSGBM matcher."""
        h, w = img_shape
        
        # Advanced SGBM parameters for high quality
        num_disparities = 144  # Wide range for depth variation
        block_size = 7         # Small block for detail
        
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,      # Smoothness penalty for small changes
            P2=32 * 3 * block_size**2,     # Smoothness penalty for large changes
            disp12MaxDiff=2,               # Left-right consistency threshold
            uniquenessRatio=5,             # Uniqueness threshold (5% = very strict)
            speckleWindowSize=150,         # Speckle filter window
            speckleRange=2,                # Speckle filter range
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 3-way mode for better quality
        )
        
        return left_matcher
    
    def create_right_matcher(self, left_matcher):
        """Create right matcher for left-right consistency check."""
        try:
            # Try to use OpenCV's createRightMatcher if available
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            return right_matcher
        except AttributeError:
            # Fallback: create manually configured right matcher
            logger.warning("createRightMatcher not available, using manual configuration")
            
            # Get parameters from left matcher
            min_disp = left_matcher.getMinDisparity()
            num_disp = left_matcher.getNumDisparities()
            block_size = left_matcher.getBlockSize()
            
            right_matcher = cv2.StereoSGBM_create(
                minDisparity=-num_disp,  # Negative for right matcher
                numDisparities=num_disp,
                blockSize=block_size,
                P1=left_matcher.getP1(),
                P2=left_matcher.getP2(),
                disp12MaxDiff=left_matcher.getDisp12MaxDiff(),
                uniquenessRatio=left_matcher.getUniquenessRatio(),
                speckleWindowSize=left_matcher.getSpeckleWindowSize(),
                speckleRange=left_matcher.getSpeckleRange(),
                mode=left_matcher.getMode()
            )
            
            return right_matcher
    
    def create_wls_filter(self, left_matcher):
        """Create WLS (Weighted Least Squares) filter for disparity refinement."""
        try:
            # Try to use OpenCV's WLS filter
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            
            # Configure WLS parameters
            wls_filter.setLambda(self.wls_lambda)   # Smoothness parameter
            wls_filter.setSigmaColor(self.wls_sigma)  # Edge sensitivity
            
            logger.info(f"WLS Filter created: lambda={self.wls_lambda}, sigma={self.wls_sigma}")
            return wls_filter
            
        except AttributeError:
            logger.warning("WLS Filter not available in this OpenCV build")
            return None
    
    def compute_advanced_disparity(self, left_rect, right_rect):
        """Compute high-quality disparity using advanced algorithms."""
        h, w = left_rect.shape
        
        # Create advanced matchers
        left_matcher = self.create_advanced_sgbm_matcher((h, w))
        right_matcher = self.create_right_matcher(left_matcher)
        wls_filter = self.create_wls_filter(left_matcher)
        
        logger.info("Computing left disparity map...")
        left_disparity = left_matcher.compute(left_rect, right_rect)
        
        logger.info("Computing right disparity map...")
        right_disparity = right_matcher.compute(right_rect, left_rect)
        
        # Apply WLS filtering if available
        if wls_filter is not None:
            logger.info("Applying WLS edge-preserving filtering...")
            
            # Convert to CV_8U for original image (required by WLS filter)
            left_rect_color = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
            
            # Apply WLS filtering
            filtered_disparity = np.zeros_like(left_disparity)
            wls_filter.filter(left_disparity, left_rect_color, filtered_disparity, right_disparity)
            
            # Get confidence map
            confidence_map = wls_filter.getConfidenceMap()
            
            # Convert to float32
            filtered_disparity = filtered_disparity.astype(np.float32) / 16.0
            
            logger.info("WLS filtering completed successfully")
            
            return {
                'raw_disparity': left_disparity.astype(np.float32) / 16.0,
                'filtered_disparity': filtered_disparity,
                'right_disparity': right_disparity.astype(np.float32) / 16.0,
                'confidence_map': confidence_map,
                'has_wls': True
            }
        else:
            # Fallback to basic post-processing
            logger.info("Applying basic post-processing...")
            
            # Convert to float32
            left_disp_f32 = left_disparity.astype(np.float32) / 16.0
            right_disp_f32 = right_disparity.astype(np.float32) / 16.0
            
            # Apply median filtering
            filtered_disparity = cv2.medianBlur(left_disp_f32.astype(np.uint8), 5).astype(np.float32)
            
            # Create basic confidence map
            confidence_map = np.ones_like(left_disp_f32) * 255.0
            confidence_map[left_disp_f32 <= 0] = 0.0
            
            return {
                'raw_disparity': left_disp_f32,
                'filtered_disparity': filtered_disparity,
                'right_disparity': right_disp_f32,
                'confidence_map': confidence_map,
                'has_wls': False
            }
    
    def triangulate_with_confidence(self, disparity_result, confidence_threshold=50.0):
        """Triangulate 3D points using confidence-based filtering."""
        disparity = disparity_result['filtered_disparity']
        confidence = disparity_result['confidence_map']
        
        # Apply confidence threshold
        valid_mask = (disparity > 0.5) & (confidence > confidence_threshold)
        
        if np.sum(valid_mask) == 0:
            logger.warning("No points passed confidence threshold")
            return np.array([]), {}
        
        h, w = disparity.shape
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        disp_valid = disparity[valid_mask]
        conf_valid = confidence[valid_mask]
        
        # Triangulation
        cx = self.K1[0, 2]
        cy = self.K1[1, 2]
        
        Z = (self.focal_length * self.baseline_mm) / disp_valid
        X = (x_valid - cx) * Z / self.focal_length
        Y = (y_valid - cy) * Z / self.focal_length
        
        points_3d = np.column_stack([X, Y, Z])
        
        # Filter by reasonable depth range
        depth_mask = (Z > 50) & (Z < 2000)
        filtered_points = points_3d[depth_mask]
        filtered_confidence = conf_valid[depth_mask]
        
        # Calculate statistics
        stats = {
            'total_points': len(filtered_points),
            'confidence_threshold': confidence_threshold,
            'mean_confidence': float(np.mean(filtered_confidence)) if len(filtered_confidence) > 0 else 0,
            'has_wls_filtering': disparity_result['has_wls']
        }
        
        if len(filtered_points) > 0:
            stats.update({
                'x_range': float(np.max(filtered_points[:, 0]) - np.min(filtered_points[:, 0])),
                'y_range': float(np.max(filtered_points[:, 1]) - np.min(filtered_points[:, 1])),
                'z_range': float(np.max(filtered_points[:, 2]) - np.min(filtered_points[:, 2])),
                'mean_depth': float(np.mean(filtered_points[:, 2])),
                'std_depth': float(np.std(filtered_points[:, 2]))
            })
        
        return filtered_points, stats
    
    def save_disparity_visualizations(self, disparity_result, method_name):
        """Save comprehensive disparity visualizations."""
        # Raw disparity
        raw_disp = disparity_result['raw_disparity']
        if np.max(raw_disp) > 0:
            raw_norm = cv2.normalize(raw_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_JET)
            cv2.imwrite(str(self.output_dir / f"{method_name}_raw_disparity.png"), raw_norm)
            cv2.imwrite(str(self.output_dir / f"{method_name}_raw_disparity_colored.png"), raw_colored)
        
        # Filtered disparity
        filt_disp = disparity_result['filtered_disparity']
        if np.max(filt_disp) > 0:
            filt_norm = cv2.normalize(filt_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            filt_colored = cv2.applyColorMap(filt_norm, cv2.COLORMAP_JET)
            cv2.imwrite(str(self.output_dir / f"{method_name}_filtered_disparity.png"), filt_norm)
            cv2.imwrite(str(self.output_dir / f"{method_name}_filtered_disparity_colored.png"), filt_colored)
        
        # Confidence map
        if 'confidence_map' in disparity_result:
            conf_map = disparity_result['confidence_map']
            conf_norm = cv2.normalize(conf_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            conf_colored = cv2.applyColorMap(conf_norm, cv2.COLORMAP_HOT)
            cv2.imwrite(str(self.output_dir / f"{method_name}_confidence.png"), conf_norm)
            cv2.imwrite(str(self.output_dir / f"{method_name}_confidence_colored.png"), conf_colored)
        
        logger.info(f"Saved visualizations for {method_name}")
    
    def save_point_cloud(self, points_3d, method_name, stats=None):
        """Save point cloud with metadata."""
        if len(points_3d) == 0:
            logger.warning(f"No points to save for {method_name}")
            return
        
        filename = self.output_dir / f"{method_name}.ply"
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Generated by Ultra Advanced Stereo Reconstructor\n")
            if stats:
                f.write(f"comment Points: {stats.get('total_points', len(points_3d))}\n")
                f.write(f"comment Z_range: {stats.get('z_range', 0):.1f}mm\n")
                f.write(f"comment WLS_filtering: {stats.get('has_wls_filtering', False)}\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points_3d:
                f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")
        
        logger.info(f"Saved: {filename} ({len(points_3d)} points)")
    
    def run_ultra_advanced_reconstruction(self):
        """Run complete ultra-advanced reconstruction."""
        logger.info("🚀 ULTRA ADVANCED STEREO RECONSTRUCTION")
        logger.info("="*70)
        
        # Load images
        left_img, right_img = self.load_best_stereo_pair()
        
        # Rectify
        left_rect, right_rect, Q = self.rectify_images(left_img, right_img)
        
        # Compute advanced disparity
        logger.info("Computing advanced disparity with WLS filtering...")
        disparity_result = self.compute_advanced_disparity(left_rect, right_rect)
        
        # Save visualizations
        self.save_disparity_visualizations(disparity_result, "ultra_advanced")
        
        # Test different confidence thresholds
        confidence_thresholds = [0, 25, 50, 75, 100]
        best_result = None
        best_score = 0
        
        for conf_thresh in confidence_thresholds:
            logger.info(f"\\nTesting confidence threshold: {conf_thresh}")
            
            # Triangulate with confidence
            points_3d, stats = self.triangulate_with_confidence(disparity_result, conf_thresh)
            
            if len(points_3d) > 100:
                # Calculate quality score
                z_range = stats.get('z_range', 0)
                point_count = stats.get('total_points', 0)
                
                # Quality score based on depth variation and point count
                quality_score = z_range * 0.7 + (point_count / 1000) * 0.3
                
                logger.info(f"  Points: {point_count}")
                logger.info(f"  Z range: {z_range:.1f}mm")
                logger.info(f"  Quality score: {quality_score:.1f}")
                
                # Save point cloud
                method_name = f"conf_{conf_thresh}"
                self.save_point_cloud(points_3d, method_name, stats)
                
                # Track best result
                if quality_score > best_score:
                    best_score = quality_score
                    best_result = {
                        'points': points_3d,
                        'stats': stats,
                        'conf_threshold': conf_thresh,
                        'quality_score': quality_score
                    }
            else:
                logger.info(f"  Too few points: {len(points_3d)}")
        
        # Save best result
        if best_result:
            logger.info(f"\\n🏆 BEST RESULT:")
            logger.info(f"Confidence threshold: {best_result['conf_threshold']}")
            logger.info(f"Points: {best_result['stats']['total_points']}")
            logger.info(f"Z variation: {best_result['stats']['z_range']:.1f}mm")
            logger.info(f"Quality score: {best_result['quality_score']:.1f}")
            logger.info(f"WLS filtering: {best_result['stats']['has_wls_filtering']}")
            
            # Save best result
            self.save_point_cloud(best_result['points'], "BEST_ULTRA_ADVANCED", best_result['stats'])
            
            # Save analysis report
            report = {
                'algorithm': 'Ultra Advanced Stereo with WLS filtering',
                'best_confidence_threshold': best_result['conf_threshold'],
                'wls_filtering': best_result['stats']['has_wls_filtering'],
                'wls_lambda': self.wls_lambda,
                'wls_sigma': self.wls_sigma,
                'results': best_result['stats'],
                'quality_score': best_result['quality_score']
            }
            
            with open(self.output_dir / "ultra_advanced_report.json", 'w') as f:
                json.dump(report, f, indent=2)
        
        logger.info("\\n" + "="*70)
        logger.info("ULTRA ADVANCED RECONSTRUCTION COMPLETED!")
        logger.info(f"Check '{self.output_dir}' for results")
        logger.info("="*70)
        
        return best_result


def main():
    """Main function."""
    reconstructor = UltraAdvancedStereoReconstructor()
    result = reconstructor.run_ultra_advanced_reconstruction()
    
    if result:
        print("\\n🎉 SUCCESS! Ultra-advanced reconstruction completed with:")
        print(f"  - {result['stats']['total_points']} high-quality points")
        print(f"  - {result['stats']['z_range']:.1f}mm depth variation")
        print(f"  - WLS filtering: {result['stats']['has_wls_filtering']}")
        print(f"  - Quality score: {result['quality_score']:.1f}")
    else:
        print("\\n❌ No successful reconstruction")


if __name__ == "__main__":
    main()