#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Epipolar Lines Verification Tool for UnLook Stereo Calibration

This tool performs comprehensive epipolar geometry verification to ensure
your stereo calibration is accurate. It provides both visual and numerical
feedback about calibration quality.

Usage:
    python check_epipolar_lines.py --calibration calibration_2k.json --images ../calibration_images
    python check_epipolar_lines.py --calibration calibration_2k.json --images ../calibration_images --save-report
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("epipolar_check")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


class EpipolarVerifier:
    """Comprehensive epipolar geometry verification for stereo calibration"""
    
    def __init__(self, calibration_file):
        """
        Initialize verifier with calibration data.
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_file = Path(calibration_file)
        self.load_calibration()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'calibration_file': str(calibration_file),
            'epipolar_errors': [],
            'rectification_errors': [],
            'quality_metrics': {}
        }
        
    def load_calibration(self):
        """Load calibration data from JSON file"""
        if not self.calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")
        
        with open(self.calibration_file, 'r') as f:
            self.calibration = json.load(f)
        
        # Extract calibration matrices
        self.K1 = np.array(self.calibration['K1'])
        self.D1 = np.array(self.calibration['D1'])
        self.K2 = np.array(self.calibration['K2'])
        self.D2 = np.array(self.calibration['D2'])
        self.R = np.array(self.calibration['R'])
        self.T = np.array(self.calibration['T'])
        self.E = np.array(self.calibration['E'])
        self.F = np.array(self.calibration['F'])
        self.R1 = np.array(self.calibration['R1'])
        self.R2 = np.array(self.calibration['R2'])
        self.P1 = np.array(self.calibration['P1'])
        self.P2 = np.array(self.calibration['P2'])
        self.Q = np.array(self.calibration['Q'])
        
        self.image_size = tuple(self.calibration['image_size'])
        self.baseline_mm = self.calibration['baseline_mm']
        
        # Create rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1
        )
        
        logger.info(f"Loaded calibration with baseline: {self.baseline_mm:.2f}mm")
        logger.info(f"Image size: {self.image_size}")
        
    def verify_epipolar_constraint(self, pts1, pts2, F, tolerance=1.0):
        """
        Verify epipolar constraint: p2^T * F * p1 = 0
        
        Args:
            pts1: Points in first image
            pts2: Points in second image
            F: Fundamental matrix
            tolerance: Maximum allowed error
            
        Returns:
            errors: Array of epipolar errors
            valid_mask: Boolean mask of valid correspondences
        """
        if len(pts1) == 0 or len(pts2) == 0:
            return np.array([]), np.array([])
        
        # Convert to homogeneous coordinates
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
        
        errors = []
        for p1, p2 in zip(pts1_h, pts2_h):
            # Compute epipolar line in second image: l2 = F * p1
            l2 = F @ p1
            
            # Distance from p2 to line l2
            error = abs(p2.T @ l2) / np.sqrt(l2[0]**2 + l2[1]**2)
            errors.append(error)
        
        errors = np.array(errors)
        valid_mask = errors < tolerance
        
        return errors, valid_mask
    
    def find_correspondences(self, img1, img2, method='sift'):
        """
        Find feature correspondences between two images.
        
        Args:
            img1: First image
            img2: Second image
            method: Feature detection method ('sift', 'orb', 'akaze')
            
        Returns:
            pts1: Points in first image
            pts2: Points in second image
            matches: Feature matches
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Create feature detector
        if method == 'sift':
            detector = cv2.SIFT_create(nfeatures=1000)
        elif method == 'orb':
            detector = cv2.ORB_create(nfeatures=1000)
        else:  # akaze
            detector = cv2.AKAZE_create()
        
        # Detect features
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return np.array([]), np.array([]), []
        
        # Match features
        if method == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
        else:
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for m in matches:
                if len(m) == 2:
                    m1, m2 = m
                    if m1.distance < 0.75 * m2.distance:
                        good_matches.append(m1)
            matches = good_matches
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return pts1, pts2, matches
    
    def verify_rectification_quality(self, img1_rect, img2_rect):
        """
        Verify rectification quality by checking horizontal alignment.
        
        Args:
            img1_rect: Rectified left image
            img2_rect: Rectified right image
            
        Returns:
            dict: Rectification quality metrics
        """
        # Find correspondences in rectified images
        pts1, pts2, _ = self.find_correspondences(img1_rect, img2_rect)
        
        if len(pts1) < 10:
            logger.warning("Too few correspondences found for rectification check")
            return {'status': 'insufficient_data'}
        
        # Calculate vertical disparities (should be ~0 for good rectification)
        vertical_disparities = pts2[:, 1] - pts1[:, 1]
        
        metrics = {
            'num_correspondences': len(pts1),
            'vertical_disparity_mean': np.mean(vertical_disparities),
            'vertical_disparity_std': np.std(vertical_disparities),
            'vertical_disparity_max': np.max(np.abs(vertical_disparities)),
            'vertical_disparity_rms': np.sqrt(np.mean(vertical_disparities**2))
        }
        
        # Quality assessment
        if metrics['vertical_disparity_rms'] < 0.5:
            metrics['quality'] = 'EXCELLENT'
        elif metrics['vertical_disparity_rms'] < 1.0:
            metrics['quality'] = 'GOOD'
        elif metrics['vertical_disparity_rms'] < 2.0:
            metrics['quality'] = 'FAIR'
        else:
            metrics['quality'] = 'POOR'
        
        return metrics
    
    def create_epipolar_visualization(self, img1, img2, pts1, pts2, F, output_path=None):
        """
        Create detailed epipolar line visualization.
        
        Args:
            img1, img2: Input images
            pts1, pts2: Corresponding points
            F: Fundamental matrix
            output_path: Optional path to save visualization
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display images
        ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax1.set_title('Left Image', fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax2.set_title('Right Image with Epipolar Lines', fontsize=14)
        ax2.axis('off')
        
        # Sample points for visualization
        if len(pts1) > 20:
            indices = random.sample(range(len(pts1)), 20)
            pts1_vis = pts1[indices]
            pts2_vis = pts2[indices]
        else:
            pts1_vis = pts1
            pts2_vis = pts2
        
        # Colors for different points
        colors = plt.cm.rainbow(np.linspace(0, 1, len(pts1_vis)))
        
        # Draw points and epipolar lines
        for i, (p1, p2, color) in enumerate(zip(pts1_vis, pts2_vis, colors)):
            # Draw point in left image
            ax1.scatter(p1[0], p1[1], c=[color], s=100, marker='o', edgecolors='white', linewidth=2)
            ax1.text(p1[0]+5, p1[1]-5, f'{i+1}', color='white', fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            # Draw corresponding point in right image
            ax2.scatter(p2[0], p2[1], c=[color], s=100, marker='o', edgecolors='white', linewidth=2)
            
            # Calculate epipolar line in right image
            p1_h = np.array([p1[0], p1[1], 1])
            l2 = F @ p1_h
            
            # Draw epipolar line
            x = np.array([0, img2.shape[1]])
            y = -(l2[0] * x + l2[2]) / l2[1]
            ax2.plot(x, y, color=color, linewidth=2, alpha=0.7)
            
            # Calculate and display error
            p2_h = np.array([p2[0], p2[1], 1])
            error = abs(p2_h.T @ l2) / np.sqrt(l2[0]**2 + l2[1]**2)
            
            # Draw error visualization
            # Project p2 onto epipolar line
            t = -(l2[0] * p2[0] + l2[1] * p2[1] + l2[2]) / (l2[0]**2 + l2[1]**2)
            proj_x = p2[0] + t * l2[0]
            proj_y = p2[1] + t * l2[1]
            
            # Draw error line (perpendicular to epipolar line)
            ax2.plot([p2[0], proj_x], [p2[1], proj_y], 'r-', linewidth=2)
            ax2.text(p2[0]+5, p2[1]+5, f'{error:.1f}px', color='red', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.suptitle(f'Epipolar Geometry Verification - {len(pts1)} correspondences', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved epipolar visualization to {output_path}")
        
        plt.close()
    
    def create_rectification_visualization(self, img1_rect, img2_rect, output_path=None):
        """
        Create rectification quality visualization.
        
        Args:
            img1_rect, img2_rect: Rectified images
            output_path: Optional path to save visualization
        """
        # Find correspondences
        pts1, pts2, _ = self.find_correspondences(img1_rect, img2_rect)
        
        if len(pts1) < 5:
            logger.warning("Too few correspondences for visualization")
            return
        
        # Create combined image
        h, w = img1_rect.shape[:2]
        combined = np.hstack((img1_rect, img2_rect))
        
        # Sample points for visualization
        if len(pts1) > 30:
            indices = random.sample(range(len(pts1)), 30)
            pts1_vis = pts1[indices]
            pts2_vis = pts2[indices]
        else:
            pts1_vis = pts1
            pts2_vis = pts2
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        
        # Draw horizontal reference lines
        for y in range(0, h, 50):
            ax.axhline(y=y, color='green', alpha=0.3, linewidth=1)
        
        # Draw correspondences
        colors = plt.cm.rainbow(np.linspace(0, 1, len(pts1_vis)))
        
        for i, (p1, p2, color) in enumerate(zip(pts1_vis, pts2_vis, colors)):
            # Draw points
            ax.scatter(p1[0], p1[1], c=[color], s=100, marker='o', edgecolors='white', linewidth=2)
            ax.scatter(p2[0] + w, p2[1], c=[color], s=100, marker='o', edgecolors='white', linewidth=2)
            
            # Draw connecting line
            ax.plot([p1[0], p2[0] + w], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.7)
            
            # Calculate vertical error
            vertical_error = abs(p2[1] - p1[1])
            
            # Annotate with error
            mid_x = (p1[0] + p2[0] + w) / 2
            mid_y = (p1[1] + p2[1]) / 2
            if vertical_error > 0.5:
                ax.text(mid_x, mid_y, f'{vertical_error:.1f}px', 
                       color='red', fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Add labels
        ax.text(10, 30, 'LEFT (Rectified)', color='white', fontsize=14, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        ax.text(w + 10, 30, 'RIGHT (Rectified)', color='white', fontsize=14, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # Add vertical error statistics
        vertical_errors = np.abs(pts2[:, 1] - pts1[:, 1])
        stats_text = f'Vertical Error: Mean={np.mean(vertical_errors):.2f}px, Max={np.max(vertical_errors):.2f}px'
        ax.text(w//2, h - 20, stats_text, color='white', fontsize=12, ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.7))
        
        ax.set_title('Rectification Quality Check - Lines Should Be Horizontal', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved rectification visualization to {output_path}")
        
        plt.close()
    
    def verify_image_pair(self, left_img, right_img, idx=0):
        """
        Perform comprehensive verification on an image pair.
        
        Args:
            left_img: Left image
            right_img: Right image
            idx: Image pair index
            
        Returns:
            dict: Verification results
        """
        logger.info(f"\nVerifying image pair {idx}...")
        
        results = {
            'image_idx': idx,
            'original_epipolar': {},
            'rectified_quality': {}
        }
        
        # Find correspondences in original images
        pts1, pts2, _ = self.find_correspondences(left_img, right_img)
        
        if len(pts1) > 0:
            # Verify epipolar constraint
            epipolar_errors, valid_mask = self.verify_epipolar_constraint(pts1, pts2, self.F)
            
            results['original_epipolar'] = {
                'num_correspondences': len(pts1),
                'epipolar_error_mean': np.mean(epipolar_errors),
                'epipolar_error_std': np.std(epipolar_errors),
                'epipolar_error_max': np.max(epipolar_errors),
                'epipolar_error_median': np.median(epipolar_errors),
                'valid_percentage': np.sum(valid_mask) / len(valid_mask) * 100
            }
            
            logger.info(f"  Original epipolar error: {results['original_epipolar']['epipolar_error_mean']:.2f} ± "
                       f"{results['original_epipolar']['epipolar_error_std']:.2f} pixels")
        
        # Rectify images
        left_rect = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)
        
        # Verify rectification quality
        rect_metrics = self.verify_rectification_quality(left_rect, right_rect)
        results['rectified_quality'] = rect_metrics
        
        if 'vertical_disparity_rms' in rect_metrics:
            logger.info(f"  Rectification quality: {rect_metrics['quality']} "
                       f"(RMS vertical error: {rect_metrics['vertical_disparity_rms']:.2f} pixels)")
        
        return results, pts1, pts2, left_rect, right_rect
    
    def generate_report(self, output_dir):
        """
        Generate comprehensive verification report.
        
        Args:
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create report document
        report = {
            'timestamp': self.results['timestamp'],
            'calibration_file': self.results['calibration_file'],
            'calibration_params': {
                'baseline_mm': self.baseline_mm,
                'image_size': self.image_size,
                'rms_error': self.calibration.get('rms_error', 'N/A')
            },
            'verification_results': {
                'num_images_tested': len(self.results['epipolar_errors']),
                'epipolar_analysis': {},
                'rectification_analysis': {}
            }
        }
        
        # Aggregate epipolar errors
        if self.results['epipolar_errors']:
            all_errors = []
            for result in self.results['epipolar_errors']:
                if 'original_epipolar' in result and 'epipolar_error_mean' in result['original_epipolar']:
                    all_errors.append(result['original_epipolar']['epipolar_error_mean'])
            
            if all_errors:
                report['verification_results']['epipolar_analysis'] = {
                    'mean_error': float(np.mean(all_errors)),
                    'std_error': float(np.std(all_errors)),
                    'max_error': float(np.max(all_errors)),
                    'quality': 'GOOD' if np.mean(all_errors) < 2.0 else 'NEEDS IMPROVEMENT'
                }
        
        # Aggregate rectification errors
        if self.results['rectification_errors']:
            all_rect_errors = []
            for result in self.results['rectification_errors']:
                if 'rectified_quality' in result and 'vertical_disparity_rms' in result['rectified_quality']:
                    all_rect_errors.append(result['rectified_quality']['vertical_disparity_rms'])
            
            if all_rect_errors:
                report['verification_results']['rectification_analysis'] = {
                    'mean_vertical_error': float(np.mean(all_rect_errors)),
                    'max_vertical_error': float(np.max(all_rect_errors)),
                    'quality': 'EXCELLENT' if np.mean(all_rect_errors) < 1.0 else 'GOOD' if np.mean(all_rect_errors) < 2.0 else 'NEEDS IMPROVEMENT'
                }
        
        # Save JSON report
        report_path = output_path / 'epipolar_verification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EPIPOLAR VERIFICATION SUMMARY")
        print("="*60)
        print(f"Calibration Baseline: {self.baseline_mm:.2f}mm")
        print(f"Images Tested: {len(self.results['epipolar_errors'])}")
        
        if 'mean_error' in report['verification_results']['epipolar_analysis']:
            print(f"\nEpipolar Analysis:")
            print(f"  Mean Error: {report['verification_results']['epipolar_analysis']['mean_error']:.2f} pixels")
            print(f"  Quality: {report['verification_results']['epipolar_analysis']['quality']}")
        
        if 'mean_vertical_error' in report['verification_results']['rectification_analysis']:
            print(f"\nRectification Analysis:")
            print(f"  Mean Vertical Error: {report['verification_results']['rectification_analysis']['mean_vertical_error']:.2f} pixels")
            print(f"  Quality: {report['verification_results']['rectification_analysis']['quality']}")
        
        print("="*60)
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Verify epipolar geometry of stereo calibration')
    parser.add_argument('--calibration', type=str, default='calibration_2k.json',
                        help='Path to calibration JSON file')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing left/right image pairs')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of image pairs to test')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--output-dir', type=str, default='epipolar_verification',
                        help='Output directory for results')
    parser.add_argument('--save-report', action='store_true',
                        help='Generate and save verification report')
    
    args = parser.parse_args()
    
    # Initialize verifier
    try:
        verifier = EpipolarVerifier(args.calibration)
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")
        return 1
    
    # Setup output directory
    output_path = Path(args.output_dir)
    if args.save_visualizations or args.save_report:
        output_path.mkdir(exist_ok=True)
    
    # Load images
    images_path = Path(args.images)
    left_dir = images_path / 'left'
    right_dir = images_path / 'right'
    
    if not left_dir.exists() or not right_dir.exists():
        logger.error(f"Image directories not found: {left_dir} or {right_dir}")
        return 1
    
    # Find image pairs
    left_images = sorted(list(left_dir.glob('*.png')) + list(left_dir.glob('*.jpg')))
    right_images = sorted(list(right_dir.glob('*.png')) + list(right_dir.glob('*.jpg')))
    
    num_pairs = min(len(left_images), len(right_images), args.num_samples)
    logger.info(f"Found {len(left_images)} left and {len(right_images)} right images")
    logger.info(f"Testing {num_pairs} image pairs")
    
    # Process image pairs
    for i in range(num_pairs):
        left_img = cv2.imread(str(left_images[i]))
        right_img = cv2.imread(str(right_images[i]))
        
        if left_img is None or right_img is None:
            logger.warning(f"Failed to load image pair {i}")
            continue
        
        # Verify image pair
        results, pts1, pts2, left_rect, right_rect = verifier.verify_image_pair(
            left_img, right_img, i
        )
        
        # Store results
        verifier.results['epipolar_errors'].append(results)
        verifier.results['rectification_errors'].append(results)
        
        # Save visualizations if requested
        if args.save_visualizations and len(pts1) > 0:
            # Epipolar visualization
            verifier.create_epipolar_visualization(
                left_img, right_img, pts1, pts2, verifier.F,
                output_path / f'epipolar_verification_{i:02d}.png'
            )
            
            # Rectification visualization
            verifier.create_rectification_visualization(
                left_rect, right_rect,
                output_path / f'rectification_verification_{i:02d}.png'
            )
    
    # Generate report if requested
    if args.save_report:
        report = verifier.generate_report(output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())