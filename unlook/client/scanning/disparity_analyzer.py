"""
Disparity Analyzer - Analyzes and validates disparity maps for coherence with input images.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DisparityAnalyzer:
    """Analyzes disparity maps to ensure they match the input images."""
    
    def __init__(self):
        self.debug_output_dir = None
        
    def set_debug_output(self, output_dir: str):
        """Set directory for debug output images."""
        self.debug_output_dir = Path(output_dir)
        self.debug_output_dir.mkdir(exist_ok=True)
        
    def analyze_disparity_coherence(self, left_img: np.ndarray, right_img: np.ndarray, 
                                   disparity_map: np.ndarray) -> Dict[str, Any]:
        """
        Analyze if the disparity map is coherent with the input images.
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'coherent': False,
            'issues': [],
            'metrics': {}
        }
        
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
            
        h, w = left_gray.shape
        
        # Test 1: Check disparity range
        valid_disp = disparity_map[disparity_map > 0]
        if len(valid_disp) > 0:
            disp_min = valid_disp.min() / 16.0
            disp_max = valid_disp.max() / 16.0
            disp_mean = valid_disp.mean() / 16.0
            
            results['metrics']['disparity_range'] = {
                'min': float(disp_min),
                'max': float(disp_max),
                'mean': float(disp_mean)
            }
            
            # Check if disparity is within reasonable range
            if disp_max > w / 4:  # Disparity shouldn't be more than 1/4 of image width
                results['issues'].append(f"Disparity too large: {disp_max:.1f} pixels (max should be < {w/4:.1f})")
        else:
            results['issues'].append("No valid disparity values found")
            return results
            
        # Test 2: Verify correspondence by warping
        logger.info("Testing disparity coherence by warping...")
        
        # Create a sample of test points
        test_points = []
        step = 50  # Test every 50 pixels
        for y in range(step, h-step, step):
            for x in range(step, w-step, step):
                if disparity_map[y, x] > 0:
                    test_points.append((x, y))
                    
        if len(test_points) == 0:
            results['issues'].append("No valid test points for correspondence check")
            return results
            
        # Test correspondences
        good_matches = 0
        total_tests = min(len(test_points), 100)  # Test up to 100 points
        
        for i, (x, y) in enumerate(test_points[:total_tests]):
            disp = disparity_map[y, x] / 16.0
            
            # The corresponding point in right image should be at (x - disp, y)
            x_right = int(x - disp)
            
            if 0 <= x_right < w:
                # Compare local patches
                patch_size = 5
                if (x >= patch_size and x < w - patch_size and 
                    x_right >= patch_size and x_right < w - patch_size):
                    
                    left_patch = left_gray[y-patch_size:y+patch_size+1, 
                                          x-patch_size:x+patch_size+1]
                    right_patch = right_gray[y-patch_size:y+patch_size+1, 
                                           x_right-patch_size:x_right+patch_size+1]
                    
                    # Calculate patch similarity
                    if left_patch.size > 0 and right_patch.size > 0:
                        correlation = cv2.matchTemplate(left_patch, right_patch, 
                                                      cv2.TM_CCORR_NORMED)[0, 0]
                        if correlation > 0.7:  # Good match threshold
                            good_matches += 1
                            
        match_ratio = good_matches / total_tests if total_tests > 0 else 0
        results['metrics']['correspondence_ratio'] = float(match_ratio)
        
        if match_ratio < 0.5:
            results['issues'].append(f"Poor correspondence: only {match_ratio:.1%} of points match")
            
        # Test 3: Check if disparity follows image features
        # Disparity should be smooth except at object boundaries
        logger.info("Checking disparity smoothness...")
        
        # Compute disparity gradient
        disp_normalized = disparity_map.astype(np.float32) / 16.0
        grad_x = cv2.Sobel(disp_normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disp_normalized, cv2.CV_32F, 0, 1, ksize=3)
        disp_gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute image gradient
        img_grad_x = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
        img_grad_y = cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3)
        img_gradient = np.sqrt(img_grad_x**2 + img_grad_y**2)
        
        # Normalize gradients
        disp_gradient_norm = cv2.normalize(disp_gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img_gradient_norm = cv2.normalize(img_gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Save debug images if output directory is set
        if self.debug_output_dir:
            cv2.imwrite(str(self.debug_output_dir / "disparity_gradient.png"), disp_gradient_norm)
            cv2.imwrite(str(self.debug_output_dir / "image_gradient.png"), img_gradient_norm)
            
            # Create visualization of correspondence check
            vis_img = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
            for x, y in test_points[:total_tests]:
                disp = disparity_map[y, x] / 16.0
                x_right = int(x - disp)
                if 0 <= x_right < w:
                    # Draw line from left to right correspondence
                    color = (0, 255, 0) if match_ratio > 0.5 else (0, 0, 255)
                    cv2.circle(vis_img, (x, y), 3, color, -1)
                    cv2.line(vis_img, (x, y), (x_right, y), color, 1)
            cv2.imwrite(str(self.debug_output_dir / "correspondence_check.png"), vis_img)
        
        # Overall coherence check
        results['coherent'] = len(results['issues']) == 0 and match_ratio > 0.6
        
        if results['coherent']:
            logger.info("Disparity map is coherent with input images")
        else:
            logger.warning(f"Disparity coherence issues: {results['issues']}")
            
        return results
    
    def suggest_fixes(self, analysis_results: Dict[str, Any]) -> list:
        """Suggest fixes based on analysis results."""
        suggestions = []
        
        if not analysis_results['coherent']:
            for issue in analysis_results['issues']:
                if "too large" in issue:
                    suggestions.append("Reduce numDisparities parameter in stereo matcher")
                    suggestions.append("Check if cameras are properly aligned")
                elif "No valid disparity" in issue:
                    suggestions.append("Check image quality and texture")
                    suggestions.append("Verify stereo rectification")
                elif "Poor correspondence" in issue:
                    suggestions.append("Recalibrate stereo cameras")
                    suggestions.append("Check for synchronization issues between cameras")
                    suggestions.append("Verify that images are not swapped (left/right)")
                    
        return suggestions