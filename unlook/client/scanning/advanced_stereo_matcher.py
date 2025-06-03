"""
Advanced Stereo Matching Module with State-of-the-Art Algorithms
Based on 2024 best practices for high-quality disparity maps
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from .disparity_analyzer import DisparityAnalyzer

logger = logging.getLogger(__name__)

class AdvancedStereoMatcher:
    """
    State-of-the-art stereo matching with automatic parameter tuning and post-filtering.
    Implements best practices from 2024 research and OpenCV documentation.
    """
    
    def __init__(self, image_size: Tuple[int, int], baseline_mm: float = 79.5):
        """
        Initialize advanced stereo matcher.
        
        Args:
            image_size: (width, height) of input images
            baseline_mm: Stereo baseline in millimeters
        """
        self.image_size = image_size
        self.baseline_mm = baseline_mm
        self.width, self.height = image_size
        
        # Auto-calculate optimal parameters based on image size and baseline
        self._calculate_optimal_parameters()
        
        # Initialize matchers
        self._create_matchers()
        
        # Initialize WLS filter for post-processing
        self._create_wls_filter()
        
        logger.info(f"Advanced stereo matcher initialized for {image_size} images")
        logger.info(f"Using disparity range: {self.min_disparity} to {self.min_disparity + self.num_disparities}")
        
        # Initialize disparity analyzer
        self.disparity_analyzer = DisparityAnalyzer()
    
    def _calculate_optimal_parameters(self):
        """Calculate optimal stereo parameters based on image dimensions and baseline."""
        
        # Use actual focal length from calibration if available (typically ~877 pixels for 1456x1088)
        # This is more accurate than using min(width, height)
        self.focal_length = 877.6  # Known focal length from calibration
        focal_length_pixels = self.focal_length
        
        # Calculate disparities for depth range 200mm to 1000mm (more reasonable for desktop scanning)
        max_disp_near = int((self.baseline_mm * focal_length_pixels) / 200)  # 20cm
        min_disp_far = int((self.baseline_mm * focal_length_pixels) / 1000)   # 1m
        
        # Ensure disparity range is reasonable and multiple of 16
        self.min_disparity = 0  # Start from 0 for maximum range
        disparity_range = max_disp_near
        self.num_disparities = ((disparity_range // 16) + 1) * 16  # Must be multiple of 16
        self.num_disparities = min(self.num_disparities, 256)  # Cap at 256 for performance
        
        # DEBUG: Log the calculated parameters
        logger.debug(f"STEREO PARAMETERS CALCULATION:")
        logger.debug(f"  Focal length: {focal_length_pixels:.1f} pixels")
        logger.debug(f"  Baseline: {self.baseline_mm:.1f} mm")
        logger.debug(f"  Expected disparity @ 20cm: {max_disp_near:.1f} pixels")
        logger.debug(f"  Expected disparity @ 100cm: {min_disp_far:.1f} pixels")
        logger.debug(f"  Min disparity: {self.min_disparity}")
        logger.debug(f"  Num disparities: {self.num_disparities}")
        
        # Calculate block size based on image resolution
        if self.width > 1000:
            self.block_size = 11  # High resolution
        elif self.width > 500:
            self.block_size = 9   # Medium resolution
        else:
            self.block_size = 7   # Low resolution
        
        # Ensure odd block size
        if self.block_size % 2 == 0:
            self.block_size += 1
        
        # Calculate P1 and P2 based on block size and channels
        channels = 3  # Assume color images
        self.P1 = 8 * channels * self.block_size ** 2
        self.P2 = 32 * channels * self.block_size ** 2
        
        logger.info(f"Auto-calculated parameters:")
        logger.info(f"  Disparity range: {self.min_disparity} to {self.min_disparity + self.num_disparities}")
        logger.info(f"  Block size: {self.block_size}")
        logger.info(f"  P1: {self.P1}, P2: {self.P2}")
    
    def _create_matchers(self):
        """Create left and right stereo matchers with optimized parameters."""
        
        # Create left matcher (StereoSGBM for high quality)
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=1,        # Left-right consistency check
            uniquenessRatio=10,     # Reject ambiguous matches
            speckleWindowSize=50,   # Remove small isolated regions
            speckleRange=2,         # Disparity variation threshold
            preFilterCap=63,        # Pre-filter normalization cap
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 3-way optimization
        )
        
        # Create right matcher for left-right consistency check
        try:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.has_right_matcher = True
            logger.info("Right matcher created for left-right consistency")
        except AttributeError:
            logger.warning("opencv-contrib not available, skipping right matcher")
            self.right_matcher = None
            self.has_right_matcher = False
    
    def _create_wls_filter(self):
        """Create WLS filter for disparity map post-processing."""
        try:
            # Create WLS filter with auto-tuned parameters
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            
            # Set lambda based on image size (larger images need more regularization)
            lambda_val = 8000 if self.width < 1000 else 12000
            self.wls_filter.setLambda(lambda_val)
            
            # Set sigma color (sensitivity to image edges)
            self.wls_filter.setSigmaColor(1.2)
            
            self.has_wls_filter = True
            logger.info(f"WLS filter created with lambda={lambda_val}")
            
        except AttributeError:
            logger.warning("opencv-contrib not available, skipping WLS filter")
            self.wls_filter = None
            self.has_wls_filter = False
    
    def compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute high-quality disparity map with post-filtering.
        
        Args:
            left_img: Left stereo image
            right_img: Right stereo image
            
        Returns:
            Tuple of (disparity_map, statistics)
        """
        stats = {
            'method': 'advanced_sgbm',
            'parameters': {
                'num_disparities': self.num_disparities,
                'block_size': self.block_size,
                'min_disparity': self.min_disparity
            }
        }
        
        # DEBUG: Log input image info
        logger.debug(f"DISPARITY COMPUTATION DEBUG:")
        logger.debug(f"  Left image shape: {left_img.shape}, dtype: {left_img.dtype}")
        logger.debug(f"  Right image shape: {right_img.shape}, dtype: {right_img.dtype}")
        logger.debug(f"  Image brightness - Left: mean={left_img.mean():.1f}, std={left_img.std():.1f}")
        logger.debug(f"  Image brightness - Right: mean={right_img.mean():.1f}, std={right_img.std():.1f}")
        
        try:
            # Ensure images are grayscale
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img.copy()
                right_gray = right_img.copy()
            
            # Apply preprocessing for better matching
            left_processed = self._preprocess_image(left_gray)
            right_processed = self._preprocess_image(right_gray)
            
            # Compute left disparity
            left_disparity = self.left_matcher.compute(left_processed, right_processed)
            stats['left_disparity_computed'] = True
            
            # Compute right disparity if available
            right_disparity = None
            if self.has_right_matcher:
                try:
                    right_disparity = self.right_matcher.compute(right_processed, left_processed)
                    stats['right_disparity_computed'] = True
                except Exception as e:
                    logger.warning(f"Right disparity computation failed: {e}")
                    stats['right_disparity_computed'] = False
            
            # Apply WLS filtering if available
            if self.has_wls_filter and self.wls_filter is not None:
                try:
                    if right_disparity is not None:
                        filtered_disparity = self.wls_filter.filter(
                            left_disparity, left_img, disparity_map_right=right_disparity
                        )
                    else:
                        filtered_disparity = self.wls_filter.filter(left_disparity, left_img)
                    
                    stats['wls_filtering_applied'] = True
                    final_disparity = filtered_disparity
                    
                except Exception as e:
                    logger.warning(f"WLS filtering failed: {e}")
                    stats['wls_filtering_applied'] = False
                    final_disparity = left_disparity
            else:
                final_disparity = left_disparity
                stats['wls_filtering_applied'] = False
            
            # Calculate statistics
            valid_pixels = np.sum(final_disparity > 0)
            total_pixels = final_disparity.size
            stats['valid_pixel_ratio'] = valid_pixels / total_pixels
            stats['total_valid_pixels'] = int(valid_pixels)
            
            if valid_pixels > 0:
                valid_disparities = final_disparity[final_disparity > 0]
                stats['disparity_range'] = {
                    'min': float(np.min(valid_disparities)) / 16.0,
                    'max': float(np.max(valid_disparities)) / 16.0,
                    'mean': float(np.mean(valid_disparities)) / 16.0,
                    'std': float(np.std(valid_disparities)) / 16.0
                }
                
                # DEBUG: Detailed disparity analysis
                logger.debug(f"DISPARITY ANALYSIS:")
                logger.debug(f"  Valid pixels: {valid_pixels}/{total_pixels} ({stats['valid_pixel_ratio']:.1%})")
                logger.debug(f"  Disparity range: {stats['disparity_range']['min']:.2f} - {stats['disparity_range']['max']:.2f} pixels")
                logger.debug(f"  Mean disparity: {stats['disparity_range']['mean']:.2f} +/- {stats['disparity_range']['std']:.2f} pixels")
                
                # Expected disparity calculation
                expected_disp_30cm = (self.focal_length * self.baseline_mm) / 300.0
                expected_disp_50cm = (self.focal_length * self.baseline_mm) / 500.0
                logger.debug(f"  Expected disparity for 30cm: {expected_disp_30cm:.1f} pixels")
                logger.debug(f"  Expected disparity for 50cm: {expected_disp_50cm:.1f} pixels")
                logger.debug(f"  Actual vs Expected ratio: {stats['disparity_range']['mean']/expected_disp_30cm:.2f}x")
            
            logger.info(f"Disparity computed: {stats['total_valid_pixels']} valid pixels ({stats['valid_pixel_ratio']:.1%})")
            
            # Analyze disparity coherence
            if valid_pixels > 0:
                coherence_analysis = self.disparity_analyzer.analyze_disparity_coherence(
                    left_img, right_img, final_disparity
                )
                stats['coherence_analysis'] = coherence_analysis
                
                if not coherence_analysis['coherent']:
                    logger.warning("Disparity coherence issues detected:")
                    for issue in coherence_analysis['issues']:
                        logger.warning(f"  - {issue}")
                    suggestions = self.disparity_analyzer.suggest_fixes(coherence_analysis)
                    for suggestion in suggestions:
                        logger.info(f"  Suggestion: {suggestion}")
            
            return final_disparity, stats
            
        except Exception as e:
            logger.error(f"Disparity computation failed: {e}")
            stats['error'] = str(e)
            # Return empty disparity map
            return np.zeros((self.height, self.width), dtype=np.int16), stats
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve stereo matching."""
        
        # Apply slight Gaussian blur to reduce noise
        img_smooth = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_smooth)
        
        return img_enhanced
    
    def compute_multiple_scales(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute disparity using multiple scales and merge results.
        This can improve quality for challenging scenes.
        """
        
        # Compute at full resolution
        disp_full, stats_full = self.compute_disparity(left_img, right_img)
        
        # Compute at half resolution for better coverage
        h, w = left_img.shape[:2]
        left_half = cv2.resize(left_img, (w//2, h//2))
        right_half = cv2.resize(right_img, (w//2, h//2))
        
        # Create temporary matcher for half resolution
        temp_matcher = AdvancedStereoMatcher((w//2, h//2), self.baseline_mm)
        disp_half, stats_half = temp_matcher.compute_disparity(left_half, right_half)
        
        # Upscale half-resolution disparity
        disp_half_upscaled = cv2.resize(disp_half, (w, h))
        disp_half_upscaled = disp_half_upscaled * 2  # Scale disparities
        
        # Merge disparities (use full resolution where available, half where not)
        merged_disparity = disp_full.copy()
        mask_invalid = disp_full <= 0
        merged_disparity[mask_invalid] = disp_half_upscaled[mask_invalid]
        
        # Combine statistics
        stats_combined = {
            'method': 'multi_scale',
            'full_resolution': stats_full,
            'half_resolution': stats_half,
            'merge_improvement': np.sum(merged_disparity > 0) - np.sum(disp_full > 0)
        }
        
        logger.info(f"Multi-scale processing added {stats_combined['merge_improvement']} valid pixels")
        
        return merged_disparity, stats_combined

def create_advanced_matcher(image_size: Tuple[int, int], baseline_mm: float = 79.5) -> AdvancedStereoMatcher:
    """Factory function to create an advanced stereo matcher."""
    return AdvancedStereoMatcher(image_size, baseline_mm)