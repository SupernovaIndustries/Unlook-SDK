#!/usr/bin/env python3
"""
Camera Auto-Optimizer Module

This module implements automatic camera settings optimization for
structured light scanning. It analyzes test images to find the optimal
exposure, gain, and contrast settings for maximum pattern visibility
while preserving object details.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    """Data class for camera settings."""
    exposure_time: int  # microseconds
    analog_gain: float
    digital_gain: float
    contrast: float
    brightness: float
    awb_mode: str
    sharpness: float
    saturation: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for picamera2."""
        return {
            'ExposureTime': self.exposure_time,
            'AnalogueGain': self.analog_gain,
            # Digital gain might be controlled differently
            'Contrast': self.contrast,
            'Brightness': self.brightness,
            'Sharpness': self.sharpness,
            'Saturation': self.saturation
        }


@dataclass 
class OptimizationResult:
    """Results from the optimization process."""
    settings: CameraSettings
    quality_score: float
    pattern_visibility: float
    object_clarity: float
    dynamic_range: float
    noise_level: float
    metadata: Dict


class CameraAutoOptimizer:
    """
    Automatic camera settings optimizer for structured light scanning.
    
    This class analyzes test images to find optimal camera settings
    that maximize pattern visibility while maintaining object details.
    """
    
    def __init__(self, camera_client):
        """
        Initialize the optimizer.
        
        Args:
            camera_client: Camera client instance
        """
        self.camera = camera_client
        self.current_settings = None
        self.optimal_settings = None
        self.test_results = []
        
    def optimize_settings(self, projector_client=None) -> OptimizationResult:
        """
        Run the full optimization process.
        
        Args:
            projector_client: Optional projector for test patterns
            
        Returns:
            OptimizationResult with optimal settings
        """
        logger.info("Starting camera optimization process")
        
        # Step 1: Capture reference images
        references = self._capture_references(projector_client)
        
        # Step 2: Analyze ambient conditions
        ambient_analysis = self._analyze_ambient(references['ambient'])
        
        # Step 3: Test exposure range
        exposure_results = self._test_exposure_range(
            references, 
            projector_client,
            ambient_analysis
        )
        
        # Step 4: Test gain range  
        gain_results = self._test_gain_range(
            references,
            projector_client,
            exposure_results['optimal_exposure']
        )
        
        # Step 5: Fine-tune contrast and brightness
        final_settings = self._fine_tune_settings(
            exposure_results['optimal_exposure'],
            gain_results['optimal_gain'],
            references,
            projector_client
        )
        
        # Step 6: Validate results
        validation = self._validate_settings(final_settings, projector_client)
        
        return OptimizationResult(
            settings=final_settings,
            quality_score=validation['overall_score'],
            pattern_visibility=validation['pattern_visibility'],
            object_clarity=validation['object_clarity'],
            dynamic_range=validation['dynamic_range'],
            noise_level=validation['noise_level'],
            metadata={
                'ambient_analysis': ambient_analysis,
                'exposure_results': exposure_results,
                'gain_results': gain_results,
                'validation': validation
            }
        )
    
    def _capture_references(self, projector_client) -> Dict[str, np.ndarray]:
        """Capture reference images for analysis."""
        logger.info("Capturing reference images for camera optimization")
        references = {}
        
        # Step 1: Capture ambient (projector off) - baseline lighting
        if projector_client:
            projector_client.show_solid_field('Black')
        time.sleep(0.5)
        references['ambient'] = self._capture_image()
        logger.info("Captured ambient reference image")
        
        # Step 2: Capture with test pattern (lines) for pattern visibility
        if projector_client:
            projector_client.show_horizontal_lines(
                foreground_color='White',
                background_color='Black',
                foreground_width=20,
                background_width=20
            )
            time.sleep(0.5)
            references['pattern'] = self._capture_image()
            logger.info("Captured pattern reference image")
            
            # Step 3: Capture with white projection for dynamic range
            projector_client.show_solid_field('White')
            time.sleep(0.5)
            references['white'] = self._capture_image()
            logger.info("Captured white reference image")
            
            # Step 4: Capture with black projection for noise floor
            projector_client.show_solid_field('Black')
            time.sleep(0.5)
            references['black'] = self._capture_image()
            logger.info("Captured black reference image")
            
            # Step 5: Capture with checkerboard for contrast analysis
            projector_client.show_checkerboard(
                horizontal_count=10,
                vertical_count=10
            )
            time.sleep(0.5)
            references['checkerboard'] = self._capture_image()
            logger.info("Captured checkerboard reference image")
            
            projector_client.turn_off()
        
        return references
    
    def _analyze_ambient(self, ambient_image: np.ndarray) -> Dict:
        """Analyze ambient lighting conditions."""
        logger.info("Analyzing ambient conditions")
        
        # Convert to grayscale if needed
        if len(ambient_image.shape) == 3:
            gray = cv2.cvtColor(ambient_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = ambient_image
            
        analysis = {
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'min_intensity': np.min(gray),
            'max_intensity': np.max(gray),
            'dynamic_range': np.max(gray) - np.min(gray),
            'histogram': cv2.calcHist([gray], [0], None, [256], [0, 256])
        }
        
        # Check for color cast
        if len(ambient_image.shape) == 3:
            b, g, r = cv2.split(ambient_image)
            analysis['color_cast'] = {
                'red_bias': np.mean(r) - np.mean(gray),
                'green_bias': np.mean(g) - np.mean(gray),
                'blue_bias': np.mean(b) - np.mean(gray)
            }
            
        # Classify lighting conditions
        if analysis['mean_intensity'] < 30:
            analysis['condition'] = 'dark'
            analysis['recommended_exposure'] = (20000, 100000)  # High exposure for dark conditions
            analysis['recommended_gain'] = (2.0, 8.0)  # Higher gain for dark conditions
        elif analysis['mean_intensity'] < 100:
            analysis['condition'] = 'low_light'
            analysis['recommended_exposure'] = (10000, 50000)  # Medium exposure
            analysis['recommended_gain'] = (1.5, 4.0)  # Medium gain
        elif analysis['mean_intensity'] < 180:
            analysis['condition'] = 'normal'
            analysis['recommended_exposure'] = (5000, 20000)  # Standard exposure
            analysis['recommended_gain'] = (1.0, 2.0)  # Standard gain
        else:
            analysis['condition'] = 'bright'
            analysis['recommended_exposure'] = (1000, 10000)  # Low exposure for bright conditions
            analysis['recommended_gain'] = (0.5, 1.5)  # Low gain for bright conditions
            
        return analysis
    
    def _test_exposure_range(self, references: Dict, projector_client, 
                           ambient_analysis: Dict) -> Dict:
        """Test different exposure values to find optimal."""
        logger.info("Testing exposure range")
        
        # Define exposure range based on ambient conditions
        if ambient_analysis['condition'] == 'dark':
            exposure_range = np.logspace(3, 5, 10)  # 1ms to 100ms
        elif ambient_analysis['condition'] == 'bright':
            exposure_range = np.logspace(2, 4, 10)  # 0.1ms to 10ms
        else:
            exposure_range = np.logspace(2.5, 4.5, 10)  # 0.3ms to 30ms
            
        results = []
        
        for exposure in exposure_range:
            # Set exposure
            self._set_exposure(int(exposure))
            time.sleep(0.1)
            
            # Capture test image
            if projector_client:
                projector_client.show_horizontal_lines()
                time.sleep(0.1)
                
            test_image = self._capture_image()
            
            # Analyze pattern visibility
            visibility = self._analyze_pattern_visibility(
                test_image, 
                references['ambient']
            )
            
            # Check for saturation
            saturation = self._check_saturation(test_image)
            
            results.append({
                'exposure': exposure,
                'visibility': visibility,
                'saturation': saturation,
                'score': visibility * (1 - saturation)
            })
            
        # Find optimal exposure
        best_result = max(results, key=lambda x: x['score'])
        
        return {
            'optimal_exposure': best_result['exposure'],
            'results': results
        }
    
    def _test_gain_range(self, references: Dict, projector_client,
                        optimal_exposure: float) -> Dict:
        """Test different gain values to find optimal."""
        logger.info("Testing gain range")
        
        # Set optimal exposure
        self._set_exposure(int(optimal_exposure))
        
        # Test gain range
        gain_range = np.linspace(1.0, 8.0, 8)
        results = []
        
        for gain in gain_range:
            # Set gain
            self._set_gain(gain)
            time.sleep(0.1)
            
            # Capture test image
            if projector_client:
                projector_client.show_horizontal_lines()
                time.sleep(0.1)
                
            test_image = self._capture_image()
            
            # Analyze image quality
            quality = self._analyze_image_quality(test_image)
            noise = self._estimate_noise(test_image)
            
            results.append({
                'gain': gain,
                'quality': quality,
                'noise': noise,
                'score': quality / (1 + noise)
            })
            
        # Find optimal gain
        best_result = max(results, key=lambda x: x['score'])
        
        return {
            'optimal_gain': best_result['gain'],
            'results': results
        }
    
    def _fine_tune_settings(self, exposure: float, gain: float,
                           references: Dict, projector_client) -> CameraSettings:
        """Fine-tune contrast and brightness."""
        logger.info("Fine-tuning camera settings")
        
        # Start with base settings
        settings = CameraSettings(
            exposure_time=int(exposure),
            analog_gain=gain,
            digital_gain=1.0,
            contrast=1.0,
            brightness=0.0,
            awb_mode='off',
            sharpness=1.0,
            saturation=1.0
        )
        
        # Test contrast adjustments
        contrast_range = np.linspace(0.8, 1.5, 7)
        best_contrast = 1.0
        best_score = 0
        
        for contrast in contrast_range:
            settings.contrast = contrast
            self._apply_settings(settings)
            time.sleep(0.1)
            
            # Test with pattern
            if projector_client:
                projector_client.show_horizontal_lines()
                time.sleep(0.1)
                
            test_image = self._capture_image()
            score = self._evaluate_overall_quality(test_image, references)
            
            if score > best_score:
                best_score = score
                best_contrast = contrast
                
        settings.contrast = best_contrast
        
        return settings
    
    def _validate_settings(self, settings: CameraSettings, 
                          projector_client) -> Dict:
        """Validate the optimized settings."""
        logger.info("Validating optimized settings")
        
        # Apply settings
        self._apply_settings(settings)
        time.sleep(0.5)
        
        validation = {
            'pattern_visibility': 0,
            'object_clarity': 0,
            'dynamic_range': 0,
            'noise_level': 0,
            'overall_score': 0
        }
        
        # Test with various patterns
        test_patterns = [
            ('horizontal_lines', {'foreground_width': 20, 'background_width': 20}),
            ('vertical_lines', {'foreground_width': 20, 'background_width': 20}),
            ('checkerboard', {'horizontal_count': 10, 'vertical_count': 10})
        ]
        
        for pattern_name, pattern_args in test_patterns:
            if projector_client:
                getattr(projector_client, f'show_{pattern_name}')(**pattern_args)
                time.sleep(0.1)
                
            test_image = self._capture_image()
            
            # Analyze different aspects
            validation['pattern_visibility'] += self._analyze_pattern_visibility(
                test_image, None
            )
            validation['object_clarity'] += self._analyze_clarity(test_image)
            validation['dynamic_range'] += self._analyze_dynamic_range(test_image)
            validation['noise_level'] += self._estimate_noise(test_image)
            
        # Average the scores
        num_tests = len(test_patterns)
        for key in validation:
            if key != 'overall_score':
                validation[key] /= num_tests
                
        # Calculate overall score
        validation['overall_score'] = (
            validation['pattern_visibility'] * 0.4 +
            validation['object_clarity'] * 0.3 +
            validation['dynamic_range'] * 0.2 +
            (1 - validation['noise_level']) * 0.1
        )
        
        return validation
    
    def _capture_image(self) -> np.ndarray:
        """Capture an image from the camera."""
        # Use the camera client to capture
        return self.camera.capture()
    
    def _set_exposure(self, exposure_time: int):
        """Set camera exposure time."""
        self.camera.set_exposure(exposure_time)
    
    def _set_gain(self, gain: float):
        """Set camera gain."""
        # This will depend on the camera client implementation
        pass
    
    def _apply_settings(self, settings: CameraSettings):
        """Apply all camera settings."""
        # This will depend on the camera client implementation
        pass
    
    def _analyze_pattern_visibility(self, image: np.ndarray, 
                                   ambient: Optional[np.ndarray]) -> float:
        """Analyze how visible the projected pattern is."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # If we have ambient, subtract it
        if ambient is not None:
            if len(ambient.shape) == 3:
                ambient_gray = cv2.cvtColor(ambient, cv2.COLOR_BGR2GRAY)
            else:
                ambient_gray = ambient
            pattern = gray.astype(float) - ambient_gray.astype(float)
        else:
            pattern = gray.astype(float)
            
        # Calculate pattern strength
        pattern_std = np.std(pattern)
        pattern_range = np.max(pattern) - np.min(pattern)
        
        # Normalize to 0-1
        visibility = min(pattern_std / 50.0, 1.0)  # Assuming 50 is good std
        
        return visibility
    
    def _check_saturation(self, image: np.ndarray) -> float:
        """Check for pixel saturation."""
        # Count pixels near max value
        saturated = np.sum(image > 250) / image.size
        return saturated
    
    def _analyze_image_quality(self, image: np.ndarray) -> float:
        """Analyze overall image quality."""
        # Check contrast
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        
        # Check sharpness using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Combine metrics
        quality = contrast * 0.5 + min(sharpness / 1000, 1.0) * 0.5
        
        return quality
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate image noise level."""
        # Use difference of Gaussians to estimate noise
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian filters
        smooth1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
        smooth2 = cv2.GaussianBlur(gray, (5, 5), 2.0)
        
        # Difference gives us high-frequency noise
        diff = smooth1.astype(float) - smooth2.astype(float)
        noise_std = np.std(diff)
        
        # Normalize to 0-1 (assuming 10 is high noise)
        noise_level = min(noise_std / 10.0, 1.0)
        
        return noise_level
    
    def _evaluate_overall_quality(self, image: np.ndarray, 
                                 references: Dict) -> float:
        """Evaluate overall image quality for structured light."""
        visibility = self._analyze_pattern_visibility(image, references.get('ambient'))
        quality = self._analyze_image_quality(image)
        noise = self._estimate_noise(image)
        saturation = self._check_saturation(image)
        
        # Weighted combination
        score = (
            visibility * 0.4 +
            quality * 0.3 +
            (1 - noise) * 0.2 +
            (1 - saturation) * 0.1
        )
        
        return score
    
    def _analyze_clarity(self, image: np.ndarray) -> float:
        """Analyze image clarity/sharpness."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Use Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Average gradient magnitude as clarity measure
        clarity = np.mean(gradient_mag) / 100.0  # Normalize
        
        return min(clarity, 1.0)
    
    def _analyze_dynamic_range(self, image: np.ndarray) -> float:
        """Analyze the dynamic range of the image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate percentiles to avoid outliers
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        
        dynamic_range = (p95 - p5) / 255.0
        
        return dynamic_range