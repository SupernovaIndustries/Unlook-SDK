"""
Adaptive Pattern Strategies for Structured Light Scanning.

This module implements intelligent pattern selection and density adaptation
based on scene complexity and scanning requirements.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Supported pattern types."""
    GRAY_CODE = "gray_code"
    PHASE_SHIFT = "phase_shift"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class SceneComplexity(Enum):
    """Scene complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PatternConfig:
    """Configuration for pattern generation."""
    pattern_type: PatternType
    num_patterns: int
    density_factor: float = 1.0
    hybrid_ratio: float = 0.5  # Ratio of Gray code vs phase shift in hybrid mode
    adaptive_threshold: float = 0.1
    min_patterns: int = 4
    max_patterns: int = 32


@dataclass
class SceneAnalysis:
    """Results from scene complexity analysis."""
    complexity: SceneComplexity
    edge_density: float
    texture_variance: float
    noise_level: float
    recommended_patterns: int
    pattern_type: PatternType


class AdaptivePatternGenerator:
    """
    Adaptive pattern generator that selects optimal patterns based on scene analysis.
    """
    
    def __init__(self):
        """Initialize adaptive pattern generator."""
        self.pattern_cache = {}
        self.scene_analysis_cache = {}
        
        # Default configurations for different complexity levels
        self.complexity_configs = {
            SceneComplexity.SIMPLE: PatternConfig(
                pattern_type=PatternType.GRAY_CODE,
                num_patterns=6,
                density_factor=0.5,
                min_patterns=4,
                max_patterns=8
            ),
            SceneComplexity.MODERATE: PatternConfig(
                pattern_type=PatternType.HYBRID,
                num_patterns=12,
                density_factor=0.75,
                hybrid_ratio=0.7,
                min_patterns=8,
                max_patterns=16
            ),
            SceneComplexity.COMPLEX: PatternConfig(
                pattern_type=PatternType.PHASE_SHIFT,
                num_patterns=24,
                density_factor=1.0,
                min_patterns=16,
                max_patterns=32
            )
        }
        
        logger.info("Adaptive pattern generator initialized")
    
    def analyze_scene_complexity(self, reference_images: List[np.ndarray]) -> SceneAnalysis:
        """
        Analyze scene complexity from reference images.
        
        Args:
            reference_images: List of reference images (white, black, etc.)
            
        Returns:
            SceneAnalysis with complexity assessment
        """
        if not reference_images:
            return SceneAnalysis(
                complexity=SceneComplexity.MODERATE,
                edge_density=0.1,
                texture_variance=0.5,
                noise_level=0.1,
                recommended_patterns=12,
                pattern_type=PatternType.HYBRID
            )
        
        try:
            # Use the first reference image for analysis
            ref_image = reference_images[0]
            
            # Convert to grayscale if needed
            if len(ref_image.shape) == 3:
                gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = ref_image
            
            # Calculate edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges > 0)
            
            # Calculate texture variance using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # Estimate noise level
            noise_level = self._estimate_noise_level(gray)
            
            # Determine complexity based on metrics
            complexity = self._classify_complexity(edge_density, texture_variance, noise_level)
            
            # Get recommended configuration
            config = self.complexity_configs[complexity]
            
            # Adjust pattern count based on specific metrics
            base_patterns = config.num_patterns
            if edge_density > 0.15:  # High edge density
                recommended_patterns = min(config.max_patterns, int(base_patterns * 1.5))
            elif edge_density < 0.05:  # Low edge density
                recommended_patterns = max(config.min_patterns, int(base_patterns * 0.7))
            else:
                recommended_patterns = base_patterns
            
            # Select optimal pattern type
            if complexity == SceneComplexity.SIMPLE and noise_level < 0.05:
                pattern_type = PatternType.GRAY_CODE
            elif complexity == SceneComplexity.COMPLEX or noise_level > 0.1:
                pattern_type = PatternType.PHASE_SHIFT
            else:
                pattern_type = PatternType.HYBRID
            
            analysis = SceneAnalysis(
                complexity=complexity,
                edge_density=edge_density,
                texture_variance=texture_variance,
                noise_level=noise_level,
                recommended_patterns=recommended_patterns,
                pattern_type=pattern_type
            )
            
            logger.info(f"Scene analysis: {complexity.value}, patterns: {recommended_patterns}, type: {pattern_type.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            # Return default moderate complexity
            return SceneAnalysis(
                complexity=SceneComplexity.MODERATE,
                edge_density=0.1,
                texture_variance=0.5,
                noise_level=0.1,
                recommended_patterns=12,
                pattern_type=PatternType.HYBRID
            )
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in the image."""
        try:
            # Use standard deviation of Laplacian to estimate noise
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            
            # Apply Gaussian blur to get noise estimate
            blurred = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 0)
            noise = np.std(image.astype(np.float32) - blurred)
            
            # Normalize noise level
            return min(1.0, noise / 50.0)  # Assuming max noise std of 50
            
        except Exception:
            return 0.1  # Default moderate noise level
    
    def _classify_complexity(self, edge_density: float, texture_variance: float, noise_level: float) -> SceneComplexity:
        """Classify scene complexity based on metrics."""
        # Weighted scoring
        complexity_score = (
            edge_density * 0.4 +
            min(1.0, texture_variance / 1000.0) * 0.4 +
            noise_level * 0.2
        )
        
        if complexity_score < 0.3:
            return SceneComplexity.SIMPLE
        elif complexity_score < 0.7:
            return SceneComplexity.MODERATE
        else:
            return SceneComplexity.COMPLEX
    
    def generate_adaptive_pattern_sequence(self, 
                                         scene_analysis: SceneAnalysis,
                                         projector_resolution: Tuple[int, int] = (1280, 720)) -> List[Dict[str, Any]]:
        """
        Generate optimal pattern sequence based on scene analysis.
        
        Args:
            scene_analysis: Results from scene complexity analysis
            projector_resolution: Projector resolution (width, height)
            
        Returns:
            List of pattern configurations
        """
        try:
            pattern_sequence = []
            
            # Add reference patterns (always needed)
            pattern_sequence.extend([
                {'type': 'white', 'priority': 'high'},
                {'type': 'black', 'priority': 'high'}
            ])
            
            # Generate patterns based on recommended type
            if scene_analysis.pattern_type == PatternType.GRAY_CODE:
                pattern_sequence.extend(self._generate_gray_code_patterns(
                    scene_analysis.recommended_patterns - 2,  # Subtract reference patterns
                    projector_resolution,
                    scene_analysis.complexity
                ))
                
            elif scene_analysis.pattern_type == PatternType.PHASE_SHIFT:
                pattern_sequence.extend(self._generate_phase_shift_patterns(
                    scene_analysis.recommended_patterns - 2,
                    projector_resolution,
                    scene_analysis.complexity
                ))
                
            elif scene_analysis.pattern_type == PatternType.HYBRID:
                # Mix Gray code and phase shift
                config = self.complexity_configs[scene_analysis.complexity]
                total_patterns = scene_analysis.recommended_patterns - 2
                
                gray_patterns = int(total_patterns * config.hybrid_ratio)
                phase_patterns = total_patterns - gray_patterns
                
                # Add Gray code patterns first (more robust)
                pattern_sequence.extend(self._generate_gray_code_patterns(
                    gray_patterns, projector_resolution, scene_analysis.complexity
                ))
                
                # Add phase shift patterns for fine details
                pattern_sequence.extend(self._generate_phase_shift_patterns(
                    phase_patterns, projector_resolution, scene_analysis.complexity
                ))
            
            # Add priority and timing information
            for i, pattern in enumerate(pattern_sequence):
                if pattern['type'] in ['white', 'black']:
                    pattern['capture_time_ms'] = 100  # Longer exposure for reference
                else:
                    pattern['capture_time_ms'] = 50   # Standard capture time
                
                pattern['sequence_index'] = i
            
            logger.info(f"Generated adaptive pattern sequence: {len(pattern_sequence)} patterns")
            
            return pattern_sequence
            
        except Exception as e:
            logger.error(f"Pattern generation failed: {e}")
            # Return minimal pattern sequence
            return [
                {'type': 'white', 'priority': 'high', 'sequence_index': 0, 'capture_time_ms': 100},
                {'type': 'black', 'priority': 'high', 'sequence_index': 1, 'capture_time_ms': 100},
                {'type': 'gray_code', 'bit_index': 0, 'direction': 'horizontal', 'inverted': False, 'sequence_index': 2, 'capture_time_ms': 50},
                {'type': 'gray_code', 'bit_index': 0, 'direction': 'horizontal', 'inverted': True, 'sequence_index': 3, 'capture_time_ms': 50}
            ]
    
    def _generate_gray_code_patterns(self, num_patterns: int, resolution: Tuple[int, int], complexity: SceneComplexity) -> List[Dict[str, Any]]:
        """Generate Gray code pattern sequence."""
        patterns = []
        
        # Calculate number of bits needed
        width, height = resolution
        bits_horizontal = int(np.ceil(np.log2(width)))
        bits_vertical = int(np.ceil(np.log2(height)))
        
        # Adjust based on complexity
        if complexity == SceneComplexity.SIMPLE:
            # Use fewer bits for simple scenes
            bits_horizontal = min(bits_horizontal, 8)
            bits_vertical = min(bits_vertical, 6)
        elif complexity == SceneComplexity.COMPLEX:
            # Use full resolution for complex scenes
            pass
        
        # Generate horizontal patterns
        patterns_added = 0
        for bit in range(bits_horizontal):
            if patterns_added >= num_patterns:
                break
                
            # Normal pattern
            patterns.append({
                'type': 'gray_code',
                'bit_index': bit,
                'direction': 'horizontal',
                'inverted': False,
                'priority': 'high' if bit < 4 else 'medium'
            })
            patterns_added += 1
            
            if patterns_added >= num_patterns:
                break
                
            # Inverted pattern
            patterns.append({
                'type': 'gray_code',
                'bit_index': bit,
                'direction': 'horizontal',
                'inverted': True,
                'priority': 'high' if bit < 4 else 'medium'
            })
            patterns_added += 1
        
        # Generate vertical patterns if we have remaining pattern budget
        for bit in range(bits_vertical):
            if patterns_added >= num_patterns:
                break
                
            # Normal pattern
            patterns.append({
                'type': 'gray_code',
                'bit_index': bit,
                'direction': 'vertical',
                'inverted': False,
                'priority': 'medium'
            })
            patterns_added += 1
            
            if patterns_added >= num_patterns:
                break
                
            # Inverted pattern
            patterns.append({
                'type': 'gray_code',
                'bit_index': bit,
                'direction': 'vertical',
                'inverted': True,
                'priority': 'medium'
            })
            patterns_added += 1
        
        return patterns[:num_patterns]
    
    def _generate_phase_shift_patterns(self, num_patterns: int, resolution: Tuple[int, int], complexity: SceneComplexity) -> List[Dict[str, Any]]:
        """Generate phase shift pattern sequence."""
        patterns = []
        
        # Standard 4-step phase shift
        steps_per_cycle = 4
        
        # Calculate number of frequencies we can fit
        num_frequencies = max(1, num_patterns // steps_per_cycle)
        
        # Base frequencies
        base_frequencies = [1, 2, 4, 8, 16]  # Different pattern densities
        
        # Select frequencies based on complexity
        if complexity == SceneComplexity.SIMPLE:
            frequencies = base_frequencies[:min(num_frequencies, 2)]
        elif complexity == SceneComplexity.MODERATE:
            frequencies = base_frequencies[:min(num_frequencies, 3)]
        else:
            frequencies = base_frequencies[:min(num_frequencies, 5)]
        
        pattern_count = 0
        for freq_idx, frequency in enumerate(frequencies):
            if pattern_count >= num_patterns:
                break
                
            for step in range(steps_per_cycle):
                if pattern_count >= num_patterns:
                    break
                    
                phase = (2 * np.pi * step) / steps_per_cycle
                
                patterns.append({
                    'type': 'phase_shift',
                    'frequency': frequency,
                    'phase': phase,
                    'step': step,
                    'frequency_index': freq_idx,
                    'direction': 'horizontal',
                    'priority': 'high' if freq_idx == 0 else 'medium'
                })
                pattern_count += 1
        
        return patterns
    
    def optimize_pattern_sequence(self, pattern_sequence: List[Dict[str, Any]], 
                                time_budget_ms: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Optimize pattern sequence for time constraints and quality.
        
        Args:
            pattern_sequence: Original pattern sequence
            time_budget_ms: Maximum time budget in milliseconds
            
        Returns:
            Optimized pattern sequence
        """
        if not time_budget_ms:
            return pattern_sequence
        
        try:
            # Calculate current total time
            total_time = sum(p.get('capture_time_ms', 50) for p in pattern_sequence)
            
            if total_time <= time_budget_ms:
                return pattern_sequence  # Already within budget
            
            # Sort patterns by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            sorted_patterns = sorted(
                pattern_sequence,
                key=lambda p: (priority_order.get(p.get('priority', 'medium'), 1), p.get('sequence_index', 999))
            )
            
            # Select patterns within time budget
            optimized_sequence = []
            current_time = 0
            
            for pattern in sorted_patterns:
                pattern_time = pattern.get('capture_time_ms', 50)
                if current_time + pattern_time <= time_budget_ms:
                    optimized_sequence.append(pattern)
                    current_time += pattern_time
                else:
                    # Check if we can fit a shorter exposure version
                    if pattern.get('priority') == 'high' and current_time + 30 <= time_budget_ms:
                        pattern_copy = pattern.copy()
                        pattern_copy['capture_time_ms'] = 30
                        optimized_sequence.append(pattern_copy)
                        current_time += 30
            
            # Re-index sequence
            for i, pattern in enumerate(optimized_sequence):
                pattern['sequence_index'] = i
            
            logger.info(f"Optimized pattern sequence: {len(pattern_sequence)} -> {len(optimized_sequence)} patterns "
                       f"({total_time}ms -> {current_time}ms)")
            
            return optimized_sequence
            
        except Exception as e:
            logger.error(f"Pattern optimization failed: {e}")
            return pattern_sequence
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'cache_size': len(self.pattern_cache),
            'scene_analysis_cache_size': len(self.scene_analysis_cache),
            'supported_pattern_types': [pt.value for pt in PatternType],
            'supported_complexities': [sc.value for sc in SceneComplexity]
        }