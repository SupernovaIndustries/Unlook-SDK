"""
Pattern Manager for Structured Light 3D Scanning.

This module handles the generation and management of different pattern types
for structured light scanning, including Gray code, phase shift, and mixed patterns.
It provides a unified interface for pattern creation with metadata tracking.

This module acts as a wrapper around the existing pattern generation functions
in the patterns submodule, providing a consistent interface for the capture module.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from unlook.core.constants import *

# Import existing pattern generation functions
try:
    from .patterns import (
        generate_enhanced_gray_code_patterns,
        generate_phase_shift_patterns,
        generate_multi_scale_patterns,
        EnhancedPatternProcessor
    )
    PATTERNS_AVAILABLE = True
except ImportError as e:
    PATTERNS_AVAILABLE = False
    logger.debug(f"Enhanced patterns not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class PatternInfo:
    """
    Information about a scanning pattern.
    
    Attributes:
        pattern_type: Type of pattern (solid_field, vertical_lines, horizontal_lines, etc.)
        name: Descriptive name for the pattern
        metadata: Additional pattern-specific metadata
        parameters: Pattern generation parameters
    """
    pattern_type: str
    name: str
    metadata: Dict[str, Any]
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pattern_type': self.pattern_type,
            'name': self.name,
            'metadata': self.metadata,
            'parameters': self.parameters
        }


class PatternManager:
    """
    Manages pattern generation for structured light scanning.
    
    This class provides methods to create different types of patterns
    used in 3D scanning, including Gray code, phase shift, and mixed patterns.
    All patterns are generated with Protocol V2 optimization in mind.
    """
    
    def __init__(self):
        """Initialize the pattern manager."""
        self.default_num_bits = 8  # Default Gray code bits
        self.default_phase_steps = 4
        self.default_frequencies = [1, 8, 64]  # Multi-frequency phase shift
        
    def create_gray_code_patterns(self, 
                                 num_bits: int = None,
                                 use_blue: bool = True,
                                 include_inverse: bool = True) -> List[PatternInfo]:
        """
        Create Gray code pattern sequence.
        
        This method can use the enhanced pattern generation from the patterns module
        if available, otherwise falls back to basic pattern generation.
        
        Args:
            num_bits: Number of Gray code bits (default: 8)
            use_blue: Use blue stripes instead of white for better contrast
            include_inverse: Include inverse patterns for robust decoding
            
        Returns:
            List of PatternInfo objects describing the patterns
        """
        if num_bits is None:
            num_bits = self.default_num_bits
        
        # Try to use enhanced pattern generation if available
        if PATTERNS_AVAILABLE:
            try:
                # Use enhanced gray code generation
                enhanced_patterns = generate_enhanced_gray_code_patterns(
                    width=1024,  # Default projector width
                    height=768,   # Default projector height
                    num_bits=num_bits,
                    orientation="vertical"
                )
                
                # Convert enhanced patterns to PatternInfo format
                patterns = []
                for idx, ep in enumerate(enhanced_patterns):
                    # Extract pattern info from enhanced format
                    enhanced_pattern_type = ep.get("pattern_type", "vertical_lines")
                    # Map enhanced pattern types to projector pattern types
                    if enhanced_pattern_type == "gray_code":
                        projector_pattern_type = "vertical_lines"
                    else:
                        projector_pattern_type = enhanced_pattern_type
                    
                    pattern_info = PatternInfo(
                        pattern_type=projector_pattern_type,
                        name=ep.get("name", f"pattern_{idx}"),
                        metadata={
                            "pattern_set": "gray_code",
                            "enhanced": True,
                            "index": idx,
                            "bit": ep.get("bit", -1),
                            "inverted": ep.get("inverted", False),
                            "original_type": enhanced_pattern_type
                        },
                        parameters={
                            "color": ep.get("color", "White"),
                            "foreground_color": "Blue" if use_blue and projector_pattern_type != "solid_field" else ep.get("foreground_color", "White"),
                            "background_color": ep.get("background_color", "Black"),
                            "foreground_width": ep.get("stripe_width", 1),
                            "background_width": ep.get("stripe_width", 1)
                        }
                    )
                    patterns.append(pattern_info)
                
                logger.info(f"Created {len(patterns)} enhanced Gray code patterns with {num_bits} bits")
                return patterns
                
            except Exception as e:
                logger.warning(f"Failed to use enhanced patterns, falling back to basic: {e}")
        
        # Fallback to basic pattern generation
        patterns = []
        
        # Reference patterns (all white and all black)
        patterns.append(PatternInfo(
            pattern_type="solid_field",
            name="reference_white",
            metadata={
                "pattern_set": "gray_code",
                "reference": True,
                "index": 0
            },
            parameters={
                "color": "White"
            }
        ))
        
        patterns.append(PatternInfo(
            pattern_type="solid_field",
            name="reference_black",
            metadata={
                "pattern_set": "gray_code",
                "reference": True,
                "index": 1
            },
            parameters={
                "color": "Black"
            }
        ))
        
        # Gray code patterns
        foreground_color = "Blue" if use_blue else "White"
        pattern_index = 2
        
        for bit in range(num_bits):
            # Calculate stripe width for this bit - make MUCH larger for hardware compatibility
            # Use minimum 400px for the finest patterns - hardware needs very thick stripes
            base_width = 2 ** (num_bits - bit - 1)
            stripe_width = max(400, base_width * 32)  # Much thicker - 32x multiplier
            
            # Normal pattern
            patterns.append(PatternInfo(
                pattern_type="vertical_lines",
                name=f"gray_code_bit_{bit}",
                metadata={
                    "pattern_set": "gray_code",
                    "gray_code": True,
                    "bit": bit,
                    "inverted": False,
                    "index": pattern_index
                },
                parameters={
                    "foreground_color": foreground_color,
                    "background_color": "Black",
                    "foreground_width": stripe_width,
                    "background_width": stripe_width
                }
            ))
            pattern_index += 1
            
            # Inverted pattern
            if include_inverse:
                patterns.append(PatternInfo(
                    pattern_type="vertical_lines",
                    name=f"gray_code_bit_{bit}_inv",
                    metadata={
                        "pattern_set": "gray_code",
                        "gray_code": True,
                        "bit": bit,
                        "inverted": True,
                        "index": pattern_index
                    },
                    parameters={
                        "foreground_color": "Black",
                        "background_color": foreground_color,
                        "foreground_width": stripe_width,
                        "background_width": stripe_width
                    }
                ))
                pattern_index += 1
        
        logger.info(f"Created {len(patterns)} basic Gray code patterns with {num_bits} bits")
        return patterns
    
    def create_phase_shift_patterns(self,
                                   num_steps: int = None,
                                   frequencies: List[int] = None,
                                   use_blue: bool = True) -> List[PatternInfo]:
        """
        Create phase shift pattern sequence.
        
        Args:
            num_steps: Number of phase steps (default: 4)
            frequencies: List of frequencies for multi-frequency approach
            use_blue: Use blue instead of white for patterns
            
        Returns:
            List of PatternInfo objects describing the patterns
        """
        if num_steps is None:
            num_steps = self.default_phase_steps
        if frequencies is None:
            frequencies = self.default_frequencies
            
        patterns = []
        pattern_index = 0
        
        # Reference patterns
        patterns.append(PatternInfo(
            pattern_type="solid_field",
            name="reference_white",
            metadata={
                "pattern_set": "phase_shift",
                "reference": True,
                "index": pattern_index
            },
            parameters={
                "color": "White"
            }
        ))
        pattern_index += 1
        
        patterns.append(PatternInfo(
            pattern_type="solid_field",
            name="reference_black",
            metadata={
                "pattern_set": "phase_shift",
                "reference": True,
                "index": pattern_index
            },
            parameters={
                "color": "Black"
            }
        ))
        pattern_index += 1
        
        # Phase shift patterns for each frequency - use vertical lines instead of sinusoidal
        for freq_idx, frequency in enumerate(frequencies):
            for step in range(num_steps):
                phase = (step * 2 * np.pi) / num_steps
                
                # Convert frequency to stripe width for vertical lines
                stripe_width = max(10, 1024 // (frequency * 8))  # Approximate conversion
                
                patterns.append(PatternInfo(
                    pattern_type="vertical_lines",
                    name=f"phase_shift_f{frequency}_s{step}",
                    metadata={
                        "pattern_set": "phase_shift",
                        "phase_shift": True,
                        "frequency": frequency,
                        "frequency_index": freq_idx,
                        "step": step,
                        "phase": phase,
                        "num_steps": num_steps,
                        "index": pattern_index
                    },
                    parameters={
                        "foreground_color": "Blue" if use_blue else "White",
                        "background_color": "Black",
                        "foreground_width": stripe_width,
                        "background_width": stripe_width,
                        "frequency": frequency,
                        "phase": phase
                    }
                ))
                pattern_index += 1
        
        logger.info(f"Created {len(patterns)} phase shift patterns with {num_steps} steps and {len(frequencies)} frequencies")
        return patterns
    
    def create_mixed_patterns(self,
                             gray_bits: int = 5,
                             phase_steps: int = 4,
                             use_blue: bool = True) -> List[PatternInfo]:
        """
        Create mixed Gray code + phase shift patterns.
        
        This approach uses Gray code for coarse correspondence and
        phase shift for sub-pixel refinement.
        
        Args:
            gray_bits: Number of Gray code bits for coarse matching
            phase_steps: Number of phase steps for refinement
            use_blue: Use blue patterns instead of white
            
        Returns:
            List of PatternInfo objects
        """
        patterns = []
        
        # Add coarse Gray code patterns (fewer bits)
        gray_patterns = self.create_gray_code_patterns(
            num_bits=gray_bits,
            use_blue=use_blue,
            include_inverse=False  # Skip inverse for mixed mode
        )
        patterns.extend(gray_patterns)
        
        # Add fine phase shift patterns (single frequency)
        phase_patterns = self.create_phase_shift_patterns(
            num_steps=phase_steps,
            frequencies=[8],  # Single medium frequency
            use_blue=use_blue
        )
        # Skip reference patterns (already in Gray code)
        patterns.extend(phase_patterns[2:])
        
        # Update metadata to indicate mixed pattern set
        for pattern in patterns:
            pattern.metadata['mixed_mode'] = True
            
        logger.info(f"Created {len(patterns)} mixed patterns (Gray code + phase shift)")
        return patterns
    
    def create_custom_sequence(self, sequence_config: List[Dict[str, Any]]) -> List[PatternInfo]:
        """
        Create a custom pattern sequence from configuration.
        
        Args:
            sequence_config: List of pattern configurations
            
        Returns:
            List of PatternInfo objects
        """
        patterns = []
        
        for idx, config in enumerate(sequence_config):
            pattern_type = config.get('pattern_type', 'solid_field')
            name = config.get('name', f'custom_pattern_{idx}')
            metadata = config.get('metadata', {})
            parameters = config.get('parameters', {})
            
            # Add index to metadata
            metadata['index'] = idx
            metadata['custom_sequence'] = True
            
            patterns.append(PatternInfo(
                pattern_type=pattern_type,
                name=name,
                metadata=metadata,
                parameters=parameters
            ))
        
        logger.info(f"Created {len(patterns)} custom patterns")
        return patterns
    
    def get_pattern_info_dict(self, patterns: List[PatternInfo]) -> Dict[str, Any]:
        """
        Get pattern information as a dictionary for metadata storage.
        
        Args:
            patterns: List of PatternInfo objects
            
        Returns:
            Dictionary with pattern sequence information
        """
        pattern_types = set()
        total_gray_bits = 0
        total_phase_steps = 0
        phase_frequencies = set()
        
        for pattern in patterns:
            if pattern.metadata.get('gray_code'):
                total_gray_bits = max(total_gray_bits, pattern.metadata.get('bit', 0) + 1)
                pattern_types.add('gray_code')
            elif pattern.metadata.get('phase_shift'):
                total_phase_steps = max(total_phase_steps, pattern.metadata.get('num_steps', 0))
                phase_frequencies.add(pattern.metadata.get('frequency', 0))
                pattern_types.add('phase_shift')
        
        return {
            'num_patterns': len(patterns),
            'pattern_types': list(pattern_types),
            'gray_code_bits': total_gray_bits if total_gray_bits > 0 else None,
            'phase_shift_steps': total_phase_steps if total_phase_steps > 0 else None,
            'phase_frequencies': list(phase_frequencies) if phase_frequencies else None,
            'uses_blue_channel': any(
                p.parameters.get('foreground_color') == 'Blue' or 
                p.parameters.get('color_channel') == 'blue' 
                for p in patterns
            ),
            'pattern_sequence': [p.to_dict() for p in patterns]
        }
    
    def estimate_capture_time(self, patterns: List[PatternInfo], 
                            pattern_switch_time: float = 0.1,
                            capture_time: float = 0.05) -> float:
        """
        Estimate total time to capture pattern sequence.
        
        Args:
            patterns: List of patterns to capture
            pattern_switch_time: Time to switch patterns (seconds)
            capture_time: Time to capture one image pair (seconds)
            
        Returns:
            Estimated total time in seconds
        """
        num_patterns = len(patterns)
        total_time = num_patterns * (pattern_switch_time + capture_time)
        
        return total_time