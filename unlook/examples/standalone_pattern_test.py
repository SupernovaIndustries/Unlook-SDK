#!/usr/bin/env python3
"""
Standalone test for enhanced pattern generation.

This script directly includes the pattern generation code to avoid import issues.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Create output directory
output_dir = Path("pattern_test_output")
output_dir.mkdir(exist_ok=True)

# Pattern generation functions copied from enhanced_patterns.py
def generate_multi_frequency_patterns(
    width: int,
    height: int,
    frequencies: List[int] = [1, 4, 8, 16, 32, 64],
    steps_per_frequency: int = 4,
    orientation: str = "both"
) -> List[Dict[str, Any]]:
    """
    Generate multi-frequency phase shift patterns for enhanced depth recovery.
    """
    patterns = []
    
    # Add white and black reference patterns first
    patterns.append({
        "pattern_type": "solid_field",
        "color": "White",
        "name": "white_reference"
    })
    
    patterns.append({
        "pattern_type": "solid_field",
        "color": "Black",
        "name": "black_reference"
    })
    
    # Generate patterns for each orientation
    orientations = []
    if orientation == "horizontal" or orientation == "both":
        orientations.append("horizontal")
    if orientation == "vertical" or orientation == "both":
        orientations.append("vertical")
    
    # Generate for each orientation and frequency
    for orient in orientations:
        for freq in frequencies:
            for step in range(steps_per_frequency):
                phase = 2 * np.pi * step / steps_per_frequency
                patterns.append({
                    "pattern_type": "multi_frequency",
                    "orientation": orient,
                    "frequency": freq,
                    "phase": phase,
                    "name": f"multi_freq_{orient[0]}_{freq}_{step}"
                })
                
    print(f"Generated {len(patterns)} multi-frequency patterns")
    return patterns

def generate_multi_scale_patterns(
    width: int,
    height: int,
    num_bits: int = 10,
    orientation: str = "both"
) -> List[Dict[str, Any]]:
    """
    Generate multi-scale Gray code patterns with varying line widths.
    """
    patterns = []
    
    # Add white and black reference patterns first
    patterns.append({
        "pattern_type": "solid_field",
        "color": "White",
        "name": "white_reference"
    })
    
    patterns.append({
        "pattern_type": "solid_field",
        "color": "Black",
        "name": "black_reference"
    })
    
    # Generate patterns for each orientation
    orientations = []
    if orientation == "horizontal" or orientation == "both":
        orientations.append("horizontal")
    if orientation == "vertical" or orientation == "both":
        orientations.append("vertical")
    
    # For each orientation, generate a set of large to small scale patterns
    for orient in orientations:
        # Generate in reverse order - larger stripes first (less bits)
        for bit in range(num_bits - 1, -1, -1):
            # Normal pattern
            patterns.append({
                "pattern_type": "multi_scale",
                "orientation": orient,
                "bit": bit,
                "inverted": False,
                "name": f"multi_scale_{orient[0]}_bit{bit}"
            })
            
            # Inverted pattern for robust decoding
            patterns.append({
                "pattern_type": "multi_scale",
                "orientation": orient,
                "bit": bit,
                "inverted": True,
                "name": f"multi_scale_{orient[0]}_bit{bit}_inv"
            })
    
    print(f"Generated {len(patterns)} multi-scale patterns")
    return patterns

def generate_variable_width_gray_code(
    width: int,
    height: int,
    min_bits: int = 4,
    max_bits: int = 10,
    orientation: str = "both"
) -> List[Dict[str, Any]]:
    """
    Generate Gray code patterns with variable stripe widths.
    """
    patterns = []
    
    # Add white and black reference patterns
    patterns.append({
        "pattern_type": "solid_field",
        "color": "White",
        "name": "white_reference"
    })
    
    patterns.append({
        "pattern_type": "solid_field",
        "color": "Black",
        "name": "black_reference"
    })
    
    # Generate patterns for each orientation
    orientations = []
    if orientation == "horizontal" or orientation == "both":
        orientations.append("horizontal")
    if orientation == "vertical" or orientation == "both":
        orientations.append("vertical")
    
    # For each orientation, generate variable width patterns
    for orient in orientations:
        # Generate from large to small stripes
        for bit in range(min_bits, max_bits + 1):
            # Normal pattern
            patterns.append({
                "pattern_type": "variable_width",
                "orientation": orient,
                "bit": bit - min_bits,  # Normalize bit index
                "width_bits": bit,      # Actual bit width
                "inverted": False,
                "name": f"var_width_{orient[0]}_bit{bit}"
            })
            
            # Inverted pattern
            patterns.append({
                "pattern_type": "variable_width",
                "orientation": orient,
                "bit": bit - min_bits,
                "width_bits": bit,
                "inverted": True,
                "name": f"var_width_{orient[0]}_bit{bit}_inv"
            })
    
    print(f"Generated {len(patterns)} variable width patterns")
    return patterns

def encode_pattern(
    pattern: Dict[str, Any],
    width: int,
    height: int
) -> np.ndarray:
    """
    Encode a single pattern based on its type and parameters.
    """
    # Create blank image
    img = np.zeros((height, width), dtype=np.uint8)
    
    if pattern["pattern_type"] == "solid_field":
        # Solid white or black field
        if pattern["color"] == "White":
            img.fill(255)
        # Black field is already zeros
        
    elif pattern["pattern_type"] == "gray_code" or pattern["pattern_type"] == "multi_scale":
        # Gray code pattern
        orientation = pattern["orientation"]
        bit = pattern["bit"]
        inverted = pattern.get("inverted", False)
        
        # Calculate stripe width based on bit position
        stripe_width = 2 ** bit
        
        # Fill with Gray code pattern
        if orientation == "horizontal":
            for x in range(width):
                if ((x // stripe_width) % 2) == 0:
                    img[:, x] = 255 if not inverted else 0
                else:
                    img[:, x] = 0 if not inverted else 255
        else:  # vertical
            for y in range(height):
                if ((y // stripe_width) % 2) == 0:
                    img[y, :] = 255 if not inverted else 0
                else:
                    img[y, :] = 0 if not inverted else 255
    
    elif pattern["pattern_type"] == "variable_width":
        # Variable width Gray code
        orientation = pattern["orientation"]
        width_bits = pattern["width_bits"]  # Actual bit width to use
        inverted = pattern.get("inverted", False)
        
        # Calculate stripe width
        stripe_width = 2 ** width_bits
        
        # Fill with Gray code pattern
        if orientation == "horizontal":
            for x in range(width):
                if ((x // stripe_width) % 2) == 0:
                    img[:, x] = 255 if not inverted else 0
                else:
                    img[:, x] = 0 if not inverted else 255
        else:  # vertical
            for y in range(height):
                if ((y // stripe_width) % 2) == 0:
                    img[y, :] = 255 if not inverted else 0
                else:
                    img[y, :] = 0 if not inverted else 255
    
    elif pattern["pattern_type"] == "multi_frequency":
        # Multi-frequency phase shift pattern
        orientation = pattern["orientation"]
        frequency = pattern["frequency"]
        phase = pattern["phase"]
        
        # Create coordinate grid
        y, x = np.mgrid[0:height, 0:width]
        
        # Calculate sinusoidal pattern 
        if orientation == "horizontal":
            pattern_values = np.cos(2 * np.pi * frequency * x / width + phase)
        else:  # vertical
            pattern_values = np.cos(2 * np.pi * frequency * y / height + phase)
        
        # Scale to 0-255 range
        img = ((pattern_values + 1) / 2 * 255).astype(np.uint8)
    
    return img

def main():
    """Generate and save sample patterns."""
    # Define pattern resolution
    width, height = 1024, 768
    
    print("Testing multi-scale patterns...")
    multi_scale_patterns = generate_multi_scale_patterns(
        width=width,
        height=height,
        num_bits=10,
        orientation="both"
    )
    
    print("Testing multi-frequency patterns...")
    multi_frequency_patterns = generate_multi_frequency_patterns(
        width=width,
        height=height,
        frequencies=[1, 8, 16],
        steps_per_frequency=4,
        orientation="both"
    )
    
    print("Testing variable width patterns...")
    variable_width_patterns = generate_variable_width_gray_code(
        width=width,
        height=height,
        min_bits=4,
        max_bits=10,
        orientation="both"
    )
    
    # Save several examples from each type
    save_samples(multi_scale_patterns, "multi_scale", width, height, output_dir)
    save_samples(multi_frequency_patterns, "multi_frequency", width, height, output_dir)
    save_samples(variable_width_patterns, "variable_width", width, height, output_dir)
    
    print(f"Saved sample patterns to {output_dir.absolute()}")
    
def save_samples(patterns, name, width, height, output_dir):
    """Save sample patterns from the list."""
    if not patterns:
        print(f"No {name} patterns generated")
        return
    
    # Save a few patterns from different parts of the sequence
    indices = [
        2,  # Skip white/black reference
        len(patterns) // 4,
        len(patterns) // 2,
        3 * len(patterns) // 4,
        min(len(patterns) - 1, len(patterns) - 3)
    ]
    
    for i, idx in enumerate(indices):
        if idx >= len(patterns):
            continue
            
        pattern = patterns[idx]
        
        # Encode the pattern
        img = encode_pattern(pattern, width, height)
        
        # Save the pattern
        pattern_name = pattern.get("name", f"pattern_{idx}")
        output_path = output_dir / f"{name}_{i}_{pattern_name}.png"
        plt.imsave(str(output_path), img, cmap='gray')
        print(f"Saved {name} pattern {i} to {output_path}")

if __name__ == "__main__":
    main()