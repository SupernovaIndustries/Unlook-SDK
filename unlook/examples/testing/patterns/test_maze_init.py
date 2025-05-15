#!/usr/bin/env python3
"""Test MazePatternDecoder initialization"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from unlook.client.patterns import (
        MazePatternGenerator, MazePatternDecoder
    )
    
    print("Imported patterns module successfully")
    
    # Test MazePatternGenerator
    generator = MazePatternGenerator(1280, 720)
    print("Created MazePatternGenerator successfully")
    
    # Test MazePatternDecoder
    encoding_info = {
        'cell_size': 16,
        'maze_width': 80,
        'maze_height': 45
    }
    decoder = MazePatternDecoder(encoding_info)
    print("Created MazePatternDecoder successfully")
    
except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())