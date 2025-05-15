"""
Pattern generation and decoding modules for structured light scanning.

This module provides various pattern types for 3D scanning:
- Maze patterns: Unique local topologies for correspondence matching
- Voronoi patterns: Dense surface reconstruction with rich features
- Hybrid ArUco patterns: Combined structured light with fiducial markers
"""

from .maze_pattern import MazePatternGenerator, MazePatternDecoder
from .voronoi_pattern import VoronoiPatternGenerator, VoronoiPatternDecoder
from .hybrid_aruco_pattern import HybridArUcoPatternGenerator, HybridArUcoPatternDecoder

__all__ = [
    'MazePatternGenerator',
    'MazePatternDecoder',
    'VoronoiPatternGenerator',
    'VoronoiPatternDecoder',
    'HybridArUcoPatternGenerator',
    'HybridArUcoPatternDecoder'
]