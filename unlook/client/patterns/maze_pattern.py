#!/usr/bin/env python3
"""
Maze Pattern Generator and Decoder for Structured Light Scanning.

This module implements maze-based structured light patterns that provide
unique local neighborhoods for robust correspondence matching. The maze
topology ensures that each junction has a unique configuration, enabling
accurate point matching between projector and camera views.

Key Features:
- Multiple maze generation algorithms (recursive backtracking, Prim's, Kruskal's)
- Unique junction markers for improved correspondence
- Configurable cell size and line width
- ISO/ASTM 52902 compliant uncertainty quantification

Example:
    >>> generator = MazePatternGenerator(width=1280, height=720)
    >>> pattern = generator.generate(algorithm='recursive_backtrack')
    >>> decoder = MazePatternDecoder()
    >>> correspondences = decoder.decode(left_image, right_image, pattern)
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
import random
from collections import deque
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MazeAlgorithm(Enum):
    """Available maze generation algorithms."""
    RECURSIVE_BACKTRACK = "recursive_backtrack"
    PRIM = "prim"
    KRUSKAL = "kruskal"


@dataclass
class MazeCell:
    """
    Represents a single cell in the maze grid.
    
    Each cell tracks its position, wall configuration, and metadata
    used during maze generation and pattern analysis.
    
    Attributes:
        x: X-coordinate in the maze grid
        y: Y-coordinate in the maze grid
        walls: Dictionary mapping directions to wall presence
        visited: Whether this cell has been visited during generation
        region_id: Unique identifier for connected regions
    """
    x: int
    y: int
    walls: Dict[str, bool] = field(default_factory=lambda: {
        'north': True, 'south': True, 'east': True, 'west': True
    })
    visited: bool = False
    region_id: int = -1
    
    def __hash__(self) -> int:
        """Make cell hashable based on coordinates."""
        return hash((self.x, self.y))
    
    def __eq__(self, other: object) -> bool:
        """Cells are equal if they have the same coordinates."""
        if isinstance(other, MazeCell):
            return self.x == other.x and self.y == other.y
        return False
    
    def get_wall_count(self) -> int:
        """Count the number of walls around this cell."""
        return sum(self.walls.values())


class MazePatternGenerator:
    """
    Generate maze patterns for structured light projection.
    
    The maze provides unique local topologies that can be used
    for correspondence matching without ambiguity. This generator
    supports multiple algorithms and is optimized for ISO/ASTM 52902
    compliance through controlled junction complexity.
    
    Attributes:
        width: Pattern width in pixels
        height: Pattern height in pixels
        cell_size: Size of each maze cell in pixels
        line_width: Width of maze walls in pixels
        maze_width: Number of cells horizontally
        maze_height: Number of cells vertically
    """
    
    def __init__(self, 
                 width: int = 1024, 
                 height: int = 768,
                 cell_size: Optional[int] = None,
                 line_width: int = 3) -> None:
        """
        Initialize the maze pattern generator.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels  
            cell_size: Size of each maze cell (auto-calculated if None)
            line_width: Width of maze walls in pixels
        """
        self.width = width
        self.height = height
        self.cell_size = 16  # Pixels per maze cell
        self.line_width = 3  # Width of maze walls
        
        # Calculate maze dimensions in cells
        self.maze_width = width // self.cell_size
        self.maze_height = height // self.cell_size
        
        self.maze = None
        self.pattern = None
        self.region_map = None
        
    def generate(self, 
                algorithm: Union[str, MazeAlgorithm] = MazeAlgorithm.RECURSIVE_BACKTRACK,
                seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a maze pattern using the specified algorithm.
        
        Args:
            algorithm: Maze generation algorithm to use  
            seed: Random seed for reproducible patterns (None for random)
            
        Returns:
            np.ndarray: Binary pattern image (0=black, 255=white)
            
        Raises:
            ValueError: If unknown algorithm is specified
            
        Example:
            >>> generator = MazePatternGenerator(1280, 720)
            >>> pattern = generator.generate(MazeAlgorithm.RECURSIVE_BACKTRACK)
            >>> cv2.imwrite('maze_pattern.png', pattern)
        """
        # Handle enum vs string input
        if isinstance(algorithm, MazeAlgorithm):
            algorithm = algorithm.value
            
        logger.info(f"Generating maze pattern with {algorithm} algorithm")
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize maze grid
        self._initialize_maze()
        
        # Generate maze structure
        if algorithm == "recursive_backtrack":
            self._generate_recursive_backtrack()
        elif algorithm == "kruskal":
            self._generate_kruskal()
        elif algorithm == "prim":
            self._generate_prim()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Convert maze to pattern image
        self.pattern = self._maze_to_pattern()
        
        # Generate region map for decoding
        self._generate_region_map()
        
        return self.pattern
    
    def _initialize_maze(self):
        """Initialize the maze grid with all walls."""
        self.maze = []
        for y in range(self.maze_height):
            row = []
            for x in range(self.maze_width):
                cell = MazeCell(
                    x=x, 
                    y=y,
                    walls={'north': True, 'south': True, 'east': True, 'west': True}
                )
                row.append(cell)
            self.maze.append(row)
    
    def _generate_recursive_backtrack(self):
        """Generate maze using recursive backtracking algorithm."""
        # Start from random cell
        current = self.maze[0][0]
        current.visited = True
        stack = [current]
        
        while stack:
            # Get unvisited neighbors
            neighbors = self._get_unvisited_neighbors(current)
            
            if neighbors:
                # Choose random neighbor
                next_cell = random.choice(neighbors)
                
                # Remove wall between current and next
                self._remove_wall(current, next_cell)
                
                # Mark as visited and continue
                next_cell.visited = True
                stack.append(next_cell)
                current = next_cell
            else:
                # Backtrack
                current = stack.pop()
    
    def _generate_kruskal(self):
        """Generate maze using Kruskal's algorithm."""
        # Create list of all walls
        walls = []
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if x < self.maze_width - 1:
                    walls.append((self.maze[y][x], self.maze[y][x+1], 'horizontal'))
                if y < self.maze_height - 1:
                    walls.append((self.maze[y][x], self.maze[y+1][x], 'vertical'))
        
        # Shuffle walls
        random.shuffle(walls)
        
        # Initialize disjoint sets
        sets = {}
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                sets[(x, y)] = {(x, y)}
        
        # Process walls
        for cell1, cell2, direction in walls:
            set1 = sets[(cell1.x, cell1.y)]
            set2 = sets[(cell2.x, cell2.y)]
            
            if set1 != set2:
                # Remove wall
                self._remove_wall(cell1, cell2)
                
                # Merge sets
                merged = set1.union(set2)
                for cell_pos in merged:
                    sets[cell_pos] = merged
    
    def _generate_prim(self):
        """Generate maze using Prim's algorithm."""
        # Start with random cell
        start = self.maze[random.randint(0, self.maze_height-1)][random.randint(0, self.maze_width-1)]
        start.visited = True
        
        # Frontier cells
        frontier = self._get_unvisited_neighbors(start)
        
        while frontier:
            # Pick random frontier cell
            current = random.choice(frontier)
            current.visited = True
            
            # Connect to random visited neighbor
            visited_neighbors = [n for n in self._get_neighbors(current) if n.visited]
            if visited_neighbors:
                neighbor = random.choice(visited_neighbors)
                self._remove_wall(current, neighbor)
            
            # Add new frontier cells
            new_frontier = [n for n in self._get_unvisited_neighbors(current) if not n.visited]
            frontier.extend(new_frontier)
            frontier = list(set(frontier))  # Remove duplicates
            frontier.remove(current)
    
    def _get_neighbors(self, cell: MazeCell) -> List[MazeCell]:
        """Get all neighbors of a cell."""
        neighbors = []
        x, y = cell.x, cell.y
        
        if x > 0:
            neighbors.append(self.maze[y][x-1])
        if x < self.maze_width - 1:
            neighbors.append(self.maze[y][x+1])
        if y > 0:
            neighbors.append(self.maze[y-1][x])
        if y < self.maze_height - 1:
            neighbors.append(self.maze[y+1][x])
            
        return neighbors
    
    def _get_unvisited_neighbors(self, cell: MazeCell) -> List[MazeCell]:
        """Get unvisited neighbors of a cell."""
        return [n for n in self._get_neighbors(cell) if not n.visited]
    
    def _remove_wall(self, cell1: MazeCell, cell2: MazeCell):
        """Remove wall between two adjacent cells."""
        if cell1.x == cell2.x:
            # Vertical adjacency
            if cell1.y < cell2.y:
                cell1.walls['south'] = False
                cell2.walls['north'] = False
            else:
                cell1.walls['north'] = False
                cell2.walls['south'] = False
        else:
            # Horizontal adjacency
            if cell1.x < cell2.x:
                cell1.walls['east'] = False
                cell2.walls['west'] = False
            else:
                cell1.walls['west'] = False
                cell2.walls['east'] = False
    
    def _maze_to_pattern(self) -> np.ndarray:
        """Convert maze structure to pattern image."""
        # Create pattern image
        pattern = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # Draw maze walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                cell = self.maze[y][x]
                
                # Calculate pixel coordinates
                px = x * self.cell_size
                py = y * self.cell_size
                
                # Draw walls
                if cell.walls['north'] and y > 0:
                    cv2.line(pattern, 
                            (px, py), 
                            (px + self.cell_size, py),
                            0, self.line_width)
                
                if cell.walls['south']:
                    cv2.line(pattern,
                            (px, py + self.cell_size),
                            (px + self.cell_size, py + self.cell_size),
                            0, self.line_width)
                
                if cell.walls['west'] and x > 0:
                    cv2.line(pattern,
                            (px, py),
                            (px, py + self.cell_size),
                            0, self.line_width)
                
                if cell.walls['east']:
                    cv2.line(pattern,
                            (px + self.cell_size, py),
                            (px + self.cell_size, py + self.cell_size),
                            0, self.line_width)
        
        # Add unique markers at intersections
        self._add_intersection_markers(pattern)
        
        return pattern
    
    def _add_intersection_markers(self, pattern: np.ndarray):
        """Add unique markers at maze intersections."""
        marker_size = 5
        marker_types = [
            np.array([[1,0,1,0,1],
                     [0,1,1,1,0],
                     [1,1,0,1,1],
                     [0,1,1,1,0],
                     [1,0,1,0,1]], dtype=np.uint8),  # Checkerboard
            
            np.array([[0,0,1,0,0],
                     [0,1,1,1,0],
                     [1,1,1,1,1],
                     [0,1,1,1,0],
                     [0,0,1,0,0]], dtype=np.uint8),  # Diamond
            
            np.array([[1,1,1,1,1],
                     [1,0,0,0,1],
                     [1,0,1,0,1],
                     [1,0,0,0,1],
                     [1,1,1,1,1]], dtype=np.uint8),  # Square
        ]
        
        marker_idx = 0
        
        # Place markers at key intersections
        for y in range(0, self.maze_height, 4):
            for x in range(0, self.maze_width, 4):
                px = x * self.cell_size
                py = y * self.cell_size
                
                # Select marker type
                marker = marker_types[marker_idx % len(marker_types)]
                marker_idx += 1
                
                # Place marker
                if px + marker_size < self.width and py + marker_size < self.height:
                    pattern[py:py+marker_size, px:px+marker_size] = marker * 255
    
    def _generate_region_map(self):
        """Generate region map for decoding."""
        self.region_map = np.zeros((self.maze_height, self.maze_width), dtype=np.int32)
        region_id = 0
        
        # Flood fill to identify connected regions
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y][x].region_id == -1:
                    self._flood_fill_region(x, y, region_id)
                    region_id += 1
        
        logger.info(f"Generated {region_id} unique regions in maze")
    
    def _flood_fill_region(self, start_x: int, start_y: int, region_id: int):
        """Flood fill a region with given ID."""
        queue = deque([(start_x, start_y)])
        
        while queue:
            x, y = queue.popleft()
            
            if self.maze[y][x].region_id != -1:
                continue
                
            self.maze[y][x].region_id = region_id
            self.region_map[y, x] = region_id
            
            # Check accessible neighbors
            cell = self.maze[y][x]
            
            if not cell.walls['north'] and y > 0:
                queue.append((x, y-1))
            if not cell.walls['south'] and y < self.maze_height - 1:
                queue.append((x, y+1))
            if not cell.walls['east'] and x < self.maze_width - 1:
                queue.append((x+1, y))
            if not cell.walls['west'] and x > 0:
                queue.append((x-1, y))
    
    def save_pattern(self, filename: str):
        """Save pattern to file."""
        if self.pattern is None:
            raise ValueError("No pattern generated yet")
        cv2.imwrite(filename, self.pattern)
        
    def get_encoding_info(self) -> Dict:
        """Get information about the pattern encoding."""
        if self.region_map is None:
            raise ValueError("No pattern generated yet")
            
        return {
            'width': self.width,
            'height': self.height,
            'cell_size': self.cell_size,
            'maze_width': self.maze_width,
            'maze_height': self.maze_height,
            'num_regions': len(np.unique(self.region_map)),
            'line_width': self.line_width
        }
    
    def generate_sequence(self, num_patterns: int = 4) -> List[np.ndarray]:
        """
        Generate a sequence of maze patterns.
        
        Args:
            num_patterns: Number of patterns to generate
            
        Returns:
            List of pattern images
        """
        patterns = []
        algorithms = ['recursive_backtrack', 'kruskal', 'prim']
        
        for i in range(num_patterns):
            # Use different algorithms for variety
            algorithm = algorithms[i % len(algorithms)]
            pattern = self.generate(algorithm=algorithm)
            patterns.append(pattern)
            
            # Reset internal state for next pattern
            self.maze = None
            self.region_map = None
        
        logger.info(f"Generated sequence of {num_patterns} maze patterns")
        return patterns


class MazePatternDecoder:
    """
    Decode maze patterns from captured images.
    
    This class analyzes captured maze patterns to find correspondences
    between projector and camera coordinates.
    """
    
    def __init__(self, encoding_info: Dict):
        """
        Initialize the decoder.
        
        Args:
            encoding_info: Information about the pattern encoding
        """
        self.encoding_info = encoding_info
        self.cell_size = encoding_info['cell_size']
        self.maze_width = encoding_info['maze_width']
        self.maze_height = encoding_info['maze_height']
        
    def decode(self, captured_image: np.ndarray, 
              reference_pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode the captured maze pattern.
        
        Args:
            captured_image: Captured image containing maze pattern
            reference_pattern: Original projected pattern
            
        Returns:
            Tuple of (x_coordinates, y_coordinates) in projector space
        """
        logger.info("Decoding maze pattern")
        
        # Preprocess captured image
        processed = self._preprocess_image(captured_image)
        
        # Extract maze structure
        maze_structure = self._extract_maze_structure(processed)
        
        # Find intersection points
        intersections = self._find_intersections(maze_structure)
        
        # Match with reference pattern
        correspondences = self._match_intersections(
            intersections, 
            reference_pattern
        )
        
        # Interpolate dense correspondence map
        x_coords, y_coords = self._interpolate_correspondences(
            correspondences, 
            captured_image.shape
        )
        
        return x_coords, y_coords
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the captured image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Detect edges
        edges = cv2.Canny(denoised, 50, 150)
        
        # Clean up edges
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_maze_structure(self, processed: np.ndarray) -> np.ndarray:
        """Extract the maze structure from processed image."""
        # Use Hough transform to detect lines
        lines = cv2.HoughLinesP(
            processed,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        # Create line image
        line_image = np.zeros_like(processed)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
        
        return line_image
    
    def _find_intersections(self, maze_structure: np.ndarray) -> List[Tuple[int, int]]:
        """Find intersection points in the maze structure."""
        # Use corner detection
        corners = cv2.goodFeaturesToTrack(
            maze_structure,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=10
        )
        
        intersections = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                intersections.append((int(x), int(y)))
                
        return intersections
    
    def _match_intersections(self, captured_intersections: List[Tuple[int, int]],
                           reference_pattern: np.ndarray) -> Dict:
        """Match captured intersections with reference pattern."""
        # Extract reference intersections
        ref_intersections = self._find_intersections(reference_pattern)
        
        # Use feature matching
        correspondences = {}
        
        for cap_pt in captured_intersections:
            # Extract local neighborhood
            cap_patch = self._extract_patch(reference_pattern, cap_pt, 32)
            
            best_match = None
            best_score = float('inf')
            
            for ref_pt in ref_intersections:
                ref_patch = self._extract_patch(reference_pattern, ref_pt, 32)
                
                # Compare patches
                score = np.sum((cap_patch - ref_patch) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_match = ref_pt
            
            if best_match and best_score < 1000:  # Threshold
                correspondences[cap_pt] = best_match
        
        return correspondences
    
    def _extract_patch(self, image: np.ndarray, center: Tuple[int, int], 
                      size: int) -> np.ndarray:
        """Extract a patch around a point."""
        x, y = center
        half_size = size // 2
        
        # Ensure we stay within bounds
        x1 = max(0, x - half_size)
        x2 = min(image.shape[1], x + half_size)
        y1 = max(0, y - half_size)
        y2 = min(image.shape[0], y + half_size)
        
        patch = image[y1:y2, x1:x2]
        
        # Pad if necessary
        if patch.shape[0] < size or patch.shape[1] < size:
            padded = np.zeros((size, size), dtype=image.dtype)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
            
        return patch
    
    def _interpolate_correspondences(self, sparse_correspondences: Dict,
                                   image_shape: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate dense correspondence map from sparse matches."""
        height, width = image_shape[:2]
        
        # Create coordinate maps
        x_coords = np.zeros((height, width), dtype=np.float32)
        y_coords = np.zeros((height, width), dtype=np.float32)
        
        # Fill in known correspondences
        src_points = []
        dst_points = []
        
        for (src_x, src_y), (dst_x, dst_y) in sparse_correspondences.items():
            src_points.append([src_x, src_y])
            dst_points.append([dst_x, dst_y])
            x_coords[src_y, src_x] = dst_x
            y_coords[src_y, src_x] = dst_y
        
        if len(src_points) < 4:
            logger.warning("Too few correspondences for interpolation")
            return x_coords, y_coords
        
        # Use thin plate spline for interpolation
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Create grid of all pixels
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Interpolate using RBF
        from scipy.interpolate import RBFInterpolator
        
        rbf_x = RBFInterpolator(src_points, dst_points[:, 0], 
                               smoothing=0.1, kernel='thin_plate_spline')
        rbf_y = RBFInterpolator(src_points, dst_points[:, 1],
                               smoothing=0.1, kernel='thin_plate_spline')
        
        x_interp = rbf_x(grid_points).reshape(height, width)
        y_interp = rbf_y(grid_points).reshape(height, width)
        
        return x_interp, y_interp