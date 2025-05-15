"""
Voronoi pattern generator and decoder for structured light scanning.

This module implements a Voronoi-based pattern system that creates
unique cell patterns across the projected image. This approach offers:
- Dense surface reconstruction with rich local features
- Robust correspondence matching in textured regions
- Better performance with curved surfaces
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
import json

try:
    from scipy.spatial import Voronoi, voronoi_plot_2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Voronoi patterns will not work.")

logger = logging.getLogger(__name__)


class VoronoiPatternGenerator:
    """Generate Voronoi patterns for structured light projection."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        """
        Initialize Voronoi pattern generator.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
        """
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        logger.info(f"Initialized VoronoiPatternGenerator with resolution {width}x{height}")
    
    def generate(self, 
                num_points: int = 100,
                color_scheme: str = 'grayscale') -> np.ndarray:
        """
        Generate a Voronoi pattern.
        
        Args:
            num_points: Number of Voronoi seed points
            color_scheme: Color scheme to use ('grayscale', 'binary', 'colored')
            
        Returns:
            Pattern image as numpy array
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Voronoi patterns. Install with: pip install scipy")
            
        logger.info(f"Generating Voronoi pattern with {num_points} points and {color_scheme} color scheme")
        
        # Generate seed points
        points = self._generate_seed_points(num_points)
        
        # Generate Voronoi diagram
        vor = Voronoi(points)
        
        # Render pattern
        if color_scheme == 'grayscale':
            pattern = self._render_grayscale(vor)
        elif color_scheme == 'binary':
            pattern = self._render_binary(vor)
        elif color_scheme == 'colored':
            pattern = self._render_colored(vor)
        else:
            raise ValueError(f"Unknown color scheme: {color_scheme}")
        
        return pattern
    
    def _generate_seed_points(self, num_points: int) -> np.ndarray:
        """Generate seed points for Voronoi diagram."""
        # Use a combination of regular and random points
        regular_points = int(num_points * 0.3)
        random_points = num_points - regular_points
        
        # Create regular grid points
        grid_size = int(np.sqrt(regular_points))
        x_step = self.width / (grid_size + 1)
        y_step = self.height / (grid_size + 1)
        
        regular = []
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                # Add some jitter to regular points
                x = i * x_step + self.rng.uniform(-x_step/4, x_step/4)
                y = j * y_step + self.rng.uniform(-y_step/4, y_step/4)
                regular.append([x, y])
        
        # Add random points
        random = self.rng.uniform(0, [self.width, self.height], size=(random_points, 2))
        
        points = np.vstack([regular, random])
        
        # Add boundary points to ensure complete coverage
        boundary_points = self._generate_boundary_points(20)
        points = np.vstack([points, boundary_points])
        
        return points
    
    def _generate_boundary_points(self, num_per_side: int) -> np.ndarray:
        """Generate points along the boundary."""
        points = []
        
        # Top edge
        for i in range(num_per_side):
            x = (i + 1) * self.width / (num_per_side + 1)
            points.append([x, 0])
        
        # Bottom edge
        for i in range(num_per_side):
            x = (i + 1) * self.width / (num_per_side + 1)
            points.append([x, self.height])
        
        # Left edge
        for i in range(num_per_side):
            y = (i + 1) * self.height / (num_per_side + 1)
            points.append([0, y])
        
        # Right edge
        for i in range(num_per_side):
            y = (i + 1) * self.height / (num_per_side + 1)
            points.append([self.width, y])
        
        return np.array(points)
    
    def _render_grayscale(self, vor: Voronoi) -> np.ndarray:
        """Render Voronoi pattern in grayscale."""
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        coords = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Compute distances to all Voronoi centers
        centers = vor.points
        distances = np.zeros((coords.shape[0], centers.shape[0]))
        
        for i, center in enumerate(centers):
            distances[:, i] = np.sqrt(((coords - center) ** 2).sum(axis=1))
        
        # Find nearest center for each pixel
        nearest = np.argmin(distances, axis=1)
        
        # Assign grayscale values based on distance to nearest center
        min_distances = distances[np.arange(len(nearest)), nearest]
        
        # Normalize distances
        max_dist = np.sqrt((self.width/2)**2 + (self.height/2)**2) / 4
        normalized = np.clip(min_distances / max_dist, 0, 1)
        
        # Create pattern with gradient from center to edges
        pattern_flat = (255 * (1 - normalized)).astype(np.uint8)
        pattern = pattern_flat.reshape(self.height, self.width)
        
        # Add edges
        edges = self._extract_edges(vor)
        pattern = self._draw_edges(pattern, edges, value=128)
        
        return pattern
    
    def _render_binary(self, vor: Voronoi) -> np.ndarray:
        """Render Voronoi pattern in binary (checkerboard style)."""
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        coords = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Find nearest Voronoi center for each pixel
        centers = vor.points
        tree = cv2.flann_Index(centers.astype(np.float32), 
                             {'algorithm': 1, 'trees': 5})
        
        indices, _ = tree.knnSearch(coords.astype(np.float32), 1)
        nearest = indices.reshape(self.height, self.width)
        
        # Create checkerboard pattern
        for i in range(len(centers)):
            mask = (nearest == i)
            if i % 2 == 0:
                pattern[mask] = 255
        
        # Add edges
        edges = self._extract_edges(vor)
        pattern = self._draw_edges(pattern, edges, value=128)
        
        return pattern
    
    def _render_colored(self, vor: Voronoi) -> np.ndarray:
        """Render Voronoi pattern with unique colors for each cell."""
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        coords = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Find nearest Voronoi center for each pixel
        centers = vor.points
        tree = cv2.flann_Index(centers.astype(np.float32), 
                             {'algorithm': 1, 'trees': 5})
        
        indices, _ = tree.knnSearch(coords.astype(np.float32), 1)
        nearest = indices.reshape(self.height, self.width)
        
        # Generate unique colors for each cell
        num_cells = len(centers)
        colors = self._generate_distinct_colors(num_cells)
        
        # Color each cell
        for i in range(num_cells):
            mask = (nearest == i)
            pattern[mask] = colors[i]
        
        # Add white edges
        edges = self._extract_edges(vor)
        pattern = self._draw_edges(pattern, edges, value=(255, 255, 255))
        
        return pattern
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors."""
        colors = []
        
        # Use HSV color space for better distribution
        for i in range(n):
            hue = (i * 360 / n) % 360
            saturation = 0.7 + 0.3 * (i % 3) / 3
            value = 0.7 + 0.3 * (i % 5) / 5
            
            # Convert HSV to RGB
            h = hue / 60
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            
            colors.append((r, g, b))
        
        return colors
    
    def _extract_edges(self, vor: Voronoi) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Extract Voronoi edges for rendering."""
        edges = []
        
        for simplex in vor.ridge_vertices:
            if -1 not in simplex:
                i, j = simplex
                p1 = vor.vertices[i]
                p2 = vor.vertices[j]
                
                # Check if edge is within bounds
                if (0 <= p1[0] <= self.width and 0 <= p1[1] <= self.height and
                    0 <= p2[0] <= self.width and 0 <= p2[1] <= self.height):
                    edges.append((p1, p2))
        
        return edges
    
    def _draw_edges(self, pattern: np.ndarray, edges: List, value) -> np.ndarray:
        """Draw edges on the pattern."""
        result = pattern.copy()
        
        for p1, p2 in edges:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if len(result.shape) == 2:
                cv2.line(result, pt1, pt2, value, 1, cv2.LINE_AA)
            else:
                cv2.line(result, pt1, pt2, value, 1, cv2.LINE_AA)
        
        return result
    
    def generate_sequence(self, 
                        num_patterns: int = 4,
                        num_points_range: Tuple[int, int] = (50, 200)) -> List[np.ndarray]:
        """
        Generate a sequence of Voronoi patterns.
        
        Args:
            num_patterns: Number of patterns to generate
            num_points_range: Range of number of points (min, max)
            
        Returns:
            List of pattern images
        """
        patterns = []
        
        for i in range(num_patterns):
            # Vary the number of points for each pattern
            num_points = int(np.linspace(num_points_range[0], 
                                       num_points_range[1], 
                                       num_patterns)[i])
            
            # Alternate between different render styles
            if i % 3 == 0:
                pattern = self.generate(num_points, 'grayscale')
            elif i % 3 == 1:
                pattern = self.generate(num_points, 'binary')
            else:
                pattern = self.generate(num_points // 2, 'colored')
            
            patterns.append(pattern)
        
        return patterns


class VoronoiPatternDecoder:
    """Decode Voronoi patterns from captured images."""
    
    def __init__(self):
        """Initialize Voronoi pattern decoder."""
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        logger.info("Initialized VoronoiPatternDecoder")
    
    def decode(self, 
              captured_patterns: List[np.ndarray],
              reference_patterns: List[np.ndarray]) -> np.ndarray:
        """
        Decode correspondence from captured Voronoi patterns.
        
        Args:
            captured_patterns: List of captured pattern images
            reference_patterns: List of reference pattern images
            
        Returns:
            Correspondence map
        """
        logger.info(f"Decoding {len(captured_patterns)} Voronoi patterns")
        
        height, width = captured_patterns[0].shape[:2]
        correspondence_map = np.zeros((height, width, 2), dtype=np.float32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        
        # Process each pattern pair
        for i, (captured, reference) in enumerate(zip(captured_patterns, reference_patterns)):
            logger.debug(f"Processing pattern pair {i+1}/{len(captured_patterns)}")
            
            # Extract and match features
            matches, keypoints_cap, keypoints_ref = self._match_features(captured, reference)
            
            # Build correspondence from matches
            local_correspondence = self._build_correspondence(
                matches, keypoints_cap, keypoints_ref, (height, width))
            
            # Accumulate correspondence
            mask = local_correspondence[..., 0] > 0
            correspondence_map[mask] = local_correspondence[mask]
            confidence_map[mask] += 1
        
        # Normalize by confidence
        valid_mask = confidence_map > 0
        correspondence_map[valid_mask] /= confidence_map[valid_mask, np.newaxis]
        
        return correspondence_map
    
    def _match_features(self, 
                       captured: np.ndarray, 
                       reference: np.ndarray) -> Tuple[List, List, List]:
        """Extract and match features between captured and reference patterns."""
        # Detect features
        kp_cap, desc_cap = self.feature_detector.detectAndCompute(captured, None)
        kp_ref, desc_ref = self.feature_detector.detectAndCompute(reference, None)
        
        if desc_cap is None or desc_ref is None:
            return [], [], []
        
        # Match features
        matches = self.matcher.knnMatch(desc_cap, desc_ref, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return good_matches, kp_cap, kp_ref
    
    def _build_correspondence(self,
                            matches: List,
                            keypoints_cap: List,
                            keypoints_ref: List,
                            output_shape: Tuple[int, int]) -> np.ndarray:
        """Build correspondence map from feature matches."""
        correspondence = np.zeros((*output_shape, 2), dtype=np.float32)
        
        if not matches:
            return correspondence
        
        # Get matched points
        src_pts = np.float32([keypoints_cap[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in matches])
        
        # Find homography
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Warp correspondence grid
                x, y = np.meshgrid(np.arange(output_shape[1]), 
                                 np.arange(output_shape[0]))
                grid_pts = np.stack([x.ravel(), y.ravel(), 
                                   np.ones_like(x.ravel())], axis=1)
                
                # Apply homography
                warped_pts = (M @ grid_pts.T).T
                warped_pts = warped_pts[:, :2] / warped_pts[:, 2:3]
                
                # Reshape to correspondence map
                correspondence = warped_pts.reshape(*output_shape, 2)
                
                # Clip to valid range
                correspondence[..., 0] = np.clip(correspondence[..., 0], 
                                               0, output_shape[1] - 1)
                correspondence[..., 1] = np.clip(correspondence[..., 1], 
                                               0, output_shape[0] - 1)
        except cv2.error:
            logger.warning("Failed to compute homography")
        
        return correspondence
    
    def decode_with_cells(self,
                         captured_patterns: List[np.ndarray],
                         reference_cells: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Decode using pre-computed Voronoi cells.
        
        Args:
            captured_patterns: List of captured pattern images
            reference_cells: Dictionary mapping cell ID to mask
            
        Returns:
            Cell ID map
        """
        height, width = captured_patterns[0].shape[:2]
        cell_map = np.zeros((height, width), dtype=np.int32)
        
        # Process each captured pattern
        for pattern in captured_patterns:
            # Extract cell features
            cell_features = self._extract_cell_features(pattern, reference_cells)
            
            # Match cells
            for y in range(height):
                for x in range(width):
                    best_cell = self._find_best_cell(pattern[y, x], cell_features)
                    if best_cell >= 0:
                        cell_map[y, x] = best_cell
        
        return cell_map
    
    def _extract_cell_features(self, 
                             pattern: np.ndarray,
                             reference_cells: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """Extract features for each Voronoi cell."""
        features = {}
        
        for cell_id, mask in reference_cells.items():
            # Extract cell region
            cell_region = pattern[mask]
            
            if len(cell_region) > 0:
                features[cell_id] = {
                    'mean': np.mean(cell_region),
                    'std': np.std(cell_region),
                    'histogram': cv2.calcHist([cell_region], [0], None, 
                                            [32], [0, 256]).flatten()
                }
        
        return features
    
    def _find_best_cell(self, 
                       pixel_value: float,
                       cell_features: Dict[int, Dict]) -> int:
        """Find the best matching cell for a pixel."""
        best_cell = -1
        best_score = float('inf')
        
        for cell_id, features in cell_features.items():
            # Simple distance metric
            score = abs(pixel_value - features['mean'])
            
            if score < best_score:
                best_score = score
                best_cell = cell_id
        
        return best_cell
    
    def save_pattern_data(self, 
                        patterns: List[np.ndarray],
                        filename: str):
        """Save pattern data for later use."""
        data = {
            'width': patterns[0].shape[1],
            'height': patterns[0].shape[0],
            'num_patterns': len(patterns),
            'patterns': [pattern.tolist() for pattern in patterns]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved pattern data to {filename}")
    
    def load_pattern_data(self, filename: str) -> List[np.ndarray]:
        """Load pattern data from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        patterns = [np.array(pattern, dtype=np.uint8) 
                   for pattern in data['patterns']]
        
        logger.info(f"Loaded {len(patterns)} patterns from {filename}")
        return patterns


if __name__ == "__main__":
    # Test the Voronoi pattern generator
    generator = VoronoiPatternGenerator(1280, 720)
    
    # Generate different types of patterns
    grayscale_pattern = generator.generate(100, 'grayscale')
    cv2.imwrite('voronoi_grayscale.png', grayscale_pattern)
    
    binary_pattern = generator.generate(100, 'binary')
    cv2.imwrite('voronoi_binary.png', binary_pattern)
    
    colored_pattern = generator.generate(50, 'colored')
    cv2.imwrite('voronoi_colored.png', colored_pattern)
    
    # Generate a sequence
    sequence = generator.generate_sequence(4, (50, 150))
    for i, pattern in enumerate(sequence):
        cv2.imwrite(f'voronoi_sequence_{i}.png', pattern)
    
    print("Voronoi patterns generated successfully")