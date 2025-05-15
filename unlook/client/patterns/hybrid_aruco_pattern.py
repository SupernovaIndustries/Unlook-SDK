"""
Hybrid ArUco pattern generator and decoder for structured light scanning.

This module combines traditional structured light patterns with ArUco markers
to provide robust correspondence matching and global registration. Benefits:
- Absolute position reference from ArUco markers
- High-resolution correspondence from structured light
- Automatic calibration and registration
- Robust to partial occlusions
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ArUcoMarkerInfo:
    """Information about an ArUco marker."""
    id: int
    position: Tuple[int, int]  # Center position in pattern
    size: int  # Size in pixels
    corners: np.ndarray  # 4x2 array of corner positions


class HybridArUcoPatternGenerator:
    """Generate hybrid patterns combining structured light with ArUco markers."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        """
        Initialize hybrid ArUco pattern generator.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
        """
        self.width = width
        self.height = height
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.marker_size = 80  # Default marker size in pixels
        self.marker_spacing = 200  # Spacing between markers
        logger.info(f"Initialized HybridArUcoPatternGenerator with resolution {width}x{height}")
    
    def generate(self,
                base_pattern: str = 'gray_code',
                num_markers: int = 9) -> Tuple[np.ndarray, List[ArUcoMarkerInfo]]:
        """
        Generate a hybrid pattern with ArUco markers overlaid on base pattern.
        
        Args:
            base_pattern: Type of base pattern ('gray_code', 'phase_shift', 'checkerboard')
            num_markers: Number of ArUco markers to embed
            
        Returns:
            Tuple of (pattern image, list of marker info)
        """
        logger.info(f"Generating hybrid pattern with {base_pattern} and {num_markers} markers")
        
        # Generate base pattern
        if base_pattern == 'gray_code':
            base = self._generate_gray_code_pattern()
        elif base_pattern == 'phase_shift':
            base = self._generate_phase_shift_pattern()
        elif base_pattern == 'checkerboard':
            base = self._generate_checkerboard_pattern()
        else:
            raise ValueError(f"Unknown base pattern: {base_pattern}")
        
        # Plan marker positions
        marker_positions = self._plan_marker_positions(num_markers)
        
        # Embed ArUco markers
        pattern, markers_info = self._embed_aruco_markers(base, marker_positions)
        
        return pattern, markers_info
    
    def _generate_gray_code_pattern(self) -> np.ndarray:
        """Generate a base Gray code pattern."""
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Vertical stripes
        num_stripes = 32
        stripe_width = self.width // num_stripes
        
        for i in range(num_stripes):
            if i % 2 == 0:
                pattern[:, i*stripe_width:(i+1)*stripe_width] = 255
        
        return pattern
    
    def _generate_phase_shift_pattern(self) -> np.ndarray:
        """Generate a base phase shift pattern."""
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Sinusoidal pattern
        x = np.arange(self.width)
        frequency = 8  # cycles across width
        phase = np.sin(2 * np.pi * frequency * x / self.width)
        
        # Normalize to 0-255
        normalized = ((phase + 1) / 2 * 255).astype(np.uint8)
        pattern[:] = normalized
        
        return pattern
    
    def _generate_checkerboard_pattern(self) -> np.ndarray:
        """Generate a base checkerboard pattern."""
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create checkerboard
        square_size = 40
        for y in range(0, self.height, square_size * 2):
            for x in range(0, self.width, square_size * 2):
                pattern[y:y+square_size, x:x+square_size] = 255
                pattern[y+square_size:y+2*square_size, 
                       x+square_size:x+2*square_size] = 255
        
        return pattern
    
    def _plan_marker_positions(self, num_markers: int) -> List[Tuple[int, int]]:
        """Plan positions for ArUco markers in a grid layout."""
        positions = []
        
        # Calculate grid dimensions
        aspect_ratio = self.width / self.height
        rows = int(np.sqrt(num_markers / aspect_ratio))
        cols = int(np.ceil(num_markers / rows))
        
        # Calculate spacing
        x_spacing = self.width / (cols + 1)
        y_spacing = self.height / (rows + 1)
        
        # Generate positions
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= num_markers:
                    break
                
                x = int((col + 1) * x_spacing)
                y = int((row + 1) * y_spacing)
                positions.append((x, y))
        
        return positions
    
    def _embed_aruco_markers(self, 
                           base_pattern: np.ndarray,
                           positions: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[ArUcoMarkerInfo]]:
        """Embed ArUco markers into the base pattern."""
        pattern = base_pattern.copy()
        markers_info = []
        
        for i, (x, y) in enumerate(positions):
            # Generate ArUco marker
            marker_id = i
            marker_img = np.zeros((self.marker_size, self.marker_size), 
                                dtype=np.uint8)
            cv2.aruco.drawMarker(self.aruco_dict, marker_id, 
                               self.marker_size, marker_img)
            
            # Add white border to marker
            border_size = 10
            marker_with_border = np.ones((self.marker_size + 2*border_size,
                                        self.marker_size + 2*border_size), 
                                       dtype=np.uint8) * 255
            marker_with_border[border_size:-border_size, 
                             border_size:-border_size] = marker_img
            
            # Calculate placement position
            half_size = marker_with_border.shape[0] // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(self.width, x + half_size)
            y2 = min(self.height, y + half_size)
            
            # Embed marker
            marker_region = marker_with_border[
                max(0, half_size - x):half_size + (x2 - x),
                max(0, half_size - y):half_size + (y2 - y)
            ]
            pattern[y1:y2, x1:x2] = marker_region
            
            # Store marker info
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            
            markers_info.append(ArUcoMarkerInfo(
                id=marker_id,
                position=(x, y),
                size=self.marker_size,
                corners=corners
            ))
        
        return pattern, markers_info
    
    def generate_sequence(self,
                         num_patterns: int = 8,
                         base_patterns: List[str] = None) -> List[Tuple[np.ndarray, List[ArUcoMarkerInfo]]]:
        """
        Generate a sequence of hybrid patterns.
        
        Args:
            num_patterns: Number of patterns to generate
            base_patterns: List of base pattern types to use
            
        Returns:
            List of (pattern, markers) tuples
        """
        if base_patterns is None:
            base_patterns = ['gray_code', 'phase_shift', 'checkerboard']
        
        sequence = []
        
        for i in range(num_patterns):
            base_type = base_patterns[i % len(base_patterns)]
            
            # Vary the number of markers
            if i < num_patterns // 3:
                num_markers = 4  # Sparse markers
            elif i < 2 * num_patterns // 3:
                num_markers = 9  # Medium density
            else:
                num_markers = 16  # Dense markers
            
            pattern, markers = self.generate(base_type, num_markers)
            sequence.append((pattern, markers))
        
        return sequence
    
    def generate_calibration_pattern(self) -> Tuple[np.ndarray, List[ArUcoMarkerInfo]]:
        """Generate a special calibration pattern with known marker positions."""
        # Create a white background
        pattern = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # Add a regular grid of ArUco markers
        rows, cols = 4, 6
        x_spacing = self.width / (cols + 1)
        y_spacing = self.height / (rows + 1)
        
        markers_info = []
        marker_id = 0
        
        for row in range(rows):
            for col in range(cols):
                x = int((col + 1) * x_spacing)
                y = int((row + 1) * y_spacing)
                
                # Generate marker
                marker_img = np.zeros((self.marker_size, self.marker_size), 
                                    dtype=np.uint8)
                cv2.aruco.drawMarker(self.aruco_dict, marker_id, 
                                   self.marker_size, marker_img)
                
                # Place marker
                half_size = self.marker_size // 2
                pattern[y-half_size:y+half_size, 
                       x-half_size:x+half_size] = marker_img
                
                # Store info
                corners = np.array([
                    [x-half_size, y-half_size],
                    [x+half_size, y-half_size],
                    [x+half_size, y+half_size],
                    [x-half_size, y+half_size]
                ], dtype=np.float32)
                
                markers_info.append(ArUcoMarkerInfo(
                    id=marker_id,
                    position=(x, y),
                    size=self.marker_size,
                    corners=corners
                ))
                
                marker_id += 1
        
        return pattern, markers_info


class HybridArUcoPatternDecoder:
    """Decode hybrid ArUco patterns from captured images."""
    
    def __init__(self):
        """Initialize hybrid pattern decoder."""
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        logger.info("Initialized HybridArUcoPatternDecoder")
    
    def decode(self,
              captured_image: np.ndarray,
              reference_markers: List[ArUcoMarkerInfo]) -> Dict[str, Any]:
        """
        Decode a captured hybrid pattern.
        
        Args:
            captured_image: Captured pattern image
            reference_markers: Reference marker positions
            
        Returns:
            Dictionary with detected markers and homography
        """
        logger.info("Decoding hybrid ArUco pattern")
        
        # Detect ArUco markers
        detected_markers = self._detect_aruco_markers(captured_image)
        
        # Match with reference markers
        matches = self._match_markers(detected_markers, reference_markers)
        
        # Compute homography if enough matches
        homography = None
        if len(matches) >= 4:
            homography = self._compute_homography(matches)
        
        # Extract structured light pattern
        pattern_data = self._extract_pattern_data(captured_image, 
                                                detected_markers)
        
        return {
            'detected_markers': detected_markers,
            'matches': matches,
            'homography': homography,
            'pattern_data': pattern_data
        }
    
    def _detect_aruco_markers(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect ArUco markers in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        
        detected = []
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                detected.append({
                    'id': marker_id,
                    'corners': corners[i][0],
                    'center': np.mean(corners[i][0], axis=0)
                })
        
        logger.info(f"Detected {len(detected)} ArUco markers")
        return detected
    
    def _match_markers(self,
                      detected: List[Dict[str, Any]],
                      reference: List[ArUcoMarkerInfo]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Match detected markers with reference markers."""
        matches = []
        
        for det in detected:
            for ref in reference:
                if det['id'] == ref.id:
                    matches.append((det['center'], ref.position))
                    break
        
        logger.info(f"Matched {len(matches)} markers")
        return matches
    
    def _compute_homography(self, 
                          matches: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Compute homography from marker matches."""
        src_pts = np.array([m[0] for m in matches], dtype=np.float32)
        dst_pts = np.array([m[1] for m in matches], dtype=np.float32)
        
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        inliers = np.sum(mask) if mask is not None else 0
        logger.info(f"Computed homography with {inliers} inliers")
        
        return homography
    
    def _extract_pattern_data(self,
                            image: np.ndarray,
                            markers: List[Dict[str, Any]]) -> np.ndarray:
        """Extract structured light pattern data, masking out marker regions."""
        pattern = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Mask out marker regions
        for marker in markers:
            # Get bounding box of marker
            corners = marker['corners'].astype(np.int32)
            cv2.fillPoly(pattern, [corners], 128)  # Fill with neutral gray
        
        return pattern
    
    def decode_sequence(self,
                       captured_sequence: List[np.ndarray],
                       reference_sequence: List[Tuple[np.ndarray, List[ArUcoMarkerInfo]]]) -> np.ndarray:
        """
        Decode a sequence of hybrid patterns.
        
        Args:
            captured_sequence: List of captured images
            reference_sequence: List of (pattern, markers) reference tuples
            
        Returns:
            Correspondence map
        """
        height, width = captured_sequence[0].shape[:2]
        correspondence_map = np.zeros((height, width, 2), dtype=np.float32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        
        # Process each pattern
        for i, (captured, (ref_pattern, ref_markers)) in enumerate(
                zip(captured_sequence, reference_sequence)):
            
            # Decode pattern
            result = self.decode(captured, ref_markers)
            
            if result['homography'] is not None:
                # Use homography to establish correspondence
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                pts = np.stack([x.ravel(), y.ravel(), 
                              np.ones_like(x.ravel())], axis=1)
                
                # Apply homography
                H_inv = np.linalg.inv(result['homography'])
                warped_pts = (H_inv @ pts.T).T
                warped_pts = warped_pts[:, :2] / warped_pts[:, 2:3]
                
                # Update correspondence map
                local_corr = warped_pts.reshape(height, width, 2)
                
                # Only update where we have valid correspondences
                mask = ((local_corr[..., 0] >= 0) & 
                       (local_corr[..., 0] < width) &
                       (local_corr[..., 1] >= 0) & 
                       (local_corr[..., 1] < height))
                
                correspondence_map[mask] = local_corr[mask]
                confidence_map[mask] += 1
        
        # Normalize by confidence
        valid_mask = confidence_map > 0
        correspondence_map[valid_mask] /= confidence_map[valid_mask, np.newaxis]
        
        return correspondence_map
    
    def calibrate_from_pattern(self,
                             captured_pattern: np.ndarray,
                             calibration_info: Tuple[np.ndarray, List[ArUcoMarkerInfo]],
                             camera_matrix: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Calibrate camera using the calibration pattern.
        
        Args:
            captured_pattern: Captured calibration pattern
            calibration_info: Reference calibration pattern and markers
            camera_matrix: Optional camera intrinsics
            
        Returns:
            Dictionary with calibration results
        """
        ref_pattern, ref_markers = calibration_info
        
        # Detect markers
        result = self.decode(captured_pattern, ref_markers)
        
        if len(result['matches']) < 4:
            raise ValueError(f"Not enough markers detected: {len(result['matches'])}")
        
        # Prepare object and image points
        obj_points = []
        img_points = []
        
        for det in result['detected_markers']:
            for ref in ref_markers:
                if det['id'] == ref.id:
                    # Convert to 3D object points (assume Z=0)
                    obj_corners = np.array([
                        [ref.corners[0][0], ref.corners[0][1], 0],
                        [ref.corners[1][0], ref.corners[1][1], 0],
                        [ref.corners[2][0], ref.corners[2][1], 0],
                        [ref.corners[3][0], ref.corners[3][1], 0]
                    ], dtype=np.float32)
                    
                    obj_points.append(obj_corners)
                    img_points.append(det['corners'])
                    break
        
        if camera_matrix is None:
            # Estimate camera matrix if not provided
            image_size = captured_pattern.shape[:2][::-1]
            camera_matrix = cv2.initCameraMatrix2D(
                [np.concatenate(obj_points)],
                [np.concatenate(img_points)],
                image_size
            )
        
        # Refine camera parameters
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [np.concatenate(obj_points).reshape(-1, 1, 3)],
            [np.concatenate(img_points).reshape(-1, 1, 2)],
            captured_pattern.shape[:2][::-1],
            camera_matrix,
            None
        )
        
        logger.info(f"Camera calibration complete. RMS error: {retval}")
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'rms_error': retval
        }
    
    def save_marker_data(self,
                        markers: List[ArUcoMarkerInfo],
                        filename: str):
        """Save marker data for later use."""
        data = {
            'markers': [
                {
                    'id': m.id,
                    'position': m.position,
                    'size': m.size,
                    'corners': m.corners.tolist()
                }
                for m in markers
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved marker data to {filename}")
    
    def load_marker_data(self, filename: str) -> List[ArUcoMarkerInfo]:
        """Load marker data from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        markers = []
        for m in data['markers']:
            markers.append(ArUcoMarkerInfo(
                id=m['id'],
                position=tuple(m['position']),
                size=m['size'],
                corners=np.array(m['corners'], dtype=np.float32)
            ))
        
        logger.info(f"Loaded {len(markers)} markers from {filename}")
        return markers


if __name__ == "__main__":
    # Test the hybrid ArUco pattern generator
    generator = HybridArUcoPatternGenerator(1280, 720)
    
    # Generate patterns with different base types
    gray_pattern, gray_markers = generator.generate('gray_code', 9)
    cv2.imwrite('hybrid_gray_code.png', gray_pattern)
    
    phase_pattern, phase_markers = generator.generate('phase_shift', 9)
    cv2.imwrite('hybrid_phase_shift.png', phase_pattern)
    
    checker_pattern, checker_markers = generator.generate('checkerboard', 16)
    cv2.imwrite('hybrid_checkerboard.png', checker_pattern)
    
    # Generate calibration pattern
    calib_pattern, calib_markers = generator.generate_calibration_pattern()
    cv2.imwrite('hybrid_calibration.png', calib_pattern)
    
    # Save marker data
    decoder = HybridArUcoPatternDecoder()
    decoder.save_marker_data(calib_markers, 'calibration_markers.json')
    
    print("Hybrid ArUco patterns generated successfully")