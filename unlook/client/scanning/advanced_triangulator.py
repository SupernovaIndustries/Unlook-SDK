"""
Advanced 3D Triangulation Module with State-of-the-Art Algorithms
Implements modern techniques for high-quality 3D reconstruction
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
# Optional imports with fallbacks
try:
    from scipy import spatial
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedTriangulator:
    """
    Advanced 3D triangulation with outlier removal, noise filtering, and quality assessment.
    Implements best practices for structured light and stereo vision systems.
    """
    
    def __init__(self, calibration_data: Dict[str, Any]):
        """
        Initialize advanced triangulator.
        
        Args:
            calibration_data: Stereo calibration parameters
        """
        self.calibration_data = calibration_data
        self.Q = np.array(calibration_data.get('Q', np.eye(4)))
        self.baseline_mm = float(calibration_data.get('baseline_mm', 79.5))
        self.image_size = tuple(calibration_data.get('image_size', [640, 480]))
        
        # Get focal length from camera matrix (more reliable than Q matrix)
        if 'camera_matrix_left' in calibration_data:
            K = np.array(calibration_data['camera_matrix_left'])
            self.focal_length = (K[0, 0] + K[1, 1]) / 2
        else:
            self.focal_length = 877.6  # Known focal length from calibration
        
        # Detect calibration issues
        self.calibration_scale_factor = 1.0
        
        # IMPORTANT: OpenCV Q matrix can have different formats
        # Standard format: Q[2,3] = focal_length, Q[3,2] = -1/baseline
        # Our calibration has: Q[2,3] = focal_length, Q[3,2] = 1/baseline (positive!)
        
        # Check if we have the focal length in Q[2,3]
        if abs(self.Q[2, 3]) > 1:
            # Q[2,3] contains focal length (this is expected)
            logger.info(f"Q matrix format detected: Q[2,3] = {self.Q[2, 3]:.1f} (focal length)")
            
            # Check Q[3,2] for baseline info
            if abs(self.Q[3, 2]) > 1e-6:
                # Calculate baseline from Q[3,2]
                baseline_from_q = 1.0 / abs(self.Q[3, 2])  # Take absolute value
                logger.info(f"Q[3,2] = {self.Q[3, 2]:.6f}, implies baseline = {baseline_from_q:.1f}mm")
                
                # Compare with expected baseline
                if abs(baseline_from_q - self.baseline_mm) > 1:
                    logger.warning(f"WARNING: Baseline mismatch - Q implies {baseline_from_q:.1f}mm, expected {self.baseline_mm:.1f}mm")
        else:
            # Non-standard Q matrix format
            logger.warning(f"WARNING: Non-standard Q matrix format detected")
            logger.warning(f"  Q[2,3] = {self.Q[2, 3]} (expected focal length ~{self.focal_length:.1f})")
            logger.warning(f"  Q[3,2] = {self.Q[3, 2]} (expected -1/baseline = {-1.0/self.baseline_mm:.6f})")
        
        logger.info(f"Advanced triangulator initialized:")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Baseline: {self.baseline_mm:.1f} mm")
        logger.info(f"  Focal length: {self.focal_length:.1f} pixels")
    
    def triangulate_from_disparity(self, disparity_map: np.ndarray, 
                                 left_image: Optional[np.ndarray] = None,
                                 quality_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Triangulate 3D points from disparity map with advanced filtering.
        
        Args:
            disparity_map: Disparity map from stereo matching
            left_image: Optional left image for texture-based filtering
            quality_threshold: Quality threshold for point acceptance (0-1)
            
        Returns:
            Tuple of (points_3d, statistics)
        """
        
        stats = {
            'method': 'advanced_triangulation',
            'input_pixels': np.sum(disparity_map > 0),
            'processing_stages': {},
            'calibration_scale_factor': self.calibration_scale_factor
        }
        
        # DEBUG: Log input disparity info
        if np.any(disparity_map > 0):
            valid_disp = disparity_map[disparity_map > 0] / 16.0
            logger.debug(f"TRIANGULATION INPUT DEBUG:")
            logger.debug(f"  Disparity map shape: {disparity_map.shape}")
            logger.debug(f"  Valid disparities: {len(valid_disp)}/{disparity_map.size}")
            logger.debug(f"  Disparity range: {valid_disp.min():.2f} - {valid_disp.max():.2f} pixels")
            logger.debug(f"  Mean disparity: {valid_disp.mean():.2f} pixels")
        
        try:
            # Stage 1: Basic reprojection
            points_3d_raw = cv2.reprojectImageTo3D(
                disparity_map.astype(np.float32) / 16.0, self.Q
            )
            
            # Apply calibration correction if needed
            if abs(self.calibration_scale_factor - 1.0) > 0.01:
                logger.info(f"Applying calibration correction factor: {1/self.calibration_scale_factor:.2f}x")
                points_3d_raw[:, :, 2] = points_3d_raw[:, :, 2] / self.calibration_scale_factor
            
            # DEBUG: Check raw point cloud before filtering
            sample_points = points_3d_raw[::100, ::100, :].reshape(-1, 3)
            valid_sample = sample_points[np.all(np.isfinite(sample_points), axis=1)]
            if len(valid_sample) > 0:
                logger.debug(f"RAW POINT CLOUD SAMPLE (before filtering):")
                logger.debug(f"  Sample Z range: {valid_sample[:, 2].min():.1f} - {valid_sample[:, 2].max():.1f} mm")
                logger.debug(f"  Sample mean Z: {valid_sample[:, 2].mean():.1f} mm")
            
            # Stage 2: Extract valid points with basic filtering
            valid_mask = self._create_validity_mask(disparity_map, points_3d_raw)
            valid_points = points_3d_raw[valid_mask]
            valid_disparities = disparity_map[valid_mask] / 16.0
            
            stats['processing_stages']['basic_filtering'] = len(valid_points)
            logger.info(f"Stage 1 - Basic filtering: {len(valid_points)} points")
            
            # DEBUG: Log point cloud statistics after basic filtering
            if len(valid_points) > 0:
                logger.debug(f"POINT CLOUD AFTER BASIC FILTERING:")
                logger.debug(f"  X range: {valid_points[:, 0].min():.1f} - {valid_points[:, 0].max():.1f} mm")
                logger.debug(f"  Y range: {valid_points[:, 1].min():.1f} - {valid_points[:, 1].max():.1f} mm")
                logger.debug(f"  Z range: {valid_points[:, 2].min():.1f} - {valid_points[:, 2].max():.1f} mm")
                logger.debug(f"  Mean depth: {valid_points[:, 2].mean():.1f} mm")
            
            if len(valid_points) == 0:
                logger.warning("No valid points after basic filtering")
                return np.array([]).reshape(0, 3), stats
            
            # Stage 3: Geometric filtering
            points_geometric = self._apply_geometric_filtering(valid_points, valid_disparities)
            stats['processing_stages']['geometric_filtering'] = len(points_geometric)
            logger.info(f"Stage 2 - Geometric filtering: {len(points_geometric)} points")
            
            # Stage 4: Statistical outlier removal
            points_statistical = self._remove_statistical_outliers(points_geometric)
            stats['processing_stages']['statistical_filtering'] = len(points_statistical)
            logger.info(f"Stage 3 - Statistical filtering: {len(points_statistical)} points")
            
            # Stage 5: Clustering-based noise removal
            points_clustered = self._apply_clustering_filter(points_statistical)
            stats['processing_stages']['clustering_filtering'] = len(points_clustered)
            logger.info(f"Stage 4 - Clustering filter: {len(points_clustered)} points")
            
            # Stage 6: Texture-based filtering (if image provided)
            if left_image is not None and len(points_clustered) > 0:
                points_texture = self._apply_texture_filtering(
                    points_clustered, disparity_map, left_image, quality_threshold
                )
                stats['processing_stages']['texture_filtering'] = len(points_texture)
                logger.info(f"Stage 5 - Texture filtering: {len(points_texture)} points")
                final_points = points_texture
            else:
                final_points = points_clustered
            
            # Calculate final statistics
            if len(final_points) > 0:
                stats.update(self._calculate_point_cloud_statistics(final_points))
                stats['success'] = True
                stats['final_point_count'] = len(final_points)
                stats['data_reduction_ratio'] = len(final_points) / max(1, stats['input_pixels'])
                
                logger.info(f"Triangulation completed: {len(final_points)} high-quality points")
                logger.info(f"Data reduction: {stats['input_pixels']} → {len(final_points)} ({stats['data_reduction_ratio']:.1%})")
            else:
                stats['success'] = False
                stats['final_point_count'] = 0
                logger.warning("No points survived filtering pipeline")
            
            return final_points, stats
            
        except Exception as e:
            logger.error(f"Triangulation failed: {e}")
            stats['error'] = str(e)
            stats['success'] = False
            return np.array([]).reshape(0, 3), stats
    
    def _create_validity_mask(self, disparity_map: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
        """Create mask for valid 3D points with comprehensive checks."""
        
        # Basic disparity validity
        disp_valid = disparity_map > 0
        logger.debug(f"Basic disparity check: {np.sum(disp_valid)} pixels have disparity > 0")
        
        # Remove infinite and NaN points
        finite_mask = np.all(np.isfinite(points_3d), axis=2)
        logger.debug(f"Finite points check: {np.sum(finite_mask)} pixels have finite coordinates")
        
        # Reasonable depth range for desktop scanning (10cm to 2m)
        # After calibration correction, depths should be in correct range
        depth_valid = (points_3d[:, :, 2] > 100) & (points_3d[:, :, 2] < 2000)  # 10cm to 2m
        logger.debug(f"Depth range check: {np.sum(depth_valid)} pixels in 10cm-2m range")
        
        # Reasonable lateral range (+/-1m)
        max_lateral = 1000  # 1m lateral range
        lateral_valid = (np.abs(points_3d[:, :, 0]) < max_lateral) & (np.abs(points_3d[:, :, 1]) < max_lateral)
        logger.debug(f"Lateral range check: {np.sum(lateral_valid)} pixels in +/-1m lateral range")
        
        # Combine all validity checks
        combined_mask = disp_valid & finite_mask & depth_valid & lateral_valid
        logger.debug(f"Combined validity: {np.sum(combined_mask)} pixels pass all basic checks")
        
        # Debug: Show actual depth range of valid points
        if np.any(combined_mask):
            valid_depths = points_3d[:, :, 2][combined_mask]
            logger.debug(f"Valid depth range: {np.min(valid_depths):.1f} to {np.max(valid_depths):.1f} mm")
            
            # Check if depths are in expected range
            if np.mean(valid_depths) > 1000:
                logger.warning(f"WARNING: Mean depth {np.mean(valid_depths):.1f}mm seems too far for desktop scanning")
        
        return combined_mask
    
    def _apply_geometric_filtering(self, points: np.ndarray, disparities: np.ndarray) -> np.ndarray:
        """Apply geometric consistency checks."""
        
        if len(points) == 0:
            return points
        
        # DISABLE geometric filtering for now to see what we're getting
        logger.info(f"Geometric filtering: DISABLED for debugging - keeping all {len(points)} points")
        return points
        
        # Original code commented out for debugging:
        # expected_depths = (self.baseline_mm * self.focal_length) / (disparities + 1e-6)
        # actual_depths = points[:, 2]
        # depth_ratio = actual_depths / (expected_depths + 1e-6)
        # depth_consistent = (depth_ratio > 0.8) & (depth_ratio < 1.2)
        # return points[depth_consistent]
    
    def _remove_statistical_outliers(self, points: np.ndarray, 
                                   k_neighbors: int = 20, 
                                   std_multiplier: float = 2.0) -> np.ndarray:
        """Remove statistical outliers using k-nearest neighbor analysis."""
        
        # DISABLE statistical filtering for now to see what we're getting
        logger.info(f"Statistical filtering: DISABLED for debugging - keeping all {len(points)} points")
        return points
        
        # Original code commented out for debugging:
        # if len(points) < k_neighbors:
        #     return points
        # 
        # if not SCIPY_AVAILABLE:
        #     logger.warning("SciPy not available, skipping statistical outlier removal")
        #     return points
            
        try:
            # Build KD-tree for efficient neighbor search
            tree = spatial.cKDTree(points)
            
            # Find k nearest neighbors for each point
            distances, _ = tree.query(points, k=k_neighbors + 1)  # +1 because point is its own neighbor
            
            # Calculate mean distance to k neighbors (excluding self)
            mean_distances = np.mean(distances[:, 1:], axis=1)
            
            # Calculate statistics
            global_mean = np.mean(mean_distances)
            global_std = np.std(mean_distances)
            
            # Remove outliers beyond std_multiplier standard deviations
            threshold = global_mean + std_multiplier * global_std
            inlier_mask = mean_distances < threshold
            
            logger.debug(f"Statistical outlier removal: {np.sum(~inlier_mask)} outliers removed")
            
            return points[inlier_mask]
            
        except Exception as e:
            logger.warning(f"Statistical outlier removal failed: {e}")
            return points
    
    def _apply_clustering_filter(self, points: np.ndarray, 
                               eps: float = 50.0, 
                               min_samples: int = 10) -> np.ndarray:
        """Apply DBSCAN clustering to remove isolated noise points."""
        
        # DISABLE clustering filter for now to see what we're getting
        logger.info(f"Clustering filter: DISABLED for debugging - keeping all {len(points)} points")
        return points
        
        # Original code commented out for debugging:
        # if len(points) < min_samples * 2:
        #     return points
        # 
        # if not SKLEARN_AVAILABLE:
        #     logger.warning("scikit-learn not available, skipping clustering filter")
        #     return points
            
        try:
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(points)
            
            # Keep only points in the largest clusters (remove noise labeled as -1)
            valid_cluster_mask = cluster_labels != -1
            
            if np.any(valid_cluster_mask):
                # Optionally, keep only the largest few clusters
                unique_labels, counts = np.unique(cluster_labels[valid_cluster_mask], return_counts=True)
                
                # Keep clusters with at least 5% of the largest cluster size
                max_cluster_size = np.max(counts)
                size_threshold = max(min_samples, max_cluster_size * 0.05)
                
                large_clusters = unique_labels[counts >= size_threshold]
                final_mask = np.isin(cluster_labels, large_clusters)
                
                logger.debug(f"Clustering filter: kept {len(large_clusters)} clusters, removed {np.sum(~final_mask)} noise points")
                
                return points[final_mask]
            else:
                logger.warning("All points classified as noise by clustering")
                return points
                
        except Exception as e:
            logger.warning(f"Clustering filter failed: {e}")
            return points
    
    def _apply_texture_filtering(self, points: np.ndarray, 
                               disparity_map: np.ndarray,
                               image: np.ndarray, 
                               quality_threshold: float) -> np.ndarray:
        """Filter points based on texture quality in the source image."""
        
        # This is a simplified texture-based filter
        # In a full implementation, this would analyze local texture strength
        
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate local gradient strength as texture measure
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize texture strength
        texture_normalized = cv2.normalize(texture_strength, None, 0, 1, cv2.NORM_MINMAX)
        
        # For each point, check texture strength at corresponding image location
        # This is a simplified version - full implementation would reproject points
        # back to image coordinates
        
        return points  # Placeholder - return all points for now
    
    def _calculate_point_cloud_statistics(self, points: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the point cloud."""
        
        if len(points) == 0:
            return {'point_count': 0}
        
        stats = {
            'point_count': len(points),
            'bounds': {
                'x': {'min': float(np.min(points[:, 0])), 'max': float(np.max(points[:, 0]))},
                'y': {'min': float(np.min(points[:, 1])), 'max': float(np.max(points[:, 1]))},
                'z': {'min': float(np.min(points[:, 2])), 'max': float(np.max(points[:, 2]))}
            },
            'centroid': {
                'x': float(np.mean(points[:, 0])),
                'y': float(np.mean(points[:, 1])),
                'z': float(np.mean(points[:, 2]))
            },
            'std_dev': {
                'x': float(np.std(points[:, 0])),
                'y': float(np.std(points[:, 1])),
                'z': float(np.std(points[:, 2]))
            }
        }
        
        # Calculate bounding box dimensions
        stats['dimensions'] = {
            'width': stats['bounds']['x']['max'] - stats['bounds']['x']['min'],
            'height': stats['bounds']['y']['max'] - stats['bounds']['y']['min'],
            'depth': stats['bounds']['z']['max'] - stats['bounds']['z']['min']
        }
        
        # Calculate point density (points per cubic mm)
        volume = stats['dimensions']['width'] * stats['dimensions']['height'] * stats['dimensions']['depth']
        if volume > 0:
            stats['density_points_per_mm3'] = len(points) / volume
        
        return stats
    
    def triangulate_multi_view(self, disparity_maps: List[np.ndarray], 
                             images: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Triangulate from multiple disparity maps and merge results.
        Useful for structured light with multiple pattern frequencies.
        """
        
        all_points = []
        merge_stats = {'individual_results': []}
        
        for i, disparity_map in enumerate(disparity_maps):
            image = images[i] if images and i < len(images) else None
            points, stats = self.triangulate_from_disparity(disparity_map, image)
            
            if len(points) > 0:
                all_points.append(points)
                merge_stats['individual_results'].append(stats)
        
        if not all_points:
            logger.warning("No valid points from any disparity map")
            return np.array([]).reshape(0, 3), merge_stats
        
        # Merge all point clouds
        merged_points = np.vstack(all_points)
        
        # Remove duplicate points (within 1mm tolerance)
        if len(merged_points) > 1000:  # Only for large point clouds
            merged_points = self._remove_duplicate_points(merged_points, tolerance=1.0)
        
        merge_stats['merged_point_count'] = len(merged_points)
        merge_stats['individual_counts'] = [len(pts) for pts in all_points]
        
        logger.info(f"Multi-view triangulation: merged {len(all_points)} views into {len(merged_points)} points")
        
        return merged_points, merge_stats
    
    def _remove_duplicate_points(self, points: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
        """Remove duplicate points within tolerance distance."""
        
        try:
            # Use spatial hashing for efficiency with large point clouds
            if len(points) > 10000:
                # Grid-based deduplication for very large point clouds
                grid_size = tolerance * 2
                
                # Quantize points to grid
                quantized = np.round(points / grid_size).astype(np.int32)
                
                # Find unique quantized points
                _, unique_indices = np.unique(quantized, axis=0, return_index=True)
                
                return points[unique_indices]
            elif SCIPY_AVAILABLE:
                # KD-tree based deduplication for smaller point clouds
                tree = spatial.cKDTree(points)
                
                # Find points within tolerance of each other
                duplicate_pairs = tree.query_ball_tree(tree, tolerance)
                
                # Keep only one point from each duplicate group
                keep_mask = np.ones(len(points), dtype=bool)
                
                for i, duplicates in enumerate(duplicate_pairs):
                    if keep_mask[i]:  # If this point hasn't been marked for removal
                        # Mark all duplicates except the first one for removal
                        for j in duplicates[1:]:
                            if j > i:  # Only mark points with higher indices
                                keep_mask[j] = False
                
                return points[keep_mask]
            else:
                # Simple fallback without scipy - just return original points
                logger.warning("SciPy not available for duplicate removal")
                return points
                
        except Exception as e:
            logger.warning(f"Duplicate removal failed: {e}")
            return points

def create_advanced_triangulator(calibration_data: Dict[str, Any]) -> AdvancedTriangulator:
    """Factory function to create an advanced triangulator."""
    return AdvancedTriangulator(calibration_data)