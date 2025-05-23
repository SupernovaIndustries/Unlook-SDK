"""
Point cloud processing utilities for 3D scanning.
"""

import logging
from typing import Union, Optional, Tuple
import numpy as np

# Try to import optional dependencies
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    # Create placeholder for open3d when not available
    class PlaceholderO3D:
        class geometry:
            class PointCloud:
                pass
            class TriangleMesh:
                pass
    o3d = PlaceholderO3D()
    o3dg = PlaceholderO3D.geometry

logger = logging.getLogger(__name__)


class PointCloudProcessor:
    """Handles point cloud filtering and mesh generation."""
    
    @staticmethod
    def filter_point_cloud(
        points_3d: np.ndarray,
        max_distance: float = 1000.0,
        voxel_size: float = 0.5,
        remove_outliers: bool = True,
        outlier_neighbors: int = 20,
        outlier_std_ratio: float = 2.0
    ) -> Union[o3dg.PointCloud, np.ndarray]:
        """
        Filter and clean 3D point cloud.
        
        Args:
            points_3d: 3D points (Nx3)
            max_distance: Maximum distance from origin
            voxel_size: Voxel size for downsampling (mm)
            remove_outliers: Whether to remove statistical outliers
            outlier_neighbors: Number of neighbors for outlier detection
            outlier_std_ratio: Standard deviation ratio for outlier detection
            
        Returns:
            Filtered point cloud
        """
        if len(points_3d) == 0:
            logger.warning("Empty point cloud, nothing to filter")
            return points_3d
            
        # Filter by distance
        distances = np.linalg.norm(points_3d, axis=1)
        mask = distances < max_distance
        filtered_points = points_3d[mask]
        
        if len(filtered_points) == 0:
            logger.warning("All points filtered out by distance threshold")
            return filtered_points
        
        logger.info(f"Filtered {len(points_3d) - len(filtered_points)} points by distance")
        
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available, returning numpy array without advanced filtering")
            return filtered_points
        
        # Convert to Open3D point cloud
        pcd = o3dg.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Voxel downsampling
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
            logger.info(f"Downsampled to {len(pcd.points)} points with voxel size {voxel_size}")
        
        # Remove statistical outliers
        if remove_outliers and len(pcd.points) > outlier_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=outlier_neighbors,
                std_ratio=outlier_std_ratio
            )
            logger.info(f"Removed outliers, {len(pcd.points)} points remaining")
        
        return pcd
    
    @staticmethod
    def create_mesh(
        point_cloud: Union[o3dg.PointCloud, np.ndarray],
        method: str = "poisson",
        depth: int = 9,
        scale: float = 1.1,
        linear_fit: bool = False,
        n_threads: int = -1,
        remove_degenerate: bool = True,
        remove_duplicated: bool = True
    ) -> Optional[o3dg.TriangleMesh]:
        """
        Create mesh from point cloud using Poisson reconstruction.
        
        Args:
            point_cloud: Input point cloud
            method: Reconstruction method ('poisson' or 'ball_pivoting')
            depth: Octree depth for Poisson reconstruction
            scale: Scale factor for Poisson reconstruction
            linear_fit: Use linear interpolation for Poisson
            n_threads: Number of threads (-1 for all)
            remove_degenerate: Remove degenerate triangles
            remove_duplicated: Remove duplicated triangles/vertices
            
        Returns:
            Generated mesh or None if failed
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for mesh generation")
            return None
        
        # Convert numpy array to Open3D point cloud if needed
        if isinstance(point_cloud, np.ndarray):
            pcd = o3dg.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            pcd = point_cloud
        
        if len(pcd.points) < 100:
            logger.error("Not enough points for mesh generation")
            return None
        
        # Estimate normals if not present
        if not pcd.has_normals():
            logger.info("Estimating normals...")
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(30)
        
        mesh = None
        
        if method == "poisson":
            logger.info(f"Creating mesh using Poisson reconstruction (depth={depth})...")
            try:
                mesh, _ = o3dg.TriangleMesh.create_from_point_cloud_poisson(
                    pcd,
                    depth=depth,
                    scale=scale,
                    linear_fit=linear_fit,
                    n_threads=n_threads
                )
            except Exception as e:
                logger.error(f"Poisson reconstruction failed: {e}")
                return None
                
        elif method == "ball_pivoting":
            logger.info("Creating mesh using Ball Pivoting Algorithm...")
            try:
                # Estimate radius for ball pivoting
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = avg_dist * 2
                
                # Create mesh using ball pivoting
                radii = [radius, radius * 2, radius * 4]
                mesh = o3dg.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
            except Exception as e:
                logger.error(f"Ball pivoting failed: {e}")
                return None
        else:
            logger.error(f"Unknown reconstruction method: {method}")
            return None
        
        if mesh is None:
            return None
        
        # Post-process mesh
        if remove_degenerate:
            mesh.remove_degenerate_triangles()
        if remove_duplicated:
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
        
        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        logger.info(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        return mesh
    
    @staticmethod
    def estimate_normals(
        point_cloud: Union[o3dg.PointCloud, np.ndarray],
        search_radius: Optional[float] = None,
        max_nn: int = 30
    ) -> o3dg.PointCloud:
        """
        Estimate normals for point cloud.
        
        Args:
            point_cloud: Input point cloud
            search_radius: Search radius for normal estimation
            max_nn: Maximum number of nearest neighbors
            
        Returns:
            Point cloud with normals
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D is required for normal estimation")
            return point_cloud
            
        # Convert numpy array to Open3D point cloud if needed
        if isinstance(point_cloud, np.ndarray):
            pcd = o3dg.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            pcd = point_cloud
            
        if search_radius is None:
            # Estimate search radius from average point distance
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=max_nn
            ))
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=max_nn
            ))
            
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(max_nn)
        
        return pcd