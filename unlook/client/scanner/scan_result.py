"""
Scan result data structure for 3D scanning.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import logging
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


class ScanResult:
    """Result of a 3D scan."""
    
    def __init__(
        self,
        point_cloud: Optional[Union[o3dg.PointCloud, np.ndarray]] = None,
        mesh: Optional[o3dg.TriangleMesh] = None,
        images: Optional[Dict[str, List[np.ndarray]]] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize scan result.
        
        Args:
            point_cloud: 3D point cloud (Open3D or numpy array)
            mesh: 3D mesh
            images: Dictionary of captured images
            debug_info: Debug information
        """
        self.point_cloud = point_cloud
        self.mesh = mesh
        self.images = images or {}
        self.debug_info = debug_info or {}
    
    def has_point_cloud(self) -> bool:
        """Check if result contains a point cloud."""
        return self.point_cloud is not None and (
            (isinstance(self.point_cloud, np.ndarray) and len(self.point_cloud) > 0) or
            (OPEN3D_AVAILABLE and hasattr(self.point_cloud, 'points') and 
             len(self.point_cloud.points) > 0)
        )
    
    def has_mesh(self) -> bool:
        """Check if result contains a mesh."""
        return self.mesh is not None and (
            OPEN3D_AVAILABLE and hasattr(self.mesh, 'vertices') and 
            len(self.mesh.vertices) > 0
        )
    
    def save(self, output_dir: str):
        """
        Save scan result to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save point cloud
        if self.has_point_cloud():
            if OPEN3D_AVAILABLE and hasattr(self.point_cloud, 'points'):
                pc_file = output_path / "point_cloud.ply"
                o3d.io.write_point_cloud(str(pc_file), self.point_cloud)
                logger.info(f"Saved point cloud to {pc_file}")
            elif isinstance(self.point_cloud, np.ndarray):
                pc_file = output_path / "point_cloud.xyz"
                np.savetxt(pc_file, self.point_cloud, fmt='%.6f')
                logger.info(f"Saved point cloud to {pc_file}")
        
        # Save mesh
        if self.has_mesh() and OPEN3D_AVAILABLE:
            mesh_file = output_path / "mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_file), self.mesh)
            logger.info(f"Saved mesh to {mesh_file}")
        
        # Save metadata
        metadata = {
            "has_point_cloud": self.has_point_cloud(),
            "has_mesh": self.has_mesh(),
            "num_points": (
                len(self.point_cloud) if isinstance(self.point_cloud, np.ndarray)
                else len(self.point_cloud.points) if self.has_point_cloud() and OPEN3D_AVAILABLE
                else 0
            ),
            "num_triangles": (
                len(self.mesh.triangles) if self.has_mesh() and OPEN3D_AVAILABLE
                else 0
            ),
            "image_categories": list(self.images.keys()),
            "debug_info": self.debug_info
        }
        
        metadata_file = output_path / "scan_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")