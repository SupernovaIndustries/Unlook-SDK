"""
Utility functions for structured light scanning.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Point cloud visualization will be limited.")
    OPEN3D_AVAILABLE = False


def apply_adaptive_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply contrast-limited adaptive histogram equalization (CLAHE) to an image.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compute_shadow_mask(black_img: np.ndarray, white_img: np.ndarray, threshold: int = 40) -> np.ndarray:
    """
    Compute shadow mask from white and black reference images.
    
    Args:
        black_img: Image captured with black pattern
        white_img: Image captured with white pattern
        threshold: Threshold for shadow detection
        
    Returns:
        Binary mask (1 for valid pixels, 0 for shadowed)
    """
    # Convert to grayscale if needed
    if len(black_img.shape) == 3:
        black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
    if len(white_img.shape) == 3:
        white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
    
    # Compute difference
    diff = cv2.absdiff(white_img, black_img)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Apply adaptive thresholding
    try:
        # Normalize for improved contrast
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply adaptive threshold
        mask = cv2.adaptiveThreshold(
            normalized,
            1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -5
        )
    except Exception as e:
        logger.warning(f"Adaptive thresholding failed, using simple threshold: {e}")
        # Fall back to simple thresholding
        _, mask = cv2.threshold(blurred, threshold, 1, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def rectify_stereo_images(
        left_img: np.ndarray, 
        right_img: np.ndarray, 
        map_left_x: np.ndarray, 
        map_left_y: np.ndarray, 
        map_right_x: np.ndarray, 
        map_right_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify stereo images.
    
    Args:
        left_img: Left camera image
        right_img: Right camera image
        map_left_x: X-coordinate map for left camera undistortion
        map_left_y: Y-coordinate map for left camera undistortion
        map_right_x: X-coordinate map for right camera undistortion
        map_right_y: Y-coordinate map for right camera undistortion
        
    Returns:
        Tuple of (rectified_left, rectified_right)
    """
    # Rectify left image
    left_rect = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
    
    # Rectify right image
    right_rect = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    return left_rect, right_rect


def filter_point_cloud(points: np.ndarray, max_distance: float = 5000.0) -> np.ndarray:
    """
    Filter point cloud to remove outliers.
    
    Args:
        points: 3D points
        max_distance: Maximum distance from origin
        
    Returns:
        Filtered points
    """
    # Remove NaN and infinite values
    valid_mask = np.all(~np.isnan(points), axis=1) & np.all(~np.isinf(points), axis=1)
    points = points[valid_mask]
    
    if len(points) == 0:
        return points
    
    # Filter by distance
    distances = np.linalg.norm(points, axis=1)
    distance_mask = distances < max_distance
    points = points[distance_mask]
    
    # Filter using Open3D if available
    if OPEN3D_AVAILABLE and len(points) > 30:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Statistical outlier removal
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Radius outlier removal
            pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=1.0)
            
            points = np.asarray(pcd.points)
        except Exception as e:
            logger.warning(f"Open3D filtering failed: {e}")
    
    return points


def visualize_point_cloud(points: np.ndarray) -> None:
    """
    Visualize point cloud using Open3D.
    
    Args:
        points: 3D points
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D not available, cannot visualize")
        return
    
    # Remove NaN and infinite values
    valid_mask = np.all(~np.isnan(points), axis=1) & np.all(~np.isinf(points), axis=1)
    points = points[valid_mask]
    
    if len(points) == 0:
        logger.error("No valid points to visualize")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])


def create_colored_point_cloud(points: np.ndarray, image: np.ndarray, points_2d: np.ndarray) -> Union[np.ndarray, Any]:
    """
    Create colored point cloud using image for texture.
    
    Args:
        points: 3D points
        image: Image for texture
        points_2d: 2D points in image coordinates
        
    Returns:
        Colored point cloud (Open3D PointCloud if available, otherwise numpy array with colors)
    """
    if len(points) == 0 or len(points_2d) == 0:
        return points
    
    # Ensure points and points_2d have the same length
    min_len = min(len(points), len(points_2d))
    points = points[:min_len]
    points_2d = points_2d[:min_len]
    
    # Convert image to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Get colors from image
    colors = []
    for point_2d in points_2d:
        x, y = int(point_2d[0]), int(point_2d[1])
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            color = image[y, x] / 255.0
            colors.append(color[::-1])  # BGR to RGB
        else:
            colors.append([0.7, 0.7, 0.7])  # Default gray
    
    if OPEN3D_AVAILABLE:
        # Create colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    else:
        # Return points with colors
        return np.hstack((points, np.array(colors)))


def save_point_cloud(points: np.ndarray, filepath: str, colors: Optional[np.ndarray] = None) -> None:
    """
    Save point cloud to PLY file.
    
    Args:
        points: 3D points
        filepath: Output path
        colors: Optional point colors
    """
    if OPEN3D_AVAILABLE:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        # Save as numpy file
        if colors is not None and len(colors) == len(points):
            np.save(filepath, np.hstack((points, colors)))
        else:
            np.save(filepath, points)


def create_mesh_from_point_cloud(
        points: np.ndarray, 
        colors: Optional[np.ndarray] = None, 
        depth: int = 9, 
        smoothing: int = 5) -> Any:
    """
    Create mesh from point cloud using Poisson reconstruction.
    
    Args:
        points: 3D points
        colors: Optional point colors
        depth: Depth parameter for Poisson reconstruction
        smoothing: Number of smoothing iterations
        
    Returns:
        Mesh (Open3D TriangleMesh if available, otherwise None)
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D not available, cannot create mesh")
        return None
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Create mesh using Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=True
    )
    
    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Apply smoothing
    if smoothing > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing)
    
    return mesh


def save_mesh(mesh: Any, filepath: str) -> None:
    """
    Save mesh to file.
    
    Args:
        mesh: Mesh (Open3D TriangleMesh)
        filepath: Output path
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D not available, cannot save mesh")
        return
    
    # Save mesh
    o3d.io.write_triangle_mesh(filepath, mesh)


def determine_projector_resolution() -> Tuple[int, int]:
    """
    Determine projector resolution based on commonly used resolutions.
    
    Returns:
        Tuple of (width, height)
    """
    # Common projector resolutions
    resolutions = [
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (1024, 768),   # XGA
        (800, 600),    # SVGA
        (640, 480)     # VGA
    ]
    
    return resolutions[0]  # Default to Full HD


def create_timestamp_folder(base_dir: str, prefix: str = "scan") -> str:
    """
    Create a timestamped folder for scan results.
    
    Args:
        base_dir: Base directory
        prefix: Folder name prefix
        
    Returns:
        Path to created folder
    """
    # Create timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    
    # Create folder
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path