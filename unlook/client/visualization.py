"""
Visualization utilities for 3D scanning results.

This module provides helper functions for visualizing and analyzing 3D scan results,
including point clouds, meshes, and debugging visualizations.
"""

import os
import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
import matplotlib.pyplot as plt

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Visualization capabilities will be limited.")
    OPEN3D_AVAILABLE = False


class ScanVisualizer:
    """
    Utility class for visualizing 3D scanning results and debugging information.
    """
    
    def __init__(self, use_window: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            use_window: Whether to use interactive window (True) or offscreen rendering (False)
        """
        self.use_window = use_window
        
        if not OPEN3D_AVAILABLE:
            logger.error("open3d is required for ScanVisualizer")
            raise ImportError("open3d is required for ScanVisualizer")
        
        # Check if we have a GUI
        # Different versions of Open3D have different ways to detect GUI capabilities
        try:
            # Try several methods to detect GUI capability
            if hasattr(o3d.visualization, 'GLFW_KEY_SPACE'):
                self.has_gui = o3d.visualization.GLFW_KEY_SPACE > 0
            elif hasattr(o3d.visualization, 'gui') and hasattr(o3d.visualization.gui, 'Application'):
                # Newer versions use the gui module
                self.has_gui = True
            else:
                # Fall back to assuming GUI is available
                self.has_gui = True
        except Exception:
            # If any error occurs, assume no GUI
            logger.warning("Could not determine GUI capabilities, assuming GUI is available")
            self.has_gui = True
        
        if use_window and not self.has_gui:
            logger.warning("No GUI available. Falling back to offscreen rendering.")
            self.use_window = False
    
    @staticmethod
    def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from file.
        
        Args:
            filepath: Path to the point cloud file (ply, pcd, xyz, etc.)
            
        Returns:
            Loaded point cloud
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return o3d.geometry.PointCloud()
        
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            logger.info(f"Loaded point cloud with {len(pcd.points)} points from {filepath}")
            return pcd
        except Exception as e:
            logger.error(f"Error loading point cloud: {e}")
            return o3d.geometry.PointCloud()
    
    @staticmethod
    def load_mesh(filepath: str) -> o3d.geometry.TriangleMesh:
        """
        Load a mesh from file.
        
        Args:
            filepath: Path to the mesh file (ply, obj, stl, etc.)
            
        Returns:
            Loaded mesh
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return o3d.geometry.TriangleMesh()
        
        try:
            mesh = o3d.io.read_triangle_mesh(filepath)
            logger.info(f"Loaded mesh with {len(mesh.triangles)} triangles from {filepath}")
            return mesh
        except Exception as e:
            logger.error(f"Error loading mesh: {e}")
            return o3d.geometry.TriangleMesh()
    
    def visualize_point_cloud(self, pcd: Union[o3d.geometry.PointCloud, str], 
                             screenshot_path: Optional[str] = None) -> None:
        """
        Visualize a point cloud.
        
        Args:
            pcd: Point cloud object or path to point cloud file
            screenshot_path: Optional path to save a screenshot
        """
        # Load point cloud if a string path is provided
        if isinstance(pcd, str):
            pcd = self.load_point_cloud(pcd)
        
        if len(pcd.points) == 0:
            logger.warning("Empty point cloud, nothing to visualize")
            return
        
        # Add normals if not present (for better visualization)
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        
        # Use interactive visualization if window mode and GUI available
        try:
            if self.use_window and self.has_gui:
                try:
                    # Try to use the Visualizer class
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Point Cloud Visualization", width=800, height=600)
                    vis.add_geometry(pcd)

                    # Add coordinate frame for scale reference
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                    vis.add_geometry(coordinate_frame)

                    # Better default view
                    opt = vis.get_render_option()
                    opt.point_size = 2.0
                    opt.background_color = np.asarray([0.1, 0.1, 0.1])

                    # Update view
                    vis.update_renderer()
                    vis.poll_events()
                    vis.run()

                    # Save screenshot if requested
                    if screenshot_path:
                        vis.capture_screen_image(screenshot_path, True)
                        logger.info(f"Saved screenshot to {screenshot_path}")

                    vis.destroy_window()
                    return
                except Exception as e:
                    logger.warning(f"Visualizer class failed: {e}, trying draw_geometries instead")
                    # Fall back to simpler method
                    if hasattr(o3d.visualization, 'draw_geometries'):
                        geometries = [pcd]
                        if hasattr(o3d.geometry, 'TriangleMesh') and hasattr(o3d.geometry.TriangleMesh, 'create_coordinate_frame'):
                            geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10))
                        o3d.visualization.draw_geometries(geometries, window_name="Point Cloud Visualization")
                        logger.info("Used draw_geometries for visualization")
                        return

            # Try non-interactive rendering for screenshot
            if screenshot_path:
                try:
                    # Try to use newer rendering API
                    if hasattr(o3d.visualization, 'rendering') and hasattr(o3d.visualization.rendering, 'OffscreenRenderer'):
                        render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
                        render.scene.add_geometry("point_cloud", pcd)
                        try:
                            render.scene.add_geometry("coord_frame",
                                                   o3d.geometry.TriangleMesh.create_coordinate_frame(size=10))
                        except Exception:
                            pass  # Skip coordinate frame if it fails
                        render.scene.camera.look_at([0, 0, 0], [10, 10, 10], [0, 1, 0])
                        img = render.render_to_image()
                        o3d.io.write_image(screenshot_path, img)
                        logger.info(f"Saved screenshot to {screenshot_path}")
                        return
                    else:
                        # Older versions might not have OffscreenRenderer
                        raise NotImplementedError("OffscreenRenderer not available")

                except Exception as e:
                    logger.warning(f"OffscreenRenderer failed: {e}, trying simpler visualization")

                    # Last resort: Try using a temporary Visualizer
                    try:
                        vis = o3d.visualization.Visualizer()
                        vis.create_window(visible=False)
                        vis.add_geometry(pcd)
                        vis.update_geometry(pcd)
                        vis.poll_events()
                        vis.update_renderer()
                        vis.capture_screen_image(screenshot_path)
                        vis.destroy_window()
                        logger.info(f"Saved screenshot to {screenshot_path}")
                        return
                    except Exception as e2:
                        logger.error(f"All visualization methods failed: {e2}")
                        logger.error("Unable to visualize or save screenshot")
            else:
                logger.warning("Non-interactive mode requires a screenshot_path")
        except Exception as general_error:
            logger.error(f"Visualization failed: {general_error}")
            logger.error("Check your Open3D installation and version compatibility")
    
    def visualize_mesh(self, mesh: Union[o3d.geometry.TriangleMesh, str],
                      screenshot_path: Optional[str] = None,
                      show_wireframe: bool = False) -> None:
        """
        Visualize a mesh.
        
        Args:
            mesh: Mesh object or path to mesh file
            screenshot_path: Optional path to save a screenshot
            show_wireframe: Whether to show wireframe
        """
        # Load mesh if a string path is provided
        if isinstance(mesh, str):
            mesh = self.load_mesh(mesh)
        
        if len(mesh.triangles) == 0:
            logger.warning("Empty mesh, nothing to visualize")
            return
        
        # Use interactive visualization if window mode and GUI available
        try:
            if self.use_window and self.has_gui:
                try:
                    # Try to use the Visualizer class
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Mesh Visualization", width=800, height=600)
                    vis.add_geometry(mesh)

                    # Add coordinate frame for scale reference
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                    vis.add_geometry(coordinate_frame)

                    # Better default view
                    opt = vis.get_render_option()
                    if show_wireframe:
                        opt.mesh_show_wireframe = True
                    opt.mesh_show_back_face = True
                    opt.background_color = np.asarray([0.1, 0.1, 0.1])

                    # Update view
                    vis.update_renderer()
                    vis.poll_events()
                    vis.run()

                    # Save screenshot if requested
                    if screenshot_path:
                        vis.capture_screen_image(screenshot_path, True)
                        logger.info(f"Saved screenshot to {screenshot_path}")

                    vis.destroy_window()
                    return
                except Exception as e:
                    logger.warning(f"Visualizer class failed: {e}, trying draw_geometries instead")
                    # Fall back to simpler method
                    if hasattr(o3d.visualization, 'draw_geometries'):
                        geometries = [mesh]
                        if hasattr(o3d.geometry, 'TriangleMesh') and hasattr(o3d.geometry.TriangleMesh, 'create_coordinate_frame'):
                            geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10))
                        o3d.visualization.draw_geometries(geometries, window_name="Mesh Visualization")
                        logger.info("Used draw_geometries for visualization")
                        return

            # Try non-interactive rendering for screenshot
            if screenshot_path:
                try:
                    # Try to use newer rendering API
                    if hasattr(o3d.visualization, 'rendering') and hasattr(o3d.visualization.rendering, 'OffscreenRenderer'):
                        render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
                        render.scene.add_geometry("mesh", mesh)
                        try:
                            render.scene.add_geometry("coord_frame",
                                                    o3d.geometry.TriangleMesh.create_coordinate_frame(size=10))
                        except Exception:
                            pass  # Skip coordinate frame if it fails
                        render.scene.camera.look_at([0, 0, 0], [10, 10, 10], [0, 1, 0])
                        img = render.render_to_image()
                        o3d.io.write_image(screenshot_path, img)
                        logger.info(f"Saved screenshot to {screenshot_path}")
                        return
                    else:
                        # Older versions might not have OffscreenRenderer
                        raise NotImplementedError("OffscreenRenderer not available")

                except Exception as e:
                    logger.warning(f"OffscreenRenderer failed: {e}, trying simpler visualization")

                    # Last resort: Try using a temporary Visualizer
                    try:
                        vis = o3d.visualization.Visualizer()
                        vis.create_window(visible=False)
                        vis.add_geometry(mesh)
                        vis.update_geometry(mesh)
                        vis.poll_events()
                        vis.update_renderer()
                        vis.capture_screen_image(screenshot_path)
                        vis.destroy_window()
                        logger.info(f"Saved screenshot to {screenshot_path}")
                        return
                    except Exception as e2:
                        logger.error(f"All visualization methods failed: {e2}")
                        logger.error("Unable to visualize or save screenshot")
            else:
                logger.warning("Non-interactive mode requires a screenshot_path")
        except Exception as general_error:
            logger.error(f"Visualization failed: {general_error}")
            logger.error("Check your Open3D installation and version compatibility")
    
    def compare_point_clouds(self, pcd1: Union[o3d.geometry.PointCloud, str],
                           pcd2: Union[o3d.geometry.PointCloud, str],
                           labels: Tuple[str, str] = ("Point Cloud 1", "Point Cloud 2"),
                           screenshot_path: Optional[str] = None) -> None:
        """
        Compare two point clouds side by side.
        
        Args:
            pcd1: First point cloud or path
            pcd2: Second point cloud or path
            labels: Labels for the point clouds
            screenshot_path: Optional path to save a screenshot
        """
        # Load point clouds if string paths are provided
        if isinstance(pcd1, str):
            pcd1 = self.load_point_cloud(pcd1)
        if isinstance(pcd2, str):
            pcd2 = self.load_point_cloud(pcd2)
        
        # Clone to avoid modifying originals
        pcd1_vis = o3d.geometry.PointCloud(pcd1)
        pcd2_vis = o3d.geometry.PointCloud(pcd2)
        
        # Add normals if not present
        if not pcd1_vis.has_normals():
            pcd1_vis.estimate_normals()
        if not pcd2_vis.has_normals():
            pcd2_vis.estimate_normals()
        
        # Color the point clouds differently
        pcd1_vis.paint_uniform_color([1, 0.7, 0])  # Orange
        pcd2_vis.paint_uniform_color([0, 0.7, 1])  # Blue
        
        try:
            # Use interactive visualization if window mode and GUI available
            if self.use_window and self.has_gui:
                try:
                    # Create two side-by-side windows
                    vis1 = o3d.visualization.Visualizer()
                    vis1.create_window(window_name=labels[0], width=600, height=600, left=50, top=50)
                    vis1.add_geometry(pcd1_vis)

                    vis2 = o3d.visualization.Visualizer()
                    vis2.create_window(window_name=labels[1], width=600, height=600, left=650, top=50)
                    vis2.add_geometry(pcd2_vis)

                    # Add coordinate frames
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                    vis1.add_geometry(coordinate_frame)
                    vis2.add_geometry(coordinate_frame)

                    # Set rendering options
                    for vis in [vis1, vis2]:
                        opt = vis.get_render_option()
                        opt.point_size = 2.0
                        opt.background_color = np.asarray([0.1, 0.1, 0.1])
                        vis.update_renderer()
                        vis.poll_events()

                    # Run both visualizers
                    while True:
                        vis1.update_renderer()
                        vis1.poll_events()
                        vis2.update_renderer()
                        vis2.poll_events()

                        if not vis1.poll_events() or not vis2.poll_events():
                            break

                    # Save screenshots if requested
                    if screenshot_path:
                        base_path, ext = os.path.splitext(screenshot_path)
                        vis1.capture_screen_image(f"{base_path}_1{ext}", True)
                        vis2.capture_screen_image(f"{base_path}_2{ext}", True)
                        logger.info(f"Saved screenshots to {base_path}_1{ext} and {base_path}_2{ext}")

                    vis1.destroy_window()
                    vis2.destroy_window()
                    return
                except Exception as e:
                    logger.warning(f"Dual visualizer failed: {e}, trying simpler method")

                    # Fall back to simpler visualization using draw_geometries
                    if hasattr(o3d.visualization, 'draw_geometries'):
                        # Show them one after another
                        logger.info(f"Showing {labels[0]}...")
                        o3d.visualization.draw_geometries([pcd1_vis], window_name=labels[0])
                        logger.info(f"Showing {labels[1]}...")
                        o3d.visualization.draw_geometries([pcd2_vis], window_name=labels[1])
                        return

            # Try non-interactive rendering for screenshots
            if screenshot_path:
                try:
                    base_path, ext = os.path.splitext(screenshot_path)

                    # Try newer rendering API
                    if hasattr(o3d.visualization, 'rendering') and hasattr(o3d.visualization.rendering, 'OffscreenRenderer'):
                        # Create renderers for both point clouds
                        for i, (pcd, label) in enumerate(zip([pcd1_vis, pcd2_vis], labels)):
                            render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
                            render.scene.add_geometry("point_cloud", pcd)
                            try:
                                render.scene.add_geometry("coord_frame",
                                                        o3d.geometry.TriangleMesh.create_coordinate_frame(size=10))
                            except Exception:
                                pass  # Skip coordinate frame if it fails

                            render.scene.camera.look_at([0, 0, 0], [10, 10, 10], [0, 1, 0])
                            img = render.render_to_image()
                            o3d.io.write_image(f"{base_path}_{i+1}{ext}", img)

                        logger.info(f"Saved screenshots to {base_path}_1{ext} and {base_path}_2{ext}")
                        return
                    else:
                        # Older versions might not have OffscreenRenderer
                        raise NotImplementedError("OffscreenRenderer not available")

                except Exception as e:
                    logger.warning(f"OffscreenRenderer failed: {e}, trying simpler visualization")

                    # Last resort: Try using temporary Visualizers
                    try:
                        base_path, ext = os.path.splitext(screenshot_path)

                        for i, (pcd, label) in enumerate(zip([pcd1_vis, pcd2_vis], labels)):
                            vis = o3d.visualization.Visualizer()
                            vis.create_window(visible=False)
                            vis.add_geometry(pcd)
                            vis.update_geometry(pcd)
                            vis.poll_events()
                            vis.update_renderer()
                            vis.capture_screen_image(f"{base_path}_{i+1}{ext}")
                            vis.destroy_window()

                        logger.info(f"Saved screenshots to {base_path}_1{ext} and {base_path}_2{ext}")
                        return
                    except Exception as e2:
                        logger.error(f"All visualization methods failed: {e2}")
                        logger.error("Unable to visualize or save screenshots")
            else:
                logger.warning("Non-interactive mode requires a screenshot_path")

        except Exception as general_error:
            logger.error(f"Comparison visualization failed: {general_error}")
            logger.error("Check your Open3D installation and version compatibility")
    
    @staticmethod
    def analyze_point_cloud(pcd: Union[o3d.geometry.PointCloud, str]) -> Dict[str, Any]:
        """
        Analyze a point cloud and return statistics.
        
        Args:
            pcd: Point cloud or path to point cloud file
            
        Returns:
            Dictionary with point cloud statistics
        """
        # Load point cloud if a string path is provided
        if isinstance(pcd, str):
            pcd = ScanVisualizer.load_point_cloud(pcd)
        
        if len(pcd.points) == 0:
            return {"points": 0, "empty": True}
        
        # Extract points for analysis
        points = np.asarray(pcd.points)
        
        # Calculate statistics
        stats = {
            "points": len(points),
            "empty": len(points) == 0,
            "min": points.min(axis=0).tolist() if len(points) > 0 else None,
            "max": points.max(axis=0).tolist() if len(points) > 0 else None,
            "mean": points.mean(axis=0).tolist() if len(points) > 0 else None,
            "std": points.std(axis=0).tolist() if len(points) > 0 else None,
            "has_normals": pcd.has_normals(),
            "has_colors": pcd.has_colors(),
            "dimension": {
                "x": (points[:, 0].max() - points[:, 0].min()) if len(points) > 0 else 0,
                "y": (points[:, 1].max() - points[:, 1].min()) if len(points) > 0 else 0,
                "z": (points[:, 2].max() - points[:, 2].min()) if len(points) > 0 else 0
            },
            "density": {
                "average_distance": None,
                "std_distance": None
            }
        }
        
        # Calculate point density (if enough points)
        if len(points) > 10:
            try:
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                distances = []
                for i in range(min(100, len(points))):  # Sample up to 100 points
                    _, idx, dist = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)  # Find nearest neighbor
                    if len(dist) > 1:  # Skip the point itself (first match)
                        distances.append(np.sqrt(dist[1]))
                
                if distances:
                    stats["density"]["average_distance"] = np.mean(distances)
                    stats["density"]["std_distance"] = np.std(distances)
            except Exception as e:
                logger.warning(f"Could not compute density statistics: {e}")
        
        return stats
    
    @staticmethod
    def print_point_cloud_analysis(stats: Dict[str, Any]) -> str:
        """
        Format point cloud analysis as a human-readable string.
        
        Args:
            stats: Statistics from analyze_point_cloud
            
        Returns:
            Formatted analysis string
        """
        if stats.get("empty", True):
            return "Empty point cloud (no points)"
        
        info = []
        info.append(f"Point count: {stats['points']}")
        
        if stats.get("min") and stats.get("max"):
            info.append("Bounding box:")
            info.append(f"  Min: ({stats['min'][0]:.2f}, {stats['min'][1]:.2f}, {stats['min'][2]:.2f})")
            info.append(f"  Max: ({stats['max'][0]:.2f}, {stats['max'][1]:.2f}, {stats['max'][2]:.2f})")
        
        if stats.get("dimension"):
            info.append("Dimensions:")
            info.append(f"  X: {stats['dimension']['x']:.2f}")
            info.append(f"  Y: {stats['dimension']['y']:.2f}")
            info.append(f"  Z: {stats['dimension']['z']:.2f}")
        
        if stats.get("mean"):
            info.append(f"Centroid: ({stats['mean'][0]:.2f}, {stats['mean'][1]:.2f}, {stats['mean'][2]:.2f})")
        
        info.append(f"Has normals: {stats['has_normals']}")
        info.append(f"Has colors: {stats['has_colors']}")
        
        if stats.get("density", {}).get("average_distance"):
            info.append("Density:")
            info.append(f"  Average nearest neighbor distance: {stats['density']['average_distance']:.4f}")
            info.append(f"  Standard deviation of distances: {stats['density']['std_distance']:.4f}")
        
        return "\n".join(info)


# Standalone function for quick visualization
def visualize_scan_result(filepath: str, output_dir: Optional[str] = None, show_gui: bool = True) -> None:
    """
    Visualize a scan result (point cloud or mesh).
    
    Args:
        filepath: Path to the point cloud or mesh file
        output_dir: Optional directory to save visualization outputs
        show_gui: Whether to show an interactive GUI
    """
    if not OPEN3D_AVAILABLE:
        logger.error("open3d is required for visualization")
        print("Error: open3d is required for visualization. Install with 'pip install open3d'")
        return
    
    # Check if file exists
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        print(f"Error: File not found: {filepath}")
        return
    
    # Determine file type
    ext = os.path.splitext(filepath)[1].lower()
    is_mesh = ext in ['.obj', '.stl', '.off', '.gltf']
    
    # Create visualizer
    visualizer = ScanVisualizer(use_window=show_gui)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Analysis output path
    analysis_path = os.path.join(output_dir, "scan_analysis.txt") if output_dir else None
    screenshot_path = os.path.join(output_dir, "scan_visualization.png") if output_dir else None
    
    if is_mesh:
        # Visualize mesh
        mesh = visualizer.load_mesh(filepath)
        if len(mesh.triangles) > 0:
            print(f"Loaded mesh with {len(mesh.triangles)} triangles")
            visualizer.visualize_mesh(mesh, screenshot_path=screenshot_path)
        else:
            print("Mesh is empty or could not be loaded correctly")
    else:
        # Visualize point cloud
        pcd = visualizer.load_point_cloud(filepath)
        stats = visualizer.analyze_point_cloud(pcd)
        
        print(visualizer.print_point_cloud_analysis(stats))
        
        if len(pcd.points) > 0:
            visualizer.visualize_point_cloud(pcd, screenshot_path=screenshot_path)
            
            # Save analysis to file
            if analysis_path:
                with open(analysis_path, 'w') as f:
                    f.write(visualizer.print_point_cloud_analysis(stats))
                print(f"Analysis saved to {analysis_path}")
        else:
            print("Point cloud is empty or could not be loaded correctly")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize 3D scan results")
    parser.add_argument("filepath", help="Path to point cloud or mesh file")
    parser.add_argument("--output", "-o", help="Output directory for visualization files")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (only save screenshots)")
    
    args = parser.parse_args()
    
    visualize_scan_result(args.filepath, args.output, not args.no_gui)