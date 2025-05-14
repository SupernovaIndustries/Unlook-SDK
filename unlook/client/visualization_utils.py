"""
Visualization Utilities for 3D Scanning Debugging.

This module provides helpful visualization tools for debugging 3D scanning issues.
It allows for better visualization of the scanning process, pattern projection,
and intermediate results.

Features:
1. Point cloud visualization with uncertainty heatmaps for ISO/ASTM 52902 compliance
2. Disparity map and depth map visualization
3. Correspondence visualization between stereo images
4. Error metrics visualization and analysis
5. Complete scanning process debug visualizations
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union

# Import for uncertainty visualization
try:
    from .direct_triangulator import PointCloudWithUncertainty, PointUncertainty
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False
    # Create fallback definitions for documentation purposes
    class PointUncertainty:
        pass
    class PointCloudWithUncertainty:
        pass

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. 3D visualization will be limited.")
    OPEN3D_AVAILABLE = False


class DebugVisualizer:
    """
    Debug visualizer for 3D scanning process.
    
    This class provides methods for creating helpful visualizations to
    debug the 3D scanning process, especially focused on structured light.
    """
    
    def __init__(self, output_dir: str = "debug_output"):
        """
        Initialize debug visualizer.
        
        Args:
            output_dir: Directory for debug output
        """
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "patterns"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rectified"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "correspondence"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "stereo"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pointcloud"), exist_ok=True)
        
        logger.info(f"Debug visualizer initialized with output dir: {output_dir}")
    
    def save_image(self, img: np.ndarray, subdir: str, name: str) -> bool:
        """
        Save an image to the debug output directory.
        
        Args:
            img: Image to save
            subdir: Subdirectory within output_dir
            name: Filename (with extension)
        
        Returns:
            True if successful, False otherwise
        """
        path = os.path.join(self.output_dir, subdir)
        os.makedirs(path, exist_ok=True)
        
        try:
            cv2.imwrite(os.path.join(path, name), img)
            return True
        except Exception as e:
            logger.error(f"Error saving image {name} to {path}: {e}")
            return False
    
    def save_pattern_images(self, 
                           patterns: List[Tuple[np.ndarray, np.ndarray]], 
                           pattern_names: Optional[List[str]] = None) -> bool:
        """
        Save pattern image pairs for debugging.
        
        Args:
            patterns: List of (left, right) image pairs
            pattern_names: Optional list of pattern names
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for i, (left, right) in enumerate(patterns):
                name = f"pattern_{i:02d}"
                if pattern_names and i < len(pattern_names):
                    name = pattern_names[i]
                
                self.save_image(left, "patterns", f"{name}_left.png")
                self.save_image(right, "patterns", f"{name}_right.png")
            
            logger.info(f"Saved {len(patterns)} pattern image pairs")
            return True
        except Exception as e:
            logger.error(f"Error saving pattern images: {e}")
            return False
    
    def save_rectified_images(self, 
                             images: List[Tuple[np.ndarray, np.ndarray]],
                             names: Optional[List[str]] = None) -> bool:
        """
        Save rectified image pairs for debugging.
        
        Args:
            images: List of (left, right) rectified image pairs
            names: Optional list of image names
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for i, (left, right) in enumerate(images):
                name = f"rect_{i:02d}"
                if names and i < len(names):
                    name = names[i]
                
                self.save_image(left, "rectified", f"{name}_left.png")
                self.save_image(right, "rectified", f"{name}_right.png")
                
                # Create a stereo anaglyph for 3D visualization (red-cyan)
                if left.shape == right.shape:
                    anaglyph = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
                    
                    # Handle grayscale or color images
                    if len(left.shape) == 2:
                        anaglyph[:, :, 0] = left  # Red channel from left eye
                        anaglyph[:, :, 1] = right  # Green channel from right eye
                        anaglyph[:, :, 2] = right  # Blue channel from right eye
                    else:
                        # For color images, split channels properly
                        anaglyph[:, :, 0] = left[:, :, 0]  # Red from left
                        anaglyph[:, :, 1] = right[:, :, 1]  # Green from right
                        anaglyph[:, :, 2] = right[:, :, 2]  # Blue from right
                    
                    self.save_image(anaglyph, "stereo", f"{name}_anaglyph.png")
            
            logger.info(f"Saved {len(images)} rectified image pairs with anaglyphs")
            return True
        except Exception as e:
            logger.error(f"Error saving rectified images: {e}")
            return False
    
    def create_correspondence_visualization(self, 
                                           left_img: np.ndarray,
                                           right_img: np.ndarray,
                                           correspondences: Dict[Tuple[int, int], Tuple[int, int]],
                                           max_points: int = 200) -> np.ndarray:
        """
        Create visualization of stereo correspondences.
        
        Args:
            left_img: Left image
            right_img: Right image
            correspondences: Dictionary of point correspondences
            max_points: Maximum number of points to visualize
        
        Returns:
            Visualization image with correspondence lines
        """
        # Create visualization image by concatenating left and right images horizontally
        if len(left_img.shape) == 2:
            # Convert grayscale to color
            left_color = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            right_color = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        else:
            left_color = left_img.copy()
            right_color = right_img.copy()
        
        h, w = left_img.shape[:2]
        vis_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis_img[:, :w] = left_color
        vis_img[:, w:] = right_color
        
        # Draw correspondence lines
        num_points = min(len(correspondences), max_points)
        selected_keys = list(correspondences.keys())[:num_points]
        
        for i, (x_left, y_left) in enumerate(selected_keys):
            x_right, y_right = correspondences[(x_left, y_left)]
            
            # Use a different color for each point based on position
            color_val = int(255 * (i / num_points))
            color = (0, color_val, 255 - color_val)
            
            # Draw points and line
            cv2.circle(vis_img, (x_left, y_left), 3, color, -1)
            cv2.circle(vis_img, (x_right + w, y_right), 3, color, -1)
            cv2.line(vis_img, (x_left, y_left), (x_right + w, y_right), color, 1)
        
        return vis_img
    
    def save_correspondence_visualization(self,
                                         left_img: np.ndarray,
                                         right_img: np.ndarray,
                                         correspondences: Dict[Tuple[int, int], Tuple[int, int]],
                                         name: str = "correspondences") -> bool:
        """
        Create and save visualization of stereo correspondences.
        
        Args:
            left_img: Left image
            right_img: Right image
            correspondences: Dictionary of point correspondences
            name: Base filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Full visualization with all points
            vis_full = self.create_correspondence_visualization(
                left_img, right_img, correspondences, max_points=1000)
            
            # Simplified visualization with fewer points
            vis_simple = self.create_correspondence_visualization(
                left_img, right_img, correspondences, max_points=50)
            
            # Save visualizations
            self.save_image(vis_full, "correspondence", f"{name}_full.png")
            self.save_image(vis_simple, "correspondence", f"{name}_simple.png")
            
            logger.info(f"Saved correspondence visualizations for {len(correspondences)} points")
            return True
        except Exception as e:
            logger.error(f"Error saving correspondence visualization: {e}")
            return False
    
    def create_disparity_visualization(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Create a colorized visualization of a disparity map.
        
        Args:
            disparity_map: Disparity map (float32)
        
        Returns:
            Colorized disparity visualization
        """
        # Normalize disparity to 0-255 range
        if np.max(disparity_map) > np.min(disparity_map):
            norm_disp = ((disparity_map - np.min(disparity_map)) / 
                        (np.max(disparity_map) - np.min(disparity_map)) * 255).astype(np.uint8)
        else:
            norm_disp = np.zeros_like(disparity_map, dtype=np.uint8)
        
        # Apply colormap
        color_disp = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)
        
        # Draw color bar
        h, w = disparity_map.shape[:2]
        color_bar_width = 30
        color_bar = np.zeros((h, color_bar_width, 3), dtype=np.uint8)
        
        for y in range(h):
            color_val = int(255 * (h - y - 1) / h)
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            color_bar[y, :] = color
        
        # Add values to color bar
        min_val = np.min(disparity_map)
        max_val = np.max(disparity_map)
        
        # Add colorbar to visualization
        vis_img = np.zeros((h, w + color_bar_width, 3), dtype=np.uint8)
        vis_img[:, :w] = color_disp
        vis_img[:, w:] = color_bar
        
        # Add min and max text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, f"{max_val:.1f}", (w + 2, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"{min_val:.1f}", (w + 2, h - 5), font, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def save_disparity_visualization(self, disparity_map: np.ndarray, name: str = "disparity") -> bool:
        """
        Create and save visualization of disparity map.
        
        Args:
            disparity_map: Disparity map
            name: Base filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create visualization
            vis_img = self.create_disparity_visualization(disparity_map)
            
            # Save visualization
            self.save_image(vis_img, "stereo", f"{name}.png")
            
            # Save raw disparity map for later analysis
            np.save(os.path.join(self.output_dir, "stereo", f"{name}.npy"), disparity_map)
            
            logger.info(f"Saved disparity visualization")
            return True
        except Exception as e:
            logger.error(f"Error saving disparity visualization: {e}")
            return False
    
    def save_mask_visualization(self, shadow_mask: np.ndarray, name: str = "shadow_mask") -> bool:
        """
        Save visualization of shadow mask.
        
        Args:
            shadow_mask: Shadow mask image
            name: Base filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save raw mask
            self.save_image(shadow_mask, "masks", f"{name}.png")
            
            # Create a more visible version with contours
            if len(shadow_mask.shape) == 2:
                contour_vis = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
            else:
                contour_vis = shadow_mask.copy()
            
            # Find contours
            contours, _ = cv2.findContours(
                shadow_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Draw contours
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            
            # Calculate and display coverage percentage
            coverage = 100 * np.count_nonzero(shadow_mask) / shadow_mask.size
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(contour_vis, f"Coverage: {coverage:.1f}%", 
                      (20, 30), font, 0.7, (0, 255, 0), 2)
            
            # Save contour visualization
            self.save_image(contour_vis, "masks", f"{name}_contours.png")
            
            logger.info(f"Saved mask visualization (coverage: {coverage:.1f}%)")
            return True
        except Exception as e:
            logger.error(f"Error saving mask visualization: {e}")
            return False
    
    def create_coordinate_map_visualization(self, 
                                           coordinate_map: np.ndarray, 
                                           shadow_mask: np.ndarray) -> np.ndarray:
        """
        Create visualization of decoded coordinate map.
        
        Args:
            coordinate_map: Map of decoded coordinates
            shadow_mask: Shadow mask
        
        Returns:
            Visualization image of coordinate map
        """
        h, w = coordinate_map.shape[:2]
        
        # Normalize coordinates for visualization
        if np.max(coordinate_map) > 0:
            norm_coords = (coordinate_map.astype(np.float32) / np.max(coordinate_map) * 255).astype(np.uint8)
        else:
            norm_coords = np.zeros_like(coordinate_map, dtype=np.uint8)
        
        # Apply colormap for better visualization
        color_coords = cv2.applyColorMap(norm_coords, cv2.COLORMAP_JET)
        
        # Create mask visualization
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[shadow_mask > 0] = [0, 255, 0]  # Green for areas with mask
        
        # Combine coordinate and mask visualizations
        alpha = 0.7
        combined = cv2.addWeighted(color_coords, alpha, mask_vis, 1 - alpha, 0)
        
        # Add statistics
        valid_coords = np.count_nonzero(coordinate_map)
        valid_percentage = 100 * valid_coords / max(1, np.count_nonzero(shadow_mask))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f"Valid coordinates: {valid_coords}", 
                  (20, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Coverage: {valid_percentage:.1f}%", 
                  (20, 60), font, 0.7, (255, 255, 255), 2)
        
        return combined
    
    def save_coordinate_map_visualization(self, 
                                         coordinate_map: np.ndarray, 
                                         shadow_mask: np.ndarray,
                                         name: str = "coordinate_map") -> bool:
        """
        Save visualization of coordinate map.
        
        Args:
            coordinate_map: Map of decoded coordinates
            shadow_mask: Shadow mask
            name: Base filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create visualization
            vis_img = self.create_coordinate_map_visualization(coordinate_map, shadow_mask)
            
            # Save visualization
            self.save_image(vis_img, "masks", f"{name}.png")
            
            logger.info(f"Saved coordinate map visualization")
            return True
        except Exception as e:
            logger.error(f"Error saving coordinate map visualization: {e}")
            return False
    
    def save_pointcloud_snapshot(self, 
                                points: Union[np.ndarray, PointCloudWithUncertainty], 
                                name: str = "pointcloud",
                                colorize_by: str = "z") -> bool:
        """
        Save a snapshot of the current point cloud.
        
        Args:
            points: 3D points array (Nx3) or PointCloudWithUncertainty object
            name: Base filename
            colorize_by: How to colorize the point cloud ('z', 'uncertainty', or 'confidence')
                - 'z': Color by depth (default)
                - 'uncertainty': Color by spatial uncertainty (red=high, blue=low)
                - 'confidence': Color by confidence score (blue=high, red=low)
        
        Returns:
            True if successful, False otherwise
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available, cannot save point cloud")
            return False
        
        try:
            # Handle different input types
            has_uncertainty = False
            uncertainties = []
            confidence_values = []
            
            # Extract points and uncertainty data based on input type
            if isinstance(points, PointCloudWithUncertainty) and UNCERTAINTY_AVAILABLE:
                has_uncertainty = True
                points_array = points.points
                uncertainties = [u.spatial_uncertainty for u in points.uncertainties]
                confidence_values = [u.disparity_confidence for u in points.uncertainties]
                uncertainty_report = points.uncertainty_report
                logger.info(f"Point cloud has uncertainty data (mean: {np.mean(uncertainties):.2f}mm)")
            else:
                # Regular point array
                points_array = points
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_array)
            
            # Determine coloring method
            if colorize_by == "uncertainty" and has_uncertainty:
                # Color by uncertainty (red = high uncertainty, blue = low uncertainty)
                if uncertainties:
                    u_min, u_max = np.min(uncertainties), np.max(uncertainties)
                    if u_max > u_min:
                        normalized_u = [(u - u_min) / (u_max - u_min) for u in uncertainties]
                        colors = np.zeros((len(points_array), 3))
                        colors[:, 0] = normalized_u  # Red channel (high uncertainty)
                        colors[:, 2] = [1.0 - u for u in normalized_u]  # Blue channel (low uncertainty)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        colorby_text = f"uncertainty ({u_min:.2f}-{u_max:.2f}mm)"
            elif colorize_by == "confidence" and has_uncertainty:
                # Color by confidence (blue = high confidence, red = low confidence)
                if confidence_values:
                    colors = np.zeros((len(points_array), 3))
                    colors[:, 0] = [1.0 - c for c in confidence_values]  # Red (low confidence)
                    colors[:, 2] = confidence_values  # Blue (high confidence)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    colorby_text = "confidence score (0-1)"
            else:
                # Default: Color by Z value
                z_values = points_array[:, 2]
                z_min, z_max = np.min(z_values), np.max(z_values)
                
                # Create color gradient based on Z value
                if z_max > z_min:
                    normalized_z = (z_values - z_min) / (z_max - z_min)
                    colors = np.zeros((len(points_array), 3))
                    colors[:, 0] = normalized_z  # Red
                    colors[:, 2] = 1.0 - normalized_z  # Blue
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    colorby_text = f"depth ({z_min:.1f}-{z_max:.1f}mm)"
            
            # Save point cloud in PLY format
            filepath = os.path.join(self.output_dir, "pointcloud", f"{name}.ply")
            o3d.io.write_point_cloud(filepath, pcd)
            
            # Save uncertainty data separately if available
            if has_uncertainty:
                import json
                uncertainty_path = os.path.join(self.output_dir, "pointcloud", f"{name}_uncertainty.json")
                with open(uncertainty_path, "w") as f:
                    json.dump(uncertainty_report, f, indent=2)
            
            # Create a rendered image of the point cloud
            if len(points) > 10:
                # Create a visualization to render the point cloud
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=800, height=600)
                vis.add_geometry(pcd)
                
                # Create coordinate system
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.max(z_values) * 0.1)
                vis.add_geometry(coord_frame)
                
                # Set view parameters
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                ctr.set_front([0, 0, -1])
                ctr.set_lookat([0, 0, 0])
                ctr.set_up([0, -1, 0])
                
                # Render and capture image
                vis.poll_events()
                vis.update_renderer()
                image = vis.capture_screen_float_buffer(do_render=True)
                
                # Convert to OpenCV format
                image_np = np.asarray(image) * 255
                image_cv = image_np.astype(np.uint8)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                
                # Add colorbar label
                if colorize_by != "z" and 'colorby_text' in locals():
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image_cv, f"Color by: {colorby_text}", 
                              (20, 30), font, 0.7, (255, 255, 255), 2)
                
                # Save rendered image
                self.save_image(image_cv, "pointcloud", f"{name}_render.png")
                
                # Clean up
                vis.destroy_window()
            
            if isinstance(points, PointCloudWithUncertainty):
                logger.info(f"Saved point cloud snapshot with {len(points.points)} points and uncertainty data")
            else:
                logger.info(f"Saved point cloud snapshot with {len(points)} points")
            return True
        except Exception as e:
            logger.error(f"Error saving point cloud snapshot: {e}")
            return False
    
    def save_debug_summary(self, 
                          info: Dict[str, Any], 
                          name: str = "debug_summary") -> bool:
        """
        Save a summary of debug information.
        
        Args:
            info: Dictionary of debug information
            name: Base filename
        
        Returns:
            True if successful, False otherwise
        """
        import json
        
        try:
            # Add timestamp
            info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save JSON file
            with open(os.path.join(self.output_dir, f"{name}.json"), "w") as f:
                json.dump(info, f, indent=2)
            
            # Create a text summary
            summary_text = f"Scan Debug Summary ({info['timestamp']})\n"
            summary_text += "=" * 50 + "\n\n"
            
            # Add key information
            for key, value in info.items():
                if key == "timestamp":
                    continue
                
                if isinstance(value, dict):
                    summary_text += f"{key}:\n"
                    for k, v in value.items():
                        summary_text += f"  {k}: {v}\n"
                    summary_text += "\n"
                else:
                    summary_text += f"{key}: {value}\n"
            
            # Save text summary
            with open(os.path.join(self.output_dir, f"{name}.txt"), "w") as f:
                f.write(summary_text)
            
            logger.info(f"Saved debug summary to {self.output_dir}/{name}.json")
            return True
        except Exception as e:
            logger.error(f"Error saving debug summary: {e}")
            return False
    
    def create_uncertainty_visualization(self,
                                     cloud_with_uncertainty: PointCloudWithUncertainty,
                                     colormap_type: str = "jet") -> np.ndarray:
        """
        Create visualization of point cloud uncertainty according to ISO/ASTM 52902.
        
        Args:
            cloud_with_uncertainty: Point cloud with uncertainty data
            colormap_type: OpenCV colormap type (default: jet)
            
        Returns:
            Visualization image with uncertainty heatmap
        """
        if not OPEN3D_AVAILABLE or not UNCERTAINTY_AVAILABLE:
            logger.warning("Open3D or uncertainty module not available")
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_with_uncertainty.points)
            
            # Get uncertainty values
            uncertainties = [u.spatial_uncertainty for u in cloud_with_uncertainty.uncertainties]
            uncertainty_min = min(uncertainties)
            uncertainty_max = max(uncertainties)
            
            # Normalize uncertainties to 0-1 range
            normalized = [(u - uncertainty_min) / (uncertainty_max - uncertainty_min)
                        if uncertainty_max > uncertainty_min else 0.5 
                        for u in uncertainties]
            
            # Create colors (red = high uncertainty, blue = low uncertainty)
            colors = np.zeros((len(normalized), 3))
            colors[:, 0] = normalized  # Red channel for high uncertainty
            colors[:, 2] = [1.0 - n for n in normalized]  # Blue for low uncertainty
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Create a visualization to render the point cloud
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=800, height=600)
            vis.add_geometry(pcd)
            
            # Set view parameters
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])
            
            # Render and capture image
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            
            # Convert to OpenCV format
            image_np = np.asarray(image) * 255
            image_cv = image_np.astype(np.uint8)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # Add uncertainty colorbar
            colorbar_width = 50
            h, w = image_cv.shape[:2]
            colorbar = np.zeros((h, colorbar_width, 3), dtype=np.uint8)
            
            for y in range(h):
                # Color from red (high uncertainty) to blue (low uncertainty)
                progress = 1.0 - (y / h)  # 1 at top, 0 at bottom
                red = int(255 * progress)
                blue = int(255 * (1.0 - progress))
                colorbar[y, :] = [blue, 0, red]  # BGR format
            
            # Add colorbar labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(colorbar, f"{uncertainty_max:.2f}mm", (5, 30), font, 0.5, (255, 255, 255), 1)
            cv2.putText(colorbar, f"{uncertainty_min:.2f}mm", (5, h-20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(colorbar, "Uncertainty", (5, h//2), font, 0.5, (255, 255, 255), 1)
            
            # Combine image and colorbar
            result = np.zeros((h, w + colorbar_width, 3), dtype=np.uint8)
            result[:, :w] = image_cv
            result[:, w:] = colorbar
            
            # Add title
            title = f"ISO/ASTM 52902 Uncertainty Visualization"
            cv2.putText(result, title, (20, 30), font, 0.7, (255, 255, 255), 2)
            stats = f"Mean: {np.mean(uncertainties):.2f}mm | Max: {np.max(uncertainties):.2f}mm"
            cv2.putText(result, stats, (20, 60), font, 0.6, (255, 255, 255), 1)
            
            # Clean up
            vis.destroy_window()
            
            return result
        except Exception as e:
            logger.error(f"Error creating uncertainty visualization: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def save_uncertainty_visualization(self,
                                      cloud_with_uncertainty: PointCloudWithUncertainty,
                                      name: str = "uncertainty") -> bool:
        """
        Create and save visualization of point cloud uncertainty for ISO/ASTM 52902 compliance.
        
        Args:
            cloud_with_uncertainty: Point cloud with uncertainty data
            name: Base filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create visualization
            vis_img = self.create_uncertainty_visualization(cloud_with_uncertainty)
            
            # Save visualization
            self.save_image(vis_img, "pointcloud", f"{name}_map.png")
            
            # Save uncertainty report
            import json
            report_path = os.path.join(self.output_dir, "pointcloud", f"{name}_report.json")
            with open(report_path, "w") as f:
                json.dump(cloud_with_uncertainty.uncertainty_report, f, indent=2)
            
            # Create a simplified text report
            report = cloud_with_uncertainty.uncertainty_report
            text_report = "ISO/ASTM 52902 Uncertainty Report\n"
            text_report += "=" * 40 + "\n\n"
            
            # Add statistics
            if "statistics" in report:
                text_report += "Measurement Statistics:\n"
                for key, value in report["statistics"].items():
                    if key != "uncertainty_distribution":
                        text_report += f"  {key}: {value}\n"
                text_report += "\n"
            
            # Add measurement uncertainties
            if "measurement_uncertainties" in report:
                text_report += "Measurement Uncertainties:\n"
                for key, value in report["measurement_uncertainties"].items():
                    text_report += f"  {key}: {value}\n"
                text_report += "\n"
                
            # Add certification status
            if "certification_status" in report:
                text_report += "ISO/ASTM 52902 Certification Status:\n"
                for key, value in report["certification_status"].items():
                    status = "✓ COMPLIANT" if value else "✗ NON-COMPLIANT"
                    text_report += f"  {key}: {status}\n"
            
            # Save text report
            with open(os.path.join(self.output_dir, "pointcloud", f"{name}_report.txt"), "w") as f:
                f.write(text_report)
            
            # Create histogram visualization
            if "statistics" in report and "uncertainty_distribution" in report["statistics"]:
                dist = report["statistics"]["uncertainty_distribution"]
                if dist:
                    # Create histogram image
                    hist_height = 300
                    hist_width = 600
                    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
                    
                    # Get the bin values and labels
                    bins = list(dist.keys())
                    values = list(dist.values())
                    max_value = max(values) if values else 1
                    
                    # Draw histogram bars
                    bar_width = hist_width // (len(bins) + 1)
                    for i, (bin_label, count) in enumerate(zip(bins, values)):
                        # Calculate bar height relative to max value
                        bar_height = int((count / max_value) * (hist_height - 50))
                        
                        # Get position for this bar
                        x = (i + 1) * bar_width
                        y = hist_height - 30 - bar_height
                        
                        # Draw the bar
                        cv2.rectangle(hist_img, (x, hist_height-30), (x+bar_width-5, y), (0, 255, 0), -1)
                        
                        # Add the count label above bar
                        cv2.putText(hist_img, str(count), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Add bin label below bar
                        cv2.putText(hist_img, bin_label, (x, hist_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Add title
                    title = "Uncertainty Distribution (ISO/ASTM 52902)"
                    cv2.putText(hist_img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save histogram
                    self.save_image(hist_img, "pointcloud", f"{name}_histogram.png")
            
            logger.info(f"Saved uncertainty visualization and report")
            return True
        except Exception as e:
            logger.error(f"Error saving uncertainty visualization: {e}")
            return False

    def create_iso_astm_52902_report(self,
                                    cloud_with_uncertainty: PointCloudWithUncertainty,
                                    scan_params: Dict[str, Any],
                                    name: str = "iso_astm_52902_report") -> bool:
        """
        Create comprehensive ISO/ASTM 52902 compliance report with visualizations.
        
        This generates the complete certification report including uncertainty visualizations,
        statistics, and compliance status according to the standard.
        
        Args:
            cloud_with_uncertainty: Point cloud with uncertainty data
            scan_params: Scanning parameters (configuration)
            name: Base filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory structure
            report_dir = os.path.join(self.output_dir, "certification")
            os.makedirs(report_dir, exist_ok=True)
            
            # Save point cloud with different uncertainty visualizations
            self.save_pointcloud_snapshot(cloud_with_uncertainty, name="iso_52902_uncertainty", colorize_by="uncertainty")
            self.save_pointcloud_snapshot(cloud_with_uncertainty, name="iso_52902_confidence", colorize_by="confidence")
            
            # Save uncertainty visualization
            self.save_uncertainty_visualization(cloud_with_uncertainty, name="iso_52902")
            
            # Create a comprehensive HTML report
            report = cloud_with_uncertainty.uncertainty_report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ISO/ASTM 52902 Compliance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #3c78d8; }}
                    .section {{ margin-bottom: 20px; }}
                    .info-box {{ background-color: #f1f1f1; padding: 10px; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .compliant {{ color: #2e7d32; font-weight: bold; }}
                    .non-compliant {{ color: #c62828; font-weight: bold; }}
                    .gallery {{ display: flex; flex-wrap: wrap; }}
                    .gallery img {{ margin: 5px; max-width: 300px; }}
                </style>
            </head>
            <body>
                <h1>ISO/ASTM 52902 Compliance Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Scan Configuration</h2>
                    <div class="info-box">
                        <table>
                            <tr><th>Parameter</th><th>Value</th></tr>
            """
            
            # Add scan parameters
            for key, value in scan_params.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            
            # Add statistics
            html += """
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Uncertainty Statistics</h2>
                    <div class="info-box">
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
            """
            
            if "statistics" in report:
                for key, value in report["statistics"].items():
                    if key != "uncertainty_distribution":
                        html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            
            # Add measurement uncertainties
            html += """
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Measurement Uncertainties</h2>
                    <div class="info-box">
                        <table>
                            <tr><th>Measurement Type</th><th>Uncertainty</th></tr>
            """
            
            if "measurement_uncertainties" in report:
                for key, value in report["measurement_uncertainties"].items():
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            
            # Add certification status
            html += """
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <h2>ISO/ASTM 52902 Certification Status</h2>
                    <div class="info-box">
                        <table>
                            <tr><th>Requirement</th><th>Status</th></tr>
            """
            
            if "certification_status" in report:
                for key, value in report["certification_status"].items():
                    status_text = "<span class='compliant'>✓ COMPLIANT</span>" if value else "<span class='non-compliant'>✗ NON-COMPLIANT</span>"
                    html += f"<tr><td>{key}</td><td>{status_text}</td></tr>\n"
            
            # Add visualizations
            html += """
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Uncertainty Visualizations</h2>
                    <div class="gallery">
                        <div>
                            <img src="../pointcloud/iso_52902_uncertainty_render.png" alt="Uncertainty Visualization">
                            <div>Uncertainty Heatmap</div>
                        </div>
                        <div>
                            <img src="../pointcloud/iso_52902_map.png" alt="ISO/ASTM 52902 Uncertainty Map">
                            <div>ISO/ASTM 52902 Uncertainty Map</div>
                        </div>
                        <div>
                            <img src="../pointcloud/iso_52902_histogram.png" alt="Uncertainty Distribution">
                            <div>Uncertainty Distribution</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Standard Information</h2>
                    <div class="info-box">
                        <p>This report complies with ISO/ASTM 52902 - "Additive manufacturing — Test artifacts — 
                        Geometric capability assessment of additive manufacturing systems"</p>
                        <p>The standard requires quantitative uncertainty measurements for reliable geometric assessment.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML report
            with open(os.path.join(report_dir, f"{name}.html"), "w") as f:
                f.write(html)
            
            logger.info(f"Created ISO/ASTM 52902 compliance report")
            return True
        except Exception as e:
            logger.error(f"Error creating ISO/ASTM 52902 report: {e}")
            return False
            
    def create_scan_report(self, 
                          config: Dict[str, Any], 
                          stats: Dict[str, Any],
                          diagnostics: Dict[str, Any]) -> bool:
        """
        Create a comprehensive scan report with all debug information.
        
        Args:
            config: Scanner configuration
            stats: Scanning statistics
            diagnostics: Scanner diagnostics
        
        Returns:
            True if successful, False otherwise
        """
        import json
        
        try:
            # Create report data
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": config,
                "stats": stats,
                "diagnostics": diagnostics
            }
            
            # Save JSON report
            with open(os.path.join(self.output_dir, "scan_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            
            # Create HTML report (more user-friendly)
            html = self._create_html_report(report)
            
            # Save HTML report
            with open(os.path.join(self.output_dir, "scan_report.html"), "w") as f:
                f.write(html)
            
            logger.info(f"Created scan report in {self.output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creating scan report: {e}")
            return False
    
    def _create_html_report(self, report: Dict[str, Any]) -> str:
        """
        Create HTML report from report data.
        
        Args:
            report: Report data
        
        Returns:
            HTML report as string
        """
        # Basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unlook SDK Scan Report - {report["timestamp"]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #3c78d8; }}
                .section {{ margin-bottom: 20px; }}
                .info-box {{ background-color: #f1f1f1; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .warning {{ color: #e65100; font-weight: bold; }}
                .error {{ color: #c62828; font-weight: bold; }}
                .success {{ color: #2e7d32; font-weight: bold; }}
                .gallery {{ display: flex; flex-wrap: wrap; }}
                .gallery img {{ margin: 5px; max-width: 300px; }}
            </style>
        </head>
        <body>
            <h1>Unlook SDK Scan Report</h1>
            <p>Generated: {report["timestamp"]}</p>
            
            <div class="section">
                <h2>Scanner Configuration</h2>
                <div class="info-box">
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        # Add configuration parameters
        for key, value in report.get("config", {}).items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
        
        # Add statistics
        html += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>Scan Statistics</h2>
                <div class="info-box">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for key, value in report.get("stats", {}).items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
        
        # Add diagnostics
        html += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>Diagnostics</h2>
                <div class="info-box">
        """
        
        # Process each diagnostic category
        for category, info in report.get("diagnostics", {}).items():
            html += f"<h3>{category}</h3>\n<table><tr><th>Parameter</th><th>Value</th></tr>\n"
            
            if isinstance(info, dict):
                for key, value in info.items():
                    # Add color coding for certain values
                    if key.endswith("status"):
                        if "OK" in str(value):
                            html += f"<tr><td>{key}</td><td class='success'>{value}</td></tr>\n"
                        elif "WARNING" in str(value) or "MODERATE" in str(value):
                            html += f"<tr><td>{key}</td><td class='warning'>{value}</td></tr>\n"
                        elif "ERROR" in str(value) or "POOR" in str(value) or "FAILED" in str(value):
                            html += f"<tr><td>{key}</td><td class='error'>{value}</td></tr>\n"
                        else:
                            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
                    else:
                        html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            else:
                html += f"<tr><td>{category}</td><td>{info}</td></tr>\n"
            
            html += "</table>\n"
        
        # Add image gallery with available debug images
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>Debug Images</h2>
                <div class="gallery">
        """
        
        # Find image files in the debug output directory
        image_categories = ["masks", "correspondence", "stereo", "pointcloud"]
        for category in image_categories:
            path = os.path.join(self.output_dir, category)
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.endswith((".png", ".jpg")):
                        rel_path = f"{category}/{filename}"
                        img_title = f"{category}: {filename}"
                        html += f"<div><img src='{rel_path}' alt='{img_title}' title='{img_title}'><div>{img_title}</div></div>\n"
        
        # Close HTML document
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html