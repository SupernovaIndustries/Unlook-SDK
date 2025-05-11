"""
Visualization Utilities for 3D Scanning Debugging.

This module provides helpful visualization tools for debugging 3D scanning issues.
It allows for better visualization of the scanning process, pattern projection,
and intermediate results.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union

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
                                points: np.ndarray, 
                                name: str = "pointcloud") -> bool:
        """
        Save a snapshot of the current point cloud.
        
        Args:
            points: 3D points array (Nx3)
            name: Base filename
        
        Returns:
            True if successful, False otherwise
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available, cannot save point cloud")
            return False
        
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Add colors based on Z value
            z_values = points[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            
            # Create color gradient based on Z value
            if z_max > z_min:
                normalized_z = (z_values - z_min) / (z_max - z_min)
                colors = np.zeros((len(points), 3))
                colors[:, 0] = normalized_z  # Red
                colors[:, 2] = 1.0 - normalized_z  # Blue
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save point cloud in PLY format
            filepath = os.path.join(self.output_dir, "pointcloud", f"{name}.ply")
            o3d.io.write_point_cloud(filepath, pcd)
            
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
                
                # Save rendered image
                self.save_image(image_cv, "pointcloud", f"{name}_render.png")
                
                # Clean up
                vis.destroy_window()
            
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