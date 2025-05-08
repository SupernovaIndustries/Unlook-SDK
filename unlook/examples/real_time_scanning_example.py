#!/usr/bin/env python3
"""
Example script demonstrating real-time 3D scanning with the UnLook scanner.

This example shows how to:
1. Configure and use the RealTimeScanner
2. Perform different types of scans (single-shot, continuous)
3. Use callbacks to monitor the scanning process
4. Process and save the scan results as point clouds
5. Generate 3D meshes from point clouds using Open3D
6. Visualize the generated 3D meshes

The example will save:
- Raw point cloud files (.ply)
- Processed point clouds with noise removal (.ply)
- 3D mesh files generated from the point clouds (.ply)

Requirements:
- Open3D library for mesh generation (pip install open3d)
- NumPy for array operations
- OpenCV for visualization

Usage:
    python real_time_scanning_example.py [--continuous] [--quality {low,medium,high,ultra}] [--show-mesh]

Options:
    --continuous    Run in continuous scanning mode instead of single scan
    --quality       Set scanning quality (low, medium, high, ultra)
    --show-mesh     Show the generated 3D mesh after scanning (requires Open3D)
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
import threading
from typing import Dict, Any, Optional, Tuple

# Add parent directories to path to make imports work properly
# First add the parent directory (unlook)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
# Also add the SDK root directory
sdk_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)

# Import Open3D for point cloud processing and meshing
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not found. Install with 'pip install open3d' to enable mesh generation.")
    logging.warning("You can find Open3D at https://github.com/isl-org/Open3D")

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Now import from unlook
from unlook import UnlookClient
from unlook.client.real_time_scanner import RealTimeScanner, ScanResult, ScanFrameData
from unlook.client.scan_config import RealTimeScannerConfig, ScanningMode, ScanningQuality, PatternType


# Ensure output directory exists
OUTPUT_DIR = "scan_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
scanner = None
visualization_window = None
visualization_image = None
visualization_lock = threading.Lock()


def on_scan_progress(progress: float, metadata: Dict[str, Any]):
    """Callback for scan progress updates."""
    logger.info(f"Scan progress: {progress*100:.1f}% - Frames: {metadata.get('frame_count', 0)}")


def on_scan_completed(result: ScanResult):
    """Callback for scan completion."""
    logger.info(f"Scan completed with {len(result.point_cloud) if result.point_cloud is not None else 0} points")
    
    # Save the result with mesh generation
    try:
        # Get the raw scan folder from the scanner
        if scanner is None:
            logger.error("Scanner object is not available")
            return
            
        # Determine the raw folder path used by the scanner
        # This is the same folder where raw images are saved (raw_TIMESTAMP)
        raw_folder = os.path.join(
            scanner.config.output_directory,
            f"raw_{int(scanner.start_time)}"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(raw_folder, exist_ok=True)
        
        # Create a base filename in the raw folder
        base_filepath = os.path.join(raw_folder, "scan_result")
        
        # Display some stats about the point cloud
        if result.point_cloud is not None:
            points = result.point_cloud
            logger.info(f"Point cloud stats:")
            logger.info(f"  - Points: {len(points)}")
            logger.info(f"  - X range: {np.min(points[:,0]):.2f} to {np.max(points[:,0]):.2f}")
            logger.info(f"  - Y range: {np.min(points[:,1]):.2f} to {np.max(points[:,1]):.2f}")
            logger.info(f"  - Z range: {np.min(points[:,2]):.2f} to {np.max(points[:,2]):.2f}")
        
        # Save both point cloud and mesh
        point_cloud_path, mesh_path = save_point_cloud_and_mesh(result, base_filepath)
        
        if point_cloud_path:
            logger.info(f"Saved scan data to {point_cloud_path}")
            if mesh_path:
                logger.info(f"Mesh created and saved to {mesh_path}")
                
                # Visualize the mesh if Open3D is available
                if OPEN3D_AVAILABLE:
                    logger.info("You can view the mesh with Open3D using:")
                    logger.info(f"    import open3d as o3d")
                    logger.info(f"    mesh = o3d.io.read_triangle_mesh('{mesh_path}')")
                    logger.info(f"    o3d.visualization.draw_geometries([mesh])")
        
    except Exception as e:
        logger.error(f"Error saving scan result: {e}")
        import traceback
        logger.error(traceback.format_exc())


def on_scan_error(error_message: str, exception: Exception):
    """Callback for scan errors."""
    logger.error(f"Scan error: {error_message}")
    if exception:
        logger.error(f"Exception: {str(exception)}")


def on_frame_captured(frame_data: ScanFrameData):
    """Callback for frame captures."""
    global visualization_window, visualization_image
    
    # Log frame info periodically
    if frame_data.index % 5 == 0:
        logger.debug(f"Captured frame {frame_data.index} from camera {frame_data.camera_id} "
                   f"with pattern: {frame_data.pattern_info.get('pattern_type', 'unknown')}")
    
    # Visualize the frame (optional)
    if visualization_window is not None:
        with visualization_lock:
            # Copy the image for display
            visualization_image = frame_data.image.copy()
            
            # Add information overlay
            cv2.putText(
                visualization_image,
                f"Camera: {frame_data.camera_id} | Pattern: {frame_data.index} | "
                f"Type: {frame_data.pattern_info.get('pattern_type', 'unknown')}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


def on_result_ready(result: ScanResult):
    """Callback for when a new result is ready (for continuous mode)."""
    logger.info(f"New scan result ready with {len(result.point_cloud) if result.point_cloud is not None else 0} points")
    
    # In continuous mode, we could save each result or just the final one
    # For this example, we'll save every 5th result
    if result.metadata.get("sequence_index", 0) % 5 == 0:
        try:
            # Get the raw scan folder from the scanner
            if scanner is None:
                logger.error("Scanner object is not available")
                return
                
            # Determine the raw folder path used by the scanner
            # This is the same folder where raw images are saved (raw_TIMESTAMP)
            raw_folder = os.path.join(
                scanner.config.output_directory,
                f"raw_{int(scanner.start_time)}"
            )
            
            # Create directory if it doesn't exist
            os.makedirs(raw_folder, exist_ok=True)
            
            # Create a base filename in the raw folder
            sequence_index = result.metadata.get("sequence_index", 0)
            base_filepath = os.path.join(raw_folder, f"scan_continuous_{sequence_index}")
            
            # Save both point cloud and mesh
            point_cloud_path, mesh_path = save_point_cloud_and_mesh(result, base_filepath)
            
            if point_cloud_path:
                logger.info(f"Saved continuous scan point cloud to {point_cloud_path}")
                if mesh_path:
                    logger.info(f"Saved continuous scan mesh to {mesh_path}")
                    
        except Exception as e:
            logger.error(f"Error saving continuous scan result: {e}")
            import traceback
            logger.error(traceback.format_exc())


def create_mesh_from_point_cloud(point_cloud: np.ndarray, colors: Optional[np.ndarray] = None) -> Optional[Tuple[Any, Any]]:
    """
    Create a mesh from a point cloud using Open3D.
    
    Args:
        point_cloud: Nx3 array of 3D points
        colors: Optional Nx3 array of RGB colors (0-255)
        
    Returns:
        Tuple containing (pcd, mesh) objects from Open3D, or None if Open3D is not available
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available. Cannot create mesh.")
        return None
        
    if point_cloud is None or len(point_cloud) == 0:
        logger.warning("No point cloud data provided for mesh creation")
        return None
        
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Add colors if available
        if colors is not None:
            # Convert to float in range [0, 1]
            colors_normalized = colors.astype(float) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        # Remove outliers
        logger.info("Removing noise from point cloud...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Estimate normals (required for surface reconstruction)
        logger.info("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
        
        # Create mesh using Poisson surface reconstruction
        logger.info("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)
        
        # Clean up the mesh - remove low density vertices
        logger.info("Cleaning up mesh...")
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        logger.info(f"Mesh created with {len(mesh.triangles)} triangles")
        return pcd, mesh
        
    except Exception as e:
        logger.error(f"Error creating mesh: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def save_point_cloud_and_mesh(result: ScanResult, base_filepath: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Save both the point cloud and reconstructed mesh to files.
    
    Args:
        result: Scan result containing point cloud data
        base_filepath: Base path/name for the files (without extension)
        
    Returns:
        Tuple of (point_cloud_path, mesh_path) or (None, None) if saving failed
    """
    if not result or not result.has_data():
        logger.warning("No valid result to save")
        return None, None
    
    # Print debugging info about the result
    logger.info(f"Saving result with {len(result.point_cloud)} points to {base_filepath}")
        
    # Define file paths with absolute paths
    point_cloud_path = os.path.abspath(f"{base_filepath}.ply")
    mesh_path_ply = os.path.abspath(f"{base_filepath}_mesh.ply")
    mesh_path_obj = os.path.abspath(f"{base_filepath}_mesh.obj")
    
    # Print the full paths for debugging
    logger.info(f"Full point cloud path: {point_cloud_path}")
    logger.info(f"Full mesh path (OBJ): {mesh_path_obj}")
    
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
        
        # Save the point cloud directly to file instead of using scanner's function
        try:
            # Direct save of PLY file using Open3D
            if OPEN3D_AVAILABLE and result.point_cloud is not None:
                # Convert numpy array to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(result.point_cloud)
                
                # Add colors if available
                if result.texture_map is not None:
                    colors = result.texture_map.astype(float) / 255.0  # Normalize to [0,1]
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Write to file
                success = o3d.io.write_point_cloud(point_cloud_path, pcd)
                if success:
                    logger.info(f"Directly saved point cloud to {point_cloud_path}")
                else:
                    logger.error(f"Failed to directly save point cloud to {point_cloud_path}")
            else:
                # Fallback to scanner's built-in function
                scanner.save_result(result, point_cloud_path)
                logger.info(f"Saved point cloud using scanner method to {point_cloud_path}")
        except Exception as e:
            logger.error(f"Error saving point cloud: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Create and save the mesh if Open3D is available
        if OPEN3D_AVAILABLE and result.point_cloud is not None:
            try:
                pcd_mesh_result = create_mesh_from_point_cloud(result.point_cloud, result.texture_map)
                
                if pcd_mesh_result:
                    pcd, mesh = pcd_mesh_result
                    
                    # Save the processed point cloud
                    processed_path = os.path.abspath(f"{base_filepath}_processed.ply")
                    try:
                        success = o3d.io.write_point_cloud(processed_path, pcd)
                        if success:
                            logger.info(f"Saved processed point cloud to {processed_path}")
                        else:
                            logger.error(f"Failed to save processed point cloud")
                    except Exception as e:
                        logger.error(f"Error saving processed point cloud: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # Save the mesh in PLY format
                    try:
                        # Explicitly compute normals
                        mesh.compute_vertex_normals()
                        success = o3d.io.write_triangle_mesh(mesh_path_ply, mesh)
                        if success:
                            logger.info(f"Saved mesh in PLY format to {mesh_path_ply}")
                        else:
                            logger.error(f"Failed to save PLY mesh (write_triangle_mesh returned False)")
                    except Exception as e:
                        logger.error(f"Error saving PLY mesh: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # Save the mesh in OBJ format (more compatible with many 3D software)
                    try:
                        # Make sure mesh has vertex colors for better OBJ export
                        if not mesh.has_vertex_colors():
                            # Add default colors if none exist
                            default_color = np.array([[0.7, 0.7, 0.7]])
                            mesh.vertex_colors = o3d.utility.Vector3dVector(
                                np.tile(default_color, (len(mesh.vertices), 1))
                            )
                        
                        # We must explicitly compute normals for OBJ export
                        mesh.compute_vertex_normals()
                        
                        # Write to OBJ format
                        success = o3d.io.write_triangle_mesh(mesh_path_obj, mesh)
                        
                        if success:
                            logger.info(f"Saved mesh in OBJ format to {mesh_path_obj}")
                        else:
                            logger.error(f"Failed to save mesh as OBJ (write_triangle_mesh returned False)")
                            
                        # Double-check if file exists
                        if os.path.exists(mesh_path_obj):
                            logger.info(f"Confirmed OBJ file exists at {mesh_path_obj}")
                            file_size = os.path.getsize(mesh_path_obj)
                            logger.info(f"OBJ file size: {file_size} bytes")
                        else:
                            logger.error(f"OBJ file was not created at {mesh_path_obj}")
                    except Exception as e:
                        logger.error(f"Error saving OBJ mesh: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # If OBJ fails, still return the PLY path
                        return point_cloud_path, mesh_path_ply
                    
                    # Return the OBJ path since that's the preferred format
                    return point_cloud_path, mesh_path_obj
                else:
                    logger.warning("Failed to create mesh")
                    return point_cloud_path, None
            except Exception as e:
                logger.error(f"Error in mesh generation process: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return point_cloud_path, None
        else:
            logger.warning("Open3D not available or no point cloud data - cannot create mesh")
            return point_cloud_path, None
            
    except Exception as e:
        logger.error(f"Error saving point cloud and mesh: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def update_visualization():
    """Update the visualization window."""
    global visualization_window, visualization_image
    
    # Create window
    window_name = "UnLook Real-Time Scanning"
    visualization_window = window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Display loop
    while visualization_window is not None:
        try:
            with visualization_lock:
                if visualization_image is not None:
                    cv2.imshow(window_name, visualization_image)
            
            # Check for key press (ESC to quit)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
                
            # Small delay
            time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            break
    
    # Clean up
    cv2.destroyAllWindows()
    visualization_window = None


def single_scan_example(client: UnlookClient, quality: ScanningQuality):
    """Perform a single 3D scan."""
    global scanner, visualization_window
    
    logger.info(f"=== Single 3D Scan Example (Quality: {quality.value}) ===")
    
    # Create scanner
    scanner = RealTimeScanner(client)
    
    # Configure scanner
    config = RealTimeScannerConfig.create_preset(quality)
    config.set_mode(ScanningMode.SINGLE)
    config.enable_save_raw_images(True, output_dir=OUTPUT_DIR)
    
    # Configure scanner with more explicit parameters
    config.pattern_type = PatternType.PHASE_SHIFT
    config.pattern_steps = 8 if quality != ScanningQuality.LOW else 4
    
    # Apply the configuration
    scanner.configure(config)
    
    # Register callbacks
    scanner.on_scan_progress(on_scan_progress)
    scanner.on_scan_completed(on_scan_completed)
    scanner.on_scan_error(on_scan_error)
    scanner.on_frame_captured(on_frame_captured)
    
    # Start visualization in a separate thread
    vis_thread = threading.Thread(target=update_visualization, daemon=True)
    vis_thread.start()
    
    try:
        # Start scanner
        logger.info("Starting single 3D scan...")
        if not scanner.start():
            logger.error("Failed to start scanner")
            return
        
        # Wait for scanning to complete - in SINGLE mode it will stop automatically
        while scanner.scanning:
            time.sleep(0.1)
        
        # Get the final result
        result = scanner.get_latest_result()
        
        if result and result.has_data():
            logger.info(f"Scan completed successfully with {len(result.point_cloud)} points")
            
            # Save the result with mesh generation
            # Use the raw folder from the scanner
            raw_folder = os.path.join(
                scanner.config.output_directory,
                f"raw_{int(scanner.start_time)}"
            )
            os.makedirs(raw_folder, exist_ok=True)
            base_filepath = os.path.join(raw_folder, "final_scan_result")
            
            point_cloud_path, mesh_path = save_point_cloud_and_mesh(result, base_filepath)
            
            if point_cloud_path:
                logger.info(f"Final point cloud saved to {point_cloud_path}")
                
                if mesh_path:
                    logger.info(f"Final mesh saved to {mesh_path}")
                    
                    # Visualize the mesh with Open3D if available
                    if OPEN3D_AVAILABLE and "--show-mesh" in sys.argv:
                        logger.info("Opening mesh visualization...")
                        mesh = o3d.io.read_triangle_mesh(mesh_path)
                        mesh.compute_vertex_normals()
                        o3d.visualization.draw_geometries([mesh])
        else:
            logger.warning("Scan completed but no valid result available")
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        scanner.stop()
    
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        scanner.stop()
    
    finally:
        # Cleanup
        visualization_window = None
        vis_thread.join(timeout=1.0)
        
        logger.info("Single scan completed")


def continuous_scan_example(client: UnlookClient, quality: ScanningQuality):
    """Perform continuous 3D scanning."""
    global scanner, visualization_window
    
    logger.info(f"=== Continuous 3D Scanning Example (Quality: {quality.value}) ===")
    
    # Create scanner
    scanner = RealTimeScanner(client)
    
    # Configure scanner
    config = RealTimeScannerConfig.create_preset(quality)
    config.set_mode(ScanningMode.CONTINUOUS)
    config.real_time_processing = True  # Enable real-time processing
    
    # For continuous scanning, we might want faster scanning with lower quality
    if quality == ScanningQuality.HIGH:
        # Override to use fewer pattern steps for better performance
        config.pattern_steps = 6
    
    # Apply the configuration
    scanner.configure(config)
    
    # Register callbacks
    scanner.on_scan_progress(on_scan_progress)
    scanner.on_scan_error(on_scan_error)
    scanner.on_frame_captured(on_frame_captured)
    scanner.on_result_ready(on_result_ready)
    
    # Start visualization in a separate thread
    vis_thread = threading.Thread(target=update_visualization, daemon=True)
    vis_thread.start()
    
    try:
        # Start scanner
        logger.info("Starting continuous 3D scanning...")
        if not scanner.start():
            logger.error("Failed to start scanner")
            return
        
        # Run for a fixed duration (15 seconds)
        scan_duration = 15.0  # seconds
        logger.info(f"Scanning will run for {scan_duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < scan_duration:
            # Check if user pressed ESC key
            if visualization_window is None:
                logger.info("Visualization window closed, stopping scan")
                break
            
            # Display stats periodically
            if int(time.time() - start_time) % 5 == 0:
                # Get latest statistics
                num_results = len(scanner.get_all_results())
                logger.info(f"Scanning in progress: {time.time() - start_time:.1f}s elapsed, "
                           f"{num_results} results generated")
            
            time.sleep(0.1)
        
        # Stop the scanner
        logger.info("Stopping continuous scanning...")
        scanner.stop()
        
        # Get and save the final result
        results = scanner.get_all_results()
        
        if results:
            logger.info(f"Scan completed with {len(results)} results")
            
            # Save the final result with mesh generation
            final_result = results[-1]
            
            # Use the raw folder from the scanner
            raw_folder = os.path.join(
                scanner.config.output_directory,
                f"raw_{int(scanner.start_time)}"
            )
            os.makedirs(raw_folder, exist_ok=True)
            base_filepath = os.path.join(raw_folder, "final_continuous_scan_result")
            
            point_cloud_path, mesh_path = save_point_cloud_and_mesh(final_result, base_filepath)
            
            if point_cloud_path:
                logger.info(f"Final continuous scan point cloud saved to {point_cloud_path}")
                
                if mesh_path:
                    logger.info(f"Final continuous scan mesh saved to {mesh_path}")
                    
                    # Visualize the mesh with Open3D if available
                    if OPEN3D_AVAILABLE and "--show-mesh" in sys.argv:
                        logger.info("Opening mesh visualization...")
                        mesh = o3d.io.read_triangle_mesh(mesh_path)
                        mesh.compute_vertex_normals()
                        o3d.visualization.draw_geometries([mesh])
        else:
            logger.warning("Continuous scan completed but no results available")
    
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        scanner.stop()
    
    except Exception as e:
        logger.error(f"Error during continuous scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        scanner.stop()
    
    finally:
        # Cleanup
        visualization_window = None
        vis_thread.join(timeout=1.0)
        
        logger.info("Continuous scanning completed")


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="UnLook Real-Time 3D Scanning Example")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--quality", choices=["low", "medium", "high", "ultra"], 
                       default="medium", help="Scanning quality")
    parser.add_argument("--show-mesh", action="store_true", 
                       help="Show the generated 3D mesh using Open3D visualization")
    
    args = parser.parse_args()
    
    # Connect to the UnLook scanner
    client = UnlookClient(client_name="RealTimeScanningDemo")
    
    # Map string to enum
    quality_map = {
        "low": ScanningQuality.LOW,
        "medium": ScanningQuality.MEDIUM,
        "high": ScanningQuality.HIGH,
        "ultra": ScanningQuality.ULTRA
    }
    quality = quality_map[args.quality]
    
    try:
        # Try to connect to any available scanner
        client.start_discovery()
        
        # Wait for scanners to be discovered
        logger.info("Waiting for scanners to be discovered...")
        time.sleep(3.0)
        
        # Get discovered scanners
        scanners = client.get_discovered_scanners()
        
        if not scanners:
            logger.error("No scanner found. Please make sure the UnLook scanner server is running.")
            return
        
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info.name} at {scanner_info.host}:{scanner_info.port}")
        
        if not client.connect(scanner_info):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Run the appropriate example
        if args.continuous:
            continuous_scan_example(client, quality)
        else:
            single_scan_example(client, quality)
        
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Always disconnect
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()