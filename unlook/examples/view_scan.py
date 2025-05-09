#!/usr/bin/env python3
"""
Command-line tool for viewing and analyzing 3D scan results.

This tool allows quick visualization and analysis of point clouds and meshes
from scan results, making it easier to diagnose scanning issues.

Usage:
    python view_scan.py [scan_directory] [--raw] [--filter] [--info-only]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import optional dependencies
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("open3d not installed. Install with 'pip install open3d' for full functionality.")
    OPEN3D_AVAILABLE = False

try:
    from unlook.client.visualization import ScanVisualizer, visualize_scan_result
except ImportError as e:
    logger.error(f"Error importing visualization module: {e}")
    print("Make sure you're running this script from the UnLook-SDK directory structure.")
    sys.exit(1)


def find_scan_result_files(scan_dir: str) -> Dict[str, List[str]]:
    """
    Find point cloud and mesh files in a scan directory structure.
    
    Args:
        scan_dir: Scan directory path
        
    Returns:
        Dictionary with 'point_clouds' and 'meshes' lists of file paths
    """
    scan_dir = os.path.abspath(scan_dir)
    results = {
        'point_clouds': [],
        'meshes': []
    }
    
    if not os.path.isdir(scan_dir):
        logger.error(f"Not a directory: {scan_dir}")
        return results
    
    # Common locations for scan results
    potential_paths = [
        os.path.join(scan_dir, "results"),
        os.path.join(scan_dir, "debug"),
        scan_dir
    ]
    
    # File extensions
    point_cloud_exts = ['.ply', '.pcd', '.xyz', '.pts']
    mesh_exts = ['.obj', '.stl', '.off', '.gltf'] 
    
    # Walk directories
    for path in potential_paths:
        if not os.path.isdir(path):
            continue
            
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in point_cloud_exts:
                    results['point_clouds'].append(filepath)
                elif ext in mesh_exts:
                    results['meshes'].append(filepath)
    
    # Sort by modification time (newest first)
    for key in results:
        results[key].sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return results


def analyze_scan_directory(scan_dir: str) -> Dict[str, Any]:
    """
    Analyze a scan directory structure and extract relevant information.
    
    Args:
        scan_dir: Scan directory path
        
    Returns:
        Dictionary with scan analysis results
    """
    scan_dir = os.path.abspath(scan_dir)
    results = {
        'directory': scan_dir,
        'name': os.path.basename(scan_dir),
        'files': find_scan_result_files(scan_dir),
        'has_data': False,
        'scan_type': None,
        'has_debug': False,
        'image_pairs': 0
    }
    
    # Check if we have any scan data
    results['has_data'] = bool(results['files']['point_clouds'] or results['files']['meshes'])
    
    # Check for debug data
    debug_dir = os.path.join(scan_dir, "debug")
    results['has_debug'] = os.path.isdir(debug_dir)
    
    # Count image pairs
    captures_dir = os.path.join(scan_dir, "captures")
    if os.path.isdir(captures_dir):
        left_images = len([f for f in os.listdir(captures_dir) if f.startswith('left_') and f.endswith('.png')])
        right_images = len([f for f in os.listdir(captures_dir) if f.startswith('right_') and f.endswith('.png')])
        results['image_pairs'] = min(left_images, right_images)
        
        # If no left/right pairs, check for generic captures
        if results['image_pairs'] == 0:
            captures = len([f for f in os.listdir(captures_dir) if f.startswith('capture_') and f.endswith('.png')])
            results['image_pairs'] = captures
    
    # Try to infer scan type from directory name
    scan_name = results['name'].lower()
    if 'robust' in scan_name:
        results['scan_type'] = 'robust'
    elif 'enhanced' in scan_name:
        results['scan_type'] = 'enhanced'
    elif 'combined' in scan_name:
        results['scan_type'] = 'combined'
    elif 'phase_shift' in scan_name:
        results['scan_type'] = 'phase_shift'
    elif 'gray_code' in scan_name:
        results['scan_type'] = 'gray_code'
    elif 'single_camera' in scan_name:
        results['scan_type'] = 'single_camera'
    else:
        results['scan_type'] = 'unknown'
    
    return results


def print_scan_info(scan_info: Dict[str, Any]) -> None:
    """
    Print scan information in a human-readable format.
    
    Args:
        scan_info: Scan analysis from analyze_scan_directory
    """
    print(f"\n=== Scan Information for: {scan_info['name']} ===")
    print(f"Directory: {scan_info['directory']}")
    print(f"Scan type: {scan_info['scan_type']}")
    print(f"Has debug data: {'Yes' if scan_info['has_debug'] else 'No'}")
    print(f"Image pairs: {scan_info['image_pairs']}")
    
    if scan_info['files']['point_clouds']:
        print("\nPoint Clouds:")
        for i, pc in enumerate(scan_info['files']['point_clouds']):
            rel_path = os.path.relpath(pc, scan_info['directory'])
            print(f"  {i+1}. {rel_path}")
    else:
        print("\nNo point clouds found")
    
    if scan_info['files']['meshes']:
        print("\nMeshes:")
        for i, mesh in enumerate(scan_info['files']['meshes']):
            rel_path = os.path.relpath(mesh, scan_info['directory'])
            print(f"  {i+1}. {rel_path}")
    else:
        print("\nNo meshes found")


def analyze_scan_results(scan_info: Dict[str, Any], raw: bool = False) -> Dict[str, Any]:
    """
    Analyze scan results in more detail.
    
    Args:
        scan_info: Scan analysis from analyze_scan_directory
        raw: Whether to include raw point clouds in analysis
        
    Returns:
        Dictionary with detailed scan analysis
    """
    if not OPEN3D_AVAILABLE:
        print("open3d is required for detailed analysis. Install with 'pip install open3d'")
        return {
            'point_clouds': {},
            'meshes': {}
        }
    
    analysis = {
        'point_clouds': {},
        'meshes': {}
    }
    
    try:
        visualizer = ScanVisualizer(use_window=False)
        
        # Analyze point clouds
        point_clouds = scan_info['files']['point_clouds']
        if point_clouds:
            # Filter out temporary/debug point clouds unless raw=True
            filtered_pcs = point_clouds
            if not raw:
                filtered_pcs = [pc for pc in point_clouds if 
                            ('result' in pc.lower() or
                                'scan_point_cloud' in pc.lower()) and
                            not ('stage' in os.path.basename(pc).lower() or
                                'raw' in os.path.basename(pc).lower())]
                
                # If no results found, use any point cloud
                if not filtered_pcs and point_clouds:
                    filtered_pcs = [point_clouds[0]]
            
            # Analyze each point cloud
            for pc_path in filtered_pcs:
                try:
                    pc_name = os.path.relpath(pc_path, scan_info['directory'])
                    print(f"Analyzing point cloud: {pc_name}")
                    
                    pcd = visualizer.load_point_cloud(pc_path)
                    stats = visualizer.analyze_point_cloud(pcd)
                    analysis['point_clouds'][pc_name] = stats
                except Exception as e:
                    logger.error(f"Error analyzing point cloud {pc_path}: {e}")
                    analysis['point_clouds'][os.path.relpath(pc_path, scan_info['directory'])] = {
                        "error": str(e),
                        "points": 0,
                        "empty": True
                    }
        
        # Analyze meshes
        meshes = scan_info['files']['meshes']
        if meshes:
            for mesh_path in meshes:
                try:
                    mesh_name = os.path.relpath(mesh_path, scan_info['directory'])
                    print(f"Analyzing mesh: {mesh_name}")
                    
                    mesh = visualizer.load_mesh(mesh_path)
                    
                    analysis['meshes'][mesh_name] = {
                        'triangles': len(mesh.triangles),
                        'vertices': len(mesh.vertices),
                        'empty': len(mesh.triangles) == 0,
                        'has_normals': mesh.has_vertex_normals(),
                        'has_colors': mesh.has_vertex_colors(),
                        'watertight': mesh.is_watertight() if hasattr(mesh, 'is_watertight') else 'unknown'
                    }
                except Exception as e:
                    logger.error(f"Error analyzing mesh {mesh_path}: {e}")
                    analysis['meshes'][os.path.relpath(mesh_path, scan_info['directory'])] = {
                        "error": str(e),
                        "triangles": 0,
                        "vertices": 0,
                        "empty": True
                    }
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    
    return analysis


def print_detailed_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print detailed analysis of scan results.
    
    Args:
        analysis: Detailed analysis from analyze_scan_results
    """
    print("\n=== Detailed Scan Analysis ===")
    
    # Point clouds
    if analysis['point_clouds']:
        print("\nPoint Clouds:")
        for name, stats in analysis['point_clouds'].items():
            print(f"\n  {name}:")
            
            if "error" in stats:
                print(f"    Error analyzing point cloud: {stats['error']}")
                continue
                
            print(f"    Points: {stats['points']}")
            
            if not stats['empty']:
                if stats.get('dimension'):
                    print(f"    Dimensions: X={stats['dimension']['x']:.2f}, " + 
                         f"Y={stats['dimension']['y']:.2f}, " +
                         f"Z={stats['dimension']['z']:.2f}")
                
                if stats.get('density', {}).get('average_distance'):
                    print(f"    Average point distance: {stats['density']['average_distance']:.4f}")
                
                print(f"    Has normals: {stats['has_normals']}")
                print(f"    Has colors: {stats['has_colors']}")
    
    # Meshes
    if analysis['meshes']:
        print("\nMeshes:")
        for name, stats in analysis['meshes'].items():
            print(f"\n  {name}:")
            
            if "error" in stats:
                print(f"    Error analyzing mesh: {stats['error']}")
                continue
                
            print(f"    Triangles: {stats['triangles']}")
            print(f"    Vertices: {stats['vertices']}")
            print(f"    Watertight: {stats['watertight']}")
            print(f"    Has normals: {stats['has_normals']}")
            print(f"    Has colors: {stats['has_colors']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="View and analyze 3D scan results")
    parser.add_argument("scan_directory", nargs='?', default="./scans", 
                       help="Directory containing scan results (default: ./scans)")
    parser.add_argument("--raw", "-r", action="store_true", 
                       help="Include raw/temporary point clouds in analysis")
    parser.add_argument("--filter", "-f", action="store_true",
                       help="Apply filtering to point clouds before visualization")
    parser.add_argument("--info-only", "-i", action="store_true",
                       help="Show information only, without visualization")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all scan directories and exit")
    parser.add_argument("--latest", action="store_true",
                       help="Automatically select the latest scan")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode for detailed error information")
    args = parser.parse_args()
    
    # Set up debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # Convert scan directory to absolute path
        scan_dir_base = os.path.abspath(args.scan_directory)
        
        # If it's a parent directory containing scan results
        if os.path.isdir(scan_dir_base) and not any(os.path.isdir(os.path.join(scan_dir_base, d)) 
                                                   for d in ["results", "debug", "captures"]):
            # List all potential scan directories
            scan_dirs = []
            for d in os.listdir(scan_dir_base):
                full_path = os.path.join(scan_dir_base, d)
                if os.path.isdir(full_path) and any(os.path.isdir(os.path.join(full_path, subdir)) 
                                                for subdir in ["results", "debug", "captures"]):
                    scan_dirs.append(full_path)
            
            # Sort by modification time (newest first)
            scan_dirs.sort(key=os.path.getmtime, reverse=True)
            
            if args.list:
                print(f"Found {len(scan_dirs)} scan directories in {scan_dir_base}:")
                for i, sd in enumerate(scan_dirs):
                    scan_name = os.path.basename(sd)
                    try:
                        scan_time = os.path.getmtime(sd)
                        time_str = f"{scan_time:.0f}"
                    except:
                        time_str = "unknown time"
                    print(f"{i+1}. {scan_name} ({time_str})")
                return
                
            if args.latest and scan_dirs:
                scan_dir = scan_dirs[0]
                print(f"Selected latest scan: {os.path.basename(scan_dir)}")
            elif scan_dirs:
                # If multiple scans, let user select
                print(f"Found {len(scan_dirs)} scan directories. Select one:")
                for i, sd in enumerate(scan_dirs):
                    print(f"{i+1}. {os.path.basename(sd)}")
                
                try:
                    selection = int(input("Enter selection number (or 0 to cancel): "))
                    if selection == 0:
                        print("Cancelled")
                        return
                    if 1 <= selection <= len(scan_dirs):
                        scan_dir = scan_dirs[selection - 1]
                    else:
                        print("Invalid selection")
                        return
                except ValueError:
                    print("Invalid input")
                    return
            else:
                print(f"No scan directories found in {scan_dir_base}")
                return
        else:
            scan_dir = scan_dir_base
        
        # Make sure it's a valid scan directory
        if not os.path.isdir(scan_dir):
            print(f"Not a directory: {scan_dir}")
            return
        
        # Analyze the scan directory
        scan_info = analyze_scan_directory(scan_dir)
        print_scan_info(scan_info)
        
        if not scan_info['has_data']:
            print(f"No scan data found in {scan_dir}")
            return
        
        # Detailed analysis
        if not args.info_only:
            try:
                detailed_analysis = analyze_scan_results(scan_info, args.raw)
                print_detailed_analysis(detailed_analysis)
                
                # Visualization
                if not args.info_only and OPEN3D_AVAILABLE:
                    # Select a point cloud to visualize
                    if scan_info['files']['point_clouds']:
                        print("\nVisualizing point cloud...")
                        pc_to_vis = None
                        
                        if args.raw and scan_info['files']['point_clouds']:
                            pc_to_vis = scan_info['files']['point_clouds'][0]
                        else:
                            # Try to find the final, filtered point cloud
                            for pc in scan_info['files']['point_clouds']:
                                if 'scan_point_cloud' in pc.lower() and not 'raw' in os.path.basename(pc).lower():
                                    pc_to_vis = pc
                                    break
                            
                            # If not found, use the first one
                            if not pc_to_vis and scan_info['files']['point_clouds']:
                                pc_to_vis = scan_info['files']['point_clouds'][0]
                        
                        if pc_to_vis:
                            try:
                                visualize_scan_result(pc_to_vis)
                            except Exception as e:
                                logger.error(f"Error visualizing point cloud: {e}")
                                if args.debug:
                                    import traceback
                                    traceback.print_exc()
                    
                    # Visualize mesh if available
                    if scan_info['files']['meshes']:
                        print("\nVisualizing mesh...")
                        try:
                            visualize_scan_result(scan_info['files']['meshes'][0])
                        except Exception as e:
                            logger.error(f"Error visualizing mesh: {e}")
                            if args.debug:
                                import traceback
                                traceback.print_exc()
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        elif not OPEN3D_AVAILABLE:
            print("\nInstall open3d for visualization: pip install open3d")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()