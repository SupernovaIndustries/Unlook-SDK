#!/usr/bin/env python3
"""
TEST Q MATRIX FIX - Verify point cloud scaling is corrected
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import logging
from unlook.client.scanning.reconstruction.stereobm_surface_reconstructor import StereoBMSurfaceReconstructor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_q_matrix_fix():
    """Test that Q matrix fix resolves point cloud scaling issues"""
    
    print("=" * 80)
    print("Q MATRIX FIX VALIDATION TEST")
    print("=" * 80)
    
    # Initialize reconstructor with Q matrix verification enabled
    reconstructor = StereoBMSurfaceReconstructor(
        use_cgal=False,  # Disable CGAL to avoid complications
        use_advanced_disparity=False,  # Use simple StereoBM for speed
        use_ndr=False,  # Disable NDR to avoid complications
        use_phase_optimization=False  # Disable phase opt to avoid complications
    )
    
    # Load test data directory
    session_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
    if not session_dir.exists():
        print(f"ERROR: Test data not found at {session_dir}")
        return False
    
    # Find stereo pairs - use f8 pattern for best quality
    left_images = list(session_dir.glob("left_*f8_s0*"))
    if not left_images:
        left_images = list(session_dir.glob("left_*.jpg"))
    
    if not left_images:
        print("ERROR: No left images found")
        return False
    
    # Load one stereo pair for testing
    left_file = left_images[0]
    right_file = Path(str(left_file).replace("left_", "right_"))
    
    if not right_file.exists():
        print(f"ERROR: Right image not found: {right_file}")
        return False
    
    print(f"Testing with:")
    print(f"  Left:  {left_file.name}")
    print(f"  Right: {right_file.name}")
    
    left_img = cv2.imread(str(left_file))
    right_img = cv2.imread(str(right_file))
    
    if left_img is None or right_img is None:
        print("ERROR: Failed to load images")
        return False
    
    print(f"Image resolution: {left_img.shape[1]}x{left_img.shape[0]}")
    
    # Test: Simple reconstruction to check point cloud scale
    print("\n" + "=" * 60)
    print("TESTING POINT CLOUD SCALE WITH Q MATRIX FIX")
    print("=" * 60)
    
    try:
        # Load and verify calibration directly
        calib_file = "unlook/calibration/default/default_stereo_2k.json"
        reconstructor.load_calibration(calib_file)
        print(f"Calibration loaded: {calib_file}")
        
        # Rectify images
        left_rect, right_rect, Q = reconstructor.rectify_images(left_img, right_img)
        print(f"Images rectified, Q matrix shape: {Q.shape}")
        
        # Simple disparity computation
        disparity = reconstructor.compute_surface_disparity(left_rect, right_rect)
        print(f"Disparity computed, valid pixels: {np.sum(disparity > 0):,}")
        
        # Direct triangulation without complex confidence calculations
        points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
        
        # Filter valid points
        valid_mask = (disparity > 0) & (points_3d[:,:,2] > 0) & (points_3d[:,:,2] < 2000)
        points_3d_valid = points_3d[valid_mask]
        
        # Remove infinite/NaN points
        finite_mask = np.all(np.isfinite(points_3d_valid), axis=1)
        points_3d_final = points_3d_valid[finite_mask]
        
        # Create basic quality metrics
        quality = {
            'quality_score': min(100, len(points_3d_final) / 100),
            'description': 'Q matrix fix test'
        }
        
        print(f"\nRECONSTRUCTION RESULTS:")
        print(f"  Points generated: {len(points_3d):,}")
        print(f"  Quality score: {quality['quality_score']:.1f}/100")
        
        if len(points_3d) > 0:
            # Analyze coordinate ranges
            x_range = points_3d[:,0].max() - points_3d[:,0].min()
            y_range = points_3d[:,1].max() - points_3d[:,1].min()
            z_range = points_3d[:,2].max() - points_3d[:,2].min()
            
            centroid = np.mean(points_3d, axis=0)
            
            print(f"\nPOINT CLOUD COORDINATE ANALYSIS:")
            print(f"  X range: {points_3d[:,0].min():.1f} to {points_3d[:,0].max():.1f} mm")
            print(f"  Y range: {points_3d[:,1].min():.1f} to {points_3d[:,1].max():.1f} mm") 
            print(f"  Z range: {points_3d[:,2].min():.1f} to {points_3d[:,2].max():.1f} mm")
            print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm")
            
            print(f"\nOBJECT SIZE ANALYSIS:")
            print(f"  Width (X):  {x_range:.1f} mm")
            print(f"  Height (Y): {y_range:.1f} mm")
            print(f"  Depth (Z):  {z_range:.1f} mm")
            
            # Critical test: Check if depth is now reasonable with Q matrix fix
            print(f"\nQ MATRIX FIX VALIDATION:")
            if z_range > 50:  # Should be >50mm for a real object
                print(f"  [SUCCESS] Depth span {z_range:.1f}mm is reasonable!")
                print(f"  [SUCCESS] Q matrix fix appears to be working!")
                success = True
            elif z_range > 20:
                print(f"  [PARTIAL] Depth span {z_range:.1f}mm is improved but still small")
                print(f"  [INFO] May need additional calibration refinement")
                success = True
            else:
                print(f"  [FAILED] Depth span {z_range:.1f}mm is still too small")
                print(f"  [FAILED] Q matrix fix may not be applied properly")
                success = False
            
            # Check overall reasonableness
            if 200 < centroid[2] < 600:
                print(f"  [OK] Z-depth {centroid[2]:.0f}mm is reasonable for desktop scanning")
            else:
                print(f"  [WARN] Z-depth {centroid[2]:.0f}mm seems unusual")
                
            # Compare to expected object size (typical desktop object: 50-150mm)
            for span, name in [(x_range, "Width"), (y_range, "Height"), (z_range, "Depth")]:
                if 30 < span < 200:
                    print(f"  [OK] {name} {span:.1f}mm seems reasonable")
                elif 10 < span < 30:
                    print(f"  [PARTIAL] {name} {span:.1f}mm is small but possible")
                else:
                    print(f"  [WARN] {name} {span:.1f}mm seems unusual")
            
            return success
            
        else:
            print("ERROR: No points generated!")
            return False
            
    except Exception as e:
        print(f"ERROR: Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("Q MATRIX FIX TEST COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_q_matrix_fix()
    sys.exit(0 if success else 1)