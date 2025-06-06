#!/usr/bin/env python3
"""
DEBUG CALIBRATION TEST
Direct test of calibration verification and debug saving methods
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

def test_calibration_diagnostics():
    """Test calibration verification and coordinate validation methods"""
    
    print("=" * 80)
    print("CALIBRATION DIAGNOSTIC TEST")
    print("=" * 80)
    
    # Initialize reconstructor
    reconstructor = StereoBMSurfaceReconstructor(
        use_cgal=True,
        use_advanced_disparity=True, 
        use_ndr=True,
        use_phase_optimization=True
    )
    
    # Load test data directory
    session_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
    if not session_dir.exists():
        print(f"ERROR: Test data not found at {session_dir}")
        return False
    
    # Find stereo pairs
    left_images = list(session_dir.glob("left_*phase_shift*f8_s0*"))
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
    
    print(f"Loading test images:")
    print(f"  Left:  {left_file.name}")
    print(f"  Right: {right_file.name}")
    
    left_img = cv2.imread(str(left_file))
    right_img = cv2.imread(str(right_file))
    
    if left_img is None or right_img is None:
        print("ERROR: Failed to load images")
        return False
    
    print(f"Image resolution: {left_img.shape[1]}x{left_img.shape[0]}")
    
    # Test 1: Calibration verification
    print("\n" + "=" * 60)
    print("TEST 1: CALIBRATION VERIFICATION")
    print("=" * 60)
    
    # Load calibration explicitly
    try:
        reconstructor.load_calibration()
        print(f"Calibration loaded from: {reconstructor.calibration_file}")
        
        # Run calibration verification
        reconstructor._verify_calibration_accuracy()
        
    except Exception as e:
        print(f"ERROR: Calibration verification failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Single reconstruction with all diagnostics
    print("\n" + "=" * 60)
    print("TEST 2: RECONSTRUCTION WITH FULL DIAGNOSTICS")
    print("=" * 60)
    
    debug_dir = session_dir / "diagnostic_test"
    debug_dir.mkdir(exist_ok=True)
    
    try:
        # Run single-frame reconstruction with full diagnostics
        points_3d, quality = reconstructor.reconstruct_surface(
            left_img, right_img, 
            debug_output_dir=str(debug_dir)
        )
        
        print(f"\nRECONSTRUCTION RESULTS:")
        print(f"  Points generated: {len(points_3d):,}")
        print(f"  Quality score: {quality['quality_score']:.1f}/100")
        
        if len(points_3d) > 0:
            # Test coordinate validation
            print("\n" + "-" * 40)
            print("COORDINATE VALIDATION:")
            print("-" * 40)
            reconstructor._validate_point_cloud_coordinates(points_3d, expected_depth=400)
            
            # Display coordinate ranges
            print(f"\nPOINT CLOUD COORDINATE ANALYSIS:")
            print(f"  X range: {points_3d[:,0].min():.1f} to {points_3d[:,0].max():.1f} mm")
            print(f"  Y range: {points_3d[:,1].min():.1f} to {points_3d[:,1].max():.1f} mm") 
            print(f"  Z range: {points_3d[:,2].min():.1f} to {points_3d[:,2].max():.1f} mm")
            
            centroid = np.mean(points_3d, axis=0)
            print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm")
            
            # Check if coordinates seem reasonable for a desktop object at ~400mm
            print(f"\nREASONABLENESS CHECK:")
            if 200 < centroid[2] < 600:
                print(f"  [OK] Z-depth seems reasonable for desktop scanning")
            else:
                print(f"  [WARN] Z-depth {centroid[2]:.0f}mm seems unusual")
                
            if abs(centroid[0]) < 200 and abs(centroid[1]) < 200:
                print(f"  [OK] X,Y coordinates seem reasonable for centered object")
            else:
                print(f"  [WARN] X,Y coordinates seem off-center: ({centroid[0]:.0f}, {centroid[1]:.0f})")
                
            # Check coordinate scales
            x_span = points_3d[:,0].max() - points_3d[:,0].min()
            y_span = points_3d[:,1].max() - points_3d[:,1].min()
            z_span = points_3d[:,2].max() - points_3d[:,2].min()
            
            print(f"\nOBJECT SIZE ANALYSIS:")
            print(f"  Width (X):  {x_span:.1f} mm")
            print(f"  Height (Y): {y_span:.1f} mm")
            print(f"  Depth (Z):  {z_span:.1f} mm")
            
            # Typical desktop object: 50-200mm in each dimension
            for span, name in [(x_span, "Width"), (y_span, "Height"), (z_span, "Depth")]:
                if 20 < span < 300:
                    print(f"  [OK] {name} {span:.1f}mm seems reasonable")
                else:
                    print(f"  [WARN] {name} {span:.1f}mm seems unusual")
        
        else:
            print("ERROR: No points generated!")
            
    except Exception as e:
        print(f"ERROR: Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check debug outputs
    print("\n" + "=" * 60)
    print("TEST 3: DEBUG OUTPUT VERIFICATION")
    print("=" * 60)
    
    debug_files = list(debug_dir.glob("*"))
    if debug_files:
        print(f"Debug files generated:")
        for file in sorted(debug_files):
            print(f"  [OK] {file.name}")
    else:
        print("[ERROR] No debug files generated!")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_calibration_diagnostics()
    sys.exit(0 if success else 1)