#!/usr/bin/env python3
"""
EMERGENCY DEMO TEST - Simple reconstruction without complex optimizations
This MUST work for the demo!
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from unlook.client.scanning.reconstruction.stereobm_surface_reconstructor import StereoBMSurfaceReconstructor
import cv2
import numpy as np

def test_simple_reconstruction():
    """Test with minimal settings that MUST work."""
    
    print("\n" + "="*70)
    print("🚨 EMERGENCY DEMO TEST - MUST WORK!")
    print("="*70)
    
    # Input path
    input_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Find ALL stereo pairs
    left_files = list(input_dir.glob("left_*.jpg"))
    stereo_pairs = []
    
    for left_file in left_files:
        right_file = Path(str(left_file).replace("left_", "right_"))
        if right_file.exists():
            stereo_pairs.append((str(left_file), str(right_file)))
    
    print(f"Found {len(stereo_pairs)} stereo pairs")
    
    if len(stereo_pairs) == 0:
        print("❌ No stereo pairs found!")
        return False
    
    # Try EVERY pair until one works
    reconstructor = StereoBMSurfaceReconstructor(
        use_cgal=False,          # Disable for simplicity
        use_advanced_disparity=False,  # Use basic StereoBM
        use_ndr=False,           # Disable neural stuff
        use_phase_optimization=False,  # Disable phase opt
        use_elas=False,          # Disable ELAS
        use_confidence_filtering=False,  # Disable confidence
        use_enhanced_subpixel=False     # Disable subpixel
    )
    
    best_result = None
    best_quality = 0
    
    for i, (left_file, right_file) in enumerate(stereo_pairs):
        print(f"\n🔍 Testing pair {i+1}/{len(stereo_pairs)}: {Path(left_file).name}")
        
        try:
            # Load images
            left_img = cv2.imread(left_file)
            right_img = cv2.imread(right_file)
            
            if left_img is None or right_img is None:
                print(f"   ❌ Failed to load images")
                continue
            
            # Try reconstruction
            points_3d, quality = reconstructor.reconstruct_surface(
                left_img, right_img, debug_output_dir=None
            )
            
            num_points = len(points_3d) if points_3d is not None else 0
            quality_score = quality.get('quality_score', 0)
            
            print(f"   📊 Result: {num_points} points, quality: {quality_score:.1f}/100")
            
            if num_points > best_quality:
                best_result = (left_file, right_file, points_3d, quality)
                best_quality = num_points
                print(f"   ✅ New best result!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Report best result
    if best_result:
        left_file, right_file, points_3d, quality = best_result
        print(f"\n" + "="*70)
        print(f"🏆 BEST RESULT:")
        print(f"   Pair: {Path(left_file).name} <-> {Path(right_file).name}")
        print(f"   Points: {len(points_3d):,}")
        print(f"   Quality: {quality['quality_score']:.1f}/100")
        print(f"   Status: {quality['description']}")
        print("="*70)
        
        # Save result
        output_file = "emergency_test_result.ply"
        if reconstructor.save_point_cloud(points_3d, output_file):
            print(f"✅ Saved to: {output_file}")
        
        return True
    else:
        print(f"\n❌ NO SUCCESSFUL RECONSTRUCTIONS!")
        return False

if __name__ == "__main__":
    success = test_simple_reconstruction()
    sys.exit(0 if success else 1)