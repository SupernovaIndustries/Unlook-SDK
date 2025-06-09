#!/usr/bin/env python3
"""
Test veloce del fix per il broadcast error
"""

import sys
from pathlib import Path
import cv2

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_fix():
    """Test rapido del fix."""
    
    print("🔧 TESTING BROADCAST FIX...")
    
    try:
        from unlook.client.scanning.reconstruction.stereobm_surface_reconstructor import StereoBMSurfaceReconstructor
        
        # Reconstructor semplice
        reconstructor = StereoBMSurfaceReconstructor(
            use_cgal=False,
            use_advanced_disparity=False,
            use_ndr=False,
            use_phase_optimization=False,
            use_elas=False,
            use_confidence_filtering=False,
            use_enhanced_subpixel=False
        )
        
        # Test con una coppia di immagini
        input_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
        left_file = input_dir / "left_003_phase_shift_f1_s1.jpg"
        right_file = input_dir / "right_003_phase_shift_f1_s1.jpg"
        
        if left_file.exists() and right_file.exists():
            print(f"🔬 Testing: {left_file.name}")
            
            left_img = cv2.imread(str(left_file))
            right_img = cv2.imread(str(right_file))
            
            if left_img is not None and right_img is not None:
                print(f"📏 Input shapes: Left={left_img.shape}, Right={right_img.shape}")
                
                # Test reconstruction
                points_3d, quality = reconstructor.reconstruct_surface(
                    left_img, right_img, debug_output_dir="test_fix_debug"
                )
                
                print(f"✅ SUCCESS!")
                print(f"📊 Points: {len(points_3d):,}")
                print(f"📊 Quality: {quality['quality_score']:.1f}/100")
                print(f"📊 Description: {quality['description']}")
                
                if len(points_3d) > 100:
                    # Check bounds
                    x_range = [points_3d[:, 0].min(), points_3d[:, 0].max()]
                    y_range = [points_3d[:, 1].min(), points_3d[:, 1].max()]
                    z_range = [points_3d[:, 2].min(), points_3d[:, 2].max()]
                    
                    print(f"📏 X: [{x_range[0]:.1f}, {x_range[1]:.1f}] mm")
                    print(f"📏 Y: [{y_range[0]:.1f}, {y_range[1]:.1f}] mm")
                    print(f"📏 Z: [{z_range[0]:.1f}, {z_range[1]:.1f}] mm")
                    
                    return True
                else:
                    print(f"⚠️  Too few points but no crash!")
                    return True  # At least no crash
            else:
                print("❌ Failed to load images")
                return False
        else:
            print("❌ Test images not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fix()
    if success:
        print("\n🎉 BROADCAST ERROR FIXED!")
        print("Now try the full command again!")
    else:
        print("\n💥 Still broken - need more debugging")
    sys.exit(0 if success else 1)