#!/usr/bin/env python3
"""
EMERGENCY DEBUG - Controlliamo tutto step by step
"""

import sys
from pathlib import Path
import cv2
import json
import numpy as np

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def check_input_data():
    """Controlla i dati di input."""
    
    print("🔍 CHECKING INPUT DATA...")
    
    input_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
    
    if not input_dir.exists():
        print(f"❌ Input directory NOT found: {input_dir}")
        return False
    
    print(f"✅ Input directory found: {input_dir}")
    
    # Check stereo pairs
    left_files = list(input_dir.glob("left_*.jpg"))
    right_files = list(input_dir.glob("right_*.jpg"))
    
    print(f"📸 Left images: {len(left_files)}")
    print(f"📸 Right images: {len(right_files)}")
    
    # List first few images
    for i, left_file in enumerate(left_files[:5]):
        right_file = Path(str(left_file).replace("left_", "right_"))
        status = "✅" if right_file.exists() else "❌"
        print(f"  {i+1}: {left_file.name} <-> {right_file.name} {status}")
    
    # Check image content
    if left_files:
        test_left = cv2.imread(str(left_files[0]))
        test_right = cv2.imread(str(left_files[0].name.replace("left_", "right_")))
        
        if test_left is not None:
            print(f"📏 Image resolution: {test_left.shape[1]}x{test_left.shape[0]}")
            
            # Check if images are different (not identical)
            if test_right is not None:
                diff = cv2.absdiff(test_left, test_right)
                diff_mean = np.mean(diff)
                print(f"🔄 Stereo difference: {diff_mean:.2f} (should be > 10)")
                
                if diff_mean < 5:
                    print("⚠️  WARNING: Images might be identical!")
                else:
                    print("✅ Images look different (good for stereo)")
    
    return len(left_files) > 0

def check_calibration():
    """Controlla i file di calibrazione."""
    
    print("\n🔧 CHECKING CALIBRATION...")
    
    # Check for calibration files
    cal_paths = [
        "calibration_2k.json",
        "unlook/calibration/2k_calibration_config.json",
        "unlook/calibration/custom/stereo_calibration.json",
        "unlook/calibration/default/default_stereo_2k.json"
    ]
    
    found_calibration = False
    for cal_path in cal_paths:
        if Path(cal_path).exists():
            print(f"✅ Found calibration: {cal_path}")
            found_calibration = True
            
            # Check calibration content
            try:
                with open(cal_path, 'r') as f:
                    cal_data = json.load(f)
                
                # Check for essential parameters
                if 'camera_matrix_left' in cal_data:
                    cam_left = cal_data['camera_matrix_left']
                    print(f"📷 Left camera fx: {cam_left[0][0]:.1f}")
                    print(f"📷 Left camera fy: {cam_left[1][1]:.1f}")
                
                if 'translation' in cal_data:
                    T = cal_data['translation']
                    baseline = np.sqrt(T[0]**2 + T[1]**2 + T[2]**2)
                    print(f"📏 Baseline: {baseline:.1f}mm")
                    
                    if baseline < 50:
                        print("⚠️  WARNING: Baseline molto piccolo!")
                    elif baseline > 200:
                        print("⚠️  WARNING: Baseline molto grande!")
                    else:
                        print("✅ Baseline sembra OK")
                        
            except Exception as e:
                print(f"❌ Error reading calibration: {e}")
            
            break
    
    if not found_calibration:
        print("❌ NO CALIBRATION FOUND!")
        
    return found_calibration

def test_basic_stereo():
    """Test stereo basic senza ottimizzazioni."""
    
    print("\n🧪 TESTING BASIC STEREO...")
    
    try:
        from unlook.client.scanning.reconstruction.stereobm_surface_reconstructor import StereoBMSurfaceReconstructor
        
        # Reconstructor SEMPLICISSIMO
        reconstructor = StereoBMSurfaceReconstructor(
            use_cgal=False,
            use_advanced_disparity=False,
            use_ndr=False,
            use_phase_optimization=False,
            use_elas=False,
            use_confidence_filtering=False,
            use_enhanced_subpixel=False
        )
        
        print("✅ Reconstructor created")
        
        # Test con una coppia di immagini
        input_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
        left_files = list(input_dir.glob("left_*.jpg"))
        
        if left_files:
            # Prova con la PRIMA immagine (più probabile che sia buona)
            for left_file in left_files:
                right_file = Path(str(left_file).replace("left_", "right_"))
                
                if right_file.exists():
                    print(f"🔬 Testing: {left_file.name}")
                    
                    left_img = cv2.imread(str(left_file))
                    right_img = cv2.imread(str(right_file))
                    
                    if left_img is not None and right_img is not None:
                        try:
                            # Test reconstruction
                            points_3d, quality = reconstructor.reconstruct_surface(
                                left_img, right_img, debug_output_dir="debug_test"
                            )
                            
                            print(f"📊 Result: {len(points_3d):,} points")
                            print(f"📊 Quality: {quality['quality_score']:.1f}/100")
                            print(f"📊 Description: {quality['description']}")
                            
                            if len(points_3d) > 100:
                                print("✅ Got some points - this pair works!")
                                
                                # Check point cloud bounds
                                if len(points_3d) > 0:
                                    x_range = [points_3d[:, 0].min(), points_3d[:, 0].max()]
                                    y_range = [points_3d[:, 1].min(), points_3d[:, 1].max()]
                                    z_range = [points_3d[:, 2].min(), points_3d[:, 2].max()]
                                    
                                    print(f"📏 X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] mm")
                                    print(f"📏 Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}] mm")
                                    print(f"📏 Z range: [{z_range[0]:.1f}, {z_range[1]:.1f}] mm")
                                    
                                    # Check if points make sense
                                    z_mean = points_3d[:, 2].mean()
                                    if z_mean < 100:
                                        print("⚠️  Points molto vicini alla camera")
                                    elif z_mean > 1000:
                                        print("⚠️  Points molto lontani dalla camera")
                                    else:
                                        print("✅ Distanza media sembra ragionevole")
                                
                                return True
                            else:
                                print(f"❌ Too few points: {len(points_3d)}")
                                
                        except Exception as e:
                            print(f"❌ Reconstruction failed: {e}")
                            continue
                    else:
                        print(f"❌ Failed to load images")
                        continue
        
        print("❌ No working image pairs found!")
        return False
        
    except Exception as e:
        print(f"❌ Import/setup error: {e}")
        return False

def check_debug_output():
    """Controlla se le immagini di debug vengono salvate."""
    
    print("\n💾 CHECKING DEBUG OUTPUT...")
    
    debug_dir = Path("debug_test")
    if debug_dir.exists():
        debug_files = list(debug_dir.glob("*"))
        print(f"📂 Debug files found: {len(debug_files)}")
        
        for file in debug_files:
            print(f"  📄 {file.name}")
            
        if len(debug_files) > 0:
            print("✅ Debug files are being saved!")
            return True
        else:
            print("❌ Debug directory empty!")
            return False
    else:
        print("❌ Debug directory not created!")
        return False

def main():
    """Run complete emergency debug."""
    
    print("🚨 EMERGENCY DEBUG SESSION")
    print("="*50)
    
    checks = [
        ("Input Data", check_input_data),
        ("Calibration", check_calibration), 
        ("Basic Stereo", test_basic_stereo),
        ("Debug Output", check_debug_output)
    ]
    
    results = {}
    
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results[name] = False
    
    print(f"\n{'='*50}")
    print("🏁 EMERGENCY DEBUG SUMMARY")
    print("="*50)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:15} : {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED - System should work!")
    else:
        print("\n💥 SOME CHECKS FAILED - Need fixing!")
        
        # Suggest fixes
        print("\n🔧 SUGGESTED FIXES:")
        if not results.get("Input Data", True):
            print("  - Check input directory path")
            print("  - Verify stereo image pairs exist")
        if not results.get("Calibration", True):
            print("  - Check calibration file exists")
            print("  - Verify calibration parameters")
        if not results.get("Basic Stereo", True):
            print("  - Try different image pairs")
            print("  - Check stereo parameters")
        if not results.get("Debug Output", True):
            print("  - Check write permissions")
            print("  - Verify debug directory creation")

if __name__ == "__main__":
    main()