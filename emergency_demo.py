#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("EMERGENCY SINGLE FRAME TEST...")
    
    try:
        from unlook.client.scanning.reconstruction.stereobm_surface_reconstructor import StereoBMSurfaceReconstructor
        
        # Create reconstructor with optimizations
        reconstructor = StereoBMSurfaceReconstructor(
            use_cgal=True,
            use_advanced_disparity=True,
            use_elas=True,
            use_confidence_filtering=True,
            use_enhanced_subpixel=True
        )
        
        print("Reconstructor created")
        
        # Load test images
        input_dir = Path("unlook/examples/scanning/captured_data/test1_2k/20250603_201954")
        left_file = input_dir / "left_003_phase_shift_f1_s1.jpg"
        right_file = input_dir / "right_003_phase_shift_f1_s1.jpg"
        
        print(f"Loading: {left_file.name}")
        
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))
        
        if left_img is None or right_img is None:
            print("ERROR: Cannot load images")
            return False
        
        print("Running reconstruction...")
        points_3d, quality = reconstructor.reconstruct_surface(
            left_img, right_img
        )
        
        num_points = len(points_3d)
        quality_score = quality['quality_score']
        
        print("="*50)
        print("RESULTS:")
        print(f"Points: {num_points:,}")
        print(f"Quality: {quality_score:.1f}/100")
        print(f"Description: {quality['description']}")
        print("="*50)
        
        if num_points > 50000:
            # Save result
            output_dir = Path("emergency_demo_result")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / "demo.ply"
            
            with open(output_file, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {num_points}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                
                for point in points_3d:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            print(f"SUCCESS! Saved to: {output_file}")
            print("DEMO READY FOR INVESTOR!")
            return True
        else:
            print("Too few points")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)