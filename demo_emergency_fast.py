#!/usr/bin/env python3
"""
DEMO EMERGENCY - FAST reconstruction for demo (max 2 minutes)
"""

import subprocess
import sys
import time

def run_emergency_demo():
    """Demo velocissimo - deve finire in max 2 minuti!"""
    
    print("🚨 DEMO EMERGENCY MODE - FAST RECONSTRUCTION!")
    print("Target: Complete in under 2 minutes")
    
    start_time = time.time()
    
    # VELOCISSIMO: Single frame, no fusion, basic algorithms
    cmd = [
        "python", "unlook/examples/scanning/process_offline.py",
        "--input", "unlook/examples/scanning/captured_data/test1_2k/20250603_201954",
        "--surface-reconstruction",
        "--single-frame",  # NO multi-frame fusion (10x faster)
        "--no-elas",      # NO ELAS (use basic StereoBM)
        "--no-confidence-filtering",  # NO confidence filtering
        "--no-enhanced-subpixel",     # NO sub-pixel refinement
        "--no-cgal",      # NO CGAL
        "--save-visualizations"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("⏱️  Starting emergency reconstruction...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Completed in {duration:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ SUCCESS!")
            # Extract points from output
            output = result.stdout
            if "Points Generated:" in output:
                lines = output.split('\n')
                for line in lines:
                    if "Points Generated:" in line:
                        print(f"🎯 {line}")
                    elif "Quality Score:" in line:
                        print(f"📊 {line}")
                    elif "Output File:" in line:
                        print(f"💾 {line}")
            
            return True
        else:
            print("❌ FAILED!")
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT dopo 2 minuti!")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = run_emergency_demo()
    if success:
        print("\n🎉 DEMO READY - usa single frame mode!")
    else:
        print("\n💥 DEMO FAILED - need debug")
    sys.exit(0 if success else 1)