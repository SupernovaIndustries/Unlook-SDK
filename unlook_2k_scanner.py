#!/usr/bin/env python3
"""
UnLook 2K High-Quality Scanner
Script ottimizzato per scansioni 2K ad alta qualità
"""

import os
import sys
from pathlib import Path

# Set 2K environment variables
os.environ.update({
    'UNLOOK_CAMERA_WIDTH': '2048',
    'UNLOOK_CAMERA_HEIGHT': '1536',
    'UNLOOK_CAMERA_FPS': '15',
    'UNLOOK_CAMERA_QUALITY': '95',
    'UNLOOK_PATTERN_WIDTH': '2048',
    'UNLOOK_PATTERN_HEIGHT': '1536',
    'UNLOOK_QUALITY_PRESET': 'ultra',
    'UNLOOK_REALTIME_MODE': 'false'
})

# Import after setting environment
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_all_images_centered import AllImagesCenteredProcessor

def main():
    """Run 2K scanning"""
    print("🔥 UNLOOK 2K HIGH-QUALITY SCANNER")
    print("="*50)
    print("Configuration:")
    print("  Resolution: 2048x1536 (2K)")
    print("  FPS: 15 (optimized for quality)")
    print("  JPEG Quality: 95%")
    print("  Pattern Resolution: 2K")
    print("  Quality Preset: Ultra")
    print("="*50)
    
    # Allow custom directory
    if len(sys.argv) > 1:
        capture_dir = sys.argv[1]
    else:
        # Look for most recent capture directory
        capture_dirs = list(Path("captured_data").glob("*"))
        if capture_dirs:
            capture_dir = str(max(capture_dirs, key=os.path.getmtime))
            print(f"Using most recent capture: {capture_dir}")
        else:
            print("❌ No capture directory found!")
            print("Usage: python unlook_2k_scanner.py [capture_directory]")
            return
    
    if not Path(capture_dir).exists():
        print(f"❌ Capture directory not found: {capture_dir}")
        return
    
    # Process with 2K-optimized settings
    processor = AllImagesCenteredProcessor(capture_dir)
    
    # Override with 2K-specific settings
    processor.output_dir = Path(capture_dir) / "2k_results"
    processor.output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Processing {capture_dir} with 2K configuration...")
    
    success = processor.process_all_images()
    
    if success:
        print("\n🎉 2K SCAN COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {processor.output_dir}")
        print("📊 2K provides:")
        print("  - 4x more pixels than 1280x720")
        print("  - Enhanced detail capture")
        print("  - Better feature matching")
        print("  - Higher precision reconstruction")
        print("\n🔍 Open FINAL_COMBINED_SCAN.ply in MeshLab for best viewing")
    else:
        print("\n❌ 2K scan failed. Check logs for details.")

if __name__ == "__main__":
    main()
