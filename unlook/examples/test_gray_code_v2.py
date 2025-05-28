#!/usr/bin/env python3
"""
Quick test for Gray code scanning with Protocol V2
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanner.scanner import UnlookClient

def test_gray_code_capture():
    """Test basic Gray code pattern capture with Protocol V2."""
    
    print("Gray Code Pattern Test with Protocol V2")
    print("=" * 50)
    
    # Create client
    client = UnlookClient(client_name="GrayCodeTest", auto_discover=True)
    
    # Wait for discovery
    print("Discovering scanners...")
    time.sleep(3)
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("❌ No scanners found!")
        return False
    
    print(f"✅ Found {len(scanners)} scanner(s)")
    
    # Connect
    if not client.connect(scanners[0]):
        print("❌ Failed to connect")
        return False
    
    print(f"✅ Connected to {scanners[0].name}")
    
    # Check Protocol V2
    if client.is_protocol_v2_enabled():
        print("✅ Protocol V2 enabled")
    else:
        print("⚠️ Protocol V2 NOT enabled")
    
    # Get cameras
    cameras = client.camera.get_cameras()
    print(f"✅ Found {len(cameras)} cameras")
    
    if len(cameras) < 2:
        print("❌ Need at least 2 cameras")
        return False
    
    camera_ids = [cam['id'] for cam in cameras[:2]]
    
    # Create output directory
    output_dir = Path("test_gray_code_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test patterns
    patterns = [
        # Reference white
        {
            "pattern_type": "solid_field",
            "color": "White",
            "name": "reference_white"
        },
        # Reference black
        {
            "pattern_type": "solid_field",
            "color": "Black", 
            "name": "reference_black"
        },
        # Blue vertical stripes (simulating Gray code bit 0)
        {
            "pattern_type": "vertical_lines",
            "foreground_color": "Blue",
            "background_color": "Black",
            "foreground_width": 128,  # Wide stripes for bit 0
            "background_width": 128,
            "name": "gray_code_bit_0"
        },
        # Inverted blue stripes
        {
            "pattern_type": "vertical_lines",
            "foreground_color": "Black",
            "background_color": "Blue",
            "foreground_width": 128,
            "background_width": 128,
            "name": "gray_code_bit_0_inv"
        },
        # Narrower stripes (bit 1)
        {
            "pattern_type": "vertical_lines",
            "foreground_color": "Blue",
            "background_color": "Black",
            "foreground_width": 64,
            "background_width": 64,
            "name": "gray_code_bit_1"
        }
    ]
    
    print(f"\nCapturing {len(patterns)} patterns...")
    
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}/{len(patterns)}: {pattern['name']}")
        
        # Project pattern based on type
        pattern_type = pattern.get("pattern_type")
        success = False
        
        try:
            if pattern_type == "solid_field":
                success = client.projector.show_solid_field(pattern.get("color", "White"))
            elif pattern_type == "vertical_lines":
                success = client.projector.show_vertical_lines(
                    foreground_color=pattern.get("foreground_color", "White"),
                    background_color=pattern.get("background_color", "Black"),
                    foreground_width=pattern.get("foreground_width", 4),
                    background_width=pattern.get("background_width", 4)
                )
            else:
                print(f"  ⚠️ Unknown pattern type: {pattern_type}")
                continue
                
            if not success:
                print(f"  ❌ Failed to project pattern")
                continue
            else:
                print(f"  ✅ Pattern projected")
        except Exception as e:
            print(f"  ❌ Error projecting pattern: {e}")
            continue
        
        # Wait for stabilization
        time.sleep(0.2)
        
        # Capture multi-camera
        print("  Capturing synchronized images...")
        start_time = time.time()
        images = client.camera.capture_multi(camera_ids)
        capture_time = (time.time() - start_time) * 1000
        
        if images and len(images) == 2:
            print(f"  ✅ Captured in {capture_time:.1f}ms")
            
            # Save images
            for cam_id, image in images.items():
                filename = output_dir / f"{pattern['name']}_{cam_id}.jpg"
                cv2.imwrite(str(filename), image)
                print(f"  💾 Saved: {filename} ({image.shape})")
        else:
            print(f"  ❌ Failed to capture")
    
    # Turn off projector
    client.projector.show_solid_field("Black")
    
    # Get compression stats
    print("\nCompression Statistics:")
    stats = client.get_compression_stats()
    if stats:
        print(f"  Compression ratio: {stats.get('avg_compression_ratio', 0):.2f}x")
        print(f"  Bandwidth savings: {stats.get('bandwidth_savings_percent', 0):.1f}%")
        print(f"  Data processed: {stats.get('total_original_mb', 0):.1f}MB")
    
    # Cleanup
    client.disconnect()
    print("\n✅ Test completed!")
    print(f"Images saved to: {output_dir}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_gray_code_capture()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)