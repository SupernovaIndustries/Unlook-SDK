#!/usr/bin/env python3
"""
Projector focus test - cycles through Gray code patterns
Press 'q' to quit
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanner.scanner import UnlookClient

def create_gray_code_patterns(num_bits=4):
    """Create the same Gray code patterns as the scanner."""
    patterns = []
    
    # Reference patterns
    patterns.append({
        "pattern_type": "solid_field",
        "color": "White",
        "name": "reference_white",
        "delay": 1.0  # Show for 1 second
    })
    patterns.append({
        "pattern_type": "solid_field",
        "color": "Black",
        "name": "reference_black",
        "delay": 1.0
    })
    
    # Gray code patterns with blue stripes
    for bit in range(num_bits):
        # Calculate stripe width for this bit
        stripe_width = 2 ** (num_bits - bit - 1) * 10  # Scale up for visibility
        
        # Normal pattern (blue on black)
        patterns.append({
            "pattern_type": "vertical_lines",
            "foreground_color": "Blue",
            "background_color": "Black",
            "foreground_width": stripe_width,
            "background_width": stripe_width,
            "name": f"gray_code_bit_{bit} (width={stripe_width})",
            "delay": 2.0  # Show for 2 seconds
        })
        
        # Inverted pattern (black on blue)
        patterns.append({
            "pattern_type": "vertical_lines",
            "foreground_color": "Black",
            "background_color": "Blue",
            "foreground_width": stripe_width,
            "background_width": stripe_width,
            "name": f"gray_code_bit_{bit}_inv (width={stripe_width})",
            "delay": 2.0
        })
    
    return patterns

def test_projector_focus():
    """Test projector focus with Gray code patterns."""
    
    print("Projector Focus Test")
    print("=" * 50)
    print("This will cycle through Gray code patterns")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Create client
    client = UnlookClient(client_name="FocusTest", auto_discover=True)
    
    # Wait for discovery
    print("\nDiscovering scanners...")
    time.sleep(2)
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("❌ No scanners found! Make sure server is running with:")
        print("   python unlook/server_bootstrap.py --enable-protocol-v2 --enable-pattern-preprocessing --enable-sync")
        return False
    
    print(f"✅ Found {len(scanners)} scanner(s)")
    
    # Connect
    scanner = scanners[0]
    success = client.connect(scanner)
    
    if not success:
        print("❌ Failed to connect to scanner")
        return False
    
    print(f"✅ Connected to scanner: {scanner.name}")
    
    # Get patterns
    patterns = create_gray_code_patterns(num_bits=4)
    
    print(f"\nCycling through {len(patterns)} patterns...")
    print("Watch the projector output and adjust focus as needed")
    print("\nPattern schedule:")
    for i, p in enumerate(patterns):
        print(f"  {i+1}. {p['name']} - {p['delay']}s")
    
    # Create a small OpenCV window for capturing key press
    cv2.namedWindow("Focus Test Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Focus Test Control", 400, 100)
    
    # Create a black image with text
    control_img = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(control_img, "Press 'q' to quit", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    pattern_index = 0
    last_pattern_time = time.time()
    running = True
    
    try:
        while running:
            # Show control window
            cv2.imshow("Focus Test Control", control_img)
            
            # Get current pattern
            current_pattern = patterns[pattern_index]
            
            # Check if it's time to change pattern
            if time.time() - last_pattern_time >= current_pattern['delay']:
                # Move to next pattern
                pattern_index = (pattern_index + 1) % len(patterns)
                current_pattern = patterns[pattern_index]
                
                print(f"\n➤ Pattern: {current_pattern['name']}")
                
                # Project pattern
                pattern_type = current_pattern['pattern_type']
                try:
                    if pattern_type == "solid_field":
                        success = client.projector.show_solid_field(current_pattern['color'])
                    elif pattern_type == "vertical_lines":
                        success = client.projector.show_vertical_lines(
                            foreground_color=current_pattern['foreground_color'],
                            background_color=current_pattern['background_color'],
                            foreground_width=current_pattern['foreground_width'],
                            background_width=current_pattern['background_width']
                        )
                    else:
                        success = False
                    
                    if not success:
                        print(f"  ⚠️ Failed to project pattern")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                
                last_pattern_time = time.time()
            
            # Check for key press
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("\n⏹️ Stopping...")
                running = False
            elif key == ord(' '):  # Space to pause
                print("\n⏸️ Paused - press space to continue")
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        print("▶️ Resuming...")
                        last_pattern_time = time.time()
                        break
                    elif key == ord('q'):
                        running = False
                        break
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        client.projector.show_solid_field("Black")
        client.disconnect()
    
    print("\n✅ Focus test completed!")
    return True


if __name__ == "__main__":
    try:
        success = test_projector_focus()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)