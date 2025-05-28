#!/usr/bin/env python3
"""
Test projector patterns directly to debug issues
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlook.client.scanner.scanner import UnlookClient

def test_projector():
    """Test projector pattern control."""
    
    print("Projector Pattern Test")
    print("=" * 50)
    
    # Create client
    client = UnlookClient(client_name="ProjectorTest", auto_discover=True)
    
    # Wait for discovery
    print("Discovering scanners...")
    time.sleep(3)
    
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("❌ No scanners found!")
        return False
    
    # Connect
    if not client.connect(scanners[0]):
        print("❌ Failed to connect")
        return False
    
    print(f"✅ Connected to {scanners[0].name}")
    
    # Test basic patterns
    tests = [
        ("White solid", lambda: client.projector.show_solid_field("White")),
        ("Black solid", lambda: client.projector.show_solid_field("Black")),
        ("Blue solid", lambda: client.projector.show_solid_field("Blue")),
        ("White lines", lambda: client.projector.show_vertical_lines("White", "Black", 10, 10)),
        ("Blue lines", lambda: client.projector.show_vertical_lines("Blue", "Black", 10, 10)),
        ("Wide blue lines", lambda: client.projector.show_vertical_lines("Blue", "Black", 50, 50)),
        ("Grid", lambda: client.projector.show_grid("Blue", "Black", 20, 20, 20, 20)),
        ("Checkerboard", lambda: client.projector.show_checkerboard("Blue", "Black", 8, 6)),
    ]
    
    for name, func in tests:
        print(f"\nTesting: {name}")
        try:
            success = func()
            if success:
                print(f"  ✅ {name} - Success")
                time.sleep(1)  # Show pattern for 1 second
            else:
                print(f"  ❌ {name} - Failed")
        except Exception as e:
            print(f"  ❌ {name} - Error: {e}")
    
    # Turn off
    print("\nTurning off projector...")
    client.projector.show_solid_field("Black")
    
    # Test pattern message directly
    print("\nTesting direct pattern message...")
    from unlook.core.protocol import MessageType
    
    success, response, _ = client.send_message(
        MessageType.PROJECTOR_PATTERN,
        {
            "pattern_type": "vertical_lines",
            "foreground_color": "Blue",
            "background_color": "Black",
            "foreground_width": 30,
            "background_width": 30
        }
    )
    
    if success:
        print("✅ Direct pattern message successful")
    else:
        print(f"❌ Direct pattern message failed: {response}")
    
    time.sleep(1)
    
    # Cleanup
    client.projector.show_solid_field("Black")
    client.disconnect()
    
    print("\n✅ Test completed!")
    return True


if __name__ == "__main__":
    try:
        success = test_projector()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)