#!/usr/bin/env python3
"""
Example script demonstrating how to use pattern sequences with the UnLook projector.

This example shows how to:
1. Create and start a simple pattern sequence
2. Use event callbacks to monitor sequence progress
3. Create a structured light sequence for 3D scanning
4. Manually step through a pattern sequence

Usage:
    python pattern_sequence_example.py
"""

import time
import logging
import sys
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, '..')

from unlook import UnlookClient


def simple_pattern_sequence_example(client: UnlookClient):
    """Demonstrate a simple pattern sequence with callbacks."""
    logger.info("=== Simple Pattern Sequence Example ===")
    
    # Get a reference to the projector client
    projector = client.projector
    
    # Register for pattern sequence events
    def on_pattern_changed(data):
        logger.info(f"Pattern changed: {data.get('pattern_type', 'unknown')}, index: {data.get('index', -1)}")
    
    def on_sequence_started(data):
        logger.info(f"Sequence started: {data.get('sequence_length', 0)} patterns")
    
    def on_sequence_completed(data):
        logger.info(f"Sequence completed: {data.get('sequence_length', 0)} patterns")
    
    # Register callbacks
    projector.on_pattern_changed(on_pattern_changed)
    projector.on_sequence_started(on_sequence_started)
    projector.on_sequence_completed(on_sequence_completed)
    
    # Create a sequence of patterns
    patterns = [
        {"pattern_type": "solid_field", "color": "White"},
        {"pattern_type": "solid_field", "color": "Red"},
        {"pattern_type": "solid_field", "color": "Green"},
        {"pattern_type": "solid_field", "color": "Blue"},
        {"pattern_type": "horizontal_lines", "foreground_color": "White", 
         "background_color": "Black", "foreground_width": 4, "background_width": 20},
        {"pattern_type": "vertical_lines", "foreground_color": "White", 
         "background_color": "Black", "foreground_width": 4, "background_width": 20},
        {"pattern_type": "grid", "foreground_color": "White", "background_color": "Black", 
         "h_foreground_width": 4, "h_background_width": 20, 
         "v_foreground_width": 4, "v_background_width": 20},
        {"pattern_type": "checkerboard", "foreground_color": "White", "background_color": "Black", 
         "horizontal_count": 8, "vertical_count": 6}
    ]
    
    # Start the sequence
    logger.info("Starting pattern sequence")
    result = projector.start_pattern_sequence(
        patterns=patterns,
        interval=1.0,  # 1 second between patterns
        loop=False,    # Don't loop
        sync_with_camera=False
    )
    
    if result:
        logger.info(f"Sequence started: {result}")
        
        # Wait for the sequence to complete
        sequence_time = (len(patterns) * 1.0) + 1.0  # Add a little buffer
        logger.info(f"Waiting {sequence_time} seconds for sequence to complete...")
        time.sleep(sequence_time)
        
        # Check if sequence is still active
        if projector.sequence_active:
            logger.info("Sequence still active, stopping it")
            projector.stop_pattern_sequence(final_pattern={"pattern_type": "solid_field", "color": "Black"})
    else:
        logger.error("Failed to start pattern sequence")


def structured_light_sequence_example(client: UnlookClient):
    """Demonstrate a structured light pattern sequence for 3D scanning."""
    logger.info("\n=== Structured Light Pattern Sequence Example ===")
    
    # Get a reference to the projector client
    projector = client.projector
    
    # Create a structured light sequence (8 phase shifts)
    patterns = projector.create_structured_light_sequence(
        base_pattern_type="horizontal_lines",
        steps=8,
        foreground_width=4,
        background_width=4
    )
    
    logger.info(f"Created structured light sequence with {len(patterns)} patterns")
    
    # Start the sequence
    result = projector.start_pattern_sequence(
        patterns=patterns,
        interval=0.5,  # 0.5 seconds between patterns
        loop=True,     # Loop continuously
        sync_with_camera=True  # Synchronize with camera captures
    )
    
    if result:
        logger.info(f"Structured light sequence started: {result}")
        
        # Wait for a few cycles
        cycle_time = len(patterns) * 0.5 * 2  # 2 cycles
        logger.info(f"Running sequence for {cycle_time} seconds...")
        time.sleep(cycle_time)
        
        # Stop the sequence
        projector.stop_pattern_sequence(final_pattern={"pattern_type": "solid_field", "color": "White"})
        logger.info("Structured light sequence stopped")
    else:
        logger.error("Failed to start structured light sequence")


def manual_stepping_example(client: UnlookClient):
    """Demonstrate manual stepping through a pattern sequence."""
    logger.info("\n=== Manual Pattern Stepping Example ===")
    
    # Get a reference to the projector client
    projector = client.projector
    
    # Create a sequence of patterns
    patterns = [
        {"pattern_type": "solid_field", "color": "White"},
        {"pattern_type": "solid_field", "color": "Red"},
        {"pattern_type": "solid_field", "color": "Green"},
        {"pattern_type": "solid_field", "color": "Blue"}
    ]
    
    # Create the sequence but don't start it automatically
    result = projector.start_pattern_sequence(
        patterns=patterns,
        start_immediately=False
    )
    
    if result:
        logger.info("Sequence defined but not started")
        
        # Step through each pattern manually
        for i in range(len(patterns)):
            logger.info(f"Stepping to pattern {i}...")
            step_result = projector.step_pattern_sequence()
            if step_result:
                logger.info(f"Step result: {step_result}")
                time.sleep(1.0)  # Wait between steps
            else:
                logger.error(f"Failed to step to pattern {i}")
        
        # Finish with black
        projector.show_solid_field("Black")
        logger.info("Manual stepping completed")
    else:
        logger.error("Failed to define pattern sequence")


def main():
    """Main function to run the example."""
    # Connect to the UnLook scanner
    client = UnlookClient()
    
    try:
        # Try to connect to any available scanner
        scanners = client.discover_scanners(timeout=2.0)
        if not scanners:
            logger.error("No scanner found. Please make sure the UnLook scanner server is running.")
            return
        
        scanner_info = scanners[0]
        logger.info(f"Connecting to scanner: {scanner_info['name']} at {scanner_info['address']}:{scanner_info['port']}")
        
        if not client.connect(scanner_info["address"], scanner_info["port"]):
            logger.error("Failed to connect to scanner")
            return
        
        logger.info("Connected to scanner")
        
        # Run the examples
        simple_pattern_sequence_example(client)
        structured_light_sequence_example(client)
        manual_stepping_example(client)
        
        # Final cleanup
        client.projector.set_standby()
        logger.info("Examples completed")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Always disconnect
        client.disconnect()
        logger.info("Disconnected from scanner")


if __name__ == "__main__":
    main()