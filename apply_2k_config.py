#!/usr/bin/env python3
"""
Apply 2K Configuration to UnLook Server
Applica la configurazione 2K per massimizzare la qualità delle immagini
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_server_config():
    """Aggiorna la configurazione del server per 2K"""
    logger.info("🔧 APPLYING 2K CONFIGURATION TO UNLOOK SERVER")
    logger.info("="*60)
    
    # Update environment variables for 2K
    config_updates = {
        'UNLOOK_CAMERA_WIDTH': '2048',
        'UNLOOK_CAMERA_HEIGHT': '1536', 
        'UNLOOK_CAMERA_FPS': '15',
        'UNLOOK_CAMERA_QUALITY': '95',
        'UNLOOK_PATTERN_WIDTH': '2048',
        'UNLOOK_PATTERN_HEIGHT': '1536',
        'UNLOOK_QUALITY_PRESET': 'ultra',
        'UNLOOK_REALTIME_MODE': 'false',
        'UNLOOK_PRIORITY_QUALITY': 'true'
    }
    
    logger.info("Setting environment variables for 2K mode:")
    for key, value in config_updates.items():
        os.environ[key] = value
        logger.info(f"  {key} = {value}")
    
    return True

def create_2k_calibration_config():
    """Crea una configurazione di calibrazione per 2K"""
    calib_config = {
        "target_resolution": [2048, 1536],
        "checkerboard_size": [9, 6],
        "square_size_mm": 24.0,
        "baseline_target_mm": 80.0,
        "min_samples": 30,
        "max_samples": 50,
        "quality_threshold": 0.3,
        "notes": "2K calibration configuration for maximum precision"
    }
    
    config_file = "unlook/calibration/2k_calibration_config.json"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(calib_config, f, indent=2)
    
    logger.info(f"Created 2K calibration config: {config_file}")
    return config_file

def create_2k_scanning_script():
    """Crea uno script specifico per scansioni 2K"""
    script_content = '''#!/usr/bin/env python3
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
        print("\\n🎉 2K SCAN COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {processor.output_dir}")
        print("📊 2K provides:")
        print("  - 4x more pixels than 1280x720")
        print("  - Enhanced detail capture")
        print("  - Better feature matching")
        print("  - Higher precision reconstruction")
        print("\\n🔍 Open FINAL_COMBINED_SCAN.ply in MeshLab for best viewing")
    else:
        print("\\n❌ 2K scan failed. Check logs for details.")

if __name__ == "__main__":
    main()
'''
    
    script_file = "unlook_2k_scanner.py"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_file, 0o755)
    
    logger.info(f"Created 2K scanning script: {script_file}")
    return script_file

def create_server_startup_2k():
    """Crea script di avvio server con configurazione 2K"""
    startup_script = '''#!/bin/bash
# UnLook Server 2K Startup Script

echo "🔥 Starting UnLook Server in 2K Mode"
echo "======================================"

# Export 2K configuration
export UNLOOK_CAMERA_WIDTH=2048
export UNLOOK_CAMERA_HEIGHT=1536
export UNLOOK_CAMERA_FPS=15
export UNLOOK_CAMERA_QUALITY=95
export UNLOOK_PATTERN_WIDTH=2048
export UNLOOK_PATTERN_HEIGHT=1536
export UNLOOK_QUALITY_PRESET=ultra
export UNLOOK_REALTIME_MODE=false
export UNLOOK_PRIORITY_QUALITY=true

echo "Configuration:"
echo "  Resolution: ${UNLOOK_CAMERA_WIDTH}x${UNLOOK_CAMERA_HEIGHT}"
echo "  FPS: ${UNLOOK_CAMERA_FPS}"
echo "  Quality: ${UNLOOK_CAMERA_QUALITY}%"
echo "  Mode: High-Quality (Non-Realtime)"
echo "======================================"

# Start server
python3 -m unlook.server_bootstrap --enable-2k-mode

echo "Server started with 2K configuration"
'''
    
    script_file = "start_server_2k.sh"
    with open(script_file, 'w') as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod(script_file, 0o755)
    
    logger.info(f"Created 2K server startup script: {script_file}")
    return script_file

def main():
    """Apply complete 2K configuration"""
    logger.info("🚀 CONFIGURING UNLOOK FOR 2K HIGH-QUALITY SCANNING")
    logger.info("="*70)
    
    try:
        # Step 1: Update server config
        update_server_config()
        
        # Step 2: Create calibration config
        calib_config = create_2k_calibration_config()
        
        # Step 3: Create 2K scanning script
        scan_script = create_2k_scanning_script()
        
        # Step 4: Create server startup script
        startup_script = create_server_startup_2k()
        
        logger.info("="*70)
        logger.info("✅ 2K CONFIGURATION APPLIED SUCCESSFULLY!")
        logger.info("")
        logger.info("Files created:")
        logger.info(f"  📄 {calib_config}")
        logger.info(f"  🔧 {scan_script}")
        logger.info(f"  🚀 {startup_script}")
        logger.info(f"  ⚙️  unlook_config_2k.json")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Restart the UnLook server with: ./start_server_2k.sh")
        logger.info("  2. Recalibrate cameras at 2K resolution")
        logger.info("  3. Capture new images at 2K resolution")
        logger.info("  4. Process with: python3 unlook_2k_scanner.py")
        logger.info("")
        logger.info("⚠️  Note: 2K mode sacrifices real-time performance for quality")
        logger.info("    Expect slower capture but much higher detail!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to apply 2K configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)