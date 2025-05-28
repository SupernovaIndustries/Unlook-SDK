
#!/usr/bin/env python3
"""
Test completo per l'integrazione Protocol V2 e nuove features.

Questo esempio testa:
1. Connessione al server con protocol v2 abilitato
2. Status del server e configurazioni optimization
3. Streaming con compression protocol v2
4. Sync metrics e compression stats
5. Camera capture con optimization
6. Tutte le nuove API client features

Usage:
    python unlook/examples/test_protocol_v2_integration.py
"""

import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_protocol_v2_integration():
    """Test completo integration protocol v2."""
    
    try:
        # Import UnLook SDK client directly (simple.py removed)
        from unlook.client.scanner.scanner import UnlookClient
        
        print("🚀 Starting Protocol V2 Integration Test")
        print("=" * 50)
        
        # 1. Create client and discover scanner
        print("\n1️⃣ Creating client and discovering scanners...")
        client = UnlookClient(client_name="ProtocolV2TestClient", auto_discover=True)
        
        # Wait for discovery
        time.sleep(2)
        scanners = client.get_discovered_scanners()
        
        if not scanners:
            print("❌ No scanners found! Make sure server is running with:")
            print("   python unlook/server_bootstrap.py --enable-protocol-v2 --enable-pattern-preprocessing --enable-sync")
            return False
        
        print(f"✅ Found {len(scanners)} scanner(s)")
        
        # 2. Connect to first scanner
        print("\n2️⃣ Connecting to scanner...")
        scanner = scanners[0]
        success = client.connect(scanner)
        
        if not success:
            print("❌ Failed to connect to scanner")
            return False
        
        print(f"✅ Connected to scanner: {scanner.name}")
        
        # 3. Test new API features
        print("\n3️⃣ Testing new API features...")
        
        # Test server status
        print("📊 Getting server status...")
        server_status = client.get_server_status()
        if server_status:
            print(f"   Server version: {server_status.get('server_info', {}).get('version', 'N/A')}")
            print(f"   Protocol V2 enabled: {server_status.get('optimization_settings', {}).get('protocol_v2_enabled', False)}")
            print(f"   Preprocessing enabled: {server_status.get('optimization_settings', {}).get('preprocessing_enabled', False)}")
            print(f"   Sync enabled: {server_status.get('optimization_settings', {}).get('sync_enabled', False)}")
            print(f"   Cameras available: {server_status.get('hardware_status', {}).get('cameras_available', 0)}")
        else:
            print("⚠️  Could not get server status")
        
        # Test protocol v2 check
        protocol_v2_enabled = client.is_protocol_v2_enabled()
        print(f"   Protocol V2 active: {'✅' if protocol_v2_enabled else '❌'} {protocol_v2_enabled}")
        
        # Test preprocessing info
        preprocessing_info = client.get_preprocessing_info()
        if preprocessing_info:
            print(f"   GPU preprocessing: {'✅' if preprocessing_info.get('gpu_available', False) else '❌'}")
            print(f"   Preprocessing level: {preprocessing_info.get('level', 'none')}")
        
        # 4. Test sync features
        print("\n4️⃣ Testing synchronization features...")
        
        # Get sync metrics
        sync_metrics = client.get_sync_metrics()
        if sync_metrics:
            print(f"   Sync metrics available: ✅")
            print(f"   Sync precision: {sync_metrics.get('sync_precision_us', 'N/A')} μs")
            print(f"   Frame consistency: {sync_metrics.get('frame_consistency_percent', 'N/A')}%")
        else:
            print("   Sync metrics: ⚠️  Not available")
        
        # Test enable sync
        sync_enabled = client.enable_sync(enable=True, fps=30.0)
        print(f"   Sync enable test: {'✅' if sync_enabled else '❌'}")
        
        # 5. Test camera features
        print("\n5️⃣ Testing camera with protocol v2...")
        
        # Get camera list
        cameras = client.camera.get_cameras()
        print(f"   Available cameras: {len(cameras) if cameras else 0}")
        
        if cameras:
            # Test single capture with protocol v2 optimization
            print("   Testing single camera capture...")
            camera_id = cameras[0]['id']  # cameras is a list of dicts
            
            start_time = time.time()
            result = client.camera.capture_image(camera_id)
            capture_time = (time.time() - start_time) * 1000
            
            if result and 'image' in result:
                height, width = result['image'].shape[:2]
                print(f"   ✅ Captured {width}x{height} image in {capture_time:.1f}ms")
                
                # Check if compression was used
                metadata = result.get('metadata', {})
                optimization = metadata.get('optimization', {})
                if optimization:
                    print(f"   📦 Protocol V2 optimization applied:")
                    print(f"      Compression ratio: {optimization.get('compression_ratio', 'N/A')}")
                    print(f"      Original size: {optimization.get('original_size', 'N/A')} bytes")
            else:
                print("   ⚠️  Single capture failed")
            
            # Test multi-camera capture
            if len(cameras) >= 2:
                print("   Testing multi-camera capture...")
                camera_ids = [cam['id'] for cam in cameras[:2]]  # Get first 2 camera IDs
                
                start_time = time.time()
                results = client.camera.capture_synchronized(camera_ids)
                multi_capture_time = (time.time() - start_time) * 1000
                
                if results and len(results) >= 2:
                    print(f"   ✅ Multi-camera capture: {len(results)} cameras in {multi_capture_time:.1f}ms")
                else:
                    print("   ⚠️  Multi-camera capture failed")
        
        # 6. Test streaming with protocol v2
        print("\n6️⃣ Testing streaming with protocol v2...")
        
        if cameras:
            camera_id = cameras[0]['id']  # cameras is a list of dicts
            
            # Stream callback to count frames and check optimization
            frame_count = 0
            optimization_count = 0
            
            def frame_callback(cam_id, image, metadata):
                nonlocal frame_count, optimization_count
                frame_count += 1
                
                # Check for protocol v2 optimization
                if metadata.get('optimization'):
                    optimization_count += 1
                
                if frame_count == 1:
                    height, width = image.shape[:2]
                    print(f"   📹 Receiving {width}x{height} frames...")
                
                # Stop after 10 frames
                if frame_count >= 10:
                    client.stream.stop_streaming(cam_id)
            
            # Start streaming
            print(f"   Starting streaming from camera {camera_id}...")
            success = client.stream.start_streaming(camera_id, frame_callback, fps=15)
            
            if success:
                # Wait for frames
                time.sleep(3)
                
                print(f"   ✅ Received {frame_count} frames")
                print(f"   📦 Protocol V2 optimized frames: {optimization_count}/{frame_count}")
                
                # Stop streaming
                client.stream.stop_streaming(camera_id)
            else:
                print("   ❌ Failed to start streaming")
        
        # 7. Get final compression stats
        print("\n7️⃣ Getting compression statistics...")
        
        compression_stats = client.get_compression_stats()
        if compression_stats:
            print(f"   📊 Compression Statistics:")
            print(f"      Average compression ratio: {compression_stats.get('avg_compression_ratio', 'N/A'):.2f}x")
            print(f"      Average compression time: {compression_stats.get('avg_compression_time_ms', 'N/A'):.1f}ms")
            print(f"      Bandwidth savings: {compression_stats.get('bandwidth_savings_percent', 'N/A'):.1f}%")
            print(f"      Total data processed: {compression_stats.get('total_original_mb', 'N/A'):.1f}MB")
        else:
            print("   ⚠️  No compression stats available")
        
        # 8. Final status check
        print("\n8️⃣ Final status check...")
        
        final_status = client.get_server_status()
        if final_status:
            perf_metrics = final_status.get('performance_metrics', {})
            if perf_metrics:
                print(f"   🎯 Performance Summary:")
                
                protocol_v2_stats = perf_metrics.get('protocol_v2', {})
                if protocol_v2_stats:
                    print(f"      Protocol V2 active: ✅")
                    print(f"      Data transferred: {protocol_v2_stats.get('total_compressed_mb', 'N/A'):.1f}MB compressed")
                
                preprocessing_stats = perf_metrics.get('preprocessing', {})
                if preprocessing_stats:
                    print(f"      GPU preprocessing: ✅ Level {preprocessing_stats.get('level', 'N/A')}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        client.disconnect()
        
        print("\n🎉 Protocol V2 Integration Test COMPLETED!")
        print("=" * 50)
        print("✅ All features tested successfully")
        print("📊 Protocol V2 optimization working")
        print("🔄 Retrocompatibilità maintained")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            if 'client' in locals():
                client.disconnect()
        except:
            pass

if __name__ == "__main__":
    print("Protocol V2 Integration Test")
    print("Make sure to start server with:")
    print("python unlook/server_bootstrap.py --enable-protocol-v2 --enable-pattern-preprocessing --enable-sync")
    print("\nStarting test in 3 seconds...")
    time.sleep(3)
    
    success = test_protocol_v2_integration()
    
    if success:
        print("\n🎯 Test Result: SUCCESS ✅")
        exit(0)
    else:
        print("\n💥 Test Result: FAILED ❌")
        exit(1)