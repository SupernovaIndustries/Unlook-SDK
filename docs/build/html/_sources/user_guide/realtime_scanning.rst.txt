Real-time Scanning
=================

The real-time scanning module provides optimized scanning capabilities for faster frame rates and continuous scanning. This makes it ideal for handheld 3D scanner applications where low latency and smooth operation are essential.

Key Features
----------

- GPU acceleration (when available)
- Neural network point cloud enhancement (optional)
- Optimized pattern sequences
- Minimal pattern sets for faster scanning
- Real-time visualization
- Data recording capabilities

Basic Usage
---------

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.realtime_scanner import create_realtime_scanner
   
   # Connect to scanner
   client = UnlookClient(auto_discover=True)
   # ... wait for discovery and connect to scanner ...
   
   # Create real-time scanner with desired quality
   scanner = create_realtime_scanner(
       client=client,
       quality="medium",  # Options: "fast", "medium", "high", "ultra"
       calibration_file="calibration/stereo_calib.json"  # Optional
   )
   
   # Start scanning
   scanner.start()
   
   # Process results
   while True:
       # Get the latest point cloud
       point_cloud = scanner.get_current_point_cloud()
       
       # Process or visualize the point cloud
       if point_cloud is not None:
           # Do something with the point cloud
           print(f"Got point cloud with {len(point_cloud.points)} points")
       
       # Check scanner status
       fps = scanner.get_fps()
       scan_count = scanner.get_scan_count()
       print(f"FPS: {fps:.1f}, Scans: {scan_count}")
       
       # Sleep briefly
       time.sleep(0.01)
   
   # When done
   scanner.stop()

Quality Presets
-------------

The SDK provides several quality presets to balance between speed and detail:

- **fast**: Optimized for highest frame rate, lower resolution patterns
- **medium**: Balanced setting, good for most use cases
- **high**: Higher detail, but slower frame rate
- **ultra**: Maximum quality, slowest frame rate

.. code-block:: python

   # Use high quality preset
   scanner = create_realtime_scanner(client=client, quality="high")

Camera Focus Check
---------------

For optimal scanning results, proper camera focus is essential. The SDK provides tools to check and adjust focus before scanning:

.. code-block:: python

   # Check focus of both cameras
   focus_results, images = client.camera.check_stereo_focus()
   
   # Interactive focus adjustment
   client.camera.interactive_stereo_focus_check()
   
   # Specify a region of interest for focused checking
   roi = (640, 360, 200, 200)  # x, y, width, height
   client.camera.interactive_stereo_focus_check(roi=roi)

Configuration Options
------------------

For more control, you can configure the scanner directly:

.. code-block:: python

   from unlook.client.realtime_scanner import RealTimeScanConfig
   
   # Create custom configuration
   config = RealTimeScanConfig()
   
   # Configure scanning parameters
   config.use_gpu = True  # Use GPU acceleration if available
   config.use_neural_network = True  # Use neural network enhancement
   config.max_fps = 15  # Target maximum FPS
   config.moving_average_frames = 5  # Temporal smoothing frames
   config.downsample_voxel_size = 2.0  # Point cloud downsampling (mm)
   config.num_gray_codes = 6  # Number of Gray code bits
   config.pattern_interval = 0.1  # Time between patterns (seconds)
   
   # Create scanner with custom config
   scanner = create_realtime_scanner(client=client, config=config)

Visualization
-----------

Visualization of the point cloud during scanning is possible with Open3D:

.. code-block:: python

   from unlook.client.visualization import ScanVisualizer
   
   # Create visualizer
   visualizer = ScanVisualizer()
   
   # Update with new point clouds during scanning
   visualizer.update(point_cloud, fps, scan_count)
   
   # Close when done
   visualizer.close()

Recording Scan Data
----------------

To save the scanning results:

.. code-block:: python

   from unlook.client.recorder import ScanRecorder
   
   # Create recorder
   recorder = ScanRecorder("output_directory")
   
   # Start recording
   recorder.start_recording()
   
   # Record frames during scanning
   recorder.record_frame(point_cloud, scan_count, fps)
   
   # Stop recording
   recorder.stop_recording()

Performance Optimization
---------------------

For best performance:

- Use GPU acceleration when available
- Choose appropriate quality presets for your needs
- Position the scanner at the optimal distance (typically 30-60cm)
- Ensure good lighting conditions
- Use a powerful computer with dedicated GPU for processing
- Keep moving_average_frames lower for more responsive scanning
- Increase downsample_voxel_size for faster but coarser results

Troubleshooting
------------

- **Low frame rate**: Try a faster quality preset, disable neural network enhancement, or increase downsampling
- **Poor point cloud quality**: Check camera focus, try higher quality preset, improve lighting
- **No points generated**: Verify calibration, check camera exposure settings
- **GPU not utilized**: Check CUDA installation, verify GPU is detected (run `python -m unlook.utils.check_gpu`)