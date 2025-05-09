Real-time Scanner API
===================

The real-time scanner API provides functionality for performing continuous, high-speed 3D scanning.

RealTimeScanner
--------------

.. autoclass:: unlook.client.realtime_scanner.RealTimeScanner
   :members:
   :undoc-members:
   :show-inheritance:

RealTimeScanConfig
-----------------

.. autoclass:: unlook.client.realtime_scanner.RealTimeScanConfig
   :members:
   :undoc-members:
   :show-inheritance:

Factory Function
--------------

.. autofunction:: unlook.client.realtime_scanner.create_realtime_scanner

Example Usage
-----------

Basic Real-time Scanning:

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.realtime_scanner import create_realtime_scanner
   
   # Connect to scanner
   client = UnlookClient()
   # ... connect to a scanner ...
   
   # Create real-time scanner
   scanner = create_realtime_scanner(
       client=client,
       quality="medium",  # Options: "fast", "medium", "high", "ultra"
   )
   
   # Start scanning
   scanner.start()
   
   # Process scanning results in a loop
   try:
       while True:
           # Get the latest point cloud
           point_cloud = scanner.get_current_point_cloud()
           
           # Process or visualize the point cloud
           if point_cloud is not None:
               print(f"New point cloud with {len(point_cloud.points)} points")
               
           # Get current performance metrics
           fps = scanner.get_fps()
           scan_count = scanner.get_scan_count()
           print(f"FPS: {fps:.1f}, Scans: {scan_count}")
           
           # Brief pause to avoid CPU overuse
           import time
           time.sleep(0.01)
   finally:
       # Always stop scanning
       scanner.stop()

Custom Configuration:

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

Using a Callback Function:

.. code-block:: python

   # Define callback for new frames
   def on_new_frame(point_cloud, scan_count, fps):
       print(f"New frame: {scan_count}, FPS: {fps:.1f}")
       if point_cloud is not None:
           print(f"Points: {len(point_cloud.points)}")
   
   # Create scanner with callback
   scanner = create_realtime_scanner(
       client=client,
       quality="high",
       on_new_frame=on_new_frame
   )
   
   # Start scanning - callback will be called for each new frame
   scanner.start()
   
   # Wait some time
   import time
   time.sleep(30)
   
   # Stop scanning
   scanner.stop()