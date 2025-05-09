Real-time Scanning Example
=======================

This example demonstrates how to set up and use the real-time scanning feature of the Unlook SDK.

Basic Real-time Scanning
---------------------

.. code-block:: python
   :linenos:
   :caption: basic_realtime_scanning.py

   #!/usr/bin/env python3
   # -*- coding: utf-8 -*-
   
   from unlook import UnlookClient
   from unlook.client.realtime_scanner import create_realtime_scanner
   import time
   import numpy as np
   import open3d as o3d
   
   def main():
       # Create client with auto-discovery
       client = UnlookClient(auto_discover=True)
       print("Starting scanner discovery...")
       client.start_discovery()
       
       # Wait for discovery
       time.sleep(5)
       
       # Connect to first available scanner
       scanners = client.get_discovered_scanners()
       if not scanners:
           print("No scanners found. Please check that scanner is powered on.")
           return
           
       print(f"Connecting to scanner: {scanners[0].name}")
       client.connect(scanners[0])
       
       # Create real-time scanner with medium quality
       scanner = create_realtime_scanner(
           client=client,
           quality="medium",  # Options: "fast", "medium", "high", "ultra"
       )
       
       # Start scanning
       print("Starting real-time scanning...")
       scanner.start()
       
       # Create Open3D visualizer
       vis = o3d.visualization.Visualizer()
       vis.create_window(window_name="Real-time 3D Scanning")
       
       # Add coordinate frame
       coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
       vis.add_geometry(coord_frame)
       
       # Create empty point cloud for visualization
       pcd = o3d.geometry.PointCloud()
       vis.add_geometry(pcd)
       
       # Configure view
       view_control = vis.get_view_control()
       view_control.set_zoom(0.5)
       view_control.set_front([0, 0, -1])
       view_control.set_lookat([0, 0, 0])
       view_control.set_up([0, -1, 0])
       
       try:
           # Main scanning loop
           scan_count = 0
           while scan_count < 100:  # Scan 100 frames
               # Get the latest point cloud
               point_cloud = scanner.get_current_point_cloud()
               
               # Update visualization if available
               if point_cloud is not None and isinstance(point_cloud, o3d.geometry.PointCloud):
                   # Copy the point cloud to avoid modifying the original
                   pcd.points = point_cloud.points
                   pcd.colors = point_cloud.colors
                   
                   # Update visualization
                   vis.update_geometry(pcd)
                   vis.poll_events()
                   vis.update_renderer()
                   
                   # Get and print statistics
                   fps = scanner.get_fps()
                   scan_count = scanner.get_scan_count()
                   print(f"FPS: {fps:.1f}, Scan count: {scan_count}, Points: {len(point_cloud.points)}")
               
               # Brief pause to avoid CPU overuse
               time.sleep(0.01)
               
       finally:
           # Always stop scanning and close visualization
           print("Stopping scanner...")
           scanner.stop()
           vis.destroy_window()
           
           # Disconnect from scanner
           client.disconnect()
           print("Disconnected from scanner")
   
   if __name__ == "__main__":
       main()

Using Focus Check Before Scanning
------------------------------

.. code-block:: python
   :linenos:
   :caption: realtime_scanning_with_focus_check.py

   #!/usr/bin/env python3
   # -*- coding: utf-8 -*-
   
   from unlook import UnlookClient
   from unlook.client.realtime_scanner import create_realtime_scanner
   import time
   import numpy as np
   
   def main():
       # Create client with auto-discovery
       client = UnlookClient(auto_discover=True)
       print("Starting scanner discovery...")
       client.start_discovery()
       
       # Wait for discovery
       time.sleep(5)
       
       # Connect to first available scanner
       scanners = client.get_discovered_scanners()
       if not scanners:
           print("No scanners found. Please check that scanner is powered on.")
           return
           
       print(f"Connecting to scanner: {scanners[0].name}")
       client.connect(scanners[0])
       
       # Run focus check before scanning
       print("Checking camera focus...")
       focus_results, _ = client.camera.check_stereo_focus()
       
       # Display focus results
       for camera_id, (score, quality) in focus_results.items():
           print(f"Camera {camera_id} focus: {score:.2f}, Quality: {quality}")
           
       # If focus is poor, run interactive focus check
       poor_focus = any(quality == "poor" for _, quality in focus_results.values())
       if poor_focus:
           print("Poor focus detected. Running interactive focus check...")
           print("Adjust camera focus until both cameras show GOOD or EXCELLENT.")
           print("Press Ctrl+C to continue when focus is good.")
           
           # Run interactive focus check
           try:
               client.camera.interactive_stereo_focus_check()
           except KeyboardInterrupt:
               print("Focus check interrupted. Continuing with scanning.")
       
       # Create real-time scanner
       scanner = create_realtime_scanner(
           client=client,
           quality="medium",  # Options: "fast", "medium", "high", "ultra"
       )
       
       # Start scanning
       print("Starting real-time scanning...")
       scanner.start()
       
       try:
           # Main scanning loop
           for _ in range(100):  # Scan for 100 iterations
               # Get the latest point cloud
               point_cloud = scanner.get_current_point_cloud()
               
               # Process point cloud if available
               if point_cloud is not None:
                   fps = scanner.get_fps()
                   scan_count = scanner.get_scan_count()
                   num_points = len(point_cloud.points) if hasattr(point_cloud, "points") else len(point_cloud)
                   print(f"FPS: {fps:.1f}, Scan count: {scan_count}, Points: {num_points}")
               
               # Brief pause
               time.sleep(0.1)
               
       finally:
           # Always stop scanning
           print("Stopping scanner...")
           scanner.stop()
           
           # Disconnect from scanner
           client.disconnect()
           print("Disconnected from scanner")
   
   if __name__ == "__main__":
       main()

Running the Examples
-----------------

To run these examples:

1. Ensure you have installed all required dependencies:
   
   .. code-block:: bash
      
      pip install -r client-requirements.txt
      
2. Connect your Unlook scanner hardware and ensure it's powered on

3. Run the example:
   
   .. code-block:: bash
      
      python basic_realtime_scanning.py
      
   Or with focus checking:
   
   .. code-block:: bash
      
      python realtime_scanning_with_focus_check.py

Example Command-line Tool
----------------------

For a more comprehensive example with command-line arguments, see the `realtime_scanning_example.py` script included in the SDK:

.. code-block:: bash
   
   # Run with visualization
   python unlook/examples/realtime_scanning_example.py --visualize
   
   # Run with focus checking before scanning
   python unlook/examples/realtime_scanning_example.py --check-focus
   
   # Focus on a specific region of interest
   python unlook/examples/realtime_scanning_example.py --check-focus --focus-roi 640,360,200,200
   
   # High quality scan with recording
   python unlook/examples/realtime_scanning_example.py --quality high --record --output scans/my_scan