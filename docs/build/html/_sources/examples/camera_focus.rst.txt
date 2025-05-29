Camera Focus Examples
==================

This example demonstrates how to use the camera focus checking functionality in the Unlook SDK.

Basic Focus Check
--------------

.. code-block:: python
   :linenos:
   :caption: basic_focus_check.py

   #!/usr/bin/env python3
   # -*- coding: utf-8 -*-
   
   from unlook import UnlookClient
   import time
   import cv2
   
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
       
       # Get available cameras
       cameras = client.camera.get_cameras()
       if not cameras:
           print("No cameras found on the scanner.")
           return
           
       # Check focus for each camera
       for camera in cameras:
           camera_id = camera["id"]
           print(f"Checking focus for camera {camera_id}...")
           
           # Check focus with multiple samples
           score, quality, image = client.camera.check_focus(camera_id, num_samples=5)
           print(f"Camera {camera_id} focus score: {score:.2f}, Quality: {quality}")
           
           # Display the image with focus information
           if image is not None:
               # Create a copy for drawing
               display_img = image.copy()
               
               # Add focus information
               cv2.putText(display_img, f"Focus: {score:.2f} - {quality}", 
                          (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               
               # Show the image
               cv2.imshow(f"Camera {camera_id}", display_img)
               cv2.waitKey(0)
               cv2.destroyAllWindows()
       
       # Disconnect from scanner
       client.disconnect()
       print("Disconnected from scanner")
   
   if __name__ == "__main__":
       main()

Interactive Focus Adjustment
-------------------------

.. code-block:: python
   :linenos:
   :caption: interactive_focus.py

   #!/usr/bin/env python3
   # -*- coding: utf-8 -*-
   
   from unlook import UnlookClient
   import time
   import argparse
   
   def parse_roi(roi_str):
       """Parse region of interest string to tuple."""
       if not roi_str:
           return None
           
       try:
           parts = roi_str.split(',')
           if len(parts) != 4:
               print(f"Invalid ROI format: {roi_str}. Expected 'x,y,width,height'")
               return None
               
           roi = tuple(int(part) for part in parts)
           return roi
       except ValueError:
           print(f"Invalid ROI values: {roi_str}. Expected integers")
           return None
   
   def main():
       # Parse command line arguments
       parser = argparse.ArgumentParser(description='Camera Focus Adjustment Tool')
       parser.add_argument('--stereo', action='store_true', 
                          help='Check focus for stereo camera pair')
       parser.add_argument('--camera-id', type=str, default=None,
                          help='Specific camera ID to check (for single camera mode)')
       parser.add_argument('--roi', type=str, default=None,
                          help='Region of interest for focus check (x,y,width,height)')
       parser.add_argument('--timeout', type=int, default=5,
                          help='Discovery timeout in seconds')
       args = parser.parse_args()
       
       # Parse ROI
       roi = parse_roi(args.roi)
       if args.roi and roi is None:
           return 1
           
       # Create client with auto-discovery
       client = UnlookClient(auto_discover=True)
       print("Starting scanner discovery...")
       client.start_discovery()
       
       # Wait for discovery
       print(f"Waiting {args.timeout} seconds for scanner discovery...")
       time.sleep(args.timeout)
       
       # Connect to first available scanner
       scanners = client.get_discovered_scanners()
       if not scanners:
           print("No scanners found. Please check that scanner is powered on.")
           return 1
           
       print(f"Connecting to scanner: {scanners[0].name}")
       client.connect(scanners[0])
       
       try:
           if args.stereo:
               # Stereo camera focus check
               print("Starting interactive stereo focus check")
               print("Adjust camera focus until both cameras show GOOD or EXCELLENT")
               print("Press Ctrl+C to exit when focus is good")
               
               client.camera.interactive_stereo_focus_check(
                   roi=roi,
                   interval=0.5
               )
           else:
               # Single camera focus check
               camera_id = args.camera_id
               
               if camera_id is None:
                   # If no camera ID specified, use first available
                   cameras = client.camera.get_cameras()
                   if not cameras:
                       print("No cameras found on the scanner.")
                       return 1
                   camera_id = cameras[0]["id"]
               
               print(f"Starting interactive focus check for camera {camera_id}")
               print("Adjust camera focus until it shows GOOD or EXCELLENT")
               print("Press Ctrl+C to exit when focus is good")
               
               client.camera.interactive_focus_check(
                   camera_id=camera_id,
                   roi=roi,
                   interval=0.5
               )
               
       except KeyboardInterrupt:
           print("Focus check interrupted by user")
       finally:
           # Final focus check
           if args.stereo:
               results, _ = client.camera.check_stereo_focus(num_samples=5, roi=roi)
               for camera_id, (score, quality) in results.items():
                   print(f"Final camera {camera_id} focus: {score:.2f}, Quality: {quality}")
           else:
               score, quality, _ = client.camera.check_focus(camera_id, num_samples=5, roi=roi)
               print(f"Final camera {camera_id} focus: {score:.2f}, Quality: {quality}")
           
           # Disconnect from scanner
           client.disconnect()
           print("Disconnected from scanner")
       
       return 0
   
   if __name__ == "__main__":
       exit(main())

Running the Examples
-----------------

To run these examples:

1. Ensure you have installed all required dependencies:
   
   .. code-block:: bash
      
      pip install -r client-requirements.txt
      
2. Connect your Unlook scanner hardware and ensure it's powered on

3. Run the basic focus check:
   
   .. code-block:: bash
      
      python basic_focus_check.py
      
4. Run the interactive focus adjustment:
   
   .. code-block:: bash
      
      # For a single camera
      python interactive_focus.py
      
      # For stereo cameras
      python interactive_focus.py --stereo
      
      # With a region of interest
      python interactive_focus.py --stereo --roi 640,360,200,200

Advanced Usage Notes
-----------------

- For the best focus check results, use a target with fine details or text
- The ROI feature is useful when your scene has varying depths, allowing you to focus on a specific area
- In stereo setups, it's important that both cameras are equally well-focused
- After achieving good focus, ensure any locking rings on the camera lenses are tightened without changing the focus position