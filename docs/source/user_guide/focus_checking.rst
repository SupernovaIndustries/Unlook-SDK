Camera Focus Checking
==================

Proper camera focus is crucial for getting high-quality 3D scans. The Unlook SDK provides tools for checking and adjusting camera focus to ensure optimal scanning results.

Importance of Camera Focus
------------------------

For structured light scanning, having properly focused cameras is critical because:

- Blurry images make pattern detection less reliable
- Poor focus reduces the precision of 3D point measurements
- Out-of-focus cameras can cause holes or noise in the 3D scan

The Focus Checking API
-------------------

The Unlook SDK provides focus checking capabilities through the `CameraClient` class:

.. code-block:: python

   # Access through the main client
   client = UnlookClient()
   # ... connect to scanner ...
   
   # Check focus for a single camera
   score, quality, image = client.camera.check_focus(camera_id)
   print(f"Focus score: {score:.2f}, Quality: {quality}")
   
   # Check focus for stereo camera setup
   results, images = client.camera.check_stereo_focus()
   for camera_id, (score, quality) in results.items():
       print(f"Camera {camera_id}: {score:.2f}, Quality: {quality}")

Interactive Focus Adjustment
-------------------------

The SDK provides interactive focus checking tools that provide real-time feedback while you manually adjust the camera lenses:

.. code-block:: python

   # Interactive focus check for a single camera
   client.camera.interactive_focus_check(camera_id)
   
   # Interactive focus check for stereo cameras
   client.camera.interactive_stereo_focus_check()

This will open a visualization window showing the camera feed with focus scores overlaid, and will provide continuous feedback as you adjust the lens focus.

Using Region of Interest (ROI)
----------------------------

For more precise focus checking, you can specify a region of interest within the image:

.. code-block:: python

   # Define ROI as (x, y, width, height)
   roi = (640, 360, 200, 200)  # Center region of a 1280x720 image
   
   # Check focus in the specified region
   client.camera.interactive_stereo_focus_check(roi=roi)

This is useful when you want to focus on a specific area of your scan, such as the center of the field of view.

Focus Quality Levels
-----------------

The SDK categorizes focus quality into four levels:

- **poor**: Very blurry, unusable for scanning
- **moderate**: Somewhat focused, but could be improved
- **good**: Well focused, suitable for scanning
- **excellent**: Perfectly focused, optimal for scanning

Integration with Real-time Scanning
-------------------------------

The real-time scanning example integrates focus checking:

.. code-block:: bash

   # Run the example with focus checking
   python unlook/examples/realtime_scanning_example.py --check-focus
   
   # Specify a region of interest
   python unlook/examples/realtime_scanning_example.py --check-focus --focus-roi 640,360,200,200

How Focus is Measured
------------------

The SDK uses the Laplacian variance method to measure focus:

1. Convert the image to grayscale
2. Apply the Laplacian operator to detect edges
3. Calculate the variance of the Laplacian result
4. Higher variance indicates more edges and better focus

This method is fast and effective, making it suitable for real-time applications.

Tips for Achieving Good Focus
--------------------------

1. Start with a well-lit scene with good contrast
2. Position your cameras at the intended working distance from the scan subject
3. If available, use a focus target with fine details or text
4. Run the interactive focus check and slowly adjust the lens until highest score is achieved
5. Once good focus is achieved, carefully tighten any locking rings without changing the focus
6. Verify focus again after locking the lens
7. For stereo systems, ensure both cameras are equally well-focused

Custom Focus Thresholds
--------------------

You can also provide custom thresholds for what constitutes good focus:

.. code-block:: python

   # Set a specific threshold for good focus
   threshold = 300.0
   client.camera.interactive_stereo_focus_check(threshold_for_good=threshold)

Focus Checking API Reference
-------------------------

For more details on the focus checking API, see the :doc:`../api_reference/camera` reference documentation.