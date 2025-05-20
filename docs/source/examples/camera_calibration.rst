Camera Calibration
================

This example demonstrates how to perform camera calibration with the UnLook SDK.

Overview
-------

Camera calibration is a crucial step for accurate 3D scanning. The UnLook SDK provides tools to:

1. Capture calibration images
2. Process calibration data
3. Save and load calibration files
4. Verify calibration accuracy

Basic Calibration Process
-----------------------

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.scanning.calibration import StereoCalibrator
   import cv2
   import numpy as np
   import time

   # Connect to scanner
   client = UnlookClient(auto_discover=True)
   client.start_discovery()
   time.sleep(5)
   
   scanners = client.get_discovered_scanners()
   if scanners:
       client.connect(scanners[0])
       
       # Create calibrator
       calibrator = StereoCalibrator(
           checkerboard_size=(9, 6),  # Number of internal corners
           square_size=25.0,          # Size in mm
           camera_resolution=(1920, 1080)
       )
       
       # Capture calibration images
       print("Capturing calibration images...")
       calibration_images = []
       
       for i in range(20):
           print(f"Capturing image pair {i+1}/20...")
           
           # Capture from left camera
           left_image = client.camera.capture("camera_left")
           
           # Capture from right camera
           right_image = client.camera.capture("camera_right")
           
           # Add to calibration set
           if left_image is not None and right_image is not None:
               calibration_images.append((left_image, right_image))
               
           # Wait for repositioning checkerboard
           time.sleep(2)
       
       # Perform calibration
       print("Performing calibration...")
       calibration_result = calibrator.calibrate(calibration_images)
       
       # Save calibration
       calibrator.save_calibration("stereo_calibration.json")
       
       print("Calibration complete!")
       print(f"Reprojection error: {calibration_result['reprojection_error']}")
       
       # When done
       client.disconnect()
   else:
       print("No scanners found.")

Loading and Using Calibration
---------------------------

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.scanning.calibration import load_calibration
   import time

   # Connect to scanner
   client = UnlookClient(auto_discover=True)
   client.start_discovery()
   time.sleep(5)
   
   scanners = client.get_discovered_scanners()
   if scanners:
       client.connect(scanners[0])
       
       # Load calibration
       calibration_data = load_calibration("stereo_calibration.json")
       
       # Use calibration for operations
       # For example, create a scanner with this calibration
       from unlook.client.scanning import create_real_time_scanner
       
       scanner = create_real_time_scanner(
           client=client,
           quality="medium",
           calibration_data=calibration_data
       )
       
       # When done
       client.disconnect()
   else:
       print("No scanners found.")

Verifying Calibration
-------------------

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.scanning.calibration import verify_calibration
   import time

   # Connect to scanner
   client = UnlookClient(auto_discover=True)
   client.start_discovery()
   time.sleep(5)
   
   scanners = client.get_discovered_scanners()
   if scanners:
       client.connect(scanners[0])
       
       # Capture verification images
       left_image = client.camera.capture("camera_left")
       right_image = client.camera.capture("camera_right")
       
       # Verify calibration
       verification_result = verify_calibration(
           "stereo_calibration.json",
           left_image,
           right_image,
           checkerboard_size=(9, 6)
       )
       
       # Print results
       print(f"Verification reprojection error: {verification_result['reprojection_error']}")
       print(f"Calibration is{'valid" if verification_result["is_valid"] else " NOT valid"}'}")
       
       # When done
       client.disconnect()
   else:
       print("No scanners found.")

Full Example
----------

For a complete, runnable example, see the file ``unlook/examples/camera_calibration_example.py`` in the SDK.