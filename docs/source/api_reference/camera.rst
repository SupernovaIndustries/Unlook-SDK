Camera API
=========

The camera API provides functionality for controlling and capturing images from cameras connected to the Unlook scanner.

CameraClient
-----------

.. autoclass:: unlook.client.camera.CameraClient
   :members:
   :undoc-members:
   :show-inheritance:

CameraConfig
-----------

.. autoclass:: unlook.client.camera_config.CameraConfig
   :members:
   :undoc-members:
   :show-inheritance:

Focus Checking API
----------------

Methods for checking and adjusting camera focus:

.. automethod:: unlook.client.camera.CameraClient.check_focus

.. automethod:: unlook.client.camera.CameraClient.check_stereo_focus

.. automethod:: unlook.client.camera.CameraClient.interactive_focus_check

.. automethod:: unlook.client.camera.CameraClient.interactive_stereo_focus_check

Enumerations
----------

.. autoclass:: unlook.client.camera_config.ColorMode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: unlook.client.camera_config.CompressionFormat
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: unlook.client.camera_config.ImageQualityPreset
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-----------

Capturing Images:

.. code-block:: python

   from unlook import UnlookClient
   
   # Connect to scanner
   client = UnlookClient()
   # ... connect to a scanner ...
   
   # Get list of available cameras
   cameras = client.camera.get_cameras()
   print(f"Available cameras: {cameras}")
   
   # Capture image from first camera
   camera_id = cameras[0]["id"]
   image = client.camera.capture(camera_id)
   
   # Configure camera settings
   from unlook.client.camera_config import CameraConfig, ColorMode
   config = CameraConfig()
   config.exposure_time = 20000  # 20ms
   config.gain = 1.5
   config.color_mode = ColorMode.COLOR
   
   client.camera.apply_camera_config(camera_id, config)
   
   # Check focus and adjust if needed
   client.camera.interactive_focus_check(camera_id)

Stereo Camera Operations:

.. code-block:: python

   # Capture from stereo pair
   left_image, right_image = client.camera.capture_stereo_pair()
   
   # Configure stereo cameras
   client.camera.configure_all(exposure_time=16000, gain=1.2)
   
   # Check focus of both cameras
   focus_results, _ = client.camera.check_stereo_focus()
   for camera_id, (score, quality) in focus_results.items():
       print(f"Camera {camera_id}: {score:.2f}, Quality: {quality}")