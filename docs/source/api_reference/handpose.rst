Handpose Module
===============

The handpose module provides advanced real-time 3D hand tracking and gesture recognition capabilities using stereo cameras and structured light projection.

Overview
--------

The UnLook SDK handpose module combines several technologies to deliver high-precision hand tracking:

* **Stereo Vision**: Uses dual cameras for 3D triangulation
* **Point Projection**: LED1 provides focused illumination for enhanced depth sensing
* **MediaPipe Integration**: Advanced hand landmark detection
* **Gesture Recognition**: Both rule-based and ML-powered gesture classification
* **Real-time Processing**: Optimized for low-latency applications

Core Components
---------------

HandTracker
~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.handpose.HandTracker
   :members:
   :undoc-members:
   :show-inheritance:

The main class for 3D hand tracking using stereo cameras.

**Key Features:**

* Automatic stereo camera calibration loading
* 3D triangulation of hand landmarks
* Temporal stability and noise reduction
* Multi-hand tracking support
* Handedness detection with stereo consistency

**Basic Usage:**

.. code-block:: python

   from unlook.client.scanning.handpose import HandTracker
   
   # Initialize with calibration
   tracker = HandTracker(
       calibration_file="path/to/stereo_calibration.json",
       max_num_hands=2,
       detection_confidence=0.6,
       tracking_confidence=0.6
   )
   
   # Track hands in stereo images
   results = tracker.track_hands_3d(left_image, right_image)
   
   # Access 3D keypoints
   if results['3d_keypoints']:
       keypoints_3d = results['3d_keypoints'][0]  # First hand
       print(f"Wrist position: {keypoints_3d[0]}")  # 3D coordinates in mm

HandDetector
~~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.handpose.HandDetector
   :members:
   :undoc-members:
   :show-inheritance:

MediaPipe-based hand detection and landmark extraction for individual camera views.

**Features:**

* 21-point hand landmark detection
* Confidence scoring
* Handedness classification
* Normalized coordinate output

GestureRecognizer
~~~~~~~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.handpose.GestureRecognizer
   :members:
   :undoc-members:
   :show-inheritance:

Advanced gesture recognition supporting both 2D and 3D hand poses.

**Supported Gestures:**

* Static Poses: Open Palm, Closed Fist, Pointing, Peace Sign, Thumbs Up/Down, OK Sign, Rock Sign, Pinch
* Dynamic Gestures: Swipes (Left/Right/Up/Down), Circle, Wave

**Recognition Methods:**

* Rule-based classification for reliability
* Machine learning models for enhanced accuracy
* Temporal smoothing for gesture stability

GestureType
~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.handpose.GestureType
   :members:
   :undoc-members:

Enumeration of all supported gesture types.

LED Controller Integration
-------------------------

The handpose module integrates with the LED controller for optimal illumination:

Point Projection (LED1)
~~~~~~~~~~~~~~~~~~~~~~~

* **Purpose**: Provides focused illumination for enhanced triangulation
* **Default**: 50mA intensity
* **Benefits**: Improves depth accuracy, reduces noise in stereo matching

Flood Illumination (LED2)
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Purpose**: General scene illumination for hand detection
* **Default**: 50mA intensity  
* **Benefits**: Consistent lighting for gesture recognition

**LED Configuration Example:**

.. code-block:: python

   from unlook.client.projector import LEDController
   
   # Initialize LED controller
   led_controller = LEDController(client)
   
   # Set both LEDs for optimal handpose tracking
   led_controller.set_intensity(
       led1=50,   # Point projection
       led2=50    # Flood illumination
   )

Coordinate Systems
------------------

The handpose module uses several coordinate systems:

Normalized Coordinates (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Range**: [0, 1] for both x and y
* **Origin**: Top-left corner of image
* **Usage**: MediaPipe hand landmarks

Pixel Coordinates (2D)
~~~~~~~~~~~~~~~~~~~~~~

* **Range**: [0, width] x [0, height]
* **Origin**: Top-left corner of image
* **Usage**: Display and annotation

World Coordinates (3D)
~~~~~~~~~~~~~~~~~~~~~~

* **Units**: Millimeters (mm)
* **Origin**: Left camera optical center
* **Axes**: X-right, Y-down, Z-forward
* **Usage**: 3D hand tracking and gesture analysis

Stereo Camera Setup
-------------------

Optimal Configuration
~~~~~~~~~~~~~~~~~~~~~

* **Baseline**: 65-120mm (similar to human IPD)
* **Convergence**: Parallel or slight toe-in
* **Synchronization**: Hardware or software sync required
* **Calibration**: Essential for accurate 3D reconstruction

Calibration Requirements
~~~~~~~~~~~~~~~~~~~~~~~

The handpose module requires stereo calibration data:

.. code-block:: json

   {
       "camera_matrix_left": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
       "camera_matrix_right": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
       "dist_coeffs_left": [k1, k2, p1, p2, k3],
       "dist_coeffs_right": [k1, k2, p1, p2, k3],
       "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
       "T": [tx, ty, tz],
       "P1": "Projection matrix for left camera",
       "P2": "Projection matrix for right camera"
   }

Performance Optimization
------------------------

Frame Rate Optimization
~~~~~~~~~~~~~~~~~~~~~~~

* **Image Preprocessing**: CLAHE and bilateral filtering
* **Downsampling**: Configurable resolution reduction
* **Temporal Filtering**: Reduces jitter without lag
* **Region of Interest**: Focus processing on hand areas

Memory Management
~~~~~~~~~~~~~~~~

* **Landmark History**: Limited buffer sizes
* **Model Loading**: Lazy initialization
* **Resource Cleanup**: Proper disposal of MediaPipe resources

Error Handling
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**No 3D Results**

* Check stereo calibration file
* Verify camera synchronization
* Ensure adequate lighting

**Poor Gesture Recognition**

* Verify hand is within detection range (30-100cm)
* Check lighting conditions
* Adjust gesture confidence thresholds

**Performance Issues**

* Reduce max_num_hands parameter
* Enable image downsampling
* Use lower detection confidence

Integration Examples
-------------------

Basic Hand Tracking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker, GestureRecognizer
   
   # Setup
   client = UnlookClient()
   client.connect()
   
   tracker = HandTracker(calibration_file="calibration.json")
   gesture_recognizer = GestureRecognizer()
   
   # Main loop
   while True:
       # Capture stereo frames
       left_frame = client.camera.capture(left_camera_id)
       right_frame = client.camera.capture(right_camera_id)
       
       # Track hands
       results = tracker.track_hands_3d(left_frame, right_frame)
       
       # Recognize gestures
       if results['3d_keypoints']:
           for keypoints_3d in results['3d_keypoints']:
               gesture_type, confidence = gesture_recognizer.recognize_gestures_3d(
                   keypoints_3d
               )
               print(f"Gesture: {gesture_type.value} ({confidence:.2f})")
       
       # Display results
       # ... visualization code ...

Enhanced Demo with LED Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from unlook.examples.handpose_demo import OpenCVHandposeDemo
   
   # Run enhanced demo with point projection
   demo = OpenCVHandposeDemo(
       led1_intensity=50,          # Point projection
       led2_intensity=50,          # Flood illumination
       enable_point_projection=True
   )
   demo.run()

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High-precision tracking
   tracker = HandTracker(
       calibration_file="high_precision_calibration.json",
       max_num_hands=1,              # Focus on single hand
       detection_confidence=0.8,     # Higher accuracy
       tracking_confidence=0.8,      # More stable tracking
       left_camera_mirror_mode=False,
       right_camera_mirror_mode=False
   )
   
   # Gesture recognition with custom thresholds
   gesture_recognizer = GestureRecognizer(
       gesture_threshold=0.9  # Very confident gestures only
   )

Troubleshooting
--------------

Debugging Tools
~~~~~~~~~~~~~~

Enable verbose logging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger('unlook.client.scanning.handpose')

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   start_time = time.time()
   results = tracker.track_hands_3d(left_frame, right_frame)
   processing_time = time.time() - start_time
   
   print(f"Processing time: {processing_time*1000:.1f}ms")
   print(f"FPS: {1.0/processing_time:.1f}")

Common Error Messages
~~~~~~~~~~~~~~~~~~~~

* ``"No valid calibration loaded"`` - Missing or invalid calibration file
* ``"HandTracker.track_hands_3d() got an unexpected keyword argument"`` - Check method parameters
* ``"'GestureRecognizer' object has no attribute 'recognize_gesture_3d'"`` - Use correct method name (plural: recognize_gestures_3d)

Best Practices
--------------

Lighting Conditions
~~~~~~~~~~~~~~~~~~

* Use both LED1 (point projection) and LED2 (flood illumination)
* Avoid harsh shadows and reflections
* Maintain consistent ambient lighting
* Consider IR illumination for low-light conditions

Hand Positioning
~~~~~~~~~~~~~~~

* Optimal distance: 40-80cm from cameras
* Keep hands within camera field of view
* Avoid occlusion between hands
* Face palms toward cameras for best detection

System Performance
~~~~~~~~~~~~~~~~~

* Use appropriate detection confidence (0.6-0.8)
* Limit max_num_hands based on application needs
* Enable preprocessing optimizations
* Monitor memory usage in long-running applications

See Also
--------

* :doc:`../examples/handpose_tracking` - Complete examples
* :doc:`../user_guide/camera_configuration` - Camera setup guide
* :doc:`./projector` - LED controller documentation
* :doc:`../troubleshooting` - Common issues and solutions