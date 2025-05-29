Handpose Setup and Configuration
================================

This guide provides step-by-step instructions for setting up 3D hand tracking and gesture recognition with the UnLook SDK.

Overview
--------

The UnLook handpose system provides real-time 3D hand tracking with millimeter-level precision using:

- **Dual stereo cameras** for 3D triangulation
- **LED point projection** (LED1) for enhanced depth sensing  
- **LED flood illumination** (LED2) for consistent lighting
- **MediaPipe integration** for robust hand detection
- **Advanced gesture recognition** with temporal stability

System Requirements
-------------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~

**Minimum Configuration:**

- UnLook scanner with stereo cameras
- LED controller (AS1170 or compatible)
- USB 3.0 connection
- 4GB RAM
- Intel i5 or equivalent CPU

**Recommended Configuration:**

- UnLook scanner with high-resolution stereo cameras
- Hardware-synchronized stereo cameras
- 8GB+ RAM
- Intel i7 or AMD Ryzen 7 CPU
- NVIDIA GPU with CUDA support (optional, for ML enhancements)

Software Requirements
~~~~~~~~~~~~~~~~~~~~

**Required Dependencies:**

.. code-block:: bash

   # Core dependencies
   pip install opencv-python>=4.5.0
   pip install numpy>=1.20.0
   pip install mediapipe>=0.8.9
   
   # Optional ML dependencies
   pip install scikit-learn>=1.0.0
   pip install tensorflow>=2.8.0  # For advanced ML models

**System Packages:**

- Python 3.8+ (Python 3.9+ recommended)
- C++ compiler for native extensions
- OpenCV with Python bindings

Initial Setup
-------------

1. Scanner Connection
~~~~~~~~~~~~~~~~~~~~

First, ensure your UnLook scanner is properly connected and recognized:

.. code-block:: python

   from unlook import UnlookClient
   
   # Test basic connection
   client = UnlookClient()
   client.start_discovery()
   
   import time
   time.sleep(3)
   
   scanners = client.get_discovered_scanners()
   print(f"Found {len(scanners)} scanners:")
   for scanner in scanners:
       print(f"  - {scanner.name} ({scanner.uuid})")
   
   # Connect to first scanner
   if scanners:
       client.connect(scanners[0])
       print(f"Connected to {scanners[0].name}")
   else:
       print("No scanners found - check connections")

2. Camera Configuration
~~~~~~~~~~~~~~~~~~~~~~

Verify stereo camera setup:

.. code-block:: python

   # Get available cameras
   cameras = client.camera.get_cameras()
   print(f"Available cameras: {len(cameras)}")
   
   for i, camera in enumerate(cameras):
       print(f"  Camera {i}: {camera['name']} (ID: {camera['id']})")
   
   # Test image capture
   if len(cameras) >= 2:
       left_id = cameras[0]['id']
       right_id = cameras[1]['id']
       
       left_image = client.camera.capture(left_id)
       right_image = client.camera.capture(right_id)
       
       if left_image is not None and right_image is not None:
           print(f"Stereo capture successful:")
           print(f"  Left: {left_image.shape}")
           print(f"  Right: {right_image.shape}")
       else:
           print("Stereo capture failed - check camera connections")

3. LED Controller Setup
~~~~~~~~~~~~~~~~~~~~~~

Configure LED illumination for optimal hand tracking:

.. code-block:: python

   from unlook.client.projector import LEDController
   
   # Initialize LED controller
   led_controller = LEDController(client)
   
   if led_controller.led_available:
       print("LED control available")
       
       # Set optimal intensities for hand tracking
       success = led_controller.set_intensity(
           led1=50,   # Point projection for enhanced triangulation
           led2=50    # Flood illumination for detection
       )
       
       if success:
           print("LED configuration successful:")
           print("  LED1: 50mA (point projection)")
           print("  LED2: 50mA (flood illumination)")
       else:
           print("LED configuration failed")
   else:
       print("LED control not available on this scanner")

4. Calibration Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Check stereo calibration status:

.. code-block:: python

   from unlook.client.scanning.handpose import HandTracker
   
   # Try to load calibration automatically
   tracker = HandTracker()
   
   if tracker.calibration_loaded:
       print("Stereo calibration loaded successfully")
       print(f"Calibration file: {tracker.calibration_file}")
   else:
       print("No calibration found - 3D tracking will not work")
       print("Please calibrate your stereo cameras first")

Quick Test
----------

Basic Hand Detection Test
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """Quick handpose system test."""
   
   import cv2
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker
   from unlook.client.projector import LEDController
   
   def test_handpose_system():
       print("UnLook Handpose System Test")
       print("=" * 40)
       
       # 1. Connect to scanner
       print("1. Connecting to scanner...")
       client = UnlookClient()
       client.start_discovery()
       
       import time
       time.sleep(3)
       
       scanners = client.get_discovered_scanners()
       if not scanners:
           print("   ❌ No scanners found")
           return False
       
       if not client.connect(scanners[0]):
           print("   ❌ Connection failed")
           return False
       
       print(f"   ✅ Connected to {scanners[0].name}")
       
       # 2. Setup LEDs
       print("2. Configuring LED illumination...")
       led_controller = LEDController(client)
       
       if led_controller.led_available:
           led_controller.set_intensity(50, 50)
           print("   ✅ LEDs configured (50mA each)")
       else:
           print("   ⚠️  LED control not available")
       
       # 3. Initialize hand tracker
       print("3. Initializing hand tracker...")
       try:
           tracker = HandTracker()
           
           if tracker.calibration_loaded:
               print("   ✅ Hand tracker initialized with calibration")
           else:
               print("   ⚠️  Hand tracker initialized without calibration")
               print("      3D tracking will not work without calibration")
       except Exception as e:
           print(f"   ❌ Hand tracker failed: {e}")
           return False
       
       # 4. Test cameras
       print("4. Testing stereo cameras...")
       cameras = client.camera.get_cameras()
       
       if len(cameras) < 2:
           print(f"   ❌ Need 2 cameras, found {len(cameras)}")
           return False
       
       left_id = cameras[0]['id']
       right_id = cameras[1]['id']
       
       left_frame = client.camera.capture(left_id)
       right_frame = client.camera.capture(right_id)
       
       if left_frame is None or right_frame is None:
           print("   ❌ Stereo capture failed")
           return False
       
       print(f"   ✅ Stereo capture successful ({left_frame.shape})")
       
       # 5. Test hand detection
       print("5. Testing hand detection...")
       print("   Place your hand in view and press any key...")
       
       # Show camera view
       while True:
           left_frame = client.camera.capture(left_id)
           right_frame = client.camera.capture(right_id)
           
           if left_frame is not None and right_frame is not None:
               # Test hand tracking
               results = tracker.track_hands_3d(left_frame, right_frame)
               
               # Annotate results
               display_frame = left_frame.copy()
               hands_detected = len(results.get('2d_left', []))
               hands_3d = len(results.get('3d_keypoints', []))
               
               cv2.putText(display_frame, f"2D Hands: {hands_detected}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               cv2.putText(display_frame, f"3D Hands: {hands_3d}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
               if hands_detected > 0:
                   cv2.putText(display_frame, "Hand detected!", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
               
               cv2.imshow('Handpose Test', display_frame)
               
               if cv2.waitKey(1) & 0xFF != 255:
                   break
       
       cv2.destroyAllWindows()
       
       if hands_detected > 0:
           print("   ✅ Hand detection working")
       else:
           print("   ⚠️  No hands detected in test")
       
       # Cleanup
       if led_controller.led_available:
           led_controller.set_intensity(0, 0)
       
       tracker.close()
       client.disconnect()
       
       print("\nSystem test completed!")
       return True
   
   if __name__ == "__main__":
       test_handpose_system()

Camera Calibration
------------------

If you don't have stereo calibration, you'll need to calibrate your cameras:

Calibration Requirements
~~~~~~~~~~~~~~~~~~~~~~~

- **Pattern**: Use a checkerboard pattern (9x6 or 8x6 squares)
- **Print size**: Accurately known square size (e.g., 25mm)
- **Image pairs**: Capture 20+ synchronized stereo pairs
- **Coverage**: Fill the entire field of view
- **Angles**: Capture from multiple angles and depths

Calibration Process
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from unlook.utils.calibration import headless_calibration
   
   # Capture calibration images
   calibration_images_left = []
   calibration_images_right = []
   
   # ... capture image pairs with checkerboard pattern ...
   
   # Run calibration
   calibration_result = headless_calibration.calibrate_stereo_cameras(
       calibration_images_left,
       calibration_images_right,
       checkerboard_size=(9, 6),  # Internal corners
       square_size_mm=25.0        # Square size in millimeters
   )
   
   if calibration_result['success']:
       print("Calibration successful!")
       print(f"Reprojection error: {calibration_result['rms']:.3f} pixels")
       
       # Save calibration
       import json
       with open('stereo_calibration.json', 'w') as f:
           json.dump(calibration_result['calibration_data'], f, indent=2)
   else:
       print("Calibration failed!")

Performance Optimization
------------------------

Lighting Optimization
~~~~~~~~~~~~~~~~~~~~

**LED Configuration:**

.. code-block:: python

   # Standard configuration (balanced)
   led_controller.set_intensity(led1=50, led2=50)
   
   # High precision (more illumination)
   led_controller.set_intensity(led1=100, led2=100)
   
   # Low power (minimal illumination)
   led_controller.set_intensity(led1=25, led2=25)
   
   # Point projection only (for triangulation focus)
   led_controller.set_intensity(led1=75, led2=0)

**Environmental Considerations:**

- Minimize ambient light variations
- Avoid reflective surfaces near the hands
- Ensure consistent background
- Consider IR illumination for challenging lighting

Tracking Parameters
~~~~~~~~~~~~~~~~~~

**High Performance (Speed Priority):**

.. code-block:: python

   tracker = HandTracker(
       max_num_hands=1,           # Single hand only
       detection_confidence=0.5,  # Lower threshold for speed
       tracking_confidence=0.5    # Lower threshold for speed
   )

**High Accuracy (Precision Priority):**

.. code-block:: python

   tracker = HandTracker(
       max_num_hands=2,           # Multiple hands
       detection_confidence=0.8,  # Higher threshold for accuracy
       tracking_confidence=0.8    # Higher threshold for stability
   )

**Balanced Configuration:**

.. code-block:: python

   tracker = HandTracker(
       max_num_hands=2,
       detection_confidence=0.6,  # Good balance
       tracking_confidence=0.6,   # Good balance
       left_camera_mirror_mode=False,  # For world-facing cameras
       right_camera_mirror_mode=False  # For world-facing cameras
   )

Image Processing
~~~~~~~~~~~~~~~

**Enable Preprocessing:**

.. code-block:: python

   # Apply image enhancement before tracking
   def preprocess_image(image):
       if image is None:
           return None
       
       # Enhance contrast and reduce noise
       enhanced = image.copy()
       lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
       l, a, b = cv2.split(lab)
       
       # CLAHE for contrast enhancement
       clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
       cl = clahe.apply(l)
       
       enhanced = cv2.merge((cl, a, b))
       enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
       
       # Bilateral filter for noise reduction
       enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
       
       return enhanced
   
   # Use in tracking loop
   left_enhanced = preprocess_image(left_frame)
   right_enhanced = preprocess_image(right_frame)
   results = tracker.track_hands_3d(left_enhanced, right_enhanced)

Troubleshooting Common Issues
----------------------------

No Hand Detection
~~~~~~~~~~~~~~~~~

**Symptoms:**
- Empty results from hand detection
- No 2D keypoints detected

**Solutions:**

1. **Check lighting conditions:**

   .. code-block:: python

      # Increase LED intensity
      led_controller.set_intensity(led1=100, led2=100)

2. **Lower detection threshold:**

   .. code-block:: python

      tracker = HandTracker(detection_confidence=0.3)

3. **Verify camera focus and exposure**
4. **Check hand positioning** (30-100cm from cameras)

No 3D Tracking
~~~~~~~~~~~~~~

**Symptoms:**
- 2D detection works, but no 3D keypoints
- Empty '3d_keypoints' in results

**Solutions:**

1. **Check calibration:**

   .. code-block:: python

      if not tracker.calibration_loaded:
          print("Calibration required for 3D tracking")

2. **Verify stereo synchronization**
3. **Check baseline distance** (6-12cm recommended)
4. **Ensure proper camera alignment**

Poor Gesture Recognition
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Inconsistent gesture detection
- Low confidence scores

**Solutions:**

1. **Improve hand positioning:**
   - Face palms toward cameras
   - Maintain 40-80cm distance
   - Avoid hand occlusion

2. **Adjust gesture thresholds:**

   .. code-block:: python

      gesture_recognizer = GestureRecognizer(gesture_threshold=0.5)

3. **Use temporal smoothing:**

   .. code-block:: python

      # Implement gesture history tracking
      gesture_history = []
      # ... smooth gestures over multiple frames

Performance Issues
~~~~~~~~~~~~~~~~~

**Symptoms:**
- Low FPS
- High CPU usage
- Memory leaks

**Solutions:**

1. **Optimize tracker settings:**

   .. code-block:: python

      # Reduce computational load
      tracker = HandTracker(
          max_num_hands=1,
          detection_confidence=0.5,
          tracking_confidence=0.5
      )

2. **Use image downsampling:**

   .. code-block:: python

      # Resize images before processing
      left_small = cv2.resize(left_frame, (320, 240))
      right_small = cv2.resize(right_frame, (320, 240))

3. **Implement threading:**

   .. code-block:: python

      # Separate capture and processing threads
      import threading
      from queue import Queue
      
      frame_queue = Queue(maxsize=2)
      result_queue = Queue(maxsize=2)

Advanced Configuration
---------------------

Multi-Camera Setup
~~~~~~~~~~~~~~~~~

For systems with more than 2 cameras:

.. code-block:: python

   # Initialize multiple stereo pairs
   trackers = {}
   for i in range(0, len(cameras), 2):
       if i + 1 < len(cameras):
           pair_name = f"stereo_{i//2}"
           trackers[pair_name] = HandTracker(
               calibration_file=f"calibration_{pair_name}.json"
           )

High-Frequency Tracking
~~~~~~~~~~~~~~~~~~~~~~

For applications requiring >30 FPS:

.. code-block:: python

   # Use streaming mode with callbacks
   def frame_callback(frame, metadata):
       # Process frame immediately
       results = tracker.track_hands_3d_fast(frame)
       # Handle results...
   
   # Start high-frequency streaming
   client.stream.start(camera_id, frame_callback, fps=60)

Integration with Other Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ROS Integration:**

.. code-block:: python

   import rospy
   from geometry_msgs.msg import PoseArray, Pose
   
   def publish_hand_poses(results):
       if results['3d_keypoints']:
           pose_array = PoseArray()
           for keypoints_3d in results['3d_keypoints']:
               for point in keypoints_3d:
                   pose = Pose()
                   pose.position.x = point[0] / 1000.0  # Convert mm to m
                   pose.position.y = point[1] / 1000.0
                   pose.position.z = point[2] / 1000.0
                   pose_array.poses.append(pose)
           
           hand_pose_pub.publish(pose_array)

**Unity Integration:**

.. code-block:: python

   import socket
   import json
   
   def send_to_unity(results):
       if results['3d_keypoints']:
           data = {
               'timestamp': time.time(),
               'hands': []
           }
           
           for keypoints_3d in results['3d_keypoints']:
               hand_data = {
                   'landmarks': keypoints_3d.tolist()
               }
               data['hands'].append(hand_data)
           
           # Send via UDP
           sock.sendto(json.dumps(data).encode(), ('localhost', 9999))

Running the Enhanced Demo
------------------------

The UnLook SDK includes a comprehensive handpose demo with all features:

.. code-block:: bash

   # Run with default settings (recommended)
   python -m unlook.examples.handpose_demo
   
   # Custom LED intensities
   python -m unlook.examples.handpose_demo --led1-intensity 75 --led2-intensity 50
   
   # Disable point projection
   python -m unlook.examples.handpose_demo --no-point-projection
   
   # Enable auto-LED control
   python -m unlook.examples.handpose_demo --auto-led-hand-control
   
   # Debug mode
   python -m unlook.examples.handpose_demo --debug

**Demo Features:**

- Real-time 3D hand tracking visualization
- Interactive LED control sliders
- Gesture recognition with confidence display
- Performance monitoring (FPS, processing time)
- Hand trajectory visualization
- Professional full-screen display

Next Steps
----------

After completing the setup:

1. **Explore Examples**: Try the comprehensive examples in :doc:`../examples/handpose_tracking`
2. **API Reference**: Review the complete API in :doc:`../api_reference/handpose`
3. **Advanced Features**: Learn about gesture customization and ML models
4. **Integration**: Integrate handpose into your applications
5. **Optimization**: Fine-tune performance for your specific use case

See Also
--------

- :doc:`../examples/handpose_tracking` - Complete examples
- :doc:`../api_reference/handpose` - API documentation
- :doc:`camera_configuration` - Camera setup guide
- :doc:`../troubleshooting` - Common issues and solutions