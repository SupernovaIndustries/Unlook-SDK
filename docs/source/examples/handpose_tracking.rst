Handpose Tracking Examples
==========================

This section provides comprehensive examples for implementing hand tracking and gesture recognition using the UnLook SDK.

Basic Hand Tracking
-------------------

Simple 3D Hand Detection
~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates basic 3D hand tracking with stereo cameras:

.. code-block:: python

   #!/usr/bin/env python3
   """Basic 3D hand tracking example."""
   
   import cv2
   import numpy as np
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker
   
   def main():
       # Connect to UnLook scanner
       client = UnlookClient()
       client.start_discovery()
       scanners = client.get_discovered_scanners()
       client.connect(scanners[0])
       
       # Initialize hand tracker
       tracker = HandTracker(
           calibration_file=None,  # Auto-load calibration
           max_num_hands=2,
           detection_confidence=0.6,
           tracking_confidence=0.6
       )
       
       # Get camera IDs
       cameras = client.camera.get_cameras()
       left_camera_id = cameras[0]['id']
       right_camera_id = cameras[1]['id']
       
       print("Starting basic hand tracking...")
       print("Press 'q' to quit")
       
       while True:
           # Capture stereo frames
           left_frame = client.camera.capture(left_camera_id)
           right_frame = client.camera.capture(right_camera_id)
           
           if left_frame is None or right_frame is None:
               continue
           
           # Track hands in 3D
           results = tracker.track_hands_3d(left_frame, right_frame)
           
           # Display results
           print(f"Detected {len(results['3d_keypoints'])} hands")
           
           for i, keypoints_3d in enumerate(results['3d_keypoints']):
               wrist_pos = keypoints_3d[0]  # Wrist is landmark 0
               print(f"Hand {i} wrist: ({wrist_pos[0]:.1f}, {wrist_pos[1]:.1f}, {wrist_pos[2]:.1f}) mm")
           
           # Simple visualization
           display_frame = np.hstack([left_frame, right_frame])
           cv2.imshow('Stereo View', display_frame)
           
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       # Cleanup
       tracker.close()
       client.disconnect()
       cv2.destroyAllWindows()
   
   if __name__ == "__main__":
       main()

Enhanced Hand Tracking with Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example adds hand skeleton visualization and gesture display:

.. code-block:: python

   #!/usr/bin/env python3
   """Enhanced hand tracking with visualization."""
   
   import cv2
   import numpy as np
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker, GestureRecognizer, GestureType
   
   # Hand skeleton connections
   HAND_CONNECTIONS = [
       # Thumb
       [0, 1], [1, 2], [2, 3], [3, 4],
       # Index finger
       [0, 5], [5, 6], [6, 7], [7, 8],
       # Middle finger  
       [0, 9], [9, 10], [10, 11], [11, 12],
       # Ring finger
       [0, 13], [13, 14], [14, 15], [15, 16],
       # Pinky
       [0, 17], [17, 18], [18, 19], [19, 20],
       # Palm
       [5, 9], [9, 13], [13, 17]
   ]
   
   def draw_hand_skeleton(image, keypoints_2d, color=(0, 255, 0)):
       """Draw hand skeleton on image."""
       if keypoints_2d is None or len(keypoints_2d) < 21:
           return
       
       h, w = image.shape[:2]
       
       # Convert normalized coordinates to pixels
       pixel_coords = []
       for kp in keypoints_2d:
           x = int(kp[0] * w)
           y = int(kp[1] * h)
           pixel_coords.append((x, y))
       
       # Draw connections
       for connection in HAND_CONNECTIONS:
           start_point = pixel_coords[connection[0]]
           end_point = pixel_coords[connection[1]]
           cv2.line(image, start_point, end_point, color, 2)
       
       # Draw landmarks
       for i, point in enumerate(pixel_coords):
           if i in [4, 8, 12, 16, 20]:  # Fingertips
               cv2.circle(image, point, 6, (0, 0, 255), -1)
           else:
               cv2.circle(image, point, 4, color, -1)
   
   def main():
       # Setup
       client = UnlookClient()
       client.start_discovery()
       scanners = client.get_discovered_scanners()
       client.connect(scanners[0])
       
       tracker = HandTracker()
       gesture_recognizer = GestureRecognizer()
       
       cameras = client.camera.get_cameras()
       left_camera_id = cameras[0]['id']
       right_camera_id = cameras[1]['id']
       
       print("Enhanced hand tracking with visualization")
       print("Supported gestures: Open Palm, Fist, Pointing, Peace, Thumbs Up/Down, OK, Rock, Pinch")
       
       while True:
           # Capture frames
           left_frame = client.camera.capture(left_camera_id)
           right_frame = client.camera.capture(right_camera_id)
           
           if left_frame is None or right_frame is None:
               continue
           
           # Track hands
           results = tracker.track_hands_3d(left_frame, right_frame)
           
           # Create display copies
           left_display = left_frame.copy()
           right_display = right_frame.copy()
           
           # Draw hand skeletons
           for keypoints_2d in results.get('2d_left', []):
               draw_hand_skeleton(left_display, keypoints_2d, (0, 255, 0))
           
           for keypoints_2d in results.get('2d_right', []):
               draw_hand_skeleton(right_display, keypoints_2d, (0, 0, 255))
           
           # Recognize and display gestures
           gesture_text = "No gesture"
           if results['3d_keypoints']:
               for keypoints_3d in results['3d_keypoints']:
                   gesture_type, confidence = gesture_recognizer.recognize_gestures_3d(keypoints_3d)
                   
                   if gesture_type != GestureType.UNKNOWN and confidence > 0.7:
                       gesture_name = gesture_type.value.replace('_', ' ').title()
                       gesture_text = f"{gesture_name} ({confidence:.2f})"
                       break
           
           # Add text overlay
           cv2.putText(left_display, f"Gesture: {gesture_text}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
           
           cv2.putText(left_display, f"Hands: {len(results['3d_keypoints'])}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
           
           # Display
           display_frame = np.hstack([left_display, right_display])
           cv2.imshow('Enhanced Hand Tracking', display_frame)
           
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       # Cleanup
       tracker.close()
       client.disconnect()
       cv2.destroyAllWindows()
   
   if __name__ == "__main__":
       main()

LED-Enhanced Hand Tracking
--------------------------

Using Point Projection for Better Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to use LED illumination for enhanced hand tracking:

.. code-block:: python

   #!/usr/bin/env python3
   """LED-enhanced hand tracking with point projection."""
   
   import time
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker
   from unlook.client.projector import LEDController
   
   def main():
       # Connect to scanner
       client = UnlookClient()
       client.start_discovery()
       time.sleep(3)
       scanners = client.get_discovered_scanners()
       client.connect(scanners[0])
       
       # Initialize LED controller
       led_controller = LEDController(client)
       
       # Configure optimal LED settings for hand tracking
       if led_controller.led_available:
           # LED1: Point projection for enhanced triangulation
           # LED2: Flood illumination for general detection
           led_controller.set_intensity(
               led1=50,   # Point projection
               led2=50    # Flood illumination
           )
           print("LED illumination configured:")
           print("  LED1: 50mA (point projection)")
           print("  LED2: 50mA (flood illumination)")
       else:
           print("LED control not available")
       
       # Initialize hand tracker
       tracker = HandTracker(
           max_num_hands=1,           # Focus on single hand for precision
           detection_confidence=0.7,  # Higher confidence with LED illumination
           tracking_confidence=0.7
       )
       
       cameras = client.camera.get_cameras()
       left_camera_id = cameras[0]['id']
       right_camera_id = cameras[1]['id']
       
       print("Starting LED-enhanced hand tracking...")
       
       frame_count = 0
       start_time = time.time()
       
       try:
           while True:
               # Capture frames
               left_frame = client.camera.capture(left_camera_id)
               right_frame = client.camera.capture(right_camera_id)
               
               if left_frame is None or right_frame is None:
                   continue
               
               # Track hands with LED enhancement
               results = tracker.track_hands_3d(left_frame, right_frame)
               
               # Display tracking results
               if results['3d_keypoints']:
                   for i, keypoints_3d in enumerate(results['3d_keypoints']):
                       wrist = keypoints_3d[0]
                       fingertip = keypoints_3d[8]  # Index fingertip
                       
                       print(f"Hand {i}:")
                       print(f"  Wrist: ({wrist[0]:.1f}, {wrist[1]:.1f}, {wrist[2]:.1f}) mm")
                       print(f"  Index tip: ({fingertip[0]:.1f}, {fingertip[1]:.1f}, {fingertip[2]:.1f}) mm")
                       
                       # Calculate hand size (wrist to middle fingertip)
                       middle_tip = keypoints_3d[12]
                       hand_size = np.linalg.norm(middle_tip - wrist)
                       print(f"  Hand size: {hand_size:.1f} mm")
               
               frame_count += 1
               
               # Print FPS every 30 frames
               if frame_count % 30 == 0:
                   elapsed = time.time() - start_time
                   fps = frame_count / elapsed
                   print(f"FPS: {fps:.1f}")
               
               time.sleep(0.033)  # ~30 FPS
       
       except KeyboardInterrupt:
           print("Stopping...")
       
       finally:
           # Turn off LEDs
           if led_controller.led_available:
               led_controller.set_intensity(0, 0)
               print("LEDs turned off")
           
           tracker.close()
           client.disconnect()
   
   if __name__ == "__main__":
       main()

Gesture Recognition Examples
---------------------------

Real-time Gesture Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """Real-time gesture recognition example."""
   
   import cv2
   import time
   from collections import deque
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker, GestureRecognizer, GestureType
   
   class GestureTracker:
       def __init__(self, window_size=10):
           self.gesture_history = deque(maxlen=window_size)
           self.last_stable_gesture = GestureType.UNKNOWN
           self.gesture_start_time = 0
           self.min_gesture_duration = 0.5  # seconds
       
       def update(self, gesture_type, confidence):
           """Update gesture history and return stable gesture."""
           current_time = time.time()
           
           # Add to history
           self.gesture_history.append((gesture_type, confidence, current_time))
           
           # Find most common gesture in recent history
           if len(self.gesture_history) >= 5:  # Need at least 5 frames
               gesture_counts = {}
               total_confidence = {}
               
               for g_type, g_conf, g_time in self.gesture_history:
                   if current_time - g_time < 1.0:  # Only last 1 second
                       if g_type not in gesture_counts:
                           gesture_counts[g_type] = 0
                           total_confidence[g_type] = 0
                       gesture_counts[g_type] += 1
                       total_confidence[g_type] += g_conf
               
               # Find most frequent gesture with good confidence
               best_gesture = GestureType.UNKNOWN
               best_score = 0
               
               for g_type, count in gesture_counts.items():
                   avg_confidence = total_confidence[g_type] / count
                   score = count * avg_confidence
                   
                   if score > best_score and avg_confidence > 0.6:
                       best_gesture = g_type
                       best_score = score
               
               # Check if gesture changed
               if best_gesture != self.last_stable_gesture:
                   self.last_stable_gesture = best_gesture
                   self.gesture_start_time = current_time
                   return best_gesture, True  # New gesture
               elif current_time - self.gesture_start_time > self.min_gesture_duration:
                   return best_gesture, False  # Stable gesture
           
           return GestureType.UNKNOWN, False
   
   def main():
       # Setup
       client = UnlookClient()
       client.start_discovery()
       time.sleep(2)
       scanners = client.get_discovered_scanners()
       client.connect(scanners[0])
       
       tracker = HandTracker()
       gesture_recognizer = GestureRecognizer(gesture_threshold=0.6)
       gesture_tracker = GestureTracker()
       
       cameras = client.camera.get_cameras()
       left_camera_id = cameras[0]['id']
       right_camera_id = cameras[1]['id']
       
       print("Real-time Gesture Recognition")
       print("=" * 40)
       print("Supported gestures:")
       for gesture_type in GestureType:
           if gesture_type != GestureType.UNKNOWN:
               print(f"  - {gesture_type.value.replace('_', ' ').title()}")
       print("=" * 40)
       
       last_gesture_name = ""
       
       while True:
           # Capture and process
           left_frame = client.camera.capture(left_camera_id)
           right_frame = client.camera.capture(right_camera_id)
           
           if left_frame is None or right_frame is None:
               continue
           
           results = tracker.track_hands_3d(left_frame, right_frame)
           
           # Process gestures
           current_gesture = GestureType.UNKNOWN
           current_confidence = 0.0
           
           if results['3d_keypoints']:
               for keypoints_3d in results['3d_keypoints']:
                   gesture_type, confidence = gesture_recognizer.recognize_gestures_3d(keypoints_3d)
                   if confidence > current_confidence:
                       current_gesture = gesture_type
                       current_confidence = confidence
           
           # Update gesture tracker
           stable_gesture, is_new = gesture_tracker.update(current_gesture, current_confidence)
           
           # Display stable gestures
           if stable_gesture != GestureType.UNKNOWN:
               gesture_name = stable_gesture.value.replace('_', ' ').title()
               
               if is_new or gesture_name != last_gesture_name:
                   timestamp = time.strftime("%H:%M:%S")
                   print(f"[{timestamp}] Gesture: {gesture_name}")
                   last_gesture_name = gesture_name
           
           # Simple visualization
           if results['2d_left']:
               left_display = left_frame.copy()
               h, w = left_display.shape[:2]
               
               # Draw simple hand indicator
               for keypoints_2d in results['2d_left']:
                   wrist = keypoints_2d[0]
                   x, y = int(wrist[0] * w), int(wrist[1] * h)
                   cv2.circle(left_display, (x, y), 10, (0, 255, 0), -1)
               
               # Add gesture text
               gesture_display = last_gesture_name if last_gesture_name else "No Gesture"
               cv2.putText(left_display, gesture_display, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
               
               cv2.imshow('Gesture Recognition', left_display)
           
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       # Cleanup
       tracker.close()
       client.disconnect()
       cv2.destroyAllWindows()
   
   if __name__ == "__main__":
       main()

Advanced Applications
---------------------

Hand-Computer Interaction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """Hand-computer interaction example using gestures."""
   
   import cv2
   import pyautogui  # pip install pyautogui
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker, GestureRecognizer, GestureType
   
   # Disable pyautogui failsafe
   pyautogui.FAILSAFE = False
   
   class HandController:
       def __init__(self):
           self.screen_width, self.screen_height = pyautogui.size()
           self.last_gesture = GestureType.UNKNOWN
           self.cursor_smoothing = 0.3
           self.last_cursor_pos = None
       
       def map_hand_to_screen(self, hand_pos_3d, workspace_bounds):
           """Map 3D hand position to screen coordinates."""
           # Define workspace bounds in mm
           x_min, x_max = workspace_bounds['x']
           y_min, y_max = workspace_bounds['y']
           z_min, z_max = workspace_bounds['z']
           
           # Map to screen space
           screen_x = int(np.interp(hand_pos_3d[0], [x_min, x_max], [0, self.screen_width]))
           screen_y = int(np.interp(hand_pos_3d[1], [y_min, y_max], [0, self.screen_height]))
           
           # Apply smoothing
           if self.last_cursor_pos:
               screen_x = int(self.last_cursor_pos[0] * (1 - self.cursor_smoothing) + 
                            screen_x * self.cursor_smoothing)
               screen_y = int(self.last_cursor_pos[1] * (1 - self.cursor_smoothing) + 
                            screen_y * self.cursor_smoothing)
           
           self.last_cursor_pos = (screen_x, screen_y)
           return screen_x, screen_y
       
       def handle_gesture(self, gesture_type):
           """Handle gesture commands."""
           if gesture_type == self.last_gesture:
               return  # Ignore repeated gestures
           
           self.last_gesture = gesture_type
           
           if gesture_type == GestureType.POINTING:
               # Move cursor
               pass  # Handled in main loop
           elif gesture_type == GestureType.CLOSED_FIST:
               pyautogui.click()
               print("Click!")
           elif gesture_type == GestureType.PEACE:
               pyautogui.rightClick()
               print("Right click!")
           elif gesture_type == GestureType.THUMBS_UP:
               pyautogui.scroll(3)
               print("Scroll up!")
           elif gesture_type == GestureType.THUMBS_DOWN:
               pyautogui.scroll(-3)
               print("Scroll down!")
           elif gesture_type == GestureType.OK:
               pyautogui.doubleClick()
               print("Double click!")
   
   def main():
       # Setup
       client = UnlookClient()
       client.start_discovery()
       time.sleep(2)
       scanners = client.get_discovered_scanners()
       client.connect(scanners[0])
       
       tracker = HandTracker(max_num_hands=1)
       gesture_recognizer = GestureRecognizer()
       controller = HandController()
       
       cameras = client.camera.get_cameras()
       left_camera_id = cameras[0]['id']
       right_camera_id = cameras[1]['id']
       
       # Define interaction workspace (adjust based on your setup)
       workspace_bounds = {
           'x': [-100, 100],   # mm left-right
           'y': [-50, 50],     # mm up-down  
           'z': [400, 600]     # mm distance from camera
       }
       
       print("Hand-Computer Interaction")
       print("Gestures:")
       print("  Pointing - Move cursor")
       print("  Fist - Click")
       print("  Peace - Right click") 
       print("  Thumbs Up - Scroll up")
       print("  Thumbs Down - Scroll down")
       print("  OK - Double click")
       print("Press 'q' to quit")
       
       while True:
           # Capture and track
           left_frame = client.camera.capture(left_camera_id)
           right_frame = client.camera.capture(right_camera_id)
           
           if left_frame is None or right_frame is None:
               continue
           
           results = tracker.track_hands_3d(left_frame, right_frame)
           
           if results['3d_keypoints']:
               keypoints_3d = results['3d_keypoints'][0]
               
               # Get hand position (use wrist or index fingertip)
               hand_pos = keypoints_3d[8]  # Index fingertip
               
               # Recognize gesture
               gesture_type, confidence = gesture_recognizer.recognize_gestures_3d(keypoints_3d)
               
               if confidence > 0.7:
                   # Map hand to screen coordinates
                   if gesture_type == GestureType.POINTING:
                       screen_x, screen_y = controller.map_hand_to_screen(hand_pos, workspace_bounds)
                       pyautogui.moveTo(screen_x, screen_y)
                   
                   # Handle other gestures
                   controller.handle_gesture(gesture_type)
           
           # Display tracking status
           if results['2d_left']:
               left_display = left_frame.copy()
               cv2.putText(left_display, f"Hands: {len(results['3d_keypoints'])}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               cv2.imshow('Hand Controller', left_display)
           
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       # Cleanup
       tracker.close()
       client.disconnect()
       cv2.destroyAllWindows()
   
   if __name__ == "__main__":
       main()

Performance Optimization
------------------------

Optimized Real-time Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """Performance-optimized hand tracking example."""
   
   import cv2
   import time
   import threading
   from queue import Queue, Empty
   from unlook import UnlookClient
   from unlook.client.scanning.handpose import HandTracker
   
   class OptimizedHandTracker:
       def __init__(self, client):
           self.client = client
           self.frame_queue = Queue(maxsize=2)
           self.result_queue = Queue(maxsize=2)
           self.processing_thread = None
           self.running = False
           
           # Initialize tracker with optimized settings
           self.tracker = HandTracker(
               max_num_hands=1,           # Single hand for speed
               detection_confidence=0.5,  # Lower for faster detection
               tracking_confidence=0.5    # Lower for faster tracking
           )
           
           # Get cameras
           cameras = client.camera.get_cameras()
           self.left_camera_id = cameras[0]['id']
           self.right_camera_id = cameras[1]['id']
       
       def capture_worker(self):
           """Capture frames in background thread."""
           while self.running:
               try:
                   left_frame = self.client.camera.capture(self.left_camera_id)
                   right_frame = self.client.camera.capture(self.right_camera_id)
                   
                   if left_frame is not None and right_frame is not None:
                       # Downsample for faster processing
                       left_small = cv2.resize(left_frame, (320, 240))
                       right_small = cv2.resize(right_frame, (320, 240))
                       
                       try:
                           self.frame_queue.put((left_small, right_small, left_frame), block=False)
                       except:
                           pass  # Queue full, skip frame
               except Exception as e:
                   print(f"Capture error: {e}")
                   time.sleep(0.1)
       
       def processing_worker(self):
           """Process frames in background thread."""
           while self.running:
               try:
                   left_small, right_small, left_full = self.frame_queue.get(timeout=0.1)
                   
                   # Track hands on downsampled images
                   start_time = time.time()
                   results = self.tracker.track_hands_3d(left_small, right_small)
                   processing_time = time.time() - start_time
                   
                   # Scale results back to full resolution
                   scale_x = left_full.shape[1] / left_small.shape[1]
                   scale_y = left_full.shape[0] / left_small.shape[0]
                   
                   for keypoints_list in [results.get('2d_left', []), results.get('2d_right', [])]:
                       for keypoints in keypoints_list:
                           keypoints[:, 0] *= scale_x
                           keypoints[:, 1] *= scale_y
                   
                   # Add timing info
                   results['processing_time'] = processing_time
                   results['full_frame'] = left_full
                   
                   try:
                       self.result_queue.put(results, block=False)
                   except:
                       pass  # Queue full, skip result
                       
               except Empty:
                   continue
               except Exception as e:
                   print(f"Processing error: {e}")
       
       def start(self):
           """Start background processing."""
           self.running = True
           self.capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
           self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
           self.capture_thread.start()
           self.processing_thread.start()
       
       def stop(self):
           """Stop background processing."""
           self.running = False
           if self.capture_thread:
               self.capture_thread.join(timeout=1.0)
           if self.processing_thread:
               self.processing_thread.join(timeout=1.0)
           self.tracker.close()
       
       def get_latest_result(self):
           """Get latest tracking result (non-blocking)."""
           try:
               return self.result_queue.get(block=False)
           except Empty:
               return None
   
   def main():
       # Setup
       client = UnlookClient()
       client.start_discovery()
       time.sleep(2)
       scanners = client.get_discovered_scanners()
       client.connect(scanners[0])
       
       # Create optimized tracker
       tracker = OptimizedHandTracker(client)
       tracker.start()
       
       print("Optimized hand tracking started")
       print("Performance metrics will be displayed")
       
       frame_count = 0
       fps_start = time.time()
       last_processing_time = 0
       
       try:
           while True:
               # Get latest result
               result = tracker.get_latest_result()
               
               if result:
                   frame_count += 1
                   last_processing_time = result.get('processing_time', 0)
                   
                   # Display frame with annotations
                   display_frame = result['full_frame'].copy()
                   
                   # Draw hand indicators
                   for keypoints_2d in result.get('2d_left', []):
                       if len(keypoints_2d) > 0:
                           h, w = display_frame.shape[:2]
                           wrist = keypoints_2d[0]
                           x, y = int(wrist[0]), int(wrist[1])
                           cv2.circle(display_frame, (x, y), 10, (0, 255, 0), -1)
                   
                   # Add performance metrics
                   current_time = time.time()
                   if frame_count > 0 and current_time > fps_start:
                       fps = frame_count / (current_time - fps_start)
                       cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       cv2.putText(display_frame, f"Process: {last_processing_time*1000:.1f}ms", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       cv2.putText(display_frame, f"Hands: {len(result.get('3d_keypoints', []))}", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
                   cv2.imshow('Optimized Hand Tracking', display_frame)
               
               if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
       
       except KeyboardInterrupt:
           print("Stopping...")
       
       finally:
           tracker.stop()
           client.disconnect()
           cv2.destroyAllWindows()
   
   if __name__ == "__main__":
       main()

Configuration Tips
------------------

Camera Setup
~~~~~~~~~~~~

**Optimal Camera Configuration:**

- **Distance**: 40-80cm from hands
- **Baseline**: 6-12cm between cameras  
- **Resolution**: 640x480 or higher
- **Frame Rate**: 30 FPS recommended
- **Exposure**: Fixed exposure for consistent lighting

**Calibration Requirements:**

- Use checkerboard pattern calibration
- Capture 20+ image pairs from different angles
- Ensure good coverage of the field of view
- Validate reprojection error < 0.5 pixels

LED Optimization
~~~~~~~~~~~~~~~

**Recommended Settings:**

.. code-block:: python

   # For optimal hand tracking
   led_controller.set_intensity(
       led1=50,   # Point projection - enhances triangulation
       led2=50    # Flood illumination - improves detection
   )

**Lighting Considerations:**

- Avoid harsh shadows
- Minimize reflections on skin
- Use consistent ambient lighting
- Consider IR illumination for low-light scenarios

Performance Tuning
~~~~~~~~~~~~~~~~~

**Speed vs Accuracy Trade-offs:**

.. code-block:: python

   # High speed configuration
   tracker = HandTracker(
       max_num_hands=1,
       detection_confidence=0.5,
       tracking_confidence=0.5
   )
   
   # High accuracy configuration  
   tracker = HandTracker(
       max_num_hands=2,
       detection_confidence=0.8,
       tracking_confidence=0.8
   )

**Memory Optimization:**

- Limit history buffer sizes
- Use appropriate image resolution
- Clean up resources properly
- Monitor memory usage in long-running applications

See Also
--------

- :doc:`../api_reference/handpose` - Complete API documentation
- :doc:`camera_calibration` - Stereo calibration guide
- :doc:`../user_guide/camera_configuration` - Camera setup
- :doc:`../troubleshooting` - Common issues and solutions