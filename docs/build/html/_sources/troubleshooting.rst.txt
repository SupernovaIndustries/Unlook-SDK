Troubleshooting
==============

This guide provides solutions to common issues you might encounter when using the UnLook SDK.

Connection Issues
---------------

Scanner Not Found
^^^^^^^^^^^^^^^^^^^

If the UnlookClient cannot find any scanners:

1. **Check network connectivity**: Ensure the scanner and the computer running the client are on the same network.

2. **Check scanner power**: Verify the scanner is powered on and operating normally.

3. **Firewall issues**: Make sure your firewall isn't blocking the discovery ports (UDP port 5353 for mDNS).

4. **Try direct connection**: If auto-discovery isn't working, try connecting directly using the IP address:

   .. code-block:: python

      client = UnlookClient()
      client.connect("192.168.1.100", 5555)  # Replace with your scanner's IP and port

5. **Restart services**: Try restarting both the client application and the scanner hardware.

Connection Timeouts
^^^^^^^^^^^^^^^^^

If connections time out:

1. **Network latency**: Check for high network latency or packet loss.

2. **Scanner load**: The scanner may be busy processing other requests.

3. **Increase timeouts**: Try increasing connection timeout values:

   .. code-block:: python

      client = UnlookClient(connection_timeout=30.0)  # 30 seconds

Camera Issues
-----------

Camera Not Available
^^^^^^^^^^^^^^^^^^

If the camera cannot be accessed:

1. **Check camera ID**: Verify you're using the correct camera ID.

2. **Camera in use**: Another process might be using the camera.

3. **Check permissions**: The scanner may not have permission to access the camera.

4. **Debug camera state**:

   .. code-block:: python

      # Get camera information
      camera_info = client.camera.get_camera_info()
      print(camera_info)

Camera Images Dark/Blurry
^^^^^^^^^^^^^^^^^^^^^^^

If images from the camera are poor quality:

1. **Adjust exposure**: Try modifying the exposure settings:

   .. code-block:: python

      from unlook.client.camera_config import CameraConfig
      
      config = CameraConfig()
      config.exposure_time = 20000  # in microseconds
      config.gain = 1.5
      
      client.camera.apply_camera_config("camera_id", config)

2. **Check focus**: Ensure the camera's focus is properly adjusted.

3. **Check lighting**: Make sure there's sufficient lighting in the scanning area.

Projector Issues
--------------

Projector Not Working
^^^^^^^^^^^^^^^^^^^

If the projector isn't projecting patterns:

1. **Check projector status**:

   .. code-block:: python

      projector_status = client.projector.get_status()
      print(projector_status)

2. **Try a basic pattern**:

   .. code-block:: python

      client.projector.project_pattern({
          "pattern_type": "solid_field", 
          "color": "White"
      })

3. **Restart projector**:

   .. code-block:: python

      client.projector.restart()

Patterns Not Synchronized
^^^^^^^^^^^^^^^^^^^^^^^

If patterns and camera captures aren't synchronized:

1. **Enable synchronization**:

   .. code-block:: python

      client.projector.start_pattern_sequence(
          patterns=patterns,
          interval=0.5,
          sync_with_camera=True
      )

2. **Check timing**: Ensure the interval between patterns allows enough time for camera exposure.

3. **Check logs**: Look for timing or synchronization warnings in the logs.

Scanning Issues
-------------

Poor Point Cloud Quality
^^^^^^^^^^^^^^^^^^^^^

If your scans produce poor-quality point clouds:

1. **Calibration**: Ensure the cameras are properly calibrated:

   .. code-block:: python

      # Load and check calibration
      from unlook.client.scanning.calibration import load_calibration, verify_calibration
      
      calibration_data = load_calibration("stereo_calibration.json")
      print(f"Calibration valid: {calibration_data is not None}")

2. **Pattern quality**: Check that patterns are clearly visible and in focus.

3. **Surface properties**: Some shiny or transparent surfaces are challenging to scan.

4. **Adjust scan parameters**:

   .. code-block:: python

      from unlook.client.scanning import StaticScanConfig
      from unlook.client.scan_config import PatternType
      
      config = StaticScanConfig()
      config.pattern_type = PatternType.PHASE_SHIFT  # Try different pattern types
      config.quality = "high"  # Increase quality
      config.set_noise_threshold(0.1)  # Adjust noise filtering

5. **Check distance**: Ensure the object is within the optimal scanning range (typically 30-60cm).

Reconstruction Errors
^^^^^^^^^^^^^^^^^^

If you encounter errors during 3D reconstruction:

1. **Check correspondence matching**: Verify that patterns are properly decoded.

2. **Check triangulation**: Look for issues in the point triangulation process.

3. **Memory issues**: For large scans, you might be running out of memory. Try reducing resolution or using more efficient settings.

4. **GPU acceleration**: If available, enable GPU acceleration to improve performance.

Performance Issues
----------------

Slow Scanning
^^^^^^^^^^^

If scanning is slower than expected:

1. **Check resolution**: Lower the resolution for faster performance:

   .. code-block:: python

      config.set_resolution(640, 480)  # Lower resolution

2. **Pattern complexity**: Use simpler patterns for faster scanning:

   .. code-block:: python

      config.pattern_type = PatternType.GRAY_CODE  # Faster than phase shift

3. **GPU acceleration**: Enable GPU acceleration if available:

   .. code-block:: python

      config.use_gpu = True
      
4. **Compression**: Use more compression for faster data transfer:

   .. code-block:: python

      config.set_compression(CompressionFormat.JPEG, jpeg_quality=75)

5. **Profiling**: Use the built-in profiler to identify bottlenecks:

   .. code-block:: python

      scanner.enable_profiling()
      # Perform scan
      profile_data = scanner.get_profile_data()
      print(profile_data)

Memory Leaks
^^^^^^^^^^

If you're experiencing memory growth:

1. **Close resources**: Ensure you're properly closing resources:

   .. code-block:: python

      # When done
      scanner.close()
      client.disconnect()

2. **Garbage collection**: Force garbage collection after large operations:

   .. code-block:: python

      import gc
      
      # After large operations
      gc.collect()

3. **Image handling**: Avoid keeping many large images in memory; process and release them.

System Logs
----------

Accessing Logs
^^^^^^^^^^^^

Enable detailed logging to troubleshoot issues:

.. code-block:: python

   import logging
   
   # Set up logging
   logging.basicConfig(level=logging.DEBUG)
   
   # More specific loggers
   logging.getLogger('unlook.client').setLevel(logging.DEBUG)
   logging.getLogger('unlook.core').setLevel(logging.DEBUG)

Log File Location
^^^^^^^^^^^^^^^

Log files are typically stored in:

- **Windows**: ``%APPDATA%\unlook\logs\``
- **Linux**: ``~/.local/share/unlook/logs/``
- **macOS**: ``~/Library/Application Support/unlook/logs/``

You can also specify a custom log location:

.. code-block:: python

   import logging
   
   # Set up file logging
   file_handler = logging.FileHandler('unlook_debug.log')
   file_handler.setLevel(logging.DEBUG)
   
   # Add the handler to the logger
   logger = logging.getLogger('unlook')
   logger.addHandler(file_handler)

Getting Help
----------

If you're still experiencing issues:

1. **Check the examples**: Review the example scripts in the SDK for proper usage patterns.

2. **SDK Documentation**: Refer to the detailed API documentation.

3. **GitHub Issues**: Check if your issue has been reported or create a new issue at https://github.com/SupernovaIndustries/unlook/issues.

4. **Contact Support**: For commercial support options, contact support@supernovaindustries.it.

5. **Gather information**: When reporting issues, include:
   - SDK version
   - Python version
   - Operating system
   - Hardware details
   - Error messages
   - Steps to reproduce the issue