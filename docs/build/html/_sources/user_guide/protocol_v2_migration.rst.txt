Protocol V2 Migration Guide
===========================

This guide explains the differences between Protocol V1 and V2, and how to migrate your applications.

Overview
--------

Protocol V2 introduces significant performance improvements and new features:

* **Bandwidth Optimization**: Up to 80% reduction in network traffic
* **Server-Side Preprocessing**: GPU-accelerated image processing on Raspberry Pi
* **Multi-Camera Support**: Synchronized capture from multiple cameras
* **Enhanced Compression**: Adaptive JPEG compression with quality assessment
* **ROI Detection**: Automatic region-of-interest extraction
* **Pattern Synchronization**: Precise projector-camera timing

Key Differences
---------------

Network Protocol
~~~~~~~~~~~~~~~~

========================  ================  ==================
Feature                   Protocol V1       Protocol V2
========================  ================  ==================
Image Compression         Client-side       Server-side
Multi-camera              Sequential        Simultaneous  
ROI Processing            None              Automatic
Bandwidth Usage           High (100%)       Low (20-40%)
Preprocessing             Client CPU        Server GPU
========================  ================  ==================

Message Types
~~~~~~~~~~~~~

New message types in Protocol V2:

.. code-block:: python

    # V2-specific message types
    MessageType.MULTI_CAMERA_CAPTURE     # Synchronized capture
    MessageType.MULTI_CAMERA_RESPONSE    # Multi-camera response
    MessageType.SET_PREPROCESSING_CONFIG  # Configure server preprocessing
    MessageType.GET_PREPROCESSING_STATS  # Get processing statistics

API Changes
-----------

Client Initialization
~~~~~~~~~~~~~~~~~~~~~

**Protocol V1:**

.. code-block:: python

    from unlook.client.scanner.scanner import UnlookClient
    
    client = UnlookClient(auto_discover=True)

**Protocol V2:**

.. code-block:: python

    from unlook.client.scanner.scanner import UnlookClient
    from unlook.core.constants import PreprocessingVersion
    
    client = UnlookClient(
        auto_discover=True,
        preprocessing_version=PreprocessingVersion.V2_ENHANCED
    )

Multi-Camera Capture
~~~~~~~~~~~~~~~~~~~~

**Protocol V1 (Sequential):**

.. code-block:: python

    # Capture cameras one by one
    left_image = client.camera.capture_image('cam0')
    right_image = client.camera.capture_image('cam1')

**Protocol V2 (Synchronized):**

.. code-block:: python

    # Synchronized multi-camera capture
    images = client.camera.capture_multi_camera(['cam0', 'cam1'])
    left_image = images['cam0']
    right_image = images['cam1']

Server Configuration
~~~~~~~~~~~~~~~~~~~~

**Protocol V2 Server Setup:**

.. code-block:: bash

    # Start server with V2 features
    python unlook/server_bootstrap.py \
        --enable-protocol-v2 \
        --enable-pattern-preprocessing \
        --enable-sync

Preprocessing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol V2 Preprocessing:**

.. code-block:: python

    from unlook.server.hardware.gpu_preprocessing import PreprocessingConfig
    
    # Configure server-side preprocessing
    config = PreprocessingConfig(
        lens_correction=True,
        roi_detection=True,
        adaptive_quality=True,
        edge_preserving_filter=True
    )
    
    client.set_preprocessing_config(config)

Migration Steps
---------------

Step 1: Update Server
~~~~~~~~~~~~~~~~~~~~~

1. **Update Server Bootstrap**:

   .. code-block:: bash
   
       # Old V1 server
       python unlook/server_bootstrap.py
       
       # New V2 server  
       python unlook/server_bootstrap.py --enable-protocol-v2

2. **Verify V2 Features**:

   .. code-block:: python
   
       if client.is_protocol_v2_enabled():
           print("✅ Protocol V2 active")
       else:
           print("❌ Using Protocol V1")

Step 2: Update Client Code
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Enable V2 in Client**:

   .. code-block:: python
   
       from unlook.core.constants import PreprocessingVersion
       
       client = UnlookClient(
           preprocessing_version=PreprocessingVersion.V2_ENHANCED
       )

2. **Use Multi-Camera API**:

   .. code-block:: python
   
       # Replace sequential capture
       images = client.camera.capture_multi_camera(['cam0', 'cam1'])

3. **Configure Preprocessing**:

   .. code-block:: python
   
       # Enable server-side optimizations
       config = PreprocessingConfig(roi_detection=True)
       client.set_preprocessing_config(config)

Step 3: Test Performance
~~~~~~~~~~~~~~~~~~~~~~~~

Compare performance between V1 and V2:

.. code-block:: python

    import time
    
    # Measure capture time
    start = time.time()
    images = client.camera.capture_multi_camera(['cam0', 'cam1'])
    duration = time.time() - start
    
    print(f"V2 capture time: {duration:.2f}s")
    
    # Check bandwidth usage
    stats = client.get_preprocessing_stats()
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")

Compatibility
-------------

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

Protocol V2 maintains backward compatibility:

* **V1 Clients** can connect to **V2 Servers** (with reduced performance)  
* **V2 Clients** can connect to **V1 Servers** (auto-fallback)
* **Mixed Deployments** are supported during migration

Version Detection
~~~~~~~~~~~~~~~~~

Automatic version detection and fallback:

.. code-block:: python

    # Client automatically detects server capabilities
    client = UnlookClient(auto_discover=True)
    
    if client.server_supports_v2():
        # Use V2 features
        client.enable_protocol_v2()
    else:
        # Fallback to V1
        print("Server only supports V1")

Troubleshooting
---------------

Common Migration Issues
~~~~~~~~~~~~~~~~~~~~~~

1. **"Protocol V2 not enabled"**:

   * Check server started with ``--enable-protocol-v2``
   * Verify client uses ``PreprocessingVersion.V2_ENHANCED``

2. **Slow Performance**:

   * Ensure server has GPU acceleration enabled
   * Check ``roi_detection=True`` in preprocessing config

3. **Connection Failures**:

   * V2 uses different ports - check firewall settings
   * Verify server and client are on same network

4. **Image Quality Issues**:

   * Adjust ``compression_level`` in preprocessing config
   * Check ``adaptive_quality`` settings

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor V2 performance improvements:

.. code-block:: python

    # Get detailed statistics
    stats = client.get_preprocessing_stats()
    
    print(f"ROI detection: {stats.get('roi_applied', False)}")
    print(f"Compression ratio: {stats.get('compression_ratio', 1):.1f}x")
    print(f"Processing time: {stats.get('processing_time_ms', 0)}ms")

Expected Improvements
---------------------

Typical performance gains with Protocol V2:

==================  ===============  =================
Metric              Protocol V1      Protocol V2
==================  ===============  =================
Network Bandwidth   100%             20-40%
Capture Latency     200-500ms        50-100ms  
Multi-camera Sync   ±50ms            ±1ms
Processing Load     Client CPU       Server GPU
Total Scan Time     30-60 seconds    10-20 seconds
==================  ===============  =================

See Also
--------

* :doc:`cpu_optimizations` - Client-side performance improvements
* :doc:`gpu_acceleration` - Server-side GPU acceleration
* :doc:`troubleshooting` - General troubleshooting guide