CPU Optimizations for ARM and Mobile Devices
==============================================

The UnLook SDK includes advanced CPU optimizations specifically designed for ARM processors and mobile devices, enabling real-time 3D scanning without GPU requirements.

Overview
--------

The CPU optimization system provides up to **20-30x performance improvement** through:

* **ROI (Region of Interest) extraction** - 70-80% pixel reduction
* **Hierarchical coarse-to-fine matching** - 5-6x speedup  
* **Optimized spatial indexing** - 10x faster lookups
* **SIMD optimizations** - 2-4x speedup on ARM NEON
* **Early termination strategies** - Smart processing limits
* **Tile-based processing** - Better cache efficiency

Target Platforms
-----------------

The optimizations are designed for:

* **ARM Processors**: Surface Pro ARM, Raspberry Pi 4, smartphones
* **Mobile Devices**: Android/iOS with ARM Cortex-A series
* **Embedded Systems**: NVIDIA Jetson, Qualcomm Snapdragon
* **Desktop ARM**: Apple M1/M2, ARM-based laptops

Key Features
------------

Spatial Hash Table
~~~~~~~~~~~~~~~~~~

Replaces dictionary-based indexing with optimized spatial hashing:

.. code-block:: python

    # Automatically enabled for large datasets
    matcher = ImprovedCorrespondenceMatcher()
    matcher.enable_simd = True  # Enable SIMD optimizations

ROI-Based Processing  
~~~~~~~~~~~~~~~~~~~~

Automatically detects object boundaries to reduce processing area:

.. code-block:: python

    # Server-side ROI detection (color-agnostic for IR cameras)
    preprocessing_config = PreprocessingConfig(
        roi_detection=True,
        roi_detection_method="reference_based"
    )

Hierarchical Matching
~~~~~~~~~~~~~~~~~~~~~

Multi-level pyramid matching from coarse to fine detail:

.. code-block:: python

    # Configure hierarchical matching
    matcher.enable_hierarchical = True
    matcher.pyramid_levels = 3       # Standard: 3 levels
    matcher.downsampling_rate = 0.5  # Half resolution per level

Early Termination
~~~~~~~~~~~~~~~~~

Smart stopping criteria to avoid unnecessary processing:

.. code-block:: python

    # Configure early termination
    matcher.enable_early_termination = True
    matcher.early_termination_threshold = 0.95  # Stop at 95% confidence
    matcher.target_match_count = 1000           # Stop at 1000 matches

SIMD Optimizations
~~~~~~~~~~~~~~~~~~

Vectorized operations using ARM NEON instructions:

.. code-block:: python

    # SIMD automatically detected on ARM
    from unlook.client.scanning.reconstruction.simd_optimizations import SIMDOptimizedMatcher
    
    # Automatically uses NEON if available
    simd_matcher = SIMDOptimizedMatcher()

Performance Tuning
-------------------

ARM-Specific Settings
~~~~~~~~~~~~~~~~~~~~~

For ARM processors, use smaller processing windows:

.. code-block:: python

    # Optimize for ARM
    if platform.machine().lower() in ['arm64', 'aarch64']:
        matcher.tile_size = 32              # Smaller tiles
        matcher.target_match_count = 500    # Lower target
        matcher.pyramid_levels = 2          # Fewer levels

Memory Optimization
~~~~~~~~~~~~~~~~~~~

Reduce memory usage for mobile devices:

.. code-block:: python

    # Configure for limited memory
    matcher.enable_local_optimization = True
    matcher.max_direct_pixels = 25000  # Limit pixel processing

Correspondence Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

Enhanced validation using Hamming distance and majority voting:

.. code-block:: python

    # Enable validation (automatically configured)
    matcher.enable_hamming_validation = True   # For Gray code patterns
    matcher.enable_majority_voting = True      # Local consensus

Adaptive Pattern Selection
--------------------------

Intelligent pattern selection based on scene complexity:

.. code-block:: python

    from unlook.client.patterns.adaptive_pattern_strategies import AdaptivePatternGenerator
    
    # Analyze scene and select optimal patterns
    generator = AdaptivePatternGenerator()
    analysis = generator.analyze_scene_complexity(reference_images)
    patterns = generator.generate_adaptive_pattern_sequence(analysis)

Performance Monitoring
----------------------

Monitor optimization effectiveness:

.. code-block:: python

    # Get performance statistics
    stats = matcher.statistics
    print(f"Processing method: {stats['processing_method']}")
    print(f"Matches found: {stats['num_matches']}")
    print(f"Processing time: {stats.get('processing_time', 0):.2f}s")

Expected Results
----------------

Performance improvements on target platforms:

=============================  ============  =================
Platform                      Before        After Optimization
=============================  ============  =================
Raspberry Pi 4                2-5 minutes   < 5 seconds
Surface Pro ARM               1-3 minutes   < 3 seconds  
Snapdragon 8xx                3-8 minutes   < 5 seconds
Apple M1/M2                   30-60 seconds < 2 seconds
NVIDIA Jetson                 1-2 minutes   < 3 seconds
=============================  ============  =================

Troubleshooting
---------------

If optimizations are not working:

1. **Check Platform Detection**:

   .. code-block:: python
   
       import platform
       print(f"Platform: {platform.machine()}")
       print(f"SIMD available: {matcher.simd_matcher is not None}")

2. **Verify Optimization Flags**:

   .. code-block:: python
   
       print(f"Hierarchical: {matcher.enable_hierarchical}")
       print(f"Early termination: {matcher.enable_early_termination}")
       print(f"SIMD: {matcher.enable_simd}")

3. **Monitor Processing Method**:

   .. code-block:: python
   
       result = pipeline.process_gray_code_scan(left_images, right_images)
       method = result.correspondence_data.statistics.get('processing_method')
       print(f"Used method: {method}")

Common Issues
~~~~~~~~~~~~~

* **Low match count**: Increase ``confidence_threshold`` or disable ``enable_early_termination``
* **Still slow**: Check if ``enable_hierarchical`` is actually being used
* **Memory errors**: Reduce ``max_direct_pixels`` and ``target_match_count``

See Also
--------

* :doc:`gpu_acceleration` - GPU acceleration alternatives
* :doc:`realtime_scanning` - Real-time scanning configuration  
* :doc:`troubleshooting` - General troubleshooting guide