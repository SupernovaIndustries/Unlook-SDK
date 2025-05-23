Triangulator Module
==================

The triangulator module provides unified 3D reconstruction from stereo images with ISO/ASTM 52902 compliant uncertainty quantification.

.. module:: unlook.client.scanning.reconstruction.triangulator

Overview
--------

The :class:`Triangulator` class consolidates all triangulation implementations into a single, robust solution that provides:

- ISO/ASTM 52902 compliant uncertainty quantification
- GPU-ready architecture (CPU implementation currently)
- Robust error handling and outlier rejection
- Consistent API across all use cases

Classes
-------

.. autoclass:: Triangulator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: triangulate
   .. automethod:: triangulate_batch

.. autoclass:: TriangulationResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PointUncertainty
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic triangulation::

    from unlook.client.scanning.reconstruction import Triangulator
    
    # Initialize with calibration data
    triangulator = Triangulator(calibration_params)
    
    # Triangulate corresponding points
    result = triangulator.triangulate(left_points, right_points)
    
    # Access results
    points_3d = result.points_3d
    uncertainties = result.uncertainties  # ISO/ASTM 52902 compliant

With uncertainty analysis::

    # Get detailed uncertainty information
    for i, uncertainty in enumerate(result.uncertainties):
        print(f"Point {i}:")
        print(f"  Position uncertainty: {uncertainty.position_uncertainty:.2f} mm")
        print(f"  Reprojection error: {uncertainty.reprojection_error:.2f} px")
        print(f"  Triangulation angle: {uncertainty.triangulation_angle:.1f}°")
        print(f"  Confidence: {uncertainty.confidence:.2%}")

Batch processing::

    # Process multiple point sets efficiently
    results = triangulator.triangulate_batch(
        left_points_list,
        right_points_list
    )

ISO/ASTM 52902 Compliance
------------------------

The triangulator provides measurement uncertainty quantification required for ISO/ASTM 52902 certification:

1. **Position Uncertainty**: Estimated 3D position uncertainty in mm
2. **Reprojection Error**: Pixel error when projecting back to image space
3. **Triangulation Angle**: Angle between camera rays (affects accuracy)
4. **Baseline Ratio**: Ratio of baseline to depth (quality metric)
5. **Confidence Score**: Overall confidence in the measurement

Migration from Legacy Code
-------------------------

If you're using the old triangulation implementations::

    # Old code
    from unlook.client.scanning.reconstruction.direct_triangulator import triangulate_with_baseline_correction
    points_3d = triangulate_with_baseline_correction(left_pts, right_pts, params)
    
    # New code
    from unlook.client.scanning.reconstruction import Triangulator
    triangulator = Triangulator(params)
    result = triangulator.triangulate(left_pts, right_pts)
    points_3d = result.points_3d

Performance Considerations
-------------------------

- CPU implementation uses optimized NumPy operations
- GPU acceleration framework in place for future implementation
- Batch processing available for multiple point sets
- Automatic outlier filtering based on statistical analysis

See Also
--------

- :doc:`structured_light` - Pattern generation and decoding
- :doc:`point_cloud_processing` - Point cloud filtering and mesh generation
- :doc:`../user_guide/iso_compliance` - ISO/ASTM 52902 compliance guide