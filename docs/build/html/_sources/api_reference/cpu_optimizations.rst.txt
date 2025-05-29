CPU Optimizations API
====================

This module provides CPU optimizations for ARM processors and mobile devices.

Improved Correspondence Matcher
-------------------------------

.. automodule:: unlook.client.scanning.reconstruction.improved_correspondence_matcher
   :members:
   :undoc-members:
   :show-inheritance:

SIMD Optimizations
------------------

.. automodule:: unlook.client.scanning.reconstruction.simd_optimizations
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Pattern Strategies
---------------------------

.. automodule:: unlook.client.patterns.adaptive_pattern_strategies
   :members:
   :undoc-members:
   :show-inheritance:

Classes and Enums
-----------------

OptimizedSpatialIndex
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.reconstruction.improved_correspondence_matcher.OptimizedSpatialIndex
   :members:
   :undoc-members:

MatchingResult
~~~~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.reconstruction.improved_correspondence_matcher.MatchingResult
   :members:
   :undoc-members:

SIMDOptimizedMatcher
~~~~~~~~~~~~~~~~~~~

.. autoclass:: unlook.client.scanning.reconstruction.simd_optimizations.SIMDOptimizedMatcher
   :members:
   :undoc-members:

AdaptivePatternGenerator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: unlook.client.patterns.adaptive_pattern_strategies.AdaptivePatternGenerator
   :members:
   :undoc-members:

PatternType
~~~~~~~~~~~

.. autoclass:: unlook.client.patterns.adaptive_pattern_strategies.PatternType
   :members:
   :undoc-members:

SceneComplexity  
~~~~~~~~~~~~~~~

.. autoclass:: unlook.client.patterns.adaptive_pattern_strategies.SceneComplexity
   :members:
   :undoc-members:

Utility Functions
-----------------

.. autofunction:: unlook.client.scanning.reconstruction.simd_optimizations.vectorized_distance_calculation

.. autofunction:: unlook.client.scanning.reconstruction.simd_optimizations.vectorized_hamming_distance

.. autofunction:: unlook.client.scanning.reconstruction.simd_optimizations.vectorized_correspondence_validation

.. autofunction:: unlook.client.scanning.reconstruction.simd_optimizations.batch_process_correspondences