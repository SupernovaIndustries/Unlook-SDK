# Structured Light CPU Optimization Plan

## Executive Summary
This document outlines the optimization strategy for UnLook's structured light 3D scanning to achieve real-time performance on CPU-only systems, including ARM processors (Surface, smartphones). The focus is on algorithmic optimizations that reduce computational complexity from O(n²) to O(n log n) or better.

## Current Problem
- Processing 598,662 x 505,049 pixels = ~300 billion possible correspondences
- Current naive O(n²) approach causes system to hang
- Need to support ARM processors and mobile devices
- Cannot rely on GPU acceleration

## Optimization Strategy

### 1. Server-Side Preprocessing Optimizations

#### 1.1 ROI (Region of Interest) Extraction
**Implementation Priority: HIGH**
- Automatically detect object boundaries using reference white/black patterns
- Reduce processing area by 70-80%
- Implementation steps:
  1. Capture reference white pattern
  2. Apply adaptive threshold to identify bright regions
  3. Morphological operations to clean mask
  4. Bounding box calculation with padding
  5. Pass ROI metadata with each frame

#### 1.2 Edge-Preserving Filtering
**Implementation Priority: MEDIUM**
- Apply bilateral filtering to reduce noise while preserving edges
- Improves pattern detection accuracy
- Can be done on Raspberry Pi using OpenCV optimized for ARM

#### 1.3 Adaptive Quality Levels
- Implement dynamic downsampling based on scene complexity
- Simple scenes: 2x downsampling
- Complex scenes: Full resolution
- Decision based on edge density analysis

### 2. Client-Side Reconstruction Optimizations

#### 2.1 Hierarchical Coarse-to-Fine Matching
**Implementation Priority: HIGH**
- Create image pyramid with 3-4 levels
- Level 0: 1/8 resolution (fast coarse matching)
- Level 1: 1/4 resolution (refinement)
- Level 2: 1/2 resolution (fine tuning)
- Level 3: Full resolution (only in ROI)
- Expected speedup: 5-6x

#### 2.2 Efficient Spatial Indexing
**Implementation Priority: MEDIUM**
- Replace dictionary-based index with spatial hash table
- Use fixed-size buckets for O(1) lookup
- Implement local search windows (5x5 or 7x7)
- Memory-efficient structure for mobile devices

#### 2.3 SIMD Optimizations
**Implementation Priority: LOW**
- Utilize ARM NEON instructions for vector operations
- Batch process 4-8 pixels simultaneously
- Optimize inner loops for cache efficiency

### 3. Pattern Strategy Optimizations

#### 3.1 Adaptive Pattern Density
- Use fewer patterns in areas with good texture
- Increase pattern density only where needed
- Dynamic pattern selection based on scene analysis

#### 3.2 Hybrid Gray Code + Phase Shift
- Gray code for robust initial correspondence
- Phase shift only for high-precision areas
- Reduces total capture time

### 4. Algorithmic Improvements

#### 4.1 Local Optimization Strategy
- Process image in tiles (e.g., 64x64)
- Parallel processing of independent tiles
- Merge results with overlap handling

#### 4.2 Early Termination
- Stop searching once confidence threshold reached
- Skip areas with low pattern visibility
- Adaptive search radius based on local texture

#### 4.3 Correspondence Validation
- Hamming distance for error correction
- Local consistency checks
- Majority voting for ambiguous matches

## Implementation Phases

### Phase 1: Critical Optimizations (Week 1)
1. ROI extraction in preprocessing
2. Basic hierarchical matching (2 levels)
3. Simple spatial indexing

### Phase 2: Performance Tuning (Week 2)
1. Full pyramid implementation (4 levels)
2. Optimized spatial hash table
3. Edge-preserving filtering

### Phase 3: Fine Tuning (Week 3)
1. SIMD optimizations
2. Adaptive pattern strategies
3. Performance profiling and bottleneck elimination

## Expected Performance Gains

### Before Optimization
- 598k x 505k pixels = ~300B comparisons
- Processing time: Several minutes
- Memory usage: >2GB

### After Optimization
- ROI reduction: 70% fewer pixels
- Hierarchical processing: 5-6x speedup
- Spatial indexing: 10x faster lookups
- **Total expected speedup: 20-30x**
- **Target processing time: <5 seconds on ARM CPU**

## Testing Strategy

### Test Platforms
1. Desktop CPU (baseline)
2. Raspberry Pi 4 (ARM Cortex-A72)
3. Surface Pro (ARM)
4. High-end smartphone (Snapdragon 8xx)

### Performance Metrics
- Total processing time
- Memory usage
- Accuracy (compared to current implementation)
- Power consumption on mobile devices

## Code Architecture

```python
class OptimizedStructuredLightPipeline:
    def __init__(self):
        self.roi_extractor = ROIExtractor()
        self.pyramid_matcher = HierarchicalMatcher()
        self.spatial_index = SpatialHashIndex()
        
    def process(self, images):
        # 1. Extract ROI
        roi_mask, bounds = self.roi_extractor.extract(images)
        
        # 2. Build pyramids
        pyramids = self.build_pyramids(images, roi_mask)
        
        # 3. Coarse-to-fine matching
        matches = self.pyramid_matcher.match(pyramids)
        
        # 4. Reconstruct 3D
        points = self.triangulate(matches)
        
        return points
```

## Success Criteria
- [ ] Process standard scan in <5 seconds on Raspberry Pi 4
- [ ] Memory usage <512MB
- [ ] Maintain accuracy within 0.1mm of current implementation
- [ ] Support real-time preview at 10+ FPS
- [ ] Run on ARM-based Surface and smartphones

## Next Steps
1. Implement server-side ROI extraction
2. Create hierarchical matcher class
3. Benchmark on target platforms
4. Iterate based on performance results