# Optimal Camera Spacing for 3D Scanning

This document provides guidance on optimal camera spacing configurations for both single-camera and stereo camera structured light 3D scanning setups.

## Stereo Camera Setup

### Basic Principles

In a stereo camera setup, two cameras are positioned at a known distance from each other, creating a baseline. The stereo system captures the same scene from two slightly different viewpoints, allowing depth to be calculated through triangulation.

### Optimal Baseline Distance

The currently used spacing of 8cm (80mm) is within the typical range for a stereo system, but may need adjustment based on your specific scanning requirements. 

#### Recommended Baselines:

- **For small objects (5-20cm)**: 50-100mm baseline
- **For medium objects (20-50cm)**: 100-150mm baseline
- **For large objects (50-100cm+)**: 150-300mm baseline

#### Baseline Tradeoffs:

- **Larger baseline**:
  - Increases depth accuracy at longer distances
  - Improves overall precision of triangulation
  - Reduces the effective near-field scanning range
  - Increases difficulty in finding correspondences (more occlusions)

- **Smaller baseline**:
  - Improves near-field scanning capabilities
  - Reduces occlusions and matching problems
  - Reduces depth precision, especially at longer distances
  - Makes it easier to find stereo correspondences

### Field of View Considerations

The camera lenses and their field of view also play a crucial role in determining the optimal spacing:

- **Narrower FOV lenses** (higher focal length) generally benefit from wider baselines
- **Wider FOV lenses** can work effectively with shorter baselines
- Ensure there is sufficient overlap in the viewable area of both cameras

### Formula for Baseline Recommendation

As a general rule, you can use this formula to determine a reasonable baseline starting point:

```
Optimal Baseline (mm) = Working Distance (mm) / 10
```

For example, if your typical scanning distance is around 500mm (50cm), a good starting baseline would be 50mm.

For precision-critical applications, you might want to increase this to:

```
Optimal Baseline (mm) = Working Distance (mm) / 5
```

Which would suggest a 100mm baseline for a 500mm working distance.

## Single Camera + Projector Setup

### Principles

In a single camera with projector setup, the camera and projector form a stereo pair. The projector is treated as an "inverse camera" that projects patterns instead of capturing them. The spacing between the camera and projector is crucial for accurate triangulation.

### Optimal Camera-Projector Baseline

#### Recommended Camera-Projector Baselines:

- **For small objects (5-20cm)**: 100-200mm baseline
- **For medium objects (20-50cm)**: 200-300mm baseline
- **For large objects (50-100cm+)**: 300-500mm baseline

Note that single camera setups typically benefit from larger baselines than stereo camera setups.

#### Baseline Placement:

Unlike stereo camera setups where the cameras are typically placed side by side horizontally, the camera-projector placement can be more flexible:

- **Horizontal offset**: Camera to the left or right of the projector (conventional approach)
- **Vertical offset**: Camera above or below the projector (useful for some scanning geometries)
- **Angled offset**: Camera at an angle to the projector (specialized cases)

The horizontal arrangement is most common and usually provides the best results for general-purpose scanning.

### Camera-Projector Angle

In addition to baseline distance, the angle between the camera and projector optical axes is important:

- **Recommended angle**: 15-30 degrees for most applications
- **Small objects**: Use angles closer to 15 degrees
- **Large objects**: Angles up to 30 degrees can be beneficial

## Adjustment Recommendations for the UnLook SDK

Based on the above principles, we recommend the following adjustments to your current setup:

### For Your Current 8cm (80mm) Stereo Camera Spacing:

This is appropriate for:
- Small to medium sized objects (5-40cm)
- Working distances of approximately 40-80cm
- General purpose scanning applications

### Recommended Adjustments:

1. **For scanning very small objects (5-10cm)**:
   - Consider reducing the baseline to 5-6cm
   - Use the "robust" scanning mode for better point cloud density

2. **For scanning larger objects (50cm+)**:
   - Increase camera spacing to 10-15cm
   - Place the scanning system farther from the object (~100cm)

3. **For single camera scanning**:
   - Set the camera-projector baseline to approximately 15-20cm
   - Position at an angle of approximately 20 degrees

## Testing Your Configuration

To find the optimal camera spacing for your specific application:

1. Create a calibration object with known dimensions
2. Scan the object at different camera spacings
3. Measure the accuracy and point density of the resulting scans
4. Choose the spacing that provides the best balance of coverage and accuracy

## Implementation Notes

When adjusting camera spacing, remember to:

1. Recalibrate the system after any physical changes
2. Ensure both cameras can view the entire object of interest
3. Check that the projector's field of view covers the entire scanning area
4. Consider the effect of the new spacing on shadow regions and occlusions

## Recommended Scanning Distance Formula

For general structured light scanning, we recommend:

```
Optimal Object Distance = Baseline × 5 to Baseline × 10
```

For your 8cm baseline setup, this means an optimal scanning distance of approximately 40-80cm from the cameras to the object.

## References

These recommendations are based on established principles in structured light scanning, research publications, and lessons learned from implementing SLStudio and similar systems.