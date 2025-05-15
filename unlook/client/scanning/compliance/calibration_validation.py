"""
Calibration validation for ISO/ASTM 52902 compliance.

This module implements validation procedures using standardized test objects
to verify scanner calibration accuracy and detect calibration drift.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTestResult:
    """Results from calibration validation test."""
    test_date: datetime
    test_object: str
    passed: bool
    measurements: Dict[str, float]
    errors: Dict[str, float]
    drift_detected: bool
    recommendations: List[str]


@dataclass
class TestObjectSpec:
    """Specification for standardized test object."""
    name: str
    type: str  # 'sphere', 'step_gauge', 'plane', 'cylinder'
    nominal_dimensions: Dict[str, float]  # mm
    tolerance: float  # mm
    features: List[Dict[str, Any]]


class CalibrationValidator:
    """
    Validates scanner calibration using standardized test objects
    according to ISO/ASTM 52902 requirements.
    """
    
    # Standard test objects for ISO/ASTM 52902
    STANDARD_TEST_OBJECTS = {
        'sphere_25mm': TestObjectSpec(
            name='Reference Sphere 25mm',
            type='sphere',
            nominal_dimensions={'diameter': 25.0},
            tolerance=0.01,
            features=[{'type': 'sphere_fit', 'min_points': 1000}]
        ),
        'step_gauge': TestObjectSpec(
            name='Step Gauge',
            type='step_gauge',
            nominal_dimensions={
                'step_1': 10.0,
                'step_2': 20.0,
                'step_3': 30.0,
                'step_4': 40.0,
                'step_5': 50.0
            },
            tolerance=0.02,
            features=[{'type': 'plane_fit', 'min_points': 500}]
        ),
        'plane_artifact': TestObjectSpec(
            name='Flatness Artifact',
            type='plane',
            nominal_dimensions={'size': 100.0},
            tolerance=0.005,
            features=[{'type': 'plane_fit', 'min_points': 5000}]
        ),
        'cylinder_50mm': TestObjectSpec(
            name='Reference Cylinder 50mm',
            type='cylinder',
            nominal_dimensions={'diameter': 50.0, 'height': 100.0},
            tolerance=0.015,
            features=[{'type': 'cylinder_fit', 'min_points': 2000}]
        )
    }
    
    def __init__(self, scanner_specs: Dict[str, Any]):
        """
        Initialize calibration validator.
        
        Args:
            scanner_specs: Scanner specifications including resolution, accuracy claims
        """
        self.scanner_specs = scanner_specs
        self.baseline_measurements = {}
        self.validation_history = []
        
    def validate_with_test_object(self,
                                 point_cloud: np.ndarray,
                                 test_object_id: str,
                                 save_results: bool = True) -> CalibrationTestResult:
        """
        Validate calibration using a standardized test object.
        
        Args:
            point_cloud: Scanned point cloud of test object
            test_object_id: ID of the test object used
            save_results: Whether to save results to history
            
        Returns:
            CalibrationTestResult with validation details
        """
        logger.info(f"Validating calibration with test object: {test_object_id}")
        
        # Get test object specification
        if test_object_id not in self.STANDARD_TEST_OBJECTS:
            raise ValueError(f"Unknown test object: {test_object_id}")
        
        test_spec = self.STANDARD_TEST_OBJECTS[test_object_id]
        
        # Perform measurements based on object type
        if test_spec.type == 'sphere':
            measurements, errors = self._validate_sphere(point_cloud, test_spec)
        elif test_spec.type == 'step_gauge':
            measurements, errors = self._validate_step_gauge(point_cloud, test_spec)
        elif test_spec.type == 'plane':
            measurements, errors = self._validate_plane(point_cloud, test_spec)
        elif test_spec.type == 'cylinder':
            measurements, errors = self._validate_cylinder(point_cloud, test_spec)
        else:
            raise ValueError(f"Unknown test object type: {test_spec.type}")
        
        # Check if measurements are within tolerance
        passed = all(abs(error) <= test_spec.tolerance for error in errors.values())
        
        # Check for calibration drift
        drift_detected = self._check_calibration_drift(test_object_id, measurements)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(errors, drift_detected)
        
        # Create result
        result = CalibrationTestResult(
            test_date=datetime.now(),
            test_object=test_object_id,
            passed=passed,
            measurements=measurements,
            errors=errors,
            drift_detected=drift_detected,
            recommendations=recommendations
        )
        
        # Save results if requested
        if save_results:
            self.validation_history.append(result)
            self._update_baseline_if_needed(test_object_id, measurements, passed)
        
        return result
    
    def _validate_sphere(self,
                        point_cloud: np.ndarray,
                        test_spec: TestObjectSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate sphere measurements."""
        logger.debug("Validating sphere artifact")
        
        # Fit sphere to point cloud
        center, radius, inliers = self._fit_sphere(point_cloud)
        
        # Check minimum points requirement
        min_points = test_spec.features[0]['min_points']
        if len(inliers) < min_points:
            logger.warning(f"Insufficient points for sphere fit: {len(inliers)} < {min_points}")
        
        # Calculate measurements
        measurements = {
            'diameter': radius * 2,
            'center_x': center[0],
            'center_y': center[1],
            'center_z': center[2],
            'num_points': len(inliers),
            'fit_residual': np.std(np.linalg.norm(point_cloud[inliers] - center, axis=1) - radius)
        }
        
        # Calculate errors
        nominal_diameter = test_spec.nominal_dimensions['diameter']
        errors = {
            'diameter_error': measurements['diameter'] - nominal_diameter,
            'sphericity_error': measurements['fit_residual']
        }
        
        return measurements, errors
    
    def _validate_step_gauge(self,
                           point_cloud: np.ndarray,
                           test_spec: TestObjectSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate step gauge measurements."""
        logger.debug("Validating step gauge")
        
        # Segment point cloud into steps
        steps = self._segment_steps(point_cloud)
        
        measurements = {}
        errors = {}
        
        for i, (step_name, nominal_height) in enumerate(test_spec.nominal_dimensions.items()):
            if i < len(steps):
                # Fit plane to each step
                plane_normal, plane_d, inliers = self._fit_plane(steps[i])
                
                # Calculate step height
                height = abs(plane_d) / np.linalg.norm(plane_normal)
                
                measurements[step_name] = height
                errors[f"{step_name}_error"] = height - nominal_height
                measurements[f"{step_name}_flatness"] = np.std(steps[i][inliers][:, 2])
            else:
                measurements[step_name] = 0
                errors[f"{step_name}_error"] = -nominal_height
        
        return measurements, errors
    
    def _validate_plane(self,
                       point_cloud: np.ndarray,
                       test_spec: TestObjectSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate plane flatness."""
        logger.debug("Validating plane artifact")
        
        # Fit plane to point cloud
        plane_normal, plane_d, inliers = self._fit_plane(point_cloud)
        
        # Calculate flatness (peak-to-valley)
        distances = np.abs(np.dot(point_cloud[inliers], plane_normal) + plane_d)
        flatness = np.max(distances) - np.min(distances)
        
        measurements = {
            'flatness': flatness,
            'rms_flatness': np.sqrt(np.mean(distances**2)),
            'num_points': len(inliers)
        }
        
        errors = {
            'flatness_error': flatness
        }
        
        return measurements, errors
    
    def _validate_cylinder(self,
                         point_cloud: np.ndarray,
                         test_spec: TestObjectSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate cylinder measurements."""
        logger.debug("Validating cylinder artifact")
        
        # Fit cylinder to point cloud
        axis, center, radius, height, inliers = self._fit_cylinder(point_cloud)
        
        measurements = {
            'diameter': radius * 2,
            'height': height,
            'axis_x': axis[0],
            'axis_y': axis[1],
            'axis_z': axis[2],
            'num_points': len(inliers),
            'cylindricity': np.std(np.linalg.norm(
                point_cloud[inliers] - center - 
                np.outer(np.dot(point_cloud[inliers] - center, axis), axis),
                axis=1
            ) - radius)
        }
        
        # Calculate errors
        nominal_diameter = test_spec.nominal_dimensions['diameter']
        nominal_height = test_spec.nominal_dimensions['height']
        
        errors = {
            'diameter_error': measurements['diameter'] - nominal_diameter,
            'height_error': measurements['height'] - nominal_height,
            'cylindricity_error': measurements['cylindricity']
        }
        
        return measurements, errors
    
    def _fit_sphere(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Fit a sphere to point cloud using least squares."""
        # Initial estimate using centroid
        center = np.mean(point_cloud, axis=0)
        
        # Iterative refinement
        for _ in range(10):
            distances = np.linalg.norm(point_cloud - center, axis=1)
            radius = np.mean(distances)
            
            # Update center using weighted average
            weights = 1.0 / (distances + 1e-6)
            center = np.average(point_cloud, axis=0, weights=weights)
        
        # Final radius calculation
        distances = np.linalg.norm(point_cloud - center, axis=1)
        radius = np.mean(distances)
        
        # Identify inliers (within 3 sigma)
        residuals = np.abs(distances - radius)
        inliers = residuals < 3 * np.std(residuals)
        
        return center, radius, inliers
    
    def _fit_plane(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Fit a plane to point cloud using SVD."""
        # Center the points
        centroid = np.mean(point_cloud, axis=0)
        centered = point_cloud - centroid
        
        # SVD to find normal
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        
        # Ensure normal points up
        if normal[2] < 0:
            normal = -normal
        
        # Calculate d coefficient (ax + by + cz + d = 0)
        d = -np.dot(normal, centroid)
        
        # Identify inliers
        distances = np.abs(np.dot(point_cloud, normal) + d)
        threshold = 3 * np.std(distances)
        inliers = distances < threshold
        
        return normal, d, inliers
    
    def _fit_cylinder(self, 
                     point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        """Fit a cylinder to point cloud."""
        # This is a simplified implementation
        # In practice, use RANSAC or more robust methods
        
        # Estimate axis using PCA
        centroid = np.mean(point_cloud, axis=0)
        centered = point_cloud - centroid
        
        _, _, vh = np.linalg.svd(centered)
        axis = vh[0, :]  # Primary axis
        
        # Project points to plane perpendicular to axis
        projections = centered - np.outer(np.dot(centered, axis), axis)
        
        # Estimate radius
        radius = np.mean(np.linalg.norm(projections, axis=1))
        
        # Estimate height
        heights = np.dot(centered, axis)
        height = np.max(heights) - np.min(heights)
        
        # Find inliers
        distances = np.linalg.norm(projections, axis=1)
        inliers = np.abs(distances - radius) < 3 * np.std(distances - radius)
        
        return axis, centroid, radius, height, inliers
    
    def _segment_steps(self, point_cloud: np.ndarray) -> List[np.ndarray]:
        """Segment step gauge point cloud into individual steps."""
        # Sort points by height (z-coordinate)
        sorted_indices = np.argsort(point_cloud[:, 2])
        sorted_points = point_cloud[sorted_indices]
        
        # Find step boundaries using height differences
        z_diff = np.diff(sorted_points[:, 2])
        step_threshold = np.std(z_diff) * 3
        step_boundaries = np.where(z_diff > step_threshold)[0]
        
        # Create step segments
        steps = []
        start = 0
        for boundary in step_boundaries:
            steps.append(sorted_points[start:boundary+1])
            start = boundary + 1
        steps.append(sorted_points[start:])
        
        return steps
    
    def _check_calibration_drift(self,
                               test_object_id: str,
                               measurements: Dict[str, float]) -> bool:
        """Check if calibration has drifted from baseline."""
        if test_object_id not in self.baseline_measurements:
            return False
        
        baseline = self.baseline_measurements[test_object_id]
        
        # Check key measurements for drift
        drift_threshold = self.scanner_specs.get('drift_threshold', 0.05)  # 5% default
        
        for key in ['diameter', 'flatness', 'height']:
            if key in measurements and key in baseline:
                relative_change = abs(measurements[key] - baseline[key]) / baseline[key]
                if relative_change > drift_threshold:
                    logger.warning(f"Calibration drift detected in {key}: {relative_change:.2%}")
                    return True
        
        return False
    
    def _generate_recommendations(self,
                                errors: Dict[str, float],
                                drift_detected: bool) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check error magnitudes
        max_error = max(abs(e) for e in errors.values())
        
        if max_error > 0.1:  # > 0.1mm error
            recommendations.append("Recalibration strongly recommended - errors exceed 0.1mm")
        elif max_error > 0.05:  # > 0.05mm error
            recommendations.append("Consider recalibration - errors approaching tolerance limits")
        
        if drift_detected:
            recommendations.append("Calibration drift detected - schedule regular validation")
        
        # Check specific error patterns
        if 'diameter_error' in errors and abs(errors['diameter_error']) > 0.02:
            recommendations.append("Check camera intrinsic parameters")
        
        if 'flatness_error' in errors and errors['flatness_error'] > 0.01:
            recommendations.append("Verify projector-camera alignment")
        
        if not recommendations:
            recommendations.append("Calibration within acceptable limits")
        
        return recommendations
    
    def _update_baseline_if_needed(self,
                                 test_object_id: str,
                                 measurements: Dict[str, float],
                                 passed: bool):
        """Update baseline measurements if this is a good calibration."""
        if passed and (test_object_id not in self.baseline_measurements or 
                      self._is_better_than_baseline(test_object_id, measurements)):
            self.baseline_measurements[test_object_id] = measurements.copy()
            logger.info(f"Updated baseline measurements for {test_object_id}")
    
    def _is_better_than_baseline(self,
                               test_object_id: str,
                               measurements: Dict[str, float]) -> bool:
        """Check if new measurements are better than baseline."""
        baseline = self.baseline_measurements[test_object_id]
        
        # Compare key quality metrics
        quality_metrics = ['fit_residual', 'flatness', 'cylindricity']
        
        for metric in quality_metrics:
            if metric in measurements and metric in baseline:
                if measurements[metric] < baseline[metric]:
                    return True
        
        return False
    
    def save_validation_history(self, filename: str):
        """Save validation history to file."""
        history_data = []
        
        for result in self.validation_history:
            history_data.append({
                'test_date': result.test_date.isoformat(),
                'test_object': result.test_object,
                'passed': result.passed,
                'measurements': result.measurements,
                'errors': result.errors,
                'drift_detected': result.drift_detected,
                'recommendations': result.recommendations
            })
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Saved validation history to {filename}")
    
    def load_validation_history(self, filename: str):
        """Load validation history from file."""
        with open(filename, 'r') as f:
            history_data = json.load(f)
        
        self.validation_history = []
        
        for data in history_data:
            result = CalibrationTestResult(
                test_date=datetime.fromisoformat(data['test_date']),
                test_object=data['test_object'],
                passed=data['passed'],
                measurements=data['measurements'],
                errors=data['errors'],
                drift_detected=data['drift_detected'],
                recommendations=data['recommendations']
            )
            self.validation_history.append(result)
        
        logger.info(f"Loaded {len(self.validation_history)} validation results")