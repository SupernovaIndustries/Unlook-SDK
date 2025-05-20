"""
Uncertainty measurement for ISO/ASTM 52902 compliance.

This module implements uncertainty quantification for different pattern types
to meet certification requirements for 3D scanning systems.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyData:
    """Container for uncertainty measurement data."""
    mean_uncertainty: float
    max_uncertainty: float
    uncertainty_map: np.ndarray
    confidence_map: np.ndarray
    statistics: Dict[str, float]


class UncertaintyMeasurement(ABC):
    """
    Abstract base class for pattern-specific uncertainty measurement.
    
    Each pattern type requires different uncertainty quantification methods
    based on its correspondence detection approach.
    """
    
    def __init__(self, resolution: Tuple[int, int]):
        """
        Initialize uncertainty measurement.
        
        Args:
            resolution: (width, height) of the scanning system
        """
        self.resolution = resolution
        self.width, self.height = resolution
        
    @abstractmethod
    def compute_uncertainty(self, 
                          correspondences: List[Dict[str, Any]],
                          pattern_data: Dict[str, Any]) -> UncertaintyData:
        """
        Compute uncertainty for detected correspondences.
        
        Args:
            correspondences: List of correspondence data
            pattern_data: Pattern-specific data
            
        Returns:
            UncertaintyData object with uncertainty metrics
        """
        pass
    
    def _create_uncertainty_map(self, 
                               points: np.ndarray,
                               uncertainties: np.ndarray) -> np.ndarray:
        """
        Create a 2D uncertainty map from point measurements.
        
        Args:
            points: Nx2 array of point coordinates
            uncertainties: N array of uncertainty values
            
        Returns:
            2D uncertainty map
        """
        uncertainty_map = np.full((self.height, self.width), np.inf)
        
        # Use nearest neighbor interpolation for initial map
        for i, (x, y) in enumerate(points):
            if 0 <= x < self.width and 0 <= y < self.height:
                uncertainty_map[int(y), int(x)] = uncertainties[i]
        
        # Fill gaps using distance-weighted interpolation
        mask = uncertainty_map == np.inf
        if np.any(mask):
            try:
                from scipy.interpolate import griddata
                y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
                valid_mask = ~mask
                
                if np.any(valid_mask):
                    valid_points = np.column_stack([x_coords[valid_mask], y_coords[valid_mask]])
                    valid_values = uncertainty_map[valid_mask]
                    
                    uncertainty_map[mask] = griddata(
                        valid_points, valid_values,
                        (x_coords[mask], y_coords[mask]),
                        method='nearest'
                    )
            except ImportError:
                # Fallback to simple filling if scipy is not available
                logger.warning("scipy not available for interpolation, using simple filling")
                uncertainty_map[mask] = np.mean(uncertainties)
        
        return uncertainty_map


class MazeUncertaintyMeasurement(UncertaintyMeasurement):
    """
    Uncertainty measurement for maze patterns.
    
    Quantifies uncertainty based on junction detection confidence
    and neighborhood matching quality.
    """
    
    def compute_uncertainty(self,
                          correspondences: List[Dict[str, Any]],
                          pattern_data: Dict[str, Any]) -> UncertaintyData:
        """
        Compute uncertainty for maze pattern correspondences.
        
        Uses junction detection confidence and local topology matching
        to quantify measurement uncertainty.
        """
        logger.info("Computing uncertainty for maze pattern")
        
        points = []
        uncertainties = []
        confidences = []
        
        for corr in correspondences:
            point = corr.get('point', [0, 0])
            
            # Junction detection confidence
            junction_confidence = corr.get('junction_confidence', 0.0)
            
            # Topology matching score
            topology_score = corr.get('topology_score', 0.0)
            
            # Local contrast quality
            contrast_quality = corr.get('contrast_quality', 0.0)
            
            # If we have valid confidence values (>0), use them
            if junction_confidence > 0 or topology_score > 0 or contrast_quality > 0:
                # Compute uncertainty as inverse of confidence
                weights = [0.4, 0.4, 0.2]  # Weights for each metric
                values = [junction_confidence, topology_score, contrast_quality]
                # Only use metrics that have values
                valid_values = [v for v in values if v > 0]
                if valid_values:
                    confidence = sum(v for v in valid_values) / len(valid_values)
                else:
                    confidence = 0.5  # Default value
            else:
                # If no confidence metrics are available, use reprojection error
                reprojection_error = corr.get('reprojection_error', 1.0)
                confidence = 1.0 / (1.0 + reprojection_error)
            
            # Ensure confidence is normalized
            confidence = max(0.0, min(1.0, confidence))
            
            # Scale uncertainty to mm based on system resolution and confidence
            uncertainty = 1.0 - confidence
            
            # Use pattern_data for scaling if available, otherwise use default
            pixel_to_mm = pattern_data.get('pixel_to_mm', 0.1)
            # Scale uncertainty based on confidence (higher confidence = lower uncertainty)
            uncertainty_mm = uncertainty * pixel_to_mm
            
            points.append(point)
            uncertainties.append(uncertainty_mm)
            confidences.append(confidence)
        
        # If we don't have any valid points, return default values
        if not points:
            logger.warning("No valid correspondence points for uncertainty calculation")
            # Create default data with uniform uncertainty
            uncertainty_map = np.ones((self.height, self.width)) * 0.1  # 0.1mm default uncertainty
            confidence_map = np.ones((self.height, self.width)) * 0.5  # 0.5 default confidence
            
            return UncertaintyData(
                mean_uncertainty=0.1,
                max_uncertainty=0.1,
                uncertainty_map=uncertainty_map,
                confidence_map=confidence_map,
                statistics={
                    'mean_confidence': 0.5,
                    'std_confidence': 0.0,
                    'min_confidence': 0.5,
                    'max_confidence': 0.5,
                    'num_correspondences': 0,
                    'coverage_percent': 0.0
                }
            )
        
        points = np.array(points)
        uncertainties = np.array(uncertainties)
        confidences = np.array(confidences)
        
        # Create uncertainty and confidence maps
        uncertainty_map = self._create_uncertainty_map(points, uncertainties)
        confidence_map = self._create_uncertainty_map(points, confidences)
        
        # Compute statistics
        statistics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences) if len(confidences) > 0 else 0.0,
            'max_confidence': np.max(confidences) if len(confidences) > 0 else 0.0,
            'num_correspondences': len(correspondences),
            'coverage_percent': (len(correspondences) / (self.width * self.height)) * 100
        }
        
        return UncertaintyData(
            mean_uncertainty=np.mean(uncertainties),
            max_uncertainty=np.max(uncertainties) if len(uncertainties) > 0 else 0.0,
            uncertainty_map=uncertainty_map,
            confidence_map=confidence_map,
            statistics=statistics
        )


class VoronoiUncertaintyMeasurement(UncertaintyMeasurement):
    """
    Uncertainty measurement for Voronoi patterns.
    
    Measures reconstruction uncertainty using cell boundary
    detection quality and region matching confidence.
    """
    
    def compute_uncertainty(self,
                          correspondences: List[Dict[str, Any]],
                          pattern_data: Dict[str, Any]) -> UncertaintyData:
        """
        Compute uncertainty for Voronoi pattern correspondences.
        
        Uses cell boundary sharpness and region descriptor matching
        to quantify measurement uncertainty.
        """
        logger.info("Computing uncertainty for Voronoi pattern")
        
        points = []
        uncertainties = []
        confidences = []
        
        for corr in correspondences:
            point = corr.get('point', [0, 0])
            
            # Check for specific Voronoi metrics
            has_voronoi_metrics = False
            
            # Boundary detection sharpness
            boundary_sharpness = corr.get('boundary_sharpness', 0.0)
            
            # Region descriptor matching score
            descriptor_score = corr.get('descriptor_score', 0.0)
            
            # Cell size consistency
            size_consistency = corr.get('size_consistency', 0.0)
            
            # Edge gradient quality
            edge_quality = corr.get('edge_quality', 0.0)
            
            if (boundary_sharpness > 0 or descriptor_score > 0 or 
                size_consistency > 0 or edge_quality > 0):
                has_voronoi_metrics = True
                
                # Compute weighted confidence
                weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each metric
                scores = [boundary_sharpness, descriptor_score, size_consistency, edge_quality]
                # Only use metrics that have values
                valid_scores = [s for s in scores if s > 0]
                valid_weights = weights[:len(valid_scores)]
                
                # Normalize weights
                if sum(valid_weights) > 0:
                    valid_weights = [w/sum(valid_weights) for w in valid_weights]
                    confidence = sum(w * s for w, s in zip(valid_weights, valid_scores))
                else:
                    confidence = 0.5  # Default value
            else:
                # Fallback to using general correspondence metrics
                confidence = corr.get('confidence', 0.0)
                reprojection_error = corr.get('reprojection_error', 0.0)
                
                if confidence > 0:
                    # Use provided confidence
                    pass
                elif reprojection_error > 0:
                    # Convert reprojection error to confidence
                    confidence = 1.0 / (1.0 + reprojection_error)
                else:
                    confidence = 0.5  # Default value
            
            # Ensure confidence is normalized
            confidence = max(0.0, min(1.0, confidence))
            
            # Convert to uncertainty in mm
            uncertainty = 1.0 - confidence
            
            # Use pattern_data for scaling if available, otherwise use default
            pixel_to_mm = pattern_data.get('pixel_to_mm', 0.1)
            # Voronoi patterns typically have lower uncertainty due to better regional matching
            uncertainty_factor = 0.8 if has_voronoi_metrics else 1.0
            uncertainty_mm = uncertainty * pixel_to_mm * uncertainty_factor
            
            points.append(point)
            uncertainties.append(uncertainty_mm)
            confidences.append(confidence)
        
        # If we don't have any valid points, return default values
        if not points:
            logger.warning("No valid correspondence points for uncertainty calculation")
            # Create default data with uniform uncertainty
            uncertainty_map = np.ones((self.height, self.width)) * 0.08  # 0.08mm default (lower for Voronoi)
            confidence_map = np.ones((self.height, self.width)) * 0.6  # 0.6 default confidence (higher for Voronoi)
            
            return UncertaintyData(
                mean_uncertainty=0.08,
                max_uncertainty=0.08,
                uncertainty_map=uncertainty_map,
                confidence_map=confidence_map,
                statistics={
                    'mean_confidence': 0.6,
                    'std_confidence': 0.0,
                    'min_confidence': 0.6,
                    'max_confidence': 0.6,
                    'num_correspondences': 0,
                    'coverage_percent': 0.0
                }
            )
        
        points = np.array(points)
        uncertainties = np.array(uncertainties)
        confidences = np.array(confidences)
        
        # Create maps
        uncertainty_map = self._create_uncertainty_map(points, uncertainties)
        confidence_map = self._create_uncertainty_map(points, confidences)
        
        # Compute statistics
        voronoi_specific_stats = {}
        if any(corr.get('boundary_sharpness', 0) > 0 for corr in correspondences):
            voronoi_specific_stats['mean_boundary_sharpness'] = np.mean([
                corr.get('boundary_sharpness', 0) for corr in correspondences 
                if corr.get('boundary_sharpness', 0) > 0
            ])
        
        if any(corr.get('descriptor_score', 0) > 0 for corr in correspondences):
            voronoi_specific_stats['mean_descriptor_score'] = np.mean([
                corr.get('descriptor_score', 0) for corr in correspondences
                if corr.get('descriptor_score', 0) > 0
            ])
        
        statistics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences) if len(confidences) > 0 else 0.0,
            'max_confidence': np.max(confidences) if len(confidences) > 0 else 0.0,
            'num_correspondences': len(correspondences),
            'coverage_percent': (len(correspondences) / (self.width * self.height)) * 100,
            **voronoi_specific_stats
        }
        
        return UncertaintyData(
            mean_uncertainty=np.mean(uncertainties),
            max_uncertainty=np.max(uncertainties) if len(uncertainties) > 0 else 0.0,
            uncertainty_map=uncertainty_map,
            confidence_map=confidence_map,
            statistics=statistics
        )


class HybridArUcoUncertaintyMeasurement(UncertaintyMeasurement):
    """
    Uncertainty measurement for hybrid ArUco patterns.
    
    Leverages ArUco marker detection confidence along with
    structured light pattern quality for uncertainty quantification.
    """
    
    def compute_uncertainty(self,
                          correspondences: List[Dict[str, Any]],
                          pattern_data: Dict[str, Any]) -> UncertaintyData:
        """
        Compute uncertainty for hybrid ArUco pattern correspondences.
        
        Combines ArUco marker reprojection error with local pattern
        matching confidence for comprehensive uncertainty assessment.
        """
        logger.info("Computing uncertainty for hybrid ArUco pattern")
        
        points = []
        uncertainties = []
        confidences = []
        
        # Get ArUco marker data if available
        marker_data = pattern_data.get('marker_data', {})
        has_marker_data = len(marker_data) > 0
        
        # Track marker-based correspondences
        marker_correspondences = []
        
        for corr in correspondences:
            point = corr.get('point', [0, 0])
            
            # Check for ArUco-specific metrics
            has_aruco_metrics = False
            
            # ArUco marker detection confidence
            marker_confidence = corr.get('marker_confidence', 0.0)
            
            # Reprojection error for nearby markers
            reprojection_error = corr.get('reprojection_error', 0.0)
            
            # Structured light pattern quality
            pattern_quality = corr.get('pattern_quality', 0.0)
            
            # Distance to nearest marker (normalized)
            marker_distance = corr.get('marker_distance_normalized', 0.0)
            
            if marker_confidence > 0 or marker_distance > 0:
                has_aruco_metrics = True
                
                # Compute confidence with higher weight for marker-based measurements
                if marker_confidence > 0:
                    # Near a marker - high confidence
                    confidence = 0.6 * marker_confidence
                    if reprojection_error > 0:
                        confidence += 0.3 * (1.0 / (1.0 + reprojection_error))
                    if pattern_quality > 0:
                        confidence += 0.1 * pattern_quality
                    marker_correspondences.append(corr)
                else:
                    # Away from markers - rely on pattern quality
                    if pattern_quality > 0:
                        confidence = 0.8 * pattern_quality
                    else:
                        confidence = 0.5  # Default
                    if marker_distance > 0:
                        confidence += 0.2 * (1.0 - marker_distance)
            else:
                # Fallback to using general correspondence metrics
                confidence = corr.get('confidence', 0.0)
                
                if confidence <= 0:
                    # Use reprojection error if available
                    if reprojection_error > 0:
                        confidence = 1.0 / (1.0 + reprojection_error)
                    else:
                        confidence = 0.5  # Default value
            
            # Ensure confidence is normalized
            confidence = max(0.0, min(1.0, confidence))
            
            # Convert to uncertainty in mm
            uncertainty = 1.0 - confidence
            
            # Use pattern_data for scaling if available, otherwise use default
            pixel_to_mm = pattern_data.get('pixel_to_mm', 0.1)
            
            # ArUco provides better accuracy when markers are detected
            uncertainty_factor = 0.6 if has_aruco_metrics else 0.9
            uncertainty_mm = uncertainty * pixel_to_mm * uncertainty_factor
            
            points.append(point)
            uncertainties.append(uncertainty_mm)
            confidences.append(confidence)
        
        # If we don't have any valid points, return default values
        if not points:
            logger.warning("No valid correspondence points for uncertainty calculation")
            # Create default data with uniform uncertainty
            uncertainty_map = np.ones((self.height, self.width)) * 0.06  # 0.06mm default (lower for ArUco)
            confidence_map = np.ones((self.height, self.width)) * 0.7  # 0.7 default confidence (higher for ArUco)
            
            return UncertaintyData(
                mean_uncertainty=0.06,
                max_uncertainty=0.06,
                uncertainty_map=uncertainty_map,
                confidence_map=confidence_map,
                statistics={
                    'mean_confidence': 0.7,
                    'std_confidence': 0.0,
                    'min_confidence': 0.7,
                    'max_confidence': 0.7,
                    'num_correspondences': 0,
                    'coverage_percent': 0.0,
                    'num_marker_correspondences': 0,
                    'markers_detected': len(marker_data)
                }
            )
        
        points = np.array(points)
        uncertainties = np.array(uncertainties)
        confidences = np.array(confidences)
        
        # Create maps
        uncertainty_map = self._create_uncertainty_map(points, uncertainties)
        confidence_map = self._create_uncertainty_map(points, confidences)
        
        # Compute statistics for marker-based correspondences
        marker_stats = {}
        if marker_correspondences:
            marker_reprojection_errors = [
                c.get('reprojection_error', 0) for c in marker_correspondences
                if c.get('reprojection_error', 0) > 0
            ]
            if marker_reprojection_errors:
                marker_stats['mean_reprojection_error'] = np.mean(marker_reprojection_errors)
        
        # Compute statistics
        statistics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences) if len(confidences) > 0 else 0.0,
            'max_confidence': np.max(confidences) if len(confidences) > 0 else 0.0,
            'num_correspondences': len(correspondences),
            'num_marker_correspondences': len(marker_correspondences),
            'coverage_percent': (len(correspondences) / (self.width * self.height)) * 100,
            'markers_detected': len(marker_data),
            **marker_stats
        }
        
        return UncertaintyData(
            mean_uncertainty=np.mean(uncertainties),
            max_uncertainty=np.max(uncertainties) if len(uncertainties) > 0 else 0.0,
            uncertainty_map=uncertainty_map,
            confidence_map=confidence_map,
            statistics=statistics
        )