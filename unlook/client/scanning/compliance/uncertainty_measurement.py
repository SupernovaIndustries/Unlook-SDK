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
            junction_confidence = corr.get('junction_confidence', 0.5)
            
            # Topology matching score
            topology_score = corr.get('topology_score', 0.5)
            
            # Local contrast quality
            contrast_quality = corr.get('contrast_quality', 0.5)
            
            # Compute uncertainty as inverse of confidence
            confidence = (junction_confidence + topology_score + contrast_quality) / 3
            uncertainty = 1.0 - confidence
            
            # Scale uncertainty to mm based on system resolution
            pixel_to_mm = pattern_data.get('pixel_to_mm', 0.1)
            uncertainty_mm = uncertainty * pixel_to_mm
            
            points.append(point)
            uncertainties.append(uncertainty_mm)
            confidences.append(confidence)
        
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
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'num_correspondences': len(correspondences),
            'coverage_percent': (len(correspondences) / (self.width * self.height)) * 100
        }
        
        return UncertaintyData(
            mean_uncertainty=np.mean(uncertainties),
            max_uncertainty=np.max(uncertainties),
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
            
            # Boundary detection sharpness
            boundary_sharpness = corr.get('boundary_sharpness', 0.5)
            
            # Region descriptor matching score
            descriptor_score = corr.get('descriptor_score', 0.5)
            
            # Cell size consistency
            size_consistency = corr.get('size_consistency', 0.5)
            
            # Edge gradient quality
            edge_quality = corr.get('edge_quality', 0.5)
            
            # Compute weighted confidence
            weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each metric
            scores = [boundary_sharpness, descriptor_score, size_consistency, edge_quality]
            confidence = sum(w * s for w, s in zip(weights, scores))
            
            # Convert to uncertainty in mm
            uncertainty = 1.0 - confidence
            pixel_to_mm = pattern_data.get('pixel_to_mm', 0.1)
            uncertainty_mm = uncertainty * pixel_to_mm * 0.8  # Voronoi typically has lower uncertainty
            
            points.append(point)
            uncertainties.append(uncertainty_mm)
            confidences.append(confidence)
        
        points = np.array(points)
        uncertainties = np.array(uncertainties)
        confidences = np.array(confidences)
        
        # Create maps
        uncertainty_map = self._create_uncertainty_map(points, uncertainties)
        confidence_map = self._create_uncertainty_map(points, confidences)
        
        # Compute statistics
        statistics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'num_correspondences': len(correspondences),
            'coverage_percent': (len(correspondences) / (self.width * self.height)) * 100,
            'mean_boundary_sharpness': np.mean([c.get('boundary_sharpness', 0.5) for c in correspondences]),
            'mean_descriptor_score': np.mean([c.get('descriptor_score', 0.5) for c in correspondences])
        }
        
        return UncertaintyData(
            mean_uncertainty=np.mean(uncertainties),
            max_uncertainty=np.max(uncertainties),
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
        
        # Get ArUco marker data
        marker_data = pattern_data.get('marker_data', {})
        
        for corr in correspondences:
            point = corr.get('point', [0, 0])
            
            # ArUco marker detection confidence
            marker_confidence = corr.get('marker_confidence', 0.0)
            
            # Reprojection error for nearby markers
            reprojection_error = corr.get('reprojection_error', 1.0)
            
            # Structured light pattern quality
            pattern_quality = corr.get('pattern_quality', 0.5)
            
            # Distance to nearest marker (normalized)
            marker_distance = corr.get('marker_distance_normalized', 1.0)
            
            # Compute confidence with higher weight for marker-based measurements
            if marker_confidence > 0:
                # Near a marker - high confidence
                confidence = 0.6 * marker_confidence + 0.3 * (1.0 / (1.0 + reprojection_error)) + 0.1 * pattern_quality
            else:
                # Away from markers - rely on pattern quality
                confidence = 0.8 * pattern_quality + 0.2 * (1.0 - marker_distance)
            
            # Convert to uncertainty in mm
            uncertainty = 1.0 - confidence
            pixel_to_mm = pattern_data.get('pixel_to_mm', 0.1)
            
            # ArUco provides better accuracy
            uncertainty_mm = uncertainty * pixel_to_mm * 0.6
            
            points.append(point)
            uncertainties.append(uncertainty_mm)
            confidences.append(confidence)
        
        points = np.array(points)
        uncertainties = np.array(uncertainties)
        confidences = np.array(confidences)
        
        # Create maps
        uncertainty_map = self._create_uncertainty_map(points, uncertainties)
        confidence_map = self._create_uncertainty_map(points, confidences)
        
        # Compute statistics
        marker_correspondences = [c for c in correspondences if c.get('marker_confidence', 0) > 0]
        
        statistics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'num_correspondences': len(correspondences),
            'num_marker_correspondences': len(marker_correspondences),
            'coverage_percent': (len(correspondences) / (self.width * self.height)) * 100,
            'mean_reprojection_error': np.mean([c.get('reprojection_error', 0) for c in marker_correspondences]) if marker_correspondences else 0,
            'markers_detected': len(marker_data)
        }
        
        return UncertaintyData(
            mean_uncertainty=np.mean(uncertainties),
            max_uncertainty=np.max(uncertainties),
            uncertainty_map=uncertainty_map,
            confidence_map=confidence_map,
            statistics=statistics
        )