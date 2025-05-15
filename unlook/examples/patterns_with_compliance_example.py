"""
Comprehensive example demonstrating advanced pattern generation with
ISO/ASTM 52902 compliance features.

This example shows how to:
1. Generate patterns with different algorithms
2. Measure uncertainty for each pattern type
3. Validate calibration using test objects
4. Generate certification reports
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from datetime import datetime

# Import pattern generators
from unlook.client.patterns.maze_pattern import MazePatternGenerator, MazeAlgorithm
from unlook.client.patterns.voronoi_pattern import VoronoiPatternGenerator
from unlook.client.patterns.hybrid_aruco_pattern import HybridArUcoPatternGenerator

# Import compliance modules
from unlook.client.scanning.compliance.uncertainty_measurement import (
    MazeUncertaintyMeasurement,
    VoronoiUncertaintyMeasurement,
    HybridArUcoUncertaintyMeasurement
)
from unlook.client.scanning.compliance.calibration_validation import CalibrationValidator
from unlook.client.scanning.compliance.certification_reporting import CertificationReporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_patterns_example():
    """Generate example patterns with all available algorithms."""
    output_dir = Path("./pattern_examples")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Maze Patterns
    logger.info("Generating maze patterns...")
    maze_gen = MazePatternGenerator(width=1280, height=720)
    
    for algorithm in MazeAlgorithm:
        pattern = maze_gen.generate(algorithm=algorithm, seed=42)
        filename = output_dir / f"maze_{algorithm.value}.png"
        cv2.imwrite(str(filename), pattern)
        logger.info(f"Saved {filename}")
    
    # 2. Voronoi Patterns
    logger.info("Generating Voronoi patterns...")
    voronoi_gen = VoronoiPatternGenerator(width=1280, height=720)
    
    for color_scheme in ['grayscale', 'binary', 'colored']:
        pattern = voronoi_gen.generate(num_points=100, color_scheme=color_scheme)
        filename = output_dir / f"voronoi_{color_scheme}.png"
        cv2.imwrite(str(filename), pattern)
        logger.info(f"Saved {filename}")
    
    # 3. Hybrid ArUco Patterns
    logger.info("Generating hybrid ArUco patterns...")
    hybrid_gen = HybridArUcoPatternGenerator(width=1280, height=720)
    
    for base_pattern in ['gray_code', 'phase_shift', 'checkerboard']:
        pattern, markers = hybrid_gen.generate(base_pattern=base_pattern, num_markers=9)
        filename = output_dir / f"hybrid_aruco_{base_pattern}.png"
        cv2.imwrite(str(filename), pattern)
        logger.info(f"Saved {filename} with {len(markers)} markers")


def demonstrate_uncertainty_measurement():
    """Demonstrate uncertainty measurement for ISO/ASTM 52902 compliance."""
    
    # Generate test patterns
    maze_gen = MazePatternGenerator(width=1280, height=720)
    voronoi_gen = VoronoiPatternGenerator(width=1280, height=720)
    hybrid_gen = HybridArUcoPatternGenerator(width=1280, height=720)
    
    # Generate patterns
    maze_pattern = maze_gen.generate(seed=42)
    voronoi_pattern = voronoi_gen.generate(num_points=100)
    hybrid_pattern, markers = hybrid_gen.generate(num_markers=9)
    
    # Create uncertainty measurements
    maze_uncertainty = MazeUncertaintyMeasurement((1280, 720))
    voronoi_uncertainty = VoronoiUncertaintyMeasurement((1280, 720))
    hybrid_uncertainty = HybridArUcoUncertaintyMeasurement((1280, 720))
    
    # Simulate correspondences (in real usage, these come from pattern matching)
    fake_correspondences = [
        {
            'point': [640, 360],
            'junction_confidence': 0.95,
            'topology_score': 0.88,
            'contrast_quality': 0.92,
            'marker_confidence': 0.0,
            'pattern_quality': 0.9
        }
        for _ in range(100)
    ]
    
    # Compute uncertainties
    pattern_data = {'pixel_to_mm': 0.1}
    
    maze_result = maze_uncertainty.compute_uncertainty(
        fake_correspondences, pattern_data
    )
    logger.info(f"Maze pattern - Mean uncertainty: {maze_result.mean_uncertainty:.3f}mm")
    
    voronoi_result = voronoi_uncertainty.compute_uncertainty(
        fake_correspondences, pattern_data
    )
    logger.info(f"Voronoi pattern - Mean uncertainty: {voronoi_result.mean_uncertainty:.3f}mm")
    
    hybrid_result = hybrid_uncertainty.compute_uncertainty(
        fake_correspondences, pattern_data
    )
    logger.info(f"Hybrid ArUco - Mean uncertainty: {hybrid_result.mean_uncertainty:.3f}mm")
    
    # Save uncertainty maps
    output_dir = Path("./uncertainty_maps")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "maze_uncertainty.png"), 
                (maze_result.uncertainty_map * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "voronoi_uncertainty.png"), 
                (voronoi_result.uncertainty_map * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "hybrid_uncertainty.png"), 
                (hybrid_result.uncertainty_map * 255).astype(np.uint8))


def demonstrate_calibration_validation():
    """Demonstrate calibration validation with test objects."""
    
    # Scanner specifications
    scanner_specs = {
        'name': 'UnLook Scanner',
        'model': 'UL-1000',
        'accuracy': 0.1,  # mm
        'resolution': (1280, 720),
        'drift_threshold': 0.05  # 5%
    }
    
    # Create validator
    validator = CalibrationValidator(scanner_specs)
    
    # Simulate point cloud of a 25mm reference sphere
    # In real usage, this comes from actual scanning
    num_points = 10000
    radius = 12.5  # 25mm diameter
    
    # Generate sphere points with some noise
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    
    noise = np.random.normal(0, 0.01, num_points)  # 0.01mm noise
    r = radius + noise
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    sphere_points = np.column_stack([x, y, z])
    
    # Validate with sphere
    result = validator.validate_with_test_object(
        sphere_points, 
        'sphere_25mm',
        save_results=True
    )
    
    logger.info(f"Calibration validation: {'PASSED' if result.passed else 'FAILED'}")
    logger.info(f"Diameter error: {result.errors['diameter_error']:.3f}mm")
    logger.info(f"Recommendations: {result.recommendations}")
    
    # Save validation history
    validator.save_validation_history("calibration_history.json")


def generate_certification_report():
    """Generate ISO/ASTM 52902 certification report."""
    
    # Scanner information
    scanner_info = {
        'name': 'UnLook Scanner',
        'model': 'UL-1000',
        'serial': 'UL2024001',
        'software_version': '1.0.0',
        'calibration_date': '2024-01-15'
    }
    
    # Create reporter
    reporter = CertificationReporter(scanner_info)
    
    # Prepare test data (in real usage, these come from actual tests)
    calibration_results = [
        {
            'test_date': datetime.now().isoformat(),
            'test_object': 'sphere_25mm',
            'passed': True,
            'measurements': {'diameter': 25.02},
            'errors': {'diameter_error': 0.02},
            'drift_detected': False
        }
    ]
    
    uncertainty_measurements = {
        'maze': {
            'mean_uncertainty': 0.08,
            'max_uncertainty': 0.15,
            'statistics': {
                'coverage_percent': 85.2,
                'num_correspondences': 5000
            }
        },
        'voronoi': {
            'mean_uncertainty': 0.06,
            'max_uncertainty': 0.12,
            'statistics': {
                'coverage_percent': 92.5,
                'num_correspondences': 8000
            }
        },
        'hybrid_aruco': {
            'mean_uncertainty': 0.05,
            'max_uncertainty': 0.10,
            'statistics': {
                'coverage_percent': 95.0,
                'num_correspondences': 10000
            }
        }
    }
    
    pattern_test_results = {
        'maze': {
            'test_date': datetime.now().isoformat(),
            'num_correspondences': 5000,
            'coverage_percent': 85.2,
            'mean_uncertainty': 0.08
        }
    }
    
    # Generate report
    report = reporter.generate_report(
        calibration_results,
        uncertainty_measurements,
        pattern_test_results,
        save_pdf=True  # Requires reportlab
    )
    
    logger.info(f"Generated certification report: {report.report_id}")
    logger.info(f"Overall compliance: {'PASS' if report.overall_compliance else 'FAIL'}")
    
    # Generate summary from multiple reports
    reporter.generate_summary_report([report])


def main():
    """Run all examples."""
    logger.info("Starting pattern generation and compliance examples...")
    
    # Generate all pattern types
    generate_patterns_example()
    
    # Demonstrate uncertainty measurement
    demonstrate_uncertainty_measurement()
    
    # Demonstrate calibration validation
    demonstrate_calibration_validation()
    
    # Generate certification report
    generate_certification_report()
    
    logger.info("Examples completed successfully!")


if __name__ == "__main__":
    main()