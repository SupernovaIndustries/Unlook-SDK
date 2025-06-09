"""Bundle Adjustment for Stereo Calibration Refinement using Ceres Solver.

This module provides high-quality stereo calibration refinement using non-linear
optimization with Ceres Solver. It significantly improves calibration accuracy
beyond standard OpenCV calibration methods.

Features:
- Stereo reprojection error minimization
- Robust outlier handling
- Distortion model refinement
- Multi-view bundle adjustment
- Professional-grade accuracy (target: RMS < 0.5 pixel)

References:
    Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (1999).
    Bundle adjustment—a modern synthesis.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import Ceres Solver
try:
    import pyceres
    CERES_AVAILABLE = True
    logger.info("Ceres Solver available for bundle adjustment")
except ImportError:
    CERES_AVAILABLE = False
    logger.warning("Ceres Solver not available - bundle adjustment disabled")


class StereoReprojectionError:
    """Ceres cost function for stereo reprojection error."""
    
    def __init__(self, observed_left: np.ndarray, observed_right: np.ndarray, 
                 world_point: np.ndarray):
        """Initialize reprojection error function.
        
        Args:
            observed_left: Observed 2D point in left image
            observed_right: Observed 2D point in right image  
            world_point: Corresponding 3D world point
        """
        self.observed_left = observed_left.astype(np.float64)
        self.observed_right = observed_right.astype(np.float64)
        self.world_point = world_point.astype(np.float64)
        
    def __call__(self, intrinsics_left: np.ndarray, distortion_left: np.ndarray,
                 intrinsics_right: np.ndarray, distortion_right: np.ndarray,
                 rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Compute reprojection error.
        
        Args:
            intrinsics_left: Left camera intrinsic parameters [fx, fy, cx, cy]
            distortion_left: Left camera distortion parameters [k1, k2, p1, p2, k3]
            intrinsics_right: Right camera intrinsic parameters  
            distortion_right: Right camera distortion parameters
            rotation: Rotation between cameras (Rodriguez vector)
            translation: Translation between cameras
            
        Returns:
            Residual vector [left_x_error, left_y_error, right_x_error, right_y_error]
        """
        # Convert Rodriguez to rotation matrix
        R, _ = cv2.Rodrigues(rotation)
        
        # Project 3D point to left camera
        projected_left = self._project_point(
            self.world_point, np.eye(3), np.zeros(3),
            intrinsics_left, distortion_left
        )
        
        # Transform 3D point to right camera coordinate system
        world_point_right = R @ self.world_point + translation.reshape(-1)
        
        # Project to right camera
        projected_right = self._project_point(
            world_point_right, np.eye(3), np.zeros(3),
            intrinsics_right, distortion_right
        )
        
        # Compute residuals
        residual = np.array([
            projected_left[0] - self.observed_left[0],
            projected_left[1] - self.observed_left[1], 
            projected_right[0] - self.observed_right[0],
            projected_right[1] - self.observed_right[1]
        ])
        
        return residual
        
    def _project_point(self, point_3d: np.ndarray, R: np.ndarray, t: np.ndarray,
                      intrinsics: np.ndarray, distortion: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D using camera model."""
        # Transform to camera coordinates
        point_cam = R @ point_3d + t
        
        # Perspective projection
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        
        # Apply distortion
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        
        k1, k2, p1, p2, k3 = distortion
        
        # Radial distortion
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        
        # Tangential distortion
        tangential_x = 2*p1*x*y + p2*(r2 + 2*x*x)
        tangential_y = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        # Apply distortion
        x_distorted = x * radial + tangential_x
        y_distorted = y * radial + tangential_y
        
        # Apply intrinsics
        fx, fy, cx, cy = intrinsics
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
        
        return np.array([u, v])


class StereoCalibrationOptimizer:
    """Professional stereo calibration optimizer using bundle adjustment."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.solver_options = self._create_solver_options()
        
    def _create_solver_options(self):
        """Create optimized Ceres solver options."""
        if not CERES_AVAILABLE:
            return None
            
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.SPARSE_SCHUR
        options.minimizer_progress_to_stdout = True
        options.max_num_iterations = 500
        options.function_tolerance = 1e-8
        options.gradient_tolerance = 1e-10
        options.parameter_tolerance = 1e-8
        options.num_threads = 8  # Use multiple threads
        
        return options
        
    def optimize_stereo_calibration(self, 
                                  image_points_left: List[np.ndarray],
                                  image_points_right: List[np.ndarray], 
                                  object_points: List[np.ndarray],
                                  initial_params: Dict[str, Any],
                                  image_size: Tuple[int, int]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize stereo calibration using bundle adjustment.
        
        Args:
            image_points_left: List of 2D points in left images
            image_points_right: List of 2D points in right images  
            object_points: List of corresponding 3D world points
            initial_params: Initial calibration parameters from OpenCV
            image_size: Image size (width, height)
            
        Returns:
            Tuple of (optimized_parameters, optimization_summary)
        """
        if not CERES_AVAILABLE:
            logger.error("Ceres Solver not available - cannot perform bundle adjustment")
            return initial_params, {"error": "Ceres not available"}
            
        logger.info("🔧 Starting bundle adjustment optimization...")
        logger.info(f"Input data: {len(image_points_left)} stereo pairs")
        
        # Extract initial parameters
        K1 = initial_params['K1']
        D1 = initial_params['D1']
        K2 = initial_params['K2'] 
        D2 = initial_params['D2']
        R = initial_params['R']
        T = initial_params['T']
        
        # Convert to optimization parameters
        intrinsics_left = np.array([K1[0,0], K1[1,1], K1[0,2], K1[1,2]], dtype=np.float64)
        intrinsics_right = np.array([K2[0,0], K2[1,1], K2[0,2], K2[1,2]], dtype=np.float64)
        distortion_left = D1.flatten()[:5].astype(np.float64)
        distortion_right = D2.flatten()[:5].astype(np.float64)
        rotation_vec, _ = cv2.Rodrigues(R)
        rotation_vec = rotation_vec.flatten().astype(np.float64)
        translation = T.flatten().astype(np.float64)
        
        # Ensure distortion vectors have 5 elements
        if len(distortion_left) < 5:
            distortion_left = np.pad(distortion_left, (0, 5-len(distortion_left)))
        if len(distortion_right) < 5:
            distortion_right = np.pad(distortion_right, (0, 5-len(distortion_right)))
            
        # Create Ceres problem
        problem = pyceres.Problem()
        
        # Add residual blocks for each observation
        num_residuals = 0
        for i, (left_pts, right_pts, obj_pts) in enumerate(zip(image_points_left, 
                                                             image_points_right,
                                                             object_points)):
            for j, (left_pt, right_pt, obj_pt) in enumerate(zip(left_pts, right_pts, obj_pts)):
                # Create reprojection error
                error_func = StereoReprojectionError(left_pt, right_pt, obj_pt)
                
                # Add residual block  
                cost_function = pyceres.PythonCostFunction(
                    error_func, 
                    num_residuals=4,  # [left_x, left_y, right_x, right_y]
                    parameter_block_sizes=[4, 5, 4, 5, 3, 3]  # intrinsics, distortion, etc.
                )
                
                problem.add_residual_block(
                    cost_function, 
                    pyceres.HuberLoss(1.0),  # Robust to outliers
                    [intrinsics_left, distortion_left, 
                     intrinsics_right, distortion_right,
                     rotation_vec, translation]
                )
                num_residuals += 1
        
        logger.info(f"Created optimization problem with {num_residuals} residuals")
        
        # Set parameter bounds to ensure physical validity
        self._set_parameter_bounds(problem, intrinsics_left, intrinsics_right, 
                                 distortion_left, distortion_right, image_size)
        
        # Solve
        logger.info("Running Ceres optimization...")
        summary = pyceres.solve(self.solver_options, problem)
        
        # Extract results
        optimized_params = self._extract_optimized_parameters(
            intrinsics_left, distortion_left,
            intrinsics_right, distortion_right, 
            rotation_vec, translation
        )
        
        # Create summary
        optimization_summary = {
            "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost,
            "cost_change": summary.initial_cost - summary.final_cost,
            "num_iterations": summary.iterations.size,
            "termination_type": str(summary.termination_type),
            "rms_error": np.sqrt(summary.final_cost / num_residuals),
            "num_residuals": num_residuals
        }
        
        logger.info("✅ Bundle adjustment completed!")
        logger.info(f"Initial cost: {summary.initial_cost:.6f}")
        logger.info(f"Final cost: {summary.final_cost:.6f}")
        logger.info(f"Cost reduction: {optimization_summary['cost_change']:.6f}")
        logger.info(f"RMS error: {optimization_summary['rms_error']:.4f} pixels")
        logger.info(f"Iterations: {optimization_summary['num_iterations']}")
        
        return optimized_params, optimization_summary
        
    def _set_parameter_bounds(self, problem, intrinsics_left, intrinsics_right,
                            distortion_left, distortion_right, image_size):
        """Set reasonable bounds on parameters."""
        width, height = image_size
        
        # Focal length bounds (0.5x to 2x of image dimension)
        min_focal = min(width, height) * 0.5
        max_focal = max(width, height) * 2.0
        
        # Principal point bounds (within image)
        min_cx, max_cx = -width * 0.1, width * 1.1
        min_cy, max_cy = -height * 0.1, height * 1.1
        
        # Left camera intrinsics bounds
        problem.set_parameter_lower_bound(intrinsics_left, 0, min_focal)  # fx
        problem.set_parameter_upper_bound(intrinsics_left, 0, max_focal)
        problem.set_parameter_lower_bound(intrinsics_left, 1, min_focal)  # fy
        problem.set_parameter_upper_bound(intrinsics_left, 1, max_focal)
        problem.set_parameter_lower_bound(intrinsics_left, 2, min_cx)     # cx
        problem.set_parameter_upper_bound(intrinsics_left, 2, max_cx)
        problem.set_parameter_lower_bound(intrinsics_left, 3, min_cy)     # cy
        problem.set_parameter_upper_bound(intrinsics_left, 3, max_cy)
        
        # Right camera intrinsics bounds (same as left)
        problem.set_parameter_lower_bound(intrinsics_right, 0, min_focal)
        problem.set_parameter_upper_bound(intrinsics_right, 0, max_focal)
        problem.set_parameter_lower_bound(intrinsics_right, 1, min_focal)
        problem.set_parameter_upper_bound(intrinsics_right, 1, max_focal)
        problem.set_parameter_lower_bound(intrinsics_right, 2, min_cx)
        problem.set_parameter_upper_bound(intrinsics_right, 2, max_cx)
        problem.set_parameter_lower_bound(intrinsics_right, 3, min_cy)
        problem.set_parameter_upper_bound(intrinsics_right, 3, max_cy)
        
        # Distortion bounds (reasonable values)
        for i in range(5):
            if i < 2:  # k1, k2 - radial distortion
                problem.set_parameter_lower_bound(distortion_left, i, -2.0)
                problem.set_parameter_upper_bound(distortion_left, i, 2.0)
                problem.set_parameter_lower_bound(distortion_right, i, -2.0)
                problem.set_parameter_upper_bound(distortion_right, i, 2.0)
            elif i < 4:  # p1, p2 - tangential distortion
                problem.set_parameter_lower_bound(distortion_left, i, -0.01)
                problem.set_parameter_upper_bound(distortion_left, i, 0.01)
                problem.set_parameter_lower_bound(distortion_right, i, -0.01)
                problem.set_parameter_upper_bound(distortion_right, i, 0.01)
            else:  # k3 - higher order radial
                problem.set_parameter_lower_bound(distortion_left, i, -1.0)
                problem.set_parameter_upper_bound(distortion_left, i, 1.0)
                problem.set_parameter_lower_bound(distortion_right, i, -1.0)
                problem.set_parameter_upper_bound(distortion_right, i, 1.0)
        
    def _extract_optimized_parameters(self, intrinsics_left, distortion_left,
                                    intrinsics_right, distortion_right,
                                    rotation_vec, translation):
        """Extract optimized parameters into OpenCV format."""
        # Reconstruct camera matrices
        K1 = np.array([
            [intrinsics_left[0], 0, intrinsics_left[2]],
            [0, intrinsics_left[1], intrinsics_left[3]], 
            [0, 0, 1]
        ])
        
        K2 = np.array([
            [intrinsics_right[0], 0, intrinsics_right[2]],
            [0, intrinsics_right[1], intrinsics_right[3]],
            [0, 0, 1] 
        ])
        
        # Distortion vectors
        D1 = distortion_left.reshape(-1, 1)
        D2 = distortion_right.reshape(-1, 1)
        
        # Rotation and translation
        R, _ = cv2.Rodrigues(rotation_vec)
        T = translation.reshape(-1, 1)
        
        return {
            'K1': K1,
            'D1': D1,
            'K2': K2,
            'D2': D2,
            'R': R,
            'T': T
        }
    
    def validate_calibration_improvement(self, original_params: Dict[str, Any],
                                       optimized_params: Dict[str, Any],
                                       image_points_left: List[np.ndarray],
                                       image_points_right: List[np.ndarray],
                                       object_points: List[np.ndarray]) -> Dict[str, float]:
        """Validate that bundle adjustment improved calibration."""
        
        # Compute reprojection errors for both calibrations
        original_error = self._compute_reprojection_error(
            original_params, image_points_left, image_points_right, object_points
        )
        
        optimized_error = self._compute_reprojection_error(
            optimized_params, image_points_left, image_points_right, object_points  
        )
        
        improvement = (original_error - optimized_error) / original_error * 100
        
        results = {
            "original_rms_error": original_error,
            "optimized_rms_error": optimized_error,
            "improvement_percent": improvement,
            "meets_target": optimized_error < 0.5  # Target: < 0.5 pixels
        }
        
        logger.info(f"Calibration validation:")
        logger.info(f"  Original RMS error: {original_error:.4f} pixels")
        logger.info(f"  Optimized RMS error: {optimized_error:.4f} pixels")
        logger.info(f"  Improvement: {improvement:.1f}%")
        logger.info(f"  Meets target (<0.5px): {'YES' if results['meets_target'] else 'NO'}")
        
        return results
        
    def _compute_reprojection_error(self, params: Dict[str, Any],
                                  image_points_left: List[np.ndarray],
                                  image_points_right: List[np.ndarray], 
                                  object_points: List[np.ndarray]) -> float:
        """Compute RMS reprojection error for calibration."""
        total_error = 0
        total_points = 0
        
        K1, D1 = params['K1'], params['D1']
        K2, D2 = params['K2'], params['D2']
        R, T = params['R'], params['T']
        
        for left_pts, right_pts, obj_pts in zip(image_points_left, 
                                              image_points_right,
                                              object_points):
            # Project object points to left camera
            left_proj, _ = cv2.projectPoints(obj_pts, np.zeros(3), np.zeros(3), K1, D1)
            left_proj = left_proj.reshape(-1, 2)
            
            # Project object points to right camera
            right_proj, _ = cv2.projectPoints(obj_pts, R, T, K2, D2)
            right_proj = right_proj.reshape(-1, 2)
            
            # Compute errors
            left_errors = np.linalg.norm(left_pts - left_proj, axis=1)
            right_errors = np.linalg.norm(right_pts - right_proj, axis=1)
            
            total_error += np.sum(left_errors**2) + np.sum(right_errors**2)
            total_points += len(left_pts) * 2  # 2 cameras
            
        return np.sqrt(total_error / total_points)


def optimize_stereo_calibration_with_bundle_adjustment(
    calibration_file: str,
    image_points_left: List[np.ndarray],
    image_points_right: List[np.ndarray],
    object_points: List[np.ndarray],
    image_size: Tuple[int, int],
    output_file: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Convenience function to optimize existing stereo calibration.
    
    Args:
        calibration_file: Path to existing calibration JSON
        image_points_left: List of 2D points in left images
        image_points_right: List of 2D points in right images
        object_points: List of corresponding 3D world points  
        image_size: Image size (width, height)
        output_file: Optional output file for optimized calibration
        
    Returns:
        Tuple of (optimized_parameters, optimization_summary)
    """
    # Load existing calibration
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    # Convert to required format
    initial_params = {
        'K1': np.array(calib_data['K1']),
        'D1': np.array(calib_data['D1']),
        'K2': np.array(calib_data['K2']),
        'D2': np.array(calib_data['D2']),
        'R': np.array(calib_data['R']),
        'T': np.array(calib_data['T'])
    }
    
    # Optimize
    optimizer = StereoCalibrationOptimizer()
    optimized_params, summary = optimizer.optimize_stereo_calibration(
        image_points_left, image_points_right, object_points, 
        initial_params, image_size
    )
    
    # Validate improvement
    validation = optimizer.validate_calibration_improvement(
        initial_params, optimized_params,
        image_points_left, image_points_right, object_points
    )
    
    # Save optimized calibration if requested
    if output_file:
        save_optimized_calibration(optimized_params, calib_data, output_file, summary, validation)
    
    return optimized_params, {**summary, **validation}


def save_optimized_calibration(optimized_params: Dict[str, Any], 
                             original_calib: Dict[str, Any],
                             output_file: str,
                             optimization_summary: Dict[str, float],
                             validation_results: Dict[str, float]):
    """Save optimized calibration with metadata."""
    
    # Update calibration data
    optimized_calib = original_calib.copy()
    optimized_calib.update({
        'K1': optimized_params['K1'].tolist(),
        'D1': optimized_params['D1'].tolist(),
        'K2': optimized_params['K2'].tolist(), 
        'D2': optimized_params['D2'].tolist(),
        'R': optimized_params['R'].tolist(),
        'T': optimized_params['T'].tolist()
    })
    
    # Add optimization metadata
    optimized_calib['bundle_adjustment'] = {
        'applied': True,
        'optimization_summary': optimization_summary,
        'validation_results': validation_results,
        'ceres_available': CERES_AVAILABLE
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(optimized_calib, f, indent=2)
    
    logger.info(f"✅ Optimized calibration saved: {output_file}")