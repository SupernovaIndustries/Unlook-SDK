"""
Single camera structured light scanning implementation for UnLook SDK.

This module provides classes and functions for performing 3D scanning with a
single camera and projector setup. It implements structured light scanning
techniques that allow for 3D reconstruction without needing stereo cameras.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import open3d as o3d
    from open3d import geometry as o3dg
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.info("open3d not installed. 3D mesh visualization and advanced filtering will be limited.")
    OPEN3D_AVAILABLE = False
    # Create placeholder for o3dg when open3d is not available
    class PlaceholderO3DG:
        class PointCloud:
            pass
        class TriangleMesh:
            pass
    o3dg = PlaceholderO3DG


class SingleCameraCalibrator:
    """
    Calibrates a single camera and projector system for structured light scanning.
    
    This class handles the calibration of both the camera and projector intrinsics
    and the extrinsic relationship between them, which is necessary for accurate
    3D reconstruction with a single camera setup.
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), 
                 checkerboard_square_size: float = 25.0):
        """
        Initialize the calibrator.
        
        Args:
            checkerboard_size: Number of inner corners (width, height) of the calibration board
            checkerboard_square_size: Size of checkerboard squares in mm
        """
        self.checkerboard_size = checkerboard_size
        self.checkerboard_square_size = checkerboard_square_size
        
        # Initialize calibration data containers
        self.camera_matrix = None
        self.camera_dist_coeffs = None
        self.projector_matrix = None
        self.projector_dist_coeffs = None
        self.R = None  # Rotation from camera to projector
        self.T = None  # Translation from camera to projector
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= checkerboard_square_size  # Convert to mm
        
        logger.info(f"Initialized SingleCameraCalibrator with {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard")
    
    def calibrate_camera(self, images: List[np.ndarray]) -> bool:
        """
        Calibrate the camera using checkerboard images.
        
        Args:
            images: List of calibration images containing checkerboard
            
        Returns:
            Success flag
        """
        if not images:
            logger.error("No calibration images provided")
            return False
        
        # Arrays to store object points and image points from all images
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        img_shape = None
        
        logger.info(f"Calibrating camera with {len(images)} images")
        
        for i, img in enumerate(images):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            if img_shape is None:
                img_shape = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                objpoints.append(self.objp)
                
                # Refine corner locations for better accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                
                # Optionally draw and display the corners for debugging
                # img_corners = cv2.drawChessboardCorners(img.copy(), self.checkerboard_size, corners, ret)
                # cv2.imwrite(f"debug_corners_{i}.jpg", img_corners)
            else:
                logger.warning(f"Could not find checkerboard corners in image {i}")
        
        if not objpoints:
            logger.error("Could not find checkerboard corners in any image")
            return False
        
        logger.info(f"Found checkerboard corners in {len(objpoints)}/{len(images)} images")
        
        # Calibrate camera
        ret, self.camera_matrix, self.camera_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        # Calculate re-projection error to evaluate calibration quality
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.camera_dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        if len(objpoints) > 0:
            mean_error /= len(objpoints)
        
        logger.info(f"Camera calibration complete. Re-projection error: {mean_error}")
        return True
    
    def calibrate_projector(self, camera_images: List[np.ndarray], 
                           projected_patterns: List[np.ndarray]) -> bool:
        """
        Calibrate the projector using checkerboard and projected patterns.
        
        The checkerboard should be visible in the camera images, and the projected
        patterns should be visible on the checkerboard.
        
        Args:
            camera_images: List of camera images with checkerboard
            projected_patterns: List of patterns projected onto the checkerboard
            
        Returns:
            Success flag
        """
        if not camera_images or not projected_patterns:
            logger.error("No calibration images provided")
            return False
        
        if len(camera_images) != len(projected_patterns):
            logger.error("Number of camera images and projected patterns don't match")
            return False
        
        if self.camera_matrix is None:
            logger.error("Camera must be calibrated first")
            return False
        
        # Arrays to store object points and image points
        objpoints = []  # 3D world points
        cam_imgpoints = []  # 2D points in camera
        proj_imgpoints = []  # 2D points in projector
        
        proj_shape = projected_patterns[0].shape[::-1] if projected_patterns else None
        
        for i, (cam_img, proj_pattern) in enumerate(zip(camera_images, projected_patterns)):
            if len(cam_img.shape) == 3:
                gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cam_img
            
            # Find checkerboard corners in camera image
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Find the projected pattern on the checkerboard
                # This is a simplified approach - in practice, you'd use Gray code or phase shift
                # patterns to establish more accurate correspondence
                
                # For this example, we'll estimate the projector corners based on camera corners
                # In a real implementation, you'd decode the actual projected pattern
                
                # Get pose of checkerboard relative to camera
                _, rvec, tvec = cv2.solvePnP(
                    self.objp, corners, self.camera_matrix, self.camera_dist_coeffs
                )
                
                # Project corners to projector (simplified - assumes projector is similar to camera)
                # In reality, you'd use the decoded patterns to find the actual projector coordinates
                # of the checkerboard corners
                
                # Placeholder for actual projector corner detection
                proj_corners = np.zeros_like(corners)
                
                # Store the points
                objpoints.append(self.objp)
                cam_imgpoints.append(corners)
                proj_imgpoints.append(proj_corners)
        
        if not objpoints:
            logger.error("Could not process any calibration images")
            return False
        
        # For proper projector calibration, you'd use the method described in
        # "Accurate and efficient stereo matching by the projector-camera calibration" by Wu et al.
        # or similar approaches that specifically handle projector calibration
        
        # This simplification just demonstrates the structure - actual projector calibration
        # requires more sophisticated pattern decoding
        
        # Use camera-projector correspondence to calibrate projector
        # stereo_ret, self.camera_matrix, self.camera_dist_coeffs, \
        # self.projector_matrix, self.projector_dist_coeffs, \
        # self.R, self.T, E, F = cv2.stereoCalibrate(
        #     objpoints, cam_imgpoints, proj_imgpoints,
        #     self.camera_matrix, self.camera_dist_coeffs,
        #     None, None, None,
        #     flags=cv2.CALIB_FIX_INTRINSIC
        # )
        
        # Simulated projector calibration (for demo purposes)
        self.projector_matrix = np.array([
            [1000.0, 0, proj_shape[0]/2 if proj_shape else 640],
            [0, 1000.0, proj_shape[1]/2 if proj_shape else 480],
            [0, 0, 1]
        ])
        self.projector_dist_coeffs = np.zeros(5)
        self.R = np.eye(3)  # Identity rotation
        self.T = np.array([[-100.0], [0.0], [0.0]])  # 100mm to the left of camera
        
        logger.info("Projector calibration complete (simulated)")
        return True
    
    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration data to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            Success flag
        """
        if (self.camera_matrix is None or self.projector_matrix is None or 
            self.R is None or self.T is None):
            logger.error("Calibration not complete")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save calibration data
        np.savez(
            filepath,
            camera_matrix=self.camera_matrix,
            camera_dist_coeffs=self.camera_dist_coeffs,
            projector_matrix=self.projector_matrix,
            projector_dist_coeffs=self.projector_dist_coeffs,
            R=self.R,
            T=self.T
        )
        
        logger.info(f"Saved calibration data to {filepath}")
        return True
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration data from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Success flag
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        try:
            data = np.load(filepath)
            self.camera_matrix = data['camera_matrix']
            self.camera_dist_coeffs = data['camera_dist_coeffs']
            self.projector_matrix = data['projector_matrix']
            self.projector_dist_coeffs = data['projector_dist_coeffs']
            self.R = data['R']
            self.T = data['T']
            
            logger.info(f"Loaded calibration data from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
    
    def generate_calibration_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a calibration pattern for projector.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
            
        Returns:
            Calibration pattern image
        """
        # Create a checkerboard pattern
        pattern = np.zeros((height, width), dtype=np.uint8)
        square_size = min(width, height) // 20  # Smaller squares for better resolution
        
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    y_end = min(y + square_size, height)
                    x_end = min(x + square_size, width)
                    pattern[y:y_end, x:x_end] = 255
        
        return pattern


class SingleCameraStructuredLight:
    """
    Structured light scanning implementation using a single camera and projector.
    
    This class handles pattern generation, projection, image capture, and 3D
    reconstruction using a single camera and projector setup.
    """
    
    def __init__(self, camera_matrix: np.ndarray, camera_dist_coeffs: np.ndarray,
                projector_matrix: np.ndarray, projector_dist_coeffs: np.ndarray,
                R: np.ndarray, T: np.ndarray,
                projector_width: int = 1920, projector_height: int = 1080):
        """
        Initialize the structured light scanner.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            camera_dist_coeffs: Camera distortion coefficients
            projector_matrix: Projector intrinsic matrix
            projector_dist_coeffs: Projector distortion coefficients
            R: Rotation matrix from camera to projector
            T: Translation vector from camera to projector
            projector_width: Projector width in pixels
            projector_height: Projector height in pixels
        """
        self.camera_matrix = camera_matrix
        self.camera_dist_coeffs = camera_dist_coeffs
        self.projector_matrix = projector_matrix
        self.projector_dist_coeffs = projector_dist_coeffs
        self.R = R
        self.T = T
        self.projector_width = projector_width
        self.projector_height = projector_height
        
        # Compute number of patterns needed for gray code
        self.bits_x = int(np.ceil(np.log2(projector_width)))
        self.bits_y = int(np.ceil(np.log2(projector_height)))
        self.num_patterns = 2 + 2 * (self.bits_x + self.bits_y)  # Including white and black
        
        logger.info(f"Initialized SingleCameraStructuredLight scanner")
    
    @classmethod
    def from_calibration_file(cls, calib_file: str, 
                             projector_width: int = 1920, 
                             projector_height: int = 1080) -> 'SingleCameraStructuredLight':
        """
        Create scanner from calibration file.
        
        Args:
            calib_file: Path to calibration file
            projector_width: Projector width
            projector_height: Projector height
            
        Returns:
            Initialized scanner
        """
        # Load calibration data
        try:
            data = np.load(calib_file)
            camera_matrix = data['camera_matrix']
            camera_dist_coeffs = data['camera_dist_coeffs']
            projector_matrix = data['projector_matrix']
            projector_dist_coeffs = data['projector_dist_coeffs']
            R = data['R']
            T = data['T']
        except Exception as e:
            logger.error(f"Error loading calibration file: {e}")
            logger.warning("Using default calibration parameters")
            # Create default parameters
            camera_matrix = np.array([
                [1000.0, 0, 640.0],
                [0, 1000.0, 480.0],
                [0, 0, 1]
            ])
            camera_dist_coeffs = np.zeros(5)
            projector_matrix = np.array([
                [1000.0, 0, 960.0],
                [0, 1000.0, 540.0],
                [0, 0, 1]
            ])
            projector_dist_coeffs = np.zeros(5)
            R = np.eye(3)  # Identity rotation
            T = np.array([[-100.0], [0.0], [0.0]])  # 100mm to the left of camera
        
        return cls(
            camera_matrix, camera_dist_coeffs,
            projector_matrix, projector_dist_coeffs,
            R, T, projector_width, projector_height
        )
    
    def generate_gray_code_patterns(self) -> List[Dict[str, Any]]:
        """
        Generate Gray code patterns for projector.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Generate white and black calibration images
        white_image = np.ones((self.projector_height, self.projector_width), dtype=np.uint8) * 255
        black_image = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
        
        # Add white and black images
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(white_image),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(black_image),
            "name": "black"
        })
        
        # Generate horizontal patterns (X axis)
        for bit in range(self.bits_x):
            # Generate Gray code pattern for this bit
            pattern_img = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
            
            for x in range(self.projector_width):
                # Convert binary to Gray code
                gray_x = x ^ (x >> 1)
                # Check if this bit is set in the Gray code
                if (gray_x >> bit) & 1:
                    pattern_img[:, x] = 255
            
            # Add pattern and its inverse
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(pattern_img),
                "name": f"gray_code_x_{bit}"
            })
            
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(255 - pattern_img),
                "name": f"gray_code_x_inv_{bit}"
            })
        
        # Generate vertical patterns (Y axis)
        for bit in range(self.bits_y):
            # Generate Gray code pattern for this bit
            pattern_img = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
            
            for y in range(self.projector_height):
                # Convert binary to Gray code
                gray_y = y ^ (y >> 1)
                # Check if this bit is set in the Gray code
                if (gray_y >> bit) & 1:
                    pattern_img[y, :] = 255
            
            # Add pattern and its inverse
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(pattern_img),
                "name": f"gray_code_y_{bit}"
            })
            
            patterns.append({
                "pattern_type": "raw_image",
                "image": self._encode_image(255 - pattern_img),
                "name": f"gray_code_y_inv_{bit}"
            })
        
        logger.info(f"Generated {len(patterns)} Gray code patterns")
        return patterns
    
    def generate_phase_shift_patterns(self, num_shifts: int = 3, 
                                     frequencies: List[int] = None) -> List[Dict[str, Any]]:
        """
        Generate phase shift patterns for projector.
        
        Args:
            num_shifts: Number of phase shifts per frequency
            frequencies: List of frequencies to use (default: [16, 32, 64])
            
        Returns:
            List of pattern dictionaries
        """
        if frequencies is None:
            frequencies = [16, 32, 64]
            
        patterns = []
        
        # Generate white and black calibration images
        white_image = np.ones((self.projector_height, self.projector_width), dtype=np.uint8) * 255
        black_image = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(white_image),
            "name": "white"
        })
        
        patterns.append({
            "pattern_type": "raw_image",
            "image": self._encode_image(black_image),
            "name": "black"
        })
        
        # Generate horizontal phase shift patterns
        for freq in frequencies:
            for shift in range(num_shifts):
                phase_offset = 2 * np.pi * shift / num_shifts
                
                # Create horizontal phase pattern
                h_pattern = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
                for x in range(self.projector_width):
                    val = 127.5 + 127.5 * np.cos(2 * np.pi * x / freq + phase_offset)
                    h_pattern[:, x] = val
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._encode_image(h_pattern),
                    "name": f"h_phase_{freq}_{shift}"
                })
        
        # Generate vertical phase shift patterns
        for freq in frequencies:
            for shift in range(num_shifts):
                phase_offset = 2 * np.pi * shift / num_shifts
                
                # Create vertical phase pattern
                v_pattern = np.zeros((self.projector_height, self.projector_width), dtype=np.uint8)
                for y in range(self.projector_height):
                    val = 127.5 + 127.5 * np.cos(2 * np.pi * y / freq + phase_offset)
                    v_pattern[y, :] = val
                
                patterns.append({
                    "pattern_type": "raw_image",
                    "image": self._encode_image(v_pattern),
                    "name": f"v_phase_{freq}_{shift}"
                })
        
        logger.info(f"Generated {len(patterns)} phase shift patterns")
        return patterns
    
    def decode_gray_code(self, captured_images: List[np.ndarray], 
                        threshold: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to obtain projector-camera correspondence.
        
        Args:
            captured_images: List of captured images (white, black, patterns)
            threshold: Threshold for valid pixels
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        if len(captured_images) < 2 + 2 * (self.bits_x + self.bits_y):
            logger.error(f"Not enough patterns: {len(captured_images)} provided, " 
                        f"{2 + 2 * (self.bits_x + self.bits_y)} required")
            return None, None
        
        # Extract white and black images
        white_img = captured_images[0]
        black_img = captured_images[1]
        
        # Convert to grayscale if needed
        if len(white_img.shape) == 3:
            white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        if len(black_img.shape) == 3:
            black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask
        mask = self._compute_shadow_mask(black_img, white_img, threshold)
        
        # Extract pattern images (starting from index 2)
        pattern_images = captured_images[2:2 + 2 * (self.bits_x + self.bits_y)]
        
        # Convert pattern images to grayscale if needed
        gray_patterns = []
        for img in pattern_images:
            if len(img.shape) == 3:
                gray_patterns.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                gray_patterns.append(img)
        
        # Get image dimensions
        height, width = mask.shape
        
        # Initialize projector correspondence map
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue  # Skip shadowed pixels
                
                # Decode X coordinate (horizontal patterns)
                x_code = 0
                for bit in range(self.bits_x):
                    # Get normal pattern and its inverse for this bit
                    idx = 2 * bit
                    normal = gray_patterns[idx][y, x]
                    inverse = gray_patterns[idx + 1][y, x]
                    
                    # Check if difference is significant enough
                    if abs(int(normal) - int(inverse)) > threshold:
                        # Set bit if normal is brighter than inverse
                        if normal > inverse:
                            x_code |= (1 << bit)
                
                # Decode Y coordinate (vertical patterns)
                y_code = 0
                for bit in range(self.bits_y):
                    # Get normal pattern and its inverse for this bit
                    idx = 2 * self.bits_x + 2 * bit
                    normal = gray_patterns[idx][y, x]
                    inverse = gray_patterns[idx + 1][y, x]
                    
                    # Check if difference is significant enough
                    if abs(int(normal) - int(inverse)) > threshold:
                        # Set bit if normal is brighter than inverse
                        if normal > inverse:
                            y_code |= (1 << bit)
                
                # Convert from Gray code to binary
                x_proj = self._gray_to_binary(x_code)
                y_proj = self._gray_to_binary(y_code)
                
                # Store projector coordinates if within bounds
                if 0 <= x_proj < self.projector_width and 0 <= y_proj < self.projector_height:
                    cam_proj[y, x, 0] = y_proj  # Row (V)
                    cam_proj[y, x, 1] = x_proj  # Column (U)
        
        return cam_proj, mask
    
    def decode_phase_shift(self, captured_images: List[np.ndarray], 
                          num_shifts: int = 3, 
                          frequencies: List[int] = None,
                          threshold: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode phase shift patterns to obtain projector-camera correspondence.
        
        Args:
            captured_images: List of captured images (white, black, patterns)
            num_shifts: Number of phase shifts per frequency
            frequencies: List of frequencies used
            threshold: Threshold for valid pixels
            
        Returns:
            Tuple of (cam_proj_correspondence, mask)
        """
        if frequencies is None:
            frequencies = [16, 32, 64]
            
        if len(captured_images) < 2 + len(frequencies) * num_shifts * 2:  # *2 for horizontal and vertical
            logger.error(f"Not enough patterns: {len(captured_images)} provided, " 
                        f"{2 + len(frequencies) * num_shifts * 2} required")
            return None, None
        
        # Extract white and black images
        white_img = captured_images[0]
        black_img = captured_images[1]
        
        # Convert to grayscale if needed
        if len(white_img.shape) == 3:
            white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        if len(black_img.shape) == 3:
            black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        
        # Compute shadow mask
        mask = self._compute_shadow_mask(black_img, white_img, threshold)
        
        # Get image dimensions
        height, width = mask.shape
        
        # Initialize projector correspondence map
        cam_proj = np.zeros((height, width, 2), dtype=np.float32)
        
        # Extract pattern images (starting from index 2)
        pattern_images = captured_images[2:]
        
        # Convert pattern images to grayscale if needed
        gray_patterns = []
        for img in pattern_images:
            if len(img.shape) == 3:
                gray_patterns.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                gray_patterns.append(img)
        
        # Split patterns into horizontal and vertical
        h_patterns = gray_patterns[:len(frequencies) * num_shifts]
        v_patterns = gray_patterns[len(frequencies) * num_shifts:]
        
        # Decode phase shifts for each frequency
        h_phases = []  # Horizontal phase maps
        v_phases = []  # Vertical phase maps
        
        # Process horizontal patterns
        for f_idx, freq in enumerate(frequencies):
            # Get patterns for this frequency
            freq_patterns = h_patterns[f_idx * num_shifts:(f_idx + 1) * num_shifts]
            
            # Calculate phase using least squares
            A = np.zeros((num_shifts, 2))
            for i in range(num_shifts):
                phase = 2 * np.pi * i / num_shifts
                A[i, 0] = np.cos(phase)
                A[i, 1] = np.sin(phase)
            
            # Calculate pseudo-inverse
            A_inv = np.linalg.pinv(A)
            
            # Decode phase for each pixel
            phase_map = np.zeros((height, width), dtype=np.float32)
            
            for y in range(height):
                for x in range(width):
                    if mask[y, x] == 0:
                        continue
                    
                    # Get intensity values for this pixel across all shifts
                    I = np.array([freq_patterns[i][y, x] for i in range(num_shifts)])
                    
                    # Solve for parameters (a * cos(phi) + b * sin(phi))
                    params = A_inv.dot(I)
                    a, b = params
                    
                    # Calculate phase
                    phase = np.arctan2(b, a)
                    
                    # Store phase
                    phase_map[y, x] = phase
            
            h_phases.append(phase_map)
        
        # Process vertical patterns
        for f_idx, freq in enumerate(frequencies):
            # Get patterns for this frequency
            freq_patterns = v_patterns[f_idx * num_shifts:(f_idx + 1) * num_shifts]
            
            # Calculate phase using least squares
            A = np.zeros((num_shifts, 2))
            for i in range(num_shifts):
                phase = 2 * np.pi * i / num_shifts
                A[i, 0] = np.cos(phase)
                A[i, 1] = np.sin(phase)
            
            # Calculate pseudo-inverse
            A_inv = np.linalg.pinv(A)
            
            # Decode phase for each pixel
            phase_map = np.zeros((height, width), dtype=np.float32)
            
            for y in range(height):
                for x in range(width):
                    if mask[y, x] == 0:
                        continue
                    
                    # Get intensity values for this pixel across all shifts
                    I = np.array([freq_patterns[i][y, x] for i in range(num_shifts)])
                    
                    # Solve for parameters (a * cos(phi) + b * sin(phi))
                    params = A_inv.dot(I)
                    a, b = params
                    
                    # Calculate phase
                    phase = np.arctan2(b, a)
                    
                    # Store phase
                    phase_map[y, x] = phase
            
            v_phases.append(phase_map)
        
        # Unwrap phases using multi-frequency information
        h_unwrapped = self._unwrap_phase_multi_freq(h_phases, frequencies, mask)
        v_unwrapped = self._unwrap_phase_multi_freq(v_phases, frequencies, mask)
        
        # Convert to projector coordinates
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    # Scale phase to projector coordinates
                    proj_x = (h_unwrapped[y, x] / (2 * np.pi)) * max(frequencies)
                    proj_y = (v_unwrapped[y, x] / (2 * np.pi)) * max(frequencies)
                    
                    # Store if within bounds
                    if 0 <= proj_x < self.projector_width and 0 <= proj_y < self.projector_height:
                        cam_proj[y, x, 1] = proj_x  # Column (U)
                        cam_proj[y, x, 0] = proj_y  # Row (V)
        
        return cam_proj, mask
    
    def reconstruct_point_cloud(self, camera_img: np.ndarray, 
                               cam_proj: np.ndarray, 
                               mask: np.ndarray) -> o3dg.PointCloud:
        """
        Reconstruct 3D point cloud from camera-projector correspondence.
        
        Args:
            camera_img: Camera image (for color information)
            cam_proj: Camera-projector correspondence map
            mask: Validity mask
            
        Returns:
            3D point cloud
        """
        if cam_proj is None or mask is None:
            logger.error("Invalid input data")
            return o3dg.PointCloud()
        
        # Get image dimensions
        height, width = mask.shape
        
        # Calculate 3D points using triangulation
        points = []
        colors = []
        
        # Get camera color image if available
        has_color = len(camera_img.shape) == 3 and camera_img.shape[2] == 3
        
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                
                # Get projector coordinates
                proj_y = cam_proj[y, x, 0]
                proj_x = cam_proj[y, x, 1]
                
                if proj_x <= 0 or proj_y <= 0:
                    continue
                
                # Create camera ray
                cam_point = np.array([x, y, 1.0])
                cam_ray = np.linalg.inv(self.camera_matrix).dot(cam_point)
                
                # Create projector ray
                proj_point = np.array([proj_x, proj_y, 1.0])
                proj_ray = np.linalg.inv(self.projector_matrix).dot(proj_point)
                
                # Convert projector ray to camera coordinate system
                proj_ray_cam = self.R.T.dot(proj_ray)
                
                # Create ray origins
                cam_origin = np.array([0, 0, 0])
                proj_origin_cam = -self.R.T.dot(self.T).flatten()
                
                # Calculate the 3D point using closest point between rays (triangulation)
                # This is a simplified method - more robust methods exist
                
                # Direction vectors
                v1 = cam_ray
                v2 = proj_ray_cam
                
                # Origins
                o1 = cam_origin
                o2 = proj_origin_cam
                
                # Compute closest point between the two rays
                n = np.cross(v1, v2)
                n1 = np.cross(v1, n)
                n2 = np.cross(v2, n)
                
                c1 = o1 + v1 * np.dot(o2 - o1, n2) / np.dot(v1, n2)
                c2 = o2 + v2 * np.dot(o1 - o2, n1) / np.dot(v2, n1)
                
                # 3D point is the average of the closest points on each ray
                point_3d = (c1 + c2) / 2
                
                # Check if point is in front of camera
                if point_3d[2] <= 0:
                    continue
                
                # Add point
                points.append(point_3d)
                
                # Add color if available
                if has_color:
                    colors.append(camera_img[y, x] / 255.0)
                else:
                    # Use a default color (gray)
                    colors.append([0.7, 0.7, 0.7])
        
        # Create Open3D point cloud
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, returning numpy array of points")
            return np.array(points)
        
        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Reconstructed point cloud with {len(pcd.points)} points")
        return pcd
    
    def process_scan(self, captured_images: List[np.ndarray], 
                    use_gray_code: bool = True, 
                    use_phase_shift: bool = False,
                    mask_threshold: int = 10,
                    output_dir: Optional[str] = None) -> o3dg.PointCloud:
        """
        Process scan data to generate point cloud.
        
        Args:
            captured_images: List of captured camera images
            use_gray_code: Whether to use Gray code for decoding
            use_phase_shift: Whether to use phase shift for decoding
            mask_threshold: Threshold for shadow mask
            output_dir: Optional directory to save debug images
            
        Returns:
            Point cloud
        """
        if not captured_images:
            logger.error("No images provided")
            return o3dg.PointCloud()
        
        # Create debug directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
        else:
            debug_dir = None
        
        # Decode structured light patterns
        if use_gray_code:
            logger.info("Decoding Gray code patterns")
            cam_proj, mask = self.decode_gray_code(captured_images, mask_threshold)
            
            # Save debug images
            if debug_dir:
                self._save_debug_images(captured_images, cam_proj, mask, debug_dir)
            
            # If phase shift is enabled, use it for refinement
            if use_phase_shift:
                # Ideally, you would have both gray code and phase shift patterns
                # but here we'll simulate this by using the same images
                logger.info("Refining with phase shift patterns")
                phase_cam_proj, phase_mask = self.decode_phase_shift(captured_images, threshold=mask_threshold)
                
                # Refine correspondence
                if phase_cam_proj is not None and phase_mask is not None:
                    cam_proj = self._refine_correspondence(cam_proj, phase_cam_proj, mask)
        else:
            # Use phase shift decoding only
            logger.info("Decoding phase shift patterns")
            cam_proj, mask = self.decode_phase_shift(captured_images, threshold=mask_threshold)
            
            # Save debug images
            if debug_dir:
                self._save_debug_images(captured_images, cam_proj, mask, debug_dir)
        
        if cam_proj is None or mask is None:
            logger.error("Pattern decoding failed")
            return o3dg.PointCloud()
        
        # Reconstruct point cloud
        logger.info("Reconstructing point cloud")
        pcd = self.reconstruct_point_cloud(captured_images[0], cam_proj, mask)
        
        # Filter outliers
        if OPEN3D_AVAILABLE and len(pcd.points) > 0:
            logger.info("Filtering outliers")
            pcd = self._filter_point_cloud(pcd)
        
        return pcd
    
    def _compute_shadow_mask(self, black_img: np.ndarray, white_img: np.ndarray, 
                           threshold: int = 10) -> np.ndarray:
        """
        Compute shadow mask from white and black images.
        
        Args:
            black_img: Image captured with black pattern
            white_img: Image captured with white pattern
            threshold: Threshold for shadow detection
            
        Returns:
            Binary mask (1 for valid pixels, 0 for shadowed)
        """
        # Compute difference
        diff = cv2.absdiff(white_img, black_img)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Apply threshold
        _, mask = cv2.threshold(blurred, threshold, 1, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _gray_to_binary(self, n: int) -> int:
        """
        Convert Gray code to binary.
        
        Args:
            n: Gray code value
            
        Returns:
            Binary value
        """
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    def _unwrap_phase_multi_freq(self, wrapped_phases: List[np.ndarray], 
                               frequencies: List[int], 
                               mask: np.ndarray) -> np.ndarray:
        """
        Unwrap phase using multi-frequency approach.
        
        Args:
            wrapped_phases: List of wrapped phase maps
            frequencies: List of frequencies
            mask: Validity mask
            
        Returns:
            Unwrapped phase map
        """
        if not wrapped_phases:
            return np.zeros_like(mask, dtype=np.float32)
        
        # Start with the highest frequency
        freq_idx = len(frequencies) - 1
        high_freq = frequencies[freq_idx]
        unwrapped = wrapped_phases[freq_idx].copy()
        
        # Unwrap using hierarchical approach (from high to low frequency)
        for i in range(freq_idx - 1, -1, -1):
            low_freq = frequencies[i]
            ratio = high_freq / low_freq
            
            # Calculate phase difference
            diff = unwrapped - wrapped_phases[i] * ratio
            
            # Adjust difference to be in [-π, π]
            diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
            
            # Update unwrapped phase
            unwrapped = wrapped_phases[i] * ratio + diff
        
        # Apply mask
        unwrapped = unwrapped * mask
        
        return unwrapped
    
    def _refine_correspondence(self, gray_correspondence: np.ndarray, 
                              phase_correspondence: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
        """
        Refine Gray code correspondence with phase shift.
        
        Args:
            gray_correspondence: Gray code correspondence map
            phase_correspondence: Phase shift correspondence map
            mask: Validity mask
            
        Returns:
            Refined correspondence map
        """
        refined = gray_correspondence.copy()
        
        # Combine only where both are valid
        valid_mask = (gray_correspondence[:,:,0] > 0) & (phase_correspondence[:,:,0] > 0) & (mask > 0)
        
        # Weighted average (gray code has higher weight for integer values)
        gray_weight = 0.7
        phase_weight = 0.3
        
        refined[valid_mask, 0] = (gray_correspondence[valid_mask, 0] * gray_weight + 
                                 phase_correspondence[valid_mask, 0] * phase_weight)
        refined[valid_mask, 1] = (gray_correspondence[valid_mask, 1] * gray_weight + 
                                 phase_correspondence[valid_mask, 1] * phase_weight)
        
        return refined
    
    def _filter_point_cloud(self, pcd: o3dg.PointCloud) -> o3dg.PointCloud:
        """
        Filter point cloud to remove outliers.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Filtered point cloud
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, skipping filtering")
            return pcd
        
        if len(pcd.points) == 0:
            return pcd
        
        try:
            # Statistical outlier removal
            nb_neighbors = min(30, len(pcd.points) // 2)
            filtered_pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=max(nb_neighbors, 2), 
                std_ratio=1.0
            )
            
            # Radius outlier removal
            filtered_pcd, _ = filtered_pcd.remove_radius_outlier(
                nb_points=16, 
                radius=2.0
            )
            
            logger.info(f"Filtered point cloud: {len(pcd.points)} → {len(filtered_pcd.points)} points")
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"Error filtering point cloud: {e}")
            return pcd
    
    def _save_debug_images(self, captured_images: List[np.ndarray], 
                          cam_proj: np.ndarray, 
                          mask: np.ndarray, 
                          debug_dir: str) -> None:
        """
        Save debug images.
        
        Args:
            captured_images: List of captured images
            cam_proj: Camera-projector correspondence map
            mask: Validity mask
            debug_dir: Output directory
        """
        if not debug_dir:
            return
        
        try:
            # Save white and black images
            if len(captured_images) >= 2:
                cv2.imwrite(os.path.join(debug_dir, "white.png"), captured_images[0])
                cv2.imwrite(os.path.join(debug_dir, "black.png"), captured_images[1])
            
            # Save mask
            cv2.imwrite(os.path.join(debug_dir, "mask.png"), mask * 255)
            
            # Create and save a visualization of the correspondence
            if cam_proj is not None and mask is not None:
                height, width = mask.shape
                
                # Visualize projector coordinates
                proj_vis = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    for x in range(width):
                        if mask[y, x] > 0:
                            # Scale projector coordinates to 0-255 for visualization
                            r = int(255 * cam_proj[y, x, 1] / self.projector_width)
                            g = int(255 * cam_proj[y, x, 0] / self.projector_height)
                            b = 128
                            proj_vis[y, x] = [b, g, r]
                
                cv2.imwrite(os.path.join(debug_dir, "proj_coords_vis.png"), proj_vis)
                
        except Exception as e:
            logger.warning(f"Error saving debug images: {e}")
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """
        Encode image as JPEG.
        
        Args:
            image: Image to encode
            
        Returns:
            JPEG encoded image as bytes
        """
        success, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return data.tobytes() if success else b''
    
    def create_mesh(self, pcd: o3dg.PointCloud, 
                   depth: int = 8, 
                   smoothing: int = 2) -> o3dg.TriangleMesh:
        """
        Create mesh from point cloud.
        
        Args:
            pcd: Input point cloud
            depth: Depth parameter for Poisson reconstruction
            smoothing: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, cannot create mesh")
            return o3dg.TriangleMesh()
        
        if len(pcd.points) < 100:
            logger.warning(f"Too few points ({len(pcd.points)}) to create a mesh")
            return o3dg.TriangleMesh()
        
        try:
            # Ensure normals are available
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(k=15)
            
            # Create mesh using Poisson reconstruction
            logger.info(f"Creating mesh with Poisson reconstruction (depth={depth})")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=0, scale=1.1, linear_fit=True
            )
            
            # Remove low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Apply smoothing
            if smoothing > 0:
                logger.info(f"Smoothing mesh with {smoothing} iterations")
                mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing)
            
            logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            return o3dg.TriangleMesh()
    
    def save_point_cloud(self, pcd: Union[o3dg.PointCloud, np.ndarray], filepath: str) -> None:
        """
        Save point cloud to file.
        
        Args:
            pcd: Point cloud to save
            filepath: Output filepath
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        if not OPEN3D_AVAILABLE:
            # Save as numpy if open3d not available
            if isinstance(pcd, np.ndarray):
                np.save(filepath, pcd)
                logger.info(f"Saved point cloud as NumPy array: {filepath}")
            else:
                logger.error("Cannot save point cloud: open3d not available and input is not numpy array")
            return
        
        # Convert numpy array to point cloud if needed
        if isinstance(pcd, np.ndarray):
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
            pcd = o3d_pcd
        
        # Get file extension
        ext = os.path.splitext(filepath)[1].lower()
        
        # Save based on extension
        if ext in ['.ply', '.pcd', '.xyz', '.pts']:
            o3d.io.write_point_cloud(filepath, pcd)
        else:
            # Default to PLY
            new_path = os.path.splitext(filepath)[0] + '.ply'
            o3d.io.write_point_cloud(new_path, pcd)
            logger.info(f"Changed extension to .ply: {new_path}")
        
        logger.info(f"Saved point cloud with {len(pcd.points)} points to {filepath}")
    
    def save_mesh(self, mesh: o3dg.TriangleMesh, filepath: str) -> None:
        """
        Save mesh to file.
        
        Args:
            mesh: Mesh to save
            filepath: Output filepath
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Cannot save mesh: open3d not available")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Get file extension
        ext = os.path.splitext(filepath)[1].lower()
        
        # Save based on extension
        if ext in ['.ply', '.obj', '.off', '.stl', '.gltf']:
            o3d.io.write_triangle_mesh(filepath, mesh)
        else:
            # Default to OBJ
            new_path = os.path.splitext(filepath)[0] + '.obj'
            o3d.io.write_triangle_mesh(new_path, mesh)
            logger.info(f"Changed extension to .obj: {new_path}")
        
        logger.info(f"Saved mesh with {len(mesh.triangles)} triangles to {filepath}")