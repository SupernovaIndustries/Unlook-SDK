"""
Neural Network-based Point Cloud Enhancement Module.

This module provides neural network-based enhancement for point clouds,
including noise reduction, outlier removal, feature enhancement,
upsampling, and meshing with real-time scanning optimizations.

It integrates with Open3D and its ML modules when available, falling back
to PyTorch-based implementations when needed, with a focus on real-time
performance for structured light scanning applications.
"""

import os
import logging
import numpy as np
import time
from typing import Union, Optional, Tuple, Dict, Any, List

# Configure logger
logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    TORCH_CUDA = torch.cuda.is_available()
    if TORCH_CUDA:
        logger.info(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("PyTorch available but CUDA not detected")
except ImportError:
    logger.warning("PyTorch not installed. Neural network enhancement will be disabled.")
    TORCH_AVAILABLE = False
    TORCH_CUDA = False

# Try to import Open3D and Open3D-ML
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    
    # Try importing Open3D-ML
    try:
        import open3d.ml as ml3d
        if hasattr(ml3d, 'torch'):
            OPEN3D_ML_TORCH = True
            logger.info("Open3D-ML with PyTorch backend available")
        elif hasattr(ml3d, 'tf'):
            OPEN3D_ML_TF = True
            logger.info("Open3D-ML with TensorFlow backend available")
        else:
            OPEN3D_ML_TORCH = False
            OPEN3D_ML_TF = False
            logger.warning("Open3D-ML found but no ML backend detected")
        OPEN3D_ML_AVAILABLE = True
    except ImportError:
        logger.warning("Open3D-ML not installed. Using basic models only.")
        OPEN3D_ML_AVAILABLE = False
        OPEN3D_ML_TORCH = False
        OPEN3D_ML_TF = False
except ImportError:
    logger.warning("Open3D not installed. Neural network enhancement will be limited.")
    OPEN3D_AVAILABLE = False
    OPEN3D_ML_AVAILABLE = False
    OPEN3D_ML_TORCH = False
    OPEN3D_ML_TF = False


class PointNetFeatureExtractor(nn.Module):
    """
    PointNet-based feature extractor for point clouds.
    
    This network extracts features from point clouds which can be used
    for classification, segmentation, or enhancement tasks.
    
    Based on the paper: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
    by Qi et al.
    """
    
    def __init__(self, input_channels=3, feature_dim=64):
        """
        Initialize the PointNet feature extractor.
        
        Args:
            input_channels: Number of input channels (typically 3 for XYZ)
            feature_dim: Dimension of output features
        """
        super(PointNetFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x):
        """Forward pass through network."""
        # x shape: B x N x 3 (batch, num_points, channels)
        
        # Convert to B x 3 x N for Conv1d
        x = x.transpose(2, 1)
        
        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling across points
        x, _ = torch.max(x, dim=2, keepdim=True)
        
        return x


class PointCloudDenoiser(nn.Module):
    """
    Neural network for point cloud denoising.
    
    This network implements a simple yet effective approach for
    removing noise from 3D point clouds while preserving details.
    """
    
    def __init__(self, feature_dim=64):
        """
        Initialize the denoiser network.
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        super(PointCloudDenoiser, self).__init__()
        
        # Feature extractor
        self.feature_extractor = PointNetFeatureExtractor(input_channels=3, feature_dim=feature_dim)
        
        # Point-wise MLP for denoising
        self.mlp1 = nn.Conv1d(feature_dim + 3, 128, 1)
        self.mlp2 = nn.Conv1d(128, 64, 1)
        self.mlp3 = nn.Conv1d(64, 3, 1)  # Output XYZ offset
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        """Forward pass to denoise point cloud."""
        # x shape: B x N x 3
        batch_size, num_points, _ = x.shape
        
        # Extract global features
        global_feat = self.feature_extractor(x)  # B x F x 1
        
        # Expand and concatenate with input points
        global_feat = global_feat.expand(-1, -1, num_points)  # B x F x N
        
        # Prepare input points for concatenation
        point_feat = x.transpose(2, 1)  # B x 3 x N
        
        # Concatenate global features with point coordinates
        features = torch.cat([global_feat, point_feat], dim=1)  # B x (F+3) x N
        
        # Apply MLP
        features = F.relu(self.bn1(self.mlp1(features)))
        features = F.relu(self.bn2(self.mlp2(features)))
        offsets = self.mlp3(features)  # B x 3 x N
        
        # Apply offsets to input points
        denoised = point_feat + offsets
        
        # Reshape to B x N x 3
        denoised = denoised.transpose(2, 1)
        
        return denoised


class PointCloudUpsampler(nn.Module):
    """
    Neural network for point cloud upsampling.
    
    This model increases the density of a point cloud by generating new points
    that preserve the geometry of the underlying surface.
    """
    
    def __init__(self, up_ratio=4, feature_dim=128):
        """
        Initialize the upsampler network.
        
        Args:
            up_ratio: Upsampling ratio (number of new points per original point)
            feature_dim: Dimension of feature vectors
        """
        super(PointCloudUpsampler, self).__init__()
        
        self.up_ratio = up_ratio
        
        # Feature extractor
        self.feature_extractor = PointNetFeatureExtractor(input_channels=3, feature_dim=feature_dim)
        
        # Upsampling MLP
        self.mlp1 = nn.Conv1d(feature_dim + 3, 256, 1)
        self.mlp2 = nn.Conv1d(256, 128, 1)
        self.mlp3 = nn.Conv1d(128, 3 * up_ratio, 1)  # Output XYZ for each new point
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        """Forward pass to upsample point cloud."""
        # x shape: B x N x 3
        batch_size, num_points, _ = x.shape
        
        # Extract features
        global_feat = self.feature_extractor(x)  # B x F x 1
        
        # Expand and concatenate with input points
        global_feat = global_feat.expand(-1, -1, num_points)  # B x F x N
        
        # Prepare input points for concatenation
        point_feat = x.transpose(2, 1)  # B x 3 x N
        
        # Concatenate global features with point coordinates
        features = torch.cat([global_feat, point_feat], dim=1)  # B x (F+3) x N
        
        # Apply MLP
        features = F.relu(self.bn1(self.mlp1(features)))
        features = F.relu(self.bn2(self.mlp2(features)))
        new_points = self.mlp3(features)  # B x (3*up_ratio) x N
        
        # Reshape to create up_ratio new points for each input point
        new_points = new_points.view(batch_size, 3, self.up_ratio, num_points)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # B x N x up_ratio x 3
        new_points = new_points.view(batch_size, num_points * self.up_ratio, 3)
        
        return new_points


class PointCloudEnhancer:
    """
    Point cloud enhancement using neural networks.
    
    This class orchestrates different neural network models to enhance
    point clouds by denoising, upsampling, completing, or enhancing features.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        enable_open3d_ml: bool = True,
        model_dir: Optional[str] = None,
        denoiser_weights_path: Optional[str] = None,
        upsampler_weights_path: Optional[str] = None
    ):
        """
        Initialize the point cloud enhancer.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            enable_open3d_ml: Whether to use Open3D-ML models if available
            model_dir: Directory to store trained models
            denoiser_weights_path: Path to pre-trained denoiser weights
            upsampler_weights_path: Path to pre-trained upsampler weights
        """
        self.use_gpu = use_gpu and (TORCH_CUDA or OPEN3D_ML_TORCH or OPEN3D_ML_TF)
        self.model_dir = model_dir or os.path.join(os.path.expanduser("~"), ".unlook", "models")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize the models
        self._init_models(denoiser_weights_path, upsampler_weights_path)
        
        # Initialize Open3D-ML if available
        if enable_open3d_ml and OPEN3D_ML_AVAILABLE:
            self._init_open3d_ml()
        
        logger.info(f"Point Cloud Enhancer initialized (GPU: {self.use_gpu})")
    
    def _init_models(self, denoiser_weights_path, upsampler_weights_path):
        """Initialize neural network models."""
        self.denoiser = None
        self.upsampler = None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, models will not be initialized")
            return
        
        # Initialize denoiser
        try:
            self.denoiser = PointCloudDenoiser(feature_dim=64)
            
            # Load weights if provided
            if denoiser_weights_path and os.path.exists(denoiser_weights_path):
                self.denoiser.load_state_dict(torch.load(denoiser_weights_path))
                logger.info(f"Loaded denoiser weights from {denoiser_weights_path}")
            
            # Move to GPU if available
            if self.use_gpu and TORCH_CUDA:
                self.denoiser = self.denoiser.cuda()
            
            self.denoiser.eval()
            logger.info("Point cloud denoiser initialized")
        except Exception as e:
            logger.error(f"Failed to initialize denoiser: {e}")
            self.denoiser = None
        
        # Initialize upsampler
        try:
            self.upsampler = PointCloudUpsampler(up_ratio=4, feature_dim=128)
            
            # Load weights if provided
            if upsampler_weights_path and os.path.exists(upsampler_weights_path):
                self.upsampler.load_state_dict(torch.load(upsampler_weights_path))
                logger.info(f"Loaded upsampler weights from {upsampler_weights_path}")
            
            # Move to GPU if available
            if self.use_gpu and TORCH_CUDA:
                self.upsampler = self.upsampler.cuda()
            
            self.upsampler.eval()
            logger.info("Point cloud upsampler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize upsampler: {e}")
            self.upsampler = None
    
    def _init_open3d_ml(self):
        """Initialize Open3D-ML models if available."""
        if not OPEN3D_ML_AVAILABLE:
            return
        
        try:
            # Use Open3D-ML models based on available backend
            if OPEN3D_ML_TORCH:
                logger.info("Initializing Open3D-ML with PyTorch backend")
                
                # In a full implementation, you would load KPConv or other Open3D-ML models here
                # For example:
                # pipeline = ml3d.pipelines.SemanticSegmentation(model=ml3d.models.KPCONV())
                
            elif OPEN3D_ML_TF:
                logger.info("Initializing Open3D-ML with TensorFlow backend")
                
                # TensorFlow implementation would go here
            
            logger.info("Open3D-ML models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Open3D-ML: {e}")
    
    def denoise(
        self,
        point_cloud: Union[np.ndarray, "o3d.geometry.PointCloud"],
        strength: float = 0.5
    ) -> Union[np.ndarray, "o3d.geometry.PointCloud"]:
        """
        Denoise a point cloud using neural network.
        
        Args:
            point_cloud: Input point cloud (numpy array or Open3D point cloud)
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Denoised point cloud in the same format as input
        """
        # If denoiser isn't available, return the original point cloud
        if not self.denoiser or not TORCH_AVAILABLE:
            logger.warning("Denoiser not available, returning original point cloud")
            return point_cloud
        
        # Convert to numpy array if needed
        is_o3d = isinstance(point_cloud, o3d.geometry.PointCloud) if OPEN3D_AVAILABLE else False
        if is_o3d:
            points = np.asarray(point_cloud.points)
        else:
            points = point_cloud
        
        # Skip if empty
        if len(points) == 0:
            return point_cloud
        
        # Denoise using PyTorch model
        try:
            # Convert to tensor
            with torch.no_grad():
                # Add batch dimension
                points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
                
                # Move to GPU if available
                if self.use_gpu and TORCH_CUDA:
                    points_tensor = points_tensor.cuda()
                
                # Apply denoising
                denoised_tensor = self.denoiser(points_tensor)
                
                # Move back to CPU if needed
                if self.use_gpu and TORCH_CUDA:
                    denoised_tensor = denoised_tensor.cpu()
                
                # Remove batch dimension
                denoised_points = denoised_tensor.squeeze(0).numpy()
                
                # Apply denoising strength
                if strength < 1.0:
                    denoised_points = points + strength * (denoised_points - points)
            
            # Return in the same format
            if is_o3d:
                result = o3d.geometry.PointCloud()
                result.points = o3d.utility.Vector3dVector(denoised_points)
                
                # Preserve colors if they exist
                if point_cloud.has_colors():
                    result.colors = point_cloud.colors
                
                return result
            else:
                return denoised_points
        
        except Exception as e:
            logger.error(f"Error during denoising: {e}")
            return point_cloud
    
    def upsample(
        self,
        point_cloud: Union[np.ndarray, "o3d.geometry.PointCloud"],
        target_points: Optional[int] = None
    ) -> Union[np.ndarray, "o3d.geometry.PointCloud"]:
        """
        Upsample a point cloud using neural network.
        
        Args:
            point_cloud: Input point cloud (numpy array or Open3D point cloud)
            target_points: Target number of points (if None, uses upsampler's default ratio)
            
        Returns:
            Upsampled point cloud in the same format as input
        """
        # If upsampler isn't available, return the original point cloud
        if not self.upsampler or not TORCH_AVAILABLE:
            logger.warning("Upsampler not available, returning original point cloud")
            return point_cloud
        
        # Convert to numpy array if needed
        is_o3d = isinstance(point_cloud, o3d.geometry.PointCloud) if OPEN3D_AVAILABLE else False
        if is_o3d:
            points = np.asarray(point_cloud.points)
            has_colors = point_cloud.has_colors()
            if has_colors:
                colors = np.asarray(point_cloud.colors)
        else:
            points = point_cloud
            has_colors = False
        
        # Skip if empty
        if len(points) == 0:
            return point_cloud
        
        # Upsample using PyTorch model
        try:
            # Convert to tensor
            with torch.no_grad():
                # Add batch dimension
                points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
                
                # Move to GPU if available
                if self.use_gpu and TORCH_CUDA:
                    points_tensor = points_tensor.cuda()
                
                # Apply upsampling
                upsampled_tensor = self.upsampler(points_tensor)
                
                # Move back to CPU if needed
                if self.use_gpu and TORCH_CUDA:
                    upsampled_tensor = upsampled_tensor.cpu()
                
                # Remove batch dimension
                upsampled_points = upsampled_tensor.squeeze(0).numpy()
                
                # If target_points is specified, randomly sample to match
                if target_points is not None and len(upsampled_points) > target_points:
                    indices = np.random.choice(len(upsampled_points), target_points, replace=False)
                    upsampled_points = upsampled_points[indices]
            
            # Return in the same format
            if is_o3d:
                result = o3d.geometry.PointCloud()
                result.points = o3d.utility.Vector3dVector(upsampled_points)
                
                # Interpolate colors if they exist
                if has_colors:
                    # This is a simple nearest-neighbor color assignment
                    # In a real implementation, you would use a more sophisticated interpolation
                    if len(upsampled_points) > len(points):
                        # Find nearest neighbor for each new point
                        from scipy.spatial import cKDTree
                        tree = cKDTree(points)
                        _, indices = tree.query(upsampled_points, k=1)
                        upsampled_colors = colors[indices]
                        
                        result.colors = o3d.utility.Vector3dVector(upsampled_colors)
                
                return result
            else:
                return upsampled_points
        
        except Exception as e:
            logger.error(f"Error during upsampling: {e}")
            return point_cloud
    
    def enhance(
        self,
        point_cloud: Union[np.ndarray, "o3d.geometry.PointCloud"],
        denoise_strength: float = 0.5,
        upsample: bool = False,
        target_points: Optional[int] = None,
        outlier_removal: bool = True,
        normal_estimation: bool = False,
        meshing: bool = False,
        voxel_size: float = 0.01
    ) -> Union[np.ndarray, "o3d.geometry.PointCloud", "o3d.geometry.TriangleMesh"]:
        """
        Enhance a point cloud with multiple operations.

        This method applies a sequence of enhancement operations:
        1. Denoising
        2. Outlier removal (optional)
        3. Upsampling (optional)
        4. Normal estimation (optional)
        5. Meshing (optional)

        Args:
            point_cloud: Input point cloud
            denoise_strength: Denoising strength (0.0 to 1.0)
            upsample: Whether to upsample the point cloud
            target_points: Target number of points after upsampling
            outlier_removal: Whether to remove outlier points
            normal_estimation: Whether to estimate normals
            meshing: Whether to create a mesh (requires normal_estimation)
            voxel_size: Voxel size for processing (smaller = more detail)

        Returns:
            Enhanced point cloud or mesh in the same format as input
        """
        # Skip processing if input is invalid
        if point_cloud is None:
            logger.warning("Input point cloud is None")
            return point_cloud

        # Check if we should use Open3D processing for advanced operations
        use_open3d_processing = OPEN3D_AVAILABLE and (outlier_removal or normal_estimation or meshing)

        # Start timer
        start_time = time.time()

        if use_open3d_processing:
            try:
                # Use integrated Open3D processing pipeline
                result = process_point_cloud_with_open3d(
                    point_cloud=point_cloud,
                    voxel_size=voxel_size,
                    denoise=denoise_strength > 0,
                    outlier_removal=outlier_removal,
                    normal_estimation=normal_estimation,
                    meshing=meshing,
                    num_neighbors=20,
                    std_ratio=2.0
                )

                # Apply additional processing if needed
                if upsample and not meshing and target_points is not None:
                    # Don't upsample if we already created a mesh
                    result = self.upsample(result, target_points=target_points)

                # Log processing time
                elapsed = time.time() - start_time
                logger.info(f"Point cloud enhancement with Open3D completed in {elapsed:.3f} seconds")

                return result

            except Exception as e:
                logger.error(f"Open3D processing failed: {e}, falling back to basic enhancement")
                # Fall back to basic enhancement

        # Basic enhancement pipeline
        try:
            # Apply denoising
            if denoise_strength > 0:
                point_cloud = self.denoise(point_cloud, strength=denoise_strength)

            # Apply upsampling if requested
            if upsample and target_points is not None:
                point_cloud = self.upsample(point_cloud, target_points=target_points)

            # Log processing time
            elapsed = time.time() - start_time
            logger.info(f"Point cloud enhancement completed in {elapsed:.3f} seconds")

            return point_cloud

        except Exception as e:
            logger.error(f"Point cloud enhancement failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return point_cloud


# Open3D integration helper functions
def process_point_cloud_with_open3d(
    point_cloud: Union[np.ndarray, "o3d.geometry.PointCloud"],
    voxel_size: float = 0.01,
    denoise: bool = True,
    outlier_removal: bool = True,
    normal_estimation: bool = True,
    meshing: bool = False,
    num_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Union[np.ndarray, "o3d.geometry.PointCloud", "o3d.geometry.TriangleMesh"]:
    """
    Process point cloud with Open3D algorithms for better quality.

    Args:
        point_cloud: Input point cloud
        voxel_size: Voxel size for downsampling (smaller = more detail but slower)
        denoise: Apply denoising
        outlier_removal: Remove outliers
        normal_estimation: Estimate normals
        meshing: Generate mesh (requires normal estimation)
        num_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio for outlier removal

    Returns:
        Processed point cloud or mesh
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, returning original point cloud")
        return point_cloud

    try:
        # Convert to Open3D point cloud if needed
        is_o3d = isinstance(point_cloud, o3d.geometry.PointCloud)
        if not is_o3d:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            o3d_cloud = point_cloud

        # Skip processing if empty
        if len(o3d_cloud.points) == 0:
            return point_cloud

        # Make a copy to avoid modifying the original
        processed_cloud = o3d_cloud.clone()

        # Down-sample to make processing faster and more robust
        processed_cloud = processed_cloud.voxel_down_sample(voxel_size)

        # Apply statistical outlier removal if requested
        if outlier_removal and len(processed_cloud.points) > 50:
            try:
                processed_cloud, _ = processed_cloud.remove_statistical_outlier(
                    nb_neighbors=num_neighbors,
                    std_ratio=std_ratio
                )
                logger.debug(f"Outlier removal: {len(o3d_cloud.points)} -> {len(processed_cloud.points)} points")
            except Exception as e:
                logger.warning(f"Outlier removal failed: {e}")

        # Apply bilateral filtering if denoising is requested
        if denoise and len(processed_cloud.points) > 50:
            try:
                # First compute normals if not done yet and not explicitly disabled
                if not processed_cloud.has_normals() and normal_estimation:
                    processed_cloud.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=voxel_size*2, max_nn=30))

                # Apply smoothing - using a small custom implementation since Open3D lacks built-in point denoising
                points = np.asarray(processed_cloud.points)
                normals = np.asarray(processed_cloud.normals) if processed_cloud.has_normals() else None

                if normals is not None:
                    # Project small movements along normal direction (simple bilateral filter approximation)
                    tree = o3d.geometry.KDTreeFlann(processed_cloud)
                    smoothed_points = points.copy()

                    # Only process a random subset of points for speed in real-time applications
                    sample_size = min(5000, len(points))
                    indices = np.random.choice(len(points), sample_size, replace=False)

                    for idx in indices:
                        # Find neighbors
                        _, neighbor_indices, _ = tree.search_knn_vector_3d(points[idx], 10)
                        neighbor_points = points[neighbor_indices]

                        # Compute weighted average based on distance and normal similarity
                        weights = np.exp(-np.linalg.norm(neighbor_points - points[idx], axis=1) / (voxel_size*2)**2)
                        avg_point = np.average(neighbor_points, axis=0, weights=weights)

                        # Project back along normal direction to preserve surface
                        normal = normals[idx]
                        diff = avg_point - points[idx]
                        proj_dist = np.dot(diff, normal)
                        smoothed_points[idx] = points[idx] + 0.5 * (diff - proj_dist * normal)

                    # Update points
                    processed_cloud.points = o3d.utility.Vector3dVector(smoothed_points)
                    logger.debug("Applied bilateral smoothing to point cloud")
            except Exception as e:
                logger.warning(f"Denoising failed: {e}")

        # Estimate normals if requested
        if normal_estimation and not processed_cloud.has_normals():
            try:
                processed_cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size*2, max_nn=30))
                processed_cloud.orient_normals_consistent_tangent_plane(k=15)
                logger.debug("Estimated point cloud normals")
            except Exception as e:
                logger.warning(f"Normal estimation failed: {e}")

        # Create mesh if requested
        if meshing and processed_cloud.has_normals() and len(processed_cloud.points) > 100:
            try:
                # First try Poisson reconstruction
                try:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        processed_cloud, depth=8, width=0, scale=1.1, linear_fit=True)

                    # Remove low-density vertices
                    vertices_to_remove = densities < np.quantile(densities, 0.1)
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                    logger.debug(f"Created mesh with {len(mesh.triangles)} triangles using Poisson reconstruction")

                    return mesh
                except Exception as e:
                    logger.warning(f"Poisson reconstruction failed: {e}, trying Ball Pivoting")

                    # Fall back to Ball Pivoting
                    radii = [voxel_size*2, voxel_size*4, voxel_size*8]
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        processed_cloud, o3d.utility.DoubleVector(radii))

                    if len(mesh.triangles) > 0:
                        logger.debug(f"Created mesh with {len(mesh.triangles)} triangles using Ball Pivoting")
                        return mesh
                    else:
                        logger.warning("Ball Pivoting created an empty mesh, returning point cloud")
            except Exception as e:
                logger.warning(f"Meshing failed: {e}")

        # Return in the same format as input
        if not is_o3d:
            return np.asarray(processed_cloud.points)
        else:
            return processed_cloud

    except Exception as e:
        logger.error(f"Error during Open3D processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return point_cloud


# Function to get a preconfigured enhancer
def get_point_cloud_enhancer(
    use_gpu: bool = True,
    model_dir: Optional[str] = None
) -> PointCloudEnhancer:
    """
    Get a preconfigured point cloud enhancer.

    Args:
        use_gpu: Whether to use GPU acceleration if available
        model_dir: Directory for model storage

    Returns:
        Configured PointCloudEnhancer instance
    """
    enhancer = PointCloudEnhancer(
        use_gpu=use_gpu,
        enable_open3d_ml=True,
        model_dir=model_dir
    )

    return enhancer