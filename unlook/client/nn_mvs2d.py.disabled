"""
MVS2D Neural Network Integration for Unlook SDK

This module integrates the MVS2D (Multi-view Stereo via Attention-Driven 2D Convolutions)
neural network model for enhanced point cloud processing in the Unlook SDK.

Reference: 
    - MVS2D: Efficient Multi-view Stereo via Attention-Driven 2D Convolutions
    - Paper: https://arxiv.org/abs/2104.13325
    - Original implementation: https://github.com/zhenpeiyang/MVS2D
"""

import os
import logging
import numpy as np
import time
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple, Any, Union

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available for MVS2D neural network processing")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. MVS2D neural network processing disabled.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    logger.info("Open3D is available for point cloud processing")
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not found. Some point cloud operations will be limited.")

# Import GPU utilities if available
try:
    from .gpu_utils import is_gpu_available
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    logger.warning("GPU utilities not available. Using CPU for neural network processing.")

# Constants
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/mvs2d")

class MVS2DConfig:
    """Configuration for MVS2D neural network."""
    
    def __init__(self):
        # Basic network configuration
        self.use_gpu = True
        self.model_path = DEFAULT_MODEL_PATH
        self.model_type = "scannet"  # Options: "scannet", "demon", "dtu"
        
        # Input/output configuration
        self.min_depth = 0.1  # Minimum depth in meters
        self.max_depth = 10.0  # Maximum depth in meters
        self.width = 640  # Input image width
        self.height = 480  # Input image height
        
        # Network parameters
        self.nlabel = 64  # Number of depth labels
        self.nhead = 8  # Number of attention heads
        self.att_rate = 4  # Attention dimension reduction rate
        self.depth_embedding = "learned"  # Type of depth embedding
        self.num_frame = 3  # Number of input frames
        self.multi_view_agg = True  # Enable multi-view aggregation
        self.robust = False  # Enable robust multi-scale attention
        self.output_scale = 2  # Output scale (1=1/4, 2=1/8, 3=1/16, 4=1/32)
        self.input_scale = 1  # Input scale
        self.num_depth_regressor_anchor = 80  # Number of depth regressor anchors
        self.pred_conf = True  # Predict confidence
        self.use_skip = True  # Use skip connections
        self.use_unet = True  # Use UNet decoder
        self.unet_channel_mode = "same"  # UNet channel mode
        self.inv_depth = True  # Use inverse depth representation
        
        # Processing parameters
        self.batch_size = 1  # Batch size for inference
        self.confidence_threshold = 0.5  # Threshold for confidence filtering
        

class ConvBlock(nn.Module):
    """Basic convolutional block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ConvBlock, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CosineEmbedding(nn.Module):
    """Cosine positional embedding."""
    
    def __init__(self, num_positions, dim):
        super(CosineEmbedding, self).__init__()
        self.num_positions = num_positions
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(num_positions, dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        position = torch.arange(0, self.num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2) * (-np.log(10000.0) / self.dim))
        self.weight[:, 0::2] = torch.sin(position * div_term)
        self.weight[:, 1::2] = torch.cos(position * div_term)


def compute_depth_expectation(probs, depth_values):
    """Compute depth expectation from probability distribution."""
    depth = torch.sum(probs * depth_values, dim=1)
    return depth


def homo_warping(src_feature, src_proj, ref_proj, depth_values):
    """
    Homography warping of source features to reference view.
    
    Args:
        src_feature: Source view features
        src_proj: Source view projection matrix
        ref_proj: Reference view projection matrix
        depth_values: Depth values for warping
        
    Returns:
        Warped features, projection mask, and grid coordinates
    """
    batch, channels, height, width = src_feature.shape
    num_depth = depth_values.shape[1]
    device = src_feature.device
    
    # Compute pixel coordinates in the reference view
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                          torch.arange(0, width, dtype=torch.float32, device=device)])
    x, y = x.contiguous(), y.contiguous()
    x, y = x.view(height * width), y.view(height * width)
    
    # Homogeneous coordinates
    ones = torch.ones_like(x)
    homogeneous = torch.stack((x, y, ones), dim=0).unsqueeze(0)  # [1, 3, H*W]
    
    # Back-project to 3D points for each depth hypothesis
    ref_proj_inv = torch.inverse(ref_proj[:, :3, :3])  # [B, 3, 3]
    ref_points = (ref_proj_inv @ homogeneous).unsqueeze(2).repeat(1, 1, num_depth, 1)  # [B, 3, D, H*W]
    
    # Scale by depth values
    depth_values = depth_values.view(batch, 1, num_depth, height * width)
    ref_points = ref_points * depth_values
    
    # Add homogeneous coordinate for projection
    ones = torch.ones_like(ref_points[:, :1, :, :])
    ref_points_hom = torch.cat((ref_points, ones), dim=1)  # [B, 4, D, H*W]
    
    # Project to source view
    src_proj_new = src_proj.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)  # [B, 3, D, 4, 4]
    ref_points_hom = ref_points_hom.unsqueeze(1)  # [B, 1, 4, D, H*W]
    src_points = torch.matmul(src_proj_new[:, :, :, :3, :], ref_points_hom)  # [B, 3, D, H*W]
    
    # Convert to pixel coordinates
    src_xy = src_points[:, :2, :, :] / (src_points[:, 2:3, :, :] + 1e-6)  # [B, 2, D, H*W]
    src_xy = src_xy.view(batch, 2, num_depth, height, width)
    
    # Normalize to [-1, 1] for grid_sample
    src_x_norm = src_xy[:, 0, :, :, :] / ((width - 1) / 2) - 1  # [B, D, H, W]
    src_y_norm = src_xy[:, 1, :, :, :] / ((height - 1) / 2) - 1  # [B, D, H, W]
    src_grid = torch.stack((src_x_norm, src_y_norm), dim=-1)  # [B, D, H, W, 2]
    
    # Apply grid sample to warp the source feature
    warped_src_feature = F.grid_sample(src_feature, src_grid.view(batch, num_depth * height, width, 2),
                                      mode='bilinear', padding_mode='zeros', align_corners=True)
    warped_src_feature = warped_src_feature.view(batch, channels, num_depth, height, width)
    
    # Compute mask for valid projections
    mask = ((src_grid >= -1) & (src_grid <= 1)).all(dim=-1)  # [B, D, H, W]
    mask = mask.unsqueeze(1).float()  # [B, 1, D, H, W]
    
    return warped_src_feature.permute(0, 2, 1, 3, 4), mask, src_grid


class UNet(nn.Module):
    """
    U-Net model for depth feature processing.
    """
    def __init__(self, inp_ch, output_chal, down_sample_times=3, channel_mode="same"):
        super(UNet, self).__init__()
        self.down_sample_times = down_sample_times
        
        # Determine channel dimensions
        if channel_mode == "same":
            channels = [inp_ch] * (down_sample_times + 1)
        elif channel_mode == "double":
            channels = [inp_ch * (2 ** i) for i in range(down_sample_times + 1)]
        else:
            raise ValueError(f"Invalid channel mode: {channel_mode}")
        
        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        for i in range(down_sample_times):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ))
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        for i in range(down_sample_times):
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(channels[down_sample_times-i] * 2, channels[down_sample_times-i-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[down_sample_times-i-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[down_sample_times-i-1], channels[down_sample_times-i-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[down_sample_times-i-1]),
                nn.ReLU(inplace=True)
            ))
        
        # Final output layer
        self.final = nn.Conv2d(channels[0], output_chal, kernel_size=1)
        
    def forward(self, x):
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder path
        for i in range(self.down_sample_times):
            encoder_features.append(x)
            x = self.down_blocks[i](x)
            if i < self.down_sample_times - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Decoder path with skip connections
        for i in range(self.down_sample_times):
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, encoder_features[self.down_sample_times - 1 - i]], dim=1)
            x = self.up_blocks[i](x)
        
        return self.final(x)


class MVS2DModel(nn.Module):
    """
    MVS2D neural network model for multi-view stereo depth prediction.
    Simplified implementation based on the original paper.
    """
    def __init__(self, config):
        super(MVS2DModel, self).__init__()
        self.config = config
        self.output_layer = 0
        
        # Define attention layers
        if config.multi_view_agg:
            if config.robust:
                self.attn_layers = ['layer2', 'layer3', 'layer4']
            else:
                self.attn_layers = ['layer2']
        else:
            self.attn_layers = []
            
        # Initialize ResNet18 feature extractors
        self.base_model = models.resnet18(pretrained=True)
        self.base_model2 = models.resnet18(pretrained=True)
        delattr(self.base_model2, 'avgpool')
        delattr(self.base_model2, 'fc')
        
        # Remove unused layers from base_model
        cnt = 0
        to_delete = []
        for k, v in self.base_model._modules.items():
            if cnt == len(self.attn_layers):
                to_delete.append(k)
            if k in self.attn_layers:
                cnt += 1
        for k in to_delete:
            delattr(self.base_model, k)
        
        # ResNet-18 feature channels
        self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
        self.feat_name2ch = {
            'relu': 64,
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512
        }
        self.feat_channels = [self.feat_name2ch[x] for x in self.feat_names]
        
        # Initialize layers dictionary for modules
        self.layers = nn.ModuleDict()
        
        # Epipolar attention layers
        for k in self.attn_layers:
            inp_dim = self.feat_name2ch[k]
            att_dim = inp_dim // config.att_rate
            
            self.layers[f'query_{k}'] = nn.Conv2d(inp_dim, att_dim, kernel_size=1)
            self.layers[f'm_embed_{k}'] = nn.Embedding(2, att_dim)
            self.layers[f'key_{k}'] = nn.Conv2d(inp_dim, att_dim, kernel_size=1, bias=False)
            
            # Depth embedding options
            if config.depth_embedding == 'learned':
                self.layers[f'pos_embed_{k}'] = nn.Embedding(config.nlabel, att_dim)
            elif config.depth_embedding == 'cosine':
                self.layers[f'pos_embed_{k}'] = CosineEmbedding(config.nlabel, att_dim)
            else:
                raise NotImplementedError(f"Depth embedding '{config.depth_embedding}' not implemented")
            
            self.layers[f'linear_out_{k}'] = nn.Conv2d(att_dim, inp_dim, kernel_size=1)
            self.layers[f'linear_out2_{k}'] = nn.Conv2d(inp_dim, inp_dim, kernel_size=1)
            self.layers[f'linear_out3_{k}'] = nn.Conv2d(inp_dim, inp_dim, kernel_size=1)
        
        # Decoder layers
        self.num_ch_dec = [64, 64, 64, 128, 256]
        ch_cur = self.feat_channels[-1]
        for i in range(4, self.output_layer, -1):
            k = 1 if i == 4 else 3
            self.layers[f'upconv_{i}_0'] = ConvBlock(ch_cur, self.num_ch_dec[i], kernel_size=k)
            ch_mid = self.num_ch_dec[i]
            if config.use_skip:
                ch_mid += self.feat_channels[i - 1]
            self.layers[f'upconv_{i}_1'] = ConvBlock(ch_mid, self.num_ch_dec[i], kernel_size=k)
            ch_cur = self.num_ch_dec[i]
        
        # Depth regressor
        ch_cur = self.num_ch_dec[config.output_scale - config.input_scale - 1]
        odim = 256
        output_chal = odim if not config.pred_conf else odim + 1
        
        if config.use_unet:
            self.conv_out = UNet(inp_ch=ch_cur,
                                output_chal=output_chal,
                                down_sample_times=3,
                                channel_mode=config.unet_channel_mode)
        else:
            self.conv_out = nn.Conv2d(ch_cur, output_chal, kernel_size=1)
        
        self.depth_regressor = nn.Sequential(
            nn.Conv2d(odim, config.num_depth_regressor_anchor, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.num_depth_regressor_anchor, config.num_depth_regressor_anchor, kernel_size=1),
        )
        
        # Initialize depth hypothesis
        if config.inv_depth:
            depth_values = torch.from_numpy(
                1.0 / np.linspace(1.0 / config.max_depth, 1.0 / config.min_depth, config.nlabel)
            ).float()
        else:
            depth_values = torch.linspace(config.min_depth, config.max_depth, config.nlabel)
        
        self.register_buffer('depth_values', depth_values)
        self.register_buffer(
            'depth_expectation_anchor',
            torch.from_numpy(
                1.0 / np.linspace(1.0 / config.max_depth, 1.0 / config.min_depth,
                                config.num_depth_regressor_anchor)
            ).float()
        )
        
        # Up-sample layers
        h, w = config.height // 4, config.width // 4
        idv, idu = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        self.register_buffer('meshgrid', torch.from_numpy(np.stack((idu, idv))).float())
        
        self.conv_up = nn.Sequential(
            nn.Conv2d(1 + 2 + odim, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, padding=0),
        )
    
    def upsample(self, x, scale_factor=2):
        """Upsample input tensor by a factor"""
        return F.interpolate(x, scale_factor=scale_factor, mode="nearest")
    
    def epipolar_fusion(self, ref_feature, src_features, ref_proj, src_projs, depth_values, layer, ref_img, src_imgs):
        """
        Epipolar attention fusion mechanism.
        
        Args:
            ref_feature: Reference view feature map
            src_features: List of source view feature maps
            ref_proj: Reference view projection matrix
            src_projs: List of source view projection matrices
            depth_values: Depth hypotheses
            layer: Current layer name
            ref_img: Reference view image
            src_imgs: List of source view images
            
        Returns:
            Fused feature map
        """
        # Get layer-specific modules
        query = self.layers[f'query_{layer}']
        m_embed = self.layers[f'm_embed_{layer}']
        pos_embed = self.layers[f'pos_embed_{layer}']
        key = self.layers[f'key_{layer}']
        linear_out = self.layers[f'linear_out_{layer}']
        linear_out2 = self.layers[f'linear_out2_{layer}']
        
        # Shape information
        num_depth = depth_values.shape[1]
        num_views = len(src_features) + 1
        b, _, h, w = ref_feature.shape
        nhead = self.config.nhead
        device = ref_feature.device
        agg = 0
        
        # Query features from reference view
        q = query(ref_feature)
        
        # Process each source view
        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            # Key features from source view
            k = key(src_fea)
            
            # Warp source view features to reference view
            k, proj_mask, grid = homo_warping(k, src_proj, ref_proj, depth_values)
            
            # Embedding for projection mask
            m = m_embed(proj_mask.long()).permute(0, 4, 1, 2, 3)
            
            # Position embedding for depth values
            pos = pos_embed.weight[None, :, None, None, :].repeat(
                b, 1, h, w, 1).permute(0, 4, 1, 2, 3) * m
            
            # Attention calculation
            att_dim = k.shape[1]
            attn = ((q.unsqueeze(2) * k).view(b, -1, nhead, num_depth, h, w).sum(1, keepdim=True) / 
                   np.sqrt(att_dim // nhead)).softmax(dim=3)
            
            # Apply attention to position embeddings
            v = pos.view(b, -1, nhead, num_depth, h, w)
            agg = agg + (attn * v).sum(3).view(b, -1, h, w)
        
        # Scale aggregation based on number of views
        if len(src_features) + 1 != self.config.num_frame:
            agg = agg / float(len(src_features)) * (self.config.num_frame - 1)
        
        # Final processing of aggregated features
        agg = linear_out(agg) + linear_out2(ref_feature)
        return agg
    
    def decoder(self, ref_feature):
        """
        Decoder for feature maps.
        
        Args:
            ref_feature: List of feature maps from encoder
            
        Returns:
            Decoded feature map
        """
        x = ref_feature[-1]
        for i in range(4, self.output_layer, -1):
            x = self.layers[f'upconv_{i}_0'](x)
            if i >= 2 - self.config.input_scale:
                x = self.upsample(x)
                if self.config.use_skip:
                    x = torch.cat((x, ref_feature[i - 1]), 1)
                x = self.layers[f'upconv_{i}_1'](x)
            else:
                break
        return x
    
    def regress_depth(self, feature_map_d):
        """
        Regress depth map from feature map.
        
        Args:
            feature_map_d: Feature map for depth prediction
            
        Returns:
            Predicted depth map
        """
        x = self.depth_regressor(feature_map_d).softmax(dim=1)
        d = compute_depth_expectation(
            x,
            self.depth_expectation_anchor.unsqueeze(0).repeat(x.shape[0], 1)
        ).unsqueeze(1)
        return d
    
    def forward(self, ref_img, src_imgs, ref_proj, src_projs, inv_K):
        """
        Forward pass through the MVS2D network.
        
        Args:
            ref_img: Reference view image
            src_imgs: List of source view images
            ref_proj: Reference view projection matrix
            src_projs: List of source view projection matrices
            inv_K: Inverse of camera intrinsic matrix
            
        Returns:
            Dictionary of outputs including depth prediction
        """
        outputs = {}
        ref_feature = ref_img
        src_features = [x for x in src_imgs]
        V = len(src_imgs) + 1
        ref_feature2 = ref_img
        ref_skip_feat2 = []
        cnt = 0
        
        # Process through encoder with multi-view attention
        for k, v in self.base_model2._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            if cnt < len(self.attn_layers):
                ref_feature = getattr(self.base_model, k)(ref_feature)
                for i in range(V - 1):
                    src_features[i] = getattr(self.base_model, k)(src_features[i])
            ref_feature2 = v(ref_feature2)
            
            # Apply epipolar attention at specified layers
            if k in self.attn_layers:
                b = ref_img.shape[0]
                depth_values = self.depth_values[None, :, None, None].repeat(
                    b, 1, ref_feature.shape[2], ref_feature.shape[3])
                
                sz_ref = (ref_feature.shape[2], ref_feature.shape[3])
                sz_src = (src_features[0].shape[2], src_features[0].shape[3])
                linear_out3 = self.layers[f'linear_out3_{k}']
                
                att_f = self.epipolar_fusion(
                    ref_feature,
                    src_features,
                    ref_proj[sz_ref],
                    [proj[sz_src] for proj in src_projs],
                    depth_values,
                    k,
                    ref_img,
                    src_imgs,
                )
                ref_feature2 = ref_feature2 + linear_out3(ref_feature2) + att_f
                cnt += 1
            
            # Store skip connections
            if any(x in k for x in self.feat_names):
                ref_skip_feat2.append(ref_feature2)
        
        # Decode into depth map
        feature_map = self.decoder(ref_skip_feat2)
        
        # Extract confidence if needed
        if self.config.pred_conf:
            feature_map = self.conv_out(feature_map)
            outputs[('log_conf_pred', self.config.output_scale)] = feature_map[:, -1:, :, :]
            feature_map_d = feature_map[:, :-1, :, :]
        else:
            feature_map_d = self.conv_out(feature_map)
        
        # Regress depth
        depth_pred = self.regress_depth(feature_map_d)
        
        # Upsample depth map to original resolution
        depth_pred = self.upsample(depth_pred, scale_factor=4) + 1e-1 * self.conv_up(
            torch.cat((depth_pred, self.meshgrid.unsqueeze(0).repeat(
                depth_pred.shape[0], 1, 1, 1).to(depth_pred), feature_map_d), 1))
        
        # Format confidence output
        if self.config.pred_conf:
            outputs[('log_conf_pred', 0)] = F.interpolate(
                outputs[('log_conf_pred', self.config.output_scale)],
                size=(self.config.height, self.config.width))
        
        outputs[('depth_pred', 0)] = depth_pred
        return outputs


class MVS2DProcessor:
    """
    Processor class for MVS2D neural network model.
    """
    def __init__(self, config=None, device=None):
        self.config = config or MVS2DConfig()
        
        # Check if CUDA is available
        if device is None:
            if GPU_UTILS_AVAILABLE and is_gpu_available() and self.config.use_gpu:
                self.device = torch.device("cuda:0")
                logger.info("Using CUDA for MVS2D neural network processing")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for MVS2D neural network processing")
        else:
            self.device = device
            
        # Initialize model path
        if self.config.model_path is None:
            self.config.model_path = DEFAULT_MODEL_PATH
            
        # Check if model directory exists
        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path, exist_ok=True)
            logger.info(f"Created model directory at {self.config.model_path}")
            
        # Find model file based on model type
        model_file = None
        if os.path.exists(os.path.join(self.config.model_path, f"{self.config.model_type}_model.pth")):
            model_file = os.path.join(self.config.model_path, f"{self.config.model_type}_model.pth")
            
        # Initialize model
        self.model = None
        if TORCH_AVAILABLE:
            try:
                self.model = MVS2DModel(self.config)
                if model_file and os.path.exists(model_file):
                    logger.info(f"Loading MVS2D model from {model_file}")
                    self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                else:
                    logger.warning(f"Model file not found at {model_file if model_file else self.config.model_path}")
                    logger.warning("Using model with random weights. Results may not be optimal.")
                
                self.model.to(self.device)
                self.model.eval()
                logger.info("MVS2D model initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize MVS2D model: {e}")
                self.model = None
        else:
            logger.warning("PyTorch not available. MVS2D model not initialized.")
            
    def process_images(self, ref_img, src_imgs, ref_proj, src_projs, inv_K=None):
        """
        Process images with MVS2D model to generate depth map.
        
        Args:
            ref_img: Reference view image (H, W, 3) or (B, 3, H, W)
            src_imgs: List of source view images (H, W, 3) or (B, 3, H, W)
            ref_proj: Reference view projection matrix (3, 4) or (B, 3, 4)
            src_projs: List of source view projection matrices (3, 4) or (B, 3, 4)
            inv_K: Inverse of camera intrinsic matrix (optional)
            
        Returns:
            Dictionary with depth map and confidence map
        """
        if not TORCH_AVAILABLE or self.model is None:
            logger.error("MVS2D model not available")
            return None
        
        try:
            with torch.no_grad():
                # Preprocess images
                if isinstance(ref_img, np.ndarray):
                    if ref_img.ndim == 3:  # (H, W, 3)
                        ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0
                        ref_img = ref_img.unsqueeze(0)  # (1, 3, H, W)
                    elif ref_img.ndim == 4:  # (B, H, W, 3)
                        ref_img = torch.from_numpy(ref_img).permute(0, 3, 1, 2).float() / 255.0
                
                # Process source images
                if isinstance(src_imgs[0], np.ndarray):
                    processed_src_imgs = []
                    for img in src_imgs:
                        if img.ndim == 3:  # (H, W, 3)
                            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                            img = img.unsqueeze(0)  # (1, 3, H, W)
                        elif img.ndim == 4:  # (B, H, W, 3)
                            img = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
                        processed_src_imgs.append(img)
                    src_imgs = processed_src_imgs
                
                # Process projection matrices
                if isinstance(ref_proj, np.ndarray):
                    ref_proj = torch.from_numpy(ref_proj).float()
                    if ref_proj.ndim == 2:  # (3, 4)
                        ref_proj = ref_proj.unsqueeze(0)  # (1, 3, 4)
                
                if isinstance(src_projs[0], np.ndarray):
                    processed_src_projs = []
                    for proj in src_projs:
                        proj = torch.from_numpy(proj).float()
                        if proj.ndim == 2:  # (3, 4)
                            proj = proj.unsqueeze(0)  # (1, 3, 4)
                        processed_src_projs.append(proj)
                    src_projs = processed_src_projs
                
                # Create inverse K if not provided
                if inv_K is None:
                    if ref_proj.ndim == 3:
                        K = ref_proj[:, :3, :3]
                        inv_K = torch.inverse(K)
                    else:
                        logger.warning("Could not compute inverse K. Using identity matrix.")
                        inv_K = torch.eye(3).unsqueeze(0)
                elif isinstance(inv_K, np.ndarray):
                    inv_K = torch.from_numpy(inv_K).float()
                    if inv_K.ndim == 2:
                        inv_K = inv_K.unsqueeze(0)
                
                # Move data to device
                ref_img = ref_img.to(self.device)
                src_imgs = [img.to(self.device) for img in src_imgs]
                ref_proj = {(ref_img.shape[2], ref_img.shape[3]): ref_proj.to(self.device)}
                src_projs = [{(img.shape[2], img.shape[3]): proj.to(self.device)} for img, proj in zip(src_imgs, src_projs)]
                inv_K = inv_K.to(self.device)
                
                # Forward pass
                outputs = self.model(ref_img, src_imgs, ref_proj, src_projs, inv_K)
                
                # Convert outputs to numpy arrays
                result = {}
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.detach().cpu().numpy()
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing images with MVS2D: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def enhance_point_cloud(self, point_cloud, images=None, camera_matrices=None):
        """
        Enhance point cloud using MVS2D neural network.
        
        Args:
            point_cloud: Open3D point cloud
            images: List of images for texture mapping (optional)
            camera_matrices: List of camera matrices (optional)
            
        Returns:
            Enhanced point cloud
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D not available for point cloud processing")
            return point_cloud
        
        if self.model is None:
            logger.error("MVS2D model not available")
            return point_cloud
        
        # If we don't have images and camera matrices, we can't do much
        if images is None or camera_matrices is None:
            logger.warning("Images or camera matrices not provided. Skipping point cloud enhancement.")
            return point_cloud
        
        try:
            # Process the point cloud
            logger.info(f"Enhancing point cloud with {len(images)} views")
            
            # Convert point cloud to numpy array
            points = np.asarray(point_cloud.points)
            
            # Process each view to get depth maps
            depth_maps = []
            for i in range(1, len(images)):
                ref_img = images[0]
                src_imgs = [images[i]]
                ref_proj = camera_matrices[0]
                src_projs = [camera_matrices[i]]
                
                result = self.process_images(ref_img, src_imgs, ref_proj, src_projs)
                if result is not None and ('depth_pred', 0) in result:
                    depth_maps.append(result[('depth_pred', 0)][0, 0])
            
            # If we have depth maps, we can use them to enhance the point cloud
            if depth_maps:
                # Create enhanced point cloud from depth maps
                enhanced_points = []
                for i, depth_map in enumerate(depth_maps):
                    # Project depth map to 3D points
                    h, w = depth_map.shape
                    K = camera_matrices[i+1][:3, :3]  # Intrinsic matrix
                    R = camera_matrices[i+1][:3, :3]  # Rotation matrix
                    t = camera_matrices[i+1][:3, 3]   # Translation vector
                    
                    # Create pixel coordinates
                    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
                    xs = xs.flatten()
                    ys = ys.flatten()
                    depth = depth_map.flatten()
                    
                    # Filter out invalid depth values
                    mask = depth > 0
                    xs = xs[mask]
                    ys = ys[mask]
                    depth = depth[mask]
                    
                    # Convert pixel coordinates to camera coordinates
                    x_cam = (xs - K[0, 2]) * depth / K[0, 0]
                    y_cam = (ys - K[1, 2]) * depth / K[1, 1]
                    z_cam = depth
                    
                    # Convert camera coordinates to world coordinates
                    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
                    points_world = np.dot(points_cam, R.T) + t
                    
                    enhanced_points.append(points_world)
                
                # Combine enhanced points
                if enhanced_points:
                    enhanced_points = np.vstack(enhanced_points)
                    
                    # Create enhanced point cloud
                    enhanced_pcd = o3d.geometry.PointCloud()
                    enhanced_pcd.points = o3d.utility.Vector3dVector(enhanced_points)
                    
                    # Merge with original point cloud
                    merged_pcd = point_cloud + enhanced_pcd
                    
                    # Remove duplicates and outliers
                    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)
                    merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    
                    logger.info(f"Enhanced point cloud from {len(point_cloud.points)} to {len(merged_pcd.points)} points")
                    return merged_pcd
            
            return point_cloud
            
        except Exception as e:
            logger.error(f"Error enhancing point cloud: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return point_cloud


# Function to check if MVS2D is available
def is_mvs2d_available():
    """Check if MVS2D neural network processing is available."""
    return TORCH_AVAILABLE


# Singleton instance
_mvs2d_processor = None

def get_mvs2d_processor(config=None, device=None):
    """Get or create the MVS2D processor singleton instance."""
    global _mvs2d_processor
    if _mvs2d_processor is None:
        _mvs2d_processor = MVS2DProcessor(config=config, device=device)
    return _mvs2d_processor


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check availability
    if is_mvs2d_available():
        logger.info("MVS2D neural network processing is available")
        
        # Initialize processor
        processor = get_mvs2d_processor()
        
        logger.info(f"MVS2D processor initialized with device: {processor.device}")
    else:
        logger.warning("MVS2D neural network processing is not available")