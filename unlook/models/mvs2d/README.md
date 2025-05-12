# MVS2D Integration for Unlook SDK

This directory contains the model files and integration code for the MVS2D (Multi-view Stereo via Attention-Driven 2D Convolutions) neural network.

## Overview

MVS2D is a deep learning approach for multi-view stereo depth estimation. It uses 2D convolutions with attention mechanisms to efficiently process multi-view inputs and produce high-quality depth maps.

## Integration with Unlook SDK

The MVS2D model has been integrated with the Unlook SDK to enhance point cloud processing. The model can be used to:

1. Improve depth map quality from structured light scanning
2. Enhance point cloud density and detail
3. Fill holes in sparse point clouds
4. Reduce noise and outliers

## Model Files

- `networks/`: Core model code from the original MVS2D implementation
- Place pre-trained model weights in this directory with naming format: `{model_type}_model.pth`
  - Supported model types: `scannet`, `demon`, `dtu`

## References

- Paper: [MVS2D: Efficient Multi-view Stereo via Attention-Driven 2D Convolutions](https://arxiv.org/abs/2104.13325)
- Original implementation: [https://github.com/zhenpeiyang/MVS2D](https://github.com/zhenpeiyang/MVS2D)

## Usage

Download pre-trained models from the original repository and place them in this directory. The Unlook SDK's neural network processing module will automatically detect and use these models when enabled in the scanning configuration.

```python
from unlook.client import StaticScanConfig, create_static_scanner

# Create scanner configuration
config = StaticScanConfig()
config.use_neural_network = True
config.nn_model_path = "path/to/model_weights.pth"  # Optional, will use default if not specified

# Create scanner with neural network enhancement
scanner = create_static_scanner(client, config=config)
```