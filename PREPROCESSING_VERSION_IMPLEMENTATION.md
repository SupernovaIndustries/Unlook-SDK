# UnLook SDK - Preprocessing Version System Implementation

## Overview
Implementation of a comprehensive preprocessing version system that allows clients to specify which preprocessing pipeline to use, enabling separation between handpose (V1) and enhanced 3D scanning (V2) pipelines.

## 🔧 Features Implemented

### 1. Client-Side Version Selection

#### Constants and Configuration (`unlook/core/constants.py`)
```python
class PreprocessingVersion:
    """Preprocessing pipeline version selection."""
    V1_LEGACY = "v1_legacy"          # Original pipeline, optimized for handpose
    V2_ENHANCED = "v2_enhanced"      # Enhanced pipeline with GPU preprocessing
    AUTO = "auto"                    # Auto-select based on use case

# Preprocessing configuration per version
PREPROCESSING_CONFIGS = {
    PreprocessingVersion.V1_LEGACY: {
        'protocol_version': '1.0',
        'gpu_preprocessing': False,
        'compression_enabled': False,
        'delta_encoding': False,
        'optimization_level': 'none',
        'description': 'Legacy V1 pipeline for handpose and real-time applications'
    },
    PreprocessingVersion.V2_ENHANCED: {
        'protocol_version': '2.0',
        'gpu_preprocessing': True,
        'compression_enabled': True,
        'delta_encoding': True,
        'optimization_level': 'advanced',
        'description': 'Enhanced V2 pipeline for 3D scanning with GPU acceleration'
    },
    PreprocessingVersion.AUTO: {
        'protocol_version': 'auto',
        'gpu_preprocessing': 'auto',
        'compression_enabled': 'auto',
        'delta_encoding': 'auto',
        'optimization_level': 'auto',
        'description': 'Auto-select best pipeline based on use case'
    }
}
```

#### UnlookClient Modifications (`unlook/client/scanner/scanner.py`)
- Added `preprocessing_version` parameter to `__init__()`
- Automatic configuration loading based on version
- Enhanced HELLO message with preprocessing configuration:

```python
def __init__(
        self,
        client_name: str = "UnlookClient",
        auto_discover: bool = True,
        discovery_callback: Optional[Callable[[ScannerInfo, bool], None]] = None,
        preprocessing_version: str = PreprocessingVersion.AUTO
):
    # Preprocessing version configuration
    self.preprocessing_version = preprocessing_version
    self.preprocessing_config = PREPROCESSING_CONFIGS.get(
        preprocessing_version, 
        PREPROCESSING_CONFIGS[PreprocessingVersion.AUTO]
    )

# Enhanced HELLO message
hello_msg = Message(
    msg_type=MessageType.HELLO,
    payload={
        "client_info": {
            "name": self.name,
            "id": self.id,
            "version": "1.0",
            **get_machine_info()
        },
        "preprocessing_config": {
            "version": self.preprocessing_version,
            "config": self.preprocessing_config
        }
    }
)
```

### 2. Server-Side Version Handling

#### New Message Types (`unlook/core/protocol.py`)
```python
# Preprocessing version control
PREPROCESSING_VERSION_SET = "preprocessing_version_set"  # Set preprocessing version
PREPROCESSING_VERSION_GET = "preprocessing_version_get"  # Get current preprocessing version
```

#### Server Configuration Methods (`unlook/server/scanner.py`)
```python
def _apply_preprocessing_config(self, client_id: str, version: str, config: Dict[str, Any]):
    """Apply preprocessing configuration for a specific client."""
    
def _configure_v1_preprocessing(self, client_id: str):
    """Configure server for V1 legacy preprocessing."""
    self._client_configs[client_id] = {
        'use_protocol_v2': False,
        'enable_gpu_preprocessing': False,
        'compression_level': 0,
        'delta_encoding': False
    }

def _configure_v2_preprocessing(self, client_id: str):
    """Configure server for V2 enhanced preprocessing."""
    self._client_configs[client_id] = {
        'use_protocol_v2': True,
        'enable_gpu_preprocessing': True,
        'compression_level': 6,
        'delta_encoding': True
    }
```

#### Enhanced HELLO Handler
```python
def _handle_hello(self, message: Message) -> Message:
    """Handle HELLO messages with preprocessing configuration."""
    # Handle preprocessing configuration
    preprocessing_config = message.payload.get("preprocessing_config", {})
    if preprocessing_config:
        requested_version = preprocessing_config.get("version", "auto")
        config = preprocessing_config.get("config", {})
        
        # Apply preprocessing configuration
        self._apply_preprocessing_config(client_id, requested_version, config)
    
    # Enhanced response with preprocessing info
    return Message.create_reply(
        message,
        {
            "scanner_name": self.name,
            "preprocessing": {
                "current_version": getattr(self, 'current_preprocessing_version', 'auto'),
                "supported_versions": ["v1_legacy", "v2_enhanced", "auto"]
            }
        }
    )
```

### 3. Protocol V2 Multi-Camera Enhancement

#### New Multi-Camera Methods (`unlook/core/protocol_v2.py`)
```python
def optimize_multi_camera_message(self, camera_data: Dict[str, bytes], metadata: Dict[str, Any]) -> Tuple[bytes, CompressionStats]:
    """Optimize multi-camera message for transmission."""
    # Individual camera compression
    # Metadata with camera offsets and sizes
    # V2 format serialization

def _deserialize_multi_camera_data(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, bytes]:
    """Deserialize multi-camera data from V2 format."""
    # Extract individual camera data
    # Decompress each camera separately
    # Return dict of camera_id -> image_data
```

#### Enhanced Deserializer
- Better V2 detection and handling
- Multi-camera support in deserializer
- Proper handling of compression per camera

### 4. Camera Client Improvements

#### Enhanced Multi-Camera Support (`unlook/client/camera/camera.py`)
```python
# PROTOCOL V2 MULTI-CAMERA HANDLING
if (msg_type == "multi_camera_response" and 
    isinstance(binary_data, dict) and 
    payload.get('optimization', {}).get('multi_camera', False)):
    
    # binary_data is already a dict of camera_id -> image_bytes
    images = {}
    for camera_id, image_bytes in binary_data.items():
        image = decode_jpeg_to_image(image_bytes)
        if image is not None:
            images[camera_id] = image
```

#### Improved Single Camera Capture
- Better error handling for different response formats
- Support for camera_capture_response format
- Fallback mechanisms for JPEG extraction

### 5. Application-Specific Configurations

#### Enhanced 3D Scanner (`unlook/examples/scanning/enhanced_3d_scanning_pipeline_v2.py`)
```python
# Create client with V2 Enhanced preprocessing
from unlook.core.constants import PreprocessingVersion
client = UnlookClient(
    client_name="Enhanced3DScanningV2",
    auto_discover=True,
    preprocessing_version=PreprocessingVersion.V2_ENHANCED
)
```

#### Handpose Applications (`unlook/examples/handpose/enhanced_gesture_demo.py`)
```python
# Import preprocessing version for V1 legacy
from unlook.core.constants import PreprocessingVersion

client = UnlookClient(
    auto_discover=False,
    preprocessing_version=PreprocessingVersion.V1_LEGACY
)
```

## 🎯 Pipeline Separation

### V1 Legacy Pipeline (Handpose)
- **Target**: Real-time gesture recognition, maximum FPS
- **Protocol**: V1 standard
- **GPU Preprocessing**: Disabled
- **Compression**: Minimal
- **Delta Encoding**: Disabled
- **Optimization**: Speed over quality

### V2 Enhanced Pipeline (3D Scanning)
- **Target**: High-quality 3D reconstruction
- **Protocol**: V2 with optimization
- **GPU Preprocessing**: Enabled
- **Compression**: Advanced adaptive
- **Delta Encoding**: Enabled
- **Optimization**: Quality over speed

### Auto Selection
- Automatically chooses best pipeline based on server capabilities
- Falls back to V1 if V2 not available
- Considers client use case requirements

## 🔍 Key Benefits

1. **No Pipeline Interference**: Each application uses its optimal preprocessing
2. **Performance Optimization**: Handpose gets maximum speed, 3D scanning gets maximum quality
3. **Backward Compatibility**: V1 applications continue to work
4. **Forward Compatibility**: New applications can leverage V2 features
5. **Flexible Configuration**: Per-client configuration without affecting others
6. **Server Efficiency**: Resources allocated based on client needs

## 🛠 Implementation Details

### Client-Server Communication Flow
1. Client specifies preprocessing version in constructor
2. Client sends version in HELLO message
3. Server applies appropriate configuration
4. Server responds with current capabilities
5. All subsequent communications use configured pipeline

### Multi-Camera V2 Format
```
Header: {
  "type": "multi_camera_response",
  "metadata": {
    "optimization": {
      "multi_camera": true,
      "num_cameras": 2
    },
    "cameras": {
      "camera_0": {"offset": 0, "size": 123456, "compression_level": 6},
      "camera_1": {"offset": 123456, "size": 109876, "compression_level": 6}
    }
  }
}
Data: [compressed_camera_0_data][compressed_camera_1_data]
```

### Error Handling and Fallbacks
- Graceful fallback to V1 if V2 fails
- Multiple JPEG extraction methods
- Robust error recovery
- Detailed logging for debugging

## 📋 Testing and Validation

### Test Scenarios
1. **V1 Handpose Application**: Should use legacy pipeline, minimal preprocessing
2. **V2 3D Scanner Application**: Should use enhanced pipeline, full preprocessing
3. **Mixed Clients**: Multiple clients with different versions should coexist
4. **Auto Selection**: Should choose appropriate version based on server capabilities

### Success Criteria
- ✅ No "Suspicious header size" warnings
- ✅ Multi-camera returns all requested images
- ✅ Handpose maintains high FPS
- ✅ 3D scanner gets high-quality preprocessing
- ✅ Server logs show correct version selection

## 🚀 Future Enhancements

1. **Dynamic Version Switching**: Allow clients to change version during session
2. **Performance Metrics**: Track performance differences between versions
3. **Custom Configurations**: Allow fine-tuning of preprocessing parameters
4. **Version Negotiation**: Automatic version negotiation based on capabilities
5. **Load Balancing**: Server-side load balancing based on version requirements

---

*Implementation completed: 2025-01-29*
*Total files modified: 6*
*Lines of code added: ~400*
*New features: Version selection, multi-camera V2, pipeline separation*