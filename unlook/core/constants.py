"""
Costanti condivise tra client e server.
"""

# ==================== NETWORK CONSTANTS ====================
# Porte di default
DEFAULT_CONTROL_PORT = 5555
DEFAULT_STREAM_PORT = 5556
DEFAULT_DISCOVERY_PORT = 5557
DEFAULT_DIRECT_PORT = 5557  # Alias for compatibility

# Timeout e tentativi
DEFAULT_TIMEOUT = 5000  # ms
MESSAGE_TIMEOUT = 5000  # milliseconds
DISCOVERY_TIMEOUT = 5.0  # seconds
MAX_RETRIES = 3

# Service discovery
SERVICE_TYPE = "_unlook._tcp.local."
SERVICE_NAME = "UnLook Scanner"

# Versione protocollo
PROTOCOL_VERSION = "1.0"

# ==================== IMAGE PROCESSING CONSTANTS ====================
# Dimensioni immagine
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720

# Compressione
DEFAULT_JPEG_QUALITY = 85  # Increased from 80 for better quality
DEFAULT_PNG_COMPRESSION = 9
DEFAULT_STREAM_FPS = 30

# ==================== CAMERA CONSTANTS ====================
DEFAULT_EXPOSURE_TIME = 10000  # microseconds
DEFAULT_CAMERA_GAIN = 1.0
DEFAULT_FPS = 30
MAX_CAMERAS = 10
CAMERA_WARMUP_TIME = 2.0  # seconds

# ==================== PATTERN GENERATION CONSTANTS ====================
DEFAULT_PATTERN_WIDTH = 1024
DEFAULT_PATTERN_HEIGHT = 768
DEFAULT_NUM_GRAY_CODES = 10
DEFAULT_NUM_PHASE_SHIFTS = 8
DEFAULT_CHECKERBOARD_SIZE = (9, 6)
DEFAULT_SQUARE_SIZE = 24.0  # mm

# ==================== SCANNING CONSTANTS ====================
DEFAULT_PATTERN_INTERVAL = 0.5  # seconds
DEFAULT_VOXEL_SIZE = 0.5  # mm
DEFAULT_MAX_DEPTH = 5000.0  # mm
DEFAULT_MIN_DEPTH = 50.0  # mm
DEFAULT_OUTLIER_STD = 2.0
DEFAULT_MESH_DEPTH = 9

# ==================== HANDPOSE CONSTANTS ====================
DEFAULT_DETECTION_CONFIDENCE = 0.6
DEFAULT_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 2
GESTURE_THRESHOLD = 0.7
GESTURE_STABILITY_FRAMES = 2
NUM_HAND_LANDMARKS = 21

# ==================== STREAMING CONSTANTS ====================
STREAM_TIMEOUT = 1000  # milliseconds
WATCHDOG_INTERVAL = 5.0  # seconds
FRAME_BUFFER_SIZE = 10

# ==================== FILE FORMAT CONSTANTS ====================
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SUPPORTED_POINTCLOUD_FORMATS = ['.ply', '.pcd', '.xyz', '.pts']
SUPPORTED_MESH_FORMATS = ['.ply', '.obj', '.stl', '.off']
CALIBRATION_FILE_EXTENSION = '.json'

# ==================== DEBUG CONSTANTS ====================
DEBUG_IMAGE_FORMAT = '.png'
DEBUG_DIR_PREFIX = 'unlook_debug'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ==================== PERFORMANCE CONSTANTS ====================
CPU_THREAD_COUNT = 4
GPU_BATCH_SIZE = 1000
MEMORY_POOL_SIZE = 100 * 1024 * 1024  # 100MB
CACHE_SIZE = 50

# ==================== QUALITY PRESETS ====================
QUALITY_PRESETS = {
    'fast': {
        'pattern_resolution': (640, 480),
        'num_patterns': 10,
        'voxel_size': 1.0,
        'processing_threads': 2
    },
    'balanced': {
        'pattern_resolution': (800, 600),
        'num_patterns': 20,
        'voxel_size': 0.5,
        'processing_threads': 4
    },
    'high': {
        'pattern_resolution': (1024, 768),
        'num_patterns': 30,
        'voxel_size': 0.25,
        'processing_threads': 8
    }
}

# ==================== ERROR MESSAGES ====================
ERROR_NO_CAMERA = "No camera found with ID: {}"
ERROR_NO_CALIBRATION = "No calibration data available"
ERROR_INVALID_PATTERN = "Invalid pattern type: {}"
ERROR_CAPTURE_FAILED = "Failed to capture image from camera: {}"
ERROR_CONNECTION_FAILED = "Failed to connect to scanner: {}"

# ==================== SUCCESS MESSAGES ====================
SUCCESS_CONNECTED = "Successfully connected to scanner: {}"
SUCCESS_CALIBRATION_LOADED = "Calibration loaded from: {}"
SUCCESS_SCAN_COMPLETE = "Scan completed with {} points"