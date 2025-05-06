"""
Costanti condivise tra client e server.
"""

# Porte di default
DEFAULT_CONTROL_PORT = 5555
DEFAULT_STREAM_PORT = 5556
DEFAULT_DISCOVERY_PORT = 5557

# Timeout e tentativi
DEFAULT_TIMEOUT = 5000  # ms
MAX_RETRIES = 3

# Service discovery
SERVICE_TYPE = "_unlook._tcp.local."
SERVICE_NAME = "UnLook Scanner"

# Compressione video
DEFAULT_JPEG_QUALITY = 80
DEFAULT_STREAM_FPS = 30

# Versione protocollo
PROTOCOL_VERSION = "1.0"