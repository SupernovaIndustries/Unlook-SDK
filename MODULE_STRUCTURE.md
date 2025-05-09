# Unlook SDK Module Structure

## Server-Client Architecture

The Unlook SDK is divided into two main parts:

1. **Client** - For applications that connect to and control the scanner
2. **Server** - For implementing the scanner server itself

These components are meant to be used independently, with the server typically running on a Raspberry Pi and clients connecting remotely.

## Dependency Structure

The modules are organized to minimize dependencies:

```
unlook/
├── __init__.py           # Main package initialization 
├── core/                 # Shared core modules
│   ├── events.py         # Event handling
│   ├── discovery.py      # Scanner discovery
│   └── protocol.py       # Communication protocol
├── client/               # Client modules
│   ├── scanner.py        # Main client implementation
│   ├── camera.py         # Camera control
│   └── ...
├── server/               # Server modules
│   ├── scanner.py        # Main server implementation
│   ├── hardware/         # Hardware interfaces
│   └── ...
└── server_bootstrap.py   # Server startup script
```

## Import Rules

To prevent circular dependencies and minimize server resource usage:

1. Server modules should **never** import from client modules
2. Client modules can import from core modules
3. Server modules can import from core modules
4. Core modules should not import from either client or server modules

## Server-Only Mode

When running `server_bootstrap.py`, a special flag is set in the global namespace to prevent client module imports:

```python
# Set in server_bootstrap.py
import builtins
builtins._SERVER_ONLY_MODE = True
```

This flag is checked in the main `__init__.py` file to conditionally skip client imports when in server mode.

## Placeholder Modules

Some client modules that require heavy dependencies (like PyTorch for neural networks) have lightweight server-side placeholders to avoid unnecessary dependencies on the server.

Example:
- `client/point_cloud_nn.py` - Full implementation with PyTorch
- `server/point_cloud_nn.py` - Placeholder that returns dummy values

## Best Practices

1. Always use relative imports within modules:
   ```python
   # In client modules
   from ..core import events  # Good
   from unlook.core import events  # Avoid
   ```

2. For cyclical imports, use deferred imports:
   ```python
   def my_function():
       # Import only when needed
       from .other_module import OtherClass
       return OtherClass()
   ```

3. Limit what's exposed in `__init__.py` files to minimize import side effects.

## Testing Import Independence

You can test if server and client are properly separated by running:

```bash
# Should work without errors
python -c "import builtins; builtins._SERVER_ONLY_MODE=True; import unlook.server"

# Should not import client modules
python -c "import builtins; builtins._SERVER_ONLY_MODE=True; import unlook"
```