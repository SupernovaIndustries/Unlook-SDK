# Camera Import Fix Solution

## Problem
Python was importing the `camera/` directory module instead of the `camera.py` file when code attempted to import from `unlook.client.camera`. This caused the error:
```
module 'unlook.client.camera' has no attribute 'CameraClient'
```

## Root Cause
There's a naming conflict between:
- `/unlook/client/camera.py` (the file containing CameraClient class)
- `/unlook/client/camera/` (the directory containing camera_auto_optimizer.py)

When Python sees `from unlook.client.camera import CameraClient`, it imports the camera directory's `__init__.py` instead of the `camera.py` file.

## Solution
Use Python's `importlib.util` to explicitly load the `camera.py` file by its exact path:

```python
@property
def camera(self):
    """Lazy-loading of the camera client."""
    if self._camera is None:
        # Use importlib to directly import from camera.py file
        import importlib.util
        import os
        camera_file = os.path.join(os.path.dirname(__file__), 'camera.py')
        spec = importlib.util.spec_from_file_location("camera_module", camera_file)
        camera_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(camera_module)
        self._camera = camera_module.CameraClient(self)
    return self._camera
```

## Applied Changes
The fix has been applied to:
1. `/unlook/client/scanner.py` - UnlookClient.camera property
2. The static scanner inherits from UnlookClient, so no additional changes needed

## Alternative Solutions (Not Implemented)
1. Rename the camera.py file to camera_client.py
2. Rename the camera/ directory to camera_utils/
3. Use absolute imports throughout the codebase

## Testing
To test if the fix works:
1. Activate the virtual environment: `.venv\Scripts\activate` (Windows)
2. Run the scanner example: `python unlook/examples/scanning/static_scanning_example_fixed.py --server_ip <ip> --debug`
3. The scanner should now properly access the camera without import errors

## Notes
- This fix uses Python's import machinery to bypass the standard module resolution
- The solution is backward compatible and doesn't require refactoring existing code
- The camera module is loaded lazily only when first accessed