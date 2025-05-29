Exception Handling
==================

The exceptions module provides custom exception classes for consistent error handling across the UnLook SDK.

.. module:: unlook.client.exceptions

Overview
--------

All UnLook exceptions inherit from :class:`UnlookException`, providing:

- Consistent error messages
- Additional error details
- Proper exception hierarchy
- Error handling decorators

Exception Hierarchy
------------------

.. inheritance-diagram:: 
   unlook.client.exceptions.UnlookException
   unlook.client.exceptions.ConnectionError
   unlook.client.exceptions.CameraError
   unlook.client.exceptions.CameraCaptureError
   unlook.client.exceptions.CameraNotFoundError
   unlook.client.exceptions.CalibrationError
   unlook.client.exceptions.CalibrationNotFoundError
   unlook.client.exceptions.CalibrationInvalidError
   unlook.client.exceptions.ProjectorError
   unlook.client.exceptions.PatternError
   unlook.client.exceptions.ScanningError
   unlook.client.exceptions.InsufficientDataError
   unlook.client.exceptions.ReconstructionError
   unlook.client.exceptions.StreamingError
   unlook.client.exceptions.StreamTimeoutError
   unlook.client.exceptions.ConfigurationError
   unlook.client.exceptions.HardwareError
   unlook.client.exceptions.DependencyError
   :parts: 1

Base Exception
-------------

.. autoclass:: UnlookException
   :members:
   :undoc-members:
   :show-inheritance:

Connection Errors
----------------

.. autoclass:: ConnectionError
   :members:
   :undoc-members:
   :show-inheritance:

Camera Errors
------------

.. autoclass:: CameraError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CameraCaptureError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CameraNotFoundError
   :members:
   :undoc-members:
   :show-inheritance:

Calibration Errors
-----------------

.. autoclass:: CalibrationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CalibrationNotFoundError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CalibrationInvalidError
   :members:
   :undoc-members:
   :show-inheritance:

Projector Errors
---------------

.. autoclass:: ProjectorError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PatternError
   :members:
   :undoc-members:
   :show-inheritance:

Scanning Errors
--------------

.. autoclass:: ScanningError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InsufficientDataError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ReconstructionError
   :members:
   :undoc-members:
   :show-inheritance:

Streaming Errors
---------------

.. autoclass:: StreamingError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StreamTimeoutError
   :members:
   :undoc-members:
   :show-inheritance:

Other Errors
-----------

.. autoclass:: ConfigurationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HardwareError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DependencyError
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
-------------

Basic exception handling::

    from unlook.client.exceptions import CameraNotFoundError, CameraCaptureError
    
    try:
        image = camera.capture('left')
    except CameraNotFoundError as e:
        print(f"Camera not found: {e.message}")
        print(f"Camera ID: {e.details['camera_id']}")
    except CameraCaptureError as e:
        print(f"Capture failed: {e.message}")
        print(f"Reason: {e.details['reason']}")

Checking exception details::

    from unlook.client.exceptions import InsufficientDataError
    
    try:
        scan_result = scanner.scan()
    except InsufficientDataError as e:
        print(f"Error: {e.message}")
        print(f"Expected {e.details['expected']} {e.details['data_type']}")
        print(f"Got only {e.details['actual']}")

Using error decorators::

    from unlook.client.exceptions import handle_camera_error
    
    @handle_camera_error
    def capture_stereo():
        left = camera.capture('left')
        right = camera.capture('right')
        return left, right

Best Practices
-------------

1. **Catch specific exceptions**::

    # Good - specific handling
    try:
        result = scanner.scan()
    except CalibrationNotFoundError:
        # Load default calibration
        scanner.load_default_calibration()
        result = scanner.scan()
    
    # Bad - too broad
    try:
        result = scanner.scan()
    except Exception:
        pass

2. **Use exception details**::

    try:
        stream.start()
    except StreamTimeoutError as e:
        timeout = e.details['timeout']
        logger.error(f"Stream timed out after {timeout}s")
        # Adjust timeout and retry
        stream.start(timeout=timeout * 2)

3. **Re-raise with context**::

    try:
        points = triangulate(left, right)
    except ValueError as e:
        raise ReconstructionError(
            stage='triangulation',
            reason=str(e)
        ) from e

4. **Log exceptions properly**::

    import logging
    logger = logging.getLogger(__name__)
    
    try:
        camera.configure(settings)
    except ConfigurationError as e:
        logger.error(
            "Configuration failed: %s",
            e.message,
            exc_info=True,  # Include traceback
            extra=e.details  # Log details
        )

Creating Custom Exceptions
-------------------------

If you need to create custom exceptions::

    class MyCustomError(UnlookException):
        '''Custom error for my module.'''
        
        def __init__(self, param: str, value: Any):
            message = f"Invalid {param}: {value}"
            super().__init__(
                message,
                details={'param': param, 'value': value}
            )

See Also
--------

- :doc:`logging_config` - Logging configuration
- :doc:`../user_guide/troubleshooting` - Troubleshooting guide
- Python exception documentation