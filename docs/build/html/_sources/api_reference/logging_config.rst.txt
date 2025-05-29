Logging Configuration
====================

The logging configuration module provides standardized logging setup for the UnLook SDK.

.. module:: unlook.client.logging_config

Overview
--------

The logging system provides:

- Consistent log formatting across all modules
- Module-specific log levels
- File and console output options
- Debug mode with automatic session management
- Log rotation to prevent huge files

Classes
-------

.. autoclass:: UnlookLogger
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: setup_logging
   .. automethod:: get_logger
   .. automethod:: setup_debug_logging

Quick Start
-----------

Basic setup::

    from unlook.client.logging_config import setup_logging
    
    # Setup with default INFO level
    setup_logging()
    
    # Setup with DEBUG level and file output
    setup_logging(
        level='DEBUG',
        log_file='unlook.log',
        console=True
    )

Get a logger in your module::

    from unlook.client.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Starting operation...")
    logger.debug("Detailed information here")

Debug Mode
----------

Enable debug mode for troubleshooting::

    from unlook.client.logging_config import debug_mode
    
    # Start debug session with automatic file management
    log_file = debug_mode("my_debug_session")
    print(f"Debug logs: {log_file}")

Debug logs are saved to: ``~/.unlook/logs/<session_name>/debug.log``

Module-Specific Levels
---------------------

Default log levels by module:

- ``unlook.client.camera``: INFO
- ``unlook.client.scanner``: INFO  
- ``unlook.client.streaming``: WARNING (less verbose)
- ``unlook.client.projector``: INFO
- ``unlook.client.scanning.handpose``: WARNING (less verbose)
- ``unlook.client.scanning.reconstruction``: INFO
- ``unlook.core``: WARNING
- ``unlook.server``: INFO

Override module levels::

    setup_logging(
        level='INFO',
        module_levels={
            'unlook.client.streaming': 'DEBUG',
            'unlook.client.camera': 'WARNING'
        }
    )

Logging Guidelines
-----------------

Use appropriate log levels:

**DEBUG**
  - Detailed diagnostic information
  - Variable values, function entry/exit
  - Performance metrics
  - Not shown to users by default

**INFO**
  - General operational messages
  - Progress updates
  - Successful operations
  - Shown to users by default

**WARNING**
  - Recoverable issues
  - Deprecated functionality
  - Performance concerns
  - Suboptimal configurations

**ERROR**
  - Failed operations that can be retried
  - Missing optional dependencies
  - Invalid user input
  - Handled exceptions

**CRITICAL**
  - System failures
  - Unrecoverable errors
  - Data corruption risks
  - Security issues

Best Practices
--------------

1. Use structured logging::

    logger.info("Captured image", extra={
        'camera_id': 'left',
        'resolution': (1920, 1080),
        'exposure': 10000
    })

2. Include context in errors::

    try:
        result = camera.capture()
    except Exception as e:
        logger.error(
            "Camera capture failed",
            exc_info=True,  # Include traceback
            extra={'camera_id': camera_id}
        )

3. Use lazy formatting::

    # Good - formatting only if logged
    logger.debug("Processing %d points", len(points))
    
    # Bad - always formats
    logger.debug(f"Processing {len(points)} points")

4. Avoid logging in tight loops::

    # Log summary instead
    logger.info("Processing %d images", len(images))
    for i, img in enumerate(images):
        process(img)
        if i % 100 == 0:  # Progress every 100
            logger.debug("Processed %d/%d", i, len(images))

Configuration File
-----------------

You can also configure logging via environment variables:

- ``UNLOOK_LOG_LEVEL``: Default log level
- ``UNLOOK_LOG_FILE``: Log file path
- ``UNLOOK_DEBUG``: Enable debug mode

Example::

    export UNLOOK_LOG_LEVEL=DEBUG
    export UNLOOK_LOG_FILE=~/unlook_debug.log
    python my_script.py

See Also
--------

- Python logging documentation
- :doc:`exceptions` - Custom exception handling
- :doc:`../user_guide/troubleshooting` - Debugging guide