"""
Standardized logging configuration for UnLook client.

This module provides consistent logging setup across all client modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from unlook.core.constants import LOG_FORMAT, MAX_LOG_FILE_SIZE, DEBUG_DIR_PREFIX


class UnlookLogger:
    """Standardized logger configuration for UnLook SDK."""
    
    # Standard log levels for different scenarios
    LEVELS = {
        'DEBUG': logging.DEBUG,      # Detailed debugging info
        'INFO': logging.INFO,        # General informational messages
        'WARNING': logging.WARNING,  # Warning messages
        'ERROR': logging.ERROR,      # Error messages
        'CRITICAL': logging.CRITICAL # Critical errors
    }
    
    # Module-specific default levels
    MODULE_LEVELS = {
        'unlook.client.camera': logging.INFO,
        'unlook.client.scanner': logging.INFO,
        'unlook.client.streaming': logging.WARNING,  # Less verbose for streaming
        'unlook.client.projector': logging.INFO,
        'unlook.client.scanning.handpose': logging.WARNING,  # Less verbose for handpose
        'unlook.client.scanning.reconstruction': logging.INFO,
        'unlook.core': logging.WARNING,
        'unlook.server': logging.INFO,
    }
    
    @classmethod
    def setup_logging(
        cls,
        level: str = 'INFO',
        log_file: Optional[str] = None,
        console: bool = True,
        module_levels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Setup standardized logging configuration.
        
        Args:
            level: Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            console: Whether to log to console
            module_levels: Optional module-specific log levels
        """
        # Convert string level to logging constant
        numeric_level = cls.LEVELS.get(level.upper(), logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Create log directory if needed
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler to prevent huge log files
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_FILE_SIZE,
                backupCount=3
            )
            file_handler.setLevel(logging.DEBUG)  # File gets everything
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
        
        # Set module-specific levels
        for module, default_level in cls.MODULE_LEVELS.items():
            module_logger = logging.getLogger(module)
            module_logger.setLevel(default_level)
        
        # Apply custom module levels if provided
        if module_levels:
            for module, level_str in module_levels.items():
                module_logger = logging.getLogger(module)
                module_level = cls.LEVELS.get(level_str.upper(), logging.INFO)
                module_logger.setLevel(module_level)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance with the given name.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    @classmethod
    def setup_debug_logging(cls, session_name: Optional[str] = None) -> str:
        """
        Setup debug logging with automatic session directory.
        
        Args:
            session_name: Optional session name
            
        Returns:
            Path to debug log file
        """
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = session_name or f"session_{timestamp}"
        debug_dir = Path.home() / '.unlook' / 'logs' / session
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with debug file
        log_file = debug_dir / 'debug.log'
        cls.setup_logging(
            level='DEBUG',
            log_file=str(log_file),
            console=True
        )
        
        # Log session start
        logger = logging.getLogger('unlook.client')
        logger.info(f"Debug session started: {session}")
        logger.debug(f"Debug logs: {log_file}")
        
        return str(log_file)


# Convenience functions
def setup_logging(**kwargs) -> None:
    """Setup logging with default configuration."""
    UnlookLogger.setup_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return UnlookLogger.get_logger(name)


def debug_mode(session_name: Optional[str] = None) -> str:
    """Enable debug mode with full logging."""
    return UnlookLogger.setup_debug_logging(session_name)


# Usage guidelines as constants
LOGGING_GUIDELINES = """
Logging Level Guidelines:

1. DEBUG: 
   - Detailed diagnostic information
   - Variable values, function entry/exit
   - Performance metrics
   - Not shown to users by default

2. INFO:
   - General operational messages
   - Progress updates
   - Successful operations
   - Shown to users by default

3. WARNING:
   - Recoverable issues
   - Deprecated functionality
   - Performance concerns
   - Suboptimal configurations

4. ERROR:
   - Failed operations that can be retried
   - Missing optional dependencies
   - Invalid user input
   - Exceptions that are handled

5. CRITICAL:
   - System failures
   - Unrecoverable errors
   - Data corruption risks
   - Security issues

Best Practices:
- Use logger.debug() for developer info
- Use logger.info() for user-facing progress
- Use logger.warning() for issues users should know about
- Use logger.error() with exc_info=True for exceptions
- Use logger.critical() sparingly for severe issues
"""