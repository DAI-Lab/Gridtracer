import logging
import sys
from pathlib import Path


def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Set up a centralized logger for the application.

    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file (str, optional): Path to log file. If None, no file logging is set up.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if needed and log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create handlers list - always include console handler
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if log_file is specified
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Get the root logger or create a new one
    logger = logging.getLogger('gridtracer')

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers = []
        for handler in handlers:
            logger.addHandler(handler)

    return logger


# Create a default logger instance at module level
logger = setup_logger()
