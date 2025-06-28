import logging
import sys
from pathlib import Path
from typing import Optional


def create_logger(
    name: str, log_level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name (str): The name for the logger, typically __name__ of the
            calling module.
        log_level (int): The minimum logging level to be processed (e.g.,
            logging.INFO).
        log_file (Optional[str]): Path to the log file. If provided, logs
            will also be written to this file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Always add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add a file handler if a log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
