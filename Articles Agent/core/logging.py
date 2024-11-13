import sys
from loguru import logger


def setup_logging() -> None:
    """Configure logging for the application."""
    logger.remove()  # Remove default handler

    # Add custom logging format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level="INFO",
        colorize=True
    )

    # Add file handler for errors
    logger.add(
        "logs/error.log",
        format=log_format,
        level="ERROR",
        rotation="10 MB",
        compression="zip"
    )