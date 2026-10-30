"""Structured logging configuration using Loguru."""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from app.config import settings

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


def configure_application_logger():
    """Configures console and rotating file sinks for application logs."""
    logger.remove()
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger.add(sys.stderr, format=CONSOLE_FORMAT, level=settings.log_level, colorize=True)
    logger.add(
        logs_dir / "app.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format=FILE_FORMAT,
        level="INFO",
    )
    logger.add(
        logs_dir / "errors.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format=FILE_FORMAT,
        level="ERROR",
    )
    return logger


def get_logger(name: Optional[str] = None):
    """Returns a bound logger instance for the given module name."""
    return logger.bind(name=name) if name else logger


configure_application_logger()
