"""
This module sets up the logger configuration.

It utilizes Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file.
"""

from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """
    Logger configuration settings for the application.

    Attributes:
        model_config (SettingsConfigDict): Model config, loaded from .env file.
        log_level (str): Logging level for the application.
        log_rotation (str): Log rotation policy.
        log_retention (str): Log retention policy.
        log_compression (str): Log compression format.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    log_rotation: str = "1 day"
    log_retention: str = "2 days"
    log_compression: str = "zip"


def configure_logging(log_level: str | None = None) -> None:
    """
    Configure the logging for the application.

    Args:
        log_level (str, optional): The log level to be set for the logger.
                                  If None, uses the LoggerSettings default.
    """
    logger_settings = LoggerSettings()
    
    if log_level is None:
        log_level = logger_settings.log_level

    # Configure logs directory with proper path resolution
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        str(logs_dir / "app.log"),
        rotation=logger_settings.log_rotation,
        retention=logger_settings.log_retention,
        compression=logger_settings.log_compression,
        level=log_level,
    )


# Initialize logging configuration
logger_settings = LoggerSettings()
configure_logging(logger_settings.log_level)
