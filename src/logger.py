"""
Centralized logging configuration for the ShopVite FAQ Assistant.
Provides consistent logging across all modules.
"""

import logging
import sys
from src.config import config

# Configure logger
logger = logging.getLogger(config.APP_NAME)
logger.setLevel(getattr(logging, config.LOG_LEVEL))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)

# Add handler to logger
if not logger.handlers:
    logger.addHandler(console_handler)

__all__ = ["logger"]
