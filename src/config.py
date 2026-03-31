"""
Configuration management for the ShopVite FAQ Assistant.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""

    # LLM Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Application Settings
    APP_NAME = os.getenv("APP_NAME", "ShopVite FAQ Assistant")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    K_RETRIEVALS = int(os.getenv("K_RETRIEVALS", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))

    # Vector Store
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
    PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")

    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Data Directory
    DATA_DIR = os.getenv("DATA_DIR", "./data/documents")

    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration values."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return True


# Initialize configuration
config = Config()
config.validate()
