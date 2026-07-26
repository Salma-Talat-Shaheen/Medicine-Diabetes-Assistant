"""Configuration settings for the Medicine Assistant."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded dynamically from environment variables."""

    @property
    def OPENROUTER_API_KEY(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def OPENROUTER_BASE_URL(self) -> str:
        return "https://openrouter.ai/api/v1"

    @property
    def MODEL_NAME(self) -> str:
        return os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

    @property
    def CHUNK_SIZE(self) -> int:
        return int(os.getenv("CHUNK_SIZE", "1000"))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(os.getenv("CHUNK_OVERLAP", "200"))

    @property
    def TOP_K_RESULTS(self) -> int:
        return int(os.getenv("TOP_K_RESULTS", "5"))

    @property
    def CHROMA_PERSIST_DIRECTORY(self) -> str:
        return os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    @property
    def COLLECTION_NAME(self) -> str:
        return os.getenv("COLLECTION_NAME", "medicine_docs")

    def validate(self) -> None:
        """Validate that required settings are configured."""
        if not self.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )


settings = Settings()
