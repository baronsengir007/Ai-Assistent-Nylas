"""
Pydantic Settings Quickstart

This example demonstrates how to:
1. Define application settings using Pydantic
2. Load configuration from environment variables
3. Use type validation and default values
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application configuration settings.

    Automatically reads from environment variables matching the field names.
    """

    database_url: str

    # OpenAI settings
    openai_api_key: str
    openai_model: str = "gpt-4o"
    temperature: float = 0.5

    # Application settings
    debug: bool = False
    max_retries: int = 3
    timeout: float = 10.0

    class Config:
        case_sensitive = False
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: The application settings.
    """
    return Settings()


# Create a global settings instance
settings = Settings()


def main():
    try:
        # Access settings as normal Python attributes
        print("\nCurrent Settings:")
        print(f"OpenAI API Key: {settings.openai_api_key[:5]}...")
        print(f"Model: {settings.openai_model}")
        print(f"Temperature: {settings.temperature}")
        print(f"Debug Mode: {settings.debug}")

    except Exception as e:
        print(f"\nError loading settings: {e}")
        print("\nMake sure you have a .env file with required variables:")


if __name__ == "__main__":
    main()
