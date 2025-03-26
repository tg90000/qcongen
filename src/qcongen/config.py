"""Configuration management for QConGen."""

import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    """Configuration class for QConGen."""

    def __init__(self, ibm_token: str | None = None) -> None:
        """Initialize configuration.
        
        Args:
            ibm_token: Optional IBM Quantum token. If not provided,
                      will try to load from environment or .env file
        """
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

        # Set IBM token with priority:
        # 1. Explicitly provided token
        # 2. Environment variable
        # 3. None (will raise error when trying to use quantum features)
        self.ibm_token: str | None = (
            ibm_token or 
            os.getenv("IBM_TOKEN")
        )

    @property
    def has_quantum_access(self) -> bool:
        """Check if quantum access is configured."""
        return bool(self.ibm_token)


# Global config instance
config = Config()


def initialize_config(ibm_token: str | None = None) -> None:
    """Initialize global configuration.
    
    Args:
        ibm_token: Optional IBM Quantum token
    """
    global config
    config = Config(ibm_token=ibm_token)
