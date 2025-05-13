"""
Configuration settings for the LLM Ethics Service.

This module provides configuration settings for the LLM ethics service,
including API keys, model settings, caching parameters, and other
service-specific configurations.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field, validator

class Settings(BaseSettings):
    """
    Configuration settings for the LLM ethics service.
    
    Uses Pydantic for validation and environment variable loading.
    """
    # Service settings
    SERVICE_NAME: str = "LLM Ethics Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(False, env="ETHICS_SERVICE_DEBUG")
    
    # Server settings
    HOST: str = Field("0.0.0.0", env="ETHICS_SERVICE_HOST")
    PORT: int = Field(8000, env="ETHICS_SERVICE_PORT")
    
    # API settings
    API_PREFIX: str = "/api/v1"
    
    # LLM API settings
    CLAUDE_API_KEY: Optional[str] = Field(None, env="CLAUDE_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Default LLM model settings
    DEFAULT_LLM_PROVIDER: str = Field("claude", env="DEFAULT_LLM_PROVIDER")
    CLAUDE_MODEL: str = Field("claude-3-opus-20240229", env="CLAUDE_MODEL")
    GPT_MODEL: str = Field("gpt-4o", env="GPT_MODEL")
    
    # Caching settings
    CACHE_ENABLED: bool = Field(True, env="ETHICS_CACHE_ENABLED")
    CACHE_TTL: int = Field(3600, env="ETHICS_CACHE_TTL")  # Time to live in seconds
    CACHE_MAX_SIZE: int = Field(1000, env="ETHICS_CACHE_MAX_SIZE")
    
    # Logging settings
    LOG_LEVEL: str = Field("INFO", env="ETHICS_LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Ethics evaluation settings
    ETHICS_CATEGORIES: List[str] = [
        "privacy", 
        "security", 
        "fairness", 
        "transparency", 
        "accountability", 
        "gurbani_alignment"
    ]
    
    ETHICS_PASSING_THRESHOLD: float = Field(0.7, env="ETHICS_PASSING_THRESHOLD")
    GURBANI_ALIGNMENT_THRESHOLD: float = Field(0.8, env="GURBANI_ALIGNMENT_THRESHOLD")
    
    # Path to Gurbani principles file
    GURBANI_PRINCIPLES_PATH: str = Field(
        "../../../guidance/design_principles.md", 
        env="GURBANI_PRINCIPLES_PATH"
    )
    
    @validator("DEFAULT_LLM_PROVIDER")
    def validate_llm_provider(cls, v):
        """Validate that the LLM provider is supported."""
        if v not in ["claude", "gpt", "local"]:
            raise ValueError(f"Unsupported LLM provider: {v}")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True

# Create a global settings instance
settings = Settings()