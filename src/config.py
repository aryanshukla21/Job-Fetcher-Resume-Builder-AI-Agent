"""
Configuration management for the AI Job Agent.
Fails fast if required environment variables are missing.
"""
from __future__ import annotations
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class AppSettings(BaseSettings):
    # LLM Provider Configuration
    llm_provider: str = "anthropic"  # 'anthropic', 'openai', or 'google'
    tier1_model: str = "claude-3-haiku-20240307"
    tier2_model: str = "claude-3-sonnet-20240229"
    tier3_model: str = "claude-3-opus-20240229"

    # API Keys (Validated based on provider)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Tool Keys
    apify_api_token: str
    github_token: str
    
    # LangSmith Observability
    langchain_tracing_v2: str = "true"
    langchain_api_key: str
    langchain_project: str = "job-agent-dev"

    # Database URLs
    database_url: str
    state_database_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> AppSettings:
        provider = self.llm_provider.lower()
        if provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("anthropic_api_key is required when llm_provider is 'anthropic'")
        if provider == "openai" and not self.openai_api_key:
            raise ValueError("openai_api_key is required when llm_provider is 'openai'")
        if provider == "google" and not self.google_api_key:
            raise ValueError("google_api_key is required when llm_provider is 'google'")
        return self

settings = AppSettings()