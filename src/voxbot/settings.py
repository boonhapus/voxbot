import os

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import structlog

_LOGGER = structlog.get_logger(__name__)


class Settings(BaseSettings):
    """Bot configuration loaded from .env file."""

    discord_token: str
    mistral_api_key: str
    mistral_model: str = "voxtral-mini-tts-2603"
    debug_guild: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("discord_token", "mistral_api_key", mode="before")
    @classmethod
    def require_non_empty(cls, v):
        if not v:
            raise ValueError("must be non-empty")
        return v

    @model_validator(mode="after")
    def sync_to_environ(self):
        """Sync loaded settings to os.environ for downstream libraries."""
        os.environ["DISCORD_TOKEN"] = self.discord_token
        os.environ["MISTRAL_API_KEY"] = self.mistral_api_key
        os.environ["MISTRAL_MODEL"] = self.mistral_model
        if self.debug_guild:
            os.environ["DEBUG_GUILD"] = self.debug_guild
        return self
