from typing import Self
import os

import pydantic_settings
import pydantic

import structlog

_LOGGER = structlog.get_logger(__name__)


class Settings(pydantic_settings.BaseSettings):
    """Bot configuration loaded from .env file."""

    discord_token: str = pydantic.Field(json_schema_extra={"mirror_to_os.environ": True})
    """https://discord.com/developers/applications/1487888280061083708/bot"""

    mistral_api_key: str = pydantic.Field(json_schema_extra={"mirror_to_os.environ": True})
    """https://console.mistral.ai/home?profile_dialog=api-keys"""

    ollama_base_url: str = "http://169.254.83.107:11434/v1"
    """Ollama server URL (mapped to OPENAI_BASE_URL for pydantic-ai)."""

    mistral_model: str = "voxtral-mini-tts-2603"
    text_model: str = "openai:qwen3.5:4b"
    debug_guild: str | None = None

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @pydantic.model_validator(mode="after")
    def export_to_environ(self) -> Self:
        """Sync loaded settings to os.environ for downstream libraries."""
        for name, field in Settings.model_fields.items():
            schema_is_empty = field.json_schema_extra is None
            schema_is_typed = isinstance(field.json_schema_extra, dict)
            schema_mirrored = (field.json_schema_extra or {}).get("mirror_to_os.environ", False)

            if schema_is_empty or not schema_is_typed or not schema_mirrored:
                continue

            val = getattr(self, name)

            if hasattr(val, "get_secret_value"):
                val = val.get_secret_value()

            if os.getenv(env_key := name.upper()) != str(val):
                os.environ[env_key] = str(val)

        os.environ["OPENAI_BASE_URL"] = self.ollama_base_url
        os.environ["OPENAI_API_KEY"] = "ollama"

        return self


settings = Settings()
