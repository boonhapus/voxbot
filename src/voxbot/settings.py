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

    google_api_key: str = pydantic.Field(json_schema_extra={"mirror_to_os.environ": True})

    voc_model: str = "voxtral-mini-tts-2603"
    txt_model: str = "google-gla:gemini-2.5-flash-lite"
    debug_guild: str | None = None

    redis_url: str = "redis://localhost:6379/1"
    docket_url: str | None = None
    docket_name: str = "voxbot"
    docket_enabled: bool = True
    discord_owner_ids: str | None = None

    soul_home_guild_id: str | None = None
    soul_channel_id: str | None = None
    soul_name_check_interval_seconds: int = pydantic.Field(default=21600, gt=0)
    soul_memory_backend: str = "json"
    soul_memory_server_url: str = "http://localhost:8000"
    soul_memory_namespace: str = "voxbot:soul"
    soul_auto_extract_enabled: bool = False
    soul_auto_extract_channel_id: str | None = "1306464265703522325"
    soul_auto_extract_namespace: str = "voxbot:soul:auto-test"

    health_enabled: bool = True
    health_heartbeat_seconds: int = pydantic.Field(default=10, gt=0)
    deployment_id: str | None = None

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

        return self


settings = Settings()
