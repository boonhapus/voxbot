from typing import Self
import os

import pydantic_settings
import pydantic
import discord

import structlog

_LOGGER = structlog.get_logger(__name__)


class Settings(pydantic_settings.BaseSettings):
    """Bot configuration loaded from .env file."""

    # ── SECRETS ───────────────────────────────────────────────────────────────────────

    discord_token: str = pydantic.Field(repr=False, json_schema_extra={"mirror_to_os.environ": True})
    """https://discord.com/developers/applications/1487888280061083708/bot"""

    mistral_api_key: str = pydantic.Field(repr=False, json_schema_extra={"mirror_to_os.environ": True})
    """https://console.mistral.ai/home?profile_dialog=api-keys"""

    google_api_key: str = pydantic.Field(repr=False, json_schema_extra={"mirror_to_os.environ": True})
    """https://aistudio.google.com/projects?project=gen-lang-client-0686028511"""


    # ── AI SETTINGS ───────────────────────────────────────────────────────────────────

    voc_model: str = "voxtral-mini-tts-2603"
    txt_model: str = "google-gla:gemini-2.5-flash-lite"


    # ── METADATA ──────────────────────────────────────────────────────────────────────

    debug_guild: int | None = None
    bot_owner_id: int


    # ── METADATA ──────────────────────────────────────────────────────────────────────

    voxbot_release_sha: str | None = None


    # ── DURABLE STORE ─────────────────────────────────────────────────────────────────

    redis_url: str = pydantic.Field(default="redis://localhost:6379/1", repr=False)


    # ── FEATURE FLAGS ─────────────────────────────────────────────────────────────────

    soul_home_guild_id: str | None = None
    soul_channel_ids: list[str] = pydantic.Field(default_factory=list)
    soul_name_check_interval_seconds: int = pydantic.Field(default=21600, gt=0)
    soul_memory_backend: str = "json"
    soul_memory_server_url: str = "http://localhost:8000"
    soul_memory_namespace: str = "voxbot:soul"
    soul_auto_extract_enabled: bool = False
    soul_auto_extract_channel_id: str | None = "1306464265703522325"
    soul_auto_extract_namespace: str = "voxbot:soul:auto-test"


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
    
    @property
    def required_intents(self) -> discord.Intents:
        """
        The bot permissions in order to run.
        
        Further reading:
          https://discord.com/developers/applications/1487888280061083708/bot
          https://support-dev.discord.com/hc/en-us/articles/6207308062871-What-are-Privileged-Intents
          https://discordpy.readthedocs.io/en/latest/intents.html
        """
        # DEV NOTE:
        #   We used to be selective about intents, but then we realized all the cool
        #   interactivity was behind all the privileged intents anyway.
        return discord.Intents.all()


settings = Settings()
