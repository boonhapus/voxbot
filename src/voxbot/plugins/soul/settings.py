import pydantic
import pydantic_settings


class SoulSettings(pydantic_settings.BaseSettings):
    """Soul plugin configuration loaded from .env."""

    channel_ids: list[int] = pydantic.Field(default_factory=list)

    memory_backend: str = "json"
    memory_server_url: str = "http://localhost:8000"
    memory_namespace: str = "voxbot:soul"

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="SOUL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


soul_settings = SoulSettings()
