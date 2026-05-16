from typing import Any, Literal

import pydantic
import pydantic_settings


class SoulSettings(pydantic_settings.BaseSettings):
    """Soul plugin configuration loaded from .env."""

    channel_ids: list[int] = pydantic.Field(default_factory=list)

    memory_backend: Literal["json", "redis"] = "json"
    memory_server_url: str = "http://localhost:8000"
    memory_namespace: str = "voxbot:soul"
    memory_embedding_model: str = "gemini/text-embedding-004"
    memory_semantic_relevance: Literal["loose", "balanced", "strict"] = "balanced"

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="SOUL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @pydantic.field_validator("memory_backend", "memory_semantic_relevance", mode="before")
    def _sanitize(cls, value: Any) -> str:
        return str(value).strip().casefold()


soul_settings = SoulSettings()
