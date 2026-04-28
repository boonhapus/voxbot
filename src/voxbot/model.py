import json
import os
import pathlib
from mistralai.client import Mistral
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import structlog

_LOGGER = structlog.get_logger(__name__)

DEFAULT_VOICES_FILE = pathlib.Path.home() / ".voxbot" / "voices.json"


class Config(BaseModel):
    """Bot configuration."""

    discord_token: str
    mistral_api_key: str
    mistral_model: str = "voxtral-mini-tts-2603"
    voices_file: pathlib.Path = DEFAULT_VOICES_FILE

    model_config = ConfigDict(frozen=True)


class VoxModel(BaseModel):
    """Shared model holding bot state and API clients."""

    config: Config
    mistral: Mistral | None = None
    custom_voices: dict[str, str] = {}
    deleted_voices: set[str] = set()
    voice_client: object | None = None
    last_active: dict[int, float] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="after")
    def initialize_mistral(self):
        if self.mistral is None:
            self.mistral = Mistral(api_key=self.config.mistral_api_key)
            self._load_voices()
        return self

    def _load_voices(self) -> None:
        """Load voice name→ID mappings from disk."""
        vf = self.config.voices_file
        if not vf.exists():
            return
        try:
            with open(vf, "r") as f:
                data = json.load(f)
            self.custom_voices.update(data.get("voices", {}))
            self.deleted_voices = set(data.get("deleted", []))
            _LOGGER.info("voices_loaded", count=len(self.custom_voices), deleted=len(self.deleted_voices))
        except Exception as err:
            _LOGGER.error("voices_load_failed", error=str(err))

    def save_voices(self) -> None:
        """Persist voice name→ID mappings to disk."""
        vf = self.config.voices_file
        vf.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(vf, "w") as f:
                json.dump(
                    {"voices": self.custom_voices, "deleted": list(self.deleted_voices)},
                    f,
                    indent=2,
                )
            _LOGGER.info("voices_saved", count=len(self.custom_voices), deleted=len(self.deleted_voices))
        except Exception as err:
            _LOGGER.error("voices_save_failed", error=str(err))

    async def sync_voices(self) -> dict[str, str]:
        """Sync local mappings with Mistral API; returns newly added voices."""
        try:
            response = await self.mistral.audio.voices.list_async()
        except Exception as err:
            _LOGGER.error("voices_sync_failed", error=str(err))
            return {}

        added = {}
        for voice in response.items:
            if voice.name in self.deleted_voices:
                continue
            if voice.name not in self.custom_voices:
                self.custom_voices[voice.name] = voice.id
                added[voice.name] = voice.id

        if added:
            self.save_voices()
            _LOGGER.info("voices_synced", added=list(added.keys()))
        return added
