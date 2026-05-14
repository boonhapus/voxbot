"""Mistral API client wrapper."""

import json
import pathlib
from typing import Annotated

from mistralai.client import Mistral
import structlog

from voxbot.errors import MistralError
from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)

type VoiceID = Annotated[str, "Mistral voice ID"]
DEFAULT_VOICES_FILE = pathlib.Path.home() / ".voxbot" / "voices.json"


class VoiceData:
    """In-memory voice state: custom voices, deleted voices, hero origins."""

    def __init__(
        self,
        custom_voices: dict[str, str] | None = None,
        deleted_voices: set[str] | None = None,
        hero_origins: dict[str, str] | None = None,
    ):
        self.custom_voices = custom_voices or {}
        self.deleted_voices = deleted_voices or set()
        self.hero_origins = hero_origins or {}

    def to_dict(self) -> dict:
        return {
            "voices": self.custom_voices,
            "deleted": list(self.deleted_voices),
            "hero_origins": self.hero_origins,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceData":
        return cls(
            custom_voices=data.get("voices", {}),
            deleted_voices=set(data.get("deleted", [])),
            hero_origins=data.get("hero_origins", {}),
        )


class MistralService:
    """Wraps Mistral API and manages voice persistence."""

    def __init__(self, voices_file: pathlib.Path = DEFAULT_VOICES_FILE):
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.voices_file = voices_file
        self.voices = VoiceData()
        self._load_voices()

    def _load_voices(self) -> None:
        if not self.voices_file.exists():
            return
        try:
            with open(self.voices_file) as f:
                data = json.load(f)
            self.voices = VoiceData.from_dict(data)
            _LOGGER.info(
                "voices_loaded",
                count=len(self.voices.custom_voices),
                deleted=len(self.voices.deleted_voices),
            )
        except Exception as exc:
            _LOGGER.error("voices_load_failed", error=str(exc))

    def save_voices(self) -> None:
        self.voices_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.voices_file, "w") as f:
                json.dump(self.voices.to_dict(), f, indent=2)
            _LOGGER.info(
                "voices_saved",
                count=len(self.voices.custom_voices),
                deleted=len(self.voices.deleted_voices),
            )
        except Exception as exc:
            _LOGGER.error("voices_save_failed", error=str(exc))

    async def sync_voices(self) -> dict[str, str]:
        """Fetch available voices from the Mistral API and persist any new ones."""
        try:
            response = await self.client.audio.voices.list_async()
        except Exception as exc:
            _LOGGER.error("voices_sync_failed", error=str(exc))
            raise MistralError(f"failed to sync voices: {exc}") from exc

        added = {}
        for voice in response.items:
            if voice.name in self.voices.deleted_voices:
                continue
            if voice.name not in self.voices.custom_voices:
                self.voices.custom_voices[voice.name] = voice.id
                added[voice.name] = voice.id

        if added:
            self.save_voices()
            _LOGGER.info("voices_synced", added=list(added.keys()))
        return added

    async def train_voice(
        self, name: str, audio_b64: str, sample_filename: str
    ) -> VoiceID:
        """Train a new custom voice on the Mistral API and persist it locally."""
        try:
            voice = await self.client.audio.voices.create_async(
                name=name,
                sample_audio=audio_b64,
                sample_filename=sample_filename,
            )
        except Exception as exc:
            _LOGGER.error("voice_train_failed", error=str(exc), name=name)
            raise MistralError(f"voice training failed: {exc}") from exc

        self.voices.custom_voices[name] = voice.id
        self.save_voices()
        return voice.id

    async def delete_voice(self, voice_id: VoiceID) -> None:
        """Delete a voice from Mistral."""
        try:
            await self.client.audio.voices.delete_async(voice_id=voice_id)
        except Exception as exc:
            _LOGGER.error("voice_delete_failed", error=str(exc), voice_id=voice_id)
            raise MistralError(f"voice deletion failed: {exc}") from exc

    async def text_to_speech(self, text: str, voice_id: VoiceID) -> str:
        """Generate TTS audio from text using the Mistral API."""
        try:
            response = await self.client.audio.speech.complete_async(
                model=settings.voc_model,
                input=text,
                voice_id=voice_id,
                response_format="mp3",
            )
        except Exception as exc:
            _LOGGER.error("tts_failed", error=str(exc))
            raise MistralError(f"TTS generation failed: {exc}") from exc

        return response.audio_data
