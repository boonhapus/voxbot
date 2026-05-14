"""Voice plugin runtime state and voice-name resolution helpers."""

import random

import pydantic

from voxbot.services.mistral import MistralService

VOICE_PREFIX = "en_paul_"
DEFAULT_VOICES = ["sad", "frustrated", "excited", "confident", "cheerful", "angry"]
MAX_AUTOCOMPLETE_RESULTS = 25


class VoxModel(pydantic.BaseModel):
    """In-memory state for voice activity and hero-trained voice origins."""

    hero_origins: dict[str, str] = pydantic.Field(default_factory=dict)
    last_active: dict[int, float] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(validate_assignment=True)


vox_model = VoxModel()
mistral_service = MistralService()


def resolve_voice(voice_name: str | None, custom_voices: dict[str, str]) -> str | None:
    """Resolve a voice name to a Mistral voice ID, defaulting to a random Paul variant."""
    if voice_name is None or voice_name == "Paul":
        return f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"

    return custom_voices.get(voice_name)


def list_voices(custom_voices: dict[str, str]) -> list[str]:
    """List voice names for autocomplete display, with Paul first."""
    custom = [v for v in custom_voices if not v.startswith(VOICE_PREFIX)]
    return ["Paul", *custom]


def filter_voices(voices: list[str], query: str) -> list[str]:
    """Filter voice names by case-insensitive substring, capped at the autocomplete limit."""
    if not query:
        return voices[:MAX_AUTOCOMPLETE_RESULTS]

    needle = query.lower()
    return [v for v in voices if needle in v.lower()][:MAX_AUTOCOMPLETE_RESULTS]
