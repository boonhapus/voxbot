"""Voice connection state management."""

import random

VOICE_PREFIX = "en_paul_"
DEFAULT_VOICES = ["sad", "frustrated", "excited", "confident", "cheerful", "angry"]


class VoiceState:
    """Resolves and manages voice selection."""

    @staticmethod
    def resolve_voice(
        voice_name: str | None, custom_voices: dict[str, str]
    ) -> str | None:
        """
        Resolve a voice name to a voice ID.
        If voice_name is "Paul" or None, pick a random default voice.
        Otherwise look it up in custom_voices.
        Returns None if voice not found.
        """
        if voice_name is None or voice_name == "Paul":
            return f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"
        return custom_voices.get(voice_name)

    @staticmethod
    def get_voice_list(custom_voices: dict[str, str]) -> list[str]:
        """Return list of available voices: 'Paul' + custom voices."""
        custom = [v for v in custom_voices.keys() if not v.startswith(VOICE_PREFIX)]
        return ["Paul", *custom]

    @staticmethod
    def filter_voice_list(voices: list[str], query: str) -> list[str]:
        """Filter voice list by query string, limit to 25."""
        if not query:
            return voices[:25]
        filtered = [v for v in voices if query.lower() in v.lower()]
        return filtered[:25]
