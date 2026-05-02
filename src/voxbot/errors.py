"""Custom exceptions for voxbot."""


class VoxBotError(Exception):
    """Base exception for all voxbot errors."""


class MistralError(VoxBotError):
    """Mistral API error."""


class VoiceNotFoundError(MistralError):
    """Requested voice does not exist."""


class TTSError(VoxBotError):
    """Text-to-speech generation error."""


class WikiError(VoxBotError):
    """Dota wiki scraping error."""
