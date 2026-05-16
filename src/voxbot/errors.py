"""Custom exceptions for voxbot."""

from discord.ext import commands


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


class RedisError(VoxBotError):
    """Redis error."""


# ── BOT ERRORS ────────────────────────────────────────────────────────────────────────


class VoxCheckFailure(commands.CheckFailure):
    """When a user fails any of our custom checks."""


class NotAnAdmin(VoxCheckFailure):
    """User is not an admin."""
