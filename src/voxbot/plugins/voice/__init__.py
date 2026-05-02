"""Voice TTS plugin."""

import structlog

from disnake.ext import commands

from voxbot.plugins.voice.cog import VoiceCog

_LOGGER = structlog.get_logger(__name__)


def setup(bot: commands.Bot) -> None:
    """Plugin setup hook."""
    bot.add_cog(VoiceCog(bot))
    _LOGGER.info("voice_plugin_loaded")
