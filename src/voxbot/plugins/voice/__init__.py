"""Voice TTS plugin."""

import structlog

from discord.ext import commands

from voxbot.plugins.voice.cog import VoiceCog

_LOGGER = structlog.get_logger(__name__)


async def setup(bot: commands.Bot) -> None:
    """Plugin setup hook."""
    await bot.add_cog(VoiceCog(bot))
    _LOGGER.info("voice_plugin_loaded")
