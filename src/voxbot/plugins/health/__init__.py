"""Voice TTS plugin."""

import structlog

from discord.ext import commands

from . import cog

_LOGGER = structlog.get_logger(__name__)


async def setup(bot: commands.Bot) -> None:
    """Setup the cog."""
    await bot.add_cog(cog.HealthCog(bot))
