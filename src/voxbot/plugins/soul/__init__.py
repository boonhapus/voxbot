from discord.ext import commands
import structlog

from . import cog

_LOGGER = structlog.get_logger(__name__)


async def setup(bot: commands.Bot) -> None:
    """Setup the cog."""
    await bot.add_cog(cog.SoulCog(bot))
