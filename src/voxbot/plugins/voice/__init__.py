from discord.ext import commands
import structlog

from . import cog, jobs  # noqa: F401

_LOGGER = structlog.get_logger(__name__)


async def setup(bot: commands.Bot) -> None:
    """Plugin setup hook."""
    # await bot.add_cog(cog.VoiceCog(bot))
