from voxbot.bot import VoxBot

from . import cog, jobs  # noqa: F401


async def setup(bot: VoxBot) -> None:
    """Plugin setup hook."""
    # await bot.add_cog(cog.VoiceCog(bot))
