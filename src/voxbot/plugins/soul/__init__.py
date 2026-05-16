from voxbot.bot import VoxBot

from . import cog, jobs  # noqa: F401


async def setup(bot: VoxBot) -> None:
    """Setup the cog."""
    await bot.add_cog(cog.SoulCog(bot))
