from voxbot.bot import VoxBot

from . import cog


async def setup(bot: VoxBot) -> None:
    """Setup the cog."""
    await bot.add_cog(cog.HealthCog(bot))
