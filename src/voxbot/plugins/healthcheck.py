import discord
from discord import app_commands
from discord.ext import commands
import structlog

_LOGGER = structlog.get_logger(__name__)


class HealthcheckCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)

        latency_ms = self.bot.latency * 1000
        await interaction.edit_original_response(content=f"Pong! **{latency_ms:.0f}ms**")

        _LOGGER.info("ping_command_used", user=interaction.user.display_name, latency=latency_ms)

    def cog_load(self):
        _LOGGER.info("healthcheck_plugin_loaded")


async def setup(bot):
    await bot.add_cog(HealthcheckCog(bot))
