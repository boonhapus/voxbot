from discord.ext import commands
from discord import app_commands
import discord
import structlog

_LOGGER = structlog.get_logger(__name__)


class HealthCog(commands.GroupCog, name="health"):
    """Health plugin cog."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    async def cog_load(self) -> None:
        """Called when cog is loaded."""
        _LOGGER.info("health_cog_loaded")

    # ── COMMANDS ──────────────────────────────────────────────────────────────────────

    @app_commands.command(name="ping", description="Check bot latency")
    async def health_ping(self, interaction: discord.Interaction) -> None:
        """Measure the latency of the Bot."""
        await interaction.response.defer(thinking=True)

        latency_ms = self.bot.latency * 1000

        await interaction.edit_original_response(content=f"Pong! **{latency_ms:.0f}ms**")

        _LOGGER.info("ping_command_used", user=interaction.user.display_name, latency=latency_ms)
