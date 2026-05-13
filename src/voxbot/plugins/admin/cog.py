from discord.ext import commands
from discord import app_commands
import discord
import structlog

from voxbot import checks

from . import schema

_LOGGER = structlog.get_logger(__name__)


class AdminCog(commands.GroupCog, name="admin"):
    """Owner-only deployment and health commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    async def cog_load(self) -> None:
        """Called when cog is loaded."""
        _LOGGER.info("admin_cog_loaded")

    # ── COMMANDS ──────────────────────────────────────────────────────────────────────

    @app_commands.command(name="health", description="Show bot deployment health")
    @checks.is_bot_admin()
    async def admin_health(self, interaction: discord.Interaction):
        """Fetch the health of the bot."""
        await interaction.response.defer(thinking=True, ephemeral=True)

        report = await schema.HealthReport.from_redis()
        await interaction.edit_original_response(content=str(report))

    @app_commands.command(name="restart", description="Restart the bot process")
    @checks.is_bot_admin()
    @app_commands.describe(reason="Optional reason to record in health state")
    async def admin_restart(self, interaction: discord.Interaction, reason: str | None = None):
        """Ad-hoc restart the bot."""
        restart_reason = reason or f"Requested by {interaction.user.name} ({interaction.user.id})"

        await interaction.response.send_message(f"🚀 Restarting: {restart_reason}", ephemeral=True)
        await self.bot.request_restart(reason=restart_reason)  # type: ignore
