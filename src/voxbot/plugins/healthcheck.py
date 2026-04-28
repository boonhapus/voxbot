from disnake.ext import commands
import structlog

_LOGGER = structlog.get_logger(__name__)


class HealthcheckCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(name="ping", description="Check bot latency")
    async def ping(self, inter):
        await inter.response.defer(sending_message=False)

        latency_ms = self.bot.latency * 1000
        await inter.edit_original_response(content=f"Pong! **{latency_ms:.0f}ms**")

        _LOGGER.info("ping_command_used", user=inter.author.display_name, latency=latency_ms)

    def cog_load(self):
        _LOGGER.info("healthcheck_plugin_loaded")


def setup(bot):
    bot.add_cog(HealthcheckCog(bot))
