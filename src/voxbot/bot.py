import asyncio
import importlib.metadata
import pathlib

from discord.ext import commands
import discord
import structlog

from voxbot.runtime.docket import BotDocketRuntime
from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)


class VoxBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.voice_states = True
        intents.message_content = True

        super().__init__(
            command_prefix="!",  # Required by d.py, but we use slash commands
            intents=intents,
        )

        self.exit_code = 0
        self.health_runtime = RedisHealthRuntime(settings)
        self.docket_runtime = BotDocketRuntime(self)
        self._health_stopped = False

    async def setup_hook(self) -> None:
        """Called once when the bot is starting up."""
        await self._load_plugins()

        # Sync slash commands
        if settings.debug_guild:
            guild = discord.Object(id=int(settings.debug_guild))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            _LOGGER.info("synced_commands_to_debug_guild", guild_id=settings.debug_guild)
        else:
            await self.tree.sync()
            _LOGGER.info("synced_commands_globally")

        await self.health_runtime.start(self)
        asyncio.create_task(self.docket_runtime.start(), name="voxbot-docket-runtime")

    async def _load_plugins(self) -> None:
        """Load all plugins from plugins/ directory using plugin/cog pattern."""
        project_dir = pathlib.Path(__file__).parent

        for subdir in project_dir.glob("plugins/*/"):
            if not (subdir / "__init__.py").exists():
                continue

            await self.load_extension(name=f"voxbot.plugins.{subdir.name}")

    async def on_ready(self):
        _LOGGER.info("bot_online", version=importlib.metadata.version("voxbot"))
        await self.health_runtime.mark_ready(True, bot=self)

    async def on_error(self, event_method: str, /, *args, **kwargs) -> None:
        await self.health_runtime.record_error(f"{event_method} failed")
        await super().on_error(event_method, *args, **kwargs)

    async def request_shutdown(self, *, reason: str, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        _LOGGER.info("bot_shutdown_requested", reason=reason, exit_code=exit_code)
        await self.close()

    async def request_restart(self, *, reason: str) -> None:
        await self.health_runtime.record_restart_requested(reason)
        await self.request_shutdown(reason=reason, exit_code=75)

    async def close(self) -> None:
        await self.docket_runtime.stop()

        if not self._health_stopped:
            self._health_stopped = True
            await self.health_runtime.stop()

        await super().close()
