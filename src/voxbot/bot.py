import importlib.metadata
import pathlib

from discord.ext import commands
import discord
import structlog

from voxbot.settings import settings
from voxbot.tasks import VoiceBackgroundTasks

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

        self.background_tasks: VoiceBackgroundTasks | None = None

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

    async def _load_plugins(self) -> None:
        """Load all plugins from plugins/ directory using plugin/cog pattern."""
        project_dir = pathlib.Path(__file__).parent

        for subdir in project_dir.glob("plugins/*/"):
            if not (subdir / "__init__.py").exists():
                continue

            _LOGGER.info("Loading plugin..", plugin=subdir.name)
            await self.load_extension(name=f"voxbot.plugins.{subdir.name}")

    async def on_ready(self):
        _LOGGER.info("bot_online", version=importlib.metadata.version("voxbot"))

        if self.background_tasks is None:
            # Note: We rely on vox_model and mistral_service being attached by cogs
            # during their __init__ or in setup_hook. In the old code, cogs did:
            # bot.mistral_service = self.mistral_service
            # We'll ensure this still works.
            self.background_tasks = VoiceBackgroundTasks(
                self, getattr(self, "mistral_service", None), getattr(self, "vox_model", None)
            )
            self.background_tasks.start()
