import importlib.metadata
import pathlib

from disnake.ext import commands
import disnake
import structlog

from voxbot.settings import settings
from voxbot.tasks import VoiceBackgroundTasks

_LOGGER = structlog.get_logger(__name__)


class VoxBot(commands.InteractionBot):
    def __init__(self):
        intents = disnake.Intents(guilds=True, guild_messages=True, voice_states=True)

        super().__init__(
            intents=intents,
            test_guilds=[int(settings.debug_guild)] if settings.debug_guild else None,
        )

        self.background_tasks: VoiceBackgroundTasks | None = None
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Load all plugins from plugins/ directory."""
        project_dir = pathlib.Path(__file__).parent

        for py in project_dir.glob("plugins/*.py"):
            if "__init__" in py.stem:
                continue

            _LOGGER.info(
                "Loading extension..",
                ext=py.stem,
                path=py.relative_to(project_dir).as_posix(),
            )
            self.load_extension(name=f"voxbot.plugins.{py.stem}")

        # Also load voice plugin subdirectory
        for subdir in project_dir.glob("plugins/*/"):
            if not (subdir / "__init__.py").exists():
                continue
            plugin_name = subdir.name
            _LOGGER.info(
                "Loading plugin directory..",
                plugin=plugin_name,
            )
            self.load_extension(name=f"voxbot.plugins.{plugin_name}")

    async def on_ready(self):
        _LOGGER.info("bot_online", version=importlib.metadata.version("voxbot"))

        if self.background_tasks is None:
            self.background_tasks = VoiceBackgroundTasks(
                self, self.mistral_service, self.vox_model
            )
            self.background_tasks.start()
