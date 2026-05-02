"""Background tasks for voice bot."""

import time

from disnake.ext import tasks
import structlog

from voxbot.services.mistral import MistralService
from voxbot.model import VoxModel

_LOGGER = structlog.get_logger(__name__)

_SYNC_INTERVAL = 300
_AUTO_LEAVE_INTERVAL = 60
_AUTO_LEAVE_THRESHOLD = 300


class VoiceBackgroundTasks:
    """Manages background tasks: voice syncing and auto-disconnect."""

    def __init__(self, bot, mistral_service: MistralService, vox_model: VoxModel):
        self.bot = bot
        self.mistral_service = mistral_service
        self.vox_model = vox_model
        self._voice_sync_task: tasks.Loop | None = None
        self._auto_leave_task: tasks.Loop | None = None

    def start(self) -> None:
        """Start all background tasks. Call after bot is ready."""
        self._voice_sync_task = self._create_voice_sync_task()
        self._auto_leave_task = self._create_auto_leave_task()
        self._voice_sync_task.start()
        self._auto_leave_task.start()
        _LOGGER.info("background_tasks_started")

    def stop(self) -> None:
        """Stop all background tasks."""
        if self._voice_sync_task:
            self._voice_sync_task.cancel()
        if self._auto_leave_task:
            self._auto_leave_task.cancel()
        _LOGGER.info("background_tasks_stopped")

    def _create_voice_sync_task(self) -> tasks.Loop:
        @tasks.loop(seconds=_SYNC_INTERVAL)
        async def voice_sync():
            try:
                await self.mistral_service.sync_voices()
            except Exception as err:
                _LOGGER.error("background_voice_sync_failed", error=str(err))

        @voice_sync.before_loop
        async def before_voice_sync():
            await self.bot.wait_until_ready()

        return voice_sync

    def _create_auto_leave_task(self) -> tasks.Loop:
        @tasks.loop(seconds=_AUTO_LEAVE_INTERVAL)
        async def auto_leave():
            now = time.monotonic()
            expired = [
                gid
                for gid, t in self.vox_model.last_active.items()
                if now - t > _AUTO_LEAVE_THRESHOLD
            ]

            for guild_id in expired:
                for vc in self.bot.voice_clients:
                    if vc.guild.id == guild_id:
                        await vc.disconnect()
                        _LOGGER.info("auto_left_voice", guild_id=guild_id)
                        break
                del self.vox_model.last_active[guild_id]

        @auto_leave.before_loop
        async def before_auto_leave():
            await self.bot.wait_until_ready()

        return auto_leave
