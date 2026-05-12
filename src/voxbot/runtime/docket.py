from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from docket import Docket, Worker
import structlog

from voxbot.runtime.jobs import auto_leave_voice_clients, soul_identity_check, sync_voices
from voxbot.settings import Settings, settings

_LOGGER = structlog.get_logger(__name__)

PureTask = Callable[..., object]

PURE_BACKGROUND_TASKS: tuple[PureTask, ...] = (sync_voices,)
BOT_LOCAL_TASKS: tuple[PureTask, ...] = (auto_leave_voice_clients, soul_identity_check)


def register_pure_background_tasks(docket: Any) -> int:
    """Register only tasks that do not need a live Discord client."""
    for task in PURE_BACKGROUND_TASKS:
        docket.register(task)

    return len(PURE_BACKGROUND_TASKS)


def register_bot_local_tasks(docket: Any) -> int:
    """Register tasks that need a live Discord client."""
    for task in BOT_LOCAL_TASKS:
        docket.register(task)

    return len(BOT_LOCAL_TASKS)


_bot: Any | None = None


def bind_bot_runtime(bot: Any) -> None:
    global _bot
    _bot = bot


def require_bot() -> Any:
    if _bot is None:
        raise RuntimeError("bot runtime is not bound")
    return _bot


class BotDocketRuntime:
    """Runs a Docket worker inside the Discord bot process for local tasks."""

    def __init__(self, bot: Any, config: Settings = settings) -> None:
        self.bot = bot
        self.config = config
        self.docket: Docket | None = None
        self.worker: Worker | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self.config.docket_enabled or self._task is not None:
            return

        # Wait for bot to be ready so jobs have access to fully loaded state
        await self.bot.wait_until_ready()

        bind_bot_runtime(self.bot)

        docket_url = self.config.docket_url or self.config.redis_url
        self.docket = Docket(name=self.config.docket_name, url=docket_url)
        await self.docket.__aenter__()

        # Register both pure and local tasks so the bot can handle everything if needed,
        # but primarily for local tasks that the external worker cannot run.
        register_pure_background_tasks(self.docket)
        register_bot_local_tasks(self.docket)

        self.worker = Worker(self.docket)
        await self.worker.__aenter__()

        self._task = asyncio.create_task(
            self.worker.run_forever(),
            name=f"{self.config.docket_name}:bot-worker",
        )
        _LOGGER.info("bot_docket_runtime_started")

    async def stop(self) -> None:
        task = self._task
        self._task = None

        if task is not None:
            task.cancel()
            with asyncio.suppress(asyncio.CancelledError):
                await task

        if self.worker is not None:
            await self.worker.__aexit__(None, None, None)
            self.worker = None

        if self.docket is not None:
            await self.docket.__aexit__(None, None, None)
            self.docket = None

        _LOGGER.info("bot_docket_runtime_stopped")
