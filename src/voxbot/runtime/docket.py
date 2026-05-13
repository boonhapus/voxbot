from collections.abc import Callable, Awaitable
from contextlib import suppress
from typing import Any
import asyncio

from discord.ext import commands
import docket
import structlog

from voxbot.settings import settings
from voxbot import __project__

_LOGGER = structlog.get_logger(__name__)

type _TaskFuncT = Callable[..., Awaitable[Any]]


class DurableTasks:
    """Stores tasks decorated with @durable_task."""
    _TASKS: list[_TaskFuncT] = []

    @classmethod
    def register[T: _TaskFuncT](cls, func: T) -> T:
        """Add task to the registry."""
        if func not in cls._TASKS:
            cls._TASKS.append(func)

        return func

    @classmethod
    def get_tasks(cls) -> list[_TaskFuncT]:
        """Fetch tasks in the registry."""
        return cls._TASKS


durable_task = DurableTasks.register


class BotDocketRuntime:
    """Manages durable tasks across processes."""

    _BOT_INSTANCE: commands.Bot | None = None

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.__class__._BOT_INSTANCE = bot
        self._main_task: asyncio.Task[None] | None = None
    
    @classmethod
    def fetch_bot_instance(cls) -> commands.Bot:
        """Allow the Bot instance to be injected into tasks."""
        assert cls._BOT_INSTANCE is not None, "Attempted to call a background task before DurableTasks were started."
        return cls._BOT_INSTANCE

    async def _run(self) -> None:
        """Internal loop that handles the Docket lifecycle."""
        await self.bot.wait_until_ready()

        # Using context managers ensures cleanup even on task cancellation
        async with docket.Docket(name=__project__.__name__, url=settings.redis_url) as d:
            for task in DurableTasks.get_tasks():
                d.register(task)

            _LOGGER.info("docket_tasks_registered", count=len(DurableTasks.get_tasks()))

            async with docket.Worker(d) as worker:
                _LOGGER.info("bot_docket_runtime_started")
                await worker.run_forever()

    def start(self) -> None:
        """Start the background worker task. Idempotent."""
        if self._main_task:
            return

        self._main_task = asyncio.create_task(self._run(), name="docket_worker")

    async def stop(self) -> None:
        """Stop the background worker and clean up resources."""
        if not self._main_task:
            return

        self._main_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._main_task
        
        self._main_task = None
        self.__class__._BOT_INSTANCE = None
        _LOGGER.info("bot_docket_runtime_stopped")
