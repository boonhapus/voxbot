from contextlib import suppress
import asyncio
import datetime as dt
import math
import traceback

from discord.ext import commands
import redis
import structlog

from voxbot.settings import settings
from voxbot import utils

_LOGGER = structlog.get_logger(__name__)


class RedisHealthRuntime:
    """Writes process health state to Redis for launchd/deployer checks."""

    def __init__(self, key_prefix: str = "voxbot:health", *, heartbeat_seconds: int = 10) -> None:
        self.key_prefix = key_prefix.rstrip(":")
        self._heartbeat_seconds = heartbeat_seconds
        self._heartbeat_task: asyncio.Task | None = None

    def key(self, name: str) -> str:
        """Create the Redis key."""
        return f"{self.key_prefix}:{name}"

    def _base_values(self, *, bot: commands.Bot | None = None) -> dict[str, str]:
        """Fetch the core objects for each message."""
        now = dt.datetime.now(tz=dt.timezone.utc)

        values: dict[str, str] = {
            self.key("heartbeat"): now.isoformat(),
            self.key("heartbeat_unix"): str(int(now.timestamp())),
            self.key("release_sha"): settings.voxbot_release_sha or "unknown",
        }

        if bot is not None and not math.isnan(bot.latency):
            values[self.key("latency_ms")] = str(int(bot.latency * 1000))

        return values

    async def _mset_with_warning(self, **values: str) -> None:
        """MultipleSet, but safely warn when it goes wrong."""
        try:
            await utils.RedisClient.mset(values)
        except redis.exceptions.RedisError as e:
            _LOGGER.warning("health_write_failed", dev_msg=str(e))

    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    async def start(self, bot: commands.Bot | None = None) -> None:
        """Start the health runtime. Idempotent."""
        if self._heartbeat_task is not None:
            return

        # CLEAR OUT ANY OLD VALUES WHILE WE'RE STARTING THE BOT / RUNTIME.
        await self.mark_ready(False, bot=bot)

        self._heartbeat_task = asyncio.create_task(self._heartbeat(bot), name=f"{self.key_prefix}:heartbeat")

    async def stop(self) -> None:
        """Stop the health runtime."""
        if (task := self._heartbeat_task) is not None:
            self._heartbeat_task = None
            task.cancel()
            
            with suppress(asyncio.CancelledError):
                await task

        await self.mark_ready(False)
        await self.close()
    
    async def close(self) -> None:
        """Close the health runtime."""
        if self._redis is None:
            return
        
        await utils.RedisClient.close()
        self._redis = None

    # ── REDIS INFORMERS ───────────────────────────────────────────────────────────────

    async def _heartbeat(self, bot: commands.Bot | None) -> None:
        """Thump forever."""
        while True:
            await self.heartbeat_once(bot=bot)
            await asyncio.sleep(self._heartbeat_seconds)

    async def heartbeat_once(self, *, bot: commands.Bot | None = None) -> None:
        """Thump."""
        await self._mset_with_warning(**self._base_values(bot=bot))

    async def mark_ready(self, ready: bool, *, bot: commands.Bot | None = None) -> None:
        """Inform the runtime that the bot is ready."""
        values = {
            **self._base_values(bot=bot),
            self.key("ready"): "true" if ready else "false"
        }
        await self._mset_with_warning(**values)

    async def record_restart_requested(self, reason: str) -> None:
        """Inform the runtime that a reset was requested."""
        try:
            await utils.RedisClient.set(self.key("restart_requested"), value=reason)
            await utils.RedisClient.incr(self.key("restart_count"))

        except Exception as err:
            _LOGGER.warning("health_restart_record_failed", dev_msg=str(err))

    async def record_error(self, exc: BaseException) -> None:
        """Inform the runtime that an error was encountered."""
        error_data = {
            "msg": str(exc) or repr(exc),
            "type": type(exc).__name__,
            "timestamp": str(int(dt.datetime.now(tz=dt.timezone.utc).timestamp())),
            "traceback": traceback.format_exc(limit=2),
            "loc": "health_check_service"
        }

        try:
            await utils.RedisClient.hset(self.key("last_error"), mapping=error_data)  # type: ignore

        except Exception as e:
            _LOGGER.error("critical_failure_recording_error", dev_msg=str(e))
