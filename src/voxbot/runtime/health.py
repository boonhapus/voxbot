from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
import datetime
import os
from typing import Any

import structlog

from voxbot.runtime.redis import close_redis_client, create_redis_client
from voxbot.settings import Settings, settings

_LOGGER = structlog.get_logger(__name__)
_MAX_ERROR_LENGTH = 500


class RedisHealthRuntime:
    """Writes process health state to Redis for launchd/deployer checks."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        key_prefix: str = "voxbot:health",
        redis_client: Any | None = None,
        redis_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.key_prefix = key_prefix.rstrip(":")
        self.release_sha = os.getenv("VOXBOT_RELEASE_SHA") or config.deployment_id or "unknown"
        self._redis = redis_client
        self._redis_factory = redis_factory or (lambda: create_redis_client(config.redis_url))
        self._heartbeat_task: asyncio.Task[None] | None = None

    def key(self, name: str) -> str:
        return f"{self.key_prefix}:{name}"

    async def start(self, bot: Any | None = None) -> None:
        if not self.config.health_enabled or self._heartbeat_task is not None:
            return

        await self.mark_ready(False, bot=bot)
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(bot),
            name=f"{self.key_prefix}:heartbeat",
        )

    async def stop(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None

        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        await self.mark_ready(False)
        await self.close()

    async def close(self) -> None:
        if self._redis is None:
            return

        await close_redis_client(self._redis)
        self._redis = None

    async def mark_ready(self, ready: bool, *, bot: Any | None = None) -> None:
        values = self._base_values(bot=bot)
        values[self.key("ready")] = "true" if ready else "false"
        await self._set_many(values)

    async def heartbeat_once(self, *, bot: Any | None = None) -> None:
        await self._set_many(self._base_values(bot=bot))

    async def record_error(self, error: str) -> None:
        await self._set_many({self.key("last_error"): _summarize_error(error)})

    async def record_restart_requested(self, reason: str) -> None:
        client = await self._client()
        try:
            await client.set(self.key("restart_requested"), _summarize_error(reason))
            await client.incr(self.key("restart_count"))
        except Exception as err:
            _LOGGER.warning("health_restart_record_failed", error=str(err))

    async def _heartbeat_loop(self, bot: Any | None) -> None:
        while True:
            await self.heartbeat_once(bot=bot)
            await asyncio.sleep(self.config.health_heartbeat_seconds)

    def _base_values(self, *, bot: Any | None = None) -> dict[str, str]:
        now = datetime.datetime.now(datetime.UTC)
        values = {
            self.key("heartbeat"): now.isoformat(),
            self.key("heartbeat_unix"): str(int(now.timestamp())),
            self.key("release_sha"): self.release_sha,
        }

        latency_ms = _latency_ms(bot)
        if latency_ms is not None:
            values[self.key("latency_ms")] = str(latency_ms)

        return values

    async def _client(self) -> Any:
        if self._redis is None:
            self._redis = self._redis_factory()
        return self._redis

    async def _set_many(self, values: dict[str, str]) -> None:
        try:
            client = await self._client()
            for key, value in values.items():
                await client.set(key, value)
        except Exception as err:
            _LOGGER.warning("health_write_failed", error=str(err))


def _latency_ms(bot: Any | None) -> int | None:
    latency = getattr(bot, "latency", None)
    if not isinstance(latency, int | float) or not (latency > 0):
        return None
    return int(latency * 1000)


def _summarize_error(error: str) -> str:
    return " ".join(str(error).split())[:_MAX_ERROR_LENGTH]
