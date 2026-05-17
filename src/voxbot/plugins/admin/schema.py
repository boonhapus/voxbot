import datetime as dt

import pydantic
import structlog

from voxbot import utils

_LOGGER = structlog.get_logger(__name__)


class HealthReport(pydantic.BaseModel):
    """Coerce values from Redis and handle formatting."""

    ready: bool
    heartbeat: dt.datetime
    heartbeat_unix: int
    release_sha: str = "unknown"
    latency_ms: int
    last_error: str | None
    worker_ready: bool
    worker_heartbeat_unix: int
    worker_release_sha: str

    @classmethod
    async def from_redis(cls) -> HealthReport:
        """Query Redis for the health data."""
        mapping = {
            "ready": "voxbot:health:ready",
            "heartbeat": "voxbot:health:heartbeat",
            "heartbeat_unix": "voxbot:health:heartbeat_unix",
            "release_sha": "voxbot:health:release_sha",
            "latency_ms": "voxbot:health:latency_ms",
            "last_error": "voxbot:health:last_error",
            "worker_ready": "voxbot:worker:health:ready",
            "worker_heartbeat_unix": "voxbot:worker:health:heartbeat_unix",
            "worker_release_sha": "voxbot:worker:health:release_sha",
        }

        k = list(mapping.values())
        v = await utils.RedisClient.mget(k)
        return cls.model_validate(dict(zip(mapping.keys(), v, strict=True)))

    @property
    def short_sha(self) -> str:
        return self.release_sha[:12] if len(self.release_sha) > 12 else self.release_sha

    @property
    def worker_short_sha(self) -> str:
        return self.worker_release_sha[:12] if len(self.worker_release_sha) > 12 else self.worker_release_sha

    def format_age(self, timestamp: int) -> str:
        now = int(dt.datetime.now(tz=dt.UTC).timestamp())
        return f"{max(0, now - timestamp)}s"

    def __str__(self) -> str:
        lines = [
            "**Bot Status**",
            f"Ready: {self.ready}",
            f"Release: `{self.short_sha}`",
            f"Heartbeat: {self.format_age(self.heartbeat_unix)}",
            f"Latency: {self.latency_ms}ms",
            "---",
            "**Worker Status**",
            f"Ready: {self.worker_ready}",
            f"Release: `{self.worker_short_sha}`",
            f"Heartbeat: {self.format_age(self.worker_heartbeat_unix)}",
        ]
        if self.last_error and self.last_error.lower() != "none":
            lines.append(f"Last Error: `{self.last_error}`")
        return "\n".join(lines)
