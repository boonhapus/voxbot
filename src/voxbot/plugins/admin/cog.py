from __future__ import annotations

import datetime
import pathlib
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands
import structlog

from voxbot.runtime.redis import close_redis_client, create_redis_client
from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)
_DEPLOYED_SHA_PATH = pathlib.Path("/Users/voxbot/apps/voxbot/deployed_sha")


class AdminCog(commands.GroupCog, name="admin"):
    """Owner-only deployment and health commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="health", description="Show bot deployment health")
    async def admin_health(self, interaction: discord.Interaction):
        if not await self._require_owner(interaction):
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        values = await self._read_health_values()
        await interaction.edit_original_response(content=_format_health_report(values))

    @app_commands.command(name="restart", description="Restart the bot process")
    @app_commands.describe(reason="Optional reason to record in health state")
    async def admin_restart(self, interaction: discord.Interaction, reason: str | None = None):
        if not await self._require_owner(interaction):
            return

        restart_reason = reason or f"requested by {interaction.user.id}"
        await interaction.response.send_message(f"Restart requested: {restart_reason}", ephemeral=True)

        request_restart = getattr(self.bot, "request_restart", None)
        if request_restart is None:
            _LOGGER.warning("restart_unavailable")
            return

        await request_restart(reason=restart_reason)

    @app_commands.command(name="deploy", description="Show deployment status")
    async def admin_deploy(self, interaction: discord.Interaction):
        if not await self._require_owner(interaction):
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        values = await self._read_health_values()
        release_sha = values.get("release_sha") or "unknown"
        deployed_sha = _read_deployed_sha()

        lines = [
            f"Running release: {_short_sha(release_sha)}",
            f"Deployed sha file: {_short_sha(deployed_sha) if deployed_sha else 'unavailable'}",
            "Deploy command execution is intentionally disabled from Discord.",
        ]
        await interaction.edit_original_response(content="\n".join(lines))

    async def _require_owner(self, interaction: discord.Interaction) -> bool:
        if await _is_owner(self.bot, interaction.user):
            return True

        await interaction.response.send_message("Only the configured bot owner can use this command.", ephemeral=True)
        return False

    async def _read_health_values(self) -> dict[str, str]:
        client = create_redis_client(settings.redis_url)
        try:
            keys = {
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
            values: dict[str, str] = {}
            for label, key in keys.items():
                value = await client.get(key)
                if value is not None:
                    values[label] = str(value)
            return values
        except Exception as err:
            _LOGGER.warning("admin_health_read_failed", error=str(err))
            return {"last_error": f"Redis health read failed: {err}"}
        finally:
            await close_redis_client(client)

    async def cog_load(self) -> None:
        _LOGGER.info("admin_cog_loaded")


async def _is_owner(bot: commands.Bot, user: discord.abc.User) -> bool:
    owner_ids = _parse_owner_ids(settings.discord_owner_ids)
    if user.id in owner_ids:
        return True

    try:
        return await bot.is_owner(user)
    except Exception as err:
        _LOGGER.warning("owner_check_failed", error=str(err), user_id=user.id)
        return False


def _parse_owner_ids(raw_ids: str | None) -> set[int]:
    if not raw_ids:
        return set()

    owner_ids: set[int] = set()
    for token in raw_ids.replace(",", " ").split():
        try:
            owner_ids.add(int(token))
        except ValueError:
            _LOGGER.warning("invalid_owner_id", owner_id=token)

    return owner_ids


def _format_health_report(values: dict[str, str], *, now: int | None = None) -> str:
    now = now if now is not None else int(datetime.datetime.now(datetime.UTC).timestamp())
    heartbeat_age = _heartbeat_age(values.get("heartbeat_unix"), now)
    worker_heartbeat_age = _heartbeat_age(values.get("worker_heartbeat_unix"), now)

    lines = [
        f"Ready: {values.get('ready', 'unknown')}",
        f"Release: {_short_sha(values.get('release_sha'))}",
        f"Heartbeat age: {heartbeat_age}",
        f"Discord latency: {values.get('latency_ms', 'unknown')} ms",
        f"Worker ready: {values.get('worker_ready', 'unknown')}",
        f"Worker release: {_short_sha(values.get('worker_release_sha'))}",
        f"Worker heartbeat age: {worker_heartbeat_age}",
    ]

    if last_error := values.get("last_error"):
        lines.append(f"Last error: {last_error}")

    return "\n".join(lines)


def _heartbeat_age(raw_timestamp: str | None, now: int) -> str:
    if raw_timestamp is None:
        return "unknown"

    try:
        age = max(0, now - int(raw_timestamp))
    except ValueError:
        return "unknown"

    return f"{age}s"


def _short_sha(sha: Any) -> str:
    if not sha:
        return "unknown"

    text = str(sha)
    if len(text) <= 12:
        return text
    return text[:12]


def _read_deployed_sha() -> str | None:
    try:
        return _DEPLOYED_SHA_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
