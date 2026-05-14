import datetime as dt
import time

from discord.ext import commands
from docket import Depends, Perpetual
import structlog

from voxbot.runtime.docket import BotDocketRuntime, durable_task

from .state import mistral_service, vox_model

_LOGGER = structlog.get_logger(__name__)

IDLE_DISCONNECT_SECONDS = 300


@durable_task
async def sync_voices(
    perpetual: Perpetual = Perpetual(every=dt.timedelta(seconds=300), automatic=True),
) -> None:
    """Sync available voices from the Mistral API."""
    _LOGGER.info("job_sync_voices_started")

    added = await mistral_service.sync_voices()

    if added:
        _LOGGER.info("job_sync_voices_completed", added_count=len(added))
    else:
        _LOGGER.info("job_sync_voices_no_new_voices")


@durable_task
async def auto_leave_voice_clients(
    bot: commands.Bot = Depends(BotDocketRuntime.fetch_bot_instance),
    perpetual: Perpetual = Perpetual(every=dt.timedelta(seconds=60), automatic=True),
) -> None:
    """Auto-disconnect voice clients idle past the disconnect threshold."""
    _LOGGER.info("job_auto_leave_started")

    now = time.monotonic()
    expired = [gid for gid, t in vox_model.last_active.items() if now - t > IDLE_DISCONNECT_SECONDS]

    for guild_id in expired:
        for vc in bot.voice_clients:
            if vc.guild.id == guild_id:  # pyrefly: ignore[missing-attribute]
                await vc.disconnect(force=True)
                _LOGGER.info("job_auto_leave_disconnected", guild_id=guild_id)
                break

        vox_model.last_active.pop(guild_id, None)
