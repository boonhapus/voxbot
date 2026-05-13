import datetime as dt
import time

from discord.ext import commands
from docket import Depends, Perpetual
import structlog

from voxbot.runtime.docket import BotDocketRuntime, durable_task

_LOGGER = structlog.get_logger(__name__)


@durable_task
async def sync_voices(perpetual: Perpetual = Perpetual(every=dt.timedelta(seconds=300), automatic=True)) -> None:
    """Sync available voices from Mistral API."""
    from voxbot.services.mistral import MistralService

    _LOGGER.info("job_sync_voices_started")

    try:
        service = MistralService()

        if added := await service.sync_voices():
            _LOGGER.info("job_sync_voices_completed", added_count=len(added))
        else:
            _LOGGER.info("job_sync_voices_no_new_voices")

    except Exception as err:
        _LOGGER.error("job_sync_voices_failed", error=str(err))
        raise


@durable_task
async def auto_leave_voice_clients(
    bot: commands.Bot = Depends(BotDocketRuntime.fetch_bot_instance),
    perpetual: Perpetual = Perpetual(every=dt.timedelta(seconds=60), automatic=True),
) -> None:
    """Auto-disconnect voice clients inactive for 5 minutes."""
    _LOGGER.info("job_auto_leave_started")
    try:
        vox_model = getattr(bot, "vox_model", None)
        if vox_model is None:
            _LOGGER.warning("job_auto_leave_aborted_no_vox_model")
            return

        now = time.monotonic()
        threshold = 300  # 5 minutes
        expired = [
            gid
            for gid, t in vox_model.last_active.items()
            if now - t > threshold
        ]

        for guild_id in expired:
            for vc in bot.voice_clients:
                if vc.guild.id == guild_id:
                    await vc.disconnect()
                    _LOGGER.info("job_auto_leave_disconnected", guild_id=guild_id)
                    break
            vox_model.last_active.pop(guild_id, None)

    except Exception as err:
        _LOGGER.error("job_auto_leave_failed", error=str(err))
        raise

