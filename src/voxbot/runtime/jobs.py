import time
from datetime import timedelta

from docket import Perpetual

import structlog

from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)


async def sync_voices(
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=300), automatic=True),
) -> None:
    """Sync available voices from Mistral API."""
    from voxbot.services.mistral import MistralService

    _LOGGER.info("job_sync_voices_started")
    try:
        service = MistralService()
        added = await service.sync_voices()
        if added:
            _LOGGER.info("job_sync_voices_completed", added_count=len(added))
        else:
            _LOGGER.info("job_sync_voices_no_new_voices")
    except Exception as err:
        _LOGGER.error("job_sync_voices_failed", error=str(err))
        raise


async def auto_leave_voice_clients(
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=60), automatic=True),
) -> None:
    """Auto-disconnect voice clients inactive for 5 minutes."""
    from voxbot.runtime.docket import require_bot

    _LOGGER.info("job_auto_leave_started")
    try:
        bot = require_bot()
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


async def soul_identity_check(
    perpetual: Perpetual = Perpetual(
        every=timedelta(seconds=settings.soul_name_check_interval_seconds),
        automatic=True,
    ),
) -> None:
    """Periodic identity check for the bot's own display name."""
    from voxbot.runtime.docket import require_bot
    from voxbot.plugins.soul import ai

    _LOGGER.info("job_soul_identity_started")
    try:
        bot = require_bot()
        if not settings.soul_home_guild_id:
            _LOGGER.warning("job_soul_identity_skipped", reason="missing_home_guild_id")
            return

        result = await ai.soul_agent.run(
            "Background identity check: decide whether your home-guild display name should change. "
            "If it should, call change_own_display_name once. Return only a silent action.",
            deps=ai.DiscordDeps(bot=bot),
            output_type=ai.DiscordResponse,
        )
        action_count = len(result.output.actions) if result.output else 0
        _LOGGER.info("job_soul_identity_completed", action_count=action_count)
    except Exception as err:
        _LOGGER.error("job_soul_identity_failed", error=str(err))
        raise
