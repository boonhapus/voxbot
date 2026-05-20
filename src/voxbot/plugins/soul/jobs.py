from typing import cast
import datetime as dt

from discord.ext import commands
from docket import Depends, Perpetual
import structlog

from voxbot.bot import VoxBot
from voxbot.runtime.docket import BotDocketRuntime, durable_task

_LOGGER = structlog.get_logger(__name__)


@durable_task
async def soul_identity_check(
    bot: commands.Bot = Depends(BotDocketRuntime.fetch_bot_instance),
    _perpetual: Perpetual = Perpetual(every=dt.timedelta(hours=6), automatic=True),
) -> None:
    """Periodic identity check for the bot's own display name."""
    from voxbot.plugins.soul import ai

    _LOGGER.info("job_soul_identity_started")

    p = (
        "Background identity check: decide whether your home-guild display name should "
        "change. If it should, call change_own_display_name once. Return an empty actions "
        "list if no change is needed."
    )

    try:
        r = await ai.soul_agent.run(
            p,
            deps=ai.DiscordDeps(bot=cast(VoxBot, bot)),
            output_type=ai.DiscordResponse,
        )

        _LOGGER.info("job_soul_identity_completed", action_count=len(r.output.actions) if r.output else 0)

    except Exception as exc:
        _LOGGER.error("job_soul_identity_failed", error=str(exc))
        raise
