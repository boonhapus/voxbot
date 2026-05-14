from collections.abc import Mapping
from types import TracebackType
from typing import Any
import io
import traceback

import discord
import structlog

from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)

type ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


def _format_traceback(
    *,
    error: BaseException | None = None,
    exc_info: ExcInfo | None = None,
) -> tuple[BaseException, str]:
    resolved_error = error
    resolved_type = type(error) if error is not None else None
    resolved_tb = error.__traceback__ if error is not None else None

    if exc_info is not None:
        exc_type, exc, tb = exc_info

        if exc_type is not None:
            resolved_type = exc_type
        if exc is not None:
            resolved_error = exc
        if tb is not None:
            resolved_tb = tb

    if resolved_error is None:
        resolved_error = Exception("Unknown error")

    if resolved_type is None:
        resolved_type = type(resolved_error)

    trace = "".join(traceback.format_exception(resolved_type, resolved_error, resolved_tb)).strip()
    if not trace:
        trace = f"{resolved_type.__name__}: {resolved_error}"

    return resolved_error, trace


def _render_report(*, title: str, details: Mapping[str, Any], traceback_text: str) -> str:
    lines = [f"🚨 {title}"]

    for key, value in details.items():
        safe = str(value).replace("`", "'")
        lines.append(f"**{key}:** `{safe}`")

    lines.append("")
    lines.append(f"**Traceback:**\n```py\n{traceback_text}\n```")

    return "\n".join(lines)


async def dm_owner_error_report(
    bot: discord.Client,
    *,
    subject: str,
    title: str,
    details: Mapping[str, Any],
    filename: str,
    error: BaseException | None = None,
    exc_info: ExcInfo | None = None,
) -> None:
    _, traceback_text = _format_traceback(error=error, exc_info=exc_info)
    report = _render_report(title=title, details=details, traceback_text=traceback_text)

    owner = bot.get_user(settings.bot_owner_id)
    if owner is None:
        try:
            owner = await bot.fetch_user(settings.bot_owner_id)
        except discord.HTTPException:
            _LOGGER.warning("owner_fetch_failed", owner_id=settings.bot_owner_id)
            return

    if owner is None:
        _LOGGER.warning("owner_not_found", owner_id=settings.bot_owner_id)
        return

    try:
        await owner.send(
            subject,
            file=discord.File(io.BytesIO(report.encode("utf-8")), filename=filename),
        )
    except discord.HTTPException:
        _LOGGER.warning("owner_error_dm_failed", filename=filename, owner_id=settings.bot_owner_id)
