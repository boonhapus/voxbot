from collections.abc import Mapping
import datetime as dt
from types import TracebackType
from typing import Any
import io
import traceback

import discord
import structlog

from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)

type ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]
_MAX_SUMMARY_CHARS = 1900


def _shorten(value: Any, *, max_len: int = 280) -> str:
    text = str(value).replace("`", "'").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\n")
    return text if len(text) <= max_len else f"{text[: max_len - 1]}…"


def _format_traceback(
    *,
    error: BaseException | None = None,
    exc_info: ExcInfo | None = None,
) -> tuple[BaseException, str, TracebackType | None]:
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

    return resolved_error, trace, resolved_tb


def _extract_app_frame(tb: TracebackType | None) -> str | None:
    if tb is None:
        return None

    frames = traceback.extract_tb(tb)
    if not frames:
        return None

    for frame in reversed(frames):
        normalized = frame.filename.replace("\\", "/")
        if "/src/voxbot/" in normalized:
            rel = normalized.split("/src/voxbot/", maxsplit=1)[1]
            return f"src/voxbot/{rel}:{frame.lineno} ({frame.name})"

    frame = frames[-1]
    return f"{frame.filename}:{frame.lineno} ({frame.name})"


def _render_summary(*, subject: str, details: Mapping[str, Any]) -> str:
    lines = [f"🚨 {subject}"]

    for key, value in details.items():
        lines.append(f"**{key}:** `{_shorten(value, max_len=220)}`")

    summary = "\n".join(lines)
    if len(summary) <= _MAX_SUMMARY_CHARS:
        return summary

    return f"{summary[: _MAX_SUMMARY_CHARS - 1]}…"


def _render_report(*, title: str, details: Mapping[str, Any], traceback_text: str) -> str:
    lines = [f"🚨 {title}"]

    for key, value in details.items():
        lines.append(f"**{key}:** `{_shorten(value, max_len=4000)}`")

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
    resolved_error, traceback_text, resolved_tb = _format_traceback(error=error, exc_info=exc_info)

    report_details: dict[str, Any] = dict(details)
    report_details.setdefault("Exception", type(resolved_error).__name__)
    report_details.setdefault("App Frame", _extract_app_frame(resolved_tb) or "unknown")
    report_details.setdefault("Release", settings.voxbot_release_sha or "unknown")
    report_details.setdefault(
        "Timestamp (UTC)",
        dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
    )

    summary = _render_summary(subject=subject, details=report_details)
    report = _render_report(title=title, details=report_details, traceback_text=traceback_text)

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
            summary,
            file=discord.File(io.BytesIO(report.encode("utf-8")), filename=filename),
        )
    except discord.HTTPException:
        _LOGGER.warning("owner_error_dm_failed", filename=filename, owner_id=settings.bot_owner_id)
