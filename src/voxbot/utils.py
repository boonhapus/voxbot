from types import TracebackType
from typing import Any
import asyncio
import datetime as dt
import functools as ft
import hashlib
import json
import pathlib
import traceback

import discord
import redis
import yaml

from voxbot.settings import settings

RedisClient = redis.asyncio.from_url(settings.redis_url, decode_responses=True)


def hash_command_tree(tree: discord.app_commands.CommandTree) -> str:
    """Convert a discord CommandTree into a stable string for comparison."""
    payloads = [cmd.to_dict(tree) for cmd in tree.get_commands()]
    payloads.sort(key=lambda d: d.get("name", ""))
    blob = json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def no_task_dangling(task: asyncio.Task[Any], *, struct: set[asyncio.Task[Any]]) -> None:
    """
    Perform lifecycle management on asyncio.Tasks.

    Further reading:
      https://docs.astral.sh/ruff/rules/asyncio-dangling-task/
      https://textual.textualize.io/blog/2023/02/11/the-heisenbug-lurking-in-your-async-code/
    """
    struct.add(task)
    task.add_done_callback(struct.discard)


class MdExceptionFormatter:
    """Convert an Exception into a formatted Markdown string."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.when = dt.datetime.now(tz=dt.UTC)

    @property
    def exc_type(self) -> type[BaseException]:
        """The exception class."""
        return type(self.exc)

    @ft.cached_property
    def tb(self) -> traceback.StackSummary:
        """The extracted summary traceback on the exception."""
        return traceback.StackSummary.extract(traceback.walk_tb(self.exc.__traceback__))

    @property
    def metadata(self) -> dict[str, Any]:
        """The same information found in the YAML frontmatter."""
        return {
            "exc_type": f"{self.exc_type.__module__}.{self.exc_type.__qualname__}",
            "exc_message": f"{self.exc}",
            "timestamp": self.when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filename": self.tb[-1].filename,
            "line": self.tb[-1].lineno,
            "function": self.tb[-1].name,
        }

    def format(
        self,
        title: str = "Exception Report",
        width: int = 120,
        frontmatter: bool = True,
        locals: bool = False,
        limit: int | None = 50,
    ) -> str:
        """Join all the sections of the formatter together."""
        sections = [
            self._frontmatter(width=int(width * 0.75)) if frontmatter else "",
            self._header(title),
            self._summary_table(),
            self._traceback_section(limit),
            self._local_variables(var_width=int(width * 0.50)) if locals else "",
        ]

        return "\n".join(filter(None, sections))

    def _frontmatter(self, width: int = 80) -> str:
        """YAML frontmatter with key exception metadata."""
        frontmatter = yaml.safe_dump(
            self.metadata,
            sort_keys=False,
            default_flow_style=False,
            width=width,
            allow_unicode=True,
        )

        return "\n".join(["---", frontmatter, "---"])

    def _header(self, title: str) -> str:
        """Generate the header of the markdown document."""
        return "\n".join(
            (
                f"# {title}\n",
                f"> **`{self.exc_type.__qualname__}`**: {self.exc}\n",
                f"*Generated on {self.when:%Y-%m-%d %H:%M:%S%z %Z}*",
            )
        )

    def _summary_table(self) -> str:
        """Generate the concise summary table of the last frame in the exception."""
        filepath = pathlib.Path(self.metadata["filename"])

        rows = {
            "Exception Type": f"`{self.metadata['exc_type']}`",
            "Message": f"{self.exc}",
            "Origin": f"`{filepath.name}` line **{self.metadata['line']}** in `{self.metadata['function']}()`",
            "MRO": " → ".join(f"`{c.__qualname__}`" for c in self.exc_type.__mro__),
        }

        # fmt: off
        th =   "| Field | Value |"
        sp =   "|:------|:------|"
        tr = [f"| {k}   | `{v}` |" for k, v in rows.items()]
        br = ""
        # fmt: on

        return "\n".join(
            (
                "## Summary\n",
                th,
                sp,
                *tr,
                br,
            )
        )

    def _traceback_section(self, limit: int | None = None) -> str:
        """Format the exception up to the last `limit` lines."""
        tb_lines = traceback.format_exception(self.exc, chain=True)
        tb_lines = "\n".join(tb_lines).strip().split("\n")  # Clean up any trailing/leading newlines.

        if limit is not None and len(tb_lines) > limit:
            tb_lines = [f"... ({len(tb_lines) - limit} lines truncated)", "", *tb_lines[-limit:]]

        br = ""

        return "\n".join(
            (
                "## Traceback",
                "```python",
                *tb_lines,
                "```",
                br,
            )
        )

    def _local_variables(self, var_width: int = 80) -> str:
        """Capture local variables from the last frame of the traceback."""

        def unwrap_to_last_frame(tb: TracebackType) -> TracebackType:
            """Navigate the traceback linked list to the last frame."""
            while tb.tb_next:
                tb = tb.tb_next

            return tb

        def make_table_cell_repr(value: Any, *, max_length: int = 60) -> str:
            """Ensure printable variable values."""
            try:
                r = repr(value)
                r = r if len(r) < max_length else f"{r[: max_length - 2]}.."
                r = r.replace(r"|", r"\|")
            except Exception:
                r = "<unrepresentable>"

            return r

        assert self.exc.__traceback__ is not None, "Exception should have a valid Traceback."
        last_frame = unwrap_to_last_frame(self.exc.__traceback__)

        rows = [
            (name, type(value).__name__, make_table_cell_repr(value, max_length=var_width))
            for name, value in sorted(last_frame.tb_frame.f_locals.items())
        ]

        # fmt: off
        th =   "| Variable | Type  | Value |"
        sp =   "|:---------|:------|:------|"
        tr = [f"| `{n}`    | `{t}` | `{v}` |" for (n, t, v) in rows]
        br = ""
        # fmt: on

        return "\n".join(
            (
                "## Local Variables (last frame)\n",
                th,
                sp,
                *tr,
                br,
            )
        )
