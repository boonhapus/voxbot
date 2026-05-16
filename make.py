"""
Run development tasks.

>>> uv run make.py
Usage: dev COMMAND [ARGS]

Run development tasks.

╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ deadcode   Run skylos for security issues (--danger), api keys (--secrets), code (--quality), and scan CVEs (--sca). │
│ deps       Run deptry unused dependency check.                                                                       │
│ format     Run ruff formatter.                                                                                       │
│ lint       Run ruff linter with auto-fix.                                                                            │
│ test       Run pytest.                                                                                               │
│ typecheck  Run pyrefly type checker.                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Bundles ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ all      Run all tasks.                                                                                              │
│ chore    Run deadcode, deps.                                                                                         │
│ quality  Run lint, format, typecheck.                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help -h  Display this message and exit.                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

** generated on 2026/05/15
"""

from typing import Annotated, NamedTuple
import subprocess as sp

from cyclopts import Parameter
from cyclopts.help import ColumnSpec, DefaultFormatter
import cyclopts

type ExitCode = Annotated[int, "OS-level exit code."]


class Command(NamedTuple):
    """A shell command exposed as a CLI task."""

    description: str
    documentation: str
    argv: list[str]

    @property
    def docstring(self) -> str:
        """Template the docstring for the cli."""
        return (
            f"{self.description}"
            f"\n\n[bold]Documentation:[/bold] [cyan][link={self.documentation}]{self.documentation}[/link][/cyan]"
            f"\n\n  [dim][bright_yellow]uv[/bright_yellow] run [bright_green]{' '.join(self.argv)}[/bright_green][/dim]"
        )


COMMANDS: dict[str, Command] = {
    "lint": Command(
        description="Run ruff linter with auto-fix.",
        documentation="https://docs.astral.sh/ruff/linter/",
        argv=["ruff", "check", "--fix", "src/"],
    ),
    "format": Command(
        description="Run ruff formatter.",
        documentation="https://docs.astral.sh/ruff/formatter/",
        argv=["ruff", "format", "src/"],
    ),
    "typecheck": Command(
        description="Run pyrefly type checker.",
        documentation="https://pyrefly.org/en/docs/",
        argv=["pyrefly", "check", "src/"],
    ),
    "deadcode": Command(
        description="Run skylos for security issues (--danger), api keys (--secrets), code (--quality), and scan CVEs (--sca).",  # noqa: E501
        documentation="https://github.com/jendrikseipp/vulture",
        argv=["skylos", "src/", "--all", "--tree"],
    ),
    "deps": Command(
        description="Run deptry unused dependency check.",
        documentation="https://deptry.com/",
        argv=["deptry", "."],
    ),
    "test": Command(
        description="Run pytest.",
        documentation="https://docs.pytest.org/",
        argv=["pytest"],
    ),
}


def _runner(*tasks: str) -> ExitCode:
    """Execute the named tasks sequentially, returning the first non-zero exit code or 0 on success."""
    DIV = "─"

    for task in tasks:
        command = COMMANDS[task]

        print(f"\n\n# {DIV * 2} {task.upper()} {DIV * (80 - len(task) - 7)}\n")

        if (r := sp.run(["uv", "run", *command.argv])) and r.returncode != 0:
            return r.returncode

    print()
    return 0


app = cyclopts.App(
    name="dev",
    help="Run development tasks.",
    help_format="rich",
    help_flags=[],
    version_flags=(),
)

type _HELP = Annotated[
    bool,
    Parameter(
        name="--help",
        alias="-h",
        help="Display this message and exit.",
        negative="",
        show_default=False,
        group=cyclopts.Group(
            "Options",
            help_formatter=DefaultFormatter(
                column_specs=(
                    ColumnSpec(renderer=lambda entry: " ".join(entry.names[1:] + entry.shorts)),
                    ColumnSpec(renderer="description"),
                ),
            ),
        ),
    ),
]


# ── GENERATE THE CLI ──────────────────────────────────────────────────────────────────

for _name, _cmd in COMMANDS.items():

    def _make_handler(name: str, command: Command):
        def handler(help: _HELP = False) -> ExitCode:
            if help:
                app.help_print()
                return 0
            return _runner(name)

        handler.__name__ = name
        handler.__doc__ = command.docstring
        return handler

    app.command(_make_handler(_name, _cmd))


# ── COMMANDS ──────────────────────────────────────────────────────────────────────────


@app.default
def _default(_help: _HELP = False) -> ExitCode:
    """Run all tasks."""
    app.help_print()
    return 1


@app.command(group="Bundles")
def all(help: _HELP = False) -> ExitCode:
    """Run all tasks."""
    if help:
        app.help_print()
        return 0

    return _runner(*COMMANDS)


@app.command(group="Bundles")
def quality(help: _HELP = False) -> ExitCode:
    """Run lint, format, typecheck."""
    if help:
        app.help_print()
        return 0

    return _runner("lint", "format", "typecheck")


@app.command(group="Bundles")
def chore(help: _HELP = False) -> ExitCode:
    """Run deadcode, deps."""
    if help:
        app.help_print()
        return 0

    return _runner("deadcode", "deps")


if __name__ == "__main__":
    raise SystemExit(app())
