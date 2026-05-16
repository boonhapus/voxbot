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
│ --ci       Use this flag to run the equivalent, CI friendly command.                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

** generated on 2026/05/15
"""

from typing import Annotated, NamedTuple
import subprocess as sp

from cyclopts.help import ColumnSpec, DefaultFormatter
import cyclopts

type ExitCode = Annotated[int, "OS-level exit code."]


class Command(NamedTuple):
    """A shell command exposed as a CLI task."""

    description: str
    documentation: str
    argv: list[str]
    ci_argv: list[str] | None = None

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
        ci_argv=["ruff", "check", "src/"],
    ),
    "format": Command(
        description="Run ruff formatter.",
        documentation="https://docs.astral.sh/ruff/formatter/",
        argv=["ruff", "format", "src/"],
        ci_argv=["ruff", "format", "--check", "src/"],
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
        argv=["pytest", "tests/", "-v"],
    ),
}


app = cyclopts.App(
    name="dev",
    help="Run development tasks.",
    help_format="rich",
    help_flags=[],
    version_flags=(),
)


_OPTIONS_GROUP = cyclopts.Group(
    "Options",
    help_formatter=DefaultFormatter(
        column_specs=(
            ColumnSpec(renderer=lambda entry: " ".join(entry.names[1:] + entry.shorts)),
            ColumnSpec(renderer="description"),
        ),
    ),
)


type _HELP = Annotated[
    bool,
    cyclopts.Parameter(
        name="--help",
        alias="-h",
        help="Display this message and exit.",
        negative="",
        show_default=False,
        group=_OPTIONS_GROUP,
    ),
]


type _CI = Annotated[
    bool,
    cyclopts.Parameter(
        name="--ci",
        help="Use this flag to run the equivalent, CI friendly command.",
        negative="",
        show_default=False,
        group=_OPTIONS_GROUP,
        env_var="CI",
    ),
]


def _runner(*tasks: str, ci: _CI = False) -> ExitCode:
    """Execute the named tasks sequentially, returning the first non-zero exit code or 0 on success."""
    DIV = "─"

    for task in tasks:
        command = COMMANDS[task]
        argv = command.ci_argv if ci and command.ci_argv else command.argv

        print(f"\n\n# {DIV * 2} {task.upper()} {DIV * (80 - len(task) - 7)}\n")

        if (r := sp.run(["uv", "run", *argv])) and r.returncode != 0:
            return r.returncode

    print()
    return 0


# ── GENERATE THE CLI ──────────────────────────────────────────────────────────────────

for _name, _cmd in COMMANDS.items():

    def _make_handler(name: str, command: Command):
        def handler(help: _HELP = False, ci: _CI = False) -> ExitCode:
            if help:
                app.help_print()
                return 0
            return _runner(name, ci=ci)

        handler.__name__ = name
        handler.__doc__ = command.docstring
        return handler

    app.command(_make_handler(_name, _cmd))


# ── COMMANDS ──────────────────────────────────────────────────────────────────────────


@app.default
def _default(_help: _HELP = False, _ci: _CI = False) -> ExitCode:
    """Run all tasks."""
    app.help_print()
    return 1


@app.command(group="Bundles")
def all(help: _HELP = False, ci: _CI = False) -> ExitCode:
    """Run all tasks."""
    if help:
        app.help_print()
        return 0

    return _runner(*COMMANDS, ci=ci)


@app.command(group="Bundles")
def quality(help: _HELP = False, ci: _CI = False) -> ExitCode:
    """Run lint, format, typecheck."""
    if help:
        app.help_print()
        return 0

    return _runner("lint", "format", "typecheck", ci=ci)


@app.command(group="Bundles")
def chore(help: _HELP = False, ci: _CI = False) -> ExitCode:
    """Run deadcode, deps."""
    if help:
        app.help_print()
        return 0

    return _runner("deadcode", "deps", ci=ci)


if __name__ == "__main__":
    raise SystemExit(app())
