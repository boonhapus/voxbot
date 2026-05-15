"""Run development tasks."""
from typing import Annotated, NamedTuple
import subprocess as sp

import cyclopts

type ExitCode = Annotated[int, "OS-level exit code."]


class Command(NamedTuple):
    """A shell command exposed as a CLI task."""
    description: str
    argv: list[str]


COMMANDS: dict[str, Command] = {
    "lint": Command("Run ruff linter with auto-fix.", ["ruff", "check", "--fix", "src/"]),
    "format": Command("Run ruff formatter.", ["ruff", "format", "src/"]),
    "typecheck": Command("Run pyrefly type checker.", ["pyrefly", "check", "src/"]),
    "deadcode": Command("Run vulture dead code detection.", ["vulture", "src/"]),
    "deps": Command("Run deptry unused dependency check.", ["deptry", "."]),
    "test": Command("Run pytest.", ["pytest"]),
}


def _runner(*tasks: str) -> ExitCode:
    """Execute the named tasks sequentially, returning the first non-zero exit code or 0 on success."""
    for task in tasks:
        command = COMMANDS[task]

        print(f"\n--- {task} ---")

        if (r := sp.run(["uv", "run", *command.argv])) and r.returncode != 0:
            return r.returncode

    return 0


app = cyclopts.App(name="dev", help="Run development tasks.")


for _name, _cmd in COMMANDS.items():

    def _make_handler(name: str, description: str):
        def handler() -> ExitCode:
            return _runner(name)

        handler.__name__ = name
        handler.__doc__ = description
        return handler

    app.command(_make_handler(_name, _cmd.description))


@app.default
def all() -> ExitCode:
    """Run all tasks."""
    return _runner(*COMMANDS)


@app.command
def check() -> ExitCode:
    """Run lint, format, typecheck, deadcode, and deps (no tests)."""
    return _runner("lint", "format", "typecheck", "deadcode", "deps")


if __name__ == "__main__":
    raise SystemExit(app())
