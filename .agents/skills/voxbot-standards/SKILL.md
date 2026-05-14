# voxbot coding standards

Load this skill when working on any code in the voxbot repository. Follow these conventions unless the user explicitly overrides them.

## Plugin structure

```
src/voxbot/plugins/<name>/
├── __init__.py      # setup(bot), calls bot.add_cog()
├── cog.py           # commands.GroupCog
├── settings.py      # (optional) pydantic_settings.BaseSettings
├── jobs.py          # (optional) @durable_task functions
├── ai.py            # (optional) pydantic_ai.Agent definitions
├── errors.py        # (optional) plugin-local exceptions
├── schema.py        # (optional) pydantic models
├── utils.py         # (optional) helpers
├── prompts/         # (optional) .mdc Jinja2 templates with YAML frontmatter
└── ...              # domain files (actions, memory, storage, state, model, etc.)
```

## Imports — strict order

```python
import <stdlib>                       # typing, dataclasses, enum, etc.

import <third-party>                  # pydantic, discord, structlog, etc.

from voxbot import <...>             # first-party voxbot.* (blank line before)

from . import cog                     # relative local — always last
from .settings import soul_settings
```

- Do **NOT** use `from __future__ import annotations`. Python 3.14+ has deferred evaluation by default.

## Module-level logger

```python
import structlog
_LOGGER = structlog.get_logger(__name__)
```

## Type annotations

Every function/method has type annotations. Use `| None` (Python 3.14 pipe), not `Optional[]`.

## Type checking (pyrefly)

Type checking is enforced in CI via `uv run pyrefly check`. The config lives in `[tool.pyrefly]` in `pyproject.toml`.

### Running locally

```bash
uv run pyrefly check                  # full check
uv run pyrefly check --summarize-errors  # with error summary
```

### Suppressing errors

Use specific `# pyrefly: ignore[error-kind]` — never bare `# type: ignore`. The error kind is part of the diagnostic (e.g. `missing-attribute`, `bad-return`).

Common suppressions in this codebase (stub gaps in discord.py / songbird):

```python
# pyrefly: ignore[missing-attribute]    # dynamic discord.py attrs
# pyrefly: ignore[bad-specialization]    # MRO / TypeVar issues with songbird
# pyrefly: ignore[unexpected-keyword]    # extra kwargs added by songbird
```

Multiple kinds per line:

```python
# pyrefly: ignore[bad-specialization, unexpected-keyword]
```

### Maintaining on new code

- Run `uv run pyrefly check` before committing
- Prefer real fixes (guards, casts, asserts) over ignores — only suppress when upstream stubs are the limiting factor
- When upgrading pyrefly or dependencies, run `pyrefly suppress` to silence newly revealed errors, then fix them incrementally

## Error hierarchy

Global exceptions in `src/voxbot/errors.py` (base `VoxBotError`, check failures off `commands.CheckFailure`).
Plugin-local exceptions go in the plugin's own `errors.py`.

## Logging

- Lowcard_snake_case event names: `"admin_cog_loaded"`, `"soul_action_failed"`
- Key-value pairs for context: `_LOGGER.info("event_name", key=value, ...)`
- Jobs: `job_<name>_started`, `job_<name>_completed`, `job_<name>_failed`
- Secrets: use `pydantic.Field(repr=False)`, never log them

## Docstrings

Short sentence-per-line summary of purpose. Do **NOT** document arguments or return types — type hints cover that. Reference external documentation under a `Further reading:` heading when the function has external dependencies or contracts.

```
def my_function(...) -> ...:
    """Short purpose.

    Further reading:
        https://discordpy.dev/...
    """
```

## Section headers

```
# ── SECTION NAME ──────────────────────────────────────────────────────────
```

Use in classes with multiple logical groups. Pad with `─` to ~80–100 chars.

## Cog patterns

- Extend `commands.GroupCog`, pass `name=` kwarg
- `__init__` takes `bot: commands.Bot`, stores as `self.bot`
- Include `cog_load` for startup logging: `_LOGGER.info("<name>_cog_loaded")`
- `@app_commands.command(...)` with `description=`
- Always `defer(thinking=True)` before async work
- `ephemeral=True` for admin-only responses
- `@commands.Cog.listener()` for event hooks (`on_ready`, `on_message`)
- Guard listener entry with early returns (bot check, channel whitelist)
- Catch only known exception types — never bare `except Exception`. Tight exception clauses make failure modes explicit.
- Exception variable is always short: `except SomeError as exc:` — never `as e:`, `as err:`, etc.

## Jobs pattern (`jobs.py`)

```python
@durable_task
async def my_job(
    bot: commands.Bot = Depends(BotDocketRuntime.fetch_bot_instance),
    perpetual: Perpetual = Perpetual(every=dt.timedelta(hours=6), automatic=True),
) -> None:
    """Short description of the job."""
    _LOGGER.info("job_my_job_started")
    ...
    _LOGGER.info("job_my_job_completed")
```

If the job doesn't need the bot instance, omit the `bot` parameter. Import from `voxbot.runtime.docket`.

## AI agent pattern (`ai.py`)

```python
from pydantic_ai import Agent, ModelSettings, RunContext
import discord
import pydantic

from voxbot.settings import settings


@dataclasses.dataclass
class DiscordDeps:
    bot: discord.Client
    message: discord.Message | None = None


class DiscordResponse(pydantic.BaseModel):
    actions: list[SomeActionT] = pydantic.Field(default_factory=list)


my_agent = Agent(
    settings.txt_model,
    deps_type=DiscordDeps,
    model_settings=ModelSettings(temperature=0.4),
)


@my_agent.system_prompt
async def _persona(ctx: RunContext[DiscordDeps]) -> str: ...


@my_agent.tool
async def my_tool(ctx: RunContext[DiscordDeps], ...) -> str: ...
```

## Settings

Single `pydantic_settings.BaseSettings` class in the plugin's `settings.py`. Use `env_prefix="PLUGINNAME_"`. Instantiate as `soul_settings = SoulSettings()` at module level (snake_case module-level singleton).

For global bot config, use `voxbot.settings.settings` (imported as `from voxbot.settings import settings`).

## Style / taste

- Double quotes `"` always — never single quotes
- Trailing commas on every line in multi-line collections
- Guard clauses / early return — no deep nesting
- `type` keyword for type aliases (Python 3.14+): `type _MyType = ...`
- Module-level singletons (no getter functions)
- `# pyrefly: ignore[error-kind]` only when unavoidable, always specify the error kind
- No comments in implementation code — docstrings and section headers only
