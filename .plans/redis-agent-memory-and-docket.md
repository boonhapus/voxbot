# Plan: Redis Agent Memory + Docket Background Work

> **Status:** Phases 1, 4, 5 **complete**. Phases 2, 3, 6 **not started**.
> **Last audited:** 2026-05-12
> **Note:** The separate deploy plan (`.plans/mac-server-self-healing-deploy.md`) has been removed
> as completed — its deploy scripts live in `deploy/macos/`.

## Goal

Use Redis Agent Memory Server and Docket in Voxbot as real, inspectable
learning use cases:

- Replace file-backed `soul` facts with Redis Agent Memory long-term memory.
- Add controlled automatic memory extraction from the bot testing channel
  `1306464265703522325`.
- Replace existing `discord.ext.tasks` background loops with Docket where the
  execution model is a good fit.
- Keep Discord gateway and voice-client side effects in the bot process.

This is partly an infrastructure learning project, so the plan intentionally
uses Docket more than a minimal single-process bot would strictly need.

---

## Current State

Relevant code:

- `src/voxbot/plugins/soul/memory.py`
  - JSON file-backed personal facts.
  - Sync `summary()` plus async `remember()` / `forget()`.
- `src/voxbot/plugins/soul/ai.py`
  - Owns `memory_service` (singleton `MemoryService()`).
  - Exposes `remember_person_fact` and `forget_person_fact` as pydantic-ai tools.
  - System prompt calls `memory_service.summary(message)` synchronously.
- `src/voxbot/plugins/soul/models.py`
  - `DiscordDeps` has `bot` + `message` — **no `memory_summary` field yet**.
- `src/voxbot/plugins/soul/cog.py`
  - Keeps conversation history in memory.
  - `on_message` calls `soul_agent.run` directly without pre-fetching memory.
- `src/voxbot/runtime/docket.py`
  - `BotDocketRuntime` — Docker worker inside bot process.
  - `register_pure_background_tasks()` / `register_bot_local_tasks()`.
- `src/voxbot/runtime/jobs.py`
  - `sync_voices` (pure, 300s interval).
  - `auto_leave_voice_clients` (bot-local, 60s interval).
  - `soul_identity_check` (bot-local, configurable interval).
- `src/voxbot/runtime/worker.py`
  - External worker entry point for pure background tasks.
- `src/voxbot/settings.py`
  - Has `redis_url`, `docket_*`, `soul_*`, `health_*` fields.
  - `soul_auto_extract_enabled: bool = False` (default off).

Important constraint:

- The project uses Python `>=3.14`.
- `agent-memory-client` supports Python `>=3.10`.
- `agent-memory-server` currently requires Python `>=3.12,<3.13`, so run the
  server outside this bot venv, preferably in Docker or its own uv project.
- `pydocket` and `agent-memory-client` are already in `pyproject.toml`.

---

## Target Architecture

```text
Discord gateway process
  VoxBot
    SoulCog
      AgentMemoryService -> agent-memory-client -> Redis Agent Memory API
      Docket producer for auto-extraction messages
    VoiceCog
      records voice activity
    Bot-local Docket worker
      runs tasks that need live bot state

Separate Voxbot worker process
  Docket worker
    runs pure/background tasks

Redis Agent Memory Server
  REST API
  Docket task worker
  Redis / RedisVL storage

Redis
  AMS data and vectors
  AMS Docket queues
  Voxbot Docket queues
```

Use separate Docket names/prefixes:

- AMS: `DOCKET_NAME=memory-server` or its default.
- Voxbot: `DOCKET_NAME=voxbot`.

Use separate namespaces:

- Manual soul memories: `voxbot:soul`.
- Auto-extraction pilot memories: `voxbot:soul:auto-test`.

The main soul prompt should only read from `voxbot:soul` at first. The
auto-extraction pilot can be inspected separately before it affects behavior.

---

## Done: Phase 1 — Local Infrastructure

### 1.1 Redis

- `settings.redis_url` exists (default `redis://localhost:6379/1`).
- `runtime/redis.py` — async Redis client factory.
- `runtime/health.py` — `RedisHealthRuntime` (heartbeat + health state).

### 1.2 Agent Memory Server

- `deploy/macos/infra/compose.yaml` includes `agent-memory-api` and
  `agent-memory-worker` services (ready to `docker compose up`).
- `deploy/macos/infra/infra-up.sh` — bootstrap script.

**Human task:** Run on the Mac server or locally to have AMS available.

---

## Done: Phase 4 — Docket Runtime for Voxbot

### 4.1 Docket Dependency

Already in `pyproject.toml`: `pydocket>=0.20.2`, `agent-memory-client>=0.14.0`.

### 4.2 Docket Factory and Registration

`runtime/docket.py`:
- `register_pure_background_tasks(docket)` — pure tasks only
- `register_bot_local_tasks(docket)` — tasks needing live Discord client
- `bind_bot_runtime(bot)` / `require_bot()` — module-level bot holder
- `BotDocketRuntime` — starts a Docket worker inside the bot process

### 4.3 CLI Worker Entry Point

`__main__.py`:
- `voxbot` — starts Discord bot (with bot-local Docket worker)
- `voxbot worker` — external Docket worker for pure tasks only

### 4.4 Bot-Local Docket Worker

`runtime/docket.py:BotDocketRuntime` — running in `bot.py:setup_hook`.

---

## Done: Phase 5 — Replace Background Tasks

All `discord.ext.tasks` loops replaced with Docket perpetual tasks:

| Task | Location | Type | Interval |
|------|----------|------|----------|
| Voice sync | `runtime/jobs.py:sync_voices` | Pure | 300s |
| Auto leave | `runtime/jobs.py:auto_leave_voice_clients` | Bot-local | 60s |
| Identity check | `runtime/jobs.py:soul_identity_check` | Bot-local | configurable |

No `discord.ext.tasks` remain anywhere in the source.

---

## Not Started: Phase 2 — Memory Service Abstraction

### 2.1 Refactor Prompt Memory Hydration

The current prompt calls `memory_service.summary(message)` synchronously inside
the pydantic-ai system prompt. Redis AMS calls are async, so move memory lookup
before `soul_agent.run`.

Change `DiscordDeps` in `models.py`:

```python
@dataclasses.dataclass
class DiscordDeps:
    bot: discord.Client
    message: discord.Message | None = None
    memory_summary: str = ""        # <-- add this
```

Change prompt construction in `ai.py` — pass `memory_summary` from deps instead
of calling `memory_service.summary(message)`:

```python
memory_summary = ctx.deps.memory_summary
```

Change `SoulCog.on_message` in `cog.py`:

```python
memory_summary = await ai.memory_service.summary_for_message(
    message,
    query=message.content,
)

r = await ai.soul_agent.run(
    message.content,
    deps=ai.DiscordDeps(
        bot=self.bot,
        message=message,
        memory_summary=memory_summary,
    ),
    ...
)
```

Change identity check in `jobs.py`:

```python
deps=ai.DiscordDeps(
    bot=bot,
    memory_summary=await ai.memory_service.summary_for_message(None),
)
```

### 2.2 Add an Agent Memory Backend

Keep the existing JSON backend as the default until the Redis path is stable.

Proposed files:

```text
src/voxbot/plugins/soul/memory_base.py     # Protocol/ABC
src/voxbot/plugins/soul/json_memory.py     # current JSON impl, extracted
src/voxbot/plugins/soul/agent_memory.py    # new Redis AMS impl
```

Minimal protocol:

```python
class SoulMemoryService(Protocol):
    async def summary_for_message(
        self,
        message: discord.Message | None,
        query: str | None = None,
    ) -> str: ...

    async def remember(
        self,
        message: discord.Message | None,
        fact: str,
        category: MemoryCategory = "other",
        person_id: str | None = None,
        person_name: str | None = None,
    ) -> str: ...

    async def forget(
        self,
        message: discord.Message | None,
        fact_fragment: str = "",
        category: MemoryCategory | None = None,
        person_id: str | None = None,
        person_name: str | None = None,
    ) -> str: ...
```

`AgentMemoryService.remember()`:
- Resolve the target person the same way the JSON service does.
- Create a semantic long-term memory.
- Set `user_id` to the Discord user id or `name:{normalized}`.
- Set `namespace=settings.soul_memory_namespace`.
- Set `topics=[category]`.
- Set metadata fields if supported by the client model:
  - source: `discord`
  - source_message_id
  - channel_id
  - guild_id
  - person_display_name

`AgentMemoryService.summary_for_message()`:
- For `message is None`, return a background-safe summary.
- Search long-term memory with:
  - text: `query or message.content or "known facts about this user"`
  - `user_id=UserId(eq=str(message.author.id))`
  - namespace filter for `settings.soul_memory_namespace`
  - `search_mode="hybrid"` if stable, else semantic default
  - `limit=5`
- Format results as the same prompt shape:

```text
- preference: likes black coffee
- job: works in software
```

`AgentMemoryService.forget()`:
- Search candidate memories for that user/category/fragment.
- Delete matching memory IDs.
- Return a Discord-friendly count.

Success criteria:
- Existing JSON memory tests still pass.
- New Agent Memory service tests pass with a fake client.
- One optional integration smoke test can run against local AMS.

### 2.3 Backend selection

In `ai.py`, select backend based on settings:

```python
if settings.soul_memory_backend == "agent_memory":
    memory_service = AgentMemoryService(...)
else:
    memory_service = JsonMemoryService(...)
```

---

## Not Started: Phase 3 — Controlled Auto Extraction

Do not let auto-extracted memories affect the normal soul prompt initially.

### 3.1 Add a Recorder

Proposed file:

```text
src/voxbot/plugins/soul/auto_extract.py
```

Behavior:
- Only run when `settings.soul_auto_extract_enabled` is true.
- Only record messages from `settings.soul_auto_extract_channel_id`.
- Ignore bot messages.
- Use namespace `settings.soul_auto_extract_namespace`.
- Use session id:

```text
discord-auto:{guild_id}:{channel_id}:user:{author_id}
```

Per-author sessions reduce the risk of attributing another user's facts to the
wrong person in a busy channel.

### 3.2 Queue Recording With Docket

The Discord event handler should enqueue a small Docket job and return quickly.

Task payload:

```python
@dataclass
class AutoExtractMessage:
    message_id: int
    guild_id: int | None
    channel_id: int
    author_id: int
    author_display_name: str
    content: str
    created_at: str
```

Docket task:

```python
async def record_auto_extract_message(
    payload: AutoExtractMessage,
    retry: ExponentialRetry = ExponentialRetry(...),
    timeout: Timeout = Timeout(...),
) -> None:
    ...
```

The task writes a `WorkingMemory` update to AMS using:
- `session_id=discord-auto:{guild_id}:{channel_id}:user:{author_id}`
- `user_id=str(author_id)`
- `namespace=settings.soul_auto_extract_namespace`
- one `MemoryMessage(role="user", content=...)`

Use Docket key:

```text
soul:auto-extract:{message_id}
```

This makes replays idempotent at the queue layer.

### 3.3 Inspection Before Use

Add one manual command later:

```text
/soul memory auto-search query:<text> user:<optional user>
```

It searches only `voxbot:soul:auto-test`.

Do not merge auto-extracted memories into `voxbot:soul` until the search output
looks sane over real test-channel traffic.

Success criteria:
- Messages outside channel `1306464265703522325` are ignored.
- Messages inside the channel enqueue exactly one Docket task per Discord
  message id.
- The Docket task can be retried safely.
- Auto-extracted memories are visible in the pilot namespace and absent from
  normal prompt memory.

---

## Not Started: Phase 6 — Tests

Unit tests:
- [ ] JSON memory backend remains behavior-compatible after refactor.
- [ ] Agent Memory backend formats search results correctly with fake client output.
- [ ] Agent Memory forget deletes all matching fake memory ids.
- [ ] Auto-extraction channel guard accepts only `1306464265703522325`.
- [ ] Auto-extraction task payload is idempotently keyed by message id.
- [ ] Docket tasks can be called directly as normal async functions.
- [ ] Docket registration includes pure jobs in worker mode and bot-local jobs in bot mode.

---

## Migration Strategy

### Existing `people.json`

Add a one-shot migration command later:

```text
voxbot memory migrate-json-to-agent-memory
```

Keep the JSON file after migration until the Redis backend has been validated.

### Rollback

Rollback is a settings change:

```text
SOUL_MEMORY_BACKEND=json
SOUL_AUTO_EXTRACT_ENABLED=false
```

Keep the old JSON tests and backend until the Redis path has run for a while.

---

## Open Decisions

1. Whether to use OpenAI defaults for AMS generation/embeddings or wire it to
   the existing Google/Gemini key path.
2. Whether auto-extracted memories should ever affect normal `soul` prompts
   automatically, or require manual promotion.
3. Whether bot-local Docket jobs are acceptable long-term, or should eventually
   be reduced back to local loops for Discord-bound behavior.
4. Whether voice activity should stay in `VoxModel.last_active` or move to
   Redis so external workers can observe it.
5. Whether to add slash commands for memory inspection before enabling the
   Redis backend by default.

---

## Source References

- Redis Agent Memory Server docs: https://redis.github.io/agent-memory-server/
- Redis AMS Python SDK: https://redis.github.io/agent-memory-server/python-sdk/
- Redis AMS configuration: https://redis.github.io/agent-memory-server/configuration/
- Redis AMS integration patterns: https://redis.github.io/agent-memory-server/memory-integration-patterns/
- Docket docs: https://docket.lol/
- Docket task behaviors: https://docket.lol/en/latest/task-behaviors/
- Docket testing: https://docket.lol/en/latest/testing/
