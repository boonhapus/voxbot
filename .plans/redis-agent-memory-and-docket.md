# Plan: Redis Agent Memory + Docket Background Work

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
  - Owns `memory_service`.
  - Exposes `remember_person_fact` and `forget_person_fact` as pydantic-ai tools.
- `src/voxbot/plugins/soul/cog.py`
  - Keeps conversation history in memory.
  - Runs a periodic identity check using `discord.ext.tasks`.
- `src/voxbot/tasks.py`
  - Runs periodic Mistral voice sync.
  - Runs periodic inactive voice auto-leave.
- `src/voxbot/plugins/voice/cog.py`
  - Updates `vox_model.last_active` after voice playback.

Important constraint:

- The project uses Python `>=3.14`.
- `agent-memory-client` supports Python `>=3.10`.
- `agent-memory-server` currently requires Python `>=3.12,<3.13`, so run the
  server outside this bot venv, preferably in Docker or its own uv project.
- `agent-memory-server` depends on `pydocket`, but that transitive dependency
  only lands in the memory-server environment if the server runs separately.
  Add `pydocket` directly to Voxbot if Voxbot will own Docket jobs.

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

## New Settings

Add to `Settings`:

```python
redis_url: str = "redis://localhost:6379/0"

docket_url: str | None = None
docket_name: str = "voxbot"
docket_enabled: bool = True

soul_memory_backend: str = "json"  # json | agent_memory
soul_memory_server_url: str = "http://localhost:8000"
soul_memory_namespace: str = "voxbot:soul"
soul_auto_extract_enabled: bool = False
soul_auto_extract_channel_id: str | None = "1306464265703522325"
soul_auto_extract_namespace: str = "voxbot:soul:auto-test"
```

`docket_url` should default to `redis_url` in factory code, not by duplicating
logic in every call site.

---

## Phase 1: Local Infrastructure

### 1.1 Redis

Use one Redis instance for the first pass.

Development target:

```text
redis://localhost:6379/0
```

For clearer isolation later:

- DB 0: Agent Memory Server data.
- DB 1: Voxbot Docket data.

### 1.2 Agent Memory Server

Run Redis AMS outside the Voxbot Python 3.14 venv.

Development config:

```yaml
redis_url: redis://localhost:6379/0
generation_model: gpt-4o-mini
embedding_model: text-embedding-3-small
long_term_memory: true
enable_discrete_memory_extraction: true
index_all_messages_in_long_term_memory: false
auth_mode: disabled
log_level: INFO
```

Run both processes:

```bash
agent-memory api
agent-memory task-worker --concurrency 2
```

Success criteria:

- `GET /v1/health` succeeds.
- A manual memory can be created through `agent-memory-client`.
- A manual search returns the memory.

---

## Phase 2: Memory Service Abstraction

### 2.1 Refactor Prompt Memory Hydration

The current prompt calls `memory_service.summary(message)` synchronously inside
the pydantic-ai system prompt. Redis AMS calls are async, so move memory lookup
before `soul_agent.run`.

Change `DiscordDeps`:

```python
@dataclasses.dataclass
class DiscordDeps:
    bot: discord.Client
    message: discord.Message | None = None
    memory_summary: str = ""
```

Change prompt construction:

```python
memory_summary=ctx.deps.memory_summary
```

Change `SoulCog.on_message`:

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

Change identity check:

```python
deps=ai.DiscordDeps(
    bot=self.bot,
    memory_summary=await ai.memory_service.summary_for_message(None),
)
```

### 2.2 Add an Agent Memory Backend

Keep the existing JSON backend as the default until the Redis path is stable.

Proposed files:

```text
src/voxbot/plugins/soul/memory_base.py
src/voxbot/plugins/soul/json_memory.py
src/voxbot/plugins/soul/agent_memory.py
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

---

## Phase 3: Controlled Auto Extraction

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

AMS background extraction can then promote useful memories to long-term storage.

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

## Phase 4: Docket Runtime for Voxbot

### 4.1 Add Docket Dependency

Add direct dependencies:

```toml
"pydocket>=0.20.2",
"agent-memory-client>=0.14.0",
```

Keep `agent-memory-server` out of this project environment.

### 4.2 Add Docket Factory and Registration

Proposed files:

```text
src/voxbot/docket_runtime.py
src/voxbot/jobs/__init__.py
src/voxbot/jobs/voice.py
src/voxbot/jobs/soul.py
```

Factory:

```python
def docket_url() -> str:
    return settings.docket_url or settings.redis_url

def create_docket() -> Docket:
    return Docket(name=settings.docket_name, url=docket_url())

def register_pure_jobs(docket: Docket) -> None:
    docket.register(sync_voices)
    docket.register(record_auto_extract_message)

def register_bot_local_jobs(docket: Docket) -> None:
    docket.register(auto_leave_voice_clients)
    docket.register(soul_identity_check)
```

### 4.3 Add CLI Worker Entry Point

Extend `voxbot.__main__`:

```text
voxbot              # current bot behavior
voxbot worker       # external Docket worker for pure jobs
```

The external worker should register only pure jobs:

- `sync_voices`
- `record_auto_extract_message`
- future pure maintenance jobs

It must not register jobs that need the live Discord bot object.

### 4.4 Bot-Local Docket Worker

For learning, replace bot-bound `discord.ext.tasks` loops with a Docket worker
running inside the bot process.

Bot-local jobs:

- `auto_leave_voice_clients`
- `soul_identity_check`

This is intentionally bot-local because these jobs need:

- `bot.voice_clients`
- `bot.wait_until_ready()`
- `guild.me`
- `member.edit(...)`

Implementation sketch:

```python
class BotDocketRuntime:
    def __init__(self, bot: VoxBot) -> None:
        self.bot = bot
        self.docket: Docket | None = None
        self.worker: Worker | None = None
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        await self.bot.wait_until_ready()
        self.docket = create_docket()
        await self.docket.__aenter__()
        bind_bot_runtime(self.bot)
        register_bot_local_jobs(self.docket)
        self.worker = Worker(self.docket)
        await self.worker.__aenter__()
        self.task = asyncio.create_task(self.worker.run_forever())

    async def stop(self) -> None:
        ...
```

Use a narrow module-level runtime holder only for bot-local jobs:

```python
_bot: discord.Client | None = None

def bind_bot_runtime(bot: discord.Client) -> None:
    global _bot
    _bot = bot

def require_bot() -> discord.Client:
    if _bot is None:
        raise RuntimeError("bot runtime is not bound")
    return _bot
```

This is acceptable for the learning pilot because only the Discord process
registers these jobs. Do not register bot-local jobs in `voxbot worker`.

---

## Phase 5: Replace Current Background Tasks

### 5.1 Voice Sync

Current:

- `VoiceBackgroundTasks._create_voice_sync_task()`
- every 300 seconds

Docket replacement:

```python
async def sync_voices(
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=300), automatic=True),
    retry: ExponentialRetry = ExponentialRetry(...),
    timeout: Timeout = Timeout(timedelta(minutes=2)),
) -> None:
    service = MistralService()
    await service.sync_voices()
```

Run in the external `voxbot worker`.

### 5.2 Auto Leave

Current:

- `VoiceBackgroundTasks._create_auto_leave_task()`
- every 60 seconds
- disconnects guild voice clients inactive for 300 seconds

Docket replacement:

```python
async def auto_leave_voice_clients(
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=60), automatic=True),
) -> None:
    bot = require_bot()
    vox_model = require_vox_model(bot)
    ...
```

Run only in the bot-local Docket worker.

Risk:

- This is a less natural Docket fit because it depends on live process state.
- Keep it bot-local and test it directly.

### 5.3 Soul Identity Check

Current:

- `SoulCog._create_identity_task()`
- every `settings.soul_name_check_interval_seconds`

Docket replacement:

```python
async def soul_identity_check(
    perpetual: Perpetual = Perpetual(
        every=timedelta(seconds=settings.soul_name_check_interval_seconds),
        automatic=True,
    ),
    timeout: Timeout = Timeout(timedelta(minutes=2)),
) -> None:
    bot = require_bot()
    result = await soul_agent.run(...)
    ...
```

Run only in the bot-local Docket worker.

Success criteria:

- `src/voxbot/tasks.py` is removed or reduced to compatibility glue.
- No `discord.ext.tasks` loops remain for voice sync, auto-leave, or identity
  check.
- Bot startup starts the bot-local Docket runtime once.
- `voxbot worker` can process pure jobs without importing or starting Discord.

---

## Phase 6: Tests

Unit tests:

- JSON memory backend remains behavior-compatible.
- Agent Memory backend formats search results correctly with fake client output.
- Agent Memory forget deletes all matching fake memory ids.
- Auto-extraction channel guard accepts only `1306464265703522325`.
- Auto-extraction task payload is idempotently keyed by message id.
- Docket tasks can be called directly as normal async functions.
- Docket registration includes pure jobs in worker mode and bot-local jobs in
  bot mode.

Docket integration tests:

- Use `memory://` backend for local Docket unit tests.
- Schedule `record_auto_extract_message` and run worker until finished.
- Bound perpetual task tests with Docket's limited-run helpers rather than
  waiting forever.

Manual smoke tests:

1. Start Redis.
2. Start Agent Memory API.
3. Start Agent Memory task worker.
4. Start `voxbot worker`.
5. Start Voxbot.
6. Use `remember_person_fact` through natural chat.
7. Confirm search returns memory from `voxbot:soul`.
8. Post in channel `1306464265703522325`.
9. Confirm auto-extracted memory appears only in `voxbot:soul:auto-test`.
10. Wait 5 minutes and confirm voice sync runs through Docket.
11. Join voice, use `/voice speak`, wait 5 minutes, confirm auto-leave runs.

---

## Migration Strategy

### Existing `people.json`

Add a one-shot migration command later:

```text
voxbot memory migrate-json-to-agent-memory
```

Behavior:

- Read `~/.voxbot/soul/people.json`.
- Convert each fact to a semantic memory.
- Preserve category as topic.
- Preserve display names in metadata.
- Use stable generated IDs if the client supports caller-provided IDs; otherwise
  rely on AMS deduplication.

Keep the JSON file after migration until the Redis backend has been validated.

### Rollback

Rollback is a settings change:

```text
SOUL_MEMORY_BACKEND=json
DOCKET_ENABLED=false
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
