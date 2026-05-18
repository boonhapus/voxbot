# Voxbot Architecture

Discord bot. Bot + Worker on macOS host (launchd). Redis + Agent Memory Server in Docker. Soul plugin = pydantic-ai agent with semantic memory.

For deploy mechanics see [deploy/macos/ARCHITECTURE.md](../deploy/macos/ARCHITECTURE.md).

---

## General Architecture

![general architecture](assets/general-architecture.png)

```mermaid
flowchart LR
    User([Chat User]) --> Discord

    subgraph Host["macOS Host (launchd)"]
        Bot["Voxbot Bot<br/>gateway / slash / voice"]
        Worker["Worker<br/>jobs"]
        Deployer["Deployer<br/>polls GitHub"]
    end

    subgraph Docker["Docker (Colima VM)"]
        Redis[("Redis<br/>state / queue / health")]
        AMS["Agent Memory Server<br/>memory"]
    end

    Discord <--> Bot
    Deployer -.restart / deploy.-> Bot
    Deployer -.restart / deploy.-> Worker
    Bot --> Redis
    Worker --> Redis
    AMS --> Redis
    Bot --> AMS
    Bot --> Mistral["Mistral AI<br/>Voxtral TTS"]
    Bot --> Gemini["Google Gemini<br/>text + embeddings"]
    GitHub[GitHub] --> Deployer
```

Two Python processes (bot, worker). Three containers (redis, agent-memory-api, agent-memory-worker). Owner-only secrets in `/Users/voxbot/secrets/voxbot.env`.

---

## Soul — Memory Recall & Storage

![memory flow](assets/memory-recall-storage.png)

```mermaid
flowchart LR
    User([User]) --> Discord --> SoulCog --> MemoryService
    MemoryService -->|Path A| AMS["AMS + Redis<br/>vector index / ANN"]
    MemoryService -->|Path B| Embed["Gemini Embeddings"] --> File["FileStorage<br/>WAL + JSON"]

    MemoryService -.memory summary.-> Agent["Soul Agent"]
    SoulCog --> Agent
    Agent <--> Tools["Tools (optional)<br/>remember / recall / forget /<br/>react / rename / time"]
    Agent --> Resp["DiscordResponse"] --> Discord

    Tools -.remember.-> MemoryService
```

Backend chosen by `SOUL_MEMORY_BACKEND`: `redis` → AMS, `json` → FileStorage. Partition = Discord user id. Embedding = `gemini-embedding-2` (3072d); local hash fallback if API fails.

---

## Process Durability

![durability flow](assets/process-durability.png)

```mermaid
flowchart LR
    GH[GitHub origin] -->|1 poll| Dep["com.voxbot.deployer<br/>deploy.sh"]

    subgraph Host["macOS Host (launchd)"]
        Dep -->|2 build / test| FS
        Dep -.restart.-> BotJ["com.voxbot.bot"]
        Dep -.restart.-> WkJ["com.voxbot.worker"]
    end

    subgraph FS["Filesystem"]
        Code["releases/&lt;sha&gt;"] -->|3 switch symlink| Cur["current → releases/&lt;sha&gt;"] -->|4 record release| Dep2["deployed_sha<br/>last successful deploy"]
    end

    subgraph Docker["Docker (Colima VM)"]
        R[("Redis<br/>health surface")]
        AMS2["Agent Memory Server"]
    end

    BotJ -->|6 write health| R
    WkJ -->|6 write health| R
    Dep -->|7 verify in Redis| R
    R -->|success| OK([deploy committed])
    R -->|failure| Roll([rollback symlink])
```

Steps: poll → build/test → swap `current` → record `release_sha` → kill bot+worker (launchd respawns) → bot writes health to Redis → deployer verifies → commit `deployed_sha` or roll symlink back. Failed releases are deleted.

---

## User Eligibility

Single-tenant, owner-centric. No role tables.

| Tier | Detected by | Can do |
|---|---|---|
| **Owner** | `author.id == BOT_OWNER_ID` | Soul DMs, `/admin health`, `/admin restart`, receives tracebacks |
| **Whitelisted channel member** | `channel.id in SOUL_CHANNEL_IDS` | Talk to Soul, accumulate memory, `/voice`, `/health` |
| **Other guild member** | anywhere else | `/voice`, `/health` only |
| **Outsider** | DM, not owner | ignored |

Gates: `plugins/soul/cog.py:48` (whitelist), `checks.py:10` (`is_bot_admin`).

---

## File Index

| Concern | Path |
|---|---|
| Entry / CLI | `src/voxbot/__main__.py` |
| Bot class + plugin loader | `src/voxbot/bot.py` |
| Settings | `src/voxbot/settings.py` |
| Soul listener | `src/voxbot/plugins/soul/cog.py` |
| Soul agent + tools | `src/voxbot/plugins/soul/ai.py` |
| Soul prompt | `src/voxbot/plugins/soul/prompts/personality.mdc` |
| Memory service | `src/voxbot/plugins/soul/memory.py` |
| Storage (File / AMS) | `src/voxbot/plugins/soul/storage.py` |
| Embeddings | `src/voxbot/plugins/soul/embedding.py` |
| Periodic identity job | `src/voxbot/plugins/soul/jobs.py` |
| Health runtime | `src/voxbot/runtime/health.py` |
| Docket runtime | `src/voxbot/runtime/docket.py` |
| Deploy runbook | `deploy/macos/README.md` |
| Deploy rationale | `deploy/macos/ARCHITECTURE.md` |
| Docker infra | `deploy/macos/infra/compose.yaml` |
| CI | `.github/workflows/verify.yml` |
