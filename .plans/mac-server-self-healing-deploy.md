# Plan: Mac Server Self-Healing Deploy Runtime

## Progress

Updated as work proceeds. Pick up from the first unchecked box.

### Server bootstrap (Mac, run as `boonhapus` unless noted)

- [x] 1. Service account `voxbot` (uid 502) + dirs created, secrets at 700
- [x] 2. Brew tools installed: `uv`, `colima`, `docker`, `docker-compose`, `redis-cli`
       (Intel Mac → brew prefix is `/usr/local`, not `/opt/homebrew` — fix script PATHs)
- [x] 3. Python 3.14.5 installed via `sudo -H -u voxbot uv python install 3.14`
- [x] 4. Colima started (`--cpu 2 --memory 4 --disk 30`) and `docker run hello-world` passes
- [x] 5. `/Users/voxbot/secrets/voxbot.env` written with real secrets, chmod 600
- [x] 6. SSH deploy key generated for voxbot, added to GitHub as read-only deploy key
- [x] 7. `~/.ssh/config` host alias `github.com-voxbot` set up, `ssh -T` succeeds
- [ ] 8. `infra-up.sh` + `compose.yaml` placed at `/Users/voxbot/infra/`, `infra-up.sh` runs clean
- [ ] 9. Redis PING + Agent Memory `/v1/health` smoke tests pass
- [ ] 10. launchd plists installed under `/Library/LaunchDaemons/`, services bootstrapped

### Repo code (deploy-minimal subset — leaves Agent Memory bot wiring for the other plan)

- [x] 11. Settings: redis/docket/health/owner fields added
- [x] 12. Redis async helper + health runtime module
- [x] 13. Bot lifecycle: SIGTERM, `request_shutdown`, `request_restart` (exit 75)
- [x] 14. Owner-only `/admin` cog (health, restart, deploy)
- [x] 15. CLI: `voxbot worker` command (Docket background jobs)
- [x] 16. `deploy/macos/` scripts: `deploy.sh`, `run-bot.sh`, `run-worker.sh`, `infra-up.sh`, `compose.yaml`
- [x] 17. `deploy/macos/launchd/` plist templates
- [x] 18. Tests for settings, health runtime, lifecycle, admin cog

### Decisions locked in

- Branch: `personality` (stay on it, no new branch)
- GitHub repo: `git@github.com:boonhapus/voxbot.git` (deploy key path uses `github.com-voxbot` host alias)
- Embedding option: **C (Ollama `nomic-embed-text`, 768 dims)** — Gemini API does not support embeddings;
  user has Ollama installed locally (currently via Electron desktop app under boonhapus, not a system daemon)
- AMS in scope for infra (compose runs api + worker), but bot keeps `SOUL_MEMORY_BACKEND=json` until the
  Agent Memory plan (`.plans/redis-agent-memory-and-docket.md`) is folded in
- Intel Mac: brew prefix `/usr/local`; all scripts must use `/usr/local/bin` in PATH, not `/opt/homebrew/bin`

### Known follow-ups (out of deploy-minimal scope)

- Migrate Ollama from the Electron desktop app to a system LaunchDaemon before flipping
  `SOUL_MEMORY_BACKEND` away from `json`. Until then, Ollama dies on logout/reboot and AMS embedding
  is best-effort. Bot is unaffected because it doesn't call AMS yet.
- The wider Agent Memory + Docket migration lives in `.plans/redis-agent-memory-and-docket.md`.

---

## Goal

Deploy Voxbot on a clean Mac server with:

- Native process supervision through `launchd`.
- Redis-backed state, health, memory, and Docket queues.
- Redis Agent Memory Server running beside Redis.
- A git-driven deploy loop so local development can ship by pushing to the
  configured branch.
- Clean bot and worker restarts with low downtime.
- Owner-only Discord commands for health checks and intentional restarts.

After the one-time server bootstrap, normal deploy flow should be:

```text
edit locally -> commit -> push main -> Mac deployer pulls/builds/tests -> bot restarts
```

The operator should not need to SSH into the server for routine deploys once
the setup is stable.

---

## Explicit Non-Goals

- No Kubernetes.
- No public inbound webhook requirement for the first version.
- No second live Discord bot instance for zero-downtime handoff.
- No automatic production code mutation by the running bot.

A single unsharded Discord bot token should only have one production process
connected at a time. The target is low downtime, not true zero downtime.

---

## Human-Required Work

The agent can write repo files, scripts, launchd plist templates, and bot code.
The following actions require a human because they need physical/server access,
GitHub/Discord console access, or secrets:

1. Create or choose the macOS service account.
2. Install Xcode command line tools and Homebrew on the Mac.
3. Install `git`, `uv`, `colima`, Docker CLI, Docker Compose, `jq`, and
   `redis-cli`.
4. Start Colima for the first time.
5. Create the server secret file with real API keys and tokens.
6. Generate the deploy SSH key on the server and add it to GitHub as a
   read-only deploy key.
7. Copy/install launchd plist files into `/Library/LaunchDaemons`.
8. Run `sudo launchctl bootstrap ...` for the first time.
9. Confirm Discord application owner/admin IDs and set them in config.

Any step below marked `HUMAN REQUIRED` must not be assumed complete by an agent.
An agent can prepare scripts/templates for those steps, but a human must run or
approve them on the Mac.

---

## Target Architecture

```text
macOS launchd
  com.voxbot.infra
    starts Colima if needed
    runs Docker Compose for Redis and Agent Memory

  com.voxbot.deployer
    polls git
    builds candidate release in an isolated release directory
    runs tests
    atomically points current -> release
    terminates bot and worker processes
    launchd restarts them
    verifies Redis health heartbeat
    rolls back symlink on failed startup

  com.voxbot.bot
    trusted discord.py gateway process
    owns Discord token
    owns live Discord gateway and voice-client state
    writes health heartbeat to Redis
    starts bot-local Docket runtime if enabled

  com.voxbot.worker
    trusted external Docket worker
    runs pure/background jobs that do not need live Discord client state

Docker / Colima
  redis
    Redis Stack / RedisVL backing memory and Docket queues

  agent-memory-api
    Redis Agent Memory REST API

  agent-memory-worker
    Redis Agent Memory Docket worker
```

Use localhost-only bindings for Redis and Agent Memory API:

```text
127.0.0.1:6379 -> Redis
127.0.0.1:8000 -> Agent Memory API
```

Do not expose Redis or the memory API publicly.

---

## Server Directory Layout

Use a dedicated macOS user named `voxbot` unless the human chooses another
name.

```text
/Users/voxbot/
  apps/
    voxbot/
      mirror/                 # bare git mirror
      releases/
        <git-sha>/            # immutable release checkouts
      current -> releases/<git-sha>
      deploy.sh
      run-bot.sh
      run-worker.sh
      deployed_sha

  infra/
    compose.yaml
    infra-up.sh

  logs/
    bot.out.log
    bot.err.log
    worker.out.log
    worker.err.log
    deployer.out.log
    deployer.err.log
    infra.out.log
    infra.err.log

  run/
    voxbot.pid
    voxbot-worker.pid

  secrets/
    voxbot.env
```

Secrets must be readable only by the service user:

```zsh
chmod 700 /Users/voxbot/secrets
chmod 600 /Users/voxbot/secrets/voxbot.env
```

---

## Server Bootstrap

### 1. Create Service Account

`HUMAN REQUIRED` on the Mac:

```zsh
sudo sysadminctl -addUser voxbot -home /Users/voxbot
sudo mkdir -p /Users/voxbot/{apps,infra,logs,run,secrets}
sudo chown -R voxbot:staff /Users/voxbot
sudo chmod 700 /Users/voxbot/secrets
```

If the user already exists, only create the missing directories and fix
ownership.

### 2. Install Tools

`HUMAN REQUIRED` on the Mac:

```zsh
xcode-select --install

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install git uv colima docker docker-compose jq redis
uv python install 3.14
colima start --cpu 2 --memory 4 --disk 30
docker version
docker compose version
redis-cli --version
```

Notes:

- `uv python install 3.14` is needed because Voxbot currently declares
  `requires-python = ">=3.14"`.
- Colima is the first-pass Docker runtime. Docker Desktop is acceptable if the
  human prefers it, but the service scripts below assume Colima.

---

## Secrets and Environment

Create `/Users/voxbot/secrets/voxbot.env`.

`HUMAN REQUIRED` because this contains real secrets:

```env
# Discord
DISCORD_TOKEN=replace-me
DISCORD_OWNER_IDS=123456789012345678
DEBUG_GUILD=

# Voxbot model providers
MISTRAL_API_KEY=replace-me
GOOGLE_API_KEY=replace-me
GEMINI_API_KEY=replace-me

# Redis
REDIS_PASSWORD=replace-me-long-random-value
REDIS_URL=redis://:replace-me-long-random-value@127.0.0.1:6379/1

# Docket
DOCKET_URL=redis://:replace-me-long-random-value@127.0.0.1:6379/1
DOCKET_NAME=voxbot
DOCKET_ENABLED=true

# Redis Agent Memory API
SOUL_MEMORY_BACKEND=agent_memory
SOUL_MEMORY_SERVER_URL=http://127.0.0.1:8000
SOUL_MEMORY_NAMESPACE=voxbot:soul
SOUL_AUTO_EXTRACT_ENABLED=false
SOUL_AUTO_EXTRACT_CHANNEL_ID=1306464265703522325
SOUL_AUTO_EXTRACT_NAMESPACE=voxbot:soul:auto-test

# Agent Memory model config
GENERATION_MODEL=gemini/gemini-1.5-flash
FAST_MODEL=gemini/gemini-1.5-flash
SLOW_MODEL=gemini/gemini-1.5-pro

# Choose exactly one embedding approach for semantic search.
# Option A: OpenAI embeddings, simplest operationally.
OPENAI_API_KEY=replace-me-if-using-openai-embeddings
EMBEDDING_MODEL=text-embedding-3-small
REDISVL_VECTOR_DIMENSIONS=1536

# Option B: Vertex AI embeddings, Google-only but needs Vertex auth setup.
# EMBEDDING_MODEL=vertex_ai/text-embedding-004
# REDISVL_VECTOR_DIMENSIONS=768

# Option C: Local Ollama embeddings, no external embedding API.
# EMBEDDING_MODEL=ollama/nomic-embed-text
# REDISVL_VECTOR_DIMENSIONS=768
# OLLAMA_API_BASE=http://host.docker.internal:11434
```

Important model distinction:

- Generation models read, summarize, classify, and extract memory text.
- Embedding models convert memory text into vectors so Redis can search by
  semantic similarity.

Gemini can be reused for generation via `GEMINI_API_KEY` and
`GENERATION_MODEL=gemini/...`. An embedding model is still required for
semantic long-term memory search.

---

## Infrastructure: Redis and Agent Memory

Create `/Users/voxbot/infra/compose.yaml`:

```yaml
services:
  redis:
    image: redis/redis-stack-server:latest
    restart: unless-stopped
    ports:
      - "127.0.0.1:6379:6379"
    environment:
      REDIS_ARGS: "--requirepass ${REDIS_PASSWORD} --appendonly yes"
    volumes:
      - redis-data:/data

  agent-memory-api:
    image: redislabs/agent-memory-server:latest
    restart: unless-stopped
    depends_on:
      - redis
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      REDIS_URL: "redis://:${REDIS_PASSWORD}@redis:6379/0"
      GEMINI_API_KEY: "${GEMINI_API_KEY}"
      GOOGLE_API_KEY: "${GOOGLE_API_KEY}"
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      GENERATION_MODEL: "${GENERATION_MODEL}"
      FAST_MODEL: "${FAST_MODEL}"
      SLOW_MODEL: "${SLOW_MODEL}"
      EMBEDDING_MODEL: "${EMBEDDING_MODEL}"
      REDISVL_VECTOR_DIMENSIONS: "${REDISVL_VECTOR_DIMENSIONS}"
      LONG_TERM_MEMORY: "true"
      ENABLE_DISCRETE_MEMORY_EXTRACTION: "true"
      INDEX_ALL_MESSAGES_IN_LONG_TERM_MEMORY: "false"
      AUTH_MODE: "disabled"
      LOG_LEVEL: "INFO"
      DOCKET_NAME: "memory-server"
    command:
      - agent-memory
      - api
      - --host
      - "0.0.0.0"
      - --port
      - "8000"

  agent-memory-worker:
    image: redislabs/agent-memory-server:latest
    restart: unless-stopped
    depends_on:
      - redis
    environment:
      REDIS_URL: "redis://:${REDIS_PASSWORD}@redis:6379/0"
      GEMINI_API_KEY: "${GEMINI_API_KEY}"
      GOOGLE_API_KEY: "${GOOGLE_API_KEY}"
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      GENERATION_MODEL: "${GENERATION_MODEL}"
      FAST_MODEL: "${FAST_MODEL}"
      SLOW_MODEL: "${SLOW_MODEL}"
      EMBEDDING_MODEL: "${EMBEDDING_MODEL}"
      REDISVL_VECTOR_DIMENSIONS: "${REDISVL_VECTOR_DIMENSIONS}"
      DOCKET_NAME: "memory-server"
      LOG_LEVEL: "INFO"
    command:
      - agent-memory
      - task-worker
      - --concurrency
      - "2"

volumes:
  redis-data:
```

Create `/Users/voxbot/infra/infra-up.sh`:

```zsh
#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

set -a
source /Users/voxbot/secrets/voxbot.env
set +a

if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi

colima status >/dev/null 2>&1 || colima start --cpu 2 --memory 4 --disk 30

cd /Users/voxbot/infra
docker compose --env-file /Users/voxbot/secrets/voxbot.env up -d
```

`HUMAN REQUIRED` once the files exist on the Mac:

```zsh
chmod +x /Users/voxbot/infra/infra-up.sh
/Users/voxbot/infra/infra-up.sh
curl http://127.0.0.1:8000/v1/health
redis-cli -u "$REDIS_URL" PING
```

If the Agent Memory health endpoint path changes in a future version, use the
current Redis Agent Memory docs to adjust the smoke test.

---

## Git Deploy Access

Use a read-only GitHub deploy key for the production server.

`HUMAN REQUIRED` on the Mac as the `voxbot` user:

```zsh
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/voxbot_deploy -C "voxbot-prod-readonly"
cat ~/.ssh/voxbot_deploy.pub
```

`HUMAN REQUIRED` in GitHub:

1. Open the Voxbot repository settings.
2. Add the public key as a deploy key.
3. Keep it read-only.

Create `/Users/voxbot/.ssh/config`:

```sshconfig
Host github.com-voxbot
  HostName github.com
  User git
  IdentityFile ~/.ssh/voxbot_deploy
  IdentitiesOnly yes
```

`HUMAN REQUIRED` verification:

```zsh
ssh -T git@github.com-voxbot
```

Expected result is an authentication success message that may also say shell
access is not provided.

---

## Release-Based Deployer

The deployer should build new code in a release directory while the old bot
keeps running. It should only switch `current` after dependencies and tests
pass.

Create `/Users/voxbot/apps/voxbot/deploy.sh`:

```zsh
#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

APP=/Users/voxbot/apps/voxbot
REPO_URL="git@github.com-voxbot:OWNER/voxbot.git"
BRANCH=main
LOCK="$APP/deploy.lock"
LOG_PREFIX="[voxbot-deploy]"

mkdir -p "$APP/releases" /Users/voxbot/run

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$LOG_PREFIX deploy already running"
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT

set -a
source /Users/voxbot/secrets/voxbot.env
set +a

if [ ! -d "$APP/mirror" ]; then
  git clone --mirror "$REPO_URL" "$APP/mirror"
fi

git -C "$APP/mirror" fetch --prune origin
SHA="$(git -C "$APP/mirror" rev-parse "refs/heads/$BRANCH")"

if [ -f "$APP/deployed_sha" ] && [ "$(cat "$APP/deployed_sha")" = "$SHA" ]; then
  exit 0
fi

REL="$APP/releases/$SHA"

if [ ! -d "$REL" ]; then
  git clone "$APP/mirror" "$REL"
  git -C "$REL" checkout --detach "$SHA"

  cd "$REL"
  uv sync --frozen --all-groups
  uv run ruff check .
  uv run pytest
fi

OLD_CURRENT="$(readlink "$APP/current" 2>/dev/null || true)"
ln -sfn "$REL" "$APP/current"
echo "$SHA" > "$APP/deployed_sha"

if [ -f /Users/voxbot/run/voxbot-worker.pid ]; then
  kill -TERM "$(cat /Users/voxbot/run/voxbot-worker.pid)" 2>/dev/null || true
fi

if [ -f /Users/voxbot/run/voxbot.pid ]; then
  kill -TERM "$(cat /Users/voxbot/run/voxbot.pid)" 2>/dev/null || true
fi

# The launchd services should restart the worker and bot after they exit.
# Give the bot a short window to write its new heartbeat.
for i in {1..30}; do
  HEALTH_SHA="$(redis-cli -u "$REDIS_URL" GET voxbot:health:release_sha 2>/dev/null || true)"
  READY="$(redis-cli -u "$REDIS_URL" GET voxbot:health:ready 2>/dev/null || true)"

  if [ "$HEALTH_SHA" = "$SHA" ] && [ "$READY" = "true" ]; then
    echo "$LOG_PREFIX deployed $SHA"
    exit 0
  fi

  sleep 2
done

echo "$LOG_PREFIX new release failed health check: $SHA"

if [ -n "$OLD_CURRENT" ] && [ -d "$OLD_CURRENT" ]; then
  ln -sfn "$OLD_CURRENT" "$APP/current"
  OLD_SHA="$(git -C "$OLD_CURRENT" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$OLD_SHA" ]; then
    echo "$OLD_SHA" > "$APP/deployed_sha"
  fi

  if [ -f /Users/voxbot/run/voxbot-worker.pid ]; then
    kill -TERM "$(cat /Users/voxbot/run/voxbot-worker.pid)" 2>/dev/null || true
  fi

  if [ -f /Users/voxbot/run/voxbot.pid ]; then
    kill -TERM "$(cat /Users/voxbot/run/voxbot.pid)" 2>/dev/null || true
  fi
fi

exit 1
```

Agent implementation notes:

- Replace `OWNER/voxbot.git` with the real GitHub repository path.
- Ensure `pytest` is present in the dev dependency group before relying on the
  deploy script.
- If tests need external credentials, split fast deploy tests from integration
  tests and run only the fast suite here.
- Do not run migrations that mutate production data before the release is
  selected unless those migrations are explicitly backward-compatible.

---

## Bot and Worker Runner Scripts

Create `/Users/voxbot/apps/voxbot/run-bot.sh`:

```zsh
#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

set -a
source /Users/voxbot/secrets/voxbot.env
set +a

cd /Users/voxbot/apps/voxbot/current
export VOXBOT_RELEASE_SHA="$(git rev-parse HEAD)"

echo $$ > /Users/voxbot/run/voxbot.pid
trap 'rm -f /Users/voxbot/run/voxbot.pid' EXIT

exec uv run voxbot
```

Create `/Users/voxbot/apps/voxbot/run-worker.sh`:

```zsh
#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

set -a
source /Users/voxbot/secrets/voxbot.env
set +a

cd /Users/voxbot/apps/voxbot/current
export VOXBOT_RELEASE_SHA="$(git rev-parse HEAD)"

echo $$ > /Users/voxbot/run/voxbot-worker.pid
trap 'rm -f /Users/voxbot/run/voxbot-worker.pid' EXIT

exec uv run voxbot worker
```

`HUMAN REQUIRED` once copied onto the Mac:

```zsh
chmod +x /Users/voxbot/apps/voxbot/deploy.sh
chmod +x /Users/voxbot/apps/voxbot/run-bot.sh
chmod +x /Users/voxbot/apps/voxbot/run-worker.sh
```

---

## launchd Services

Create plist templates in the repo first, then the human installs them to:

```text
/Library/LaunchDaemons/com.voxbot.infra.plist
/Library/LaunchDaemons/com.voxbot.deployer.plist
/Library/LaunchDaemons/com.voxbot.bot.plist
/Library/LaunchDaemons/com.voxbot.worker.plist
```

### Infra Service

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.voxbot.infra</string>
  <key>UserName</key><string>voxbot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/voxbot/infra/infra-up.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>/Users/voxbot/logs/infra.out.log</string>
  <key>StandardErrorPath</key><string>/Users/voxbot/logs/infra.err.log</string>
</dict>
</plist>
```

### Deployer Service

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.voxbot.deployer</string>
  <key>UserName</key><string>voxbot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/voxbot/apps/voxbot/deploy.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>/Users/voxbot/logs/deployer.out.log</string>
  <key>StandardErrorPath</key><string>/Users/voxbot/logs/deployer.err.log</string>
</dict>
</plist>
```

### Bot Service

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.voxbot.bot</string>
  <key>UserName</key><string>voxbot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/voxbot/apps/voxbot/run-bot.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>/Users/voxbot/logs/bot.out.log</string>
  <key>StandardErrorPath</key><string>/Users/voxbot/logs/bot.err.log</string>
</dict>
</plist>
```

### Worker Service

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.voxbot.worker</string>
  <key>UserName</key><string>voxbot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/voxbot/apps/voxbot/run-worker.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>/Users/voxbot/logs/worker.out.log</string>
  <key>StandardErrorPath</key><string>/Users/voxbot/logs/worker.err.log</string>
</dict>
</plist>
```

`HUMAN REQUIRED` installation:

```zsh
sudo cp com.voxbot.*.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.voxbot.*.plist
sudo chmod 644 /Library/LaunchDaemons/com.voxbot.*.plist

sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.infra.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.deployer.plist

# Start these after the deployer has produced /Users/voxbot/apps/voxbot/current.
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.worker.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.bot.plist
```

Useful human operations:

```zsh
sudo launchctl print system/com.voxbot.bot
sudo launchctl kickstart -k system/com.voxbot.bot
sudo launchctl bootout system/com.voxbot.bot
tail -f /Users/voxbot/logs/bot.err.log
```

---

## Required Voxbot Code Changes

This section is for the implementation agent working in the repository.

### Dependencies

Add dependencies:

```toml
"redis>=5.0.0",
"pydocket>=0.20.2",
"agent-memory-client>=0.14.0",
```

Add dev dependency if missing:

```toml
"pytest>=8.0.0",
```

Keep `agent-memory-server` out of the Voxbot project environment because the
server runs in Docker.

### Settings

Add to `Settings`:

```python
redis_url: str = "redis://localhost:6379/1"

docket_url: str | None = None
docket_name: str = "voxbot"
docket_enabled: bool = True

discord_owner_ids: str | None = None

soul_memory_backend: str = "json"
soul_memory_server_url: str = "http://localhost:8000"
soul_memory_namespace: str = "voxbot:soul"
soul_auto_extract_enabled: bool = False
soul_auto_extract_channel_id: str | None = "1306464265703522325"
soul_auto_extract_namespace: str = "voxbot:soul:auto-test"

health_enabled: bool = True
health_heartbeat_seconds: int = 10
deployment_id: str | None = None
```

Use `settings.docket_url or settings.redis_url` in Docket factory code.

### CLI Shape

Extend `voxbot.__main__`:

```text
voxbot          # starts Discord bot
voxbot worker   # starts external Docket worker for pure jobs
```

The bot command must:

- Set up logging.
- Run the Discord bot.
- Return the bot's requested exit code.
- Handle `SIGTERM` as a clean shutdown path on macOS.

The worker command must:

- Set up logging.
- Create Docket.
- Register only pure/background jobs.
- Run until interrupted.

Do not register jobs that need a live `discord.Client` in the external worker.

### Bot Lifecycle

Add an explicit restart/shutdown path:

```python
class VoxBot(commands.Bot):
    exit_code: int = 0

    async def request_shutdown(self, *, reason: str, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        await self.close()

    async def request_restart(self, *, reason: str) -> None:
        await self.request_shutdown(reason=reason, exit_code=75)
```

Implementation details:

- The process does not restart itself.
- It exits cleanly.
- `launchd` restarts it because `KeepAlive` is true.
- Exit code `75` can be treated as "intentional restart requested" in logs.

When possible, prefer an `asyncio.run()` entrypoint with signal handlers over
`bot.run(...)` so `SIGTERM` can close Discord cleanly.

### Health Runtime

Add a module such as:

```text
src/voxbot/runtime/health.py
```

Responsibilities:

- Maintain one Redis connection through `redis.asyncio`.
- Write heartbeat keys every `settings.health_heartbeat_seconds`.
- Mark readiness true after `on_ready`.
- Mark readiness false during shutdown.
- Record release sha from `VOXBOT_RELEASE_SHA`.
- Record last error summary without secret values.

Redis keys:

```text
voxbot:health:ready              "true" | "false"
voxbot:health:heartbeat          ISO timestamp
voxbot:health:heartbeat_unix     integer seconds
voxbot:health:release_sha        git sha
voxbot:health:latency_ms         integer
voxbot:health:last_error         short string
voxbot:health:restart_requested  short string/reason
voxbot:health:restart_count      integer, optional
```

Optional health checks:

- Redis ping.
- Agent Memory `/v1/health`.
- Discord websocket latency threshold.
- Docket worker heartbeat key.
- Event loop stall detection.

Critical failures should either:

- Disable the failing subsystem if the bot can continue safely, or
- Request shutdown with nonzero exit code and let `launchd` restart.

Add a restart budget before automatic self-triggered restarts:

```text
max 3 automatic restarts per 10 minutes
after budget is exhausted, mark unhealthy and avoid crash-looping
```

### Admin Cog

Add an owner-only admin cog, for example:

```text
src/voxbot/plugins/admin/__init__.py
src/voxbot/plugins/admin/cog.py
```

Commands:

```text
/admin health
/admin restart reason:<optional text>
/admin deploy
```

Behavior:

- Use `await bot.is_owner(interaction.user)` or configured
  `DISCORD_OWNER_IDS`.
- Responses should be ephemeral.
- `/admin restart` writes `voxbot:health:restart_requested`, responds to the
  user, then calls `await bot.request_restart(...)`.
- `/admin health` reads Redis health keys and reports release sha, ready state,
  heartbeat age, Discord latency, and last error.
- `/admin deploy` can report `deployed_sha`/health state later. Do not make it
  run arbitrary shell commands.

### Worker Heartbeat

The external Docket worker should also write:

```text
voxbot:worker:health:heartbeat
voxbot:worker:health:release_sha
voxbot:worker:health:ready
```

This lets the bot and deployer distinguish "Discord is healthy" from "Docket
worker is healthy".

---

## Low-Downtime Deploy Strategy

The deployer never mutates the active checkout. It builds:

```text
/Users/voxbot/apps/voxbot/releases/<new-sha>
```

while the old release continues running.

Deployment order:

1. Fetch branch.
2. Resolve target sha.
3. If sha is already deployed, exit.
4. Clone mirror to release directory.
5. Check out target sha detached.
6. `uv sync --frozen --all-groups`.
7. Run fast checks.
8. Switch `current` symlink.
9. Send `SIGTERM` to worker.
10. Send `SIGTERM` to bot.
11. launchd restarts both from `current`.
12. Deployer polls Redis health for the expected release sha.
13. If health fails, switch `current` back and terminate processes again.

Expected downtime:

- Build/test time: zero user-facing downtime because old bot stays connected.
- Restart time: Discord disconnect/reconnect window only.

---

## Rollback

Automatic rollback:

- If the new release does not write `voxbot:health:ready=true` and the expected
  `voxbot:health:release_sha` within the deployer's timeout, the deployer
  restores `current` to the previous release and restarts bot/worker again.

Manual rollback:

`HUMAN REQUIRED` only if Discord/admin commands are unavailable:

```zsh
cd /Users/voxbot/apps/voxbot
ls -lt releases
ln -sfn /Users/voxbot/apps/voxbot/releases/<known-good-sha> current
echo <known-good-sha> > deployed_sha
kill -TERM "$(cat /Users/voxbot/run/voxbot-worker.pid)"
kill -TERM "$(cat /Users/voxbot/run/voxbot.pid)"
```

Later, add `/admin rollback` that rolls back only to a known previous release
recorded by the deployer. Do not let it accept arbitrary paths.

---

## Security Requirements

- Redis must bind only to `127.0.0.1`.
- Agent Memory API must bind only to `127.0.0.1`.
- `AUTH_MODE=disabled` is acceptable only because the API is localhost-only.
- GitHub deploy key must be read-only.
- Secrets file must be `chmod 600`.
- Do not print `.env` values in logs.
- Owner-only Discord commands must be gated before doing any work.
- Deployer must not run unbounded shell from Discord.
- Branch protection is recommended on the deployment branch.
- The deployer should run tests before switching the active release.

---

## Testing Plan

Unit tests:

- Settings parse new Redis/Docket/health/admin fields.
- Health runtime writes expected Redis keys with a fake Redis client.
- Restart request sets the bot exit code and calls close.
- Admin commands reject non-owners.
- Admin restart writes the restart reason and triggers shutdown.
- Docket worker registration includes only pure jobs.
- Bot-local registration does not run in external worker mode.

Deployment script tests:

- Use a temporary git repo and temporary app directory.
- First deploy creates mirror, release, current symlink, and deployed sha.
- Second deploy with same sha exits without restart.
- Failed tests do not change `current`.
- Failed health check rolls back to old `current`.

Manual smoke test:

1. `HUMAN REQUIRED`: run server bootstrap.
2. `HUMAN REQUIRED`: install secrets.
3. `HUMAN REQUIRED`: add GitHub deploy key.
4. Start `com.voxbot.infra`.
5. Confirm Redis responds to `PING`.
6. Confirm Agent Memory health endpoint responds.
7. Start `com.voxbot.deployer`.
8. Confirm first release appears under `releases/<sha>`.
9. Start `com.voxbot.worker`.
10. Start `com.voxbot.bot`.
11. Confirm Discord bot appears online.
12. Run `/admin health`.
13. Push a harmless commit.
14. Confirm deployer builds a new release and restarts the bot.
15. Confirm `/admin health` reports the new release sha.
16. Run `/admin restart`.
17. Confirm bot goes offline briefly and comes back with the same release sha.

---

## Implementation Order

1. Add settings for Redis, Docket, Agent Memory, health, and owner IDs.
2. Add Redis async helper/factory.
3. Add health runtime and tests.
4. Convert bot entrypoint to support clean `SIGTERM`.
5. Add `request_shutdown` and `request_restart` lifecycle methods.
6. Add owner-only admin cog.
7. Add `voxbot worker` command.
8. Add Docket factory and job registration split.
9. Add server deployment script templates under a repo path such as
   `deploy/macos/`.
10. Add launchd plist templates under `deploy/macos/launchd/`.
11. [x] Add README or plan references explaining human-required bootstrap steps.
12. Test deploy scripts locally with a temp repo where possible.
13. Human runs server bootstrap.
14. Human starts launchd services.
15. Push a test commit and validate no-SSH deployment.

---

## Source References

- Apple launchd / launchctl documentation:
  https://www.manpagez.com/man/8/launchd/
- Colima getting started:
  https://colima.readthedocs.io/
- Docker Compose documentation:
  https://docs.docker.com/compose/
- Redis Stack Docker image:
  https://hub.docker.com/r/redis/redis-stack-server
- Redis Agent Memory Server:
  https://redis.github.io/agent-memory-server/
- Redis Agent Memory configuration:
  https://redis.github.io/agent-memory-server/configuration/
- Redis Agent Memory LLM providers:
  https://redis.github.io/agent-memory-server/llm-providers/
- Redis Agent Memory Docker image:
  https://hub.docker.com/r/redislabs/agent-memory-server
- Astral uv documentation:
  https://docs.astral.sh/uv/
- GitHub deploy keys:
  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys
- discord.py API documentation:
  https://discordpy.readthedocs.io/
