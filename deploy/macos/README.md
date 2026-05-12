# Voxbot macOS Server Runbook

Step-by-step bootstrap of a new macOS server. Every command is meant to be run
as-is, top-to-bottom. After each block there is a `# verify:` line — run it and
confirm the expected output before moving on.

For *why* the system is built this way (process model, rollback strategy, etc.)
see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Assumptions

- You have **admin (sudo) access** on a Mac running macOS 13+ (Apple Silicon or Intel).
- You will run **all `sudo` commands as your admin user**, and switch into the
  `voxbot` user for non-sudo work.
- You have the values for every secret listed in step 5.

---

## 1. Install Homebrew + system tools

Run as your **admin user**.

```bash
# Install Homebrew (non-interactive).
NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Make brew available in the current shell.
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"

# Tools: git for clone, uv for python+venv, colima for the Docker VM,
# docker CLI (compose v2 ships built-in), jq for log inspection,
# redis-cli for deploy health probes.
brew install git uv colima docker jq redis

# Install Python 3.14 (required by pyproject.toml).
uv python install 3.14
```

```bash
# verify:
brew --version && git --version && uv --version && colima version && docker --version && redis-cli --version
```

---

## 2. Create the `voxbot` service user

macOS users created via `dscl` are non-login service accounts.

```bash
# Pick the next free UID >= 600 (service-account range).
NEXT_UID=$(($(dscl . -list /Users UniqueID | awk '{print $2}' | sort -n | tail -1) + 1))
echo "Using UID: $NEXT_UID"

sudo dscl . -create /Users/voxbot
sudo dscl . -create /Users/voxbot UserShell /bin/zsh
sudo dscl . -create /Users/voxbot RealName "Voxbot Service"
sudo dscl . -create /Users/voxbot UniqueID "$NEXT_UID"
sudo dscl . -create /Users/voxbot PrimaryGroupID 20   # staff
sudo dscl . -create /Users/voxbot NFSHomeDirectory /Users/voxbot
sudo mkdir -p /Users/voxbot
sudo chown -R voxbot:staff /Users/voxbot
```

```bash
# verify:
id voxbot
# expected: uid=<NEXT_UID>(voxbot) gid=20(staff) groups=20(staff)
```

---

## 3. Create the directory layout

```bash
sudo -u voxbot mkdir -p /Users/voxbot/{apps/voxbot,infra,logs,run,secrets,.ssh}
sudo chmod 700 /Users/voxbot/secrets /Users/voxbot/.ssh
```

```bash
# verify:
sudo -u voxbot ls -la /Users/voxbot
# expected: apps  infra  logs  run  secrets  .ssh  (secrets and .ssh = drwx------)
```

---

## 4. Set up the GitHub deploy key

```bash
# Generate an SSH key pair as voxbot.
sudo -u voxbot ssh-keygen -t ed25519 -N "" -f /Users/voxbot/.ssh/github_voxbot -C "voxbot-deploy"

# Configure a per-key host alias. The deployer uses `github.com-voxbot`
# to force this specific key.
sudo -u voxbot tee /Users/voxbot/.ssh/config >/dev/null <<'EOF'
Host github.com-voxbot
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_voxbot
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF
sudo chmod 600 /Users/voxbot/.ssh/config /Users/voxbot/.ssh/github_voxbot

# Print the public key. Copy it into GitHub:
# https://github.com/boonhapus/voxbot/settings/keys/new  (Allow read access only.)
sudo -u voxbot cat /Users/voxbot/.ssh/github_voxbot.pub
```

After adding the key on GitHub, test connectivity:

```bash
sudo -u voxbot ssh -T git@github.com-voxbot
# expected: "Hi boonhapus/voxbot! You've successfully authenticated..."
# (exit code 1 is normal for `ssh -T` on GitHub; the message is what matters.)
```

---

## 5. Write the secrets file

```bash
sudo -u voxbot tee /Users/voxbot/secrets/voxbot.env >/dev/null <<'EOF'
# ── Voxbot ──────────────────────────────────────────────────────────────
DISCORD_TOKEN=
DISCORD_OWNER_IDS=
MISTRAL_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=
REDIS_PASSWORD=
REDIS_URL=redis://:CHANGEME@localhost:6379/1

# Optional bot settings
SOUL_HOME_GUILD_ID=
SOUL_CHANNEL_IDS=
DEBUG_GUILD=

# ── Agent Memory Server (compose.yaml) ──────────────────────────────────
# Leave OPENAI_API_KEY empty if you only use Gemini.
OPENAI_API_KEY=
GENERATION_MODEL=gemini-2.5-flash
FAST_MODEL=gemini-2.5-flash-lite
SLOW_MODEL=gemini-2.5-pro
EMBEDDING_MODEL=text-embedding-004
REDISVL_VECTOR_DIMENSIONS=768
OLLAMA_API_BASE=
EOF
sudo chmod 600 /Users/voxbot/secrets/voxbot.env
sudo chown voxbot:staff /Users/voxbot/secrets/voxbot.env
```

Now edit the file and fill in every value:

```bash
sudo -u voxbot nano /Users/voxbot/secrets/voxbot.env
```

`REDIS_URL` **must** embed the same password as `REDIS_PASSWORD` (the bot uses
the URL; the container reads the password). Example:
`redis://:s3cret@localhost:6379/1` with `REDIS_PASSWORD=s3cret`.

```bash
# verify: no empty required values
sudo -u voxbot grep -E '^(DISCORD_TOKEN|MISTRAL_API_KEY|GOOGLE_API_KEY|REDIS_PASSWORD|REDIS_URL)=$' /Users/voxbot/secrets/voxbot.env
# expected: (no output)
```

---

## 6. Copy deploy scripts and plists into place

Clone the repo to a scratch location, copy the files, then delete the scratch
clone. The deployer maintains its own mirror going forward.

```bash
sudo -u voxbot git clone --depth 1 git@github.com-voxbot:boonhapus/voxbot.git /tmp/voxbot-bootstrap

# App scripts → /Users/voxbot/apps/voxbot/
sudo -u voxbot cp /tmp/voxbot-bootstrap/deploy/macos/apps/*.sh /Users/voxbot/apps/voxbot/
sudo -u voxbot chmod +x /Users/voxbot/apps/voxbot/*.sh

# Infra scripts + compose → /Users/voxbot/infra/
sudo -u voxbot cp /tmp/voxbot-bootstrap/deploy/macos/infra/* /Users/voxbot/infra/
sudo -u voxbot chmod +x /Users/voxbot/infra/*.sh

# launchd plists → /Library/LaunchDaemons/ (must be root-owned, mode 644)
sudo cp /tmp/voxbot-bootstrap/deploy/macos/launchd/*.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.voxbot.*.plist
sudo chmod 644 /Library/LaunchDaemons/com.voxbot.*.plist

rm -rf /tmp/voxbot-bootstrap
```

```bash
# verify:
ls -l /Users/voxbot/apps/voxbot/ /Users/voxbot/infra/ /Library/LaunchDaemons/com.voxbot.*.plist
# expected: scripts owned by voxbot:staff and executable; plists owned by root:wheel mode 644
```

---

## 7. Initialize Colima (one-time, interactive)

`infra-up.sh` will `colima start` automatically, but the first start downloads
a VM image and can prompt for credentials. Do it once interactively so launchd
runs cleanly later.

```bash
sudo -u voxbot -i colima start --cpu 2 --memory 4 --disk 30
sudo -u voxbot -i docker ps   # warms the Docker socket
sudo -u voxbot -i colima stop # infra-up.sh will restart it
```

```bash
# verify:
sudo -u voxbot -i colima status
# expected: colima is not running
```

---

## 8. Bootstrap services in order

Each `launchctl bootstrap` loads a plist and starts it immediately because all
four plists have `RunAtLoad=true`.

### 8a. Infra (Redis + Agent Memory Server)

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.infra.plist
```

```bash
# verify (give it ~60s on first run while images pull):
sudo -u voxbot -i docker ps --format 'table {{.Names}}\t{{.Status}}'
# expected: redis (healthy), agent-memory-api, agent-memory-worker — all "Up"
```

```bash
# verify Redis answers with the password:
sudo -u voxbot bash -c 'set -a; source /Users/voxbot/secrets/voxbot.env; set +a; redis-cli -a "$REDIS_PASSWORD" PING'
# expected: PONG
```

### 8b. Deployer (it will clone, build, test, then create `current`)

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.deployer.plist
```

Watch it work (first run = clone + `uv sync` + `pytest`, ~2–5 min):

```bash
tail -f /Users/voxbot/logs/deployer.out.log /Users/voxbot/logs/deployer.err.log
# Ctrl-C once you see "[voxbot-deploy] deployed <sha>"
```

Wait until the `current` symlink exists:

```bash
until [ -L /Users/voxbot/apps/voxbot/current ]; do echo "waiting for first release..."; sleep 5; done
ls -la /Users/voxbot/apps/voxbot/current
# expected: symlink → /Users/voxbot/apps/voxbot/releases/<sha>
```

### 8c. Worker, then bot

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.worker.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.bot.plist
```

```bash
# verify all four services are loaded and have a PID:
sudo launchctl list | grep com.voxbot
# expected: four lines, none with a "-" in the PID column for bot/worker/infra
```

---

## 9. Smoke test

```bash
# Bot heartbeat appears in Redis:
sudo -u voxbot bash -c 'set -a; source /Users/voxbot/secrets/voxbot.env; set +a; redis-cli -u "$REDIS_URL" GET voxbot:health:ready'
# expected: "true"

sudo -u voxbot bash -c 'set -a; source /Users/voxbot/secrets/voxbot.env; set +a; redis-cli -u "$REDIS_URL" GET voxbot:health:release_sha'
# expected: a git SHA matching /Users/voxbot/apps/voxbot/deployed_sha
```

In Discord, as a configured owner, run `/admin health`. You should see `Ready: true`
and a recent heartbeat.

Bootstrap complete.

---

## Routine operations

### Deploy a code change

```bash
git push origin main
# The server polls every 30s. Watch:
tail -f /Users/voxbot/logs/deployer.out.log
```

### Manual restart

```bash
sudo launchctl kickstart -k system/com.voxbot.bot
sudo launchctl kickstart -k system/com.voxbot.worker
```

### View logs

```bash
ls /Users/voxbot/logs/
# bot.{out,err}.log  worker.{out,err}.log  deployer.{out,err}.log  infra.{out,err}.log
```

### Roll back manually

```bash
ls -1t /Users/voxbot/apps/voxbot/releases   # newest first
sudo -u voxbot ln -sfn /Users/voxbot/apps/voxbot/releases/<old-sha> /Users/voxbot/apps/voxbot/current
echo "<old-sha>" | sudo -u voxbot tee /Users/voxbot/apps/voxbot/deployed_sha
sudo launchctl kickstart -k system/com.voxbot.worker
sudo launchctl kickstart -k system/com.voxbot.bot
```

### Tear down a service

```bash
sudo launchctl bootout system /Library/LaunchDaemons/com.voxbot.bot.plist
```
