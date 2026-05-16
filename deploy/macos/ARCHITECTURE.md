# Voxbot macOS Deployment — Architecture

The bootstrap commands live in [README.md](README.md). This document explains
the *why* behind each piece, so future maintainers can change things safely.

## Interactive debugging

The bot runs under the `voxbot` user. When you SSH in as `boonhapus` and need to
inspect or fix something in the bot's environment, use `sudo -u voxbot -i`:

```sh
sudo -u voxbot -i -- uv run --directory /Users/voxbot/apps/voxbot/current python -c "print('hello from voxbot')"
```

`-i` starts a login shell so `uv` is on `PATH` and the environment (secrets, etc.)
is set up correctly. Always use this form for any manual operation on the bot's
working tree or venv — running as `boonhapus` directly will miss env vars and
may create files owned by the wrong user.

**Quoting gotcha**: `sudo -u voxbot -i bash -c '...'` can mangle variable expansion
because the outer login shell (zsh) intercepts quoting before bash sees it. If a
command works with a literal value but fails with a `$VAR`, pipe through stdin
instead:
```sh
printf 'set -a; source /Users/voxbot/secrets/voxbot.env; set +a; redis-cli -a "$REDIS_PASSWORD" PING\n' | sudo -u voxbot -i bash
```

If you need to run multiple shell statements, pass the `bash -c` payload in
single quotes:

```sh
sudo -u voxbot -i -- bash -c 'set -a; source /Users/voxbot/secrets/voxbot.env; set +a; redis-cli -a "$REDIS_PASSWORD" PING'
```

Using double quotes around the `-c` payload can expand `$REDIS_*` in your
current shell before `sudo` switches users, which is a common source of
"works in one terminal, fails in another" behavior.

## High-level shape

```
                 ┌──────────────────────┐
                 │ GitHub (main branch) │
                 └──────────┬───────────┘
                            │ git fetch (every 30s)
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  launchd (system domain)                                   │
   │                                                            │
   │  com.voxbot.deployer  ── deploy.sh ── builds, tests, swaps │
   │  com.voxbot.bot       ── run-bot.sh ── Discord gateway     │
   │  com.voxbot.worker    ── run-worker.sh ── Docket worker    │
   │  com.voxbot.infra     ── infra-up.sh ── docker compose up  │
   └────────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Colima VM (Docker)                                        │
   │                                                            │
   │  redis (state + Docket queues)                             │
   │  agent-memory-api  ── HTTP, long-term memory               │
   │  agent-memory-worker ── ingest, summarize                  │
   └────────────────────────────────────────────────────────────┘
```

Four launchd jobs, two Python processes (bot + worker), three containers.

## Why launchd, not Docker, for the bot?

- **Audio**: the bot uses `discord-ext-songbird` (Rust) for low-latency voice.
  Running it on the host avoids Docker's audio/network indirection and gives
  direct access to the Discord voice UDP path.
- **Restart semantics**: launchd's `KeepAlive=true` + `ThrottleInterval=10`
  gives us a free crash-loop supervisor with backoff. No need to add a sidecar.
- **PID-1 problems**: Python in a container needs `tini` or similar to reap
  zombies; on the host this is free.

We still containerize Redis and the memory server because they're stateful,
upstream-maintained images and we don't need host I/O for them.

## Why poll for deploys instead of using webhooks?

- **No inbound ports**: the Mac sits behind home NAT. Polling avoids the
  reverse-tunnel / port-forward / static-IP problem.
- **Self-healing by design**: if the deployer process dies, launchd restarts
  it. If the bot crashes, launchd restarts it. If a release is broken, the
  deployer auto-rolls back. Every path converges back to a known-good state
  without operator intervention.
- **30-second cadence** is the tradeoff: deploys take up to ~30s longer than a
  webhook flow, but the system has no inbound dependency to maintain.

## Release directory layout

```
/Users/voxbot/apps/voxbot/
├── mirror/                ← git --mirror, single source for clones
├── releases/
│   ├── <sha-1>/           ← detached checkout, full .venv per release
│   ├── <sha-2>/
│   └── ...                ← `deploy.sh` keeps the 5 newest
├── current  → releases/<latest-good-sha>   (symlink)
├── release_sha            ← SHA `current` points at *right now*
├── deployed_sha           ← last SHA that passed the post-deploy health gate
├── deploy.lock/           ← mkdir-based lock to prevent concurrent runs
├── deploy.sh
├── run-bot.sh
└── run-worker.sh
```

Each release has its own `.venv`. Disk cost is real (~300MB/release) but it
means rollback is just `ln -sfn`, no env-rebuild, no race.

## The deploy loop in pseudocode

```
fetch mirror
SHA = mirror's main HEAD
if deployed_sha == SHA: exit 0

if releases/SHA doesn't exist:
    clone, checkout, uv sync, ruff check, pytest
    on any failure: rm -rf releases/SHA, exit 1

remember OLD_SHA = deployed_sha, OLD_CURRENT = readlink current
ln -sfn releases/SHA current
write SHA → release_sha          # runtime scripts read this on next launch

kill bot + worker process groups (SIGKILL; SIGTERM hangs on docket shutdown)
sleep 5

for 30 * 2s:
    if redis(voxbot:health:release_sha) == SHA AND redis(voxbot:health:ready) == "true":
        write SHA → deployed_sha    # commit: this SHA is known-good
        prune old releases (keep newest 5)
        exit 0

# health check failed → roll back
rm -rf releases/SHA
ln -sfn OLD_CURRENT current
write OLD_SHA → release_sha
write OLD_SHA → deployed_sha
kill bot + worker process groups
exit 1
```

Key invariants:
- `release_sha` tracks what `current` points at *right now* — flipped at the
  same moment as the symlink, so `run-bot.sh` and `run-worker.sh` always read
  the SHA that matches the venv they're about to exec.
- `deployed_sha` only advances *after* the new release passes the post-deploy
  health gate. On rollback, both files revert to the previous SHA together, so
  `deployed_sha == release_sha` is the steady-state invariant.
- Failed releases are deleted, so the next poll re-clones and re-tests rather
  than re-rolling a stale failed artifact.
- The `mkdir deploy.lock` lock is atomic; concurrent deployer runs (e.g. if a
  run takes longer than 30s) skip cleanly.

## Why SIGKILL the bot, not SIGTERM?

`pydocket` has a shutdown bug where its worker task can hang indefinitely on
SIGTERM. `kill -KILL` on the **process group** (negative pid) reliably tears
down the `uv run` parent *and* its Python child. launchd's KeepAlive then
respawns a fresh process pointing at the new `current` symlink within
`ThrottleInterval` seconds.

This is a tradeoff: we lose graceful Discord gateway disconnects, but we gain
deterministic deploy times. Discord handles the abrupt disconnect fine
(re-IDENTIFYs on next connect).

## Health surface

`RedisHealthRuntime` writes these keys every `health_heartbeat_seconds` (10):

| Key                                | Meaning                          |
|------------------------------------|----------------------------------|
| `voxbot:health:ready`              | "true" after `on_ready` fired    |
| `voxbot:health:release_sha`        | SHA the running bot was built at |
| `voxbot:health:heartbeat`          | ISO timestamp, last write        |
| `voxbot:health:heartbeat_unix`     | Same, unix epoch                 |
| `voxbot:health:latency_ms`         | Discord gateway latency          |
| `voxbot:health:last_error`         | Most recent error summary        |
| `voxbot:health:restart_requested`  | Reason if `request_restart` ran  |
| `voxbot:health:restart_count`      | Lifetime restart counter         |

The worker process writes the same key set under the `voxbot:worker:health:`
prefix, so both processes are observable independently.

The deployer reads `release_sha` + `ready` to decide whether a deploy
succeeded. The `/admin health` Discord command reads the same keys.

## Command-tree sync gate

Discord rate-limits global slash-command syncs to 2/hour. The bot hashes its
command tree on every startup and stores the hash at `~/.voxbot/commands.sha`.
It only calls `tree.sync()` when the hash changes, so routine deploys (which
don't touch command definitions) don't burn rate-limit budget.

If you ever need to force a re-sync, delete the file:

```bash
sudo -u voxbot rm /Users/voxbot/.voxbot/commands.sha
sudo launchctl kickstart -k system/com.voxbot.bot
```

## Failure modes and recovery

| Symptom                                  | What happened                              | Recovery                          |
|------------------------------------------|--------------------------------------------|-----------------------------------|
| `/admin health` shows old `release_sha`  | New deploy failed health check → rolled back | Check deployer logs; push fix     |
| Deployer log: "deploy already running"   | Previous run still holds `deploy.lock`     | Wait, or `rm -rf deploy.lock`     |
| Bot in crash loop                        | New release builds but fails at runtime    | Auto-rollback after 60s; check logs |
| Worker shows old SHA but bot is current  | Worker plist not kicked by deploy.sh       | `launchctl kickstart` it manually |
| `colima status` says "not running"       | VM crashed or was stopped                  | `infra-up.sh` re-runs every 5min  |
| Redis container restarting               | Likely `REDIS_PASSWORD` mismatch, or port 6379 already bound (for example an SSH tunnel) | Recheck `voxbot.env`, stop the tunnel/process on 6379, re-up infra |

## Things deliberately not done

- **No webhook deploy**: see above.
- **No blue/green or hot reload**: full process restart per deploy. Simpler,
  and the Discord reconnect is cheap.
- **No GitHub Actions runner on the box**: tests run inside the deploy loop so
  the same machine that ships the code is the one that gates it.
- **No Spotlight exclusion for `releases/`**: add `mdutil -i off
  /Users/voxbot/apps` if disk indexing becomes a problem.
- **No log rotation**: launchd doesn't rotate stdout/stderr. If logs grow,
  configure `newsyslog` or pipe through `multilog`.
