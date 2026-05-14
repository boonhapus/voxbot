#!/bin/zsh
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

APP=/Users/voxbot/apps/voxbot
REPO_URL="${VOXBOT_REPO_URL:-git@github.com-voxbot:boonhapus/voxbot.git}"
BRANCH="${VOXBOT_DEPLOY_BRANCH:-main}"
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
  git clone "$APP/mirror" "$REL" || exit 1
  git -C "$REL" checkout --detach "$SHA" || { rm -rf "$REL"; exit 1; }

  cd "$REL"
  uv sync --frozen --all-groups || { rm -rf "$REL"; exit 1; }
  uv run --frozen ruff check . || { rm -rf "$REL"; exit 1; }
  uv run --frozen pytest || { rm -rf "$REL"; exit 1; }
fi

OLD_CURRENT="$(readlink "$APP/current" 2>/dev/null || true)"
ln -sfn "$REL" "$APP/current"

# Copy deploy scripts from the new release so Mac-side script fixes
# (run-bot.sh, run-worker.sh) are picked up without manual re-bootstrap.
cp "$REL/deploy/macos/apps/"*.sh "$APP/"

# Force-kill old processes (and their children) so launchd restarts cleanly.
# Negative pid = kill process group; covers uv -> python subprocesses.
# (SIGTERM hangs due to docket shutdown crash.)
kill_release_pidfile() {
  local pidfile="$1"
  [ -f "$pidfile" ] || return 0
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [ -n "$pid" ] || return 0
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [ -n "$pgid" ]; then
    kill -KILL -"$pgid" 2>/dev/null || true
  fi
  kill -KILL "$pid" 2>/dev/null || true
}

kill_release_pidfile /Users/voxbot/run/voxbot-worker.pid
kill_release_pidfile /Users/voxbot/run/voxbot.pid

sleep 5

for i in {1..30}; do
  HEALTH_SHA="$(redis-cli -u "$REDIS_URL" GET voxbot:health:release_sha 2>/dev/null || true)"
  READY="$(redis-cli -u "$REDIS_URL" GET voxbot:health:ready 2>/dev/null || true)"

  if [ "$HEALTH_SHA" = "$SHA" ] && [ "$READY" = "true" ]; then
    echo "$SHA" > "$APP/deployed_sha"
    echo "$LOG_PREFIX deployed $SHA"
    # Prune old releases, keeping the 5 most recent plus current+previous.
    ls -1t "$APP/releases" 2>/dev/null | tail -n +6 | while read -r old; do
      [ -n "$old" ] && rm -rf "$APP/releases/$old"
    done
    exit 0
  fi

  sleep 2
done

echo "$LOG_PREFIX new release failed health check: $SHA"

# Drop the failed release so the next poll re-clones and re-tests instead of
# silently re-rolling the same bad artifact.
rm -rf "$REL"

# Restore the previous release. If OLD_CURRENT was pruned, pick the newest
# surviving release. If nothing survives, remove the broken symlink so
# run-bot.sh doesn't spin on a dead link.
if [ -n "$OLD_CURRENT" ] && [ -d "$OLD_CURRENT" ]; then
  RESTORE="$OLD_CURRENT"
else
  RESTORE="$(ls -1dt "$APP/releases"/*/ 2>/dev/null | head -1)"
  RESTORE="${RESTORE%/}"
fi

if [ -n "$RESTORE" ]; then
  ln -sfn "$RESTORE" "$APP/current"
  OLD_SHA="$(git -C "$RESTORE" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$OLD_SHA" ]; then
    echo "$OLD_SHA" > "$APP/deployed_sha"
  fi
else
  rm -f "$APP/current"
fi

kill_release_pidfile /Users/voxbot/run/voxbot-worker.pid
kill_release_pidfile /Users/voxbot/run/voxbot.pid

exit 1
