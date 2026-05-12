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
echo "$SHA" > "$APP/deployed_sha"

# Force-kill old processes so launchd restarts immediately
# (SIGTERM hangs due to docket shutdown crash)
if [ -f /Users/voxbot/run/voxbot-worker.pid ]; then
  kill -KILL "$(cat /Users/voxbot/run/voxbot-worker.pid)" 2>/dev/null || true
fi

if [ -f /Users/voxbot/run/voxbot.pid ]; then
  kill -KILL "$(cat /Users/voxbot/run/voxbot.pid)" 2>/dev/null || true
fi

sleep 5

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
    kill -KILL "$(cat /Users/voxbot/run/voxbot-worker.pid)" 2>/dev/null || true
  fi

  if [ -f /Users/voxbot/run/voxbot.pid ]; then
    kill -KILL "$(cat /Users/voxbot/run/voxbot.pid)" 2>/dev/null || true
  fi
fi

exit 1
