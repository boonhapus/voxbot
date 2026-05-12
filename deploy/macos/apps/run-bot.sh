#!/bin/zsh
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

set -a
source /Users/voxbot/secrets/voxbot.env
set +a

# Wait for the current symlink to exist
until [ -L /Users/voxbot/apps/voxbot/current ]; do
  echo "Waiting for /Users/voxbot/apps/voxbot/current to be available..."
  sleep 5
done

cd /Users/voxbot/apps/voxbot/current
export VOXBOT_RELEASE_SHA="$(git rev-parse HEAD)"

echo $$ > /Users/voxbot/run/voxbot.pid
trap 'rm -f /Users/voxbot/run/voxbot.pid' EXIT

exec uv run --frozen voxbot
