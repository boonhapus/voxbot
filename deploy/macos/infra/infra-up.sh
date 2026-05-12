#!/bin/zsh
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

set -a
source /Users/voxbot/secrets/voxbot.env
set +a

if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi

colima status >/dev/null 2>&1 || colima start --cpu 2 --memory 4 --disk 30

cd /Users/voxbot/infra
docker compose --env-file /Users/voxbot/secrets/voxbot.env up -d
