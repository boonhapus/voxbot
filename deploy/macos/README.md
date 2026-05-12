# Mac Server Deployment Guide (Voxbot)

This guide explains how to set up and maintain Voxbot on a macOS server. It uses native `launchd` for process management and a git-driven automated deployment loop.

## Architecture Overview

- **Process Supervisor (`launchd`)**: Manages the life of the bot, worker, and deployer processes. If they crash, macOS restarts them.
- **Container Runtime (`Colima/Docker`)**: Runs infrastructure services like Redis (state/queues) and Agent Memory Server (long-term memory).
- **Automated Deployer**: A script that polls GitHub, builds new code, runs tests, and swaps versions atomically.

## One-Time Bootstrap

Follow these steps to set up a new server.

### 1. System Preparation
- Create a dedicated macOS user: `voxbot`.
- Create the base directories:
  ```bash
  mkdir -p /Users/voxbot/{apps,infra,logs,run,secrets}
  ```
- Install prerequisites:
  - **Homebrew**: `/bin/bash -c "$(curl -fsSL ...)"`
  - **Tools**: `brew install git uv colima docker docker-compose jq redis`
  - **Python**: `uv python install 3.14`

### 2. Secrets Configuration
Create `/Users/voxbot/secrets/voxbot.env` (chmod 600). This file must contain:
- `DISCORD_TOKEN`, `DISCORD_OWNER_IDS`
- API Keys: `MISTRAL_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`
- Redis: `REDIS_PASSWORD`, `REDIS_URL`

### 3. Git & Infrastructure
- **Deploy Key**: Generate an SSH key for the `voxbot` user and add it to GitHub as a read-only deploy key.
- **Infra**: Copy `deploy/macos/infra/*` to `/Users/voxbot/infra/`. Run `infra-up.sh`. This starts Redis and the Agent Memory Server.

### 4. Service Activation
- Copy `deploy/macos/launchd/*.plist` to `/Library/LaunchDaemons/`.
- Change ownership to `root:wheel` and permissions to `644`.
- Bootstrap the services:
  ```bash
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.infra.plist
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.deployer.plist
  # Once deployer creates /Users/voxbot/apps/voxbot/current:
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.worker.plist
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.voxbot.bot.plist
  ```

## Routine Operations

### Deploying Updates
Push code to the **`main`** branch. The server polls for changes every 30 seconds.
1. It builds the new version in a separate folder.
2. It runs `pytest`.
3. If successful, it restarts the bot.

### Monitoring
- **Logs**: View logs in `/Users/voxbot/logs/`.
- **Status**: Run `/admin health` in Discord.
- **Manual Restart**: `sudo launchctl kickstart -k system/com.voxbot.bot`

## Why this approach?
We use `launchd` instead of Docker for the bot itself to ensure high-performance audio (Songbird/Rust) and easy access to the host's Discord gateway. We use a "release-based" folder structure so that the bot can rollback instantly if a new version fails to start.