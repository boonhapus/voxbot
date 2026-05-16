# Voxbot

<p align="center">
  <img src="assets/vox_avatar.png" width="200" alt="Voxbot Avatar">
</p>

A memory-capable Text-to-Speech (TTS) Discord bot built with **discord.py**, **Songbird (Rust)**, and powered by **Mistral** and **Gemini**.

## Features

- **High-Quality TTS**: Multi-voice support via Mistral AI.
- **Hero Origins**: Train custom voices by scraping Dota 2 hero voice lines.
- **Conversational Soul**: Pydantic-AI agent responds to whitelisted channels with personality and tool use.
- **Long-Term Memory**: Per-user facts persisted via a crash-safe file store, with an optional Redis Agent Memory Server backend.
- **Hybrid Worker**: Heavy and periodic work runs in a separate Docket worker process; the bot stays responsive for low-latency audio.
- **Git-Driven Deployment**: macOS launchd deployer polls the repo and rebuilds on new commits, with manual rollback via prior release symlinks.

## Commands

### Voice (`/voice`)
- `/voice speak`: Speak a message in your voice channel using a selected voice.
- `/voice train`: Train a custom voice from an audio sample or Dota 2 hero name.
- `/voice delete`: Remove a custom voice.
- `/voice listen`: 10-second audio "spike" that visualizes incoming voice traffic.

### Health (`/health`)
- `/health ping`: Check bot latency.

### Admin (`/admin`, owner-only)
- `/admin health`: Show deployment health, heartbeats, and last recorded error.
- `/admin restart`: Gracefully restart the bot process.

### Soul

Soul has no slash commands. It reacts to messages in whitelisted channels (`SOUL_CHANNEL_IDS`) and to DMs from the bot owner, dispatching reply/react/thread actions and storing facts it learns about people.

## Architecture

- **VoiceCog** (`plugins/voice`): Voice channel connections, TTS generation, and Songbird playback.
- **SoulCog** (`plugins/soul`): Pydantic-AI agent, memory tools, and message-driven personality.
- **AdminCog** (`plugins/admin`): Owner-only operational commands.
- **HealthCog** (`plugins/health`): Public latency check.
- **Runtimes** (`runtime/`):
  - `RedisHealthRuntime` — heartbeat, ready state, and error stream written to Redis.
  - `BotDocketRuntime` / worker — durable task scheduler (`docket`) for background jobs.

## Deployment

- [macOS deployment runbook](deploy/macos/README.md) — step-by-step bootstrap of a new server.
- [macOS deployment architecture](deploy/macos/ARCHITECTURE.md) — why the system is built this way (process model, rollback strategy, health surface).

## License

MIT
