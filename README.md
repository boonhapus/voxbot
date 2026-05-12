# Voxbot

<p align="center">
  <img src="assets/vox_avatar.png" width="200" alt="Voxbot Avatar">
</p>

A memory-capable Text-to-Speech (TTS) Discord bot built with **discord.py**, **Songbird (Rust)**, and powered by **Mistral** and **Gemini**.

## Features

- **High-Quality TTS**: Multi-voice support via Mistral AI.
- **Hero Origins**: Mimic Dota 2 hero personalities using scraped voice lines.
- **Long-Term Memory**: Remembers user preferences and past interactions using Redis Agent Memory.
- **Self-Healing**: Native macOS deployment with automatic recovery and git-driven updates.
- **Hybrid Worker**: Offloads heavy tasks to background workers while keeping audio low-latency.

## Commands

### Voice (`/voice`)
- `/voice speak`: Speak a message or generate an AI line from a prompt.
- `/voice train`: Train a custom voice from an audio file or Dota hero name.
- `/voice delete`: Remove a custom voice.
- `/voice listen`: A 10-second "spike" to visualize incoming audio data.

### Soul (`/soul`)
- `/soul chat`: Interact with the bot's personality.
- `/soul memory`: Manage what the bot remembers about you.

### Admin (Owner-only)
- `/admin health`: Check deployment status, latency, and service heartbeats.
- `/admin restart`: Gracefully restart the bot process.
- `/admin deploy`: View current release SHA and deployment state.

## Cogs & Architecture

- **VoiceCog**: Manages voice channel connections and Songbird audio playback.
- **SoulCog**: Orchestrates the AI personality and memory retrieval.
- **AdminCog**: Provides operational tools for the bot owner.
- **Health Runtime**: Reports real-time status to Redis for process supervision.

## Maintenance

For server setup and deployment details, see the [MacOS Deployment Guide](deploy/macos/README.md).

## License

MIT
