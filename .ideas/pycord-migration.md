# Migrate disnake → py-cord

## Goal

Replace `disnake` with `py-cord` so we get native voice receive (`VoiceClient.start_recording` + Sink). This is a forward port of the **current** codebase — not a revert of commit `7449632`.

Why: `disnake` has zero voice-receive support (confirmed in `disnake/voice_client.py` — only send-side methods). Issue #178 has been "planned, low priority" since Nov 2021; PR #239 was abandoned. We need receive to build any conversational voice feature.

## Library landscape

- **discord.py** — Rapptz original. No native receive. Community ext `discord-ext-voice-recv` (single maintainer, discord.py-only).
- **py-cord** — fork from the 2021 hiatus. **Native receive**: `VoiceClient.start_recording(sink, callback)`, Sink subclasses, `unpack_audio` → 48kHz stereo PCM. Opus optional via libopus.
- **disnake** — current. Slash-command focused. No receive.

py-cord wins: receive is first-class, slash-command ergonomics comparable to disnake, fork heritage means most edits are mechanical.

## Phase 0 — Streaming recording spike (decision gate)

Pure throwaway script under `.ideas/spikes/pycord_recording_spike.py`. Validates that py-cord can do **realtime** streaming, not just buffer-then-callback.

The default Sinks (`WaveSink`, `MP3Sink`, `PCMSink`, etc.) buffer everything in memory until `stop_recording()` fires the callback once. For chat use we need frames as they arrive.

Spike must prove a custom `Sink` subclass can:

1. Override `write(data, user)` to push frames into an `asyncio.Queue` from the audio thread (via `loop.call_soon_threadsafe`).
2. Yield 20ms PCM frames (~3840 bytes 48kHz/16-bit/stereo) per user with no buffering.
3. Resample 48k stereo → 24k mono via `audioop` (verify quality + CPU acceptable).
4. Survive a 60s recording without backpressure / dropped frames.

Reference snippet (full one in earlier conversation):

```python
class StreamingSink(Sink):
    def __init__(self, loop, queue):
        super().__init__()
        self.loop = loop
        self.queue = queue

    def write(self, data, user):
        # Audio thread → asyncio loop
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (user, data))

    def cleanup(self):
        self.finished = True
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
```

**Decision gate**: only proceed with the port if the spike streams cleanly. If it doesn't, fall back to disnake + Rust songbird microservice (separate idea — not covered here).

Spike runs against a throwaway py-cord bot in a dev guild. No need to migrate the real codebase first.

## API delta — disnake → py-cord

Translate the existing files in place. Mostly mechanical.

| disnake | py-cord |
|---|---|
| `import disnake` / `from disnake.ext import commands` | `import discord` / `from discord.ext import commands` |
| `disnake.Attachment` | `discord.Attachment` |
| `disnake.FFmpegPCMAudio` | `discord.FFmpegPCMAudio` |
| `commands.slash_command(name=..., group=...)` | `SlashCommandGroup` or `@bot.slash_command(...)` with `parent` |
| `commands.Param(default=..., description=..., choices=..., autocomplete=...)` | `discord.Option(type, description=..., default=..., choices=..., autocomplete=...)` |
| `inter: disnake.ApplicationCommandInteraction` | `ctx: discord.ApplicationContext` |
| `inter.response.defer()` / `inter.edit_original_response(content=...)` | `ctx.defer()` / `ctx.edit(content=...)` (or `ctx.followup.send`) |
| Autocomplete: separate callback registered on `Param` | Autocomplete: callback bound via `Option(autocomplete=...)`; signature `(ctx, value)` |
| `bot.voice_clients`, `vc.move_to(channel)` | same |
| `bot.add_cog(...)` | `bot.add_cog(...)` (sync) — verify async-vs-sync per version |

## Files to port

From `git status` and current layout:

- `src/voxbot/__main__.py` — bot construction, intents, login.
- `src/voxbot/bot.py` — `Bot` subclass / setup hooks.
- `src/voxbot/plugins/voice/cog.py` — slash group, autocomplete, attachments. Largest delta.
- `src/voxbot/plugins/voice/state.py` — provider-agnostic, no changes expected.
- `src/voxbot/plugins/healthcheck.py` — likely a plain command; small port.
- `src/voxbot/tasks.py` — background tasks (`@tasks.loop` API exists in py-cord with same shape).
- `pyproject.toml` — swap `disnake` for `py-cord`. Confirm extras (`[voice]`).
- `uv.lock` — regenerate.

## Implementation plan

1. **Phase 0 spike** (above). Decision gate.
2. Add `py-cord` to a feature branch alongside `disnake`. Pick one cog at a time and port; do not run both libs in production.
3. Port `__main__.py` + `bot.py` first — bot must boot.
4. Port `voice/cog.py` — biggest surface area. Verify `/voice speak`, `/voice train`, `/voice delete` still work end-to-end in dev guild.
5. Port `healthcheck.py` and `tasks.py`.
6. Drop `disnake` from `pyproject.toml`, regenerate lock, delete any disnake-specific compat code.
7. Single PR, mechanical only — no behavior changes, no new features. Land it cleanly so blame stays readable.

## What this idea does **not** include

- xAI / Mistral provider work — separate idea (`mistral-to-xai-migration.md`).
- `/voice chat` feature — depends on this migration **and** the xAI migration. Build after both land.

## Risks / gotchas

- Custom `Sink.write` runs on the **audio decode thread**. Always cross to asyncio with `call_soon_threadsafe` or `run_coroutine_threadsafe`. Don't `await` inside.
- 20ms frames at 48kHz/16-bit/stereo = ~3840 bytes. Bound the queue to apply backpressure.
- Multi-speaker rooms: `Sink.write` is per-user. Decide upstream how to combine (sum-and-clip, loudest, designated speaker).
- `RawData` (pre-decode Opus) is also exposed by py-cord — useful if a downstream consumer accepts Opus and we want to skip the decode/re-encode round trip.
- Verify `py-cord` supports the disnake voice-state event signatures we already rely on.

## References

- [py-cord voice API](https://docs.pycord.dev/en/stable/api/voice.html)
- [py-cord audio_recording.py example](https://github.com/Pycord-Development/pycord/blob/master/examples/audio_recording.py)
- [py-cord Sink base class](https://github.com/Pycord-Development/pycord/blob/master/discord/sinks/core.py)
- [disnake issue #178 — voice recording (open since 2021)](https://github.com/DisnakeDev/disnake/issues/178)
