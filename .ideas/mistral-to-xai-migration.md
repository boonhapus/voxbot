# Migrate TTS provider: Mistral → xAI

## Goal

Replace Mistral with xAI for all current TTS features. xAI's Custom Voices API is a superset of what Mistral provides: voice cloning, list, delete, plus a one-shot TTS endpoint. Same `voice_id` works across TTS, streaming, and Voice Agent — useful later but out of scope here.

This idea is **library-agnostic**. It changes the provider behind `/voice speak`, `/voice train`, `/voice delete`. It does not depend on the disnake → py-cord migration; the two can land independently.

## Provider mapping

| Current (Mistral) | New (xAI) |
|---|---|
| `MistralService.train_voice(name, audio_b64, filename)` | `POST /v1/custom-voices` (multipart, ≤120s, WAV/MP3/FLAC/OGG, 24kHz mono recommended) |
| `MistralService.delete_voice(voice_id)` | `DELETE /v1/custom-voices/{voice_id}` |
| `MistralService.sync_voices()` (list) | `GET /v1/custom-voices` (paginated: `limit`, `pagination_token`) |
| `MistralService.text_to_speech(text, voice_id) -> base64 mp3` | xAI TTS endpoint (URL/format confirmed in Phase 0) |
| `voices.json` (Mistral voice IDs) | `voices.json` schema repurposed to xAI `voice_id`s |

`VoxModel.hero_origins` is provider-agnostic — keeps working unchanged.

`dota_wiki.sample_voice_lines` already targets 20–30s — fits xAI's ≤120s limit.

## Phase 0 — API validation (decision gate)

Pure Python spike. `niquests` for HTTP (per project rule — don't add `requests`/`httpx`).

`.ideas/spikes/xai_clone_spike.py`:

1. POST a 30s WAV to `/v1/custom-voices` (multipart/form-data).
2. Confirm response shape, capture `voice_id`.
3. GET `/v1/custom-voices/{voice_id}` to verify metadata.
4. Hit the TTS endpoint with the cloned voice — verify audio comes back, format, sample rate.
5. DELETE the voice. Confirm subsequent GET returns 404.

**Decision gate**: only proceed once clone + TTS round-trip works.

Open questions for the spike to answer:
- Exact TTS endpoint URL and request shape (separate from realtime WS).
- Auth header format (`Authorization: Bearer ...` assumed).
- Concurrent voices per API key, rate limits.
- Audio output format defaults (PCM Linear16? MP3?). Configurable?

## `services/xai.py` — new module

REST surface:

```python
class XaiService:
    async def create_voice(name: str, audio: bytes, filename: str) -> str: ...
    async def delete_voice(voice_id: str) -> None: ...
    async def list_voices(limit: int = 100) -> list[Voice]: ...
    async def text_to_speech(text: str, voice_id: str) -> bytes: ...
```

Mirror the existing `MistralService` shape so the cog swap is mechanical. Persist the same `voices.json` (in-memory `VoiceData` class unchanged) — only the IDs inside change.

## Migration sequencing

1. **Phase 0 spike** (above). Decision gate.
2. **Build `services/xai.py`** with the four REST methods. Unit-test against the spike's findings.
3. **Add feature flag** `settings.tts_provider: Literal["mistral", "xai"] = "mistral"`. In `voice/cog.py`, branch on the flag — `mistral_service` or `xai_service`. Both wired up; only one used at a time.
4. **Re-train custom voices**: one-shot script `.ideas/spikes/migrate_voices.py` reads existing `voices.json`, fetches original samples (or re-runs `dota_wiki.sample_voice_lines` for hero voices), POSTs to xAI, writes `voices_xai.json`. Idempotent — skip already-migrated names.
5. **Flip default** to `xai`. Verify `/voice speak`, `/voice train`, `/voice delete` end-to-end in dev guild. Mistral path stays callable via flag if rollback needed.
6. **Soak** in dev guild. Confirm no regressions in voice quality, latency, error rates.
7. **Cleanup**: delete `services/mistral.py`, drop `mistralai` from `pyproject.toml`, delete old `voices.json` (or rename `voices_xai.json` → `voices.json`), remove `MISTRAL_API_KEY` from `settings.py`, remove `tts_provider` flag (xAI is now the only path).

## Files affected

- **New**: `src/voxbot/services/xai.py`, `.ideas/spikes/xai_clone_spike.py`, `.ideas/spikes/migrate_voices.py`.
- **Modified**: `src/voxbot/plugins/voice/cog.py` (flag + service swap), `src/voxbot/settings.py` (add `xai_api_key`, `tts_provider`).
- **Removed at step 7**: `src/voxbot/services/mistral.py`, `mistralai` dep, `MISTRAL_API_KEY`.

## Out of scope

- `/voice chat` realtime feature. Handled separately once both this migration and the py-cord migration land.
- xAI Voice Agent WebSocket. Not needed for one-shot TTS — that's a different endpoint.
- Library swap (disnake → py-cord) — separate idea (`pycord-migration.md`).

## Risks / gotchas

- Voice quality parity: xAI clones from up to 120s; Mistral has no documented limit. Hero scrapes (~20–30s) work for both, but xAI may want longer samples for best results — verify in spike.
- Rate limits unknown until Phase 0; bulk re-train script (step 4) may need throttling.
- API key separation: keep `MISTRAL_API_KEY` and `XAI_API_KEY` both set during steps 3–6 so rollback is one env flip.
- Step 4 script must be re-runnable. Name collisions (xAI voice with same name already exists) → either skip or DELETE+recreate; pick a strategy and document.

## References

- [xAI Voice Agent docs](https://docs.x.ai/developers/model-capabilities/audio/voice-agent)
- [xAI Custom Voices docs](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)
