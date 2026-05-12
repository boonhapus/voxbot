# Migration Plan: Mistral Voxtral → Qwen3-TTS (Self-Hosted)

> **Status:** Not started. Bot still uses `MistralService` with Mistral API for all TTS.
> **Last audited:** 2026-05-12 — code references updated for pydantic-settings + slash commands.

## Overview

Replace the managed Mistral `voxtral-mini-tts-2603` API with a self-hosted
Qwen3-TTS inference server running on a local networked machine (RTX 3060,
12 GB VRAM). The bot's `/voice speak` and `/voice train` commands must continue
to work with zero functional regression.

---

## Background: vLLM and vLLM-Omni

### What is vLLM?

vLLM is a high-throughput, memory-efficient inference engine for LLMs. Its core
innovations are **PagedAttention** (non-contiguous KV-cache blocks, eliminating
memory waste) and **continuous batching** (dynamic request scheduling, no idle
GPU time between requests). It exposes an OpenAI-compatible REST API
out of the box.

### What is vLLM-Omni?

vLLM-Omni is an official extension of vLLM that adds support for multimodal and
audio models — specifically text-to-speech, speech-to-text, and vision-language
models. It ships the same `vllm serve` CLI and OpenAI-compatible endpoint, but
adds `/v1/audio/speech` (TTS) and `/v1/audio/transcriptions` (ASR) routes.

Qwen3-TTS reached **production-ready** status in vLLM-Omni in Q1 2026:
- Real-Time Factor (RTF): **0.34** (generates 1 s of audio in ~340 ms)
- Time to First Packet (TTFP): **~131 ms**

### Why vLLM-Omni over alternatives?

| Option | Latency | Custom voice | OpenAI-compat API | Maintained |
|---|---|---|---|---|
| vLLM-Omni | Best (RTF 0.34) | Yes (Base model) | Yes | Official |
| cornball-ai/qwen3-tts-api | Good | Yes | Yes | Community |
| ValyrianTech/Qwen3-TTS_server | Good | Yes | No | Community |
| Ollama | N/A | N/A | N/A | No TTS support |

---

## Model Selection

Qwen3-TTS ships three task-type variants. Each is a separate HuggingFace
checkpoint:

| Variant | Task type | Use case |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-Base` | `Base` | **Voice cloning** via reference audio |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `CustomVoice` | Preset speaker + style instruction |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `VoiceDesign` | Describe a voice in natural language |

**Chosen model: `Qwen3-TTS-12Hz-1.7B-Base`**

Reason: matches the existing `/voice train` workflow — user uploads a voice
sample, bot stores it, TTS calls attach it as `ref_audio` on each request.
VRAM footprint is ~4.5 GB weights + ~2.5 GB runtime buffers = ~7 GB total,
well within the RTX 3060's 12 GB.

---

## Phase 1 — Remote Server Setup

### Choose your OS

| OS | Pros | Cons |
|---|---|---|
| **Ubuntu 22.04 LTS** | Native vLLM support, best GPU driver ecosystem | CLI-focused, less familiar to some |
| **Windows Server 2022** | Familiar UI, good NVIDIA driver support | More RAM overhead, GPU PASSTHROUGH required for WSL/containers |

---

### Ubuntu Server Setup (22.04 LTS)

#### 1.1 OS / Driver prerequisites

Run as a non-root user with `sudo`.

```bash
# NVIDIA driver (535+ required for CUDA 12)
sudo apt install -y nvidia-driver-535
# Confirm
nvidia-smi

# CUDA toolkit (12.x)
# Download the runfile from developer.nvidia.com/cuda-downloads
# or use the distro package:
sudo apt install -y nvidia-cuda-toolkit

# Python 3.11+
sudo apt install -y python3.11 python3.11-venv python3-pip

# ffmpeg (needed for audio format conversion)
sudo apt install -y ffmpeg
```

#### 1.2 Create isolated Python environment

```bash
python3.11 -m venv ~/vllm-omni-env
source ~/vllm-omni-env/bin/activate
pip install --upgrade pip
```

#### 1.3 Install vLLM-Omni

Install from source (recommended — the package moves fast):

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
pip install -e .
```

Or install the stable release directly:

```bash
pip install vllm-omni
```

Verify:

```bash
python -c "import vllm; print(vllm.__version__)"
vllm --version
```

#### 1.4 Download the model

The model is pulled from HuggingFace on first `vllm serve`. Pre-download to
avoid timeout on first request:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
"
```

#### 1.5 Start the TTS server

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --omni \
  --port 8091 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enforce-eager
```

Flag notes:
- `--omni` — activates vLLM-Omni audio extensions
- `--enforce-eager` — disables CUDA graph capture; required for TTS models
- `--host 0.0.0.0` — binds to all interfaces so other machines can reach it

Verify it's up:

```bash
curl http://localhost:8091/health
# → {"status":"ok"}
```

#### 1.6 Run as a systemd service (production)

```ini
# /etc/systemd/system/qwen3-tts.service
[Unit]
Description=Qwen3-TTS vLLM-Omni server
After=network.target

[Service]
User=<your-user>
WorkingDirectory=/home/<your-user>/vllm-omni
ExecStart=/home/<your-user>/vllm-omni-env/bin/vllm serve \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --omni --port 8091 --host 0.0.0.0 \
  --trust-remote-code --enforce-eager
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-tts
sudo journalctl -fu qwen3-tts
```

#### 1.7 Network access

If the server and bot machine are on the same LAN, no additional config needed.
The bot connects to `http://<server-ip>:8091`.

If you need it accessible over the internet, put nginx in front with TLS:

```nginx
server {
    listen 443 ssl;
    server_name tts.yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8091;
        proxy_read_timeout 120s;
    }
}
```

---

### Windows Server Setup (2022)

#### 1.1 OS / Driver prerequisites

Install Windows Server 2022. Enable SSH via PowerShell:

```powershell
Install-WindowsFeature -Name OpenSSH.Server
```

Install NVIDIA drivers (535+). Download from NVIDIA Enterprise.

#### 1.2 Install Python 3.11+

Download from python.org or use winget:

```powershell
winget install Python.Python.3.11
```

Restart shell. Verify:

```powershell
python --version
```

#### 1.3 Install ffmpeg

```powershell
winget install FFmpeg.FFmpeg
# Restart PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

#### 1.4 Create isolated Python environment

```powershell
python -m venv C:\vllm-omni-env
C:\vllm-omni-env\Scripts\Activate.ps1
pip install --upgrade pip
```

#### 1.5 Install vLLM-Omni

```powershell
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
pip install -e .
```

Verify:

```powershell
python -c "import vllm; print(vllm.__version__)"
vllm --version
```

#### 1.6 Download the model

```powershell
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
"
```

#### 1.7 Start the TTS server

```powershell
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base `
  --omni `
  --port 8091 `
  --host 0.0.0.0 `
  --trust-remote-code `
  --enforce-eager
```

Verify:

```powershell
Invoke-RestMethod -Uri http://localhost:8091/health
```

#### 1.8 Run as a Windows Service (production)

Use NSSM (Non-Sucking Service Manager) or create a service via PowerShell:

```powershell
# Create the service
$binaryPath = "C:\vllm-omni-env\Scripts\python.exe"
$scriptPath = "C:\vllm-omni-env\Scripts\vllm"
$args = "serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8091 --host 0.0.0.0 --trust-remote-code --enforce-eager"

New-Service -Name Qwen3-TTS `
  -BinaryPathName "$binaryPath $scriptPath $args" `
  -DisplayName "Qwen3-TTS vLLM-Omni" `
  -StartupType Automatic

Start-Service Qwen3-TTS
```

Or use NSSM for better process management:

```powershell
# Download nssm, then:
nssm install Qwen3-TTS C:\vllm-omni-env\Scripts\python.exe C:\vllm-omni-env\Scripts\vllm "serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8091 --host 0.0.0.0 --trust-remote-code --enforce-eager"
nssm start Qwen3-TTS
```

#### 1.9 Firewall

Open port 8091:

```powershell
New-NetFirewallRule -DisplayName "Qwen3-TTS" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8091
```

#### 1.10 Network access

Same as Ubuntu — bot connects to `http://<server-ip>:8091`. For internet
exposure, use IIS with URL Rewrite + ARR or nginx under WSL.

---

## Phase 2 — API Reference

### Endpoint

```
POST http://<server-ip>:8091/v1/audio/speech
Content-Type: application/json
```

### Request body (voice cloning)

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "input": "Text to synthesize.",
  "voice": "custom",
  "response_format": "mp3",
  "task_type": "Base",
  "ref_audio": "data:audio/wav;base64,<base64-encoded-audio>",
  "ref_text": "Transcript of what was said in the reference clip."
}
```

### Response

Raw audio bytes in the requested format (`mp3`, `wav`, `opus`, etc.).
No base64 wrapping — unlike Mistral's `audio_data` field.

### Quick smoke test (Ubuntu)

```bash
REF_B64=$(base64 -w0 /path/to/reference.wav)

curl http://<server-ip>:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen3-TTS-12Hz-1.7B-Base\",
    \"input\": \"Hello from the other side.\",
    \"voice\": \"custom\",
    \"response_format\": \"mp3\",
    \"task_type\": \"Base\",
    \"ref_audio\": \"data:audio/wav;base64,${REF_B64}\",
    \"ref_text\": \"What the speaker says in the clip.\"
  }" \
  --output test.mp3

ffplay test.mp3
```

### Quick smoke test (Windows)

```powershell
$refBytes = [System.IO.File]::ReadAllBytes("C:\path\to\reference.wav")
$refB64 = [Convert]::ToBase64String($refBytes)

$body = @{
    model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    input = "Hello from the other side."
    voice = "custom"
    response_format = "mp3"
    task_type = "Base"
    ref_audio = "data:audio/wav;base64,$refB64"
    ref_text = "What the speaker says in the clip."
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8091/v1/audio/speech" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" `
  -OutFile C:\test.mp3

Start-Process C:\test.mp3
```

---

## Phase 3 — Bot Migration (Current Code)

### 3.1 What changes

| Component | Before | After |
|---|---|---|
| `Settings` | `mistral_api_key`, `voc_model` | `tts_base_url`, `voices_dir` |
| `MistralService` | Mistral SDK client | `httpx.AsyncClient` (or reuse `niquests`) |
| `_custom_voices` | `dict[str, str]` (name → Mistral voice ID) | `dict[str, Path]` (name → ref audio path) |
| `voice_train` | POST audio to Mistral, store returned ID | Save attachment bytes to `voices_dir/<name>.wav` |
| `voice_speak` | `mistral.audio.speech.complete_async(voice_id=...)` | POST to `/v1/audio/speech` with `ref_audio` bytes |
| Audio decode | `b64.b64decode(response.audio_data)` | `response.content` (raw bytes) |
| Preset voices | `en_paul_<emotion>` string IDs | Remove (no equivalent; use VoiceDesign model if needed) |

### 3.2 New config fields

```python
# in src/voxbot/settings.py
tts_base_url: str = "http://localhost:8091"
voices_dir: str = str(pathlib.Path.home() / ".voxbot" / "voices")
```

### 3.3 `voice_train` — new implementation

Save the attachment to disk instead of uploading to Mistral. Currently in
`src/voxbot/plugins/voice/cog.py:VoiceCog.voice_train`:

```python
# Replace MistralService.train_voice(...) with:
voice_name = pathlib.Path(audio.filename).stem.title()
dest = pathlib.Path(settings.voices_dir) / f"{voice_name}.wav"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(audio_bytes)
self.custom_voices[voice_name] = dest
```

### 3.4 `voice_speak` — new implementation

```python
# Replace MistralService.text_to_speech(...) with:
voice_path = self.custom_voices.get(voice)
ref_audio_b64 = base64.b64encode(voice_path.read_bytes()).decode()
payload = {
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": text,
    "voice": "custom",
    "response_format": "mp3",
    "task_type": "Base",
    "ref_audio": f"data:audio/wav;base64,{ref_audio_b64}",
    "ref_text": "",
}
async with httpx.AsyncClient() as client:
    resp = await client.post(
        f"{settings.tts_base_url}/v1/audio/speech",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    audio_bytes = resp.content  # raw MP3, no base64 unwrap
```

### 3.5 Startup: restore voices from disk

On bot start, scan `voices_dir` and repopulate `custom_voices`:

```python
# in VoiceCog.cog_load
voices_path = pathlib.Path(settings.voices_dir)
voices_path.mkdir(parents=True, exist_ok=True)
for f in voices_path.iterdir():
    if f.suffix in {".wav", ".mp3", ".ogg", ".flac"}:
        self.custom_voices[f.stem] = f
```

### 3.6 `.env` changes

```diff
-MISTRAL_API_KEY=sk-...
+TTS_BASE_URL=http://<server-ip>:8091
```

`voices_dir` defaults to `~/.voxbot/voices` — no env change needed unless
overriding.

### 3.7 Dependencies

`niquests` is already in `pyproject.toml` (used by other parts of the bot).
`mistralai` can be removed once the migration is complete.

---

## Phase 4 — Latency Considerations

| Source | Expected latency |
|---|---|
| vLLM-Omni TTFP | ~131 ms |
| RTX 3060 generation (per sentence) | ~300–500 ms |
| LAN round-trip | <5 ms |
| Mistral API (current) | ~400–800 ms (internet) |

Self-hosted on LAN will match or beat Mistral latency for short messages.
For longer messages (>2 sentences), consider chunking text at sentence
boundaries and streaming audio chunks to the voice channel as they arrive
using the `/v1/audio/speech` streaming endpoint (check vLLM-Omni docs for
`stream=true` support status).

---

## Checklist

### Server (Ubuntu)
- [ ] NVIDIA driver 535+ installed, `nvidia-smi` confirms GPU
- [ ] CUDA 12.x installed
- [ ] `vllm-omni` installed and `vllm --version` works
- [ ] Model downloaded: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- [ ] Server starts, `/health` returns `{"status":"ok"}`
- [ ] Smoke test curl produces valid MP3
- [ ] systemd service enabled and survives reboot
- [ ] Port 8091 reachable from bot machine

### Server (Windows)
- [ ] NVIDIA driver 535+ installed, `nvidia-smi` confirms (WSL) or GPU-z
- [ ] CUDA 12.x installed
- [ ] `vllm-omni` installed and `vllm --version` works
- [ ] Model downloaded: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- [ ] Server starts, `/health` returns `{"status":"ok"}`
- [ ] Smoke test Invoke-RestMethod produces valid MP3
- [ ] Windows service enabled and survives reboot
- [ ] Firewall rule allows port 8091

### Bot
- [ ] `tts_base_url` and `voices_dir` added to `Settings`
- [ ] `MistralService` replaced with httpx/niquests calls to Qwen3 endpoint
- [ ] `voice_train` saves audio to `voices_dir` instead of Mistral API
- [ ] `voice_speak` POSTs to `/v1/audio/speech` with `ref_audio`
- [ ] `cog_load` restores voices from disk on startup
- [ ] Audio decoded from raw bytes (not base64)
- [ ] `.env` updated: remove `MISTRAL_API_KEY`, add `TTS_BASE_URL`
- [ ] `mistralai` removed from dependencies
- [ ] End-to-end test: train a voice, speak with it

---

## Sources

- [vLLM-Omni GPU Installation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/)
- [vLLM-Omni Quickstart](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/)
- [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [vLLM-Omni Qwen3-TTS Online Serving](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/online_serving/qwen3_tts/)
- [QwenLM/Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS-12Hz-1.7B-Base HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [vllm-omni Qwen3-TTS offline inference examples](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts)
