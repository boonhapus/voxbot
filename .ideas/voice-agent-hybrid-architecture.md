# Hybrid Voice Architecture: Mistral → x.ai Voice Agent

## Overview

Migrate from Mistral TTS to x.ai Voice Agent, supporting two interaction modes:

1. **Command-based TTS** (`/voice speak`) — one-shot text → audio synthesis
2. **Conversational voice chat** (`/voice chat on/off`) — continuous bidirectional audio + LLM reasoning

Architecture uses Python/disnake for Discord orchestration + Rust microservice for reliable audio I/O and x.ai streaming.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  voxbot (Python/disnake)                             │
│  • Discord commands (/voice speak, /voice chat)      │
│  • Voice channel connection management               │
│  • Audio playback to Discord                         │
│  • Conversation flow orchestration                   │
└──────────────────────────────────────────────────────┘
            ↕ (TCP socket or Unix domain socket)
┌──────────────────────────────────────────────────────┐
│  voxbot-audio-service (Rust/serenity + songbird)     │
│  • Connect to Discord voice independently            │
│  • Continuous PCM audio capture                      │
│  • Bidirectional streaming ↔ x.ai WebSocket          │
│  • Audio encoding/decoding (PCM ↔ Opus)              │
│  • Session state (active conversations)              │
└──────────────────────────────────────────────────────┘
            ↕ (WebSocket)
       x.ai Voice Agent API
       (grok-voice-think-fast-1.0)
```

### Why This Split?

| Concern | Why Rust service? |
|---------|---|
| **Audio receiving** | disnake/discord.py AudioSink unreliable; songbird (serenity) battle-tested |
| **Continuous capture** | Rust async handles voice packets more predictably |
| **x.ai streaming** | Dedicated WebSocket + audio buffering without blocking Discord I/O |
| **Isolation** | Bot crash doesn't kill audio service; service restart doesn't disconnect Discord |

---

## Component Breakdown

### Python Bot (voxbot)

**New/modified files:**
```
src/voxbot/
  ├── audio_service_client.py       # IPC client to Rust service
  ├── services/
  │   ├── xai.py                    # x.ai TTS + Voice Agent client
  │   └── (remove) mistral.py
  └── plugins/
      └── voice/
          ├── cog.py                # Updated: add /voice chat commands
          └── state.py              # Enhanced: track conversation sessions
```

**Key changes:**
- Replace Mistral TTS calls with x.ai TTS API
- Add IPC client to communicate with Rust service
- New command group: `/voice chat on|off`
- Maintain `/voice speak` (unchanged paradigm, new backend)

### Rust Microservice (voxbot-audio-service)

**New repository structure:**
```
voxbot-audio-service/
  ├── Cargo.toml
  ├── src/
  │   ├── main.rs                   # Server + IPC listener
  │   ├── discord_voice.rs          # serenity + songbird integration
  │   ├── xai_client.rs             # x.ai WebSocket streaming
  │   ├── audio_processor.rs        # PCM/Opus codec
  │   ├── ipc_protocol.rs           # Message types
  │   └── session.rs                # Conversation state
  ├── .env.example
  └── README.md
```

---

## IPC Protocol

### Connection

Unix domain socket (Linux/macOS) or named pipe (Windows):
```
Windows: \\.\pipe\voxbot-audio
Unix:    /tmp/voxbot-audio.sock
```

Or TCP for simplicity during dev:
```
localhost:9999
```

### Message Format

All messages are JSON. Each side sends/receives one complete JSON object per message (newline-delimited):

```json
{"type": "start_chat", "guild_id": 123456, "channel_id": 789012, "user_id": 111213}
```

### Messages

#### Bot → Service

**Start conversation:**
```json
{
  "type": "start_chat",
  "guild_id": 123456789,
  "channel_id": 987654321,
  "user_id": 111213141,
  "xai_api_key": "sk-...",
  "voice_id": "custom-voice-abc123"
}
```

**Stop conversation:**
```json
{
  "type": "stop_chat",
  "guild_id": 123456789
}
```

#### Service → Bot

**Chat started:**
```json
{
  "type": "chat_started",
  "guild_id": 123456789,
  "session_id": "sess_xyz"
}
```

**Audio response chunk** (for streaming playback):
```json
{
  "type": "audio_chunk",
  "guild_id": 123456789,
  "data": "base64-encoded-opus-or-pcm",
  "duration_ms": 2048
}
```

**Chat ended:**
```json
{
  "type": "chat_ended",
  "guild_id": 123456789,
  "reason": "user_stopped | timeout | error"
}
```

**Error:**
```json
{
  "type": "error",
  "guild_id": 123456789,
  "message": "Failed to connect to Discord voice channel"
}
```

---

## Installation

### Python Bot Changes

#### 1. Add x.ai dependency

```bash
uv add anthropic  # For x.ai (uses same SDK)
```

Or if x.ai has a dedicated SDK (check docs):
```bash
uv add xai-sdk
```

#### 2. Create `src/voxbot/services/xai.py`

```python
"""x.ai Voice Agent client."""

import asyncio
import json
from typing import AsyncGenerator
import structlog

_LOGGER = structlog.get_logger(__name__)


class XaiVoiceClient:
    """Handles x.ai Voice Agent WebSocket and TTS."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None

    async def text_to_speech(self, text: str, voice_id: str) -> bytes:
        """Generate TTS audio. Returns raw audio bytes (Opus or MP3)."""
        # TODO: Check x.ai TTS endpoint (may be separate from Voice Agent API)
        # Placeholder: assume they have a /v1/audio/speech endpoint like OpenAI
        pass

    async def start_voice_session(
        self, voice_id: str, instructions: str = ""
    ) -> str:
        """Start a Voice Agent session. Returns session_id."""
        # Connect to wss://api.x.ai/v1/realtime
        # Send session.update with voice_id and instructions
        # Return the session ID for later reference
        pass

    async def stream_audio_to_xai(
        self, audio_chunk: bytes
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio to x.ai, yield response chunks."""
        # Send audio via WebSocket
        # Yield response audio chunks as they arrive
        pass

    async def stop_session(self, session_id: str) -> None:
        """Close the voice session."""
        pass
```

#### 3. Create `src/voxbot/audio_service_client.py`

```python
"""IPC client to audio service."""

import asyncio
import json
import struct
from typing import AsyncGenerator

import structlog

_LOGGER = structlog.get_logger(__name__)


class AudioServiceClient:
    """Communicate with Rust audio service via TCP/Unix socket."""

    def __init__(self, host: str = "localhost", port: int = 9999):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self) -> None:
        """Connect to audio service."""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port
            )
            _LOGGER.info("audio_service_connected", host=self.host, port=self.port)
        except Exception as err:
            _LOGGER.error("audio_service_connection_failed", error=str(err))
            raise

    async def start_chat(
        self, guild_id: int, channel_id: int, user_id: int, voice_id: str
    ) -> str:
        """Start conversation. Returns session_id."""
        msg = {
            "type": "start_chat",
            "guild_id": guild_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "voice_id": voice_id,
        }
        await self._send(msg)

        # Expect "chat_started" response
        response = await self._recv()
        if response.get("type") == "chat_started":
            return response["session_id"]
        raise RuntimeError(f"Unexpected response: {response}")

    async def stop_chat(self, guild_id: int) -> None:
        """Stop conversation."""
        msg = {"type": "stop_chat", "guild_id": guild_id}
        await self._send(msg)

    async def listen_for_audio(self, guild_id: int) -> AsyncGenerator[bytes, None]:
        """Listen for audio chunks from service."""
        while True:
            msg = await self._recv()
            if msg.get("type") == "audio_chunk" and msg["guild_id"] == guild_id:
                audio_data = msg["data"]
                # Decode base64 if needed
                yield audio_data
            elif msg.get("type") == "chat_ended" and msg["guild_id"] == guild_id:
                break

    async def _send(self, msg: dict) -> None:
        """Send JSON message."""
        line = json.dumps(msg) + "\n"
        self.writer.write(line.encode())
        await self.writer.drain()

    async def _recv(self) -> dict:
        """Receive JSON message."""
        line = await self.reader.readline()
        return json.loads(line.decode())

    async def close(self) -> None:
        """Close connection."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
```

#### 4. Update voice cog: add chat commands

```python
# In src/voxbot/plugins/voice/cog.py

@commands.slash_command(
    name="chat",
    group="voice",
    description="Start or stop voice conversation with the bot",
)
async def voice_chat(
    self,
    inter,
    action: str = commands.Param(
        choices=["on", "off"], description="Start or stop conversation"
    ),
):
    """Voice chat: on/off."""
    await inter.response.defer()

    if inter.guild_id is None:
        await inter.edit_original_response(content="❌ Must be used in a server.")
        return

    voice_state = inter.author.voice
    if voice_state is None or voice_state.channel is None:
        await inter.edit_original_response(
            content="❌ You must be in a voice channel."
        )
        return

    if action == "on":
        try:
            session_id = await self.bot.audio_service.start_chat(
                guild_id=inter.guild_id,
                channel_id=voice_state.channel.id,
                user_id=inter.author.id,
                voice_id="custom-voice-default",  # or from user config
            )
            # Ensure bot is in voice channel
            vc = None
            for v in self.bot.voice_clients:
                if v.guild.id == inter.guild_id:
                    vc = v
                    break
            if not vc:
                vc = await voice_state.channel.connect()

            # Start listening for audio chunks
            asyncio.create_task(
                self._stream_audio_responses(inter.guild_id, vc)
            )

            await inter.edit_original_response(
                content="🎤 Voice chat started. Speak now!"
            )
        except Exception as err:
            _LOGGER.error("chat_start_failed", error=str(err))
            await inter.edit_original_response(
                content="⚠️ Failed to start voice chat."
            )

    elif action == "off":
        try:
            await self.bot.audio_service.stop_chat(inter.guild_id)
            await inter.edit_original_response(content="🛑 Voice chat stopped.")
        except Exception as err:
            _LOGGER.error("chat_stop_failed", error=str(err))
            await inter.edit_original_response(content="⚠️ Failed to stop chat.")

async def _stream_audio_responses(self, guild_id: int, vc) -> None:
    """Stream audio responses from service to Discord."""
    try:
        async for audio_chunk in self.bot.audio_service.listen_for_audio(
            guild_id
        ):
            # Decode audio and play
            # This is a simplified placeholder
            source = discord.PCMVolumeTransformer(audio_chunk)
            vc.play(source)
    except asyncio.CancelledError:
        pass
    except Exception as err:
        _LOGGER.error("stream_audio_failed", error=str(err))
```

#### 5. Update bot initialization

```python
# In src/voxbot/__main__.py or bot setup

async def setup(bot):
    """Initialize bot and services."""
    # ... existing code ...

    # Connect to audio service
    bot.audio_service = AudioServiceClient()
    await bot.audio_service.connect()

    # Load voice cog
    await bot.add_cog(VoiceCog(bot))
```

---

### Rust Microservice Installation

#### 1. Create new Rust project

```bash
cargo new voxbot-audio-service --name voxbot_audio
cd voxbot-audio-service
```

#### 2. Add dependencies

```toml
# Cargo.toml
[package]
name = "voxbot-audio-service"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
serenity = { version = "0.12", features = ["voice", "cache", "client", "gateway", "model", "rustls_backend", "twilight-http"] }
songbird = { version = "0.4", features = ["builtin-queue", "driver", "gateway", "model", "rustls_backend"] }
tokio-tungstenite = "0.23"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
structlog = "0.1"  # or tracing/log
dotenv = "0.15"
base64 = "0.22"
anyhow = "1"
thiserror = "1"

[profile.release]
opt-level = 3
lto = true
```

#### 3. Scaffold main.rs

```rust
// src/main.rs
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use serenity::Client;
use songbird::Songbird;

mod discord_voice;
mod xai_client;
mod audio_processor;
mod ipc_protocol;
mod session;

use ipc_protocol::{Message, MessageType};
use session::SessionManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    // Load .env
    dotenv::dotenv().ok();

    // Start Discord voice client
    let discord_token = std::env::var("DISCORD_TOKEN")?;
    let discord_client = create_discord_client(&discord_token).await?;

    // Start IPC server
    let listener = TcpListener::bind("127.0.0.1:9999").await?;
    println!("Audio service listening on 127.0.0.1:9999");

    // Session manager (shared state)
    let session_manager = Arc::new(SessionManager::new());

    loop {
        let (socket, _) = listener.accept().await?;
        let session_mgr = Arc::clone(&session_manager);

        tokio::spawn(async move {
            if let Err(e) = handle_client(socket, session_mgr).await {
                eprintln!("Client error: {}", e);
            }
        });
    }
}

async fn handle_client(
    socket: TcpStream,
    sessions: Arc<SessionManager>,
) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let (reader, mut writer) = socket.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    while reader.read_line(&mut line).await? > 0 {
        let msg: Message = serde_json::from_str(&line)?;

        match msg.msg_type {
            MessageType::StartChat => {
                // Validate, create session, connect to Discord voice
                let guild_id = msg.guild_id.ok_or(anyhow::anyhow!("Missing guild_id"))?;
                let channel_id = msg.channel_id.ok_or(anyhow::anyhow!("Missing channel_id"))?;

                let session_id = sessions.create_session(guild_id, channel_id).await?;

                let response = serde_json::json!({
                    "type": "chat_started",
                    "guild_id": guild_id,
                    "session_id": session_id,
                });
                writer.write_all(serde_json::to_string(&response)?.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
            MessageType::StopChat => {
                let guild_id = msg.guild_id.ok_or(anyhow::anyhow!("Missing guild_id"))?;
                sessions.stop_session(guild_id).await?;

                let response = serde_json::json!({
                    "type": "chat_ended",
                    "guild_id": guild_id,
                    "reason": "user_stopped",
                });
                writer.write_all(serde_json::to_string(&response)?.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
            _ => {}
        }

        line.clear();
    }

    Ok(())
}

async fn create_discord_client(token: &str) -> anyhow::Result<Client> {
    use serenity::prelude::*;
    use serenity::async_trait;

    struct Handler;

    #[async_trait]
    impl EventHandler for Handler {}

    let client = Client::builder(token, GatewayIntents::GUILDS | GatewayIntents::GUILD_VOICE_STATES)
        .event_handler(Handler)
        .voice_manager::<Songbird>()
        .await?;

    Ok(client)
}
```

#### 4. Key module skeletons

**`src/ipc_protocol.rs`** — Message types:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MessageType {
    #[serde(rename = "start_chat")]
    StartChat,
    #[serde(rename = "stop_chat")]
    StopChat,
    #[serde(rename = "chat_started")]
    ChatStarted,
    #[serde(rename = "audio_chunk")]
    AudioChunk,
    #[serde(rename = "chat_ended")]
    ChatEnded,
    #[serde(rename = "error")]
    Error,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    #[serde(flatten)]
    pub msg_type: MessageType,
    pub guild_id: Option<u64>,
    pub channel_id: Option<u64>,
    pub user_id: Option<u64>,
    pub voice_id: Option<String>,
    pub session_id: Option<String>,
    pub data: Option<String>,  // base64 audio
    pub message: Option<String>,  // error message
}
```

**`src/session.rs`** — Session state:
```rust
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Debug)]
pub struct Session {
    pub guild_id: u64,
    pub channel_id: u64,
    pub session_id: String,
    pub xai_session: Option<String>,  // x.ai session ID
}

pub struct SessionManager {
    sessions: RwLock<HashMap<u64, Session>>,  // guild_id -> Session
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    pub async fn create_session(&self, guild_id: u64, channel_id: u64) -> anyhow::Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = Session {
            guild_id,
            channel_id,
            session_id: session_id.clone(),
            xai_session: None,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(guild_id, session);

        Ok(session_id)
    }

    pub async fn get_session(&self, guild_id: u64) -> anyhow::Result<Session> {
        let sessions = self.sessions.read().await;
        sessions
            .get(&guild_id)
            .cloned()
            .ok_or(anyhow::anyhow!("Session not found"))
    }

    pub async fn stop_session(&self, guild_id: u64) -> anyhow::Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(&guild_id);
        Ok(())
    }
}
```

**`src/discord_voice.rs`** — Voice capture:
```rust
use serenity::async_trait;
use songbird::{
    driver::{Driver, DriverStatus},
    events::{Event, EventContext, EventHandler, TrackEnd},
    EventStore,
};

pub struct VoiceReceiver;

#[async_trait]
impl EventHandler for VoiceReceiver {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        // Decode voice packets, accumulate into PCM buffer
        // Stream to x.ai
        None
    }
}
```

**`src/xai_client.rs`** — x.ai integration:
```rust
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use futures::{SinkExt, StreamExt};

pub struct XaiClient {
    api_key: String,
}

impl XaiClient {
    pub async fn connect(&self, voice_id: &str) -> anyhow::Result<XaiSession> {
        let url = "wss://api.x.ai/v1/realtime";
        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Send session.update with voice_id
        let config = serde_json::json!({
            "type": "session.update",
            "session": {
                "voice": voice_id,
                "instructions": "You are a helpful Discord bot. Keep responses concise."
            }
        });

        write.send(WsMessage::Text(serde_json::to_string(&config)?)).await?;

        Ok(XaiSession {
            write,
            read,
        })
    }
}

pub struct XaiSession {
    // WebSocket writer/reader for x.ai
    write: futures::stream::SplitSink<_, WsMessage>,
    read: futures::stream::SplitStream<_>,
}
```

---

## Integration Checklist

### Phase 1: Setup
- [ ] Create Rust project, add dependencies
- [ ] Implement IPC protocol + message types
- [ ] Implement session manager
- [ ] Test TCP server can accept connections

### Phase 2: Discord Voice (Rust)
- [ ] Connect to Discord via serenity
- [ ] Join voice channel via songbird
- [ ] Capture raw audio packets
- [ ] Decode to PCM format
- [ ] Buffer and stream to local queue

### Phase 3: x.ai Integration (Rust)
- [ ] Implement x.ai WebSocket client
- [ ] Send audio frames to x.ai
- [ ] Receive response audio chunks
- [ ] Encode response (Opus/MP3)
- [ ] Send back to Python bot via IPC

### Phase 4: Python Bot Changes
- [ ] Add `audio_service_client.py`
- [ ] Add `services/xai.py` (TTS endpoint)
- [ ] Update voice cog: add `/voice chat` commands
- [ ] Wire up audio service connection in bot init
- [ ] Test end-to-end: `/voice chat on` → speak → hear response

### Phase 5: Polish
- [ ] Error handling (service crash, network timeout, etc.)
- [ ] Session cleanup (orphaned sessions)
- [ ] Logging + monitoring
- [ ] Configuration (env vars, timeouts, audio format)
- [ ] Docker/systemd service management

---

## Testing Strategy

### Unit Tests (Rust)
- IPC protocol serialization/deserialization
- Session manager CRUD
- x.ai message formatting

### Integration Tests (Rust)
- Rust service ↔ mock Discord voice (synthetic audio)
- Rust service ↔ mock x.ai WebSocket
- IPC message round-trip

### End-to-End (Local)
- Start Rust service
- Start bot
- Connect to Discord test server
- `/voice chat on` → manual speak → check response

---

## Gotchas & Mitigation

| Issue | Mitigation |
|-------|-----------|
| **Discord voice connection lag** | Rust service connects independently; Python bot just plays audio |
| **x.ai WebSocket timeout** | Implement ping/pong, reconnect logic, timeout detection |
| **Audio encoding mismatch** | Standardize on PCM for internal buffers; convert at boundaries |
| **IPC message loss** | Use TCP (built-in reliability); add ACK pattern if needed |
| **Memory leak from unclosed sessions** | Implement session TTL (e.g., 15 min auto-close) |
| **Bot crash during conversation** | Audio service keeps running; bot reconnects + resumes |

---

## Configuration

### Python (.env)

```bash
DISCORD_TOKEN=...
X_AI_API_KEY=...
AUDIO_SERVICE_HOST=localhost
AUDIO_SERVICE_PORT=9999
```

### Rust (.env)

```bash
DISCORD_TOKEN=...
X_AI_API_KEY=...
IPC_BIND=127.0.0.1:9999
LOG_LEVEL=info
```

---

## Future Enhancements

1. **Audio effects** (noise gate, echo cancellation) on capture
2. **Multi-user conversations** (track per-user state)
3. **Custom x.ai instructions** (per-user or per-guild)
4. **Conversation history** (context window management)
5. **Metrics** (latency, uptime, audio quality)
6. **TLS** for IPC (if exposing remotely)
7. **Fallback TTS** (if x.ai unavailable, use OpenAI or self-hosted Qwen3)

---

## References

- [x.ai Voice Agent Docs](https://docs.x.ai/developers/model-capabilities/audio/voice-agent)
- [serenity Discord.rs](https://github.com/serenity-rs/serenity)
- [songbird Voice Client](https://github.com/serenity-rs/songbird)
- [tokio-tungstenite WebSocket](https://github.com/snapview/tokio-tungstenite)
