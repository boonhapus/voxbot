import base64 as b64
import pathlib
import random
import re
import time

import crescent
from crescent.ext import tasks
import hikari
import hikariwave
import structlog

from voxbot import model

_LOGGER = structlog.get_logger(__name__)

plugin = crescent.Plugin[hikari.GatewayBot, model.VoxModel]()

# Command group: /voice <subcommand>
voice_group = crescent.Group("voice", "TTS voice commands")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_VOICES = ["sad", "frustrated", "excited", "confident", "cheerful", "angry"]
DEFAULT_MESSAGE = "woof! woofwoof! ..."
VOICE_PREFIX = "en_paul_"

VOICE_NAME_RE = re.compile(r"^[\w\-]+$")


def _clear_stale_player_task(connection: hikariwave.Connection) -> None:
    # hikariwave 0.7.0a1 bug: AudioPlayer keeps a reference to the finished
    # _player_task, so __ensure_loop() short-circuits and play() silently
    # no-ops on the second call. Drop the dead task to force a fresh loop.
    # TODO: Remove once hikariwave is updated past 0.7.0a1.
    player_task = connection.player._player_task
    if player_task is not None and player_task.done():
        connection.player._player_task = None


async def _ensure_voice_connection(
    vc: hikariwave.VoiceClient, guild_id: int, channel_id: int
) -> hikariwave.Connection:
    connection = vc.get_connection(guild_id=guild_id)
    if connection is None:
        return await vc.connect(guild_id=guild_id, channel_id=channel_id)
    if connection.channel_id != hikari.Snowflake(channel_id):
        return await vc.move(channel_id=channel_id, guild_id=guild_id)
    return connection


def _resolve_voice(voice: str | None, custom_voices: dict[str, str]) -> str | None:
    if voice is None:
        return f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"
    if voice == "Paul":
        return f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"
    voice_id = custom_voices.get(voice)
    return voice_id


@plugin.include
@voice_group.child
@crescent.command(name="train", description="Train a custom TTS voice from an audio file")
class VoiceTrain:
    audio = crescent.option(hikari.Attachment, "Audio file to train voice from")

    async def callback(self, ctx: crescent.Context) -> None:
        await ctx.defer()
        attachment = self.audio
        model_instance = plugin.model

        ext = pathlib.Path(attachment.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            await ctx.respond(f"❌ Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}")
            return

        voice_name = pathlib.Path(attachment.filename).stem
        if not VOICE_NAME_RE.match(voice_name):
            await ctx.respond("❌ Filename must be alphanumeric (underscores/hyphens OK).")
            return

        await ctx.respond(f"🎤 Training voice `{voice_name}`... this may take a moment.")

        try:
            audio_bytes = await attachment.read()
            audio_b64_str = b64.b64encode(audio_bytes).decode()
        except Exception as err:
            _LOGGER.error("voice_download_failed", error=str(err))
            await ctx.edit("⚠️ Failed to download audio file.")
            return

        try:
            voice = await model_instance.mistral.audio.voices.create_async(
                name=voice_name,
                sample_audio=audio_b64_str,
                sample_filename=attachment.filename,
            )
        except Exception as err:
            _LOGGER.error("voice_train_failed", error=str(err), name=voice_name)
            await ctx.edit("⚠️ Voice training failed. Check logs.")
            return

        model_instance.custom_voices[voice_name] = voice.id
        model_instance.save_voices()
        await ctx.edit(
            f"✅ Voice `{voice_name}` trained! Use it with: `/voice speak --voice {voice_name}`"
        )


async def _voice_autocomplete(
    ctx: crescent.AutocompleteContext, option: hikari.AutocompleteInteractionOption
) -> list[tuple[str, str]]:
    """Autocomplete for voice names: custom voices + 'Paul' preset."""
    model_instance = plugin.model
    voices = list(model_instance.custom_voices.keys())
    # Add 'Paul' as a single option for the preset voices
    if "Paul" not in voices:
        voices.append("Paul")
    
    if not voices:
        return [("No voices available", "none")]
    
    # Filter by user input
    user_input = option.value or ""
    if isinstance(user_input, str) and user_input:
        voices = [v for v in voices if user_input.lower() in v.lower()]
    
    return [(v, v) for v in voices[:25]]  # Discord max 25 choices


@plugin.include
@voice_group.child
@crescent.command(name="speak", description="Speak a message in your voice channel")
class VoiceSpeak:
    message = crescent.option(str, "Message to speak", default=DEFAULT_MESSAGE)
    voice = crescent.option(str, "Voice to use (leave empty for random)", default=None, autocomplete=_voice_autocomplete)

    async def callback(self, ctx: crescent.Context) -> None:
        await ctx.defer()
        user = ctx.user
        guild_id = ctx.guild_id
        if guild_id is None:
            await ctx.respond("❌ Must be used in a server.")
            return

        assert isinstance(ctx.app, hikari.CacheAware)
        voice_state = ctx.app.cache.get_voice_state(guild_id, user.id)
        if voice_state is None or voice_state.channel_id is None:
            states = ctx.app.cache.get_voice_states_view_for_guild(guild_id)
            _LOGGER.warning(
                "no_voice_state",
                user_id=user.id,
                guild_id=guild_id,
                cached_states=len(states),
                cached_user_ids=[int(uid) for uid in states.keys()],
            )
            await ctx.respond(
                "❌ You must be in a voice channel. "
                "(If you are, the bot may need the `GUILD_VOICE_STATES` intent.)"
            )
            return

        model_instance = plugin.model

        voice_id = _resolve_voice(self.voice, model_instance.custom_voices)
        if voice_id is None:
            available = ", ".join(model_instance.custom_voices.keys()) or "none"
            await ctx.respond(f"❌ Unknown voice `{self.voice}`. Available: {available}")
            return

        _LOGGER.info(
            "tts_request",
            author=user.display_name,
            guild_id=guild_id,
            length=len(self.message),
            voice=voice_id,
        )

        try:
            tts_response = await model_instance.mistral.audio.speech.complete_async(
                model=model_instance.config.mistral_model,
                input=self.message,
                voice_id=voice_id,
                response_format="mp3",
            )
        except Exception as err:
            _LOGGER.error("tts_failed", error=str(err))
            await ctx.respond("⚠️ Failed to generate speech. Check logs for details.")
            return

        connection = await _ensure_voice_connection(
            model_instance.voice_client, guild_id, voice_state.channel_id
        )
        assert connection is not None

        if connection.player.is_playing:
            await connection.player.stop()

        _clear_stale_player_task(connection)

        audio_data = b64.b64decode(tts_response.audio_data)
        source = hikariwave.BufferAudioSource(audio_data)
        await connection.player.play(source)

        model_instance.last_active[guild_id] = time.monotonic()
        await ctx.respond("🔊 Playing...")


@plugin.include
@voice_group.child
@crescent.command(name="delete", description="Delete a custom voice")
class VoiceDelete:
    voice = crescent.option(str, "Voice to delete", autocomplete=_voice_autocomplete)

    async def callback(self, ctx: crescent.Context) -> None:
        await ctx.defer()
        model_instance = plugin.model
        voice_name = self.voice

        if voice_name not in model_instance.custom_voices:
            await ctx.respond(f"❌ Voice `{voice_name}` not found.")
            return

        voice_id = model_instance.custom_voices[voice_name]

        try:
            await model_instance.mistral.audio.voices.delete_async(voice_id=voice_id)
        except Exception as err:
            _LOGGER.error("voice_delete_failed", error=str(err), name=voice_name)
            await ctx.respond("⚠️ Failed to delete voice from Mistral. Check logs.")
            return

        del model_instance.custom_voices[voice_name]
        model_instance.deleted_voices.add(voice_name)
        model_instance.save_voices()
        await ctx.respond(f"🗑️ Voice `{voice_name}` deleted.")


_SYNC_INTERVAL = 300  # seconds between voice syncs
_AUTO_LEAVE_INTERVAL = 60  # check every minute
_AUTO_LEAVE_THRESHOLD = 300  # 5 minutes in seconds


@plugin.include
@tasks.loop(seconds=_SYNC_INTERVAL)
async def _voice_sync() -> None:
    """Sync voices from Mistral periodically."""
    model_instance = plugin.model
    await model_instance.sync_voices()


@plugin.include
@tasks.loop(seconds=_AUTO_LEAVE_INTERVAL)
async def _auto_leave() -> None:
    """Leave voice channels with no activity in the last 5 minutes."""
    model_instance = plugin.model
    vc = model_instance.voice_client
    if vc is None:
        return

    now = time.monotonic()
    to_remove = []

    for guild_id, last_time in list(model_instance.last_active.items()):
        if now - last_time > _AUTO_LEAVE_THRESHOLD:
            connection = vc.get_connection(guild_id=guild_id)
            if connection is not None:
                await connection.disconnect()
                _LOGGER.info("auto_left_voice", guild_id=guild_id, reason="inactive")
            to_remove.append(guild_id)

    for guild_id in to_remove:
        del model_instance.last_active[guild_id]


@plugin.load_hook
def on_load() -> None:
    _LOGGER.info("voice_plugin_loaded")
