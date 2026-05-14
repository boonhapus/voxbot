"""Voice plugin main Cog."""

import asyncio
import base64
import pathlib
import re
import time
from typing import cast

from discord import app_commands
from discord.ext import commands, songbird
import discord
import structlog

from voxbot import dota_wiki
from voxbot.errors import MistralError, TTSError
from voxbot.services.tts import TTSProcessor

from . import ai, state
from .errors import VoiceCommandError

_LOGGER = structlog.get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_MESSAGE = "woof! woofwoof! ..."
VOICE_NAME_RE = re.compile(r"^[\w\-]+$")
LISTEN_DURATION_SECONDS = 10


class SpikeReceiver:
    """Songbird receiver that tracks decoded audio byte counts per user."""

    def __init__(self) -> None:
        self.bytes_per_user: dict[int | str, int] = {}
        self.ssrc_to_user: dict[int, int | str] = {}
        self.active_ssrcs: set[int] = set()

    def speaking_update(self, ssrc: int, user_id: int, speaking: bool) -> None:
        self.ssrc_to_user[ssrc] = user_id

        if speaking:
            self.active_ssrcs.add(ssrc)
        else:
            self.active_ssrcs.discard(ssrc)

    def voice_tick(self, tick: object) -> None:
        for ssrc, voice_data in tick.speaking.items():  # pyrefly: ignore[missing-attribute]
            if not voice_data.decoded_voice:
                continue

            uid = self.ssrc_to_user.get(ssrc, f"SSRC:{ssrc}")
            self.bytes_per_user[uid] = self.bytes_per_user.get(uid, 0) + len(voice_data.decoded_voice)


# ── TRAINING HELPERS ──────────────────────────────────────────────────────────────────


async def _audio_from_hero(hero: str) -> tuple[str, bytes, str, str]:
    """Scrape a Dota hero voice sample for training. Returns (voice_name, bytes, filename, canonical)."""
    try:
        canonical, audio_bytes, sample_filename = await dota_wiki.sample_voice_lines(hero)
    except dota_wiki.WikiError as exc:
        raise VoiceCommandError(f"❌ {exc}") from exc
    except (OSError, ValueError) as exc:
        raise VoiceCommandError(
            "⚠️ Failed to scrape Dota wiki. Check logs.",
            log_event="wiki_scrape_failed", error=str(exc), hero=hero,
        ) from exc

    voice_name = re.sub(r"[^A-Za-z0-9]+", "_", canonical).strip("_").title()
    return voice_name, audio_bytes, sample_filename, canonical


async def _audio_from_attachment(audio: discord.Attachment) -> tuple[str, bytes, str]:
    """Validate and download an audio attachment for training."""
    ext = pathlib.Path(audio.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise VoiceCommandError(f"❌ Unsupported format. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    voice_name = pathlib.Path(audio.filename).stem.title()
    if not VOICE_NAME_RE.match(voice_name):
        raise VoiceCommandError("❌ Filename must be alphanumeric (underscores/hyphens OK).")

    try:
        audio_bytes = await audio.read()
    except discord.HTTPException as exc:
        raise VoiceCommandError(
            "⚠️ Failed to download audio file.",
            log_event="voice_download_failed", error=str(exc),
        ) from exc

    return voice_name, audio_bytes, audio.filename


# ── SPEAK HELPERS ─────────────────────────────────────────────────────────────────────


async def _generate_speech_text(message: str | None, prompt: str | None, voice: str | None) -> str:
    """Resolve the text to be spoken, generating an AI line when a prompt is supplied."""
    if prompt is None:
        return message or DEFAULT_MESSAGE

    hero_context = state.vox_model.hero_origins.get(voice or "")

    try:
        text = await ai.generate_line(prompt, hero_context)
    except Exception as exc:
        raise VoiceCommandError(
            "⚠️ Failed to generate line. Check logs.",
            log_event="ai_generate_failed", error=str(exc),
        ) from exc

    if not text:
        raise VoiceCommandError("⚠️ AI returned an empty line.")

    return text


async def _play_in_voice(
    bot: commands.Bot,
    channel: discord.VoiceChannel,
    tmp_path: str,
) -> None:
    """Connect to the voice channel and start asynchronous playback of the prepared file."""

    def _after_play(err: Exception | None) -> None:
        if err is not None:
            _LOGGER.error("voice_playback_failed", error=str(err))

        TTSProcessor.cleanup_temp_file(tmp_path)

    try:
        vc = discord.utils.get(bot.voice_clients, guild=channel.guild)

        if vc is None:
            vc = await channel.connect()
        elif vc.channel.id != channel.id:  # pyrefly: ignore[missing-attribute]
            await vc.move_to(channel)  # pyrefly: ignore[missing-attribute]

        vc.play(discord.FFmpegPCMAudio(tmp_path), after=_after_play)  # pyrefly: ignore[missing-attribute]
    except (discord.DiscordException, asyncio.TimeoutError, OSError) as exc:
        raise VoiceCommandError(
            "⚠️ Failed to connect or play audio.",
            log_event="voice_connect_failed", error=str(exc),
        ) from exc


# ── LISTEN HELPERS ────────────────────────────────────────────────────────────────────


async def _connect_songbird(
    bot: commands.Bot, channel: discord.VoiceChannel,
) -> songbird.SongbirdClient:
    """Connect or upgrade the voice client to a SongbirdClient with decoding enabled."""
    config = songbird.ConfigBuilder().decode_mode(songbird.PyDecodeMode.Decode).build()  # pyrefly: ignore[missing-attribute]

    try:
        vc = discord.utils.get(bot.voice_clients, guild=channel.guild)

        if vc is None:
            return cast(songbird.SongbirdClient, await channel.connect(cls=songbird.SongbirdClient, config=config))  # pyrefly: ignore[bad-specialization, unexpected-keyword]

        if not isinstance(vc, songbird.SongbirdClient):
            await vc.disconnect(force=True)
            return cast(songbird.SongbirdClient, await channel.connect(cls=songbird.SongbirdClient, config=config))  # pyrefly: ignore[bad-specialization, unexpected-keyword]

        if vc.channel.id != channel.id:
            await vc.move_to(channel)

        return vc
    except (discord.DiscordException, asyncio.TimeoutError, OSError) as exc:
        raise VoiceCommandError(
            "⚠️ Failed to connect.",
            log_event="voice_connect_failed", error=str(exc),
        ) from exc


def _render_listen(elapsed: int, receiver: SpikeReceiver) -> str:
    """Render the live status text for the listen spike."""
    lines = [f"👂 **Listening Spike** ({elapsed}s / {LISTEN_DURATION_SECONDS}s)", ""]

    if not receiver.bytes_per_user:
        lines.append("_No audio data received yet... speak now!_")
        return "\n".join(lines)

    for uid, total_bytes in receiver.bytes_per_user.items():
        speaker = f"<@{uid}>" if isinstance(uid, int) else uid
        is_speaking = any(
            ssrc in receiver.active_ssrcs
            for ssrc, mapped_uid in receiver.ssrc_to_user.items()
            if mapped_uid == uid
        )
        marker = "🎙️" if is_speaking else "🔇"
        lines.append(f"{marker} {speaker}: **{total_bytes / 1024:.1f} KB** heard")

    return "\n".join(lines)


# ── COG ───────────────────────────────────────────────────────────────────────────────


class VoiceCog(commands.GroupCog, name="voice"):
    """Voice training, TTS playback, and voice-channel listening."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    async def cog_load(self) -> None:
        """Called when cog is loaded."""
        _LOGGER.info("voice_cog_loaded")

    # ── AUTOCOMPLETE ──────────────────────────────────────────────────────────────────

    async def voice_autocomplete(
        self, interaction: discord.Interaction, current: str,
    ) -> list[app_commands.Choice[str]]:
        """Autocomplete voice names from the trained voice registry."""
        voices = state.list_voices(state.mistral_service.voices.custom_voices)
        return [app_commands.Choice(name=v, value=v) for v in state.filter_voices(voices, current)]

    # ── /voice train ──────────────────────────────────────────────────────────────────

    @app_commands.command(name="train", description="Train a custom TTS voice")
    @app_commands.describe(
        audio="Audio sample to train from",
        hero="Dota hero name; scrapes 20-30s of voice lines",
    )
    async def voice_train(
        self,
        interaction: discord.Interaction,
        audio: discord.Attachment | None = None,
        hero: str | None = None,
    ) -> None:
        """Train a custom voice from an audio attachment or a Dota hero name."""
        await interaction.response.defer(thinking=True)

        if (audio is None) == (hero is None):
            await interaction.edit_original_response(
                content="❌ Provide exactly one of `audio` or `hero`.",
            )
            return

        canonical: str | None = None

        try:
            if hero is not None:
                await interaction.edit_original_response(
                    content=f"🔎 Scraping voice lines for `{hero}` from the Dota wiki...",
                )
                voice_name, audio_bytes, sample_filename, canonical = await _audio_from_hero(hero)
            else:
                assert audio is not None
                voice_name, audio_bytes, sample_filename = await _audio_from_attachment(audio)

            await interaction.edit_original_response(
                content=f"🎤 Training voice `{voice_name}`... this may take a moment.",
            )

            try:
                await state.mistral_service.train_voice(
                    voice_name, base64.b64encode(audio_bytes).decode(), sample_filename,
                )
            except MistralError as exc:
                raise VoiceCommandError("⚠️ Voice training failed. Check logs.") from exc

        except VoiceCommandError as exc:
            exc.log()
            await interaction.edit_original_response(content=exc.user_message)
            return

        if canonical is not None:
            state.vox_model.hero_origins[voice_name] = canonical
        else:
            state.vox_model.hero_origins.pop(voice_name, None)

        await interaction.edit_original_response(
            content=f"✅ Voice `{voice_name}` trained! Use it with: `/voice speak voice:{voice_name}`",
        )

    # ── /voice speak ──────────────────────────────────────────────────────────────────

    @app_commands.command(name="speak", description="Speak a message in your voice channel")
    @app_commands.describe(
        message="Message to speak",
        prompt="Short prompt — AI generates a line",
        voice="Voice to use",
    )
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def voice_speak(
        self,
        interaction: discord.Interaction,
        message: str | None = None,
        prompt: str | None = None,
        voice: str | None = None,
    ) -> None:
        """Speak a message in the user's voice channel using the chosen voice."""
        await interaction.response.defer(thinking=True)

        if message is not None and prompt is not None:
            await interaction.edit_original_response(content="❌ Provide `message` OR `prompt`, not both.")
            return

        if interaction.guild_id is None:
            await interaction.edit_original_response(content="❌ Must be used in a server.")
            return

        user_voice = interaction.user.voice  # pyrefly: ignore[missing-attribute]
        if user_voice is None or user_voice.channel is None:
            await interaction.edit_original_response(content="❌ You must be in a voice channel.")
            return

        voice_id = state.resolve_voice(voice, state.mistral_service.voices.custom_voices)
        if voice_id is None:
            available = ", ".join(state.mistral_service.voices.custom_voices) or "none"
            await interaction.edit_original_response(
                content=f"❌ Unknown voice `{voice}`. Available: {available}",
            )
            return

        tmp_path: str | None = None

        try:
            text = await _generate_speech_text(message, prompt, voice)

            if prompt is not None:
                await interaction.edit_original_response(content=f"🤖 {text}")

            _LOGGER.info(
                "tts_request",
                author=interaction.user.display_name,
                guild_id=interaction.guild_id,
                length=len(text),
                voice=voice_id,
                generated=prompt is not None,
            )

            try:
                tts_response = await state.mistral_service.text_to_speech(text, voice_id)
            except MistralError as exc:
                raise VoiceCommandError("⚠️ Failed to generate speech. Check logs.") from exc

            try:
                _, tmp_path = TTSProcessor.prepare_audio_source(tts_response)
            except TTSError as exc:
                raise VoiceCommandError("⚠️ Failed to prepare audio.") from exc

            await _play_in_voice(self.bot, user_voice.channel, tmp_path)

        except VoiceCommandError as exc:
            exc.log()
            if tmp_path is not None:
                TTSProcessor.cleanup_temp_file(tmp_path)
            await interaction.edit_original_response(content=exc.user_message)
            return

        state.vox_model.last_active[interaction.guild_id] = time.monotonic()

        if prompt is None:
            await interaction.edit_original_response(content="🔊 Playing...")

    # ── /voice delete ─────────────────────────────────────────────────────────────────

    @app_commands.command(name="delete", description="Delete a custom voice")
    @app_commands.describe(voice="Voice to delete")
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def voice_delete(self, interaction: discord.Interaction, voice: str) -> None:
        """Remove a trained voice from the registry and the Mistral API."""
        await interaction.response.defer(thinking=True)

        voices = state.mistral_service.voices

        if voice not in voices.custom_voices:
            await interaction.edit_original_response(content=f"❌ Voice `{voice}` not found.")
            return

        try:
            await state.mistral_service.delete_voice(voices.custom_voices[voice])
        except MistralError:
            await interaction.edit_original_response(content="⚠️ Failed to delete voice. Check logs.")
            return

        del voices.custom_voices[voice]
        voices.deleted_voices.add(voice)
        state.vox_model.hero_origins.pop(voice, None)
        state.mistral_service.save_voices()

        await interaction.edit_original_response(content=f"🗑️ Voice `{voice}` deleted.")

    # ── /voice listen ─────────────────────────────────────────────────────────────────

    @app_commands.command(name="listen", description="Listen to the voice channel for 10 seconds")
    async def voice_listen(self, interaction: discord.Interaction) -> None:
        """Run a 10-second receiver spike, reporting decoded audio bytes per speaker."""
        await interaction.response.defer(thinking=True)

        user_voice = interaction.user.voice  # pyrefly: ignore[missing-attribute]
        if user_voice is None or user_voice.channel is None:
            await interaction.edit_original_response(content="❌ You must be in a voice channel.")
            return

        try:
            vc = await _connect_songbird(self.bot, user_voice.channel)
        except VoiceCommandError as exc:
            exc.log()
            await interaction.edit_original_response(content=exc.user_message)
            return

        receiver = SpikeReceiver()
        await vc.register_receiver(receiver)  # pyrefly: ignore[missing-attribute]

        await interaction.edit_original_response(
            content="👂 Listening... (Initializing Rust/Songbird)",
        )

        try:
            for elapsed in range(1, LISTEN_DURATION_SECONDS + 1):
                await asyncio.sleep(1)
                await interaction.edit_original_response(content=_render_listen(elapsed, receiver))
        finally:
            await vc.unregister_receiver(receiver)  # pyrefly: ignore[missing-attribute]

        await interaction.edit_original_response(
            content=f"✅ Finished listening spike. Data received for {len(receiver.bytes_per_user)} users.",
        )
