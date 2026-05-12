"""Voice plugin main Cog."""

import asyncio
import base64
import pathlib
import re
import time
from typing import List

import discord
from discord import app_commands
from discord.ext import commands, songbird
import structlog

from voxbot import dota_wiki
from voxbot.errors import MistralError, TTSError
from voxbot.model import VoxModel
from voxbot.services.mistral import MistralService
from voxbot.services.tts import TTSProcessor

from . import ai
from . import state

_LOGGER = structlog.get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_MESSAGE = "woof! woofwoof! ..."
VOICE_NAME_RE = re.compile(r"^[\w\-]+$")


class SpikeReceiver:
    """Temporary receiver to track byte counts for the spike."""

    def __init__(self):
        self.bytes_per_user = {}  # user_id -> total_bytes
        self.ssrc_to_user = {}  # ssrc -> user_id
        self.active_ssrcs = set()

    def speaking_update(self, ssrc: int, user_id: int, speaking: bool):
        self.ssrc_to_user[ssrc] = user_id
        if speaking:
            self.active_ssrcs.add(ssrc)
        else:
            self.active_ssrcs.discard(ssrc)

    def voice_tick(self, tick):
        for ssrc, voice_data in tick.speaking.items():
            if voice_data.decoded_voice:
                user_id = self.ssrc_to_user.get(ssrc, f"SSRC:{ssrc}")
                count = len(voice_data.decoded_voice)
                self.bytes_per_user[user_id] = self.bytes_per_user.get(user_id, 0) + count


class VoiceCog(commands.GroupCog, name="voice"):
    """Main voice plugin cog."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.mistral_service = MistralService()
        self.vox_model = VoxModel()

        # Expose to bot for other components (tasks, etc.)
        bot.mistral_service = self.mistral_service
        bot.vox_model = self.vox_model

    async def voice_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete voice names."""
        voices = state.VoiceState.get_voice_list(self.mistral_service.voices.custom_voices)
        filtered = state.VoiceState.filter_voice_list(voices, current)
        return [app_commands.Choice(name=v, value=v) for v in filtered]

    @app_commands.command(name="train", description="Train a custom TTS voice")
    @app_commands.describe(
        audio="Audio sample to train from",
        hero="Dota hero name; scrapes 20-30s of voice lines",
    )
    async def voice_train(
        self,
        interaction: discord.Interaction,
        audio: discord.Attachment = None,
        hero: str = None,
    ):
        await interaction.response.defer(thinking=True)

        if (audio is None) == (hero is None):
            await interaction.edit_original_response(
                content="❌ Provide exactly one of `audio` or `hero`."
            )
            return

        if hero is not None:
            await interaction.edit_original_response(
                content=f"🔎 Scraping voice lines for `{hero}` from the Dota wiki..."
            )
            try:
                canonical, audio_bytes, sample_filename = (
                    await dota_wiki.sample_voice_lines(hero)
                )
            except dota_wiki.WikiError as err:
                await interaction.edit_original_response(content=f"❌ {err}")
                return
            except Exception as err:
                _LOGGER.error("wiki_scrape_failed", error=str(err), hero=hero)
                await interaction.edit_original_response(
                    content="⚠️ Failed to scrape Dota wiki. Check logs."
                )
                return

            voice_name = re.sub(r"[^A-Za-z0-9]+", "_", canonical).strip("_").title()
        else:
            ext = pathlib.Path(audio.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                await interaction.edit_original_response(
                    content=f"❌ Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
                )
                return

            voice_name = pathlib.Path(audio.filename).stem.title()
            if not VOICE_NAME_RE.match(voice_name):
                await interaction.edit_original_response(
                    content="❌ Filename must be alphanumeric (underscores/hyphens OK)."
                )
                return

            try:
                audio_bytes = await audio.read()
            except Exception as err:
                _LOGGER.error("voice_download_failed", error=str(err))
                await interaction.edit_original_response(
                    content="⚠️ Failed to download audio file."
                )
                return
            sample_filename = audio.filename

        await interaction.edit_original_response(
            content=f"🎤 Training voice `{voice_name}`... this may take a moment."
        )

        audio_b64_str = base64.b64encode(audio_bytes).decode()

        try:
            await self.mistral_service.train_voice(
                voice_name, audio_b64_str, sample_filename
            )
        except MistralError:
            await interaction.edit_original_response(
                content="⚠️ Voice training failed. Check logs."
            )
            return

        if hero is not None:
            self.vox_model.hero_origins[voice_name] = canonical
        else:
            self.vox_model.hero_origins.pop(voice_name, None)

        await interaction.edit_original_response(
            content=f"✅ Voice `{voice_name}` trained! Use it with: `/voice speak voice:{voice_name}`"
        )

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
        message: str = None,
        prompt: str = None,
        voice: str = None,
    ):
        await interaction.response.defer(thinking=True)

        if message is not None and prompt is not None:
            await interaction.edit_original_response(
                content="❌ Provide `message` OR `prompt`, not both."
            )
            return

        if interaction.guild_id is None:
            await interaction.edit_original_response(content="❌ Must be used in a server.")
            return

        voice_state = interaction.user.voice
        if voice_state is None or voice_state.channel is None:
            await interaction.edit_original_response(
                content="❌ You must be in a voice channel."
            )
            return

        voice_id = state.VoiceState.resolve_voice(
            voice, self.mistral_service.voices.custom_voices
        )
        if voice_id is None:
            available = (
                ", ".join(self.mistral_service.voices.custom_voices.keys())
                or "none"
            )
            await interaction.edit_original_response(
                content=f"❌ Unknown voice `{voice}`. Available: {available}"
            )
            return

        if prompt is not None:
            hero_context = self.vox_model.hero_origins.get(voice or "")
            try:
                text = await ai.generate_line(prompt, hero_context)
            except Exception as err:
                _LOGGER.error("ai_generate_failed", error=str(err))
                await interaction.edit_original_response(
                    content="⚠️ Failed to generate line. Check logs."
                )
                return
            if not text:
                await interaction.edit_original_response(
                    content="⚠️ AI returned an empty line."
                )
                return
            await interaction.edit_original_response(content=f"🤖 {text}")
        else:
            text = message or DEFAULT_MESSAGE

        _LOGGER.info(
            "tts_request",
            author=interaction.user.display_name,
            guild_id=interaction.guild_id,
            length=len(text),
            voice=voice_id,
            generated=prompt is not None,
        )

        try:
            tts_response = await self.mistral_service.text_to_speech(text, voice_id)
        except MistralError:
            await interaction.edit_original_response(
                content="⚠️ Failed to generate speech. Check logs."
            )
            return

        try:
            source_data, tmp_path = TTSProcessor.prepare_audio_source(tts_response)
        except TTSError:
            await interaction.edit_original_response(content="⚠️ Failed to prepare audio.")
            return

        try:
            # Connect using standard discord.py voice client first (more reliable)
            vc = discord.utils.get(self.bot.voice_clients, guild=interaction.guild)
            if vc:
                if vc.channel.id != voice_state.channel.id:
                    await vc.move_to(voice_state.channel)
            else:
                vc = await voice_state.channel.connect()

            # Use FFmpegPCMAudio with standard voice client
            source = discord.FFmpegPCMAudio(tmp_path)

            def _after_play(err, _path=tmp_path):
                if err is not None:
                    _LOGGER.error("voice_playback_failed", error=str(err))
                TTSProcessor.cleanup_temp_file(_path)

            vc.play(source, after=_after_play)

        except Exception as err:
            _LOGGER.error("voice_connect_failed", error=str(err))
            TTSProcessor.cleanup_temp_file(tmp_path)
            await interaction.edit_original_response(
                content="⚠️ Failed to connect or play audio."
            )
            return

        self.vox_model.last_active[interaction.guild_id] = time.monotonic()
        playing_label = f"🔊 {text}" if prompt is not None else "🔊 Playing..."
        await interaction.edit_original_response(content=playing_label)

    @app_commands.command(name="delete", description="Delete a custom voice")
    @app_commands.describe(voice="Voice to delete")
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def voice_delete(
        self,
        interaction: discord.Interaction,
        voice: str,
    ):
        await interaction.response.defer(thinking=True)

        if voice not in self.mistral_service.voices.custom_voices:
            await interaction.edit_original_response(
                content=f"❌ Voice `{voice}` not found."
            )
            return

        voice_id = self.mistral_service.voices.custom_voices[voice]

        try:
            await self.mistral_service.delete_voice(voice_id)
        except MistralError:
            await interaction.edit_original_response(
                content="⚠️ Failed to delete voice. Check logs."
            )
            return

        del self.mistral_service.voices.custom_voices[voice]
        self.mistral_service.voices.deleted_voices.add(voice)
        self.vox_model.hero_origins.pop(voice, None)
        self.mistral_service.save_voices()
        await interaction.edit_original_response(content=f"🗑️ Voice `{voice}` deleted.")

    @app_commands.command(name="listen", description="Listen to the voice channel for 10 seconds")
    async def voice_listen(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)

        voice_state = interaction.user.voice
        if not voice_state or not voice_state.channel:
            await interaction.edit_original_response(content="❌ You must be in a voice channel.")
            return

        try:
            # We must enable decoding to see PCM bytes
            config = songbird.ConfigBuilder().decode_mode(songbird.PyDecodeMode.Decode).build()

            vc = discord.utils.get(self.bot.voice_clients, guild=interaction.guild)
            if vc:
                if vc.channel.id != voice_state.channel.id:
                    await vc.move_to(voice_state.channel)
                if not isinstance(vc, songbird.SongbirdClient):
                    await vc.disconnect()
                    vc = await voice_state.channel.connect(cls=songbird.SongbirdClient, config=config)
            else:
                vc = await voice_state.channel.connect(cls=songbird.SongbirdClient, config=config)
        except Exception as err:
            _LOGGER.error("voice_connect_failed", error=str(err))
            await interaction.edit_original_response(content="⚠️ Failed to connect.")
            return

        receiver = SpikeReceiver()
        await vc.register_receiver(receiver)

        await interaction.edit_original_response(content="👂 Listening... (Initializing Rust/Songbird)")

        try:
            for i in range(1, 11):
                await asyncio.sleep(1)

                lines = [f"👂 **Listening Spike** ({i}s / 10s)", ""]
                if not receiver.bytes_per_user:
                    lines.append("_No audio data received yet... speak now!_")
                else:
                    for uid, total_bytes in receiver.bytes_per_user.items():
                        kb = total_bytes / 1024
                        speaker_label = f"<@{uid}>" if isinstance(uid, int) else uid
                        is_speaking = "🎙️" if any(s for s, u in receiver.ssrc_to_user.items() if u == uid and s in receiver.active_ssrcs) else "🔇"
                        lines.append(f"{is_speaking} {speaker_label}: **{kb:.1f} KB** heard")

                await interaction.edit_original_response(content="\n".join(lines))
        finally:
            # Always unregister to clean up the receiver thread
            await vc.unregister_receiver(receiver)

        await interaction.edit_original_response(content="✅ Finished listening spike. Data received for {} users.".format(len(receiver.bytes_per_user)))

    async def cog_load(self) -> None:
        """Called when cog is loaded."""
        _LOGGER.info("voice_cog_loaded")
