import base64 as b64
import pathlib
import random
import re
import time

from disnake import Attachment
from disnake.ext import commands, tasks
import structlog

from voxbot import model

_LOGGER = structlog.get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_VOICES = ["sad", "frustrated", "excited", "confident", "cheerful", "angry"]
DEFAULT_MESSAGE = "woof! woofwoof! ..."
VOICE_PREFIX = "en_paul_"

VOICE_NAME_RE = re.compile(r"^[\w\-]+$")

_SYNC_INTERVAL = 300
_AUTO_LEAVE_INTERVAL = 60
_AUTO_LEAVE_THRESHOLD = 300


class VoiceCog(commands.Cog):
    def __init__(self, bot, vox_model: model.VoxModel):
        self.bot = bot
        self.model = vox_model
        self._voice_sync.start()
        self._auto_leave.start()

    async def _ensure_voice_connection(self, guild_id: int, channel_id: int):
        """Ensure bot is connected to voice channel."""
        for vc in self.bot.voice_clients:
            if vc.guild.id == guild_id:
                if vc.channel.id != channel_id:
                    await vc.move_to(self.bot.get_channel(channel_id))
                return vc

        channel = self.bot.get_channel(channel_id)
        return await channel.connect()

    def _resolve_voice(self, voice: str | None) -> str | None:
        if voice is None:
            return f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"
        if voice == "Paul":
            return f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"
        return self.model.custom_voices.get(voice)

    async def voice_autocomplete(self, inter, user_input: str):
        """Autocomplete for voice names."""
        voices = list(self.model.custom_voices.keys())
        if "Paul" not in voices:
            voices.append("Paul")

        if not voices:
            return ["No voices available"]

        if user_input:
            voices = [v for v in voices if user_input.lower() in v.lower()]

        return voices[:25]

    @commands.slash_command(name="voice_train", description="Train a custom TTS voice from an audio file")
    async def voice_train(self, inter, audio: Attachment):
        await inter.response.defer()

        ext = pathlib.Path(audio.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            await inter.edit_original_response(content=f"❌ Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}")
            return

        voice_name = pathlib.Path(audio.filename).stem
        if not VOICE_NAME_RE.match(voice_name):
            await inter.edit_original_response(content="❌ Filename must be alphanumeric (underscores/hyphens OK).")
            return

        await inter.edit_original_response(content=f"🎤 Training voice `{voice_name}`... this may take a moment.")

        try:
            audio_bytes = await audio.read()
            audio_b64_str = b64.b64encode(audio_bytes).decode()
        except Exception as err:
            _LOGGER.error("voice_download_failed", error=str(err))
            await inter.edit_original_response(content="⚠️ Failed to download audio file.")
            return

        try:
            voice = await self.model.mistral.audio.voices.create_async(
                name=voice_name,
                sample_audio=audio_b64_str,
                sample_filename=audio.filename,
            )
        except Exception as err:
            _LOGGER.error("voice_train_failed", error=str(err), name=voice_name)
            await inter.edit_original_response(content="⚠️ Voice training failed. Check logs.")
            return

        self.model.custom_voices[voice_name] = voice.id
        self.model.save_voices()
        await inter.edit_original_response(
            content=f"✅ Voice `{voice_name}` trained! Use it with: `/voice_speak --voice {voice_name}`"
        )

    @commands.slash_command(name="voice_speak", description="Speak a message in your voice channel")
    async def voice_speak(
        self,
        inter,
        message: str = commands.Param(default=DEFAULT_MESSAGE, description="Message to speak"),
        voice: str = commands.Param(default=None, description="Voice to use", autocomplete=voice_autocomplete),
    ):
        await inter.response.defer()

        user = inter.author
        guild_id = inter.guild_id
        if guild_id is None:
            await inter.edit_original_response(content="❌ Must be used in a server.")
            return

        guild = inter.guild
        voice_state = guild.get_member(user.id).voice
        if voice_state is None or voice_state.channel is None:
            await inter.edit_original_response(content="❌ You must be in a voice channel.")
            return

        voice_id = self._resolve_voice(voice)
        if voice_id is None:
            available = ", ".join(self.model.custom_voices.keys()) or "none"
            await inter.edit_original_response(content=f"❌ Unknown voice `{voice}`. Available: {available}")
            return

        _LOGGER.info(
            "tts_request",
            author=user.display_name,
            guild_id=guild_id,
            length=len(message),
            voice=voice_id,
        )

        try:
            tts_response = await self.model.mistral.audio.speech.complete_async(
                model=self.model.mistral_model,
                input=message,
                voice_id=voice_id,
                response_format="mp3",
            )
        except Exception as err:
            _LOGGER.error("tts_failed", error=str(err))
            await inter.edit_original_response(content="⚠️ Failed to generate speech. Check logs for details.")
            return

        try:
            vc = await self._ensure_voice_connection(guild_id, voice_state.channel.id)
        except Exception as err:
            _LOGGER.error("voice_connect_failed", error=str(err))
            await inter.edit_original_response(content="⚠️ Failed to connect to voice channel.")
            return

        audio_data = b64.b64decode(tts_response.audio_data)

        try:
            from disnake.ext.voice import AudioSource
            source = AudioSource(audio_data)
            vc.play(source)
        except Exception as err:
            _LOGGER.error("audio_play_failed", error=str(err))
            await inter.edit_original_response(content="⚠️ Failed to play audio.")
            return

        self.model.last_active[guild_id] = time.monotonic()
        await inter.edit_original_response(content="🔊 Playing...")

    @commands.slash_command(name="voice_delete", description="Delete a custom voice")
    async def voice_delete(
        self,
        inter,
        voice: str = commands.Param(description="Voice to delete", autocomplete=voice_autocomplete),
    ):
        await inter.response.defer()

        voice_name = voice

        if voice_name not in self.model.custom_voices:
            await inter.edit_original_response(content=f"❌ Voice `{voice_name}` not found.")
            return

        voice_id = self.model.custom_voices[voice_name]

        try:
            await self.model.mistral.audio.voices.delete_async(voice_id=voice_id)
        except Exception as err:
            _LOGGER.error("voice_delete_failed", error=str(err), name=voice_name)
            await inter.edit_original_response(content="⚠️ Failed to delete voice from Mistral. Check logs.")
            return

        del self.model.custom_voices[voice_name]
        self.model.deleted_voices.add(voice_name)
        self.model.save_voices()
        await inter.edit_original_response(content=f"🗑️ Voice `{voice_name}` deleted.")

    @tasks.loop(seconds=_SYNC_INTERVAL)
    async def _voice_sync(self) -> None:
        """Sync voices from Mistral periodically."""
        await self.model.sync_voices()

    @_voice_sync.before_loop
    async def before_voice_sync(self):
        await self.bot.wait_until_ready()

    @tasks.loop(seconds=_AUTO_LEAVE_INTERVAL)
    async def _auto_leave(self) -> None:
        """Leave voice channels with no activity in the last 5 minutes."""
        now = time.monotonic()
        to_remove = []

        for guild_id, last_time in list(self.model.last_active.items()):
            if now - last_time > _AUTO_LEAVE_THRESHOLD:
                for vc in self.bot.voice_clients:
                    if vc.guild.id == guild_id:
                        await vc.disconnect()
                        _LOGGER.info("auto_left_voice", guild_id=guild_id, reason="inactive")
                        break
                to_remove.append(guild_id)

        for guild_id in to_remove:
            del self.model.last_active[guild_id]

    @_auto_leave.before_loop
    async def before_auto_leave(self):
        await self.bot.wait_until_ready()

    def cog_load(self):
        _LOGGER.info("voice_plugin_loaded")


def setup(bot):
    bot.add_cog(VoiceCog(bot, bot.vox_model))
