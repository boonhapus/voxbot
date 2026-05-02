"""Voice plugin main Cog."""

import base64
import pathlib
import re
import time

from disnake.ext import commands
from disnake import Attachment
import structlog

from voxbot import ai, dota_wiki
from voxbot.errors import MistralError, TTSError
from voxbot.model import VoxModel
from voxbot.services.mistral import MistralService
from voxbot.services.tts import TTSProcessor
from voxbot.plugins.voice.state import VoiceState

_LOGGER = structlog.get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_MESSAGE = "woof! woofwoof! ..."
VOICE_NAME_RE = re.compile(r"^[\w\-]+$")


class VoiceCog(commands.Cog):
    """Main voice plugin cog."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.mistral_service = MistralService()
        self.vox_model = VoxModel()

        # Expose to bot for other components (tasks, etc.)
        bot.mistral_service = self.mistral_service
        bot.vox_model = self.vox_model

    async def voice_autocomplete(self, inter, user_input: str):
        """Autocomplete voice names."""
        voices = VoiceState.get_voice_list(self.mistral_service.voices.custom_voices)
        return VoiceState.filter_voice_list(voices, user_input)

    @commands.slash_command(
        name="train", group="voice", description="Train a custom TTS voice"
    )
    async def voice_train(
        self,
        inter,
        audio: Attachment = commands.Param(
            default=None, description="Audio sample to train from"
        ),
        hero: str = commands.Param(
            default=None,
            description="Dota hero name; scrapes 20-30s of voice lines",
        ),
    ):
        await inter.response.defer()

        if (audio is None) == (hero is None):
            await inter.edit_original_response(
                content="❌ Provide exactly one of `audio` or `hero`."
            )
            return

        if hero is not None:
            await inter.edit_original_response(
                content=f"🔎 Scraping voice lines for `{hero}` from the Dota wiki..."
            )
            try:
                canonical, audio_bytes, sample_filename = (
                    await dota_wiki.sample_voice_lines(hero)
                )
            except dota_wiki.WikiError as err:
                await inter.edit_original_response(content=f"❌ {err}")
                return
            except Exception as err:
                _LOGGER.error("wiki_scrape_failed", error=str(err), hero=hero)
                await inter.edit_original_response(
                    content="⚠️ Failed to scrape Dota wiki. Check logs."
                )
                return

            voice_name = re.sub(r"[^A-Za-z0-9]+", "_", canonical).strip("_").title()
        else:
            ext = pathlib.Path(audio.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                await inter.edit_original_response(
                    content=f"❌ Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
                )
                return

            voice_name = pathlib.Path(audio.filename).stem.title()
            if not VOICE_NAME_RE.match(voice_name):
                await inter.edit_original_response(
                    content="❌ Filename must be alphanumeric (underscores/hyphens OK)."
                )
                return

            try:
                audio_bytes = await audio.read()
            except Exception as err:
                _LOGGER.error("voice_download_failed", error=str(err))
                await inter.edit_original_response(
                    content="⚠️ Failed to download audio file."
                )
                return
            sample_filename = audio.filename

        await inter.edit_original_response(
            content=f"🎤 Training voice `{voice_name}`... this may take a moment."
        )

        audio_b64_str = base64.b64encode(audio_bytes).decode()

        try:
            await self.mistral_service.train_voice(
                voice_name, audio_b64_str, sample_filename
            )
        except MistralError as err:
            await inter.edit_original_response(
                content="⚠️ Voice training failed. Check logs."
            )
            return

        if hero is not None:
            self.vox_model.hero_origins[voice_name] = canonical
        else:
            self.vox_model.hero_origins.pop(voice_name, None)

        await inter.edit_original_response(
            content=f"✅ Voice `{voice_name}` trained! Use it with: `/voice_speak --voice {voice_name}`"
        )

    @commands.slash_command(
        name="speak",
        group="voice",
        description="Speak a message in your voice channel",
    )
    async def voice_speak(
        self,
        inter,
        message: str = commands.Param(default=None, description="Message to speak"),
        prompt: str = commands.Param(
            default=None,
            description="Short prompt — AI generates a line",
        ),
        voice: str = commands.Param(
            default=None, description="Voice to use", autocomplete=voice_autocomplete
        ),
    ):
        await inter.response.defer()

        if message is not None and prompt is not None:
            await inter.edit_original_response(
                content="❌ Provide `message` OR `prompt`, not both."
            )
            return

        if inter.guild_id is None:
            await inter.edit_original_response(content="❌ Must be used in a server.")
            return

        voice_state = inter.author.voice
        if voice_state is None or voice_state.channel is None:
            await inter.edit_original_response(
                content="❌ You must be in a voice channel."
            )
            return

        voice_id = VoiceState.resolve_voice(
            voice, self.mistral_service.voices.custom_voices
        )
        if voice_id is None:
            available = (
                ", ".join(self.mistral_service.voices.custom_voices.keys())
                or "none"
            )
            await inter.edit_original_response(
                content=f"❌ Unknown voice `{voice}`. Available: {available}"
            )
            return

        if prompt is not None:
            hero_context = self.vox_model.hero_origins.get(voice or "")
            try:
                text = await ai.generate_line(prompt, hero_context)
            except Exception as err:
                _LOGGER.error("ai_generate_failed", error=str(err))
                await inter.edit_original_response(
                    content="⚠️ Failed to generate line. Check logs."
                )
                return
            if not text:
                await inter.edit_original_response(
                    content="⚠️ AI returned an empty line."
                )
                return
            await inter.edit_original_response(content=f"🤖 {text}")
        else:
            text = message or DEFAULT_MESSAGE

        _LOGGER.info(
            "tts_request",
            author=inter.author.display_name,
            guild_id=inter.guild_id,
            length=len(text),
            voice=voice_id,
            generated=prompt is not None,
        )

        try:
            tts_response = await self.mistral_service.text_to_speech(text, voice_id)
        except MistralError as err:
            await inter.edit_original_response(
                content="⚠️ Failed to generate speech. Check logs."
            )
            return

        try:
            source, tmp_path = TTSProcessor.prepare_audio_source(tts_response)
        except TTSError as err:
            await inter.edit_original_response(content="⚠️ Failed to prepare audio.")
            return

        try:
            for vc in self.bot.voice_clients:
                if vc.guild.id == inter.guild_id:
                    if vc.channel.id != voice_state.channel.id:
                        await vc.move_to(voice_state.channel)
                    break
            else:
                vc = await voice_state.channel.connect()

            def cleanup(error):
                if not error:
                    TTSProcessor.cleanup_temp_file(tmp_path)

            vc.play(source, after=cleanup)
        except Exception as err:
            _LOGGER.error("voice_connect_failed", error=str(err))
            TTSProcessor.cleanup_temp_file(tmp_path)
            await inter.edit_original_response(
                content="⚠️ Failed to connect or play audio."
            )
            return

        self.vox_model.last_active[inter.guild_id] = time.monotonic()
        playing_label = f"🔊 {text}" if prompt is not None else "🔊 Playing..."
        await inter.edit_original_response(content=playing_label)

    @commands.slash_command(
        name="delete", group="voice", description="Delete a custom voice"
    )
    async def voice_delete(
        self,
        inter,
        voice: str = commands.Param(
            description="Voice to delete", autocomplete=voice_autocomplete
        ),
    ):
        await inter.response.defer()

        if voice not in self.mistral_service.voices.custom_voices:
            await inter.edit_original_response(
                content=f"❌ Voice `{voice}` not found."
            )
            return

        voice_id = self.mistral_service.voices.custom_voices[voice]

        try:
            await self.mistral_service.delete_voice(voice_id)
        except MistralError as err:
            await inter.edit_original_response(
                content="⚠️ Failed to delete voice. Check logs."
            )
            return

        del self.mistral_service.voices.custom_voices[voice]
        self.mistral_service.voices.deleted_voices.add(voice)
        self.vox_model.hero_origins.pop(voice, None)
        self.mistral_service.save_voices()
        await inter.edit_original_response(content=f"🗑️ Voice `{voice}` deleted.")

    def cog_load(self) -> None:
        """Called when cog is loaded."""
        _LOGGER.info("voice_cog_loaded")
