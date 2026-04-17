import base64 as b64
import io
import random
import time
from pathlib import Path

import crescent
import hikari
import hikariwave

from voxbot.model import VoxModel

plugin = crescent.Plugin[hikari.GatewayBot, VoxModel]()

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_VOICES = ["sad", "frustrated", "excited", "confident", "cheerful", "angry"]
VOICE_PREFIX = "en_paul_"


@plugin.include
@crescent.command(name="trainvoice", description="Train a custom TTS voice")
class TrainVoice:
    async def callback(self, ctx: crescent.Context) -> None:
        msgs = ctx.messages
        if not msgs:
            await ctx.respond("❌ No message with attachment found.")
            return

        msg = msgs[0]
        if not msg.attachments:
            await ctx.respond("❌ Attach an audio file to train a voice.")
            return

        attachment = msg.attachments[0]
        ext = Path(attachment.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            await ctx.respond(
                f"❌ Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            return

        voice_name = Path(attachment.filename).stem
        if not voice_name.replace("_", "").replace("-", "").isalnum():
            await ctx.respond(
                "❌ Filename must be alphanumeric (underscores/hyphens OK)."
            )
            return

        await ctx.respond(
            f"🎤 Training voice `{voice_name}`... this may take a moment."
        )

        try:
            audio_bytes = await attachment.read()
            audio_b64_str = b64.b64encode(audio_bytes).decode()
        except Exception as err:
            plugin.model._log.error("voice_download_failed", error=str(err))
            await ctx.edit_response("⚠️ Failed to download audio file.")
            return

        try:
            voice = await plugin.model.mistral.audio.voices.create_async(
                name=voice_name,
                sample_audio=audio_b64_str,
                sample_filename=attachment.filename,
            )
        except Exception as err:
            plugin.model._log.error(
                "voice_train_failed", error=str(err), name=voice_name
            )
            await ctx.edit_response("⚠️ Voice training failed. Check logs.")
            return

        plugin.model.custom_voices[voice_name] = voice.id
        await ctx.edit_response(
            f"✅ Voice `{voice_name}` trained! Use it with: `/speak --voice {voice_name} message`"
        )


@plugin.include
@crescent.command(name="speak", description="Speak a message in voice")
class Speak:
    voice = crescent.option(str, "Voice to use", default=None)
    message = crescent.param(default="woof! woofwoof! ...")

    async def callback(self, ctx: crescent.Context) -> None:
        member = ctx.member
        if member is None:
            await ctx.respond("❌ Could not find member.")
            return

        voice_state = member.guild_voice_state
        if voice_state is None or voice_state.channel_id is None:
            await ctx.respond("❌ You must be in a voice channel to use this command.")
            return

        voice_channel_id = voice_state.channel_id

        if self.voice:
            voice_id = plugin.model.custom_voices.get(self.voice)
            if not voice_id:
                available = ", ".join(plugin.model.custom_voices.keys()) or "none"
                await ctx.respond(
                    f"❌ Unknown voice `{self.voice}`. Available: {available}"
                )
                return
        else:
            voice_id = f"{VOICE_PREFIX}{random.choice(DEFAULT_VOICES)}"

        plugin.model._log.info(
            "tts_request",
            author=member.display_name,
            length=len(self.message),
            voice=voice_id,
        )

        try:
            tts_response = await plugin.model.mistral.audio.speech.complete_async(
                model="voxtral-mini-tts-2603",
                input=self.message,
                voice_id=voice_id,
                response_format="mp3",
            )
        except Exception as err:
            plugin.model._log.error("tts_failed", error=str(err))
            await ctx.respond("⚠️ Failed to generate speech. Check logs for details.")
            return

        guild_id = ctx.guild_id
        assert guild_id is not None

        vc = plugin.model.voice_client
        assert vc is not None

        connection = await vc.connect(guild_id, voice_channel_id)

        if connection.player.is_playing:
            connection.player.stop()

        audio_data = b64.b64decode(tts_response.audio_data)
        source = hikariwave.MemoryAudioSource(audio_data)
        connection.player.play(source)

        plugin.model.last_active[guild_id] = time.monotonic()
        await ctx.respond("🔊 Playing...")


@plugin.load_hook
def on_load() -> None:
    plugin.model._log.info("voice_plugin_loaded")


@plugin.unload_hook
def on_unload() -> None:
    pass
