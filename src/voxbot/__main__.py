import base64 as b64
import io
import logging
import random
import time
from pathlib import Path
from typing import Annotated

import attrs
import cyclopts
import discord
import dotenv
import structlog
from discord.ext import commands, tasks
from mistralai.client import Mistral

dotenv.load_dotenv(Path(__file__).resolve().parents[2] / ".env")

type GuildIdT = int

LOGGER = structlog.get_logger("voxtral")

VOICES = ["sad", "frustrated", "excited", "confident", "cheerful", "angry"]
VOICE_PREFIX = "en_paul_"

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}


@attrs.define(frozen=True, kw_only=True)
class VoxBotConfig:
    discord_token: str
    command_prefix: str = "!"
    mistral_api_key: str
    mistral_default_model: str = "voxtral-mini-tts-2603"


class VoiceCog(commands.Cog):
    def __init__(self, bot: VoxBot) -> None:
        self.bot = bot

    @commands.command(name="trainvoice")
    async def _trainvoice(self, ctx: commands.Context) -> None:
        if not ctx.message.attachments:
            await ctx.send("❌ Attach an audio file to train a voice.")
            return

        attachment = ctx.message.attachments[0]
        ext = Path(attachment.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            await ctx.send(
                f"❌ Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            return

        voice_name = Path(attachment.filename).stem
        if not voice_name.replace("_", "").replace("-", "").isalnum():
            await ctx.send("❌ Filename must be alphanumeric (underscores/hyphens OK).")
            return

        await ctx.send(f"🎤 Training voice `{voice_name}`... this may take a moment.")

        try:
            audio_bytes = await attachment.read()
            audio_b64 = b64.b64encode(audio_bytes).decode()
        except Exception as err:
            await LOGGER.aerror("voice_download_failed", error=str(err))
            await ctx.send("⚠️ Failed to download audio file.")
            return

        try:
            voice = await self.bot.mistral.audio.voices.create_async(
                name=voice_name,
                sample_audio=audio_b64,
                sample_filename=attachment.filename,
            )
        except Exception as err:
            await LOGGER.aerror("voice_train_failed", error=str(err), name=voice_name)
            await ctx.send("⚠️ Voice training failed. Check logs.")
            return

        self.bot._custom_voices[voice_name] = voice.id
        await ctx.send(
            f"✅ Voice `{voice_name}` trained! Use it with: `!speak --voice {voice_name} message`"
        )
        await ctx.message.delete()

    @commands.command(name="speak")
    async def _speak(
        self,
        ctx: commands.Context,
        voice: str | None = None,
        *,
        message: str = "woof! woofwoof! ...",
    ) -> None:
        if getattr(ctx.author, "voice", None) is None:
            await ctx.send("❌ You must be in a voice channel to use this command.")
            return

        assert isinstance(ctx.author, discord.Member)
        assert ctx.author.voice is not None

        if voice:
            voice_id = self.bot._custom_voices.get(voice)
            if not voice_id:
                available = ", ".join(self.bot._custom_voices.keys()) or "none"
                await ctx.send(f"❌ Unknown voice `{voice}`. Available: {available}")
                return
        else:
            voice_id = f"{VOICE_PREFIX}{random.choice(VOICES)}"

        await LOGGER.ainfo(
            "tts_request", author=ctx.author.name, length=len(message), voice=voice_id
        )

        voice_client = await self.bot.ensure_voice(
            voice=ctx.voice_client, channel=ctx.author.voice.channel
        )

        try:
            tts_response = await self.bot.mistral.audio.speech.complete_async(
                model=self.bot.config.mistral_default_model,
                input=message,
                voice_id=voice_id,
                response_format="mp3",
            )
        except Exception as err:
            await LOGGER.aerror("tts_failed", error=str(err))
            await ctx.send("⚠️ Failed to generate speech. Check logs for details.")
            return

        if voice_client.is_playing():
            voice_client.stop()

        audio_bytes = b64.b64decode(tts_response.audio_data)
        voice_client.play(
            discord.FFmpegPCMAudio(io.BytesIO(audio_bytes), pipe=True),
            after=lambda e: LOGGER.error("playback_error", error=str(e)) if e else None,
        )

        await ctx.message.delete()


class VoxBot(commands.Bot):
    def __init__(self, config: VoxBotConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.command_prefix, intents=intents)

        self._last_active: dict[GuildIdT, float] = {}
        self._custom_voices: dict[str, str] = {}
        self.config = config
        self.mistral = Mistral(api_key=config.mistral_api_key)

    async def setup_hook(self) -> None:
        await self.add_cog(VoiceCog(bot=self))
        self.inactivity_watchdog.start()
        await LOGGER.ainfo("bot_initializing", version="2026.3.27")

    async def on_ready(self) -> None:
        await LOGGER.ainfo("bot_online", user=str(self.user), guilds=len(self.guilds))

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        await self.process_commands(message)

    @tasks.loop(seconds=15.0)
    async def inactivity_watchdog(self) -> None:
        ONE_MINUTE = 60.0
        now = time.monotonic()

        for voice in list(self.voice_clients):
            assert isinstance(voice, discord.VoiceClient)

            if voice.is_playing():
                self._last_active[voice.guild.id] = now
                continue

            if (idle := now - self._last_active.get(voice.guild.id, now)) >= ONE_MINUTE:
                await LOGGER.ainfo(
                    "idle_disconnect", guild=voice.guild.name, idle_for=round(idle)
                )
                self._last_active.pop(voice.guild.id, None)
                await voice.disconnect()

    @inactivity_watchdog.before_loop
    async def _before_inactivity_watchdog(self) -> None:
        await self.wait_until_ready()

    async def ensure_voice(
        self, voice: discord.VoiceClient | None, channel: discord.VoiceChannel
    ) -> discord.VoiceClient:
        if voice is None:
            voice = await channel.connect()
        elif voice.channel != channel:
            await voice.move_to(channel)

        self._last_active[voice.guild.id] = time.monotonic()
        return voice


# ── CLI ──────────────────────────────────────────────────────────────────────

cli = cyclopts.App(help="Voxtral TTS Discord Runner")


@cli.default
def start_bot(
    discord_token: Annotated[str, cyclopts.Parameter(env_var="DISCORD_TOKEN")],
    mistral_api_key: Annotated[str, cyclopts.Parameter(env_var="MISTRAL_API_KEY")],
    prefix: str = "!",
    voice: str = "en_paul_excited",
) -> int:
    """Validate config and launch the bot."""
    config = VoxBotConfig(
        discord_token=discord_token,
        command_prefix=prefix,
        mistral_api_key=mistral_api_key,
    )

    try:
        bot = VoxBot(config=config)

        LOGGER.info("starting")
        bot.run(config.discord_token, log_handler=None)
        return 0

    except KeyboardInterrupt:
        LOGGER.info("shutdown")
        return 0

    except Exception:
        LOGGER.exception("error")
        return 1


# ── SCRIPT ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dotenv.load_dotenv()

    # 1. This is the "Final Output" formatter
    # It takes the data and turns it into the pretty console lines you like
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        ],
    )

    # 2. Setup the Standard Library Handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # 3. Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            # CRITICAL: This wraps the dict so the stdlib Formatter can "copy()" it
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    raise SystemExit(cli())
