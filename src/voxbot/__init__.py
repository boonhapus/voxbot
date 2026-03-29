# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "discord.py>=2.3",
#     "mistralai>=1.0",
#     "cyclopts>=2.5",
#     "structlog>=24.0",
#     "attrs>=23.2",
# ]
# ///

from typing import Annotated
import io
import logging
import time

from discord.ext import commands, tasks
import attrs
import cyclopts
import discord
import mistralai
import structlog

type GuildIdT = Annotated[int, "The Guild ID."]
LOGGER = structlog.get_logger("voxtral")

# ── Configuration ────────────────────────────────────────────────────────────

@attrs.define(frozen=True, kw_only=True)
class VoxBotConfig:
    """Configuration for the bot."""
    discord_token: str
    command_prefix: str = "!"
    mistral_api_key: str
    mistral_default_voice: str = "casual_male"
    mistral_default_model: str = "voxtral-tts-2603"

# ── Bot ──────────────────────────────────────────────────────────────────────

class VoxBot(commands.Bot):
    """A TTS / voice cloning discord bot."""

    def __init__(self, config: VoxBotConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.command_prefix, intents=intents)

        self._last_active: dict[GuildIdT, float] = {}
        self.config = config
        self.mistral = mistralai.Mistral(api_key=config.mistral_api_key)

    # ── EVENT LISTENERS ──────────────────────────────────────────────────────

    async def setup_hook(self) -> None:
        """
        Perform setup after loggin, but before connecting to the Websocket.

        This is only called once, in login(), and will be called before any events are
        dispatched, making it a better solution than doing such setup in the on_ready()
        event.
        
        Further reading:
          https://discordpy.readthedocs.io/en/stable/ext/commands/api.html#discord.ext.commands.Bot.setup_hook
        """
        # ADD ALL BOT COMMANDS.
        self.add_command(commands.Command(self._speak, name="speak"))

        # START THE BACKGROUND TASKS.
        self.inactivity_watchdog.start()

        # GO. :~)
        await LOGGER.ainfo("bot_initializing", version="2026.3.27")

    async def on_ready(self) -> None:
        """
        Called when the bot is done preparing the data received from Discord.

        Further reading:
          https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
        """
        await LOGGER.ainfo("bot_online", user=str(self.user), guilds=len(self.guilds))

    # ── INTERNALS ────────────────────────────────────────────────────────────

    @tasks.loop(seconds=15.0)
    async def inactivity_watchdog(self) -> None:
        """Leave the last known voice channel after some inactivity."""
        ONE_MINUTE = 60.0
        now = time.monotonic()

        for voice in list(self.voice_clients):
            assert isinstance(voice, discord.VoiceClient)

            if voice.is_playing():
                self._last_active[voice.guild.id] = now
                continue

            if (idle := now - self._last_active.get(voice.guild.id, now)) >= ONE_MINUTE:
                await LOGGER.ainfo("idle_disconnect", guild=voice.guild.name, idle_for=round(idle))
                self._last_active.pop(voice.guild.id, None)
                await voice.disconnect()

    @inactivity_watchdog.before_loop
    async def _before_inactivity_watchdog(self) -> None:
        """Wait until the bot is ready."""
        await self.wait_until_ready()

    # ── HELPERS ──────────────────────────────────────────────────────────────

    async def ensure_voice(self, voice: discord.VoiceClient | None, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """Join the channel if you're not already in it."""
        if voice is None:
            voice = await channel.connect()

        elif voice.channel != channel:
            await voice.move_to(channel)

        # REGISTER THIS CHANNEL
        self._last_active[voice.guild.id] = time.monotonic()

        return voice

    # ── COMMANDS ─────────────────────────────────────────────────────────────

    async def _speak(self, ctx: commands.Context, *, message: str) -> None:
        """Convert text to speech and play it in the author's voice channel."""
        if getattr(ctx.author, "voice", None) is None:
            await ctx.send("❌ You must be in a voice channel to use this command.")
            return
        
        assert isinstance(ctx.author, discord.Member), "Author is not a discord.Member"
        assert ctx.author.voice is not None, "Author is not in a voice channel."

        voice = await self.ensure_voice(voice=ctx.voice_client, channel=ctx.author.voice.channel)

        await LOGGER.ainfo("tts_request", author=ctx.author.name, length=len(message))

        try:
            r = await self.mistral.audio.speech.complete_async(
                model=self.config.mistral_default_model,
                input=message,
                voice_id=self.config.mistral_default_voice,
                response_format="mp3",
            )

            if voice.is_playing():
                voice.stop()

            voice.play(
                discord.FFmpegPCMAudio(io.BytesIO(r.content), pipe=True),
                after=lambda e: LOGGER.error("playback_error", error=str(e)),
            )

            await ctx.message.add_reaction("🎙️")

        except Exception as err:
            await LOGGER.aerror("tts_failed", error=str(err))
            await ctx.send("⚠️ Failed to generate speech. Check logs for details.")

# ── CLI ──────────────────────────────────────────────────────────────────────

cli = cyclopts.App(help="Voxtral TTS Discord Runner")

@cli.default
def start_bot(
    discord_token: Annotated[str, "Discord Bot Token"],
    mistral_api_key: Annotated[str, "Mistral AI API Key"],
    prefix: str = "!",
    voice: str = "casual_male",
) -> int:
    """Validate config and launch the bot."""
    config = VoxBotConfig(
        discord_token=discord_token,
        command_prefix=prefix,
        mistral_api_key=mistral_api_key,
        mistral_default_voice=voice,
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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    raise SystemExit(cli())
