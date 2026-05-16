from typing import Any
import asyncio
import io
import signal
import sys

from discord.ext import commands
from discord.ext.commands._types import BotT
import discord
import structlog

from voxbot import __project__, errors, utils
from voxbot.runtime.docket import BotDocketRuntime
from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import settings
from voxbot.store import runtime

_LOGGER = structlog.get_logger(__name__)


class VoxBot(commands.Bot):
    """
    Vox 🦜, the bot.

    BOT LIFECYCLE
      .__init__
      .setup_hook
      .login
      .on_connect / .on_resumed
      .on_ready
      .on_error
      .on_disconnect
    """

    def __init__(self):
        super().__init__(
            command_prefix="!",  # DEV NOTE: We used /slash commands everywhere, but d.py requires this internally.
            intents=settings.required_intents,
        )

        # Internal background tasks. All external background tasks will be added to DocketRuntime.
        self._running_tasks: set[asyncio.Task[Any]] = set()

        self.exit_code = 0
        self.health_runtime = RedisHealthRuntime()
        self.docket_runtime = BotDocketRuntime(self)

    @property
    def me(self) -> discord.User:
        """
        The Bot User.

        If you need to edit the Bot's profile, user `VoxBot.user` instead.
        """
        assert self.user is not None, "Attempted to fetch a User while not connected to the Discord Gateway."
        user = self.get_user(self.user.id)
        assert user is not None, "Attempted to fetch a User while not connected to the Discord Gateway."
        return user

    @property
    def dad(self) -> discord.User:
        """The Owner User."""
        user = self.get_user(settings.bot_owner_id)
        assert user is not None, "Attempted to fetch a User while not connected to the Discord Gateway."
        return user

    def _install_bot_signal_handlers(self) -> None:
        """Handle shutdown events from the OS."""
        loop = asyncio.get_running_loop()

        def handle_signal(s: signal.Signals) -> None:
            reason = f"{s.name} received"
            coro = self.request_shutdown(reason=reason, exit_code=0)
            task = asyncio.create_task(coro, name=f"voxbot-{reason.replace(' ', '-')}")
            utils.no_task_dangling(task, struct=self._running_tasks)

        for sig_num in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig_num, handle_signal, sig_num)
            except NotImplementedError, RuntimeError:
                continue

    async def _load_plugins(self) -> None:
        """Load all Cogs from the plugins directory."""
        for subdir in runtime.library_root.glob("plugins/*/"):
            if not (subdir / "__init__.py").exists():
                continue

            await self.load_extension(name=f"voxbot.plugins.{subdir.name}")

    async def _determine_if_tree_needs_sync(self) -> None:
        """Sync the CommandTree if it's necessary."""
        # DEV NOTE:
        #   Global syncs are rate-limited (2/hr), so skip them when the
        #   command tree is unchanged since the last successful sync.
        #
        #   This bot was never intended to be sharded / global, so this
        #   is safe to run consistently, but we still can skip it if the
        #   commands haven't changed.
        this_hash = utils.hash_command_tree(self.tree)
        last_hash = runtime.commands_sha.read_text(encoding="utf-8").strip()

        if this_hash == last_hash:
            _LOGGER.info("skipped_global_sync_unchanged", hash=this_hash[:12])
            return

        if settings.debug_guild:
            guild = discord.Object(id=settings.debug_guild)
            self.tree.copy_global_to(guild=guild)
        else:
            guild = None

        await self.tree.sync(guild=guild)
        runtime.commands_sha.write_text(this_hash, encoding="utf-8")

        _LOGGER.info("synced_commands", hash=this_hash[:12], guild=settings.debug_guild or "globally")

    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    async def setup_hook(self) -> None:
        """
        Called once when the bot is starting up.

        Further reading:
          https://discordpy.readthedocs.io/en/stable/api.html#discord.setup_hook
        """
        runtime.ensure_directories()

        self._install_bot_signal_handlers()

        await self.health_runtime.start(self)
        await self._load_plugins()
        await self._determine_if_tree_needs_sync()

        # Manages its own async lifecycle.
        self.docket_runtime.start()

    async def on_ready(self) -> None:
        """
        Called when the client is done preparing the data received from Discord.

        Further reading:
          https://discordpy.readthedocs.io/en/stable/api.html#discord.on_ready
        """
        _LOGGER.info("bot_online", version=__project__.__version__)
        await self.health_runtime.mark_ready(True, bot=self)

    async def on_error(self, event_method: str, *args: Any, **kwargs: Any) -> None:
        """
        Handle errors that occur during event processing or command execution.

        Further reading:
          https://discordpy.readthedocs.io/en/stable/api.html#discord.on_error
        """
        exc_type, exc, tb = sys.exc_info()

        _LOGGER.error("bot.on_error", exc_type=exc_type, exc=str(exc), event_method=event_method)

        assert exc_type is not None, "Not handling an active Exception."
        assert exc is not None, "Not handling an active Exception."
        assert tb is not None, "Not handling an active Exception."

        # Record to health runtime
        await self.health_runtime.record_error(exc=exc)

        # DM owner with a full traceback attachment.
        mdc_exc = utils.MdExceptionFormatter(exc_info=(exc_type, exc, tb))

        await self.dad.send(
            f"🚨 **{exc}** — {event_method}",
            file=discord.File(io.BytesIO(mdc_exc.format(locals=True).encode("utf-8")), filename="error_trace.md"),
        )

        await super().on_error(event_method, *args, **kwargs)

    async def on_command_error(self, ctx: commands.Context[BotT], error: commands.CommandError) -> None:
        """
        Handle errors that occur during command execution.

        Further reading:
          https://discordpy.readthedocs.io/en/stable/api.html#discord.on_command_error
        """
        cause_exc = getattr(error, "original", error)
        exc_type, exc, tb = type(cause_exc), cause_exc, cause_exc.__traceback__

        _LOGGER.error(
            "bot.command_error",
            exc_type=exc_type,
            exc=str(exc),
            command=ctx.command.name if ctx.command else "unknown",
        )

        assert exc_type is not None, "Not handling an active Exception."
        assert exc is not None, "Not handling an active Exception."
        assert tb is not None, "Not handling an active Exception."

        # Record to health runtime
        await self.health_runtime.record_error(exc=error)

        # Notify the User what's up.
        if isinstance(error, errors.VoxCheckFailure):
            await ctx.send(str(error), ephemeral=True)

        # DM owner with a full traceback attachment.
        mdc_exc = utils.MdExceptionFormatter(exc_info=(exc_type, exc, tb))
        channel_mention = ctx.me if isinstance(ctx.channel, discord.abc.PrivateChannel) else ctx.channel.mention

        await self.dad.send(
            f"🚨 **{error}** — {ctx.author.mention} in {channel_mention}",
            file=discord.File(io.BytesIO(mdc_exc.format(locals=True).encode("utf-8")), filename="error_trace.md"),
        )

        await super().on_command_error(ctx, error)

    async def request_shutdown(self, *, reason: str, exit_code: int = 0) -> None:
        """Politely stop the Bot."""
        self.exit_code = exit_code

        _LOGGER.info("bot_shutdown_requested", reason=reason, exit_code=exit_code)

        await self.close()

    async def request_restart(self, *, reason: str) -> None:
        """Enqueue a restart.."""
        await self.health_runtime.record_restart_requested(reason)
        await self.request_shutdown(reason=reason, exit_code=75)

    async def close(self) -> None:
        """Close the bot, stopping each runtime."""
        # DEV NOTE:
        #   Ordered stop (docket -> health) because DurableTasks may try to push
        #   notifications to Redis.
        await self.docket_runtime.stop()
        await self.health_runtime.stop()
        await super().close()
