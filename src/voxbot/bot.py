import asyncio
import signal
import sys

from discord.ext import commands
import discord
import structlog

from voxbot.runtime.docket import BotDocketRuntime
from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import settings
from voxbot.store import runtime
from voxbot import __project__, error_reports, errors, utils

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
        self._running_tasks: set[asyncio.Task] = set()

        self.exit_code = 0
        self.health_runtime = RedisHealthRuntime()
        self.docket_runtime = BotDocketRuntime(self)

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
            except (NotImplementedError, RuntimeError):
                continue

    async def _load_plugins(self) -> None:
        """Load all Cogs from the plugins directory."""
        for subdir in runtime.library_root.glob("plugins/*/"):
            if not (subdir / "__init__.py").exists():
                continue

            await self.load_extension(name=f"voxbot.plugins.{subdir.name}")
    
    async def _determine_tree_needs_sync(self) -> None:
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
        await self._determine_tree_needs_sync()

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

    async def on_error(self, event_method: str, *args, **kwargs) -> None:
        """
        Handle errors that occur during event processing or command execution.
        
        Further reading:
          https://discordpy.readthedocs.io/en/stable/api.html#discord.on_error
        """
        await self.health_runtime.record_error(exc=Exception(f"{event_method} failed"))

        exc_type, exc, tb = sys.exc_info()
        error_detail = exc or Exception(f"Unknown error in {event_method}")

        # Record to health runtime
        await self.health_runtime.record_error(exc=error_detail)

        # DM owner with a full traceback attachment.
        await error_reports.dm_owner_error_report(
            self,
            subject=f"Error occurred in {event_method}",
            title=f"Bot error in `{event_method}`",
            details={
                "Details": f"{type(error_detail).__name__}: {error_detail}",
                "Args": repr(args),
                "Kwargs": repr(kwargs),
            },
            filename="error_log.txt",
            error=error_detail,
            exc_info=(exc_type, exc, tb),
        )

        await super().on_error(event_method, *args, **kwargs)

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        """
        Handle errors that occur during command execution.
        
        Further reading:
          https://discordpy.readthedocs.io/en/stable/api.html#discord.on_command_error
        """
        _LOGGER.error("command_error", error=str(error), command=ctx.command.name if ctx.command else "unknown")

        await self.health_runtime.record_error(exc=error)

        if isinstance(error, errors.VoxCheckFailure):
            await ctx.send(str(error), ephemeral=True)

        source_error = getattr(error, "original", error)
        await error_reports.dm_owner_error_report(
            self,
            subject=f"Command error in {ctx.command}",
            title=f"Command `{ctx.command}` failed",
            details={
                "Channel": str(ctx.channel),
                "User": f"{ctx.author} ({ctx.author.id})",
                "Details": f"{type(source_error).__name__}: {source_error}",
            },
            filename="command_error_log.txt",
            error=source_error,
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
