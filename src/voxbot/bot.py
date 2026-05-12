import asyncio
import hashlib
import importlib.metadata
import json
import pathlib

from discord.ext import commands
import discord
import structlog

from voxbot.runtime.docket import BotDocketRuntime
from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)
_COMMAND_HASH_PATH = pathlib.Path.home() / ".voxbot" / "commands.sha"


class VoxBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.voice_states = True
        intents.message_content = True

        super().__init__(
            command_prefix="!",  # Required by d.py, but we use slash commands
            intents=intents,
        )

        self.exit_code = 0
        self.health_runtime = RedisHealthRuntime(settings)
        self.docket_runtime = BotDocketRuntime(self)
        self._health_stopped = False

    async def setup_hook(self) -> None:
        """Called once when the bot is starting up."""
        await self.health_runtime.start(self)
        await self._load_plugins()

        # Sync slash commands. Global syncs are rate-limited (2/hr), so skip
        # them when the command tree is unchanged since the last successful sync.
        if settings.debug_guild:
            guild = discord.Object(id=int(settings.debug_guild))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            _LOGGER.info("synced_commands_to_debug_guild", guild_id=settings.debug_guild)
        else:
            current_hash = _hash_command_tree(self.tree)
            previous_hash = _read_previous_command_hash()
            if current_hash == previous_hash:
                _LOGGER.info("skipped_global_sync_unchanged", hash=current_hash[:12])
            else:
                await self.tree.sync()
                _write_command_hash(current_hash)
                _LOGGER.info("synced_commands_globally", hash=current_hash[:12])

        asyncio.create_task(self.docket_runtime.start(), name="voxbot-docket-runtime")

    async def _load_plugins(self) -> None:
        """Load all plugins from plugins/ directory using plugin/cog pattern."""
        project_dir = pathlib.Path(__file__).parent

        for subdir in project_dir.glob("plugins/*/"):
            if not (subdir / "__init__.py").exists():
                continue

            await self.load_extension(name=f"voxbot.plugins.{subdir.name}")

    async def on_ready(self):
        _LOGGER.info("bot_online", version=importlib.metadata.version("voxbot"))
        await self.health_runtime.mark_ready(True, bot=self)

    async def on_error(self, event_method: str, /, *args, **kwargs) -> None:
        """Handle errors that occur during event processing or command execution."""
        await self.health_runtime.record_error(f"{event_method} failed")

        owner_id_str = settings.discord_owner_ids
        if owner_id_str:
            try:
                owner_id = int(owner_id_str.split(',')[0]) # Take the first owner ID
                owner = self.get_user(owner_id) or await self.fetch_user(owner_id)

                if owner:
                    error_message_parts = [f"🚨 Bot Error Occurred!"]
                    error_message_parts.append(f"**Event/Method:** `{event_method}`")

                    # Attempt to extract the exception details
                    error_detail = None
                    if 'error' in kwargs: # Likely a command error
                        error_detail = kwargs['error']
                    elif args: # Could be an event error, try to get the exception from args if it's an Exception instance
                        for arg in args:
                            if isinstance(arg, Exception):
                                error_detail = arg
                                break
                    
                    if error_detail:
                        error_message_parts.append(f"**Details:**\n```\n{type(error_detail).__name__}: {error_detail}\n```")
                    else:
                        # Fallback if no clear exception is found
                        error_message_parts.append(f"**Arguments:** `{args}`")
                        error_message_parts.append(f"**Keyword Arguments:** `{kwargs}`")
                    
                    full_error_message = "\n".join(error_message_parts)

                    try:
                        await owner.send(full_error_message)
                        _LOGGER.info("sent_error_dm_to_owner", owner_id=owner_id, event=event_method)
                    except discord.Forbidden:
                        _LOGGER.warning("failed_to_send_dm_to_owner", owner_id=owner_id, reason="DM blocked by user.")
                    except Exception as e:
                        _LOGGER.warning("failed_to_send_dm_to_owner", owner_id=owner_id, reason=f"An unexpected error occurred: {e}")

            except ValueError:
                _LOGGER.warning("invalid_owner_id_format_for_dm", owner_ids=owner_id_str)
            except Exception as e:
                _LOGGER.warning("error_while_preparing_dm", error=str(e))

        # Always call the superclass method to ensure default error handling
        await super().on_error(event_method, *args, **kwargs)

    async def request_shutdown(self, *, reason: str, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        _LOGGER.info("bot_shutdown_requested", reason=reason, exit_code=exit_code)
        await self.close()

    async def request_restart(self, *, reason: str) -> None:
        await self.health_runtime.record_restart_requested(reason)
        await self.request_shutdown(reason=reason, exit_code=75)

    async def close(self) -> None:
        await self.docket_runtime.stop()

        if not self._health_stopped:
            self._health_stopped = True
            await self.health_runtime.stop()

        await super().close()


def _hash_command_tree(tree: discord.app_commands.CommandTree) -> str:
    payloads = [cmd.to_dict(tree) for cmd in tree.get_commands()]
    payloads.sort(key=lambda d: d.get("name", ""))
    blob = json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _read_previous_command_hash() -> str | None:
    try:
        return _COMMAND_HASH_PATH.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_command_hash(value: str) -> None:
    try:
        _COMMAND_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
        _COMMAND_HASH_PATH.write_text(value, encoding="utf-8")
    except OSError as err:
        _LOGGER.warning("command_hash_write_failed", error=str(err))
