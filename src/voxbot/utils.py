import asyncio
import hashlib
import json

import discord
import redis

from voxbot.settings import settings

RedisClient = redis.asyncio.from_url(settings.redis_url, decode_responses=True)


def hash_command_tree(tree: discord.app_commands.CommandTree) -> str:
    """Convert a discord CommandTree into a stable string for comparison."""
    payloads = [cmd.to_dict(tree) for cmd in tree.get_commands()]
    payloads.sort(key=lambda d: d.get("name", ""))
    blob = json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def no_task_dangling(task: asyncio.Task, *, struct: set) -> None:
    """
    Perform lifecycle management on asyncio.Tasks.
    
    Further reading:
      https://docs.astral.sh/ruff/rules/asyncio-dangling-task/
      https://textual.textualize.io/blog/2023/02/11/the-heisenbug-lurking-in-your-async-code/
    """
    struct.add(task)
    task.add_done_callback(struct.discard)
