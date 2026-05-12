import os
import unittest
from unittest.mock import AsyncMock

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.bot import VoxBot


class BotLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_shutdown_sets_exit_code_and_closes(self) -> None:
        bot = VoxBot()
        bot.close = AsyncMock()

        await bot.request_shutdown(reason="test", exit_code=12)

        self.assertEqual(bot.exit_code, 12)
        bot.close.assert_awaited_once()

    async def test_request_restart_uses_tempfail_exit_code(self) -> None:
        bot = VoxBot()
        bot.close = AsyncMock()
        bot.health_runtime.record_restart_requested = AsyncMock()

        await bot.request_restart(reason="deploy")

        self.assertEqual(bot.exit_code, 75)
        bot.health_runtime.record_restart_requested.assert_awaited_once_with("deploy")
        bot.close.assert_awaited_once()
