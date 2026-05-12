import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault('DISCORD_TOKEN', 'test-discord-token')
os.environ.setdefault('DISCORD_OWNER_IDS', '123456789')

from voxbot.bot import VoxBot
from discord.ext import commands

class TestErrorNotifications(unittest.IsolatedAsyncioTestCase):
    async def test_notify_owner_called_on_command_error(self) -> None:
        bot = VoxBot()
        bot.fetch_user = AsyncMock(return_value=AsyncMock())
        mock_owner = bot.fetch_user.return_value
        mock_owner.send = AsyncMock()

        ctx = MagicMock(spec=commands.Context)
        ctx.command = MagicMock()
        ctx.command.name = 'test_command'
        error = commands.CommandError('test error')

        # Mock super() call for command error
        with patch.object(commands.Bot, 'on_command_error', new_callable=AsyncMock):
            await bot.on_command_error(ctx, error)

        # Verify the notification contains the error detail
        mock_owner.send.assert_awaited()
        args, _ = mock_owner.send.await_args
        self.assertIn("failed!", args[0])
        self.assertIn("CommandError: test error", args[0])

    async def test_notify_owner_called_on_event_error(self) -> None:
        bot = VoxBot()
        bot.fetch_user = AsyncMock(return_value=AsyncMock())
        mock_owner = bot.fetch_user.return_value
        mock_owner.send = AsyncMock()

        error = Exception('event error')

        # Mock super() call for on_error
        with patch.object(commands.Bot, 'on_error', new_callable=AsyncMock):
            await bot.on_error('on_ready', error=error)

        mock_owner.send.assert_awaited()
        # Verify the notification contains the error detail
        args, _ = mock_owner.send.await_args
        self.assertIn('🚨 Bot Error in `on_ready`!', args[0])
        self.assertIn('Exception: event error', args[0])
