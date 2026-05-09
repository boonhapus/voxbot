import os
import unittest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.plugins.soul.cog import _MAX_CONVERSATION_TURNS, _trim_conversation


def _turn(index: int) -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest(parts=[UserPromptPart(content=f"message {index}")]),
        ModelResponse(parts=[TextPart(content=f"reply {index}")]),
    ]


class SoulCogTests(unittest.TestCase):
    def test_trim_conversation_keeps_recent_user_turns(self) -> None:
        messages = []
        for index in range(_MAX_CONVERSATION_TURNS + 5):
            messages.extend(_turn(index))

        trimmed = _trim_conversation(messages)

        self.assertEqual(len(trimmed), _MAX_CONVERSATION_TURNS * 2)
        self.assertIsInstance(trimmed[0], ModelRequest)
        self.assertEqual(trimmed[0].parts[0].content, "message 5")

    def test_trim_conversation_leaves_short_history_unchanged(self) -> None:
        messages = _turn(1)

        self.assertIs(_trim_conversation(messages), messages)
