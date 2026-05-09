import os
import unittest

import pydantic

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.plugins.soul.models import DiscordResponse, ReactAction, TextAction, ThreadAction


class SoulModelTests(unittest.TestCase):
    def test_text_action_strips_empty_messages_and_caps_at_three(self) -> None:
        action = TextAction(content=[" first ", "", "second", " third ", "fourth"])

        self.assertEqual(action.content, ["first", "second", "third"])

    def test_response_parses_discriminated_actions(self) -> None:
        response = DiscordResponse(
            actions=[
                {"kind": "silent"},
                {"kind": "text", "delivery": "reply", "content": [" hi "]},
                {"kind": "react", "emoji": " 👀 "},
            ]
        )

        self.assertEqual(len(response.actions), 3)
        self.assertEqual(response.actions[1].content, ["hi"])
        self.assertEqual(response.actions[2].emoji, "👀")

    def test_react_action_rejects_empty_emoji(self) -> None:
        with self.assertRaises(pydantic.ValidationError):
            ReactAction(emoji="   ")

    def test_thread_action_normalizes_title_and_content(self) -> None:
        action = ThreadAction(title="  side   quest  ", content=[" ok ", ""])

        self.assertEqual(action.title, "side quest")
        self.assertEqual(action.content, ["ok"])
