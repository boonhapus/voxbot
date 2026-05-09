from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import pathlib
import tempfile
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.plugins.soul.memory import MemoryService


@dataclass
class FakeUser:
    id: int
    name: str
    display_name: str
    global_name: str | None = None


@dataclass
class FakeChannel:
    id: int
    type: str = "private"


@dataclass
class FakeGuild:
    id: int


@dataclass
class FakeMessage:
    id: int = 1001
    author: FakeUser = field(default_factory=lambda: FakeUser(42, "boon", "Boon"))
    channel: FakeChannel = field(default_factory=lambda: FakeChannel(99))
    guild: FakeGuild | None = field(default_factory=lambda: FakeGuild(7))
    mentions: list[FakeUser] = field(default_factory=list)


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.people_file = pathlib.Path(self.temp_dir.name) / "people.json"
        self.service = MemoryService(self.people_file)
        self.message = FakeMessage()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_remember_stores_fact_and_summary_reads_it(self) -> None:
        result = await self.service.remember(self.message, "  likes black coffee  ", "preference")

        self.assertEqual(result, "Remembered for Boon: likes black coffee")
        self.assertEqual(self.service.summary(self.message), "- preference: likes black coffee")

        data = json.loads(self.people_file.read_text(encoding="utf-8"))
        fact = data["users"]["42"]["facts"][0]
        self.assertEqual(fact["fact"], "likes black coffee")
        self.assertEqual(fact["category"], "preference")
        self.assertEqual(fact["source_message_id"], "1001")
        self.assertEqual(fact["channel_id"], "99")
        self.assertEqual(fact["guild_id"], "7")

    async def test_remember_refreshes_duplicate_fact(self) -> None:
        await self.service.remember(self.message, "likes black coffee", "preference")
        result = await self.service.remember(self.message, "LIKES BLACK COFFEE", "preference")

        self.assertEqual(result, "Memory already existed for Boon; refreshed it.")

        data = json.loads(self.people_file.read_text(encoding="utf-8"))
        self.assertEqual(len(data["users"]["42"]["facts"]), 1)

    async def test_forget_removes_matching_fact(self) -> None:
        await self.service.remember(self.message, "likes black coffee", "preference")
        await self.service.remember(self.message, "works in software", "job")

        result = await self.service.forget(self.message, fact_fragment="coffee")

        self.assertEqual(result, "Forgot 1 memory item(s) for Boon.")
        self.assertEqual(self.service.summary(self.message), "- job: works in software")

    async def test_remember_can_target_named_mention(self) -> None:
        mentioned_user = FakeUser(77, "jane", "Jane Doe", global_name="Jane")
        self.message.mentions.append(mentioned_user)

        result = await self.service.remember(
            self.message,
            "works nights",
            "job",
            person_name="Jane",
        )

        self.assertEqual(result, "Remembered for Jane Doe: works nights")
        data = json.loads(self.people_file.read_text(encoding="utf-8"))
        self.assertIn("77", data["users"])

    async def test_none_message_returns_background_safe_messages(self) -> None:
        self.assertEqual(self.service.summary(None), "- No current author; this is a background identity check.")
        self.assertEqual(
            await self.service.remember(None, "likes coffee"),
            "No memory stored because there is no current Discord message.",
        )
        self.assertEqual(
            await self.service.forget(None, fact_fragment="coffee"),
            "No memory forgotten because there is no current Discord message.",
        )
