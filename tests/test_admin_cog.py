import os
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.plugins.admin.cog import _format_health_report, _parse_owner_ids


class AdminCogTests(unittest.TestCase):
    def test_parse_owner_ids_accepts_commas_and_spaces(self) -> None:
        self.assertEqual(_parse_owner_ids("123, 456 789"), {123, 456, 789})

    def test_format_health_report_includes_bot_and_worker_state(self) -> None:
        report = _format_health_report(
            {
                "ready": "true",
                "heartbeat_unix": "100",
                "release_sha": "abcdef1234567890",
                "latency_ms": "42",
                "worker_ready": "true",
                "worker_heartbeat_unix": "95",
                "worker_release_sha": "123456abcdef7890",
            },
            now=105,
        )

        self.assertIn("Ready: true", report)
        self.assertIn("Release: abcdef123456", report)
        self.assertIn("Heartbeat age: 5s", report)
        self.assertIn("Discord latency: 42 ms", report)
        self.assertIn("Worker ready: true", report)
        self.assertIn("Worker heartbeat age: 10s", report)
