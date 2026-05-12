import os
import unittest

import pydantic

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.settings import Settings


class SettingsTests(unittest.TestCase):
    def _settings(self, **overrides) -> Settings:
        values = {
            "discord_token": "discord",
            "mistral_api_key": "mistral",
            "google_api_key": "google",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    def test_deploy_runtime_defaults(self) -> None:
        settings = self._settings()

        self.assertEqual(settings.redis_url, "redis://localhost:6379/1")
        self.assertIsNone(settings.docket_url)
        self.assertEqual(settings.docket_name, "voxbot")
        self.assertTrue(settings.docket_enabled)
        self.assertEqual(settings.soul_memory_backend, "json")
        self.assertEqual(settings.soul_memory_server_url, "http://localhost:8000")
        self.assertTrue(settings.health_enabled)
        self.assertEqual(settings.health_heartbeat_seconds, 10)

    def test_health_heartbeat_seconds_must_be_positive(self) -> None:
        with self.assertRaises(pydantic.ValidationError):
            self._settings(health_heartbeat_seconds=0)

    def test_mirrors_secret_fields_to_environment(self) -> None:
        self._settings(discord_token="new-discord-token")

        self.assertEqual(os.environ["DISCORD_TOKEN"], "new-discord-token")
