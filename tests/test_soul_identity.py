import os
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.plugins.soul.identity import IdentityService


class IdentityServiceTests(unittest.TestCase):
    def test_normalize_display_name_strips_and_collapses_whitespace(self) -> None:
        self.assertEqual(IdentityService.normalize_display_name("  Vox   Jr  "), "Vox Jr")

    def test_normalize_display_name_rejects_empty_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "display name cannot be empty"):
            IdentityService.normalize_display_name("   ")

    def test_normalize_display_name_rejects_control_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "control characters"):
            IdentityService.normalize_display_name("Vox\nJr")

    def test_normalize_display_name_rejects_names_over_discord_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 characters"):
            IdentityService.normalize_display_name("x" * 33)
