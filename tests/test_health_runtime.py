import os
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import Settings


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.closed = False

    async def set(self, key: str, value: object) -> None:
        self.data[key] = str(value)

    async def incr(self, key: str) -> None:
        self.data[key] = str(int(self.data.get(key, "0")) + 1)

    async def aclose(self) -> None:
        self.closed = True


class FakeBot:
    latency = 0.123


class HealthRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self) -> Settings:
        return Settings(
            _env_file=None,
            discord_token="discord",
            mistral_api_key="mistral",
            google_api_key="google",
            deployment_id="test-sha",
            health_heartbeat_seconds=1,
        )

    async def test_mark_ready_writes_expected_health_keys(self) -> None:
        redis = FakeRedis()
        runtime = RedisHealthRuntime(self._settings(), redis_client=redis)

        await runtime.mark_ready(True, bot=FakeBot())

        self.assertEqual(redis.data["voxbot:health:ready"], "true")
        self.assertEqual(redis.data["voxbot:health:release_sha"], "test-sha")
        self.assertEqual(redis.data["voxbot:health:latency_ms"], "123")
        self.assertIn("voxbot:health:heartbeat", redis.data)
        self.assertIn("voxbot:health:heartbeat_unix", redis.data)

    async def test_record_restart_requested_sets_reason_and_count(self) -> None:
        redis = FakeRedis()
        runtime = RedisHealthRuntime(self._settings(), redis_client=redis)

        await runtime.record_restart_requested(" deploy requested \n now ")

        self.assertEqual(redis.data["voxbot:health:restart_requested"], "deploy requested now")
        self.assertEqual(redis.data["voxbot:health:restart_count"], "1")

    async def test_stop_marks_unready_and_closes_client(self) -> None:
        redis = FakeRedis()
        runtime = RedisHealthRuntime(self._settings(), redis_client=redis)

        await runtime.stop()

        self.assertEqual(redis.data["voxbot:health:ready"], "false")
        self.assertTrue(redis.closed)
