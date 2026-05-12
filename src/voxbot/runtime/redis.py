from __future__ import annotations

import inspect
from typing import Any

from redis import asyncio as redis_async


def create_redis_client(url: str) -> redis_async.Redis:
    """Create a decoded async Redis client."""
    return redis_async.from_url(url, decode_responses=True)


async def close_redis_client(client: Any) -> None:
    """Close Redis-like clients across redis-py versions and test fakes."""
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return

    result = close()
    if inspect.isawaitable(result):
        await result
