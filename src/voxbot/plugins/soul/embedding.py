import asyncio
import functools as ft
import hashlib
import math
import re

from google import genai
from google.genai import types
import structlog

from voxbot.settings import settings

from .settings import soul_settings

type Vector = list[float]
_LOGGER = structlog.get_logger(__name__)


def hashed_embedding(text: str) -> Vector:
    """Deterministic local embedding fallback when remote embeddings are unavailable."""
    vector = [0.0] * 256

    if not (tokens := re.findall(r"[a-z0-9_]+", text.casefold())):
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % 256
        sign = 1.0 if (digest[2] & 1) else -1.0
        vector[index] += sign

    if (norm := math.sqrt(sum(value * value for value in vector))) == 0:
        return vector

    return [value / norm for value in vector]


class GoogleEmbeddingProvider:
    """Gemini embedding wrapper with graceful fallback when unavailable."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or soul_settings.memory_embedding_model
        self.client = genai.Client(api_key=api_key or settings.google_api_key)

    def _embed(self, text: str, *, task_type: str) -> Vector:
        if not (payload := text.strip()):
            return hashed_embedding(text)

        try:
            r = self.client.models.embed_content(
                model=self.model,
                contents=[payload],
                config=types.EmbedContentConfig(task_type=task_type),
            )

            return [float(v) for v in r.embeddings[0].values]  # pyrefly: ignore[unsupported-operation, not-iterable]

        except Exception as exc:
            _LOGGER.warning("embedding_generation_failed_falling_back", error=str(exc))
            return hashed_embedding(payload)

    async def embed_document(self, text: str) -> Vector:
        fn = ft.partial(self._embed, text=text, task_type="RETRIEVAL_DOCUMENT")
        return await asyncio.to_thread(fn)

    async def embed_query(self, text: str) -> Vector:
        fn = ft.partial(self._embed, text=text, task_type="RETRIEVAL_QUERY")
        return await asyncio.to_thread(fn)
