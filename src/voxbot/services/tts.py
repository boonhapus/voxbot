"""TTS audio processing and playback."""

import base64
import os
import tempfile

import discord
import structlog

from voxbot.errors import TTSError

_LOGGER = structlog.get_logger(__name__)


class TTSProcessor:
    """Handles TTS audio generation and Discord playback setup."""

    @staticmethod
    def prepare_audio_source(audio_data_b64: str) -> tuple[discord.FFmpegPCMAudio, str]:
        """Decode base64 audio into a playable FFmpegPCMAudio source backed by a temporary file."""
        try:
            audio_bytes = base64.b64decode(audio_data_b64)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
                f.write(audio_bytes)

            source = discord.FFmpegPCMAudio(tmp_path)
            return source, tmp_path
        except Exception as exc:
            _LOGGER.error("audio_prepare_failed", error=str(exc))
            raise TTSError(f"failed to prepare audio: {exc}") from exc

    @staticmethod
    def cleanup_temp_file(path: str) -> None:
        """Remove a temp audio file."""
        try:
            os.unlink(path)
        except Exception as exc:
            _LOGGER.warning("temp_file_cleanup_failed", path=path, error=str(exc))
