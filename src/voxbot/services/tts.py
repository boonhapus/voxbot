"""TTS audio processing and playback."""

import base64
import os
import pathlib
import tempfile

import disnake
import structlog

from voxbot.errors import TTSError

_LOGGER = structlog.get_logger(__name__)


class TTSProcessor:
    """Handles TTS audio generation and Discord playback setup."""

    @staticmethod
    def prepare_audio_source(audio_data_b64: str) -> tuple[disnake.FFmpegPCMAudio, str]:
        """
        Decode base64 audio, write to temp file, return FFmpegPCMAudio source.
        Caller is responsible for cleanup via the returned temp path.
        """
        try:
            audio_bytes = base64.b64decode(audio_data_b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            tmp_path = tmp.name

            source = disnake.FFmpegPCMAudio(tmp_path)
            return source, tmp_path
        except Exception as err:
            _LOGGER.error("audio_prepare_failed", error=str(err))
            raise TTSError(f"failed to prepare audio: {err}") from err

    @staticmethod
    def cleanup_temp_file(path: str) -> None:
        """Remove a temp audio file."""
        try:
            os.unlink(path)
        except Exception as err:
            _LOGGER.warning("temp_file_cleanup_failed", path=path, error=str(err))
