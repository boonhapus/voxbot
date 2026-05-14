import structlog

from voxbot.errors import VoxBotError

_LOGGER = structlog.get_logger(__name__)


class VoiceCommandError(VoxBotError):
    """User-facing voice command failure with optional structured logging."""

    def __init__(
        self,
        user_message: str,
        *,
        log_event: str | None = None,
        **log_kwargs: object,
    ) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.log_event = log_event
        self.log_kwargs = log_kwargs

    def log(self) -> None:
        """Emit the configured structlog event when one was supplied."""
        if self.log_event is None:
            return

        _LOGGER.error(self.log_event, **self.log_kwargs)
