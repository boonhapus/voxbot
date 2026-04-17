import attrs
import structlog
from mistralai.client import Mistral

import hikariwave

_LOGGER = structlog.get_logger("voxtral")


@attrs.define(frozen=True, kw_only=True)
class Config:
    token: str
    mistral_api_key: str
    mistral_model: str = "voxtral-mini-tts-2603"


class VoxModel:
    mistral: Mistral
    custom_voices: dict[str, str]
    voice_client: hikariwave.VoiceClient | None
    last_active: dict[int, float]

    def __init__(self, config: Config) -> None:
        self.mistral = Mistral(api_key=config.mistral_api_key)
        self.custom_voices = {}
        self.voice_client = None
        self.last_active = {}
