import pydantic


class VoxModel(pydantic.BaseModel):
    """Shared state for voice management and activity tracking."""

    custom_voices: dict[str, str] = {}
    deleted_voices: set[str] = set()
    hero_origins: dict[str, str] = {}
    last_active: dict[int, float] = {}

    model_config = pydantic.ConfigDict(validate_assignment=True)
