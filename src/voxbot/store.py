import pathlib

import pydantic


class RuntimeConfig(pydantic.BaseModel):
    """The location where Voxbot stores its config info."""

    directory: pathlib.Path = pathlib.Path.home() / ".voxbot"
    library_root: pathlib.Path = pathlib.Path(__file__).parent

    @property
    def commands_sha(self) -> pathlib.Path:
        """
        Stores the calculated the command tree hash.
        
        Used to determine if we need to sync Bot.tree (slash commands) to the server.
        """
        return self.directory / "commands.sha"
    
    def ensure_directories(self) -> None:
        """Create all the necessary directories on disk."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.commands_sha.touch(exist_ok=True)


runtime = RuntimeConfig()