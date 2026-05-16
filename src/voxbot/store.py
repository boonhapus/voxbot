import pathlib

import pydantic


class RuntimeConfig(pydantic.BaseModel):
    """The location where Voxbot stores its config info."""

    directory: pathlib.Path = pathlib.Path.home() / ".voxbot"
    library_root: pathlib.Path = pathlib.Path(__file__).parent

    _extra_paths: set[pathlib.Path] = set()

    @property
    def commands_sha(self) -> pathlib.Path:
        """
        Stores the calculated the command tree hash.

        Used to determine if we need to sync Bot.tree (slash commands) to the server.
        """
        return self.directory / "commands.sha"

    def extend(self, *parts: str) -> pathlib.Path:
        """Get or create a subdirectory under the runtime directory."""
        path = self.directory.joinpath(*parts)
        self._extra_paths.add(path)
        self.ensure_directories()
        return path

    def ensure_directories(self) -> None:
        """Create all the necessary directories on disk."""
        for path in (self.directory, self.commands_sha, *self._extra_paths):
            if path.suffixes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)


runtime = RuntimeConfig()
