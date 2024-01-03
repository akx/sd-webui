from __future__ import annotations

import dataclasses
from pathlib import Path

from modules.upscaler import logger


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """
    Describes a (generally non-SD) model that can be either local already,
    or downloaded from an URL to a local directory.
    """
    name: str
    local_directory: Path
    url: str | None = None
    local_name: str | None = None

    def __post_init__(self):
        if not self.local_name:
            if not self.url:
                raise ValueError(f"Model {self.name} has no local name and no URL was provided.")
            name = self.url.rpartition("/")[2]
            object.__setattr__(self, "local_name", name)

    @property
    def local_path(self) -> Path:
        return self.local_directory / self.local_name

    def download_if_required(self) -> None:
        if self.local_path.exists():
            return
        if not self.url:
            raise ValueError(f"Model {self.name} is not available locally and no URL was provided.")
        self.local_directory.mkdir(parents=True, exist_ok=True)
        from torch.hub import download_url_to_file
        logger.info("Downloading %s to %s", self.url, self.local_path)
        download_url_to_file(self.url, str(self.local_path), progress=True)
