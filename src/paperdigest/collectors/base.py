"""Abstract collector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config
from ..models import Paper


class BaseCollector(ABC):
    """Base class for paper collectors."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def collect(self) -> list[Paper]:
        """Collect papers and return a list of Paper objects."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name for this source."""
        ...
