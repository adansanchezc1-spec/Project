from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class IDataRepository(ABC):
    """Contract for dataset persistence and loading."""

    @abstractmethod
    def load(self, path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, data: Any) -> None:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        return Path(path).exists()


class DataRepository(IDataRepository):
    """Backward-compatible alias for existing infrastructure classes."""
