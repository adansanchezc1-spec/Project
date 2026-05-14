from abc import ABC, abstractmethod
from typing import Any


class DataRepository(ABC):
    @abstractmethod
    def load(self, path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, data: Any) -> None:
        raise NotImplementedError
