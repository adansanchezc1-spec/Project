from abc import ABC, abstractmethod
from typing import Any


class IMDMService(ABC):
    """Contract for master data management operations."""

    def merge(self, datasets: list[Any]) -> Any:
        raise NotImplementedError

    def deduplicate(self, data: Any) -> Any:
        raise NotImplementedError

    def standardize(self, data: Any) -> Any:
        raise NotImplementedError


class MDMService(IMDMService):
    """Legacy service contract kept while the application migrates to IMDMService."""

    @abstractmethod
    def sync_master_data(self, dataset_id: str) -> bool:
        raise NotImplementedError
