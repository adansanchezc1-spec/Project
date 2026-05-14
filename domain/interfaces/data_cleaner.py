from abc import ABC, abstractmethod

from domain.entities.dataset import Dataset


class IDataCleaner(ABC):
    """Contract for one dataset-cleaning strategy."""

    @abstractmethod
    def clean(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError


class DataCleaner(IDataCleaner):
    """Backward-compatible alias for existing cleaning strategies."""
