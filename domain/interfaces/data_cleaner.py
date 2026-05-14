from abc import ABC, abstractmethod
from domain.entities.dataset import Dataset


class DataCleaner(ABC):
    @abstractmethod
    def clean(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
