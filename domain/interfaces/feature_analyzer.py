from abc import ABC, abstractmethod
from domain.entities.dataset import Dataset


class FeatureAnalyzer(ABC):
    @abstractmethod
    def analyze(self, dataset: Dataset) -> list:
        raise NotImplementedError
