from abc import ABC, abstractmethod

from domain.entities.dataset import Dataset


class IFeatureAnalyzer(ABC):
    """Contract for feature analysis and selection."""

    @abstractmethod
    def analyze(self, dataset: Dataset) -> list:
        raise NotImplementedError

    def remove_irrelevant(self, dataset: Dataset) -> Dataset:
        return dataset


class FeatureAnalyzer(IFeatureAnalyzer):
    """Backward-compatible alias for existing analyzers."""
