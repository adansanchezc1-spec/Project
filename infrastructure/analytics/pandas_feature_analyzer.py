import pandas as pd
from domain.entities.dataset import Dataset
from domain.entities.feature import Feature
from domain.interfaces.feature_analyzer import FeatureAnalyzer


class PandasFeatureAnalyzer(FeatureAnalyzer):
    def analyze(self, dataset: Dataset) -> list:
        if dataset.data is None:
            return []

        features = []
        for name, dtype in dataset.data.dtypes.items():
            features.append(Feature(name=name, dtype=str(dtype), importance=0.0))
        return features
