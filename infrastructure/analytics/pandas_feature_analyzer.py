import numpy as np

from domain.entities.dataset import Dataset
from domain.entities.feature import Feature
from domain.interfaces.feature_analyzer import FeatureAnalyzer


class PandasFeatureAnalyzer(FeatureAnalyzer):
    def analyze(self, dataset: Dataset) -> list:
        if dataset.data is None:
            return []

        features = []
        numeric_data = dataset.data.select_dtypes(include=[np.number])
        numeric_variance = numeric_data.var(numeric_only=True) if not numeric_data.empty else {}

        for name, dtype in dataset.data.dtypes.items():
            importance = 0.0
            if name in numeric_variance:
                importance = float(numeric_variance[name] or 0.0)
            features.append(Feature(name=name, dtype=str(dtype), importance=importance))
        return features

    def remove_irrelevant(self, dataset: Dataset) -> Dataset:
        if dataset.data is None:
            return dataset

        empty_columns = [
            column for column in dataset.data.columns if dataset.data[column].nunique() <= 1
        ]
        dataset.data = dataset.data.drop(columns=empty_columns)
        dataset.refresh_profile()
        return dataset
