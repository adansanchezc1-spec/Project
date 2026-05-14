from domain.entities.dataset import Dataset
from infrastructure.analytics.pandas_feature_analyzer import PandasFeatureAnalyzer


class FeatureEngineeringService:
    def __init__(self) -> None:
        self.analyzer = PandasFeatureAnalyzer()

    def engineer(self, dataset: Dataset) -> list:
        return self.analyzer.analyze(dataset)
