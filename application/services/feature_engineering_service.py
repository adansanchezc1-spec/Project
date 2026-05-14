from domain.entities.dataset import Dataset
from domain.enums.dataset_status import DatasetStatus
from domain.interfaces.feature_analyzer import IFeatureAnalyzer
from infrastructure.analytics.pandas_feature_analyzer import PandasFeatureAnalyzer


class FeatureEngineeringService:
    def __init__(self, analyzer: IFeatureAnalyzer | None = None) -> None:
        self._analyzer = analyzer or PandasFeatureAnalyzer()

    def engineer(self, dataset: Dataset) -> list:
        dataset.features = self._analyzer.analyze(dataset)
        dataset.update_status(DatasetStatus.PROFILED)
        return dataset.features
