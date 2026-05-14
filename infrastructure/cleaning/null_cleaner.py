from domain.entities.cleaning_report import CleaningReport
from domain.entities.dataset import Dataset
from domain.interfaces.data_cleaner import DataCleaner


class NullCleaner(DataCleaner):
    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is not None:
            nulls_removed = int(dataset.data.isna().sum().sum())
            dataset.data = dataset.data.fillna(0)
            dataset.report = dataset.report or CleaningReport()
            dataset.report.nulls_removed += nulls_removed
            dataset.report.missing_values_fixed += nulls_removed
        return dataset
