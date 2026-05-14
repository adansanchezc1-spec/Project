from domain.entities.cleaning_report import CleaningReport
from domain.entities.dataset import Dataset
from domain.interfaces.data_cleaner import DataCleaner


class DuplicateCleaner(DataCleaner):
    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is not None:
            initial_rows = len(dataset.data)
            dataset.data = dataset.data.drop_duplicates().reset_index(drop=True)
            removed_rows = initial_rows - len(dataset.data)
            dataset.report = dataset.report or CleaningReport()
            dataset.report.duplicate_rows_removed += removed_rows
            dataset.report.affected_rows += removed_rows
        return dataset
