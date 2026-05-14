from domain.entities.cleaning_report import CleaningReport
from domain.entities.dataset import Dataset
from domain.interfaces.data_cleaner import DataCleaner


class FormatCleaner(DataCleaner):
    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is not None:
            object_columns = dataset.data.select_dtypes(include=["object"]).columns
            dataset.report = dataset.report or CleaningReport()

            for column in object_columns:
                original = dataset.data[column].copy()
                dataset.data[column] = dataset.data[column].astype(str).str.strip().str.lower()
                dataset.report.formatting_issues += int(
                    (original.astype(str) != dataset.data[column]).sum()
                )
        return dataset
