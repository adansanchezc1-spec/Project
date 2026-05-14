from domain.entities.dataset import Dataset
from domain.interfaces.data_cleaner import DataCleaner


class FormatCleaner(DataCleaner):
    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is not None:
            for column in dataset.data.select_dtypes(include=['object']).columns:
                dataset.data[column] = dataset.data[column].str.strip()
        return dataset
