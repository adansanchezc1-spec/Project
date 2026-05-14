from domain.entities.dataset import Dataset
from domain.interfaces.data_cleaner import DataCleaner


class NullCleaner(DataCleaner):
    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is not None:
            dataset.data = dataset.data.fillna(0)
        return dataset
