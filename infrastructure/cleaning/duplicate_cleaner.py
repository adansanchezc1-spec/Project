from domain.entities.dataset import Dataset
from domain.interfaces.data_cleaner import DataCleaner


class DuplicateCleaner(DataCleaner):
    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is not None:
            dataset.data = dataset.data.drop_duplicates().reset_index(drop=True)
        return dataset
