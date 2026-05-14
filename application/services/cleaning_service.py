from domain.entities.dataset import Dataset
from domain.enums.dataset_status import DatasetStatus
from domain.interfaces.data_cleaner import IDataCleaner
from infrastructure.cleaning.duplicate_cleaner import DuplicateCleaner
from infrastructure.cleaning.format_cleaner import FormatCleaner
from infrastructure.cleaning.null_cleaner import NullCleaner


class CleaningService:
    def __init__(self, strategies: list[IDataCleaner] | None = None) -> None:
        self._strategies = strategies or [
            NullCleaner(),
            FormatCleaner(),
            DuplicateCleaner(),
        ]

    def clean(self, dataset: Dataset) -> Dataset:
        if dataset.data is None:
            dataset.update_status(DatasetStatus.ERROR)
            return dataset

        dataset.update_status(DatasetStatus.CLEANING)
        for strategy in self._strategies:
            dataset = strategy.clean(dataset)

        dataset.is_clean = True
        dataset.update_status(DatasetStatus.TRANSFORMED)
        dataset.refresh_profile()
        return dataset
