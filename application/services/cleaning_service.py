from domain.entities.dataset import Dataset
from infrastructure.cleaning.null_cleaner import NullCleaner
from infrastructure.cleaning.format_cleaner import FormatCleaner
from infrastructure.cleaning.duplicate_cleaner import DuplicateCleaner


class CleaningService:
    def __init__(self) -> None:
        self.cleaners = [
            NullCleaner(),
            FormatCleaner(),
            DuplicateCleaner(),
        ]

    def clean(self, dataset: Dataset) -> Dataset:
        for cleaner in self.cleaners:
            dataset = cleaner.clean(dataset)
        return dataset
