import pandas as pd
from domain.entities.dataset import Dataset
from domain.enums.dataset_status import DatasetStatus


class IngestionService:
    def load_dataset(self, path: str) -> Dataset:
        dataframe = pd.read_csv(path)
        dataset = Dataset(
            id=path,
            name=path.split('\\')[-1],
            data_path=path,
            status=DatasetStatus.NEW,
            data=dataframe,
        )
        return dataset
